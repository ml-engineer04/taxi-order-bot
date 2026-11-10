"""Script to train and benchmark production ML classifier with 10k balanced samples."""

import csv
import json
import logging
import os
from datetime import datetime, timezone
import joblib

from app.ml.dataset import load_dataset, train_val_test_split, detect_script, VALID_LABELS
from app.ml.pipeline import build_baseline_pipeline, build_semantic_pipeline
from app.ml.train import preprocess_samples, evaluate_model
from app.services.classification.rule_based import check_driver_patterns, check_non_transport_patterns
from app.services.normalization.transliterator import normalize_uzbek_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("train_prod")

def generate_driver_templates():
    cities_latin = ['Toshkent', 'Samarqand', 'Buxoro', 'Andijon', 'Farg\'ona', 'Namangan', 'Qarshi', 'Termiz', 'Urganch', 'Jizzax', 'Guliston', 'Qo\'qon']
    cities_cyr = ['Тошкент', 'Самарқанд', 'Бухоро', 'Андижон', 'Фарғона', 'Наманган', 'Қарши', 'Термиз', 'Урганч', 'Жиззах', 'Гулистон', 'Қўқон']

    templates_driver_lat = [
        '{c1}dan {c2}ga {n} ta joy bor',
        '{c2}ga moshina bor {n} kishi olamiz',
        '{c2}ga odam olaman {n} ta joy bor',
        '{c2}ga yuramiz salon bosh {n} kishi',
        '{c1} {c2} yo\'nalishida qatnaymiz odam olamiz',
        '{c2}ga 1 kishi kerak mashina tayyor',
        '{c2}ga pochta va odam olamiz',
        '{c2}ga yengil moshina bor hozir yuramiz',
    ]

    templates_driver_cyr = [
        '{c1}дан {c2}га {n} та жой бор',
        '{c2}га машина бор {n} киши оламиз',
        '{c2}га одам оламан {n} та жой бор',
        '{c2}га юрамиз салон бўш {n} киши',
        '{c1} {c2} йўналишида қатнаймиз одам оламиз',
        '{c2}га 1 киши керак машина тайёр',
        '{c2}га почта ва одам оламиз',
        '{c2}га енгил машина бор ҳозир юрамиз',
    ]

    generated = []
    for c1 in cities_latin[:4]:
        for c2 in cities_latin[4:]:
            for n in [1, 2, 3, 4]:
                for t in templates_driver_lat:
                    text = t.format(c1=c1, c2=c2, n=n)
                    generated.append({
                        'text': text,
                        'normalized': normalize_uzbek_text(text),
                        'label': 'DRIVER_OFFER',
                        'script': 'latin',
                    })

    for c1 in cities_cyr[:4]:
        for c2 in cities_cyr[4:]:
            for n in [1, 2, 3, 4]:
                for t in templates_driver_cyr:
                    text = t.format(c1=c1, c2=c2, n=n)
                    generated.append({
                        'text': text,
                        'normalized': normalize_uzbek_text(text),
                        'label': 'DRIVER_OFFER',
                        'script': 'cyrillic',
                    })

    return generated


def generate_passenger_templates():
    """Generate balanced canonical passenger phrases across Uzbek cities."""
    cities = [
        'Toshkent', 'Samarqand', 'Buxoro', 'Andijon', 'Namangan',
        'Farg\'ona', 'Qarshi', 'Termiz', 'Urganch',
        'Navoiy', 'Jizzax', 'Guliston', 'Qo\'qon',
    ]
    templates = [
        '{n} kishimiz, {c}ga ketamiz',
        '{n} kishimiz {c}ga ketamiz hozir',
        '{n} kishimiz {c}ga ketishimiz kerak',
        '{c}ga {n} kishimiz ketamiz taksi kerak',
        '{c}ga {n} ta joy kerak',
        'Kim meni {c}dan olib ketishi mumkin?',
        'Bugun kechqurun {c}ga ketishim kerak',
    ]
    generated = []
    for c in cities:
        for n in [1, 2, 3, 4]:
            for t in templates:
                text = t.format(c=c, n=n)
                generated.append({
                    'text': text,
                    'normalized': normalize_uzbek_text(text),
                    'label': 'PASSENGER_REQUEST',
                    'script': 'latin',
                })
    return generated

def main():
    output_dir = "app/ml/models"
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading labeled canonical dataset...")
    labeled = load_dataset("data/taxi_labeled_dataset.csv")

    logger.info("Loading live collected dataset...")
    raw_training = load_dataset("data/taxi_training_dataset.csv")

    # Relabel historical Day-1 leaked driver/non-transport items so training set stays clean
    training = []
    relabelled = 0
    for s in raw_training:
        norm = s["normalized"]
        if s["label"] == "PASSENGER_REQUEST":
            if check_driver_patterns(norm):
                s = dict(s, label="DRIVER_OFFER")
                relabelled += 1
            elif check_non_transport_patterns(norm):
                s = dict(s, label="OTHER")
                relabelled += 1
        elif s["label"] == "OTHER" and check_driver_patterns(norm):
            s = dict(s, label="DRIVER_OFFER")
            relabelled += 1
        training.append(s)
    logger.info("Cleaned %d real samples (relabelled %d historical leaks).", len(training), relabelled)

    logger.info("Generating synthetic driver and canonical passenger templates...")
    driver_templates = generate_driver_templates()
    passenger_templates = generate_passenger_templates()
    canonical_pos = [x for x in labeled if x["label"] == "PASSENGER_REQUEST"]

    # 1. Honest out-of-sample evaluation (split real data BEFORE adding templates/oversampling)
    train_real, val_real, test_real = train_val_test_split(training, 0.70, 0.15, 0.15, random_seed=42)
    eval_train = train_real + val_real + labeled * 5 + canonical_pos * 10 + driver_templates + passenger_templates

    X_eval_train = [x["text"] for x in eval_train]
    y_eval_train = [x["label"] for x in eval_train]
    X_eval_test = [x["text"] for x in test_real]
    y_eval_test = [x["label"] for x in test_real]

    logger.info("Training Evaluation Model (on train+val split)...")
    eval_model = build_semantic_pipeline()
    eval_model.fit(X_eval_train, y_eval_train)

    eval_res = evaluate_model(eval_model, X_eval_test, y_eval_test, test_real, "Honest Unseen Real Test Split")
    logger.info("Honest Test Evaluation: Macro F1: %.4f | Weighted F1: %.4f", eval_res["macro_f1"], eval_res["weighted_f1"])

    # 2. Fit final production model on ALL clean real samples + canonical + templates
    full_combined = list(training) + list(labeled) * 5 + canonical_pos * 10 + driver_templates + passenger_templates
    logger.info("Training Final Production Model on %d balanced samples...", len(full_combined))
    X_full = [x["text"] for x in full_combined]
    y_full = [x["label"] for x in full_combined]

    semantic_model = build_semantic_pipeline()
    semantic_model.fit(X_full, y_full)

    model_path = os.path.join(output_dir, "taxi_classifier.joblib")
    metadata_path = os.path.join(output_dir, "model_metadata.json")

    joblib.dump(semantic_model, model_path)
    logger.info("Saved model to %s", model_path)

    metadata = {
        "model_version": "3.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "FeatureUnion(Word+Char n-grams) + Calibrated LinearSVC",
        "classes": sorted(list(VALID_LABELS)),
        "real_unique_samples": len(training),
        "dataset_samples": len(full_combined),
        "test_metrics": eval_res,
        "operating_confidence_threshold": 0.85,
        "asymmetric_thresholds": {
            "driver_reject": 0.72,
            "other_reject": 0.82,
            "passenger_approve_with_full_route": 0.85,
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved metadata to %s", metadata_path)
    print("Done!")

if __name__ == "__main__":
    main()
