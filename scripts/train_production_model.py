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

def main():
    output_dir = "app/ml/models"
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading labeled canonical dataset...")
    labeled = load_dataset("data/taxi_labeled_dataset.csv")

    logger.info("Loading live collected dataset...")
    training = load_dataset("data/taxi_training_dataset.csv")

    logger.info("Generating synthetic driver templates for class balance...")
    driver_templates = generate_driver_templates()

    # Combine canonical samples (weight x3) + real-world training + driver templates
    combined = list(training) + list(labeled) * 3 + driver_templates
    logger.info("Total balanced dataset size: %d samples", len(combined))

    train_data, val_data, test_data = train_val_test_split(combined, 0.70, 0.15, 0.15, random_seed=42)

    X_train, y_train = preprocess_samples(train_data)
    X_val, y_val = preprocess_samples(val_data)
    X_test, y_test = preprocess_samples(test_data)

    logger.info("Training Semantic Multi-Gram Model...")
    X_train_full = X_train + X_val
    y_train_full = y_train + y_val

    semantic_model = build_semantic_pipeline()
    semantic_model.fit(X_train_full, y_train_full)

    eval_res = evaluate_model(semantic_model, X_test, y_test, test_data, "Semantic Multi-Gram Calibrated Model")
    logger.info("Evaluation Complete. Macro F1: %.4f | Weighted F1: %.4f", eval_res["macro_f1"], eval_res["weighted_f1"])

    model_path = os.path.join(output_dir, "taxi_classifier.joblib")
    metadata_path = os.path.join(output_dir, "model_metadata.json")

    joblib.dump(semantic_model, model_path)
    logger.info("Saved model to %s", model_path)

    metadata = {
        "model_version": "2.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "FeatureUnion(Word+Char n-grams) + Calibrated LinearSVC",
        "classes": sorted(list(VALID_LABELS)),
        "dataset_samples": len(combined),
        "test_metrics": eval_res,
        "operating_confidence_threshold": 0.85,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved metadata to %s", metadata_path)
    print("Done!")

if __name__ == "__main__":
    main()
