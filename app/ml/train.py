"""
Reproducible Training and Evaluation Pipeline for Taxi Message Classifier.
Trains and compares Baseline (TF-IDF + MNB) vs Semantic (Word+Char n-grams + Calibrated SVC).
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from app.ml.dataset import load_dataset, train_val_test_split, VALID_LABELS
from app.ml.pipeline import build_baseline_pipeline, build_semantic_pipeline
from app.services.normalization.transliterator import normalize_uzbek_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ml_train")


def preprocess_samples(samples: List[Dict[str, str]]) -> Tuple[List[str], List[str]]:
    """
    Extract feature text and labels.
    Combines raw text and normalized transliterated representation
    to provide rich dual-script signals.
    """
    X = []
    y = []
    for item in samples:
        raw = item["text"]
        norm = normalize_uzbek_text(raw)
        # Dual representation: raw text + normalized text
        combined = f"{raw} {norm}"
        X.append(combined)
        y.append(item["label"])
    return X, y


def evaluate_model(
    model,
    X_test: List[str],
    y_test: List[str],
    test_samples: List[Dict[str, str]],
    model_name: str,
) -> Dict:
    """Compute precision, recall, F1-scores, confusion matrix, and script breakdown."""
    y_pred = model.predict(X_test)
    labels = sorted(list(VALID_LABELS))

    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

    report_dict = classification_report(
        y_test, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    # Script-specific performance breakdown
    script_perf: Dict[str, Dict] = {"latin": {"correct": 0, "total": 0},
                                    "cyrillic": {"correct": 0, "total": 0},
                                    "mixed": {"correct": 0, "total": 0}}

    for true_label, pred_label, item in zip(y_test, y_pred, test_samples):
        sc = item.get("script", "latin")
        if sc in script_perf:
            script_perf[sc]["total"] += 1
            if true_label == pred_label:
                script_perf[sc]["correct"] += 1

    return {
        "model_name": model_name,
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "labels": labels,
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "script_performance": script_perf,
    }


def train_and_evaluate(dataset_csv: str = "data/taxi_labeled_dataset.csv", output_dir: str = "app/ml/models"):
    """Full end-to-end training, benchmarking, and saving pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Step 1: Loading dataset from %s...", dataset_csv)
    samples = load_dataset(dataset_csv)
    if len(samples) < 20:
        raise ValueError(f"Dataset too small ({len(samples)} samples).")

    logger.info("Step 2: Stratified split into Train (70%%), Val (15%%), Test (15%%)...")
    train_data, val_data, test_data = train_val_test_split(samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    X_train, y_train = preprocess_samples(train_data)
    X_val, y_val = preprocess_samples(val_data)
    X_test, y_test = preprocess_samples(test_data)

    # 3. Train Baseline Model (TF-IDF + MultinomialNB)
    logger.info("Step 3: Training Baseline Model (Word TF-IDF + MultinomialNB)...")
    baseline_model = build_baseline_pipeline()
    baseline_model.fit(X_train, y_train)
    baseline_eval = evaluate_model(baseline_model, X_test, y_test, test_data, "Baseline (Word TF-IDF + MNB)")

    # 4. Train Proposed Semantic Model (Word+Char n-grams + Calibrated LinearSVC)
    logger.info("Step 4: Training Semantic Model (Word+Char n-grams + Calibrated LinearSVC)...")
    # For training the final model, we combine train + val to maximize data efficiency
    X_train_full = X_train + X_val
    y_train_full = y_train + y_val
    semantic_model = build_semantic_pipeline()
    semantic_model.fit(X_train_full, y_train_full)
    semantic_eval = evaluate_model(semantic_model, X_test, y_test, test_data, "Semantic Multi-Gram Calibrated Model")

    # 5. Display Comparative Results
    print("\n" + "=" * 70)
    print("🏆 MODEL BENCHMARK COMPARISON (Held-out Test Set)")
    print("=" * 70)
    print(f"{'Metric':<30} | {'Baseline (MNB)':<18} | {'Proposed Semantic':<18}")
    print("-" * 70)
    print(f"{'Macro F1-Score':<30} | {baseline_eval['macro_f1']:<18.4f} | {semantic_eval['macro_f1']:<18.4f}")
    print(f"{'Weighted F1-Score':<30} | {baseline_eval['weighted_f1']:<18.4f} | {semantic_eval['weighted_f1']:<18.4f}")

    pass_base = baseline_eval["classification_report"].get("PASSENGER_REQUEST", {})
    pass_sem = semantic_eval["classification_report"].get("PASSENGER_REQUEST", {})
    print(f"{'PASSENGER_REQUEST Recall':<30} | {pass_base.get('recall', 0):<18.4f} | {pass_sem.get('recall', 0):<18.4f}")
    print(f"{'PASSENGER_REQUEST Precision':<30} | {pass_base.get('precision', 0):<18.4f} | {pass_sem.get('precision', 0):<18.4f}")
    print("=" * 70)

    print("\n📊 DETAILED CLASSIFICATION REPORT (Semantic Model):")
    for lbl in semantic_eval["labels"]:
        metrics = semantic_eval["classification_report"].get(lbl, {})
        print(f"  • {lbl:<20} | Prec: {metrics.get('precision', 0):.2f} | Rec: {metrics.get('recall', 0):.2f} | F1: {metrics.get('f1-score', 0):.2f} (support: {metrics.get('support', 0)})")

    print("\n🔤 SCRIPT BREAKDOWN (Semantic Model Accuracy):")
    for sc, stats in semantic_eval["script_performance"].items():
        if stats["total"] > 0:
            acc = (stats["correct"] / stats["total"]) * 100
            print(f"  • {sc.capitalize():<10}: {acc:.1f}% ({stats['correct']}/{stats['total']})")

    # 6. Save Model Artifact and Metadata
    model_path = os.path.join(output_dir, "taxi_classifier.joblib")
    metadata_path = os.path.join(output_dir, "model_metadata.json")

    logger.info("Step 5: Saving trained model to %s...", model_path)
    joblib.dump(semantic_model, model_path)

    metadata = {
        "model_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "FeatureUnion(Word+Char n-grams) + Calibrated LinearSVC",
        "classes": sorted(list(VALID_LABELS)),
        "dataset_samples": len(samples),
        "test_metrics": semantic_eval,
        "baseline_comparison": {
            "baseline_macro_f1": baseline_eval["macro_f1"],
            "semantic_macro_f1": semantic_eval["macro_f1"],
        },
        "operating_confidence_threshold": 0.65,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Metadata saved to %s.", metadata_path)
    logger.info("✅ Training pipeline finished successfully.")
    return semantic_model, metadata


if __name__ == "__main__":
    train_and_evaluate()
