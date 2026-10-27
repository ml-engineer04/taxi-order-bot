"""Dataset loader, validation, and stratified splitting without data leakage."""

import csv
import logging
import random
from typing import Dict, List, Set, Tuple
from app.services.normalization.transliterator import normalize_uzbek_text

logger = logging.getLogger(__name__)

VALID_LABELS: Set[str] = {"PASSENGER_REQUEST", "DRIVER_OFFER", "OTHER"}


def detect_script(text: str) -> str:
    """Classify writing script as latin, cyrillic, or mixed."""
    has_cyrillic = any("\u0400" <= ch <= "\u04ff" for ch in text)
    has_latin = any("a" <= ch.lower() <= "z" for ch in text)

    if has_cyrillic and has_latin:
        return "mixed"
    if has_cyrillic:
        return "cyrillic"
    return "latin"


def load_dataset(csv_path: str) -> List[Dict[str, str]]:
    """
    Load and validate dataset from CSV.
    Validates mandatory columns: text, label.
    Automatically assigns script tag if missing.
    """
    cleaned_rows: List[Dict[str, str]] = []
    seen_texts: Set[str] = set()

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            raw_text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip().upper()

            if not raw_text:
                logger.warning("Row %d: Empty text skipped.", idx)
                continue

            if label not in VALID_LABELS:
                logger.warning("Row %d: Invalid label '%s' skipped.", idx, label)
                continue

            norm = normalize_uzbek_text(raw_text)
            if norm in seen_texts:
                # Group near-identical normalized texts to avoid duplicate leakage
                continue
            seen_texts.add(norm)

            script = row.get("script") or detect_script(raw_text)

            cleaned_rows.append({
                "text": raw_text,
                "normalized": norm,
                "label": label,
                "script": script,
            })

    logger.info("Loaded %d unique validated samples from %s", len(cleaned_rows), csv_path)
    return cleaned_rows


def train_val_test_split(
    samples: List[Dict[str, str]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Perform reproducible stratified split across classes into Train, Validation, and Test sets.
    """
    rng = random.Random(random_seed)

    # Group by class label
    by_class: Dict[str, List[Dict[str, str]]] = {l: [] for l in VALID_LABELS}
    for item in samples:
        by_class[item["label"]].append(item)

    train_set: List[Dict[str, str]] = []
    val_set: List[Dict[str, str]] = []
    test_set: List[Dict[str, str]] = []

    for label, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_set.extend(items[:n_train])
        val_set.extend(items[n_train:n_train + n_val])
        test_set.extend(items[n_train + n_val:])

    # Shuffle combined sets
    rng.shuffle(train_set)
    rng.shuffle(val_set)
    rng.shuffle(test_set)

    logger.info(
        "Stratified Split: Train=%d, Val=%d, Test=%d (Total=%d)",
        len(train_set), len(val_set), len(test_set), len(samples)
    )
    return train_set, val_set, test_set
