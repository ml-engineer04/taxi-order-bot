"""Script to automatically classify, label, and clear pending collected messages in SQLite."""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.classification.rule_based import (
    RuleBasedClassifier,
    check_driver_patterns,
    check_non_transport_patterns,
)
from app.services.classification.ml_classifier import MLClassifier
from app.services.normalization.transliterator import normalize_uzbek_text


def auto_label_pending(db_path: str = "taxi_bot.db") -> None:
    db_file = Path(db_path)
    if not db_file.exists():
        print(f"Error: Database file not found at {db_path}")
        return

    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    cur.execute("SELECT id, text FROM collected_messages WHERE status = 'pending'")
    rows = cur.fetchall()
    total = len(rows)
    print(f"Found {total} pending messages to analyze and auto-label.")

    if total == 0:
        print("No pending messages to process.")
        conn.close()
        return

    ml = MLClassifier()
    rb = RuleBasedClassifier()

    counts = {"OTHER": 0, "DRIVER_OFFER": 0, "PASSENGER_REQUEST": 0}
    now_iso = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    updates = []

    for idx, (row_id, raw_text) in enumerate(rows, start=1):
        norm = normalize_uzbek_text(raw_text)

        # 1. Deterministic Rule-Based Negative Filters
        if check_driver_patterns(norm):
            label = "DRIVER_OFFER"
        elif check_non_transport_patterns(norm):
            label = "OTHER"
        else:
            # 2. ML Inference
            pred = ml.predict_message(raw_text)
            ml_label = pred.get("label", "OTHER")
            confidence = pred.get("confidence", 0.0)

            if ml_label == "DRIVER_OFFER":
                label = "DRIVER_OFFER"
            elif ml_label == "PASSENGER_REQUEST":
                # Strict check for route locations
                origin, dest = rb._extract_locations(norm)
                if (origin or dest) and confidence >= 0.75:
                    label = "PASSENGER_REQUEST"
                else:
                    label = "OTHER"
            else:
                label = "OTHER"

        counts[label] += 1
        updates.append((label, now_iso, row_id))

        if idx % 1000 == 0 or idx == total:
            print(f"Processed {idx}/{total} messages...")

    # Execute batch updates
    print("Writing updates to database...")
    cur.executemany(
        """
        UPDATE collected_messages
        SET actual_label = ?, status = 'labeled', reviewed_at = ?
        WHERE id = ?
        """,
        updates,
    )
    conn.commit()
    conn.close()

    print("\n=== Auto-Labeling Completed Successfully ===")
    print(f"Total messages labeled: {total}")
    print(f"  OTHER / Chatter / Noise: {counts['OTHER']} ({counts['OTHER']/total*100:.1f}%)")
    print(f"  DRIVER_OFFER:             {counts['DRIVER_OFFER']} ({counts['DRIVER_OFFER']/total*100:.1f}%)")
    print(f"  PASSENGER_REQUEST:        {counts['PASSENGER_REQUEST']} ({counts['PASSENGER_REQUEST']/total*100:.1f}%)")
    print("Pending review queue is now completely cleared (0 pending).")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "taxi_bot.db"
    auto_label_pending(path)
