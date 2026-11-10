"""Dataset Extraction & Preprocessing Pipeline for Yo'lchi AI.

Compiles real-world collected data from:
1. forwarded_messages (ground-truth positive transport requests)
2. collected_messages (driver announcements, spam, ambiguous cases)
3. existing labeled seed dataset

Outputs:
- data/taxi_training_dataset.csv
- data/taxi_training_dataset.jsonl
"""

import argparse
import csv
import json
import os
import re
import sqlite3
from typing import Dict, List, Set

DEFAULT_DB_PATH = "/root/taxi-order-bot/taxi_bot.db"
DEFAULT_OUT_DIR = "/root/taxi-order-bot/data"


def detect_script(text: str) -> str:
    cyrillic_chars = len(re.findall(r"[\u0400-\u04FF]", text))
    latin_chars = len(re.findall(r"[a-zA-Z]", text))
    return "cyrillic" if cyrillic_chars > latin_chars else "latin"


def clean_text(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"[\r\t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def run_export(db_path: str = DEFAULT_DB_PATH, out_dir: str = DEFAULT_OUT_DIR) -> None:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")

    out_csv = os.path.join(out_dir, "taxi_training_dataset.csv")
    out_jsonl = os.path.join(out_dir, "taxi_training_dataset.jsonl")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    dataset: List[Dict] = []
    seen_texts: Set[str] = set()

    # 1. Forwarded messages (High quality verified positive samples)
    cur.execute("""
        SELECT raw_text, origin, destination, seats, forwarded_at
        FROM forwarded_messages
        ORDER BY id ASC
    """)
    fwd_rows = cur.fetchall()

    fwd_pos = 0
    for raw_text, origin, destination, seats, _ in fwd_rows:
        cleaned = clean_text(raw_text)
        if not cleaned or len(cleaned) < 4:
            continue
        # Deduplication on lowercased normalized text
        norm_key = re.sub(r"\s+", " ", cleaned.lower())
        if norm_key in seen_texts:
            continue
        seen_texts.add(norm_key)

        is_cargo = bool(
            re.search(
                r"\b(pochta|poshta|poshda|yuk|posilka|korobka|dastavka|dostavka)\b",
                cleaned,
                re.IGNORECASE,
            )
        )
        label = "CARGO_REQUEST" if is_cargo else "PASSENGER_REQUEST"

        dataset.append({
            "text": cleaned,
            "label": label,
            "is_order": 1,
            "origin": origin or "",
            "destination": destination or "",
            "seats": str(seats or ""),
            "script": detect_script(cleaned),
            "source": "forwarded_messages",
        })
        fwd_pos += 1

    print(f"✅ Extracted {fwd_pos} unique verified positive samples from forwarded_messages.")

    # 2. Collected messages (Negative samples: Driver offers, Real Estate, Spam, Ads)
    cur.execute("""
        SELECT text, predicted_label, actual_label, confidence, status
        FROM collected_messages
        ORDER BY id ASC
    """)
    coll_rows = cur.fetchall()

    driver_cnt = 0
    other_cnt = 0
    pos_from_coll = 0
    skipped_polluted = 0

    driver_pat = re.compile(
        r"\b(ta kam|odam kam|kishi kam|kamdamiz|kamimiz|kammiz|odam olamiz|odam olaman|"
        r"poshta olamiz|pochta olamiz|odam zarur|aloqadaman|aloqadamiz|alishtir|aloshtir|"
        r"yuramiz|avto koblt|kobalt|yuraman|moshina bor|mashina bor|tom bagaj|salon bosh)\b",
        re.IGNORECASE,
    )
    non_transport_pat = re.compile(
        r"\b(kv|kvartira|arenda|ijara|xona|sherik|telefon|sotiladi|ishga|vakansiya|prava|suyunchi)\b",
        re.IGNORECASE,
    )

    for text, pred_label, actual_label, conf, _ in coll_rows:
        # Avoid label poisoning from fallback/timeout uncertain rows (confidence >= 0.25 without human review)
        if not actual_label and conf is not None and conf >= 0.25:
            skipped_polluted += 1
            continue

        cleaned = clean_text(text)
        if not cleaned or len(cleaned) < 4:
            continue
        norm_key = re.sub(r"\s+", " ", cleaned.lower())
        if norm_key in seen_texts:
            continue
        seen_texts.add(norm_key)

        act = (actual_label or "").strip().upper()
        if act in ("PASSENGER_REQUEST", "DRIVER_OFFER", "OTHER"):
            final_label = act
            is_order = 1 if act == "PASSENGER_REQUEST" else 0
            if act == "DRIVER_OFFER":
                driver_cnt += 1
            elif act == "PASSENGER_REQUEST":
                pos_from_coll += 1
            else:
                other_cnt += 1
        elif non_transport_pat.search(cleaned):
            final_label = "OTHER"
            other_cnt += 1
            is_order = 0
        elif driver_pat.search(cleaned) or "DRIVER" in (pred_label or "").upper():
            final_label = "DRIVER_OFFER"
            driver_cnt += 1
            is_order = 0
        elif "PASSENGER" in (pred_label or "").upper():
            final_label = "PASSENGER_REQUEST"
            pos_from_coll += 1
            is_order = 1
        else:
            final_label = "OTHER"
            other_cnt += 1
            is_order = 0

        dataset.append({
            "text": cleaned,
            "label": final_label,
            "is_order": is_order,
            "origin": "",
            "destination": "",
            "seats": "",
            "script": detect_script(cleaned),
            "source": "collected_messages",
        })

    print(f"✅ Extracted {driver_cnt} DRIVER_OFFER, {other_cnt} OTHER/SPAM, and {pos_from_coll} PASSENGER_REQUEST from collected_messages (skipped {skipped_polluted} unreviewed timeout/uncertain rows).")
    print(f"📊 Total unique samples compiled: {len(dataset)}")

    # Class balance stats
    counts: Dict[str, int] = {}
    for item in dataset:
        counts[item["label"]] = counts.get(item["label"], 0) + 1

    print("\n📈 Dataset Label Distribution:")
    for lbl, count in counts.items():
        pct = (count / len(dataset)) * 100
        print(f"  - {lbl}: {count} ({pct:.1f}%)")

    # Save to CSV
    os.makedirs(out_dir, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["text", "label", "is_order", "origin", "destination", "seats", "script", "source"],
        )
        writer.writeheader()
        writer.writerows(dataset)

    # Save to JSONL
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n🎉 Files created successfully:")
    print(f"  CSV:   {out_csv} ({os.path.getsize(out_csv) / 1024:.1f} KB)")
    print(f"  JSONL: {out_jsonl} ({os.path.getsize(out_jsonl) / 1024:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export training dataset from taxi_bot.db")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to SQLite database")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory")
    args = parser.parse_args()
    run_export(db_path=args.db, out_dir=args.out_dir)
