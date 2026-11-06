"""Dataset Extraction & Preprocessing Pipeline for Yo'lchi AI.

Compiles real-world collected data from:
1. forwarded_messages (ground-truth positive transport requests)
2. collected_messages (driver announcements, spam, ambiguous cases)
3. existing labeled seed dataset

Outputs:
- data/taxi_training_dataset.csv
- data/taxi_training_dataset.jsonl
"""

import csv
import json
import os
import re
import sqlite3
from typing import Dict, List, Set

DB_PATH = "/root/taxi-order-bot/taxi_bot.db"
OUT_CSV = "/root/taxi-order-bot/data/taxi_training_dataset.csv"
OUT_JSONL = "/root/taxi-order-bot/data/taxi_training_dataset.jsonl"


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


def run_export() -> None:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
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
        SELECT text, predicted_label, actual_label, status
        FROM collected_messages
        ORDER BY id ASC
    """)
    coll_rows = cur.fetchall()

    driver_cnt = 0
    other_cnt = 0
    pos_from_coll = 0

    driver_pat = re.compile(
        r"\b(kam|ta kam|odam olamiz|poshta olamiz|odam zarur|aloqadaman|aloqadamiz|"
        r"yuramiz|avto koblt|kobalt|yuraman|moshina bor|mashina bor|yuradiganlar)\b",
        re.IGNORECASE,
    )
    non_transport_pat = re.compile(
        r"\b(kv|kvartira|arenda|ijara|xona|sherik|telefon|sotiladi|ishga|vakansiya)\b",
        re.IGNORECASE,
    )

    for text, pred_label, actual_label, _ in coll_rows:
        cleaned = clean_text(text)
        if not cleaned or len(cleaned) < 4:
            continue
        norm_key = re.sub(r"\s+", " ", cleaned.lower())
        if norm_key in seen_texts:
            continue
        seen_texts.add(norm_key)

        lbl = actual_label or pred_label or "OTHER"
        lbl = lbl.upper()

        if non_transport_pat.search(cleaned):
            final_label = "OTHER"
            other_cnt += 1
            is_order = 0
        elif driver_pat.search(cleaned) or "DRIVER" in lbl:
            final_label = "DRIVER_OFFER"
            driver_cnt += 1
            is_order = 0
        elif "PASSENGER" in lbl:
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

    print(f"✅ Extracted {driver_cnt} DRIVER_OFFER, {other_cnt} OTHER/SPAM, and {pos_from_coll} PASSENGER_REQUEST from collected_messages.")
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
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["text", "label", "is_order", "origin", "destination", "seats", "script", "source"],
        )
        writer.writeheader()
        writer.writerows(dataset)

    # Save to JSONL
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n🎉 Files created successfully:")
    print(f"  CSV:   {OUT_CSV} ({os.path.getsize(OUT_CSV) / 1024:.1f} KB)")
    print(f"  JSONL: {OUT_JSONL} ({os.path.getsize(OUT_JSONL) / 1024:.1f} KB)")


if __name__ == "__main__":
    run_export()
