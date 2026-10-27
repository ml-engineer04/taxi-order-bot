"""Active Learning Service for Interactive Message Review and Continuous Model Training."""

import asyncio
import csv
import html
import logging
import os
from typing import Optional, Tuple
from aiogram import types
from aiogram.utils.keyboard import InlineKeyboardBuilder

from app.database.models.collected_message import CollectedMessageRecord
from app.ml.dataset import detect_script
from app.ml.train import train_and_evaluate
from app.services.classification import get_classifier
from app.services.classification.ml_classifier import MLClassifier

logger = logging.getLogger(__name__)


def append_example_to_dataset(
    text: str,
    label: str,
    csv_path: str = "data/taxi_labeled_dataset.csv",
) -> bool:
    """Append a human-labeled example to the training dataset CSV."""
    clean_text = text.strip()
    if not clean_text or label not in {"PASSENGER_REQUEST", "DRIVER_OFFER", "OTHER"}:
        return False

    script = detect_script(clean_text)

    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, mode="a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow([clean_text, label, script])
        logger.info("Appended new training example to '%s': [%s] '%s'", csv_path, label, clean_text[:40])
        return True
    except Exception as e:
        logger.error("Failed to append to dataset '%s': %s", csv_path, e)
        return False


def retrain_model_sync() -> Tuple[bool, str]:
    """Synchronous model retraining procedure."""
    try:
        logger.info("Initiating model retraining pipeline...")
        model, metadata = train_and_evaluate()

        # Update currently loaded classifier instance in-memory
        active_classifier = get_classifier()
        if isinstance(active_classifier, MLClassifier):
            active_classifier._load_model()
            logger.info("Successfully reloaded active in-memory MLClassifier model.")

        f1 = metadata.get("test_metrics", {}).get("macro_f1", 0.0)
        samples = metadata.get("dataset_samples", 0)
        summary = f"✅ Model qayta o'qitildi!\n📊 Dataset: {samples} ta namuna\n🎯 Macro F1: {f1:.4f}"
        return True, summary
    except Exception as e:
        logger.exception("Failed during active learning model retraining: %s", e)
        return False, f"❌ Xatolik yuz berdi: {str(e)}"


async def retrain_model_async() -> Tuple[bool, str]:
    """Execute model training in thread pool to avoid blocking asyncio event loop."""
    return await asyncio.to_thread(retrain_model_sync)


def build_review_card(
    record: CollectedMessageRecord,
    remaining_count: int,
) -> Tuple[str, types.InlineKeyboardMarkup]:
    """Build review interface text and interactive choice buttons."""
    escaped_text = html.escape(record.text)
    conf_pct = int(record.confidence * 100)

    label_names = {
        "PASSENGER_REQUEST": "Yo'lovchi (Mijoz)",
        "DRIVER_OFFER": "Haydovchi (Taklif)",
        "OTHER": "Boshqa / Spam",
    }
    pred_display = label_names.get(record.predicted_label, record.predicted_label)

    card_lines = [
        f"📝 <b>Yangi xabarni tasdiqlash</b> (Qolgan: {remaining_count} ta)",
        "━━━━━━━━━━━━━━━━━━━━",
        "💬 <b>Xabar matni:</b>",
        f"<blockquote>{escaped_text}</blockquote>",
        "",
        f"🤖 <b>Model taxmini:</b> {pred_display} <i>({conf_pct}%)</i>",
        "━━━━━━━━━━━━━━━━━━━━",
        "<b>Ushbu xabar aslida qaysi toifaga kiradi?</b>",
    ]

    builder = InlineKeyboardBuilder()
    builder.button(
        text="🚖 Yo'lovchi",
        callback_data=f"rev:label:PASSENGER_REQUEST:{record.id}",
    )
    builder.button(
        text="🚗 Haydovchi",
        callback_data=f"rev:label:DRIVER_OFFER:{record.id}",
    )
    builder.button(
        text="🗑 Boshqa / Spam",
        callback_data=f"rev:label:OTHER:{record.id}",
    )
    builder.button(
        text="⏭ O'tkazib yuborish",
        callback_data=f"rev:skip:{record.id}",
    )
    builder.button(
        text="🛑 To'xtatish",
        callback_data="rev:stop",
    )
    builder.adjust(3, 2)

    return "\n".join(card_lines), builder.as_markup()
