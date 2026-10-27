"""Comprehensive Automated Tests for Machine Learning NLP Classifier and Telegram Integration."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.config import Settings
from app.services.classification.base import ClassificationResult
from app.services.classification.ml_classifier import MLClassifier
from app.services.classification.rule_based import RuleBasedClassifier


@pytest.fixture
def ml_classifier():
    """Load the trained ML classifier."""
    return MLClassifier(
        model_path="app/ml/models/taxi_classifier.joblib",
        confidence_threshold=0.65,
    )


# =====================================================================
# 1. LATIN SCRIPT PASSENGER TESTS
# =====================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "Toshkentdan Samarqandga 2 ta joy kerak",
        "Chirchiqdan Toshkentga ketishim kerak",
        "Ertaga Buxoroga boradigan mashina kerak",
        "2 kishimiz, Andijonga ketamiz",
        "Namanganga taksi kerak, narxi qancha?",
        "Bugun kechqurun Farg'onaga ketishim kerak",
        "qo'qondan toshkentga odam bor",
        "qoqondan toshkentga odam bor",
        "toshkentga odam bor",
        "buvaydadan 2 ta odam bor toshkentga",
    ],
)
async def test_uzbek_latin_passenger_requests(ml_classifier, text):
    """Verify high-confidence detection of Latin-script passenger requests."""
    res = await ml_classifier.classify(text)
    assert res.is_passenger_request is True, f"Failed on Latin passenger: {text} ({res.reason})"
    assert res.confidence >= 0.65
    assert res.method == "ml"


# =====================================================================
# 2. CYRILLIC SCRIPT PASSENGER TESTS
# =====================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "Тошкентдан Самарқандга 2 та жой керак",
        "Чирчиқдан Тошкентга кетишим керак",
        "Самарқандга борадиган машина борми?",
        "Бухорога такси керак 1 кишига",
        "Андижонга машина борми ҳозир кетадиган?",
    ],
)
async def test_uzbek_cyrillic_passenger_requests(ml_classifier, text):
    """Verify high-confidence detection of Cyrillic-script passenger requests."""
    res = await ml_classifier.classify(text)
    assert res.is_passenger_request is True, f"Failed on Cyrillic passenger: {text} ({res.reason})"
    assert res.confidence >= 0.65
    assert res.method == "ml"


# =====================================================================
# 3. MIXED SCRIPT PASSENGER & COMMON SPELLING ERRORS
# =====================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "Toshkentdan Samarqandga 2 та жой kerak",
        "Toshkentdan Samarqandga 2ta joy kerek",
        "Samarqandga moshina kerek 1 kishiga",
        "Andijonga mashina bormikan hozir yuradigan",
    ],
)
async def test_mixed_script_and_spelling_errors(ml_classifier, text):
    """Verify robustness against typos, dialect variations, and mixed scripts."""
    res = await ml_classifier.classify(text)
    assert res.is_passenger_request is True, f"Failed on variant/mixed: {text} ({res.reason})"
    assert res.confidence >= 0.60


# =====================================================================
# 4. PASSENGER REQUESTS WITHOUT THE WORD 'TAKSI'
# =====================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "Ertaga Buxoroga boradigan mashina kerak",
        "2 kishimiz, Andijonga ketamiz",
        "Kim meni aeroportdan olib ketishi mumkin?",
        "Bugun kechqurun Farg'onaga ketishim kerak",
        "Samarqandga 1 ta joy kerak hozirga",
    ],
)
async def test_passenger_requests_without_taksi_keyword(ml_classifier, text):
    """Verify passenger requests without mentioning 'taksi' are correctly classified."""
    res = await ml_classifier.classify(text)
    assert res.is_passenger_request is True, f"Failed on non-taksi passenger: {text} ({res.reason})"
    assert res.confidence >= 0.65


# =====================================================================
# 5. DRIVER OFFERS WITH SAME DESTINATIONS (NEGATIVE REJECTION)
# =====================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "Toshkentdan Samarqandga 3 ta joy bor odam olaman",
        "Тошкентдан Самарқандга 3 та жой бор",
        "Buxoroga yo'lovchi olaman",
        "Андижонга машина бор, 2 ta joy mavjud",
        "Toshkent aeroportidan yo'lovchi olib ketaman",
        "Farg'onaga odam olamiz hozir yuradigan salon bo'sh",
    ],
)
async def test_driver_offers_rejected(ml_classifier, text):
    """Verify driver transport offers are NOT classified as passenger requests."""
    res = await ml_classifier.classify(text)
    assert res.is_passenger_request is False, f"Driver offer falsely accepted: {text} ({res.reason})"
    pred = ml_classifier.predict_message(text)
    assert pred["label"] == "DRIVER_OFFER"


# =====================================================================
# 6. UNRELATED TOPICS & CITY MENTIONS (NEGATIVE REJECTION)
# =====================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "Bugun Toshkentda havo yaxshi",
        "Samarqandga o'tgan hafta bordim",
        "Assalomu alaykum hammaga",
        "Kimda telefon zaryadkasi bor?",
        "Toshkent–Samarqand yo'lida ta'mirlash ishlari ketmoqda",
        "Chevrolet Cobalt sotiladi 2023 yil probeg 30000",
    ],
)
async def test_unrelated_messages_rejected(ml_classifier, text):
    """Verify casual chats and unrelated city mentions are classified as OTHER."""
    res = await ml_classifier.classify(text)
    assert res.is_passenger_request is False, f"Irrelevant message falsely accepted: {text}"
    pred = ml_classifier.predict_message(text)
    assert pred["label"] == "OTHER"


# =====================================================================
# 7. EMPTY AND UNSUPPORTED MESSAGES
# =====================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", "   ", "ok", "a", "??"])
async def test_empty_and_short_messages(ml_classifier, text):
    """Verify empty and very short strings safely return non-passenger."""
    res = await ml_classifier.classify(text)
    assert res.is_passenger_request is False
    assert res.confidence == 0.0

    pred = ml_classifier.predict_message(text)
    assert pred["label"] == "OTHER"
    assert pred["confidence"] == 0.0


# =====================================================================
# 8. MISSING MODEL ARTIFACT & INFERENCE EXCEPTION FALLBACK
# =====================================================================

@pytest.mark.asyncio
async def test_missing_model_fallback():
    """Verify that classifier falls back safely when the model file is missing."""
    fallback_mock = AsyncMock()
    fallback_mock.classify.return_value = ClassificationResult(
        is_passenger_request=True,
        confidence=0.80,
        origin="Toshkent",
        destination="Samarqand",
        seats=2,
        reason="Fallback rule triggered",
        method="rule_based",
    )

    classifier = MLClassifier(
        model_path="non_existent_path/fake_model.joblib",
        fallback_classifier=fallback_mock,
    )
    assert classifier.pipeline is None

    res = await classifier.classify("Toshkentdan Samarqandga 2 ta joy kerak")
    fallback_mock.classify.assert_called_once()
    assert res.is_passenger_request is True
    assert res.method == "rule_based"


@pytest.mark.asyncio
async def test_inference_exception_fallback(ml_classifier):
    """Verify exception handling during inference falls back gracefully."""
    with patch.object(ml_classifier.pipeline, "predict_proba", side_effect=RuntimeError("GPU OOM")):
        res = await ml_classifier.classify("Toshkentdan Samarqandga 2 ta joy kerak")
        # Should gracefully fall back to rule-based without crashing
        assert isinstance(res, ClassificationResult)


# =====================================================================
# 9. TELEGRAM HANDLER INTEGRATION TESTS
# =====================================================================

@pytest.mark.asyncio
async def test_telegram_public_handler_passenger_forwarded():
    """Verify that public group handler forwards eligible passenger requests."""
    from app.handlers.public_group import handle_public_group_message

    mock_bot = AsyncMock()
    mock_chat = MagicMock()
    mock_chat.id = -1003726192046
    mock_chat.username = "toshkent_taksi"

    mock_user = MagicMock()
    mock_user.id = 123456789
    mock_user.is_bot = False
    mock_user.username = "passenger_user"
    mock_user.full_name = "Ali Valiyev"

    mock_message = MagicMock()
    mock_message.chat = mock_chat
    mock_message.from_user = mock_user
    mock_message.text = "Toshkentdan Samarqandga 2 ta joy kerak"
    mock_message.caption = None
    mock_message.message_id = 999
    mock_message.bot = mock_bot

    with patch("app.handlers.public_group.settings") as mock_settings, \
         patch("app.handlers.public_group.get_forwarding_service") as mock_get_forwarder:

        mock_settings.is_monitored_public_group.return_value = True
        mock_settings.classifier_confidence_threshold = 0.65
        mock_forwarder = AsyncMock()
        mock_forwarder.is_user_on_cooldown = MagicMock(return_value=False)
        mock_get_forwarder.return_value = mock_forwarder

        await handle_public_group_message(mock_message)

        # Forwarding should be triggered
        mock_forwarder.forward_passenger_request.assert_called_once()


@pytest.mark.asyncio
async def test_telegram_public_handler_driver_offer_ignored():
    """Verify that driver offers in the public group are NOT forwarded."""
    from app.handlers.public_group import handle_public_group_message

    mock_bot = AsyncMock()
    mock_chat = MagicMock()
    mock_chat.id = -1003726192046

    mock_user = MagicMock()
    mock_user.id = 987654321
    mock_user.is_bot = False
    mock_user.username = "driver_user"
    mock_user.full_name = "Haydovchi Botir"

    mock_message = MagicMock()
    mock_message.chat = mock_chat
    mock_message.from_user = mock_user
    mock_message.text = "Toshkentdan Samarqandga 3 ta joy bor odam olaman"
    mock_message.caption = None
    mock_message.message_id = 1000
    mock_message.bot = mock_bot

    with patch("app.handlers.public_group.settings") as mock_settings, \
         patch("app.handlers.public_group.get_forwarding_service") as mock_get_forwarder:

        mock_settings.is_monitored_public_group.return_value = True
        mock_settings.classifier_confidence_threshold = 0.65
        mock_forwarder = AsyncMock()
        mock_forwarder.is_user_on_cooldown = MagicMock(return_value=False)
        mock_get_forwarder.return_value = mock_forwarder

        await handle_public_group_message(mock_message)

        # Forwarding must NOT be called for driver offer
        mock_forwarder.forward_passenger_request.assert_not_called()
