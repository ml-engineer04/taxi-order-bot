"""Unit tests for HybridClassifier (ML + DeepSeek AI Fallback)."""

from unittest.mock import AsyncMock
import pytest
from app.services.classification import HybridClassifier
from app.services.classification.base import ClassificationResult


@pytest.mark.asyncio
async def test_hybrid_fast_local_ml_accepted():
    """Verify that high-confidence local ML results with route bypass AI API calls."""
    base_mock = AsyncMock()
    base_mock.classify.return_value = ClassificationResult(
        is_passenger_request=True,
        confidence=0.95,
        origin="Toshkent",
        destination="Samarqand",
        seats=2,
        reason="ML PASSENGER_REQUEST (confidence: 0.95)",
        method="ml",
    )

    ai_mock = AsyncMock()

    hybrid = HybridClassifier(
        base_classifier=base_mock,
        ai_classifier=ai_mock,
        enable_ai=True,
        threshold=0.85,
    )

    res = await hybrid.classify("Toshkentdan Samarqandga 2 ta joy kerak")
    assert res.is_passenger_request is True
    assert res.confidence == 0.95
    assert res.method == "ml"
    ai_mock.classify.assert_not_called()


@pytest.mark.asyncio
async def test_hybrid_fast_local_rejection():
    """Verify that high-confidence local ML driver offers/noise bypass AI API calls."""
    base_mock = AsyncMock()
    base_mock.classify.return_value = ClassificationResult(
        is_passenger_request=False,
        confidence=0.98,
        reason="ML DRIVER_OFFER (confidence: 0.98)",
        method="ml",
    )

    ai_mock = AsyncMock()

    hybrid = HybridClassifier(
        base_classifier=base_mock,
        ai_classifier=ai_mock,
        enable_ai=True,
        threshold=0.85,
    )

    res = await hybrid.classify("Toshkentdan Samarqandga 3 ta joy bor odam olaman")
    assert res.is_passenger_request is False
    assert res.confidence == 0.98
    assert res.method == "ml"
    ai_mock.classify.assert_not_called()


@pytest.mark.asyncio
async def test_hybrid_delegates_on_ambiguous_confidence():
    """Verify that ambiguous confidence (0.35 <= conf < 0.85) delegates to DeepSeek."""
    base_mock = AsyncMock()
    base_mock.classify.return_value = ClassificationResult(
        is_passenger_request=False,
        confidence=0.62,
        reason="Borderline score",
        method="ml",
    )

    ai_mock = AsyncMock()
    ai_mock.classify.return_value = ClassificationResult(
        is_passenger_request=True,
        confidence=0.95,
        origin="Andijon",
        destination="Toshkent",
        seats=1,
        reason="DeepSeek semantic reasoning",
        method="deepseek",
    )

    hybrid = HybridClassifier(
        base_classifier=base_mock,
        ai_classifier=ai_mock,
        enable_ai=True,
        threshold=0.85,
    )

    res = await hybrid.classify("Andijonga mashina bormikan hozir yuradigan")
    assert res.is_passenger_request is True
    assert res.confidence == 0.95
    assert res.method == "deepseek"
    ai_mock.classify.assert_called_once()


@pytest.mark.asyncio
async def test_hybrid_delegates_on_missing_route():
    """Verify that passenger request with missing origin/destination delegates to AI."""
    base_mock = AsyncMock()
    base_mock.classify.return_value = ClassificationResult(
        is_passenger_request=True,
        confidence=0.90,
        origin=None,
        destination=None,
        reason="ML detected passenger but no route extracted",
        method="ml",
    )

    ai_mock = AsyncMock()
    ai_mock.classify.return_value = ClassificationResult(
        is_passenger_request=True,
        confidence=0.95,
        origin="Chilonzor",
        destination="Qo'qon",
        reason="DeepSeek semantic route extraction",
        method="deepseek",
    )

    hybrid = HybridClassifier(
        base_classifier=base_mock,
        ai_classifier=ai_mock,
        enable_ai=True,
        threshold=0.85,
    )

    res = await hybrid.classify("Chilonzordan Qo'qonga bitta odam bor")
    assert res.is_passenger_request is True
    assert res.origin == "Chilonzor"
    assert res.destination == "Qo'qon"
    assert res.method == "deepseek"
    ai_mock.classify.assert_called_once()


@pytest.mark.asyncio
async def test_hybrid_delegates_on_incomplete_single_city_route():
    """Verify that passenger request with only one city (destination missing) delegates to AI."""
    base_mock = AsyncMock()
    base_mock.classify.return_value = ClassificationResult(
        is_passenger_request=True,
        confidence=0.92,
        origin="Andijon",
        destination=None,
        reason="ML high score but incomplete route",
        method="ml",
    )

    ai_mock = AsyncMock()
    ai_mock.classify.return_value = ClassificationResult(
        is_passenger_request=False,
        confidence=0.20,
        reason="Lost document announcement",
        method="deepseek",
    )

    hybrid = HybridClassifier(
        base_classifier=base_mock,
        ai_classifier=ai_mock,
        enable_ai=True,
        threshold=0.85,
    )

    res = await hybrid.classify("Andijondan mexmon kelgan prava yoqotilgan")
    assert res.is_passenger_request is False
    assert res.confidence == 0.20
    assert res.method == "deepseek"
    ai_mock.classify.assert_called_once()


@pytest.mark.asyncio
async def test_hybrid_graceful_fallback_on_ai_exception():
    """Verify that when AI raises an exception, the local ML result is safely returned."""
    base_mock = AsyncMock()
    base_mock.classify.return_value = ClassificationResult(
        is_passenger_request=False,
        confidence=0.55,
        reason="ML ambiguous",
        method="ml",
    )

    ai_mock = AsyncMock()
    ai_mock.classify.side_effect = RuntimeError("DeepSeek API timeout")

    hybrid = HybridClassifier(
        base_classifier=base_mock,
        ai_classifier=ai_mock,
        enable_ai=True,
        threshold=0.85,
    )

    res = await hybrid.classify("Some message")
    assert res.is_passenger_request is False
    assert res.confidence == 0.55
    assert res.method == "ml"
