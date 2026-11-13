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


@pytest.mark.asyncio
async def test_hybrid_ttl_cache_prevents_duplicate_ai_calls():
    """Verify repeated identical texts hit TTL cache and do not re-call AI."""
    base_mock = AsyncMock()
    base_mock.classify.return_value = ClassificationResult(
        is_passenger_request=False,
        confidence=0.60,
        reason="ML OTHER/irrelevant (confidence: 0.60)",
        method="ml",
        predicted_label="OTHER",
    )

    ai_mock = AsyncMock()
    ai_mock.classify.return_value = ClassificationResult(
        is_passenger_request=False,
        confidence=0.20,
        reason="Driver ad",
        method="deepseek",
    )

    hybrid = HybridClassifier(
        base_classifier=base_mock,
        ai_classifier=ai_mock,
        enable_ai=True,
        threshold=0.85,
    )

    msg = "Toshkentdan Qaytishga 1 odam kam"
    r1 = await hybrid.classify(msg)
    r2 = await hybrid.classify(msg)
    assert r1.is_passenger_request is False
    assert r2.is_passenger_request is False
    assert ai_mock.classify.call_count == 1
    assert base_mock.classify.call_count == 1


@pytest.mark.asyncio
async def test_hybrid_timeout_preserves_high_conf_ml_order():
    """If DeepSeek fails and delegates to non-AI fallback, keep ML result (not fallback)."""
    base_mock = AsyncMock()
    base_mock.classify.return_value = ClassificationResult(
        is_passenger_request=True,
        confidence=0.75,
        origin=None,
        destination=None,
        reason="ML PASSENGER_REQUEST (confidence: 0.75)",
        method="ml",
        predicted_label="PASSENGER_REQUEST",
    )

    ai_mock = AsyncMock()
    # Simulates DeepSeek timeout falling back to a lower-confidence rule_based result
    ai_mock.classify.return_value = ClassificationResult(
        is_passenger_request=False,
        confidence=0.50,
        reason="Rule fallback",
        method="rule_based",
    )

    hybrid = HybridClassifier(
        base_classifier=base_mock,
        ai_classifier=ai_mock,
        enable_ai=True,
        threshold=0.85,
    )

    res = await hybrid.classify("+998979600650 bitta joy kerak")
    assert res.is_passenger_request is True
    assert res.confidence == 0.75
    assert res.method == "ml"

    # Transient AI failure should NOT be cached so next attempt can query AI again
    await hybrid.classify("+998979600650 bitta joy kerak")
    assert ai_mock.classify.call_count == 2


@pytest.mark.asyncio
async def test_hybrid_single_city_with_explicit_intent_skips_ai():
    """Confident ML (>=0.88) with one city and explicit passenger intent skips DeepSeek entirely."""
    base_mock = AsyncMock()
    base_mock.classify.return_value = ClassificationResult(
        is_passenger_request=True,
        confidence=0.95,
        origin="Toshkent",
        destination=None,
        reason="ML PASSENGER_REQUEST (confidence: 0.95)",
        method="ml",
        predicted_label="PASSENGER_REQUEST",
    )

    ai_mock = AsyncMock()
    hybrid = HybridClassifier(
        base_classifier=base_mock,
        ai_classifier=ai_mock,
        enable_ai=True,
        threshold=0.85,
    )

    # 1. Single city + explicit passenger syntax -> skips AI
    res = await hybrid.classify("Toshkentdan 1 kishi bor hozirga")
    assert res.is_passenger_request is True
    assert res.method == "ml"
    assert ai_mock.classify.call_count == 0

    # 2. Single city + cargo syntax -> skips AI
    res2 = await hybrid.classify("+998979600650 Toshkentdan pochta bor")
    assert res2.is_passenger_request is True
    assert res2.method == "ml"
    assert ai_mock.classify.call_count == 0


@pytest.mark.asyncio
async def test_hybrid_asymmetric_reject_thresholds():
    """Confident OTHER (>=0.82) and DRIVER_OFFER (>=0.72) skip AI, while low-conf or below-thresh PASSENGER queries AI."""
    ai_mock = AsyncMock()
    ai_mock.classify.return_value = ClassificationResult(
        is_passenger_request=True,
        confidence=0.90,
        origin="Toshkent",
        destination="Samarqand",
        method="deepseek",
    )

    # 1. ML DRIVER_OFFER at 0.75 (>= 0.72) -> skipped AI
    base_mock = AsyncMock()
    base_mock.classify.return_value = ClassificationResult(
        is_passenger_request=False,
        confidence=0.75,
        reason="ML DRIVER_OFFER",
        method="ml",
        predicted_label="DRIVER_OFFER",
    )
    hybrid = HybridClassifier(base_classifier=base_mock, ai_classifier=ai_mock, enable_ai=True)
    r = await hybrid.classify("Moshina bor Samarqandga")
    assert r.is_passenger_request is False
    ai_mock.classify.assert_not_called()

    # 2. ML OTHER at 0.83 (>= 0.82) -> skipped AI
    base_mock.classify.return_value = ClassificationResult(
        is_passenger_request=False,
        confidence=0.83,
        reason="ML OTHER",
        method="ml",
        predicted_label="OTHER",
    )
    r2 = await hybrid.classify("Assalomu alaykum yaxshimisizlar")
    assert r2.is_passenger_request is False
    ai_mock.classify.assert_not_called()

    # 3. ML PASSENGER_REQUEST below threshold (0.80 < 0.85) -> queries AI
    base_mock.classify.return_value = ClassificationResult(
        is_passenger_request=False,
        confidence=0.80,
        origin="Toshkent",
        destination="Samarqand",
        reason="ML PASSENGER_REQUEST below threshold",
        method="ml",
        predicted_label="PASSENGER_REQUEST",
    )
    r3 = await hybrid.classify("Samarqandga bir kishi ketmoqchi")
    assert r3.is_passenger_request is True
    assert r3.method == "deepseek"
    ai_mock.classify.assert_called_once()

