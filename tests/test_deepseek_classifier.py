"""Unit tests for DeepSeekClassifier."""

import pytest
from unittest.mock import AsyncMock, patch
from app.services.classification.deepseek_classifier import DeepSeekClassifier
from app.services.classification.base import BaseClassifier, ClassificationResult


class DummyFallbackClassifier(BaseClassifier):
    async def classify(self, raw_text: str) -> ClassificationResult:
        return ClassificationResult(
            is_passenger_request=False,
            confidence=0.55,
            reason="Fallback rule decision",
            method="rule_based",
        )


@pytest.mark.asyncio
async def test_deepseek_short_text():
    classifier = DeepSeekClassifier(api_key="fake_key")
    res = await classifier.classify("yo")
    assert not res.is_passenger_request
    assert res.method == "pre_filter"


@pytest.mark.asyncio
async def test_deepseek_strict_driver_rejected():
    classifier = DeepSeekClassifier(api_key="fake_key")
    # "tom bagaj bor" should be rejected immediately before API call
    res = await classifier.classify("Kobalt bor Toshkentga tom bagaj bor 2 ta kam")
    assert not res.is_passenger_request
    assert res.confidence >= 0.95
    assert res.method == "rule_based"

    # Driver picked up passengers announcements
    res2 = await classifier.classify("Baland balandka, ikkinchi so'rovga ham kelishmasa bizda ikkita odam bor. Ereportdan alib chiqdi goji.")
    assert not res2.is_passenger_request
    assert res2.method == "rule_based"

    res3 = await classifier.classify("boldi boldi bizda, 2 ta odam bor agar kelishalmasa ayraportdan olib chiqdik uje")
    assert not res3.is_passenger_request
    assert res3.method == "rule_based"

    # Inquiries and quoting previous messages
    res4 = await classifier.classify("Odam bor ddzmi shooir aka")
    assert not res4.is_passenger_request
    assert res4.method == "rule_based"

    res5 = await classifier.classify("Odam bor dedizmi shopir aka")
    assert not res5.is_passenger_request
    assert res5.method == "rule_based"

    # Real estate / apartment rental / non-transport
    res6 = await classifier.classify("Assalomu aleykum shuhrat chorraxa atrofidan kv kerak 1 kishiga")
    assert not res6.is_passenger_request
    assert res6.method == "rule_based"

    # Driver soliciting parcels / offering transport
    res7 = await classifier.classify("Xozrga Toshkonga Bogdod Rishton Buvaydadan Zaril pochtalar bolsa aloqadaman +998903060343")
    assert not res7.is_passenger_request
    assert res7.method == "rule_based"


@pytest.mark.asyncio
async def test_deepseek_success_passenger():
    classifier = DeepSeekClassifier(api_key="fake_key")

    mock_json = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"is_order": true, "order_type": "passenger", "confidence": 0.96, '
                        '"origin": "Toshkent", "destination": "Samarqand", "seats": 2, '
                        '"time": "hozir", "reason": "Passenger request"}'
                    )
                }
            }
        ]
    }

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = mock_json
        mock_post.return_value.__aenter__.return_value = mock_resp

        res = await classifier.classify("Toshkentdan Samarqandga 2 kishi bor hozir ketishga")
        assert res.is_passenger_request is True
        assert res.order_type == "passenger"
        assert res.origin == "Toshkent"
        assert res.destination == "Samarqand"
        assert res.seats == 2
        assert res.time == "hozir"
        assert res.method == "deepseek"


@pytest.mark.asyncio
async def test_deepseek_success_cargo():
    classifier = DeepSeekClassifier(api_key="fake_key")

    mock_json = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"is_order": true, "order_type": "cargo", "confidence": 0.94, '
                        '"origin": "Qo\'qon", "destination": "Toshkent", "seats": null, '
                        '"time": null, "reason": "Parcel request"}'
                    )
                }
            }
        ]
    }

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = mock_json
        mock_post.return_value.__aenter__.return_value = mock_resp

        res = await classifier.classify("Qo'qondan Toshkentga pochta bor berib yuborish kerak")
        assert res.is_passenger_request is True
        assert res.order_type == "cargo"
        assert res.origin == "Qo'qon"
        assert res.destination == "Toshkent"
        assert res.method == "deepseek"


@pytest.mark.asyncio
async def test_deepseek_fallback_on_error():
    fallback = DummyFallbackClassifier()
    classifier = DeepSeekClassifier(api_key="fake_key", fallback_classifier=fallback)

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_resp.text.return_value = "Server error"
        mock_post.return_value.__aenter__.return_value = mock_resp

        res = await classifier.classify("Toshkentga 1 kishi kerak")
        assert res.method == "rule_based"
        assert res.reason == "Fallback rule decision"
