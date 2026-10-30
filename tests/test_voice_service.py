"""Unit tests for VoiceTranscriptionService and voice card generation."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.voice_service import VoiceTranscriptionService, clean_voice_transcription
from app.services.forwarding import ForwardingService
from app.services.classification.base import ClassificationResult
from app.config import Settings
from aiogram import types


def test_clean_voice_transcription():
    # Test mapping phonetic foreign chars and regional dialect phrases
    noisy = "Šabrakil ə, Štabtomanda yürgəllərgə, bittay ayar kisi bar, yayin qırvan, dhatqa gevaradə."
    cleaned = clean_voice_transcription(noisy)
    assert "ayol kishi" in cleaned
    assert "bor" in cleaned
    assert "Yangiqo'rg'on" in cleaned
    assert "ə" not in cleaned
    assert "Š" not in cleaned
    assert "ı" not in cleaned

    # Test speech-to-text noisy transcriptions for TashMI, Buvayda, Bachqir
    noisy2 = "Mana shin umerda yangi tashmi atrafudan, bitta odam bor. Bu veydi, bachqray gelad, ha?"
    cleaned2 = clean_voice_transcription(noisy2)
    assert "shu nomerda" in cleaned2
    assert "Yangi ToshMI" in cleaned2
    assert "Buvayda" in cleaned2
    assert "Bachqirga" in cleaned2
    assert "keladi" in cleaned2


@pytest.mark.asyncio
async def test_voice_transcription_empty():
    service = VoiceTranscriptionService(groq_api_key="fake", enabled=True)
    assert await service.transcribe(b"") is None
    assert await service.transcribe(b"short") is None


@pytest.mark.asyncio
async def test_voice_transcription_disabled():
    service = VoiceTranscriptionService(groq_api_key="fake", enabled=False)
    assert await service.transcribe(b"dummy" * 50) is None


@pytest.mark.asyncio
async def test_voice_transcription_groq_success():
    service = VoiceTranscriptionService(groq_api_key="gsk_test123", enabled=True)

    with patch.object(service, "_refine_with_llm", new=AsyncMock(return_value="Toshkentga 2 kishi bor ketadigan")):
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json.return_value = {"text": "Toshkentga 2 kishi bor ketadigan"}
            mock_post.return_value.__aenter__.return_value = mock_resp

            result = await service.transcribe(b"fake_audio_content" * 20, filename="voice.ogg")
            assert result == "Toshkentga 2 kishi bor ketadigan"


@pytest.mark.asyncio
async def test_voice_refine_with_llm_success():
    service = VoiceTranscriptionService(groq_api_key="gsk_test123", enabled=True)

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Bering menda ayol kishi bor, 4-maktab yonida turibman"}}]
        }
        mock_post.return_value.__aenter__.return_value = mock_resp

        refined = await service._refine_with_llm("Bira yim enda ayol kishi bor, torti 2 mektad, yuanda turib bor")
        assert refined == "Bering menda ayol kishi bor, 4-maktab yonida turibman"


def test_build_voice_card():
    settings = Settings(
        BOT_TOKEN="123:fake",
        PRIVATE_GROUP_ID=-1004465107293,
    )
    forwarder = ForwardingService(bot=None, settings=settings)

    chat = types.Chat(id=-10012345, type="supergroup")
    user = types.User(id=777, is_bot=False, first_name="Ali", username="ali_uz")
    msg = types.Message(
        message_id=10,
        date=1700000000,
        chat=chat,
        from_user=user,
    )

    result = ClassificationResult(
        is_passenger_request=True,
        confidence=0.92,
        origin="Toshkent",
        destination="Samarqand",
        seats=2,
        reason="Voice order",
    )

    card_text, reply_markup = forwarder.build_passenger_card(
        message=msg,
        classification=result,
        is_voice=True,
        raw_text="Toshkentdan Samarqandga 2 kishi bor ketadigan",
    )

    assert "YANGI YO‘LOVCHI" in card_text
    assert "Toshkent → Samarqand" in card_text
    assert "Ovozli xabar" in card_text
    assert "Ovozli xabar matni" in card_text
    assert "Ali" in card_text
