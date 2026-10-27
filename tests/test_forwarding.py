"""Unit tests for ForwardingService and DeduplicationCache."""

import pytest
from unittest.mock import AsyncMock
from aiogram.exceptions import TelegramAPIError, TelegramRetryAfter
from aiogram import types

from app.config import Settings
from app.services.classification.base import ClassificationResult
from app.services.forwarding import DeduplicationCache, ForwardingService


def test_deduplication_cache_basic():
    cache = DeduplicationCache(ttl_seconds=60)
    user_id = 12345
    text = "Toshkentdan Samarqandga 2 ta joy kerak"

    assert cache.is_duplicate(user_id, text) is False
    cache.record(user_id, text)
    assert cache.is_duplicate(user_id, text) is True

    # Different user with same message -> Not duplicate
    assert cache.is_duplicate(67890, text) is False

    # Same user with different message -> Not duplicate
    assert cache.is_duplicate(user_id, "Andijonga ketadigan bormi?") is False


def test_deduplication_cross_script_equivalence():
    """Verify that Cyrillic and Latin versions of the same request are deduplicated."""
    cache = DeduplicationCache(ttl_seconds=60)
    user_id = 9999

    lat_text = "Toshkentdan Samarqandga 2 ta joy kerak"
    cyr_text = "Тошкентдан Самарқандга 2 та жой керак"

    assert cache.is_duplicate(user_id, lat_text) is False
    cache.record(user_id, lat_text)
    assert cache.is_duplicate(user_id, cyr_text) is True


def test_build_passenger_card():
    settings = Settings(
        BOT_TOKEN="123456:fake_token",
        ADMIN_USER_IDS="111",
        PRIVATE_GROUP_ID=-1004465107293,
    )
    forwarder = ForwardingService(bot=None, settings=settings)

    chat = types.Chat(id=-1003726192046, type="supergroup")
    user = types.User(id=777, is_bot=False, first_name="Ali", username="ali_valiyev")
    msg = types.Message(
        message_id=45,
        date=1700000000,
        chat=chat,
        from_user=user,
        text="Toshkentdan Samarqandga 2 kishi ketamiz",
    )

    result = ClassificationResult(
        is_passenger_request=True,
        confidence=0.92,
        origin="Toshkent",
        destination="Samarqand",
        seats=2,
        reason="Test",
        method="rule_based",
    )

    card_text, reply_markup = forwarder.build_passenger_card(msg, result)

    assert "YANGI YO‘LOVCHI" in card_text
    assert "Toshkent → Samarqand" in card_text
    assert "2 kishi" in card_text
    assert "aniqlanmagan" in card_text
    assert "<blockquote>" in card_text
    # Confidence is excluded per user specification
    assert "%" not in card_text

    buttons = reply_markup.inline_keyboard
    assert len(buttons) >= 2
    assert "https://t.me/ali_valiyev" in buttons[0][0].url
    assert "https://t.me/c/3726192046/45" in buttons[1][0].url


@pytest.mark.asyncio
async def test_telegram_api_error_handling():
    """Verify that Telegram API errors are caught gracefully without uncaught exceptions."""
    mock_bot = AsyncMock()
    mock_bot.send_message.side_effect = TelegramAPIError(method="sendMessage", message="Chat not found")

    settings = Settings(
        BOT_TOKEN="123456:fake",
        ADMIN_USER_IDS="111",
        PRIVATE_GROUP_ID=-1004465107293,
    )
    forwarder = ForwardingService(bot=mock_bot, settings=settings)

    chat = types.Chat(id=-1003726192046, type="supergroup")
    msg = types.Message(
        message_id=99,
        date=1700000000,
        chat=chat,
        from_user=types.User(id=555, is_bot=False, first_name="Test"),
        text="Samarqandga taksi kerak",
    )
    result = ClassificationResult(
        is_passenger_request=True,
        confidence=0.85,
        destination="Samarqand",
    )

    # Should return False on API error, not crash
    success = await forwarder.forward_passenger_request(msg, result)
    assert success is False
