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
    assert "Yo‘lovchi:" in card_text
    assert "Ali" in card_text

    buttons = reply_markup.inline_keyboard
    assert len(buttons) == 2
    assert "https://t.me/ali_valiyev" in buttons[0][0].url
    assert "💬 Yo'lovchiga yozish" == buttons[0][0].text
    assert "📍 Guruhdagi xabarga o'tish" == buttons[1][0].text
    assert "https://t.me/c/3726192046/45" in buttons[1][0].url


def test_build_passenger_card_without_username():
    """Verify card generated for user without username has direct tg://user?id profile link and group message button."""
    settings = Settings(
        BOT_TOKEN="123456:fake_token",
        ADMIN_USER_IDS="111",
        PRIVATE_GROUP_ID=-1004465107293,
    )
    forwarder = ForwardingService(bot=None, settings=settings)

    chat = types.Chat(id=-1003726192046, type="supergroup")
    user = types.User(id=8623889331, is_bot=False, first_name="M-CBDE", username=None)
    msg = types.Message(
        message_id=50,
        date=1700000000,
        chat=chat,
        from_user=user,
        text="Andijondan Toshkentga odam bor +998901234567",
    )

    result = ClassificationResult(
        is_passenger_request=True,
        confidence=0.92,
        origin="Andijon",
        destination="Toshkent",
        seats=1,
        reason="Test",
        method="rule_based",
    )

    card_text, reply_markup = forwarder.build_passenger_card(msg, result)

    assert "YANGI YO‘LOVCHI" in card_text
    assert "Andijon → Toshkent" in card_text
    assert "1 kishi" in card_text
    assert 'tg://user?id=8623889331' in card_text
    assert 'M-CBDE' in card_text
    assert '+998901234567' in card_text

    # For users without username, group message link button is attached
    assert reply_markup is not None
    buttons = reply_markup.inline_keyboard
    assert len(buttons) == 1
    assert "📍 Guruhdagi xabarga o'tish" == buttons[0][0].text
    assert "https://t.me/c/3726192046/50" in buttons[0][0].url


def test_build_cargo_card():
    """Verify card generated for cargo / parcel customer request."""
    settings = Settings(
        BOT_TOKEN="123456:fake_token",
        ADMIN_USER_IDS="111",
        PRIVATE_GROUP_ID=-1004465107293,
    )
    forwarder = ForwardingService(bot=None, settings=settings)

    chat = types.Chat(id=-1003726192046, type="supergroup")
    user = types.User(id=999, is_bot=False, first_name="Nodir", username="nodir_pochta")
    msg = types.Message(
        message_id=60,
        date=1700000000,
        chat=chat,
        from_user=user,
        text="Toshkentdan Fargonaga pochta bor berib yuborish kerak +998901234567",
    )

    result = ClassificationResult(
        is_passenger_request=True,
        confidence=0.95,
        origin="Toshkent",
        destination="Fargona",
        order_type="cargo",
        reason="Pochta request",
        method="rule_based",
    )

    card_text, reply_markup = forwarder.build_passenger_card(msg, result)

    assert "YANGI POCHTA / YUK" in card_text
    assert "Toshkent → Fargona" in card_text
    assert "Buyurtmachi:" in card_text
    assert "Pochta / Yuk" in card_text
    assert "+998901234567" in card_text
    buttons = reply_markup.inline_keyboard
    assert len(buttons) == 2
    assert "💬 Buyurtmachiga yozish" == buttons[0][0].text
    assert "https://t.me/nodir_pochta" in buttons[0][0].url
    assert "📍 Guruhdagi xabarga o'tish" == buttons[1][0].text
    assert "https://t.me/c/3726192046/60" in buttons[1][0].url


@pytest.mark.asyncio
async def test_telegram_api_error_handling():
    """Verify that Telegram API errors are caught gracefully without uncaught exceptions."""
    mock_bot = AsyncMock()
    mock_bot.send_message.side_effect = TelegramAPIError(method="sendMessage", message="Chat not found")
    mock_bot.forward_message.side_effect = TelegramAPIError(method="forwardMessage", message="Chat not found")

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


@pytest.mark.asyncio
async def test_forward_voice_dispatches_both_audio_and_card():
    """Verify that voice passenger requests forward the voice message AND also send the structured card."""
    mock_bot = AsyncMock()
    settings = Settings(
        BOT_TOKEN="123456:fake",
        PRIVATE_GROUP_ID=-1004465107293,
    )
    forwarder = ForwardingService(bot=mock_bot, settings=settings)

    chat = types.Chat(id=-1003726192046, type="supergroup")
    msg = types.Message(
        message_id=123,
        date=1700000000,
        chat=chat,
        from_user=types.User(id=987, is_bot=False, first_name="VoiceUser"),
        text=None,
    )
    result = ClassificationResult(
        is_passenger_request=True,
        confidence=0.95,
        destination="Toshkent",
        order_type="passenger",
    )

    success = await forwarder.forward_passenger_request(
        message=msg,
        classification=result,
        is_voice=True,
        raw_text="Toshkentga bitta ayol kishi bor",
        voice_bytes=b"fake_audio_bytes",
    )

    assert success is True
    # 1. Voice audio forwarded/sent
    assert mock_bot.forward_message.called or mock_bot.send_voice.called
    # 2. Card ALSO sent
    assert mock_bot.send_message.called
    sent_text = mock_bot.send_message.call_args[1]["text"]
    assert "Toshkent" in sent_text
    assert "Ovozli xabar" in sent_text


@pytest.mark.asyncio
async def test_forward_custom_voice_userbot_dispatches_both():
    """Verify that Userbot captured voice requests forward audio AND send card."""
    mock_bot = AsyncMock()
    mock_userbot_client = AsyncMock()
    settings = Settings(
        BOT_TOKEN="123456:fake",
        PRIVATE_GROUP_ID=-1004465107293,
    )
    forwarder = ForwardingService(bot=mock_bot, settings=settings)

    result = ClassificationResult(
        is_passenger_request=True,
        confidence=0.92,
        destination="Rishton",
        order_type="passenger",
    )

    success = await forwarder.forward_custom_passenger_request(
        chat_id=-100111222,
        message_id=555,
        user_id=777,
        raw_text="Rishtonga 2 kishi bor",
        classification=result,
        passenger_name="Ovozli Buyurtmachi",
        is_voice=True,
        voice_bytes=b"ogg_voice_data",
        userbot_client=mock_userbot_client,
        telethon_message=AsyncMock(),
    )

    assert success is True
    # 1. Audio forwarded via userbot or bot
    assert mock_userbot_client.forward_messages.called or mock_bot.forward_message.called or mock_bot.send_voice.called
    # 2. Card ALSO sent via bot
    assert mock_bot.send_message.called
    sent_text = mock_bot.send_message.call_args[1]["text"]
    assert "Rishton" in sent_text
    assert "Ovozli xabar" in sent_text
