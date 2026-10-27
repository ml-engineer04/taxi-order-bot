"""Unit tests for Telethon Userbot client, listener, and forwarding pipeline."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.config import Settings
from app.services.classification.base import ClassificationResult
from app.services.forwarding import DeduplicationCache, ForwardingService
from app.userbot.client import create_userbot_client
from app.userbot.listener import is_chat_monitored, setup_userbot_listener


def test_is_chat_monitored():
    # Empty public group -> monitors everything
    s_empty = Settings(BOT_TOKEN="123:abc", PUBLIC_GROUP_ID="")
    assert is_chat_monitored(-1003726192046, s_empty) is True
    assert is_chat_monitored(12345, s_empty) is True

    # Configured with negative supergroup IDs
    s_configured = Settings(
        BOT_TOKEN="123:abc",
        PUBLIC_GROUP_ID="-1003726192046, -1002213849094",
    )
    # Direct match with -100 prefix
    assert is_chat_monitored(-1003726192046, s_configured) is True
    assert is_chat_monitored(-1002213849094, s_configured) is True

    # Variations without -100 prefix
    assert is_chat_monitored(3726192046, s_configured) is True
    assert is_chat_monitored(-3726192046, s_configured) is True

    # Unmonitored chat
    assert is_chat_monitored(-1009999999999, s_configured) is False
    assert is_chat_monitored(123456789, s_configured) is False


def test_create_userbot_client():
    # Missing credentials -> returns None
    s_no_creds = Settings(BOT_TOKEN="123:abc", TG_API_ID=None, TG_API_HASH=None)
    assert create_userbot_client(s_no_creds) is None

    # Credentials provided -> returns TelegramClient instance
    s_creds = Settings(
        BOT_TOKEN="123:abc",
        TG_API_ID=36827938,
        TG_API_HASH="d9622c8700c748fa10a65a5f05abf949",
        USERBOT_SESSION_NAME="data/test_userbot.session",
    )
    client = create_userbot_client(s_creds)
    assert client is not None
    assert client.api_id == 36827938
    assert client.api_hash == "d9622c8700c748fa10a65a5f05abf949"


@pytest.mark.asyncio
async def test_forward_custom_passenger_request():
    mock_bot = AsyncMock()
    settings = Settings(
        BOT_TOKEN="123:abc",
        PRIVATE_GROUP_ID=-1004465107293,
        FORWARD_MODE="copy",
    )
    cache = DeduplicationCache(ttl_seconds=60)
    forwarder = ForwardingService(bot=mock_bot, settings=settings, deduplicator=cache)

    result = ClassificationResult(
        is_passenger_request=True,
        confidence=0.95,
        origin="Toshkent",
        destination="Buxoro",
        seats=3,
        reason="Passenger detected",
    )

    success = await forwarder.forward_custom_passenger_request(
        chat_id=-1003726192046,
        message_id=500,
        user_id=1234567,
        raw_text="Toshkentdan Buxoroga 3 kishi bor taksi bormi",
        classification=result,
        passenger_name="Sardor Aliyev",
        passenger_username="sardor_a",
        chat_username="toshkent_taxi_chat",
    )

    assert success is True
    assert mock_bot.forward_message.call_count == 1 or mock_bot.send_message.call_count == 1
    call_kwargs = mock_bot.send_message.call_args[1]
    assert call_kwargs["chat_id"] == -1004465107293
    assert "YANGI YO‘LOVCHI" in call_kwargs["text"]
    assert "Toshkent → Buxoro" in call_kwargs["text"]
    assert "3 kishi" in call_kwargs["text"]
    assert "aniqlanmagan" in call_kwargs["text"]

    # Duplicate identical message within 3 mins should be suppressed
    dup_success = await forwarder.forward_custom_passenger_request(
        chat_id=-1003726192046,
        message_id=501,
        user_id=1234567,
        raw_text="Toshkentdan Buxoroga 3 kishi bor taksi bormi",
        classification=result,
        passenger_name="Sardor Aliyev",
        passenger_username="sardor_a",
    )
    assert dup_success is False
    assert mock_bot.send_message.call_count == 1


@pytest.mark.asyncio
async def test_userbot_listener_events():
    mock_client = MagicMock()
    registered_handler = None

    def on_decorator(event_builder):
        def wrapper(func):
            nonlocal registered_handler
            registered_handler = func
            return func
        return wrapper

    mock_client.on = on_decorator

    mock_bot = AsyncMock()
    mock_forwarder = MagicMock()
    mock_forwarder.forward_custom_passenger_request = AsyncMock(return_value=True)

    mock_classifier = AsyncMock()
    mock_classifier.classify = AsyncMock(
        return_value=ClassificationResult(
            is_passenger_request=True,
            confidence=0.94,
            origin="Toshkent",
            destination="Samarqand",
            seats=1,
        )
    )

    settings = Settings(
        BOT_TOKEN="123:abc",
        PUBLIC_GROUP_ID="-1003726192046",
        PRIVATE_GROUP_ID=-1004465107293,
    )

    setup_userbot_listener(
        client=mock_client,
        bot=mock_bot,
        settings=settings,
        classifier=mock_classifier,
        forwarder=mock_forwarder,
    )

    assert registered_handler is not None

    # Test 1: Ignore outgoing message from self
    event_out = MagicMock()
    event_out.out = True
    event_out.is_private = False
    await registered_handler(event_out)
    assert mock_forwarder.forward_custom_passenger_request.call_count == 0

    # Test 2: Ignore private DM
    event_private = MagicMock()
    event_private.out = False
    event_private.is_private = True
    await registered_handler(event_private)
    assert mock_forwarder.forward_custom_passenger_request.call_count == 0

    # Test 3: Ignore unmonitored group
    event_unmonitored = MagicMock()
    event_unmonitored.out = False
    event_unmonitored.is_private = False
    event_unmonitored.chat_id = -1009999999999
    await registered_handler(event_unmonitored)
    assert mock_forwarder.forward_custom_passenger_request.call_count == 0

    # Test 4: Passenger message in monitored group
    event_valid = MagicMock()
    event_valid.out = False
    event_valid.is_private = False
    event_valid.chat_id = -1003726192046
    event_valid.id = 888
    event_valid.raw_text = "Toshkentdan Samarqandga 1 kishi bor"
    event_valid.sender_id = 998877

    sender_mock = MagicMock()
    sender_mock.bot = False
    sender_mock.first_name = "Jamshid"
    sender_mock.last_name = "Karimov"
    sender_mock.username = "jamshid_k"
    event_valid.get_sender = AsyncMock(return_value=sender_mock)

    chat_mock = MagicMock()
    chat_mock.username = "toshkent_taxi_supergroup"
    chat_mock.title = "Toshkent Taksi"
    event_valid.get_chat = AsyncMock(return_value=chat_mock)

    await registered_handler(event_valid)
    assert mock_forwarder.forward_custom_passenger_request.call_count == 1
    call_args = mock_forwarder.forward_custom_passenger_request.call_args[1]
    assert call_args["chat_id"] == -1003726192046
    assert call_args["message_id"] == 888
    assert call_args["user_id"] == 998877
    assert call_args["passenger_name"] == "Jamshid Karimov"
    assert call_args["passenger_username"] == "jamshid_k"
    assert call_args["chat_username"] == "toshkent_taxi_supergroup"


@pytest.mark.asyncio
async def test_historical_messages_skipped():
    """Verify that messages dated before userbot startup or older than 2 minutes are strictly ignored."""
    from datetime import datetime, timedelta, timezone

    mock_client = MagicMock()
    registered_handler = None

    def on_decorator(event_builder):
        def wrapper(func):
            nonlocal registered_handler
            registered_handler = func
            return func
        return wrapper

    mock_client.on = on_decorator
    mock_bot = AsyncMock()
    mock_forwarder = MagicMock()
    mock_forwarder.is_user_on_cooldown.return_value = False
    mock_forwarder.forward_custom_passenger_request = AsyncMock(return_value=True)
    mock_classifier = AsyncMock()

    settings = Settings(
        BOT_TOKEN="123:abc",
        PUBLIC_GROUP_ID="-1003726192046",
        PRIVATE_GROUP_ID=-1004465107293,
    )

    setup_userbot_listener(
        client=mock_client,
        bot=mock_bot,
        settings=settings,
        classifier=mock_classifier,
        forwarder=mock_forwarder,
    )

    # 1. Message dated 1 hour ago (historical backlog)
    event_old = MagicMock()
    event_old.out = False
    event_old.is_private = False
    event_old.chat_id = -1003726192046
    event_old.id = 101
    event_old.raw_text = "Toshkentdan Samarqandga 2 kishi bor"
    event_old.message.date = datetime.now(timezone.utc) - timedelta(hours=1)
    event_old.sender_id = 112233

    await registered_handler(event_old)
    # Must NOT classify or forward historical messages
    assert mock_classifier.classify.call_count == 0
    assert mock_forwarder.forward_custom_passenger_request.call_count == 0

    # 2. Live brand new message (dated right now)
    event_live = MagicMock()
    event_live.out = False
    event_live.is_private = False
    event_live.chat_id = -1003726192046
    event_live.id = 102
    event_live.raw_text = "Toshkentdan Samarqandga 2 kishi bor"
    event_live.message.date = datetime.now(timezone.utc)
    event_live.sender_id = 112233
    event_live.get_sender = AsyncMock(return_value=MagicMock(bot=False, first_name="Ali", last_name="", username="ali"))
    event_live.get_chat = AsyncMock(return_value=MagicMock(username="taksi", title="Taksi"))
    mock_classifier.classify.return_value = ClassificationResult(
        is_passenger_request=True,
        confidence=0.9,
        destination="Samarqand",
    )

    await registered_handler(event_live)
    assert mock_classifier.classify.call_count == 1
    assert mock_forwarder.forward_custom_passenger_request.call_count == 1


@pytest.mark.asyncio
async def test_user_cooldown_disabled():
    """Verify that 24h user blocking is disabled (users are NOT blocked from ordering again)."""
    settings = Settings(BOT_TOKEN="123:abc")
    forwarder = ForwardingService(bot=AsyncMock(), settings=settings)

    # Any user should NOT be on 24h cooldown
    assert forwarder.is_user_on_cooldown(554433) is False
    assert forwarder.get_user_remaining_cooldown_hours(554433) == 0.0


@pytest.mark.asyncio
async def test_auto_join_and_resolve_targets():
    """Verify that auto_join_and_resolve_targets parses IDs, usernames, and invite links."""
    from app.userbot.listener import auto_join_and_resolve_targets

    mock_client = AsyncMock()

    # Mock entity lookup for @public_group
    mock_entity = MagicMock()
    mock_entity.id = 1122334455
    mock_client.get_entity.return_value = mock_entity

    settings = Settings(
        BOT_TOKEN="123:abc",
        PUBLIC_GROUP_ID="-1003726192046, @toshkent_taksi, https://t.me/+fake_invite_hash",
    )

    resolved = await auto_join_and_resolve_targets(mock_client, settings)

    assert -1003726192046 in resolved
    assert 3726192046 in resolved
    assert 1122334455 in resolved

