"""Unit tests for Admin command authorization and input validation."""

import pytest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from app.config import Settings
from app.handlers.admin import is_admin
from app.services.subscription import SubscriptionService


def test_admin_authorization():
    # Configured admin ID from .env
    assert is_admin(8120241913) is True

    # Unauthorized random user
    assert is_admin(999999999) is False
    assert is_admin(None) is False


def test_subscription_date_formatting():
    settings = Settings(
        BOT_TOKEN="123456:fake_token",
        ADMIN_USER_IDS="8120241913",
        TIMEZONE="Asia/Tashkent",
    )
    service = SubscriptionService(bot=None, settings=settings)

    # 2026-10-26 18:00 UTC == 2026-10-26 23:00 Tashkent (UTC+5)
    utc_dt = datetime(2026, 10, 26, 18, 0, 0, tzinfo=timezone.utc)
    formatted = service.to_local_string(utc_dt)

    assert formatted == "2026-10-26 23:00"
    assert service.to_local_string(None) == "Ko'rsatilmagan"


@pytest.mark.asyncio
async def test_handle_start_non_admin():
    from unittest.mock import AsyncMock, MagicMock
    from aiogram import types
    from app.handlers.admin import handle_start

    mock_msg = MagicMock(spec=types.Message)
    mock_msg.from_user = types.User(id=999888777, is_bot=False, first_name="Ali", last_name="Valiyev")
    mock_msg.bot = MagicMock()
    mock_msg.answer = AsyncMock()

    await handle_start(mock_msg)

    mock_msg.answer.assert_called_once()
    args, kwargs = mock_msg.answer.call_args
    sent_text = args[0]
    assert "@rahmonjon_qosimov" in sent_text
    assert "Ali Valiyev" in sent_text
    reply_markup = kwargs.get("reply_markup")
    assert reply_markup is not None
    assert "https://t.me/rahmonjon_qosimov" in reply_markup.inline_keyboard[0][0].url
