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
