"""Unit tests for ExpirationScheduler background jobs."""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.config import Settings
from app.database.base import Base
from app.database.models.driver import Driver, SubscriptionStatus
from app.repositories.driver_repo import DriverRepository
from app.services.scheduler import ExpirationScheduler


@pytest_asyncio.fixture
async def test_env(monkeypatch):
    """Setup test database and mock bot."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Patch get_db_session in scheduler to use this test session maker
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def mock_get_session():
        async with session_maker() as session:
            yield session
            await session.commit()

    monkeypatch.setattr("app.services.scheduler.get_db_session", mock_get_session)

    mock_bot = AsyncMock()
    settings = Settings(
        BOT_TOKEN="123456:fake",
        ADMIN_USER_IDS="8120241913",
        PRIVATE_GROUP_ID=-1004465107293,
    )

    scheduler = ExpirationScheduler(bot=mock_bot, settings=settings)

    yield scheduler, session_maker, mock_bot

    await engine.dispose()


@pytest.mark.asyncio
async def test_scheduler_evicts_expired_drivers(test_env):
    scheduler, session_maker, mock_bot = test_env

    # 1. Seed database with 1 active driver and 1 already expired driver
    now = datetime.now(timezone.utc)
    async with session_maker() as session:
        repo = DriverRepository(session)

        # Active driver (expires in 20 days)
        d1 = await repo.create_or_update(telegram_id=111, full_name="Active Driver")
        await repo.activate_subscription(telegram_id=111, duration_days=20)

        # Expired driver (expired 2 days ago)
        d2 = await repo.create_or_update(telegram_id=222, full_name="Expired Driver")
        await repo.activate_subscription(telegram_id=222, duration_days=30)
        d2.end_date = now - timedelta(days=2)

        await session.commit()

    # 2. Run expiration check
    evicted_count = await scheduler.check_expired_subscriptions()

    # 3. Verify exactly 1 driver was evicted
    assert evicted_count == 1

    # Verify bot called ban_chat_member for driver 222
    mock_bot.ban_chat_member.assert_called_once_with(
        chat_id=-1004465107293,
        user_id=222,
    )

    # Verify database status is updated to EXPIRED
    async with session_maker() as session:
        repo = DriverRepository(session)
        d1_check = await repo.get_by_telegram_id(111)
        d2_check = await repo.get_by_telegram_id(222)

        assert d1_check.status == SubscriptionStatus.ACTIVE.value
        assert d2_check.status == SubscriptionStatus.EXPIRED.value
