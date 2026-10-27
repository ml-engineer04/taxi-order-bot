"""Database and Subscription Logic Unit Tests."""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta, timezone
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.database.base import Base
from app.database.models.driver import Driver, SubscriptionStatus
from app.repositories.driver_repo import DriverRepository


@pytest_asyncio.fixture
async def async_session():
    """In-memory SQLite async engine and session for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_maker() as session:
        yield session

    await engine.dispose()


@pytest.mark.asyncio
async def test_driver_registration(async_session: AsyncSession):
    repo = DriverRepository(async_session)
    driver = await repo.create_or_update(
        telegram_id=123456789,
        full_name="Botir Haydovchi",
        username="botir_taxi",
        phone_number="+998901234567",
    )
    await async_session.commit()

    assert driver.id is not None
    assert driver.telegram_id == 123456789
    assert driver.status == SubscriptionStatus.PENDING.value
    assert driver.start_date is None
    assert driver.end_date is None


@pytest.mark.asyncio
async def test_driver_activation(async_session: AsyncSession):
    repo = DriverRepository(async_session)
    await repo.create_or_update(telegram_id=987654321, full_name="Sardor Haydovchi")

    driver = await repo.activate_subscription(
        telegram_id=987654321,
        duration_days=30,
        approved_by_admin_id=8120241913,
        invite_link="https://t.me/+join_test_link",
    )
    await async_session.commit()

    assert driver.status == SubscriptionStatus.ACTIVE.value
    assert driver.start_date is not None
    assert driver.end_date is not None
    assert driver.invite_link == "https://t.me/+join_test_link"
    assert driver.approved_by_admin_id == 8120241913

    # Check remaining days
    assert driver.days_remaining() in (29, 30)
    assert driver.is_currently_active() is True


@pytest.mark.asyncio
async def test_subscription_renewal(async_session: AsyncSession):
    repo = DriverRepository(async_session)
    await repo.create_or_update(telegram_id=55555, full_name="Vali Haydovchi")
    await repo.activate_subscription(telegram_id=55555, duration_days=30)

    # Renew by adding 15 more days
    renewed = await repo.renew_subscription(telegram_id=55555, additional_days=15)
    await async_session.commit()

    assert renewed.days_remaining() in (44, 45)
    assert renewed.status == SubscriptionStatus.ACTIVE.value


@pytest.mark.asyncio
async def test_expired_detection(async_session: AsyncSession):
    repo = DriverRepository(async_session)
    await repo.create_or_update(telegram_id=77777, full_name="Muddati O'tgan Haydovchi")
    driver = await repo.activate_subscription(telegram_id=77777, duration_days=30)

    # Manually backdate end_date to yesterday
    driver.end_date = datetime.now(timezone.utc) - timedelta(days=1)
    await async_session.commit()

    assert driver.is_currently_active() is False
    assert driver.days_remaining() == 0

    # Test repository query for expired active drivers
    expired_list = await repo.get_expired_active_drivers()
    assert len(expired_list) == 1
    assert expired_list[0].telegram_id == 77777
