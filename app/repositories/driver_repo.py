"""Driver Database Repository."""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models.driver import Driver, SubscriptionStatus, ensure_utc


class DriverRepository:
    """Encapsulates database operations for the Driver model."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_telegram_id(self, telegram_id: int) -> Optional[Driver]:
        """Find a driver by their Telegram numeric User ID."""
        stmt = select(Driver).where(Driver.telegram_id == telegram_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_id(self, driver_id: int) -> Optional[Driver]:
        """Find a driver by their internal database primary key."""
        stmt = select(Driver).where(Driver.id == driver_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create_or_update(
        self,
        telegram_id: int,
        full_name: str,
        username: Optional[str] = None,
        phone_number: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Driver:
        """Create new driver or update existing driver's identity info."""
        driver = await self.get_by_telegram_id(telegram_id)
        if not driver:
            driver = Driver(
                telegram_id=telegram_id,
                full_name=full_name,
                username=username,
                phone_number=phone_number,
                notes=notes,
                status=SubscriptionStatus.PENDING.value,
            )
            self.session.add(driver)
        else:
            driver.full_name = full_name
            if username:
                driver.username = username
            if phone_number:
                driver.phone_number = phone_number
            if notes:
                driver.notes = notes

        await self.session.flush()
        return driver

    async def activate_subscription(
        self,
        telegram_id: int,
        duration_days: int = 30,
        approved_by_admin_id: Optional[int] = None,
        invite_link: Optional[str] = None,
    ) -> Driver:
        """Activate or reactivate driver subscription for the given number of days."""
        driver = await self.get_by_telegram_id(telegram_id)
        if not driver:
            raise ValueError(f"Driver with Telegram ID {telegram_id} not found in database.")

        now = datetime.now(timezone.utc)
        driver.status = SubscriptionStatus.ACTIVE.value
        driver.start_date = now
        driver.end_date = now + timedelta(days=duration_days)
        driver.approved_by_admin_id = approved_by_admin_id
        if invite_link:
            driver.invite_link = invite_link

        await self.session.flush()
        return driver

    async def renew_subscription(
        self,
        telegram_id: int,
        additional_days: int = 30,
        approved_by_admin_id: Optional[int] = None,
    ) -> Driver:
        """Extend an existing subscription from its current end_date (or from now if expired)."""
        driver = await self.get_by_telegram_id(telegram_id)
        if not driver:
            raise ValueError(f"Driver with Telegram ID {telegram_id} not found in database.")

        now = datetime.now(timezone.utc)
        driver_end = ensure_utc(driver.end_date)
        # If currently active, extend from end_date; if expired, start from now
        if driver.status == SubscriptionStatus.ACTIVE.value and driver_end and driver_end > now:
            base_date = driver_end
        else:
            base_date = now
            driver.start_date = now

        driver.status = SubscriptionStatus.ACTIVE.value
        driver.end_date = base_date + timedelta(days=additional_days)
        if approved_by_admin_id:
            driver.approved_by_admin_id = approved_by_admin_id

        await self.session.flush()
        return driver

    async def set_status(self, telegram_id: int, status: SubscriptionStatus) -> Optional[Driver]:
        """Update driver status (e.g. EXPIRED, BANNED, PENDING)."""
        driver = await self.get_by_telegram_id(telegram_id)
        if driver:
            driver.status = status.value
            await self.session.flush()
        return driver

    async def list_active(self) -> Sequence[Driver]:
        """Retrieve all currently active drivers."""
        stmt = (
            select(Driver)
            .where(Driver.status == SubscriptionStatus.ACTIVE.value)
            .order_by(Driver.end_date.asc())
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def list_expired(self) -> Sequence[Driver]:
        """Retrieve all expired drivers."""
        stmt = (
            select(Driver)
            .where(Driver.status == SubscriptionStatus.EXPIRED.value)
            .order_by(Driver.end_date.desc())
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def list_pending(self) -> Sequence[Driver]:
        """Retrieve drivers waiting for manual activation."""
        stmt = (
            select(Driver)
            .where(Driver.status == SubscriptionStatus.PENDING.value)
            .order_by(Driver.created_at.desc())
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_expired_active_drivers(self, reference_time: Optional[datetime] = None) -> Sequence[Driver]:
        """
        Find drivers marked as 'active' whose end_date is in the past.
        These drivers need their Telegram access revoked.
        """
        ref = ensure_utc(reference_time) or datetime.now(timezone.utc)
        stmt = (
            select(Driver)
            .where(
                Driver.status == SubscriptionStatus.ACTIVE.value,
                Driver.end_date <= ref,
            )
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_statistics(self) -> Dict[str, int]:
        """Aggregate driver counts by status."""
        stmt = select(Driver.status, func.count(Driver.id)).group_by(Driver.status)
        result = await self.session.execute(stmt)
        stats = {status.value: 0 for status in SubscriptionStatus}
        for status_val, count in result.all():
            stats[status_val] = count
        return stats
