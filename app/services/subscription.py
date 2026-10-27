"""Driver Subscription Business Logic Service."""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError

from app.config import Settings
from app.database.models.driver import Driver, SubscriptionStatus
from app.database.session import get_db_session
from app.repositories.driver_repo import DriverRepository

logger = logging.getLogger(__name__)


class SubscriptionService:
    """Manages driver subscription lifecycle, invite links, and group access."""

    def __init__(self, bot: Bot, settings: Settings):
        self.bot = bot
        self.settings = settings
        self.local_tz = ZoneInfo(settings.timezone)

    def to_local_string(self, dt: Optional[datetime]) -> str:
        """Format UTC datetime into Tashkent local time string."""
        if not dt:
            return "Ko'rsatilmagan"
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local_dt = dt.astimezone(self.local_tz)
        return local_dt.strftime("%Y-%m-%d %H:%M")

    async def create_driver_invite_link(self, driver_name: str) -> Optional[str]:
        """
        Create a secure, single-use, 24-hour expiring invite link for the VIP group.
        Prevents unauthorized link forwarding.
        """
        if self.settings.private_group_id == 0:
            logger.warning("PRIVATE_GROUP_ID is 0. Cannot create invite link.")
            return None

        try:
            invite = await self.bot.create_chat_invite_link(
                chat_id=self.settings.private_group_id,
                name=f"VIP: {driver_name[:25]}",
                expire_date=int(time.time() + 86400),  # Valid for 24 hours
                member_limit=1,  # Strictly one person can join
            )
            return invite.invite_link
        except TelegramAPIError as e:
            logger.error("Failed to generate single-use invite link: %s", e)
            return None

    async def register_driver(
        self,
        telegram_id: int,
        full_name: str,
        username: Optional[str] = None,
        phone_number: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Driver:
        """Register driver record in database with PENDING status."""
        async with get_db_session() as session:
            repo = DriverRepository(session)
            driver = await repo.create_or_update(
                telegram_id=telegram_id,
                full_name=full_name,
                username=username,
                phone_number=phone_number,
                notes=notes,
            )
            logger.info("Driver registered: TG ID %s (%s)", telegram_id, full_name)
            return driver

    async def activate_driver(
        self,
        telegram_id: int,
        duration_days: Optional[int] = None,
        admin_id: Optional[int] = None,
    ) -> Tuple[Driver, Optional[str]]:
        """
        Activate subscription for an approved driver:
        1. Lift any previous ban in the VIP group.
        2. Generate a single-use invite link.
        3. Save active dates in database.
        """
        days = duration_days or self.settings.default_subscription_days

        # 1. Unban in VIP group if previously removed
        if self.settings.private_group_id != 0:
            try:
                await self.bot.unban_chat_member(
                    chat_id=self.settings.private_group_id,
                    user_id=telegram_id,
                    only_if_banned=True,
                )
            except TelegramAPIError:
                pass

        async with get_db_session() as session:
            repo = DriverRepository(session)
            driver = await repo.get_by_telegram_id(telegram_id)
            driver_name = driver.full_name if driver else f"ID_{telegram_id}"

            # 2. Create single-use invite link
            invite_link = await self.create_driver_invite_link(driver_name)

            # 3. Activate in database
            driver = await repo.activate_subscription(
                telegram_id=telegram_id,
                duration_days=days,
                approved_by_admin_id=admin_id,
                invite_link=invite_link,
            )
            logger.info("Subscription activated for %s for %d days", telegram_id, days)
            return driver, invite_link

    async def renew_driver(
        self,
        telegram_id: int,
        additional_days: Optional[int] = None,
        admin_id: Optional[int] = None,
    ) -> Tuple[Driver, Optional[str]]:
        """Extend subscription after confirmed manual renewal payment."""
        days = additional_days or self.settings.default_subscription_days

        async with get_db_session() as session:
            repo = DriverRepository(session)
            driver = await repo.get_by_telegram_id(telegram_id)
            if not driver:
                raise ValueError(f"Driver with Telegram ID {telegram_id} does not exist.")

            # If driver was expired or banned, unban and provide fresh link
            invite_link = None
            if driver.status != SubscriptionStatus.ACTIVE.value or (driver.end_date and driver.end_date <= datetime.now(timezone.utc)):
                if self.settings.private_group_id != 0:
                    try:
                        await self.bot.unban_chat_member(
                            chat_id=self.settings.private_group_id,
                            user_id=telegram_id,
                            only_if_banned=True,
                        )
                    except TelegramAPIError:
                        pass
                invite_link = await self.create_driver_invite_link(driver.full_name)

            driver = await repo.renew_subscription(
                telegram_id=telegram_id,
                additional_days=days,
                approved_by_admin_id=admin_id,
            )
            if invite_link:
                driver.invite_link = invite_link

            logger.info("Subscription renewed for %s (+%d days)", telegram_id, days)
            return driver, invite_link

    async def revoke_driver_access(self, telegram_id: int, reason: str = "Muddati tugagan") -> bool:
        """
        Evict driver from VIP group and mark as EXPIRED.
        Uses ban_chat_member to prevent re-entering with old links.
        """
        if self.settings.private_group_id != 0:
            try:
                await self.bot.ban_chat_member(
                    chat_id=self.settings.private_group_id,
                    user_id=telegram_id,
                )
                logger.info("Banned expired driver %s from VIP group", telegram_id)
            except TelegramAPIError as e:
                logger.warning("Could not ban driver %s from VIP group: %s", telegram_id, e)

        async with get_db_session() as session:
            repo = DriverRepository(session)
            await repo.set_status(telegram_id, SubscriptionStatus.EXPIRED)
            return True

    async def get_driver_info(self, telegram_id: int) -> Optional[Dict]:
        """Retrieve driver profile and human-formatted subscription dates."""
        async with get_db_session() as session:
            repo = DriverRepository(session)
            driver = await repo.get_by_telegram_id(telegram_id)
            if not driver:
                return None

            now = datetime.now(timezone.utc)
            return {
                "id": driver.id,
                "telegram_id": driver.telegram_id,
                "full_name": driver.full_name,
                "username": driver.username,
                "phone_number": driver.phone_number,
                "status": driver.status,
                "start_date": self.to_local_string(driver.start_date),
                "end_date": self.to_local_string(driver.end_date),
                "is_active": driver.is_currently_active(now),
                "days_left": driver.days_remaining(now),
                "notes": driver.notes,
            }
