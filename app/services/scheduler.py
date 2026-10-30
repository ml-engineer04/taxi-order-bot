"""Background Scheduler for Automated Subscription Expiration Checks."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError

from app.config import Settings
from app.database.models.driver import SubscriptionStatus
from app.database.session import get_db_session
from app.repositories.driver_repo import DriverRepository
from app.repositories.collected_repo import CollectedMessageRepository
from app.services.subscription import SubscriptionService

logger = logging.getLogger(__name__)


class ExpirationScheduler:
    """Manages periodic checks for expired subscriptions and driver eviction."""

    def __init__(self, bot: Bot, settings: Settings):
        self.bot = bot
        self.settings = settings
        self.scheduler = AsyncIOScheduler(timezone=settings.timezone)
        self.sub_service = SubscriptionService(bot=bot, settings=settings)

    async def check_expired_subscriptions(self) -> int:
        """
        Check database for active drivers whose end_date is in the past.
        Evicts them from the VIP group, updates status, and notifies admin & driver.
        Returns the number of evicted drivers.
        """
        now = datetime.now(timezone.utc)
        logger.info("Running scheduled check for expired subscriptions (UTC %s)...", now)

        async with get_db_session() as session:
            repo = DriverRepository(session)
            expired_drivers = await repo.get_expired_active_drivers(reference_time=now)

            if not expired_drivers:
                logger.info("No expired subscriptions found.")
                return 0

            logger.warning("Found %d expired driver(s) to evict.", len(expired_drivers))

            for driver in expired_drivers:
                tg_id = driver.telegram_id
                driver_name = driver.full_name
                end_str = self.sub_service.to_local_string(driver.end_date)

                # 1. Evict / Ban driver from private VIP group
                if self.settings.private_group_id != 0:
                    try:
                        await self.bot.ban_chat_member(
                            chat_id=self.settings.private_group_id,
                            user_id=tg_id,
                        )
                        logger.info("Evicted expired driver %s from VIP group %s", tg_id, self.settings.private_group_id)
                    except TelegramAPIError as e:
                        logger.warning("Could not ban driver %s from VIP group: %s", tg_id, e)

                # 2. Update status in database
                await repo.set_status(tg_id, SubscriptionStatus.EXPIRED)

                # 3. Notify administrator
                admin_alert = (
                    "⚠️ <b>Obuna muddati tugadi va haydovchi guruhdan chiqarildi:</b>\n\n"
                    f"👤 <b>Haydovchi:</b> {driver_name}\n"
                    f"🆔 <b>Telegram ID:</b> <code>{tg_id}</code>\n"
                    f"📅 <b>Tugagan sana:</b> {end_str}\n\n"
                    f"<i>Qayta tiklash uchun: <code>/renew {tg_id} 30</code></i>"
                )
                for admin_id in self.settings.admin_user_ids:
                    try:
                        await self.bot.send_message(chat_id=admin_id, text=admin_alert, parse_mode="HTML")
                    except Exception as exc:
                        logger.warning("Could not send expiration alert to admin %s: %s", admin_id, exc)

                # 4. Notify driver in private message
                admin_user = (self.settings.admin_username or "rahmonjon_qosimov").lstrip("@").strip()
                driver_alert = (
                    f"⚠️ <b>Hurmatli {driver_name}!</b>\n\n"
                    "Sizning VIP haydovchilar guruhidagi oylik obuna muddatingiz tugadi "
                    "va guruhdan avtomatik chiqarildingiz.\n\n"
                    "🔄 <b>Obunani qayta faollashtirish va yangi buyurtmalarni qabul qilish uchun admin bilan bog'laning:</b>\n"
                    f"👉 @{admin_user}\n\n"
                    f"🆔 Sizning ID raqamingiz: <code>{tg_id}</code>"
                )
                try:
                    await self.bot.send_message(chat_id=tg_id, text=driver_alert, parse_mode="HTML")
                except Exception:
                    logger.info("Could not send PM to expired driver %s (bot blocked or not started).", tg_id)

            return len(expired_drivers)

    async def notify_pending_reviews(self) -> None:
        """Alert administrators if there are pending ambiguous messages to review."""
        try:
            async with get_db_session() as session:
                count = await CollectedMessageRepository.get_pending_count(session)
                if count >= 5:
                    alert = (
                        "🔔 <b>Tasdiqlash uchun yangi xabarlar to'plandi!</b>\n\n"
                        f"Guruhdan model to'liq aniqlay olmagan <b>{count} ta</b> xabar yig'ildi.\n"
                        "Modelni yanada aniqroq o'qitish uchun <code>/review</code> buyrug'ini bering!"
                    )
                    for admin_id in self.settings.admin_user_ids:
                        try:
                            await self.bot.send_message(chat_id=admin_id, text=alert, parse_mode="HTML")
                        except Exception:
                            pass
        except Exception as e:
            logger.debug("Failed in notify_pending_reviews: %s", e)

    def start(self) -> None:
        """Register recurring jobs and start scheduler."""
        # Run check every 15 minutes
        self.scheduler.add_job(
            self.check_expired_subscriptions,
            trigger="interval",
            minutes=15,
            id="check_expired_subscriptions",
            replace_existing=True,
        )
        # Check pending active learning review queue every 4 hours
        self.scheduler.add_job(
            self.notify_pending_reviews,
            trigger="interval",
            hours=4,
            id="notify_pending_reviews",
            replace_existing=True,
        )
        self.scheduler.start()
        logger.info("Expiration scheduler started (interval: 15 minutes, review check: 4 hours).")

    def shutdown(self) -> None:
        """Stop scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("Expiration scheduler stopped.")
