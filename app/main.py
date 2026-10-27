"""Main Application Entry Point."""

import asyncio
import logging
import sys
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand

from app.config import get_settings
from app.database.session import init_db, close_db
from app.handlers.admin import admin_router
from app.handlers.public_group import public_group_router
from app.handlers.common import common_router
from app.services.scheduler import ExpirationScheduler

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("taxi_bot")


async def setup_bot_commands(bot: Bot) -> None:
    """Register menu commands with Telegram BotFather interface."""
    commands = [
        BotCommand(command="start", description="Boshqaruv menyusi / Holat"),
        BotCommand(command="activate", description="Obunani faollashtirish"),
        BotCommand(command="renew", description="Obunani uzaytirish"),
        BotCommand(command="subscription", description="Haydovchini tekshirish"),
        BotCommand(command="active", description="Faol obunachilar"),
        BotCommand(command="expired", description="Muddati tugaganlar"),
        BotCommand(command="groups", description="Kuzatilayotgan guruhlar"),
        BotCommand(command="addgroup", description="Guruh qo'shish"),
        BotCommand(command="delgroup", description="Guruhni o'chirish"),
        BotCommand(command="stats", description="Umumiy statistika"),
        BotCommand(command="id", description="ID raqamni aniqlash"),
        BotCommand(command="help", description="Qo'llanma va yordam"),
    ]
    try:
        await bot.set_my_commands(commands)
        logger.info("Bot commands successfully registered with Telegram.")
    except Exception as exc:
        logger.warning("Could not set bot commands: %s", exc)


async def main() -> None:
    """Initialize bot, database, scheduler, routers, and start long polling."""
    logger.info("Starting Telegram Taxi Forwarder Bot...")

    # 1. Initialize Database Schema
    await init_db()

    # 2. Initialize Bot with HTML ParseMode by default
    bot = Bot(
        token=settings.bot_token.get_secret_value(),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    # 3. Initialize Dispatcher with in-memory storage
    dp = Dispatcher(storage=MemoryStorage())

    # 4. Register Handler Routers in precedence order
    dp.include_router(admin_router)
    dp.include_router(public_group_router)
    dp.include_router(common_router)

    # Verify credentials and retrieve bot profile
    try:
        me = await bot.get_me()
        logger.info("Bot authenticated: @%s (ID: %s, Name: %s)", me.username, me.id, me.first_name)
    except Exception as exc:
        logger.critical("Failed to authenticate with Telegram Bot API: %s", exc)
        await bot.session.close()
        await close_db()
        return

    # Set up menu commands
    await setup_bot_commands(bot)

    # 5. Initialize and Start Expiration Background Scheduler
    scheduler = ExpirationScheduler(bot=bot, settings=settings)
    # Immediate check for drivers who expired while the bot was stopped
    try:
        evicted = await scheduler.check_expired_subscriptions()
        if evicted > 0:
            logger.info("Startup check: Evicted %d expired driver(s).", evicted)
    except Exception as exc:
        logger.error("Error during startup expiration check: %s", exc)
    scheduler.start()

    # Notify administrator on startup
    for admin_id in settings.admin_user_ids:
        try:
            await bot.send_message(
                chat_id=admin_id,
                text="🟢 <b>Taksi Bot muvaffaqiyatli ishga tushdi!</b>\n"
                     "<i>Tizim, baza va avtomatik tekshiruv ishga tushirildi. Boshqaruv uchun /start yuboring.</i>",
                parse_mode="HTML",
            )
            logger.info("Startup notification sent to admin ID: %s", admin_id)
        except Exception as exc:
            logger.info("Could not send startup PM to admin %s: %s", admin_id, exc)

    # 6. Initialize Forwarding Service and hydrate 24-hour user cooldowns from DB
    from app.handlers.public_group import get_forwarding_service
    from app.services.group_manager import get_group_manager

    forwarder = get_forwarding_service(bot)
    try:
        loaded_cooldowns = await forwarder.load_recent_cooldowns_from_db()
        if loaded_cooldowns > 0:
            logger.info("Restored %d active 24h user cooldown(s) from database.", loaded_cooldowns)
    except Exception as exc:
        logger.warning("Could not hydrate user cooldowns from DB: %s", exc)

    # 7. Initialize Dynamic Group Manager
    group_manager = get_group_manager(settings)
    try:
        await group_manager.initialize()
    except Exception as exc:
        logger.warning("Could not initialize dynamic group manager: %s", exc)

    # 8. Initialize Userbot (MTProto) if enabled
    userbot_client = None
    if settings.userbot_enabled:
        if not settings.tg_api_id or not settings.tg_api_hash:
            logger.warning("USERBOT_ENABLED is true, but TG_API_ID or TG_API_HASH is missing in .env.")
        else:
            try:
                from app.userbot.client import create_userbot_client
                from app.userbot.listener import setup_userbot_listener, auto_join_and_resolve_targets
                from app.services.classification import get_classifier

                userbot_client = create_userbot_client(settings)
                if userbot_client:
                    await userbot_client.connect()
                    if await userbot_client.is_user_authorized():
                        ub_me = await userbot_client.get_me()
                        logger.info(
                            "Userbot authenticated: %s (@%s, ID: %s)",
                            ub_me.first_name,
                            ub_me.username,
                            ub_me.id,
                        )
                        # Attach userbot to group manager for real-time joins
                        group_manager.set_clients(bot=bot, userbot=userbot_client)

                        # Auto-join configured target groups by link or username
                        monitored_ids = await auto_join_and_resolve_targets(userbot_client, settings)

                        classifier = get_classifier()
                        setup_userbot_listener(
                            client=userbot_client,
                            bot=bot,
                            settings=settings,
                            classifier=classifier,
                            forwarder=forwarder,
                            monitored_ids=monitored_ids,
                        )
                        logger.info("Userbot listener attached and active.")
                    else:
                        logger.warning(
                            "⚠️ Userbot is enabled but NOT authorized yet. "
                            "Please run 'python -m app.userbot.auth' in your terminal to authenticate."
                        )
            except Exception as exc:
                logger.error("Failed to initialize Telethon Userbot: %s", exc)

    # Ensure bot client is attached to group_manager
    group_manager.set_clients(bot=bot, userbot=userbot_client)

    logger.info("Bot is now listening for incoming messages (Polling mode)...")

    try:
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot shutdown requested.")
    finally:
        logger.info("Closing bot session, userbot, scheduler, and database...")
        scheduler.shutdown()
        if userbot_client and userbot_client.is_connected():
            try:
                await userbot_client.disconnect()
                logger.info("Userbot client disconnected successfully.")
            except Exception as exc:
                logger.warning("Error disconnecting userbot: %s", exc)
        await bot.session.close()
        await close_db()
        logger.info("Bot shutdown completed cleanly.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
