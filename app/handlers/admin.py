"""Admin Commands and Driver Subscription Management Router."""

import logging
from typing import Optional
from aiogram import Router, types, F
from aiogram.filters import Command, CommandStart
from aiogram.enums import ChatType
from aiogram.utils.keyboard import InlineKeyboardBuilder

from app.config import get_settings
from app.database.models.driver import SubscriptionStatus
from app.database.session import get_db_session
from app.repositories.driver_repo import DriverRepository
from app.repositories.collected_repo import CollectedMessageRepository
from app.services.active_learning import (
    build_review_card,
    append_example_to_dataset,
    retrain_model_async,
)
from app.services.subscription import SubscriptionService

logger = logging.getLogger(__name__)
admin_router = Router(name="admin_router")
settings = get_settings()


def is_admin(user_id: Optional[int]) -> bool:
    """Validate administrator authorization."""
    if not user_id:
        return False
    return settings.is_admin(user_id)


@admin_router.message(CommandStart())
async def handle_start(message: types.Message) -> None:
    """Handle /start command in private chat or groups."""
    user = message.from_user
    if not user:
        return

    sub_service = SubscriptionService(bot=message.bot, settings=settings)

    if is_admin(user.id):
        text = (
            f"👑 <b>Assalomu alaykum, Hurmatli Administrator ({user.full_name})!</b>\n\n"
            "🚖 <b>Taksi Buyurtmalarini Saralash Boti Boshqaruv Paneli</b>\n\n"
            "<b>Obuna boshqaruvi:</b>\n"
            "• <code>/activate &lt;tg_id&gt; [kun]</code> - Obunani faollashtirish va havola yaratish\n"
            "• <code>/renew &lt;tg_id&gt; [kun]</code> - Obuna muddatini uzaytirish\n"
            "• <code>/subscription &lt;tg_id&gt;</code> - Haydovchi ma'lumotlarini tekshirish\n"
            "• <code>/revoke &lt;tg_id&gt;</code> - Haydovchini guruhdan chiqarish\n\n"
            "<b>Ro'yxatlar va statistika:</b>\n"
            "• /active - Faol haydovchilar ro'yxati\n"
            "• /expired - Muddati o'tgan haydovchilar\n"
            "• /pending - To'lov kutilayotganlar\n"
            "• /stats - Umumiy statistika\n\n"
            "<b>Guruhlar boshqaruvi:</b>\n"
            "• /groups - Kuzatilayotgan guruhlar ro'yxati\n"
            "• <code>/addgroup &lt;ID/link&gt;</code> - Yangi guruh qo'shish\n"
            "• <code>/delgroup &lt;ID&gt;</code> - Guruhni o'chirish\n\n"
            "<b>Mashinali o'rganish (AI/ML):</b>\n"
            "• /review - Guruhdan to'plangan noaniq xabarlarni tasdiqlash\n"
            "• /retrain - Modelni yangi xabarlar bilan qayta o'qitish\n\n"
            "<b>Tizim buyruqlari:</b>\n"
            "• /id - Chat va foydalanuvchi ID sini aniqlash\n"
            "• /status - Guruhlar va tizim sozlamalari\n"
            "• /help - To'liq yo'riqnoma"
        )
        await message.answer(text, parse_mode="HTML")
    else:
        # Check if driver has an active subscription
        driver_info = await sub_service.get_driver_info(user.id)
        builder = InlineKeyboardBuilder()

        if driver_info and driver_info["is_active"]:
            text = (
                f"👋 <b>Assalomu alaykum, {user.full_name}!</b>\n\n"
                "✅ <b>Sizning VIP obunangiz faol holatda!</b>\n\n"
                f"📅 <b>Tugash sanasi:</b> <code>{driver_info['end_date']}</code>\n"
                f"⏳ <b>Qolgan vaqt:</b> {driver_info['days_left']} kun\n\n"
                "<i>VIP guruhda barcha yangi yo'lovchi buyurtmalarini kuzatib boring!</i>"
            )
        else:
            text = (
                f"👋 <b>Assalomu alaykum, {user.full_name}!</b>\n\n"
                "🚖 <b>Viloyatlararo Taksi Haydovchilari Xizmati Botiga xush kelibsiz!</b>\n\n"
                "Ushbu bot umumiy guruhlardagi yo'lovchi buyurtmalarini saralab, "
                "faqat tasdiqlangan haydovchilar guruhiga yetkazadi.\n\n"
                "🔒 <b>Yopiq VIP guruhga qo'shilish va obunani faollashtirish uchun "
                "admin bilan bog'laning:</b>\n"
                f"👉 @{settings.admin_username}\n\n"
                f"🆔 <b>Sizning Telegram ID raqamingiz:</b> <code>{user.id}</code>\n"
                "<i>(To'lov qilgach, ushbu ID raqamingizni adminga yuboring).</i>"
            )
            builder.button(text="💬 Admin bilan bog'lanish", url=f"https://t.me/{settings.admin_username}")

        await message.answer(text, reply_markup=builder.as_markup(), parse_mode="HTML")


@admin_router.message(Command("register"))
async def handle_register(message: types.Message) -> None:
    """Register a driver in database: /register <telegram_id> <Full Name>"""
    if not is_admin(message.from_user.id):
        await message.answer("⛔ <i>Bu buyruq faqat administratorlar uchun mo'ljallangan.</i>", parse_mode="HTML")
        return

    args = message.text.split(maxsplit=2)
    if len(args) < 3 or not args[1].isdigit():
        await message.answer(
            "⚠️ <b>Noto'g'ri format!</b>\n\n"
            "Foydalanish: <code>/register &lt;telegram_id&gt; &lt;Ism Familiya&gt;</code>\n"
            "Misol: <code>/register 987654321 Sardor Aliyev</code>",
            parse_mode="HTML",
        )
        return

    tg_id = int(args[1])
    full_name = args[2].strip()

    sub_service = SubscriptionService(bot=message.bot, settings=settings)
    driver = await sub_service.register_driver(telegram_id=tg_id, full_name=full_name)

    text = (
        "✅ <b>Haydovchi muvaffaqiyatli ro'yxatga olindi!</b>\n\n"
        f"👤 <b>Foydalanuvchi:</b> {driver.full_name}\n"
        f"🆔 <b>Telegram ID:</b> <code>{driver.telegram_id}</code>\n"
        f"📊 <b>Holati:</b> {driver.status.upper()}\n\n"
        "<i>Obunani yoqish uchun:</i>\n"
        f"<code>/activate {driver.telegram_id} 30</code>"
    )
    await message.answer(text, parse_mode="HTML")


@admin_router.message(Command("activate"))
async def handle_activate(message: types.Message) -> None:
    """Activate subscription: /activate <telegram_id> [duration_days]"""
    if not is_admin(message.from_user.id):
        await message.answer("⛔ <i>Bu buyruq faqat administratorlar uchun mo'ljallangan.</i>", parse_mode="HTML")
        return

    args = message.text.split()
    if len(args) < 2 or not args[1].isdigit():
        await message.answer(
            "⚠️ <b>Noto'g'ri format!</b>\n\n"
            "Foydalanish: <code>/activate &lt;telegram_id&gt; [kunlar]</code>\n"
            "Misol: <code>/activate 987654321 30</code>",
            parse_mode="HTML",
        )
        return

    tg_id = int(args[1])
    days = int(args[2]) if len(args) > 2 and args[2].isdigit() else settings.default_subscription_days

    sub_service = SubscriptionService(bot=message.bot, settings=settings)

    # Ensure driver exists in database
    async with get_db_session() as session:
        repo = DriverRepository(session)
        driver = await repo.get_by_telegram_id(tg_id)
        if not driver:
            # Auto-register driver record
            await repo.create_or_update(telegram_id=tg_id, full_name=f"Haydovchi_{tg_id}")

    try:
        driver, invite_link = await sub_service.activate_driver(
            telegram_id=tg_id,
            duration_days=days,
            admin_id=message.from_user.id,
        )

        start_str = sub_service.to_local_string(driver.start_date)
        end_str = sub_service.to_local_string(driver.end_date)

        text = (
            "🎉 <b>Obuna Muvaffaqiyatli Faollashtirildi!</b>\n\n"
            f"👤 <b>Haydovchi:</b> {driver.full_name}\n"
            f"🆔 <b>Telegram ID:</b> <code>{driver.telegram_id}</code>\n"
            f"⏳ <b>Muddat:</b> {days} kun\n"
            f"📅 <b>Boshlanish:</b> {start_str}\n"
            f"🏁 <b>Tugash:</b> {end_str}\n\n"
        )

        if invite_link:
            text += (
                f"🔗 <b>Bir martalik taklif havolasi (24 soat amal qiladi):</b>\n"
                f"<code>{invite_link}</code>\n\n"
                "<i>Ushbu havolani haydovchiga yuboring. Undan faqat 1 kishi foydalana oladi!</i>"
            )
        else:
            text += "⚠️ <i>Taklif havolasi yaratilmadi (PRIVATE_GROUP_ID tekshiring).</i>"

        await message.answer(text, parse_mode="HTML")

        # Try to notify the driver in PM
        if invite_link:
            try:
                driver_text = (
                    "🎉 <b>Tabriklaymiz, to'lovingiz tasdiqlandi!</b>\n\n"
                    f"VIP haydovchilar guruhiga kirish havolangiz:\n{invite_link}\n\n"
                    f"📅 Obuna tugash sanasi: <b>{end_str}</b>\n"
                    "<i>Eslatma: Havola bir martalik bo'lib, uni boshqalarga ulashib bo'lmaydi.</i>"
                )
                await message.bot.send_message(chat_id=tg_id, text=driver_text, parse_mode="HTML")
            except Exception:
                logger.info("Could not PM driver %s directly (driver hasn't started bot).", tg_id)

    except Exception as exc:
        logger.exception("Error activating driver %s: %s", tg_id, exc)
        await message.answer(f"❌ Xatolik yuz berdi: {exc}")


@admin_router.message(Command("renew"))
async def handle_renew(message: types.Message) -> None:
    """Extend subscription: /renew <telegram_id> [additional_days]"""
    if not is_admin(message.from_user.id):
        await message.answer("⛔ <i>Bu buyruq faqat administratorlar uchun mo'ljallangan.</i>", parse_mode="HTML")
        return

    args = message.text.split()
    if len(args) < 2 or not args[1].isdigit():
        await message.answer(
            "⚠️ <b>Noto'g'ri format!</b>\n\n"
            "Foydalanish: <code>/renew &lt;telegram_id&gt; [qo'shimcha_kunlar]</code>\n"
            "Misol: <code>/renew 987654321 30</code>",
            parse_mode="HTML",
        )
        return

    tg_id = int(args[1])
    days = int(args[2]) if len(args) > 2 and args[2].isdigit() else settings.default_subscription_days

    sub_service = SubscriptionService(bot=message.bot, settings=settings)

    try:
        driver, invite_link = await sub_service.renew_driver(
            telegram_id=tg_id,
            additional_days=days,
            admin_id=message.from_user.id,
        )

        end_str = sub_service.to_local_string(driver.end_date)
        text = (
            "🔄 <b>Obuna Muvaffaqiyatli Uzaytirildi!</b>\n\n"
            f"👤 <b>Haydovchi:</b> {driver.full_name}\n"
            f"🆔 <b>Telegram ID:</b> <code>{driver.telegram_id}</code>\n"
            f"➕ <b>Qo'shilgan muddat:</b> {days} kun\n"
            f"🏁 <b>Yangi tugash sanasi:</b> {end_str}\n"
            f"⏳ <b>Qolgan kunlar:</b> {driver.days_remaining()} kun\n"
        )
        if invite_link:
            text += f"\n🔗 <b>Qayta kirish havolasi:</b> <code>{invite_link}</code>"

        await message.answer(text, parse_mode="HTML")
    except ValueError as val_err:
        await message.answer(f"⚠️ {val_err}")
    except Exception as exc:
        logger.exception("Error renewing driver %s: %s", tg_id, exc)
        await message.answer(f"❌ Xatolik yuz berdi: {exc}")


@admin_router.message(Command("subscription"))
async def handle_check_subscription(message: types.Message) -> None:
    """Check driver details: /subscription <telegram_id>"""
    if not is_admin(message.from_user.id):
        await message.answer("⛔ <i>Bu buyruq faqat administratorlar uchun mo'ljallangan.</i>", parse_mode="HTML")
        return

    args = message.text.split()
    if len(args) < 2 or not args[1].isdigit():
        await message.answer("Foydalanish: <code>/subscription &lt;telegram_id&gt;</code>", parse_mode="HTML")
        return

    tg_id = int(args[1])
    sub_service = SubscriptionService(bot=message.bot, settings=settings)
    info = await sub_service.get_driver_info(tg_id)

    if not info:
        await message.answer(f"❌ <code>{tg_id}</code> ID raqamli haydovchi bazadan topilmadi.", parse_mode="HTML")
        return

    status_icon = "🟢" if info["is_active"] else "🔴"

    text = (
        f"{status_icon} <b>Haydovchi Ma'lumotlari:</b>\n\n"
        f"👤 <b>Foydalanuvchi:</b> {info['full_name']}\n"
        f"🆔 <b>Telegram ID:</b> <code>{info['telegram_id']}</code>\n"
        f"📱 <b>Username:</b> @{info['username'] or 'Yo‘q'}\n"
        f"📊 <b>Holati:</b> {info['status'].upper()}\n"
        f"📅 <b>Boshlangan:</b> {info['start_date']}\n"
        f"🏁 <b>Tugaydi:</b> {info['end_date']}\n"
        f"⏳ <b>Qolgan muddat:</b> {info['days_left']} kun\n"
    )
    if info.get("notes"):
        text += f"📝 <b>Izoh:</b> {info['notes']}\n"

    await message.answer(text, parse_mode="HTML")


@admin_router.message(Command("revoke"))
async def handle_revoke(message: types.Message) -> None:
    """Revoke driver access and ban from VIP group: /revoke <telegram_id>"""
    if not is_admin(message.from_user.id):
        await message.answer("⛔ <i>Bu buyruq faqat administratorlar uchun mo'ljallangan.</i>", parse_mode="HTML")
        return

    args = message.text.split()
    if len(args) < 2 or not args[1].isdigit():
        await message.answer("Foydalanish: <code>/revoke &lt;telegram_id&gt;</code>", parse_mode="HTML")
        return

    tg_id = int(args[1])
    sub_service = SubscriptionService(bot=message.bot, settings=settings)
    await sub_service.revoke_driver_access(tg_id, reason="Admin tomonidan bekor qilindi")

    await message.answer(
        f"🚫 <b>Haydovchi ({tg_id}) obunasi bekor qilindi va VIP guruhdan chiqarildi.</b>",
        parse_mode="HTML",
    )


@admin_router.message(Command("active"))
async def handle_list_active(message: types.Message) -> None:
    """List all currently active drivers."""
    if not is_admin(message.from_user.id):
        return

    sub_service = SubscriptionService(bot=message.bot, settings=settings)
    async with get_db_session() as session:
        repo = DriverRepository(session)
        drivers = await repo.list_active()

    if not drivers:
        await message.answer("ℹ️ <i>Hozirda faol obunachilar mavjud emas.</i>", parse_mode="HTML")
        return

    lines = [f"🟢 <b>Faol Obunachilar Ro'yxati ({len(drivers)} ta):</b>\n"]
    for i, d in enumerate(drivers, start=1):
        end_str = sub_service.to_local_string(d.end_date)
        username_str = f"(@{d.username})" if d.username else ""
        lines.append(f"{i}. <b>{d.full_name}</b> {username_str}\n   🆔 <code>{d.telegram_id}</code> | Tugaydi: {end_str} ({d.days_remaining()} kun)")

    await message.answer("\n".join(lines), parse_mode="HTML")


@admin_router.message(Command("expired"))
async def handle_list_expired(message: types.Message) -> None:
    """List all expired drivers."""
    if not is_admin(message.from_user.id):
        return

    sub_service = SubscriptionService(bot=message.bot, settings=settings)
    async with get_db_session() as session:
        repo = DriverRepository(session)
        drivers = await repo.list_expired()

    if not drivers:
        await message.answer("ℹ️ <i>Muddati o'tgan haydovchilar mavjud emas.</i>", parse_mode="HTML")
        return

    lines = [f"🔴 <b>Muddati O'tgan Haydovchilar ({len(drivers)} ta):</b>\n"]
    for i, d in enumerate(drivers, start=1):
        end_str = sub_service.to_local_string(d.end_date)
        lines.append(f"{i}. <b>{d.full_name}</b> (<code>{d.telegram_id}</code>) | Tugagan: {end_str}")

    await message.answer("\n".join(lines), parse_mode="HTML")


@admin_router.message(Command("pending"))
async def handle_list_pending(message: types.Message) -> None:
    """List all registered drivers awaiting payment activation."""
    if not is_admin(message.from_user.id):
        return

    async with get_db_session() as session:
        repo = DriverRepository(session)
        drivers = await repo.list_pending()

    if not drivers:
        await message.answer("ℹ️ <i>Kutilayotgan haydovchilar ro'yxati bo'sh.</i>", parse_mode="HTML")
        return

    lines = [f"⏳ <b>To'lov Kutilayotgan Haydovchilar ({len(drivers)} ta):</b>\n"]
    for i, d in enumerate(drivers, start=1):
        lines.append(f"{i}. <b>{d.full_name}</b> | ID: <code>{d.telegram_id}</code>")

    await message.answer("\n".join(lines), parse_mode="HTML")


@admin_router.message(Command("stats"))
async def handle_stats(message: types.Message) -> None:
    """Display overall driver and system statistics."""
    if not is_admin(message.from_user.id):
        return

    async with get_db_session() as session:
        repo = DriverRepository(session)
        stats = await repo.get_statistics()

    total = sum(stats.values())
    text = (
        "📊 <b>Haydovchilar Statistikasi:</b>\n\n"
        f"🟢 <b>Faol obunachilar:</b> {stats.get('active', 0)}\n"
        f"🔴 <b>Muddati o'tganlar:</b> {stats.get('expired', 0)}\n"
        f"⏳ <b>Kutilayotganlar:</b> {stats.get('pending', 0)}\n"
        f"🚫 <b>Bloklanganlar:</b> {stats.get('banned', 0)}\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"👥 <b>Jami haydovchilar:</b> {total}\n"
    )
    await message.answer(text, parse_mode="HTML")


@admin_router.message(Command("id"))
async def handle_id(message: types.Message) -> None:
    """Return numeric Telegram IDs for the user and current chat."""
    user = message.from_user
    chat = message.chat

    user_id = user.id if user else "Noma'lum"
    user_name = user.full_name if user else "Noma'lum"
    admin_status = is_admin(user.id) if user else False

    text = (
        "🔍 <b>Telegram ID Ma'lumotlari:</b>\n\n"
        f"👤 <b>Foydalanuvchi:</b> {user_name}\n"
        f"🆔 <b>Foydalanuvchi ID:</b> <code>{user_id}</code>\n"
        f"👑 <b>Admin maqomi:</b> {'✅ Ha' if admin_status else '❌ Yo‘q'}\n\n"
        f"💬 <b>Chat nomi:</b> {chat.title or 'Shaxsiy chat'}\n"
        f"🆔 <b>Chat ID:</b> <code>{chat.id}</code>\n"
        f"📂 <b>Chat turi:</b> <code>{chat.type}</code>\n"
    )
    await message.answer(text, parse_mode="HTML")


@admin_router.message(Command("status"))
async def handle_status(message: types.Message) -> None:
    """Display current system status (Admin only)."""
    user = message.from_user
    if not user or not is_admin(user.id):
        await message.answer("⛔ <i>Bu buyruq faqat bot administratorlari uchun mo‘ljallangan.</i>", parse_mode="HTML")
        return

    groups_list = settings.public_group_id if isinstance(settings.public_group_id, list) else [settings.public_group_id]
    public_status = f"✅ {len(groups_list)} ta guruh" if groups_list else "⚠️ Barcha guruhlar (0)"
    groups_str = ", ".join(f"<code>{gid}</code>" for gid in groups_list) if groups_list else "Barcha guruhlar"
    private_status = "✅ Sozlangan" if settings.private_group_id != 0 else "⚠️ Belgilanmagan (0)"

    text = (
        "📊 <b>Tizim Holati:</b>\n\n"
        f"• <b>AI Dvigateli:</b> <code>{settings.classifier_type.upper()} AI (Anthropic Claude)</code>\n"
        f"• <b>Vaqt mintaqasi:</b> <code>{settings.timezone}</code>\n"
        f"• <b>Uzatish usuli:</b> <code>{settings.forward_mode}</code>\n"
        f"• <b>Umumiy guruhlar (Manbalar):</b> {groups_str} ({public_status})\n"
        f"• <b>VIP Haydovchilar guruhi:</b> <code>{settings.private_group_id}</code> ({private_status})\n"
        f"• <b>Saralash chegarasi:</b> <code>{settings.classifier_confidence_threshold}</code>\n"
        f"• <b>Adminlar soni:</b> {len(settings.admin_user_ids)}\n"
    )
    await message.answer(text, parse_mode="HTML")


@admin_router.message(Command("help"))
async def handle_help(message: types.Message) -> None:
    """Display help information."""
    user = message.from_user
    admin_status = is_admin(user.id) if user else False

    if admin_status:
        text = (
            "📖 <b>Administrator Yo'riqnomasi:</b>\n\n"
            "<b>Haydovchi qo'shish va obuna faollashtirish:</b>\n"
            "1. Haydovchi kartangizga pul o'tkazib, o'z Telegram ID raqamini yuboradi.\n"
            "2. To'lovni hisobingizda tekshirib, tasdiqlaysiz.\n"
            "3. <code>/activate &lt;tg_id&gt; 30</code> buyrug'ini yuborasiz.\n"
            "4. Bot bir martalik taklif havolasini taqdim etadi.\n"
            "5. Ushbu havola orqali haydovchi VIP guruhga qo'shiladi.\n\n"
            "<b>Muddatni uzaytirish:</b>\n"
            "• <code>/renew &lt;tg_id&gt; 30</code>\n\n"
            "<b>Guruhdan chiqarish:</b>\n"
            "• <code>/revoke &lt;tg_id&gt;</code>\n"
        )
    else:
        text = (
            "📖 <b>Foydalanuvchi Yo'riqnomasi:</b>\n\n"
            "Ushbu bot orqali viloyatlararo qatnovchi yo'lovchilarning buyurtmalari saralanadi.\n\n"
            "<b>Qanday qilib buyurtmalarni olish mumkin?</b>\n"
            f"1. Admin bilan bog'laning: @{settings.admin_username}\n"
            f"2. O'z ID raqamingizni (<code>{user.id if user else ''}</code>) adminga taqdim eting.\n"
            "3. Oylik to'lovni amalga oshiring.\n"
            "4. Admin yuborgan maxsus havola orqali VIP guruhga qo'shiling!"
        )
    await message.answer(text, parse_mode="HTML")


@admin_router.message(Command("review"))
async def handle_review(message: types.Message) -> None:
    """Initiate interactive active learning review session for ambiguous messages."""
    user = message.from_user
    if not user or not is_admin(user.id):
        await message.answer("⛔ <i>Bu buyruq faqat administratorlar uchun mo'ljallangan.</i>", parse_mode="HTML")
        return

    async with get_db_session() as session:
        pending_count = await CollectedMessageRepository.get_pending_count(session)
        if pending_count == 0:
            await message.answer(
                "🎉 <b>Tasdiqlash uchun xabarlar yo'q!</b>\n\n"
                "Guruhdan to'plangan barcha xabarlar ko'rib chiqilgan yoki hali noaniq xabarlar kelmadi.",
                parse_mode="HTML",
            )
            return

        record = await CollectedMessageRepository.get_next_pending(session)
        if not record:
            await message.answer("🎉 Hozircha tasdiqlanadigan yangi xabar yo'q.")
            return

        text, reply_markup = build_review_card(record, pending_count)
        await message.answer(text, reply_markup=reply_markup, parse_mode="HTML")


@admin_router.callback_query(F.data.startswith("rev:"))
async def handle_review_callback(callback: types.CallbackQuery) -> None:
    """Process active learning inline button actions."""
    user = callback.from_user
    if not user or not is_admin(user.id):
        await callback.answer("Ruxsat berilmagan!", show_alert=True)
        return

    parts = callback.data.split(":")
    action = parts[1]

    if action == "stop":
        await callback.message.edit_text(
            "🛑 <b>Ko'rib chiqish to'xtatildi.</b>\n\n"
            "Istalgan vaqtda /review buyrug'i orqali davom ettirishingiz mumkin.\n"
            "Yangi ma'lumotlarni modelga qo'llash uchun /retrain buyrug'ini bering.",
            parse_mode="HTML",
        )
        await callback.answer("To'xtatildi.")
        return

    if action == "skip":
        msg_id = int(parts[2])
        async with get_db_session() as session:
            await CollectedMessageRepository.mark_skipped(session, msg_id)
            next_record = await CollectedMessageRepository.get_next_pending(session)
            pending_count = await CollectedMessageRepository.get_pending_count(session)

            if not next_record:
                await callback.message.edit_text(
                    "🎉 <b>Barcha xabarlar ko'rib chiqildi!</b>\n\n"
                    "Yangi o'rganilgan ma'lumotlarni modelga singdirish uchun /retrain buyrug'ini bering.",
                    parse_mode="HTML",
                )
                await callback.answer("O'tkazib yuborildi.")
                return

            text, reply_markup = build_review_card(next_record, pending_count)
            await callback.message.edit_text(text, reply_markup=reply_markup, parse_mode="HTML")
            await callback.answer("O'tkazib yuborildi.")
            return

    if action == "label":
        chosen_label = parts[2]
        msg_id = int(parts[3])

        async with get_db_session() as session:
            record = await CollectedMessageRepository.mark_labeled(session, msg_id, chosen_label)
            if record:
                append_example_to_dataset(record.text, chosen_label)

            next_record = await CollectedMessageRepository.get_next_pending(session)
            pending_count = await CollectedMessageRepository.get_pending_count(session)

            if not next_record:
                await callback.message.edit_text(
                    f"✅ <b>Oxirgi xabar saqlandi! ({chosen_label})</b>\n\n"
                    "🎉 <b>Barcha to'plangan xabarlar tasdiqlandi!</b>\n\n"
                    "Modelni ushbu yangi ma'lumotlar bilan qayta o'qitish uchun quyidagi buyruqni bering:\n"
                    "👉 /retrain",
                    parse_mode="HTML",
                )
                await callback.answer("Saqlandi!")
                return

            text, reply_markup = build_review_card(next_record, pending_count)
            await callback.message.edit_text(text, reply_markup=reply_markup, parse_mode="HTML")
            await callback.answer("Saqlandi!")


@admin_router.message(Command("retrain"))
async def handle_retrain(message: types.Message) -> None:
    """Manually trigger model retraining with latest dataset."""
    user = message.from_user
    if not user or not is_admin(user.id):
        await message.answer("⛔ <i>Bu buyruq faqat administratorlar uchun mo'ljallangan.</i>", parse_mode="HTML")
        return

    status_msg = await message.answer(
        "⏳ <b>Model qayta o'qitilmoqda...</b>\n\n"
        "<i>Barcha yangi xabarlar tahlil qilinib, mashinali o'rganish modeli yangilanmoqda. Iltimos, kuting...</i>",
        parse_mode="HTML",
    )

    success, summary = await retrain_model_async()
    await status_msg.edit_text(summary, parse_mode="HTML")


@admin_router.message(Command("groups", "guruhlar"))
async def handle_list_groups(message: types.Message) -> None:
    """Display list of all dynamically monitored public groups."""
    user = message.from_user
    if not user or not is_admin(user.id):
        await message.answer("⛔ <i>Bu buyruq faqat administratorlar uchun mo'ljallangan.</i>", parse_mode="HTML")
        return

    from app.services.group_manager import get_group_manager
    group_mgr = get_group_manager()
    groups = await group_mgr.get_all_groups()

    if not groups:
        text = (
            "📋 <b>KUZATILAYOTGAN GURUHLAR RO'YXATI:</b>\n\n"
            "<i>Hozircha birorta ham guruh bazaga qo'shilmagan.</i>\n\n"
            "➕ <b>Guruh qo'shish uchun:</b>\n"
            "<code>/addgroup &lt;Chat ID yoki @username yoki havola&gt;</code>\n"
            "<i>Masalan: <code>/addgroup -1003726192046</code></i>"
        )
        await message.answer(text, parse_mode="HTML")
        return

    lines = [
        "📋 <b>KUZATILAYOTGAN GURUHLAR RO'YXATI:</b>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]
    builder = InlineKeyboardBuilder()

    for idx, grp in enumerate(groups, start=1):
        name = grp.title or f"Guruh ({grp.chat_id})"
        lines.append(f"<b>{idx}. {name}</b>")
        lines.append(f"   🆔 ID: <code>{grp.chat_id}</code>")
        if grp.username:
            lines.append(f"   🏷 Username: @{grp.username}")
        lines.append("")

        btn_label = f"🗑 O'chirish: {name[:16]}"
        builder.button(text=btn_label, callback_data=f"del_grp:{grp.chat_id}")

    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"📊 <i>Jami faol guruhlar: {len(groups)} ta</i>\n")
    lines.append("➕ <b>Yangi guruh qo'shish:</b>\n<code>/addgroup &lt;ID yoki @username yoki havola&gt;</code>\n")
    lines.append("❌ <b>Guruhni o'chirish:</b>\n<code>/delgroup &lt;ID&gt;</code>")

    builder.adjust(1)
    await message.answer("\n".join(lines), reply_markup=builder.as_markup(), parse_mode="HTML")


@admin_router.message(Command("addgroup", "qosh"))
async def handle_add_group(message: types.Message) -> None:
    """Dynamically add and join a new group: /addgroup <ID | @username | invite_link>"""
    user = message.from_user
    if not user or not is_admin(user.id):
        await message.answer("⛔ <i>Bu buyruq faqat administratorlar uchun mo'ljallangan.</i>", parse_mode="HTML")
        return

    parts = (message.text or "").strip().split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        help_text = (
            "ℹ️ <b>Guruh qo'shish bo'yicha ko'rsatma:</b>\n\n"
            "Format: <code>/addgroup &lt;ID yoki @username yoki havola&gt;</code>\n\n"
            "<b>Misollar:</b>\n"
            "• ID raqam orqali:\n"
            "  <code>/addgroup -1003726192046</code>\n"
            "  <code>/addgroup -5353349446</code>\n\n"
            "• Ommaviy username orqali:\n"
            "  <code>/addgroup @toshkent_vodiy_taksi</code>\n\n"
            "• Maxsus taklif havolasi orqali:\n"
            "  <code>/addgroup https://t.me/+AbCdEf123...</code>"
        )
        await message.answer(help_text, parse_mode="HTML")
        return

    target = parts[1].strip()
    status_msg = await message.answer("⏳ <i>Guruh tekshirilmoqda va ulanmoqda...</i>", parse_mode="HTML")

    from app.services.group_manager import get_group_manager
    group_mgr = get_group_manager()
    success, reply_text, _ = await group_mgr.add_group(target=target, admin_id=user.id)

    await status_msg.edit_text(reply_text, parse_mode="HTML")


@admin_router.message(Command("delgroup", "removegroup", "ochir"))
async def handle_remove_group(message: types.Message) -> None:
    """Dynamically remove a monitored group: /delgroup <Chat ID>"""
    user = message.from_user
    if not user or not is_admin(user.id):
        await message.answer("⛔ <i>Bu buyruq faqat administratorlar uchun mo'ljallangan.</i>", parse_mode="HTML")
        return

    parts = (message.text or "").strip().split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        help_text = (
            "ℹ️ <b>Guruhni o'chirish bo'yicha ko'rsatma:</b>\n\n"
            "Format: <code>/delgroup &lt;Chat ID&gt;</code>\n\n"
            "<b>Misol:</b>\n"
            "<code>/delgroup -5353349446</code>\n\n"
            "<i>(Mavjud barcha guruhlar ro'yxatini ko'rish uchun /groups yuboring).</i>"
        )
        await message.answer(help_text, parse_mode="HTML")
        return

    target_id = parts[1].strip()
    from app.services.group_manager import get_group_manager
    group_mgr = get_group_manager()
    success, reply_text = await group_mgr.remove_group(target_id)

    await message.answer(reply_text, parse_mode="HTML")


@admin_router.callback_query(F.data.startswith("del_grp:"))
async def handle_delete_group_callback(callback: types.CallbackQuery) -> None:
    """Handle interactive inline group deletion."""
    user = callback.from_user
    if not user or not is_admin(user.id):
        await callback.answer("⛔ Ruxsat berilmagan!", show_alert=True)
        return

    chat_id_str = callback.data.split(":", 1)[1]
    from app.services.group_manager import get_group_manager
    group_mgr = get_group_manager()

    success, reply_text = await group_mgr.remove_group(chat_id_str)
    await callback.answer("Guruh kuzatuvdan olib tashlandi!", show_alert=True)

    # Refresh the list view
    groups = await group_mgr.get_all_groups()
    if not groups:
        if callback.message:
            await callback.message.edit_text(
                "📋 <b>KUZATILAYOTGAN GURUHLAR RO'YXATI:</b>\n\n"
                "<i>Hozircha birorta ham guruh kuzatilmayapti.</i>\n\n"
                "➕ <b>Guruh qo'shish uchun:</b>\n<code>/addgroup &lt;ID yoki link&gt;</code>",
                parse_mode="HTML",
            )
        return

    lines = [
        "📋 <b>KUZATILAYOTGAN GURUHLAR RO'YXATI:</b>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]
    builder = InlineKeyboardBuilder()
    for idx, grp in enumerate(groups, start=1):
        name = grp.title or f"Guruh ({grp.chat_id})"
        lines.append(f"<b>{idx}. {name}</b>")
        lines.append(f"   🆔 ID: <code>{grp.chat_id}</code>")
        if grp.username:
            lines.append(f"   🏷 Username: @{grp.username}")
        lines.append("")
        builder.button(text=f"🗑 O'chirish: {name[:16]}", callback_data=f"del_grp:{grp.chat_id}")

    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"📊 <i>Jami faol guruhlar: {len(groups)} ta</i>\n")
    lines.append("➕ <b>Yangi guruh qo'shish:</b>\n<code>/addgroup &lt;ID yoki link&gt;</code>\n")
    lines.append("❌ <b>Guruhni o'chirish:</b>\n<code>/delgroup &lt;ID&gt;</code>")

    builder.adjust(1)
    if callback.message:
        await callback.message.edit_text("\n".join(lines), reply_markup=builder.as_markup(), parse_mode="HTML")


@admin_router.message(F.chat.type == ChatType.PRIVATE, ~F.text.startswith("/"))
async def handle_admin_private_test(message: types.Message) -> None:
    """Allow administrators to test message classification directly in private chat."""
    user = message.from_user
    if not user or not is_admin(user.id):
        return

    text = message.text or message.caption or ""
    if not text.strip():
        return

    import html
    from app.services.classification import get_classifier

    clf = get_classifier()
    res = await clf.classify(text)

    status_icon = (
        "🚖 <b>YO'LOVCHI BUYURTMASI</b> (VIP guruhga o'tadi)"
        if res.is_passenger_request
        else "❌ <b>YO'LOVCHI EMAS</b> (Guruhda qoladi)"
    )
    conf_pct = int(res.confidence * 100)
    seats_str = f"{res.seats} ta" if res.seats else "Ko'rsatilmagan"

    reply = (
        "🤖 <b>AI Tahlil Natijasi:</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"💬 <b>Xabar:</b> <i>\"{html.escape(text)}\"</i>\n\n"
        f"📊 <b>Xulosa:</b> {status_icon}\n"
        f"🎯 <b>Ishonch darajasi:</b> {conf_pct}%\n"
        f"📍 <b>Qayerdan:</b> {res.origin or 'Aniqlanmadi'}\n"
        f"🏁 <b>Qayerga:</b> {res.destination or 'Aniqlanmadi'}\n"
        f"👥 <b>O'rinlar:</b> {seats_str}\n"
        f"⚙️ <b>Model:</b> <code>{res.method}</code>\n\n"
        f"🧠 <b>Sabab / Izoh:</b> <i>{html.escape(res.reason)}</i>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "<i>💡 Eslatma: Ushbu xabar shaxsiy chatda sinov tariqasida tekshirildi. "
        "Xuddi shu xabar sozlangan guruhda yozilsa, avtomatik VIP guruhga uzatiladi!</i>"
    )
    await message.answer(reply, parse_mode="HTML")


