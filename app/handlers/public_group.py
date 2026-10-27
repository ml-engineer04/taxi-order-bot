"""Public Group Message Listener and Forwarding Router."""

from datetime import datetime, timezone
import logging
from aiogram import Router, types, F
from aiogram.enums import ChatType
from aiogram.filters import ChatMemberUpdatedFilter, JOIN_TRANSITION

from app.config import get_settings
from app.database.session import get_db_session
from app.repositories.collected_repo import CollectedMessageRepository
from app.services.classification import get_classifier
from app.services.forwarding import ForwardingService

logger = logging.getLogger(__name__)
public_group_router = Router(name="public_group_router")
settings = get_settings()
classifier = get_classifier()
handler_started_at = datetime.now(timezone.utc)

# Global forwarding service reference initialized dynamically
_forwarding_service: ForwardingService = None


def get_forwarding_service(bot) -> ForwardingService:
    global _forwarding_service
    if _forwarding_service is None or _forwarding_service.bot != bot:
        _forwarding_service = ForwardingService(bot=bot, settings=settings)
    return _forwarding_service


def extract_message_text(message: types.Message) -> str:
    """Extract text from normal message or caption from media message."""
    return (message.text or message.caption or "").strip()


@public_group_router.my_chat_member(ChatMemberUpdatedFilter(member_status_changed=JOIN_TRANSITION))
async def handle_bot_added_to_chat(event: types.ChatMemberUpdated) -> None:
    """Triggered when the bot is added to a group or channel."""
    chat = event.chat
    logger.info("Bot was added to chat: %s (ID: %s, Type: %s)", chat.title, chat.id, chat.type)

    welcome_text = (
        "🚖 <b>Viloyatlar Mijoz Boti guruhga muvaffaqiyatli qo'shildi!</b>\n\n"
        f"🏷 <b>Guruh nomi:</b> {chat.title}\n"
        f"🆔 <b>Ushbu guruh Chat ID raqami:</b> <code>{chat.id}</code>\n\n"
        "<b>Sozlash bo'yicha ko'rsatma:</b>\n"
        "• Agar bu <b>yo'lovchilar umumiy guruhi</b> bo'lsa, <code>.env</code> fayliga quyidagini yozing:\n"
        f"  <code>PUBLIC_GROUP_ID={chat.id}</code>\n\n"
        "• Agar bu <b>VIP haydovchilar guruhi</b> bo'lsa, <code>.env</code> fayliga quyidagini yozing:\n"
        f"  <code>PRIVATE_GROUP_ID={chat.id}</code>\n\n"
        "<i>Eslatma: Guruhdagi barcha xabarlarni o'qiy olishi uchun botga guruhda Administrator huquqini bering!</i>"
    )

    try:
        await event.bot.send_message(chat_id=chat.id, text=welcome_text, parse_mode="HTML")
    except Exception as exc:
        logger.warning("Could not send welcome message to group %s: %s", chat.id, exc)


@public_group_router.message(F.chat.type.in_([ChatType.GROUP, ChatType.SUPERGROUP]))
async def handle_public_group_message(message: types.Message) -> None:
    """
    Monitor incoming messages from the public group, classify passenger intent,
    and automatically forward verified ride requests to the VIP drivers' group.
    """
    user = message.from_user

    # 1. Ignore messages without a sender or messages from other bots
    if not user or user.is_bot:
        return

    # 2. Ignore historical messages sent before bot startup or lagging backlog
    msg_date = getattr(message, "date", None)
    if isinstance(msg_date, datetime):
        if msg_date.tzinfo is None:
            msg_date = msg_date.replace(tzinfo=timezone.utc)
        if msg_date < handler_started_at:
            logger.debug("Skipping historical message %s from %s", message.message_id, msg_date)
            return
        if (datetime.now(timezone.utc) - msg_date).total_seconds() > 120:
            logger.debug("Skipping backlog message %s (age > 120s)", message.message_id)
            return

    # 3. Check 24-hour user cooldown (skip repeat messages within 1 day)
    forwarder = get_forwarding_service(message.bot)
    is_cooldown = False
    if hasattr(forwarder, "is_user_on_cooldown"):
        check_val = forwarder.is_user_on_cooldown(user.id)
        if isinstance(check_val, bool):
            is_cooldown = check_val

    if is_cooldown:
        remaining = forwarder.get_user_remaining_cooldown_hours(user.id)
        logger.info(
            "⏭️ [USER 24H SKIP] User %s already forwarded within 1 day (%.1fh remaining). Skipping.",
            user.id,
            remaining,
        )
        return

    # 4. Ignore commands (e.g. /id, /start) as they are handled by admin_router
    raw_text = extract_message_text(message)
    if not raw_text or raw_text.startswith("/"):
        return

    chat_id = message.chat.id
    current_settings = get_settings()

    # 5. Only monitor messages from configured PUBLIC_GROUP_ID(s)
    if not current_settings.is_monitored_public_group(chat_id):
        logger.debug("Ignored message from unmonitored group ID: %s", chat_id)
        return

    # 4. Classify passenger transportation intent
    try:
        current_classifier = get_classifier()
        result = await current_classifier.classify(raw_text)
    except Exception as exc:
        logger.error(
            "Classification failure on message %s from user %s: %s",
            message.message_id,
            user.id,
            exc,
        )
        return

    username_display = f"@{user.username}" if user.username else f"{user.full_name} ({user.id})"

    if result.is_passenger_request:
        logger.info(
            "🎯 [PASSENGER DETECTED] Conf: %.2f | Origin: %s | Dest: %s | Seats: %s | From: %s | Method: %s",
            result.confidence,
            result.origin,
            result.destination,
            result.seats,
            username_display,
            result.method,
        )

        # 5. Dispatch to VIP drivers' private group
        try:
            forwarder = get_forwarding_service(message.bot)
            await forwarder.forward_passenger_request(message, result)
        except Exception as exc:
            logger.error("Failed to forward passenger request: %s", exc)

    else:
        # Check if score falls in the ambiguous zone for review logging & collection
        is_candidate = (
            (0.30 <= result.confidence < settings.classifier_confidence_threshold)
            or (
                result.confidence <= 0.75
                and any(
                    kw in raw_text.lower()
                    for kw in ("odam", "joy", "mashina", "moshina", "taksi", "bor", "ket", "kishi", "yo'lovchi")
                )
            )
        )

        if is_candidate and len(raw_text.strip()) >= 5:
            logger.warning(
                "⚠️ [AMBIGUOUS MESSAGE REVIEW] Conf: %.2f | Reason: %s | User: %s",
                result.confidence,
                result.reason,
                user.id,
            )
            try:
                pred_label = "OTHER"
                if hasattr(classifier, "predict_message"):
                    pred = classifier.predict_message(raw_text)
                    pred_label = pred.get("label", "OTHER")

                async with get_db_session() as session:
                    await CollectedMessageRepository.save_if_new(
                        session=session,
                        text=raw_text,
                        predicted_label=pred_label,
                        confidence=result.confidence,
                    )
            except Exception as e:
                logger.debug("Active learning save failed: %s", e)
        else:
            logger.debug(
                "⏭️ [IGNORED MSG] Conf: %.2f | Reason: %s | User: %s",
                result.confidence,
                result.reason,
                user.id,
            )
