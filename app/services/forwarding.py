"""Message Forwarding and Deduplication Service."""

import asyncio
import hashlib
import html
import logging
import re
import time
from typing import Dict, Optional, Tuple
from aiogram import Bot, types
from aiogram.exceptions import TelegramAPIError, TelegramRetryAfter
from aiogram.utils.keyboard import InlineKeyboardBuilder

from app.config import Settings
from app.services.classification.base import ClassificationResult
from app.services.normalization.transliterator import normalize_uzbek_text

logger = logging.getLogger(__name__)


def clean_display_name(raw_name: Optional[str]) -> str:
    """Clean and sanitize display name, stripping spam/control glyphs and Cuneiform spam."""
    if not raw_name:
        return "Yo'lovchi"
    # Remove control chars and high supplementary unicode (Cuneiform, unprintable blocks)
    cleaned = re.sub(r"[\u0000-\u001f\u007f-\u009f\U00010000-\U0010ffff]", "", raw_name).strip()
    cleaned = re.sub(r"[^a-zA-Z0-9\u0400-\u04ff\s\.\-\'\`’ʻ_]", "", cleaned).strip()
    if not cleaned or len(cleaned) < 2:
        return "Yo'lovchi"
    return cleaned[:35]


class DeduplicationCache:
    """In-memory sliding-window cache to prevent immediate duplicate forwarding (3 min TTL)."""

    def __init__(self, ttl_seconds: int = 180):
        self.ttl = ttl_seconds
        # Mapping: hash -> timestamp
        self._cache: Dict[str, float] = {}

    def _generate_key(self, user_id: int, text: str) -> str:
        """Create deterministic key from author ID and normalized text."""
        norm = normalize_uzbek_text(text)
        payload = f"{user_id}:{norm}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def is_duplicate(self, user_id: int, text: str) -> bool:
        """Check if identical message from this user was already forwarded recently."""
        self._cleanup()
        key = self._generate_key(user_id, text)
        return key in self._cache

    def record(self, user_id: int, text: str) -> None:
        """Record forwarded message in cache."""
        key = self._generate_key(user_id, text)
        self._cache[key] = time.time()

    def _cleanup(self) -> None:
        """Purge entries older than TTL."""
        now = time.time()
        expired_keys = [k for k, ts in self._cache.items() if now - ts > self.ttl]
        for k in expired_keys:
            del self._cache[k]


class ForwardingService:
    """Handles dispatching passenger requests to the VIP drivers' group."""

    def __init__(
        self,
        bot: Bot,
        settings: Settings,
        deduplicator: Optional[DeduplicationCache] = None,
    ):
        self.bot = bot
        self.settings = settings
        self.deduplicator = deduplicator or DeduplicationCache(ttl_seconds=180)

    def is_user_on_cooldown(self, user_id: int) -> bool:
        """User 24h cooldown removed per business logic request."""
        return False

    def get_user_remaining_cooldown_hours(self, user_id: int) -> float:
        return 0.0

    def construct_message_link(
        self,
        chat_id: int,
        message_id: int,
        chat_username: Optional[str] = None,
    ) -> str:
        """Construct a deep link back to the original public message."""
        if chat_username:
            clean_username = chat_username.lstrip("@")
            return f"https://t.me/{clean_username}/{message_id}"

        # Supergroup internal ID format
        chat_id_str = str(chat_id)
        if chat_id_str.startswith("-100"):
            internal_id = chat_id_str[4:]
            return f"https://t.me/c/{internal_id}/{message_id}"

        return f"https://t.me/c/{abs(chat_id)}/{message_id}"

    def build_message_link(self, chat: types.Chat, message_id: int) -> str:
        """Construct a deep link back to the original public message."""
        return self.construct_message_link(
            chat_id=chat.id,
            message_id=message_id,
            chat_username=chat.username,
        )

    def build_card_payload(
        self,
        origin: Optional[str],
        destination: Optional[str],
        seats: Optional[int],
        raw_text: str,
        passenger_username: Optional[str],
        chat_id: int,
        message_id: int,
        time_val: Optional[str] = None,
        chat_username: Optional[str] = None,
        user_id: Optional[int] = None,
        passenger_name: Optional[str] = None,
        **kwargs,
    ) -> Tuple[str, types.InlineKeyboardMarkup]:
        """Construct an informative HTML card matching the user's requested clean design."""
        escaped_text = html.escape(raw_text)

        # Route formatting: Qo‘qon → Toshkent
        if origin and destination:
            route_str = f"{origin} → {destination}"
        elif destination:
            route_str = f"Noma'lum → {destination}"
        elif origin:
            route_str = f"{origin} → Noma'lum"
        else:
            route_str = "aniqlanmagan"

        seats_str = f"{seats} kishi" if seats else "aniqlanmagan"
        time_str = time_val or "aniqlanmagan"

        # Determine order type (passenger or cargo/pochta)
        order_type = kwargs.get("order_type")
        if not order_type:
            order_type = "cargo" if re.search(r"\b(pochta|poshta|poshda|yuk|posilka|korobka|dastavka|dostavka)\b", raw_text.lower()) else "passenger"

        is_cargo = order_type == "cargo"
        header = "📦 <b>YANGI POCHTA / YUK</b>" if is_cargo else "🚕 <b>YANGI YO‘LOVCHI</b>"
        client_label = "Buyurtmachi" if is_cargo else "Yo‘lovchi"

        # Passenger name & profile link
        clean_name = html.escape(clean_display_name(passenger_name))
        if passenger_username:
            clean_username = passenger_username.lstrip("@").strip()
            passenger_link = f'<a href="https://t.me/{clean_username}">{clean_name}</a> (@{clean_username})'
        elif user_id and user_id > 0:
            passenger_link = f'<a href="tg://user?id={user_id}">{clean_name}</a>'
        else:
            passenger_link = clean_name

        # Phone number detection in raw_text
        phone_match = re.search(r"(\+?998[\s-]?\d{2}[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2})", raw_text)
        phone_str = None
        if phone_match:
            phone_str = phone_match.group(1).replace(" ", "").replace("-", "")
            if not phone_str.startswith("+"):
                phone_str = f"+{phone_str}"

        is_voice = kwargs.get("is_voice", False)

        card_lines = [
            header,
            f"📍 <b>{route_str}</b>",
            f"👤 <b>{client_label}:</b> {passenger_link}",
        ]
        if is_cargo:
            card_lines.append("📦 <b>Turi:</b> Pochta / Yuk")
        else:
            card_lines.append(f"👥 <b>Yo‘lovchilar:</b> {seats_str}")

        card_lines.append(f"🕒 <b>Vaqt:</b> {time_str}")
        if is_voice:
            card_lines.append("🎙 <b>Format:</b> Ovozli xabar")
        if phone_str:
            card_lines.append(f"📞 <b>Telefon:</b> {phone_str}")

        quote_header = "🎙 <i>[Ovozli xabar matni]:</i>\n" if is_voice else ""
        card_lines.extend([
            "",
            "💬 <b>Asl xabar:</b>",
            f"<blockquote>{quote_header}{escaped_text}</blockquote>",
        ])
        text = "\n".join(card_lines)

        builder = InlineKeyboardBuilder()

        # 1. Primary button: Yo'lovchiga yozish / Buyurtmachiga yozish (only if valid username exists)
        if passenger_username:
            clean_username = passenger_username.lstrip("@").strip()
            if clean_username:
                btn_text = "💬 Buyurtmachiga yozish" if is_cargo else "💬 Yo'lovchiga yozish"
                builder.button(text=btn_text, url=f"https://t.me/{clean_username}")

        # 2. 100% reliable Deep Link to original message in group (solves hidden account / no username problem)
        if chat_id and message_id:
            msg_link = self.construct_message_link(
                chat_id=chat_id,
                message_id=message_id,
                chat_username=chat_username,
            )
            builder.button(text="📍 Guruhdagi xabarga o'tish", url=msg_link)

        builder.adjust(1)
        reply_markup = builder.as_markup() if len(builder.as_markup().inline_keyboard) > 0 else None
        return text, reply_markup

    def build_passenger_card(
        self,
        message: types.Message,
        classification: ClassificationResult,
        is_voice: bool = False,
        raw_text: Optional[str] = None,
    ) -> Tuple[str, types.InlineKeyboardMarkup]:
        """Construct an informative HTML card and action buttons."""
        user = message.from_user
        msg_text = raw_text if raw_text is not None else (message.text or message.caption or "")
        return self.build_card_payload(
            origin=classification.origin,
            destination=classification.destination,
            seats=classification.seats,
            order_type=getattr(classification, "order_type", "passenger"),
            time_val=getattr(classification, "time", None),
            raw_text=msg_text,
            passenger_username=user.username if user else None,
            user_id=user.id if user else None,
            passenger_name=user.full_name if user else None,
            chat_id=message.chat.id,
            message_id=message.message_id,
            chat_username=message.chat.username,
            is_voice=is_voice,
        )

    async def forward_passenger_request(
        self,
        message: types.Message,
        classification: ClassificationResult,
        is_voice: bool = False,
        raw_text: Optional[str] = None,
        voice_bytes: Optional[bytes] = None,
        **kwargs,
    ) -> bool:
        """
        Forward or copy passenger request to the private group with flood control.
        Returns True if successful, False otherwise.
        """
        if self.settings.private_group_id == 0:
            logger.warning("PRIVATE_GROUP_ID is not configured (0). Message not forwarded.")
            return False

        user = message.from_user
        user_id = user.id if user else 0
        msg_text = raw_text if raw_text is not None else (message.text or message.caption or "")
        chat_id = message.chat.id
        message_id = message.message_id
        chat_username = message.chat.username
        passenger_name = user.full_name if user else "Yo'lovchi"

        # Check short deduplication (avoid exact duplicate sends within 3 mins)
        if self.deduplicator.is_duplicate(user_id, msg_text):
            logger.info("Skipping duplicate passenger request from user %s", user_id)
            return False

        target_chat_id = self.settings.private_group_id

        # Retry logic for Telegram rate limits (HTTP 429)
        max_retries = 2
        for attempt in range(max_retries):
            try:
                if is_voice:
                    # 1. Forward original voice message so drivers can listen to the audio directly
                    voice_forwarded = False
                    try:
                        await self.bot.forward_message(
                            chat_id=target_chat_id,
                            from_chat_id=message.chat.id,
                            message_id=message.message_id,
                        )
                        voice_forwarded = True
                        logger.info("✅ Voice message forwarded to VIP group %s", target_chat_id)
                    except Exception as fwd_err:
                        logger.debug("Bot forward voice failed (%s)", fwd_err)

                    if not voice_forwarded and voice_bytes:
                        try:
                            await self.bot.send_voice(
                                chat_id=target_chat_id,
                                voice=types.BufferedInputFile(voice_bytes, filename="voice.ogg"),
                            )
                            voice_forwarded = True
                            logger.info("✅ Sent voice note via send_voice to VIP group %s", target_chat_id)
                        except Exception as snd_err:
                            logger.warning("Bot send_voice failed (%s)", snd_err)

                    # 2. ALSO send the structured card
                    card_text, reply_markup = self.build_passenger_card(
                        message=message,
                        classification=classification,
                        is_voice=True,
                        raw_text=msg_text,
                    )
                    await self.bot.send_message(
                        chat_id=target_chat_id,
                        text=card_text,
                        reply_markup=reply_markup,
                        parse_mode="HTML",
                        disable_web_page_preview=True,
                    )
                elif self.settings.forward_mode == "forward":
                    try:
                        await self.bot.forward_message(
                            chat_id=target_chat_id,
                            from_chat_id=message.chat.id,
                            message_id=message.message_id,
                        )
                    except Exception as fwd_err:
                        logger.warning("Bot forward_message failed (%s)", fwd_err)

                    card_text, reply_markup = self.build_passenger_card(
                        message=message,
                        classification=classification,
                        is_voice=False,
                        raw_text=msg_text,
                    )
                    await self.bot.send_message(
                        chat_id=target_chat_id,
                        text=card_text,
                        reply_markup=reply_markup,
                        parse_mode="HTML",
                        disable_web_page_preview=True,
                    )
                else:
                    # Default: enriched copy mode
                    card_text, reply_markup = self.build_passenger_card(
                        message=message,
                        classification=classification,
                        is_voice=False,
                        raw_text=msg_text,
                    )
                    await self.bot.send_message(
                        chat_id=target_chat_id,
                        text=card_text,
                        reply_markup=reply_markup,
                        parse_mode="HTML",
                        disable_web_page_preview=True,
                    )

                # Record in deduplication cache
                self.deduplicator.record(user_id, msg_text)

                # Save record to audit database
                try:
                    from app.database.session import get_db_session
                    from app.repositories.forwarded_repo import ForwardedMessageRepository

                    async with get_db_session() as session:
                        await ForwardedMessageRepository.save_record(
                            session=session,
                            public_message_id=message.message_id,
                            public_chat_id=message.chat.id,
                            author_telegram_id=user_id,
                            author_username=user.username if user else None,
                            author_full_name=user.full_name if user else None,
                            origin=classification.origin,
                            destination=classification.destination,
                            seats=classification.seats,
                            confidence=classification.confidence,
                            raw_text=raw_text,
                        )
                except Exception as db_err:
                    logger.debug("Could not persist forwarded record to DB: %s", db_err)

                logger.info(
                    "✅ Successfully dispatched passenger request to private group %s | Message ID: %s",
                    target_chat_id,
                    message.message_id,
                )
                return True

            except TelegramRetryAfter as e:
                logger.warning("Telegram rate limit hit. Sleeping for %s seconds...", e.retry_after)
                await asyncio.sleep(e.retry_after)
            except TelegramAPIError as e:
                logger.error("Failed to forward card to VIP group %s: %s. Attempting fallback...", target_chat_id, e)
                # Fallback 1: Try native forward if bot is member of group
                try:
                    await self.bot.forward_message(
                        chat_id=target_chat_id,
                        from_chat_id=chat_id,
                        message_id=message_id,
                    )
                    self.deduplicator.record(user_id, raw_text)
                    logger.info("✅ Fallback native forward succeeded to VIP group %s", target_chat_id)
                    return True
                except Exception as fwd_err:
                    logger.warning("Native forward failed (%s). Sending clean text fallback...", fwd_err)
                    try:
                        clean_name = html.escape((passenger_name or "").strip() or "Yo'lovchi")
                        user_ref = f'<a href="tg://user?id={user_id}">{clean_name}</a>' if (user_id and user_id > 0) else clean_name
                        is_cargo = getattr(classification, "order_type", None) == "cargo" or bool(
                            re.search(r"\b(pochta|poshta|poshda|yuk|posilka|korobka|dastavka|dostavka)\b", raw_text.lower())
                        )
                        fallback_header = "📦 <b>YANGI POCHTA / YUK</b>" if is_cargo else "🚕 <b>YANGI YO‘LOVCHI</b>"
                        fallback_client = "Buyurtmachi" if is_cargo else "Yo‘lovchi"
                        fallback_text = (
                            f"{fallback_header}\n"
                            f"👤 <b>{fallback_client}:</b> {user_ref}\n\n"
                            "💬 <b>Asl xabar:</b>\n"
                            f"<blockquote>{html.escape(raw_text)}</blockquote>"
                        )
                        await self.bot.send_message(
                            chat_id=target_chat_id,
                            text=fallback_text,
                            parse_mode="HTML",
                            disable_web_page_preview=True,
                        )
                        self.deduplicator.record(user_id, raw_text)
                        logger.info("✅ Fallback clean text send succeeded to VIP group %s", target_chat_id)
                        return True
                    except Exception as final_err:
                        logger.error("All fallback forwarding methods failed for VIP group %s: %s", target_chat_id, final_err)
                        return False
            except Exception as e:
                logger.exception("Unexpected error while forwarding message: %s", e)
                return False

        return False

    async def forward_custom_passenger_request(
        self,
        chat_id: int,
        message_id: int,
        user_id: int,
        raw_text: str,
        classification: ClassificationResult,
        passenger_name: str = "Yo'lovchi",
        passenger_username: Optional[str] = None,
        chat_username: Optional[str] = None,
        is_voice: bool = False,
        voice_bytes: Optional[bytes] = None,
        userbot_client: Optional[object] = None,
        telethon_message: Optional[object] = None,
        **kwargs,
    ) -> bool:
        """Forward a passenger request captured from any source (e.g. Userbot) to VIP group."""
        if self.settings.private_group_id == 0:
            logger.warning("PRIVATE_GROUP_ID is not configured (0). Message not forwarded.")
            return False

        if self.deduplicator.is_duplicate(user_id, raw_text):
            logger.info("Skipping duplicate passenger request from user %s", user_id)
            return False

        target_chat_id = self.settings.private_group_id
        max_retries = 2
        for attempt in range(max_retries):
            try:
                if is_voice:
                    # 1. Forward original voice message so drivers can listen to the audio directly
                    voice_forwarded = False
                    if userbot_client and telethon_message:
                        try:
                            await userbot_client.forward_messages(
                                entity=target_chat_id,
                                messages=telethon_message,
                                drop_author=False,
                            )
                            voice_forwarded = True
                            logger.info("✅ Userbot forwarded voice message to VIP group %s", target_chat_id)
                        except Exception as ub_fwd_err:
                            try:
                                await userbot_client.forward_messages(
                                    entity=target_chat_id,
                                    messages=message_id,
                                    from_peer=chat_id,
                                    drop_author=False,
                                )
                                voice_forwarded = True
                                logger.info("✅ Userbot forwarded voice message (by ID) to VIP group %s", target_chat_id)
                            except Exception as ub_fwd_err2:
                                logger.warning("Userbot forward_messages failed: %s / %s", ub_fwd_err, ub_fwd_err2)

                    if not voice_forwarded:
                        try:
                            await self.bot.forward_message(
                                chat_id=target_chat_id,
                                from_chat_id=chat_id,
                                message_id=message_id,
                            )
                            voice_forwarded = True
                            logger.info("✅ Bot forwarded voice message to VIP group %s", target_chat_id)
                        except Exception as bot_fwd_err:
                            logger.debug("Bot forward_message failed (%s)", bot_fwd_err)

                    if not voice_forwarded and voice_bytes:
                        try:
                            await self.bot.send_voice(
                                chat_id=target_chat_id,
                                voice=types.BufferedInputFile(voice_bytes, filename="voice.ogg"),
                            )
                            voice_forwarded = True
                            logger.info("✅ Bot sent voice note to VIP group %s", target_chat_id)
                        except Exception as snd_err:
                            logger.warning("Bot send_voice failed (%s)", snd_err)

                    # 2. ALSO send the structured card
                    card_text, reply_markup = self.build_card_payload(
                        origin=classification.origin,
                        destination=classification.destination,
                        seats=classification.seats,
                        order_type=getattr(classification, "order_type", "passenger"),
                        time_val=getattr(classification, "time", None),
                        raw_text=raw_text,
                        passenger_username=passenger_username,
                        user_id=user_id,
                        passenger_name=passenger_name,
                        chat_id=chat_id,
                        message_id=message_id,
                        chat_username=chat_username,
                        is_voice=True,
                    )
                    await self.bot.send_message(
                        chat_id=target_chat_id,
                        text=card_text,
                        reply_markup=reply_markup,
                        parse_mode="HTML",
                        disable_web_page_preview=True,
                    )
                    self.deduplicator.record(user_id, raw_text)
                    logger.info("✅ Successfully dispatched voice request & card to VIP group %s", target_chat_id)

                else:
                    # Non-voice messages:
                    # 1. If FORWARD_MODE is 'forward', attempt native forward first (Userbot or Bot)
                    forwarded = False
                    if self.settings.forward_mode == "forward":
                        if userbot_client and telethon_message:
                            try:
                                await userbot_client.forward_messages(
                                    entity=target_chat_id,
                                    messages=telethon_message,
                                    drop_author=False,
                                )
                                forwarded = True
                                logger.info("✅ Userbot native forward succeeded to VIP group %s", target_chat_id)
                            except Exception as ub_err:
                                logger.warning("Userbot forward failed (%s), trying bot forward...", ub_err)

                        if not forwarded:
                            try:
                                await self.bot.forward_message(
                                    chat_id=target_chat_id,
                                    from_chat_id=chat_id,
                                    message_id=message_id,
                                )
                                forwarded = True
                                logger.info("✅ Bot native forward succeeded to VIP group %s", target_chat_id)
                            except Exception as fwd_err:
                                logger.warning(
                                    "Bot forward_message failed (%s). Falling back to enriched card...",
                                    fwd_err,
                                )

                    # 2. ALWAYS send the structured card with action buttons (including "📍 Guruhdagi xabarga o'tish")
                    card_text, reply_markup = self.build_card_payload(
                        origin=classification.origin,
                        destination=classification.destination,
                        seats=classification.seats,
                        order_type=getattr(classification, "order_type", "passenger"),
                        time_val=getattr(classification, "time", None),
                        raw_text=raw_text,
                        passenger_username=passenger_username,
                        user_id=user_id,
                        passenger_name=passenger_name,
                        chat_id=chat_id,
                        message_id=message_id,
                        chat_username=chat_username,
                        is_voice=False,
                    )
                    await self.bot.send_message(
                        chat_id=target_chat_id,
                        text=card_text,
                        reply_markup=reply_markup,
                        parse_mode="HTML",
                        disable_web_page_preview=True,
                    )
                    self.deduplicator.record(user_id, raw_text)

                # Save record to audit database
                try:
                    from app.database.session import get_db_session
                    from app.repositories.forwarded_repo import ForwardedMessageRepository

                    async with get_db_session() as session:
                        await ForwardedMessageRepository.save_record(
                            session=session,
                            public_message_id=message_id,
                            public_chat_id=chat_id,
                            author_telegram_id=user_id,
                            author_username=passenger_username,
                            author_full_name=passenger_name,
                            origin=classification.origin,
                            destination=classification.destination,
                            seats=classification.seats,
                            confidence=classification.confidence,
                            raw_text=raw_text,
                        )
                except Exception as db_err:
                    logger.debug("Could not persist custom forwarded record to DB: %s", db_err)

                logger.info(
                    "✅ Successfully dispatched userbot passenger request to VIP group %s | Original Chat: %s | Msg: %s",
                    target_chat_id,
                    chat_id,
                    message_id,
                )
                return True
            except TelegramRetryAfter as e:
                logger.warning("Telegram rate limit hit. Sleeping for %s seconds...", e.retry_after)
                await asyncio.sleep(e.retry_after)
            except TelegramAPIError as e:
                logger.error("Failed to forward card to VIP group %s: %s. Attempting fallback...", target_chat_id, e)
                # Fallback 1: Try native forward if bot is member of group
                try:
                    await self.bot.forward_message(
                        chat_id=target_chat_id,
                        from_chat_id=chat_id,
                        message_id=message_id,
                    )
                    self.deduplicator.record(user_id, raw_text)
                    logger.info("✅ Fallback native forward succeeded to VIP group %s", target_chat_id)
                    return True
                except Exception as fwd_err:
                    logger.warning("Native forward failed (%s). Sending clean text fallback...", fwd_err)
                    try:
                        clean_name = html.escape((passenger_name or "").strip() or "Yo'lovchi")
                        user_ref = f'<a href="tg://user?id={user_id}">{clean_name}</a>' if (user_id and user_id > 0) else clean_name
                        is_cargo = getattr(classification, "order_type", None) == "cargo" or bool(
                            re.search(r"\b(pochta|poshta|poshda|yuk|posilka|korobka|dastavka|dostavka)\b", raw_text.lower())
                        )
                        fallback_header = "📦 <b>YANGI POCHTA / YUK</b>" if is_cargo else "🚕 <b>YANGI YO‘LOVCHI</b>"
                        fallback_client = "Buyurtmachi" if is_cargo else "Yo‘lovchi"
                        fallback_text = (
                            f"{fallback_header}\n"
                            f"👤 <b>{fallback_client}:</b> {user_ref}\n\n"
                            "💬 <b>Asl xabar:</b>\n"
                            f"<blockquote>{html.escape(raw_text)}</blockquote>"
                        )
                        await self.bot.send_message(
                            chat_id=target_chat_id,
                            text=fallback_text,
                            parse_mode="HTML",
                            disable_web_page_preview=True,
                        )
                        self.deduplicator.record(user_id, raw_text)
                        logger.info("✅ Fallback clean text send succeeded to VIP group %s", target_chat_id)
                        return True
                    except Exception as final_err:
                        logger.error("All fallback forwarding methods failed for VIP group %s: %s", target_chat_id, final_err)
                        return False
            except Exception as e:
                logger.exception("Unexpected error while forwarding message: %s", e)
                return False

        return False
