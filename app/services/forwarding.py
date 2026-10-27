"""Message Forwarding and Deduplication Service."""

import asyncio
import hashlib
import html
import logging
import time
from typing import Dict, Optional, Tuple
from aiogram import Bot, types
from aiogram.exceptions import TelegramAPIError, TelegramRetryAfter
from aiogram.utils.keyboard import InlineKeyboardBuilder

from app.config import Settings
from app.services.classification.base import ClassificationResult
from app.services.normalization.transliterator import normalize_uzbek_text

logger = logging.getLogger(__name__)


class DeduplicationCache:
    """In-memory sliding-window cache to prevent duplicate forwarding."""

    def __init__(self, ttl_seconds: int = 86400):
        self.ttl = ttl_seconds
        # Mapping: hash -> timestamp
        self._cache: Dict[str, float] = {}

    def _generate_key(self, user_id: int, text: str) -> str:
        """Create deterministic key from author ID and normalized text."""
        norm = normalize_uzbek_text(text)
        payload = f"{user_id}:{norm}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def is_duplicate(self, user_id: int, text: str) -> bool:
        """Check if message from this user was already forwarded recently."""
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


class UserCooldownRegistry:
    """Tracks passenger users to enforce 1-day (24h) skip window."""

    def __init__(self, cooldown_seconds: int = 86400):
        self.cooldown_seconds = cooldown_seconds
        # Mapping: user_id -> timestamp
        self._cooldowns: Dict[int, float] = {}

    def is_on_cooldown(self, user_id: int) -> bool:
        if not user_id:
            return False
        self._cleanup()
        return user_id in self._cooldowns

    def get_remaining_hours(self, user_id: int) -> float:
        ts = self._cooldowns.get(user_id)
        if not ts:
            return 0.0
        remaining = self.cooldown_seconds - (time.time() - ts)
        return max(0.0, remaining / 3600.0)

    def record(self, user_id: int, timestamp: Optional[float] = None) -> None:
        if not user_id:
            return
        self._cooldowns[user_id] = timestamp or time.time()

    def populate(self, mapping: Dict[int, float]) -> None:
        now = time.time()
        for uid, ts in mapping.items():
            if now - ts < self.cooldown_seconds:
                self._cooldowns[uid] = ts

    def _cleanup(self) -> None:
        now = time.time()
        expired = [uid for uid, ts in self._cooldowns.items() if now - ts >= self.cooldown_seconds]
        for uid in expired:
            del self._cooldowns[uid]


class ForwardingService:
    """Handles dispatching passenger requests to the VIP drivers' group."""

    def __init__(
        self,
        bot: Bot,
        settings: Settings,
        deduplicator: Optional[DeduplicationCache] = None,
        user_cooldowns: Optional[UserCooldownRegistry] = None,
    ):
        self.bot = bot
        self.settings = settings
        cooldown_sec = getattr(settings, "user_cooldown_hours", 24) * 3600
        self.deduplicator = deduplicator or DeduplicationCache(ttl_seconds=cooldown_sec)
        self.user_cooldowns = user_cooldowns or UserCooldownRegistry(cooldown_seconds=cooldown_sec)

    async def load_recent_cooldowns_from_db(self) -> int:
        """Hydrate in-memory user cooldowns from database records within the last 24h."""
        try:
            from app.database.session import get_db_session
            from app.repositories.forwarded_repo import ForwardedMessageRepository

            async with get_db_session() as session:
                cooldowns = await ForwardedMessageRepository.get_recent_user_cooldowns(
                    session=session,
                    hours=self.settings.user_cooldown_hours,
                )
                self.user_cooldowns.populate(cooldowns)
                logger.info(
                    "Loaded %d active user cooldown(s) (last %dh) from database.",
                    len(cooldowns),
                    self.settings.user_cooldown_hours,
                )
                return len(cooldowns)
        except Exception as exc:
            logger.warning("Could not load user cooldowns from DB: %s", exc)
            return 0

    def is_user_on_cooldown(self, user_id: int) -> bool:
        """Check if user has already forwarded a ride request within the 24-hour window."""
        return self.user_cooldowns.is_on_cooldown(user_id)

    def get_user_remaining_cooldown_hours(self, user_id: int) -> float:
        """Get remaining cooldown hours for user."""
        return self.user_cooldowns.get_remaining_hours(user_id)

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
        confidence: float,
        raw_text: str,
        passenger_name: str,
        passenger_username: Optional[str],
        chat_id: int,
        message_id: int,
        chat_username: Optional[str] = None,
    ) -> Tuple[str, types.InlineKeyboardMarkup]:
        """Construct an informative HTML card and action buttons."""
        escaped_text = html.escape(raw_text)
        origin_val = origin or "Aniqlanmadi"
        dest_val = destination or "Aniqlanmadi"
        seats_str = f"{seats} ta joy" if seats else "Ko'rsatilmagan"
        escaped_name = html.escape(passenger_name) if passenger_name else "Yo'lovchi"

        card_lines = [
            "🚖 <b>YANGI YO'LOVCHI BUYURTMASI!</b>",
            "━━━━━━━━━━━━━━━━━━━━",
            f"📍 <b>Yo'nalish:</b> {origin_val} ➡️ <b>{dest_val}</b>",
            f"👥 <b>O'rinlar:</b> {seats_str}",
            f"👤 <b>Yo'lovchi:</b> {escaped_name}",
            "",
            "💬 <b>Asl xabar:</b>",
            f"<blockquote>{escaped_text}</blockquote>",
            "━━━━━━━━━━━━━━━━━━━━",
            f"<i>Ishonch darajasi: {int(confidence * 100)}%</i>",
        ]
        text = "\n".join(card_lines)

        builder = InlineKeyboardBuilder()
        msg_link = self.construct_message_link(chat_id, message_id, chat_username)
        builder.button(text="🔗 Xabarni guruhda ochish", url=msg_link)

        if passenger_username:
            clean_username = passenger_username.lstrip("@")
            builder.button(text="💬 Yo'lovchiga yozish", url=f"https://t.me/{clean_username}")

        builder.adjust(1)
        return text, builder.as_markup()

    def build_passenger_card(
        self,
        message: types.Message,
        classification: ClassificationResult,
    ) -> Tuple[str, types.InlineKeyboardMarkup]:
        """Construct an informative HTML card and action buttons."""
        user = message.from_user
        raw_text = message.text or message.caption or ""
        return self.build_card_payload(
            origin=classification.origin,
            destination=classification.destination,
            seats=classification.seats,
            confidence=classification.confidence,
            raw_text=raw_text,
            passenger_name=user.full_name if user else "Yo'lovchi",
            passenger_username=user.username if user else None,
            chat_id=message.chat.id,
            message_id=message.message_id,
            chat_username=message.chat.username,
        )

    async def forward_passenger_request(
        self,
        message: types.Message,
        classification: ClassificationResult,
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
        raw_text = message.text or message.caption or ""

        # 1. Check 24-hour user cooldown (1 day window)
        if self.is_user_on_cooldown(user_id):
            remaining = self.get_user_remaining_cooldown_hours(user_id)
            logger.info(
                "⏭️ [24H COOLDOWN] User %s already forwarded within 1 day (%.1fh remaining). Skipping.",
                user_id,
                remaining,
            )
            return False

        # 2. Check deduplication
        if self.deduplicator.is_duplicate(user_id, raw_text):
            logger.info("Skipping duplicate passenger request from user %s", user_id)
            return False

        target_chat_id = self.settings.private_group_id

        # Retry logic for Telegram rate limits (HTTP 429)
        max_retries = 2
        for attempt in range(max_retries):
            try:
                if self.settings.forward_mode == "forward":
                    await self.bot.forward_message(
                        chat_id=target_chat_id,
                        from_chat_id=message.chat.id,
                        message_id=message.message_id,
                    )
                else:
                    # Default: enriched copy mode
                    card_text, reply_markup = self.build_passenger_card(message, classification)
                    await self.bot.send_message(
                        chat_id=target_chat_id,
                        text=card_text,
                        reply_markup=reply_markup,
                        parse_mode="HTML",
                        disable_web_page_preview=True,
                    )

                # Record in user cooldown & deduplication cache
                self.user_cooldowns.record(user_id)
                self.deduplicator.record(user_id, raw_text)

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
                logger.error("Failed to forward message to VIP group %s: %s", target_chat_id, e)
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
    ) -> bool:
        """Forward a passenger request captured from any source (e.g. Userbot) to VIP group."""
        if self.settings.private_group_id == 0:
            logger.warning("PRIVATE_GROUP_ID is not configured (0). Message not forwarded.")
            return False

        # 1. Check 24-hour user cooldown (1 day window)
        if self.is_user_on_cooldown(user_id):
            remaining = self.get_user_remaining_cooldown_hours(user_id)
            logger.info(
                "⏭️ [24H COOLDOWN] User %s already forwarded within 1 day (%.1fh remaining). Skipping.",
                user_id,
                remaining,
            )
            return False

        # 2. Check deduplication
        if self.deduplicator.is_duplicate(user_id, raw_text):
            logger.info("Skipping duplicate passenger request from user %s", user_id)
            return False

        target_chat_id = self.settings.private_group_id
        max_retries = 2
        for attempt in range(max_retries):
            try:
                card_text, reply_markup = self.build_card_payload(
                    origin=classification.origin,
                    destination=classification.destination,
                    seats=classification.seats,
                    confidence=classification.confidence,
                    raw_text=raw_text,
                    passenger_name=passenger_name,
                    passenger_username=passenger_username,
                    chat_id=chat_id,
                    message_id=message_id,
                    chat_username=chat_username,
                )
                await self.bot.send_message(
                    chat_id=target_chat_id,
                    text=card_text,
                    reply_markup=reply_markup,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )
                # Record in user cooldown & deduplication cache
                self.user_cooldowns.record(user_id)
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
                logger.error("Failed to forward message to VIP group %s: %s", target_chat_id, e)
                return False
            except Exception as e:
                logger.exception("Unexpected error while forwarding message: %s", e)
                return False

        return False
