"""Dynamic Monitored Group Management Service."""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple
from aiogram import Bot
from telethon import TelegramClient
from telethon.tl.functions.channels import JoinChannelRequest
from telethon.tl.functions.messages import ImportChatInviteRequest, CheckChatInviteRequest
from telethon.errors import (
    UserAlreadyParticipantError,
    InviteHashExpiredError,
    InviteHashInvalidError,
    ChannelPrivateError,
    FloodWaitError,
)

from app.config import Settings
from app.database.session import get_db_session
from app.database.models.monitored_group import MonitoredGroupRecord
from app.repositories.monitored_group_repo import MonitoredGroupRepository

logger = logging.getLogger(__name__)


class GroupManager:
    """Manages real-time addition, removal, and filtering of monitored groups."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.bot: Optional[Bot] = None
        self.userbot: Optional[TelegramClient] = None
        self._active_ids: Set[int] = set()

    def set_clients(self, bot: Optional[Bot] = None, userbot: Optional[TelegramClient] = None) -> None:
        """Attach bot and userbot clients for dynamic API resolutions."""
        if bot:
            self.bot = bot
        if userbot:
            self.userbot = userbot

    def _expand_variants(self, chat_id: int) -> Set[int]:
        s = str(chat_id)
        variants = {chat_id}
        if s.startswith("-100"):
            raw = s[4:]
            variants.add(int(raw))
            variants.add(-int(raw))
        elif not s.startswith("-"):
            variants.add(int(f"-100{s}"))
            variants.add(-int(s))
        elif s.startswith("-"):
            raw = s[1:]
            variants.add(int(f"-100{raw}"))
            variants.add(abs(chat_id))
        return variants

    async def initialize(self) -> None:
        """Seed initial static groups from settings into DB and load all active chat IDs into memory."""
        async with get_db_session() as session:
            # 1. Sync any configured groups from .env into DB if not present
            static_targets = self.settings.get_public_target_strings()
            for raw_target in static_targets:
                target = str(raw_target).strip()
                if not target or target == "0":
                    continue
                if re.match(r"^-?\d+$", target):
                    num_id = int(target)
                    await MonitoredGroupRepository.add_or_activate(
                        session=session,
                        chat_id=num_id,
                        title=f"Boshlang'ich guruh ({num_id})",
                    )
                elif "+" in target or "joinchat" in target:
                    pass
                else:
                    pass

            # 2. Load all active groups from database
            active_groups = await MonitoredGroupRepository.get_all_active(session)
            self._active_ids.clear()
            for g in active_groups:
                self._active_ids.update(self._expand_variants(g.chat_id))

            # Also add static numeric IDs as fallback
            for n_id in self.settings.get_public_group_numeric_ids():
                self._active_ids.update(self._expand_variants(n_id))

        logger.info("GroupManager initialized with %d active monitored chat ID variant(s).", len(self._active_ids))

    def is_monitored(self, chat_id: int) -> bool:
        """Fast O(1) check if chat_id matches any active monitored group."""
        if not self._active_ids and not self.settings.public_group_id:
            return True

        if chat_id in self._active_ids:
            return True

        variants = self._expand_variants(chat_id)
        return bool(variants & self._active_ids)

    async def add_group(
        self,
        target: str,
        admin_id: Optional[int] = None,
    ) -> Tuple[bool, str, Optional[int]]:
        """
        Dynamically add a group by numeric ID, username (@group), or invite link.
        Returns (success: bool, message: str, chat_id: Optional[int]).
        """
        raw_target = str(target).strip()
        if not raw_target:
            return False, "❌ Guruh ID raqami, username yoki havola kiritilmadi.", None

        chat_id: Optional[int] = None
        title: Optional[str] = None
        username: Optional[str] = None
        invite_link: Optional[str] = None

        # 1. Direct Numeric ID: e.g. -5353349446 or -1003726192046
        if re.match(r"^-?\d+$", raw_target):
            chat_id = int(raw_target)
            title = f"Guruh ({chat_id})"

            # Try to fetch actual group title via Bot API or Userbot
            if self.bot:
                try:
                    c = await self.bot.get_chat(chat_id)
                    title = c.title or title
                    username = c.username
                except Exception:
                    pass

            if self.userbot and (not title or title.startswith("Guruh (")):
                try:
                    entity = await self.userbot.get_entity(chat_id)
                    title = getattr(entity, "title", title)
                    username = getattr(entity, "username", username)
                except Exception:
                    pass

        # 2. Private Invite Link: e.g. https://t.me/+AbCdEf or t.me/joinchat/AbCdEf
        elif "+" in raw_target or "joinchat" in raw_target:
            invite_link = raw_target
            hash_match = re.search(r"(\+|joinchat/)([a-zA-Z0-9_-]+)", raw_target)
            if not hash_match:
                return False, "❌ Noto'g'ri taklif havolasi formati.", None
            invite_hash = hash_match.group(2)

            if not self.userbot:
                return False, "❌ Taklif havolasi orqali qo'shish uchun Userbot yoqilgan bo'lishi kerak.", None

            try:
                res = await self.userbot(ImportChatInviteRequest(invite_hash))
                chat = res.chats[0]
                chat_id = chat.id
                title = getattr(chat, "title", f"Guruh ({chat_id})")
                username = getattr(chat, "username", None)
            except UserAlreadyParticipantError:
                try:
                    check = await self.userbot(CheckChatInviteRequest(invite_hash))
                    if hasattr(check, "chat"):
                        chat_id = check.chat.id
                        title = getattr(check.chat, "title", f"Guruh ({chat_id})")
                        username = getattr(check.chat, "username", None)
                    else:
                        return False, "❌ Guruh allaqachon qo'shilgan, lekin ID aniqlanmadi.", None
                except Exception as e:
                    return False, f"❌ Guruhni tekshirishda xatolik: {e}", None
            except Exception as e:
                return False, f"❌ Havola orqali guruhga ulanib bo'lmadi: {e}", None

        # 3. Public Username or Link: e.g. @toshkent_taxi or https://t.me/toshkent_taxi
        else:
            clean_name = re.sub(r"^(https?://)?(t\.me/|telegram\.me/)", "", raw_target).lstrip("@").strip("/")
            username = clean_name

            # Resolve using userbot first (which can also join the group)
            if self.userbot:
                try:
                    entity = await self.userbot.get_entity(clean_name)
                    chat_id = entity.id
                    title = getattr(entity, "title", f"@{clean_name}")
                    try:
                        await self.userbot(JoinChannelRequest(entity))
                    except UserAlreadyParticipantError:
                        pass
                except Exception as e:
                    logger.debug("Userbot could not join @%s: %s", clean_name, e)

            # Fallback resolve using bot
            if not chat_id and self.bot:
                try:
                    c = await self.bot.get_chat(f"@{clean_name}")
                    chat_id = c.id
                    title = c.title or f"@{clean_name}"
                except Exception as e:
                    return False, f"❌ @{clean_name} guruhini topib bo'lmadi: {e}", None

            if not chat_id:
                return False, f"❌ @{clean_name} guruhini aniqlab bo'lmadi.", None

        # Normalize supergroup ID to -100 format if positive integer
        if chat_id > 0:
            chat_id = int(f"-100{chat_id}")

        # Save to database
        async with get_db_session() as session:
            record = await MonitoredGroupRepository.add_or_activate(
                session=session,
                chat_id=chat_id,
                title=title,
                username=username,
                invite_link=invite_link,
                added_by_admin_id=admin_id,
            )

        # Update in-memory set immediately
        self._active_ids.update(self._expand_variants(chat_id))

        display_name = title or f"ID: {chat_id}"
        msg = (
            f"✅ <b>Guruh muvaffaqiyatli qo'shildi va kuzatuvga olindi!</b>\n\n"
            f"🏷 <b>Nomi:</b> {display_name}\n"
            f"🆔 <b>Chat ID:</b> <code>{chat_id}</code>\n"
            + (f"🏷 <b>Username:</b> @{username}\n" if username else "")
            + "<i>Endi ushbu guruhdagi barcha yangi yo'lovchi xabarlari avtomatik tahlil qilinadi.</i>"
        )
        return True, msg, chat_id

    async def remove_group(self, target: str) -> Tuple[bool, str]:
        """Dynamically remove/deactivate a monitored group by ID."""
        raw_target = str(target).strip()
        if not raw_target or not re.match(r"^-?\d+$", raw_target):
            return False, "❌ Guruhni o'chirish uchun to'g'ri raqamli ID kiriting (masalan: <code>/delgroup -1003726192046</code>)."

        chat_id = int(raw_target)

        async with get_db_session() as session:
            success = await MonitoredGroupRepository.deactivate_group(session, chat_id)

        # Remove from in-memory set
        variants = self._expand_variants(chat_id)
        self._active_ids.difference_update(variants)

        if success:
            return True, f"🗑 <b>Guruh kuzatuvdan olib tashlandi!</b>\n\n🆔 <b>Chat ID:</b> <code>{chat_id}</code>\n<i>Ushbu guruhdagi xabarlar endi o'qilmaydi.</i>"
        return False, f"⚠️ Ushbu ID (<code>{chat_id}</code>) bo'yicha faol guruh topilmadi."

    async def get_all_groups(self) -> List[MonitoredGroupRecord]:
        """Fetch all active monitored groups from database."""
        async with get_db_session() as session:
            return await MonitoredGroupRepository.get_all_active(session)


# Global singleton instance
_group_manager: Optional[GroupManager] = None


def get_group_manager(settings: Optional[Settings] = None) -> GroupManager:
    """Retrieve or create singleton GroupManager instance."""
    global _group_manager
    if _group_manager is None:
        from app.config import get_settings
        _group_manager = GroupManager(settings=settings or get_settings())
    return _group_manager
