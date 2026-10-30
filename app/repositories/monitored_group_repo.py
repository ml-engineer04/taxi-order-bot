"""Repository for managing monitored public groups in database."""

import logging
from typing import List, Optional, Set
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models.monitored_group import MonitoredGroupRecord

logger = logging.getLogger(__name__)


class MonitoredGroupRepository:
    """Handles CRUD database operations for monitored groups."""

    @staticmethod
    def _get_chat_id_variants(chat_id: int) -> List[int]:
        s = str(chat_id)
        variants = [chat_id]
        if s.startswith("-100"):
            raw = s[4:]
            variants.extend([int(raw), -int(raw)])
        elif not s.startswith("-"):
            variants.extend([int(f"-100{s}"), -int(s)])
        elif s.startswith("-"):
            raw = s[1:]
            variants.extend([int(f"-100{raw}"), abs(chat_id)])
        return variants

    @classmethod
    async def add_or_activate(
        cls,
        session: AsyncSession,
        chat_id: int,
        title: Optional[str] = None,
        username: Optional[str] = None,
        invite_link: Optional[str] = None,
        added_by_admin_id: Optional[int] = None,
    ) -> MonitoredGroupRecord:
        """Add a new monitored group or reactivate an existing one."""
        variants = cls._get_chat_id_variants(chat_id)
        query = select(MonitoredGroupRecord).where(MonitoredGroupRecord.chat_id.in_(variants))
        result = await session.execute(query)
        record = result.scalars().first()

        if record:
            record.is_active = True
            if title:
                record.title = title
            if username:
                record.username = username
            if invite_link:
                record.invite_link = invite_link
            if added_by_admin_id:
                record.added_by_admin_id = added_by_admin_id
            await session.commit()
            await session.refresh(record)
            return record

        record = MonitoredGroupRecord(
            chat_id=chat_id,
            title=title or f"Guruh ({chat_id})",
            username=username,
            invite_link=invite_link,
            is_active=True,
            added_by_admin_id=added_by_admin_id,
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)
        return record

    @classmethod
    async def get_by_chat_id(cls, session: AsyncSession, chat_id: int) -> Optional[MonitoredGroupRecord]:
        """Fetch a monitored group record by chat ID (handling id variants)."""
        variants = cls._get_chat_id_variants(chat_id)
        query = select(MonitoredGroupRecord).where(MonitoredGroupRecord.chat_id.in_(variants))
        result = await session.execute(query)
        return result.scalars().first()

    @classmethod
    async def deactivate_group(cls, session: AsyncSession, chat_id: int) -> bool:
        """Mark a monitored group as inactive by chat ID."""
        variants = cls._get_chat_id_variants(chat_id)
        query = select(MonitoredGroupRecord).where(MonitoredGroupRecord.chat_id.in_(variants))
        result = await session.execute(query)
        records = result.scalars().all()

        if not records:
            return False

        for rec in records:
            rec.is_active = False

        await session.commit()
        return True

    @staticmethod
    async def get_all_active(session: AsyncSession) -> List[MonitoredGroupRecord]:
        """Fetch all active monitored groups ordered by creation time."""
        query = select(MonitoredGroupRecord).where(MonitoredGroupRecord.is_active == True).order_by(MonitoredGroupRecord.id.asc())
        result = await session.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def get_active_chat_ids(session: AsyncSession) -> Set[int]:
        """Fetch set of all active chat IDs."""
        query = select(MonitoredGroupRecord.chat_id).where(MonitoredGroupRecord.is_active == True)
        result = await session.execute(query)
        return set(result.scalars().all())
