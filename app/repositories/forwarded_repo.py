"""Repository for persisting and querying forwarded message audit records."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models.forwarded_message import ForwardedMessageRecord

logger = logging.getLogger(__name__)


class ForwardedMessageRepository:
    """Handles database transactions for forwarded passenger requests."""

    @staticmethod
    async def save_record(
        session: AsyncSession,
        public_message_id: int,
        public_chat_id: int,
        author_telegram_id: int,
        author_username: Optional[str],
        author_full_name: Optional[str],
        origin: Optional[str],
        destination: Optional[str],
        seats: Optional[int],
        confidence: float,
        raw_text: str,
    ) -> ForwardedMessageRecord:
        """Persist a newly forwarded message record."""
        record = ForwardedMessageRecord(
            public_message_id=public_message_id,
            public_chat_id=public_chat_id,
            author_telegram_id=author_telegram_id,
            author_username=author_username,
            author_full_name=author_full_name,
            origin=origin,
            destination=destination,
            seats=seats,
            confidence=confidence,
            raw_text=raw_text,
            forwarded_at=datetime.now(timezone.utc),
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)
        return record

    @staticmethod
    async def get_recent_user_cooldowns(
        session: AsyncSession,
        hours: int = 24,
    ) -> Dict[int, float]:
        """
        Fetch all user IDs forwarded within the last N hours and their forward timestamps.
        Returns mapping: user_id -> timestamp (float).
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        query = (
            select(
                ForwardedMessageRecord.author_telegram_id,
                ForwardedMessageRecord.forwarded_at,
            )
            .where(ForwardedMessageRecord.forwarded_at >= cutoff)
            .order_by(ForwardedMessageRecord.forwarded_at.asc())
        )
        result = await session.execute(query)
        rows = result.all()
        cooldowns: Dict[int, float] = {}
        for uid, f_at in rows:
            if f_at.tzinfo is None:
                f_at = f_at.replace(tzinfo=timezone.utc)
            cooldowns[uid] = f_at.timestamp()
        return cooldowns
