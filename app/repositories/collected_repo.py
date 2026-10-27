"""Repository for Collected / Review Messages."""

from datetime import datetime, timezone
import logging
from typing import Optional
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models.collected_message import CollectedMessageRecord

logger = logging.getLogger(__name__)


class CollectedMessageRepository:
    """Data access layer for active learning messages."""

    @staticmethod
    async def save_if_new(
        session: AsyncSession,
        text: str,
        predicted_label: str,
        confidence: float,
    ) -> Optional[CollectedMessageRecord]:
        """Save message if it is not already in the collection database."""
        text_clean = text.strip()
        if not text_clean:
            return None

        text_hash = CollectedMessageRecord.compute_hash(text_clean)

        stmt = select(CollectedMessageRecord).where(
            CollectedMessageRecord.text_hash == text_hash
        )
        res = await session.execute(stmt)
        existing = res.scalar_one_or_none()
        if existing:
            return None

        record = CollectedMessageRecord(
            text=text_clean,
            text_hash=text_hash,
            predicted_label=predicted_label,
            confidence=round(confidence, 4),
            status="pending",
        )
        session.add(record)
        await session.flush()
        return record

    @staticmethod
    async def get_next_pending(session: AsyncSession) -> Optional[CollectedMessageRecord]:
        """Retrieve the oldest pending review message."""
        stmt = (
            select(CollectedMessageRecord)
            .where(CollectedMessageRecord.status == "pending")
            .order_by(CollectedMessageRecord.id.asc())
            .limit(1)
        )
        res = await session.execute(stmt)
        return res.scalar_one_or_none()

    @staticmethod
    async def get_pending_count(session: AsyncSession) -> int:
        """Count total pending review messages."""
        stmt = (
            select(func.count())
            .select_from(CollectedMessageRecord)
            .where(CollectedMessageRecord.status == "pending")
        )
        res = await session.execute(stmt)
        return res.scalar() or 0

    @staticmethod
    async def mark_labeled(
        session: AsyncSession,
        message_id: int,
        actual_label: str,
    ) -> Optional[CollectedMessageRecord]:
        """Mark message as labeled with the human-verified category."""
        stmt = select(CollectedMessageRecord).where(CollectedMessageRecord.id == message_id)
        res = await session.execute(stmt)
        record = res.scalar_one_or_none()
        if record:
            record.actual_label = actual_label
            record.status = "labeled"
            record.reviewed_at = datetime.now(timezone.utc)
            await session.flush()
        return record

    @staticmethod
    async def mark_skipped(
        session: AsyncSession,
        message_id: int,
    ) -> Optional[CollectedMessageRecord]:
        """Mark message as skipped."""
        stmt = select(CollectedMessageRecord).where(CollectedMessageRecord.id == message_id)
        res = await session.execute(stmt)
        record = res.scalar_one_or_none()
        if record:
            record.status = "skipped"
            record.reviewed_at = datetime.now(timezone.utc)
            await session.flush()
        return record
