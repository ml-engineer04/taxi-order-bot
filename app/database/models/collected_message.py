"""Collected / Ambiguous Message Database Model for Active Learning."""

from datetime import datetime, timezone
import hashlib
from typing import Optional
from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base
from app.services.normalization.transliterator import normalize_uzbek_text


class CollectedMessageRecord(Base):
    """Stores incoming messages that require human-in-the-loop verification."""

    __tablename__ = "collected_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    text_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)

    predicted_label: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    actual_label: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), default="pending", index=True, nullable=False
    )  # 'pending', 'labeled', 'skipped'

    collected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    @staticmethod
    def compute_hash(raw_text: str) -> str:
        """Compute deterministic SHA-256 hash of normalized text."""
        norm = normalize_uzbek_text(raw_text)
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()
