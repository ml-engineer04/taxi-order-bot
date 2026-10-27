"""Forwarded Message Audit Database Model."""

from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import BigInteger, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class ForwardedMessageRecord(Base):
    """Audit trail of messages classified and dispatched to VIP group."""

    __tablename__ = "forwarded_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    public_message_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    public_chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False)

    author_telegram_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    author_username: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    author_full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    origin: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    destination: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    seats: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)

    forwarded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
