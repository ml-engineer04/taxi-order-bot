"""Driver Subscription Database Model."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from sqlalchemy import BigInteger, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, TimestampMixin


def ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensure datetime is timezone-aware in UTC (fixes SQLite naive datetime returns)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class SubscriptionStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    EXPIRED = "expired"
    BANNED = "banned"


class Driver(Base, TimestampMixin):
    """Driver subscription tracking model."""

    __tablename__ = "drivers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    telegram_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    username: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    phone_number: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    status: Mapped[str] = mapped_column(
        String(32),
        default=SubscriptionStatus.PENDING.value,
        nullable=False,
        index=True,
    )

    start_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    end_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, index=True)

    approved_by_admin_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    invite_link: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def is_currently_active(self, now: Optional[datetime] = None) -> bool:
        """Check if subscription is currently valid according to end_date."""
        if self.status != SubscriptionStatus.ACTIVE.value:
            return False
        if not self.end_date:
            return False
        now_dt = ensure_utc(now) or datetime.now(timezone.utc)
        return ensure_utc(self.end_date) > now_dt

    def days_remaining(self, now: Optional[datetime] = None) -> int:
        """Calculate days left until expiration."""
        if not self.end_date:
            return 0
        now_dt = ensure_utc(now) or datetime.now(timezone.utc)
        diff = ensure_utc(self.end_date) - now_dt
        return max(0, diff.days)

    def __repr__(self) -> str:
        return f"<Driver id={self.id} tg_id={self.telegram_id} status={self.status} end_date={self.end_date}>"
