"""Automated Tests for Active Learning Review and Auto-Data Collector."""

import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.database.base import Base
from app.database.models.collected_message import CollectedMessageRecord
from app.repositories.collected_repo import CollectedMessageRepository
from app.services.active_learning import (
    append_example_to_dataset,
    build_review_card,
)


@pytest_asyncio.fixture
async def async_test_session():
    """Create in-memory SQLite database session for testing."""
    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async_session = async_sessionmaker(test_engine, expire_on_commit=False, class_=AsyncSession)

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as session:
        yield session

    await test_engine.dispose()


@pytest.mark.asyncio
async def test_collected_repo_save_and_deduplicate(async_test_session):
    """Verify that messages are stored with pending status and duplicates are prevented."""
    text = "Toshkentdan Samarqandga mashina bormi"

    # First save
    rec1 = await CollectedMessageRepository.save_if_new(
        async_test_session, text=text, predicted_label="OTHER", confidence=0.45
    )
    assert rec1 is not None
    assert rec1.status == "pending"
    assert rec1.predicted_label == "OTHER"

    # Second save with exact same text should return None (deduplicated)
    rec2 = await CollectedMessageRepository.save_if_new(
        async_test_session, text=text, predicted_label="OTHER", confidence=0.50
    )
    assert rec2 is None

    # Count check
    count = await CollectedMessageRepository.get_pending_count(async_test_session)
    assert count == 1


@pytest.mark.asyncio
async def test_collected_repo_label_and_skip(async_test_session):
    """Verify that pending messages can be labeled or skipped."""
    rec = await CollectedMessageRepository.save_if_new(
        async_test_session, text="Buxoroga 2 kishi bor", predicted_label="OTHER", confidence=0.40
    )
    assert rec is not None

    # Retrieve next pending
    pending = await CollectedMessageRepository.get_next_pending(async_test_session)
    assert pending.id == rec.id

    # Label message
    labeled = await CollectedMessageRepository.mark_labeled(
        async_test_session, message_id=rec.id, actual_label="PASSENGER_REQUEST"
    )
    assert labeled.status == "labeled"
    assert labeled.actual_label == "PASSENGER_REQUEST"
    assert labeled.reviewed_at is not None

    # Pending count should now be 0
    remaining = await CollectedMessageRepository.get_pending_count(async_test_session)
    assert remaining == 0


def test_append_example_to_dataset(tmp_path):
    """Verify appending a new verified sample to the CSV dataset."""
    csv_file = tmp_path / "test_dataset.csv"

    # Append passenger
    ok1 = append_example_to_dataset("Toshkentga 3 ta odam bor", "PASSENGER_REQUEST", str(csv_file))
    assert ok1 is True

    # Append driver
    ok2 = append_example_to_dataset("Samarqandga odam olaman", "DRIVER_OFFER", str(csv_file))
    assert ok2 is True

    content = csv_file.read_text(encoding="utf-8")
    assert "Toshkentga 3 ta odam bor,PASSENGER_REQUEST,latin" in content
    assert "Samarqandga odam olaman,DRIVER_OFFER,latin" in content


def test_build_review_card():
    """Verify formatting of the review card and inline keyboard buttons."""
    record = CollectedMessageRecord(
        id=42,
        text="Toshkentdan Samarqandga odam bor",
        text_hash="dummyhash",
        predicted_label="PASSENGER_REQUEST",
        confidence=0.55,
        status="pending",
    )

    card_text, markup = build_review_card(record, remaining_count=5)
    assert "Yangi xabarni tasdiqlash" in card_text
    assert "Toshkentdan Samarqandga odam bor" in card_text
    assert "55%" in card_text

    # Verify callback buttons
    buttons = [btn.callback_data for row in markup.inline_keyboard for btn in row]
    assert "rev:label:PASSENGER_REQUEST:42" in buttons
    assert "rev:label:DRIVER_OFFER:42" in buttons
    assert "rev:label:OTHER:42" in buttons
    assert "rev:skip:42" in buttons
    assert "rev:stop" in buttons
