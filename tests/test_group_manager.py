"""Unit tests for MonitoredGroupRepository, GroupManager, and dynamic group handlers."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.config import Settings
from app.database.session import init_db, get_db_session
from app.database.models.monitored_group import MonitoredGroupRecord
from app.repositories.monitored_group_repo import MonitoredGroupRepository
from app.services.group_manager import GroupManager


@pytest.mark.asyncio
async def test_monitored_group_repo_crud():
    await init_db()

    async with get_db_session() as session:
        # Add new group
        record = await MonitoredGroupRepository.add_or_activate(
            session=session,
            chat_id=-1005555555555,
            title="Test Taksi Guruhi",
            username="test_taksi",
            added_by_admin_id=8120241913,
        )
        assert record.id is not None
        assert record.chat_id == -1005555555555
        assert record.is_active is True

        # Check active list
        active_list = await MonitoredGroupRepository.get_all_active(session)
        assert any(g.chat_id == -1005555555555 for g in active_list)

        # Deactivate
        deactivated = await MonitoredGroupRepository.deactivate_group(session, -1005555555555)
        assert deactivated is True

        # Verify not in active list
        active_list_after = await MonitoredGroupRepository.get_all_active(session)
        assert not any(g.chat_id == -1005555555555 for g in active_list_after)


@pytest.mark.asyncio
async def test_group_manager_dynamic_add_and_remove():
    await init_db()
    settings = Settings(
        BOT_TOKEN="123:abc",
        PUBLIC_GROUP_ID="-1003726192046",
    )
    manager = GroupManager(settings=settings)
    await manager.initialize()

    # Initial group from settings should be monitored
    assert manager.is_monitored(-1003726192046) is True
    assert manager.is_monitored(3726192046) is True

    # Unknown group should NOT be monitored
    assert manager.is_monitored(-5353349446) is False

    # Admin dynamically adds -5353349446
    success, msg, added_id = await manager.add_group("-5353349446", admin_id=8120241913)
    assert success is True
    assert added_id == -5353349446
    assert "Guruh muvaffaqiyatli qo'shildi" in msg

    # Now -5353349446 (and its supergroup variant -1005353349446) is monitored immediately
    assert manager.is_monitored(-5353349446) is True
    assert manager.is_monitored(-1005353349446) is True

    # Admin dynamically removes -5353349446
    del_success, del_msg = await manager.remove_group("-5353349446")
    assert del_success is True
    assert "olib tashlandi" in del_msg

    # Now it is no longer monitored
    assert manager.is_monitored(-5353349446) is False
    assert manager.is_monitored(-1005353349446) is False


@pytest.mark.asyncio
async def test_admin_handlers_add_and_del_group():
    from app.handlers.admin import handle_add_group, handle_remove_group, handle_list_groups

    mock_bot = AsyncMock()
    mock_user = MagicMock()
    mock_user.id = 8120241913  # admin id

    # 1. Test /addgroup -777888999
    msg_add = MagicMock()
    msg_add.from_user = mock_user
    msg_add.text = "/addgroup -777888999"
    mock_status_msg = AsyncMock()
    msg_add.answer = AsyncMock(return_value=mock_status_msg)

    await handle_add_group(msg_add)
    mock_status_msg.edit_text.assert_called_once()
    assert "qo'shildi" in mock_status_msg.edit_text.call_args[0][0]

    # 2. Test /groups
    msg_list = MagicMock()
    msg_list.from_user = mock_user
    msg_list.answer = AsyncMock()
    await handle_list_groups(msg_list)
    msg_list.answer.assert_called_once()
    assert "-777888999" in msg_list.answer.call_args[0][0]

    # 3. Test /delgroup -777888999
    msg_del = MagicMock()
    msg_del.from_user = mock_user
    msg_del.text = "/delgroup -777888999"
    msg_del.answer = AsyncMock()
    await handle_remove_group(msg_del)
    msg_del.answer.assert_called_once()
    assert "olib tashlandi" in msg_del.answer.call_args[0][0]
