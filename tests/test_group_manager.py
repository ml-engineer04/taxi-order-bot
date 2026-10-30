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
    test_chat_id = -999988885555

    async with get_db_session() as session:
        # Add new group
        record = await MonitoredGroupRepository.add_or_activate(
            session=session,
            chat_id=test_chat_id,
            title="Test Taksi Guruhi",
            username="test_taksi",
            added_by_admin_id=8120241913,
        )
        assert record.id is not None
        assert record.chat_id == test_chat_id
        assert record.is_active is True

        # Check active list
        active_list = await MonitoredGroupRepository.get_all_active(session)
        assert any(g.chat_id == test_chat_id for g in active_list)

        # Deactivate
        deactivated = await MonitoredGroupRepository.deactivate_group(session, test_chat_id)
        assert deactivated is True

        # Verify not in active list
        active_list_after = await MonitoredGroupRepository.get_all_active(session)
        assert not any(g.chat_id == test_chat_id for g in active_list_after)

        # Cleanup test row
        await session.delete(record)
        await session.commit()


@pytest.mark.asyncio
async def test_group_manager_dynamic_add_and_remove():
    await init_db()
    test_id = -999988887777
    settings = Settings(
        BOT_TOKEN="123:abc",
        PUBLIC_GROUP_ID="-999988889999",
    )
    manager = GroupManager(settings=settings)
    await manager.initialize()

    # Initial group from settings should be monitored
    assert manager.is_monitored(-999988889999) is True

    # Unknown group should NOT be monitored
    assert manager.is_monitored(test_id) is False

    # Admin dynamically adds test_id
    success, msg, added_id = await manager.add_group(str(test_id), admin_id=8120241913)
    assert success is True
    assert added_id == test_id
    assert "Guruh muvaffaqiyatli qo'shildi" in msg

    # Now test_id is monitored immediately
    assert manager.is_monitored(test_id) is True

    # Admin dynamically removes test_id
    del_success, del_msg = await manager.remove_group(str(test_id))
    assert del_success is True
    assert "olib tashlandi" in del_msg

    # Now it is no longer monitored
    assert manager.is_monitored(test_id) is False

    # Cleanup DB row
    async with get_db_session() as session:
        rec = await MonitoredGroupRepository.get_by_chat_id(session, test_id)
        if rec:
            await session.delete(rec)
        rec_init = await MonitoredGroupRepository.get_by_chat_id(session, -999988889999)
        if rec_init:
            await session.delete(rec_init)
        await session.commit()


@pytest.mark.asyncio
async def test_admin_handlers_add_and_del_group():
    from app.handlers.admin import handle_add_group, handle_remove_group, handle_list_groups

    test_del_id = -999988886666
    mock_bot = AsyncMock()
    mock_user = MagicMock()
    mock_user.id = 8120241913  # admin id

    # 1. Test /addgroup
    msg_add = MagicMock()
    msg_add.from_user = mock_user
    msg_add.text = f"/addgroup {test_del_id}"
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
    assert str(test_del_id) in msg_list.answer.call_args[0][0]

    # 3. Test /delgroup
    msg_del = MagicMock()
    msg_del.from_user = mock_user
    msg_del.text = f"/delgroup {test_del_id}"
    msg_del.answer = AsyncMock()
    await handle_remove_group(msg_del)
    msg_del.answer.assert_called_once()
    assert "olib tashlandi" in msg_del.answer.call_args[0][0]

    # Cleanup DB row
    async with get_db_session() as session:
        rec = await MonitoredGroupRepository.get_by_chat_id(session, test_del_id)
        if rec:
            await session.delete(rec)
            await session.commit()
