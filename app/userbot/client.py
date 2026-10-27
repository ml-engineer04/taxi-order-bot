"""Telethon Userbot client factory and lifecycle management."""

import os
import logging
from typing import Optional
from telethon import TelegramClient

from app.config import Settings

logger = logging.getLogger(__name__)


def create_userbot_client(settings: Settings) -> Optional[TelegramClient]:
    """
    Instantiate Telethon TelegramClient using configuration parameters.
    Returns None if TG_API_ID or TG_API_HASH is not set.
    """
    if not settings.tg_api_id or not settings.tg_api_hash:
        logger.warning("TG_API_ID or TG_API_HASH is missing; cannot initialize Telethon Userbot.")
        return None

    session_path = settings.userbot_session_name
    dirname = os.path.dirname(session_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    client = TelegramClient(
        session=session_path,
        api_id=settings.tg_api_id,
        api_hash=settings.tg_api_hash,
    )
    return client
