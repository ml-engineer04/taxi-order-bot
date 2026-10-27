"""Common and Error Handlers Router."""

import logging
from aiogram import Router
from aiogram.types import ErrorEvent

logger = logging.getLogger(__name__)
common_router = Router(name="common_router")


@common_router.error()
async def error_handler(event: ErrorEvent) -> None:
    """Global error handler to catch and log exceptions gracefully."""
    logger.error(
        "Exception occurred while handling update %s: %s",
        event.update.update_id if event.update else "Unknown",
        event.exception,
        exc_info=True,
    )
