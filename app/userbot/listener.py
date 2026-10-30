"""Telethon Userbot message listener, auto-joiner, and live forwarding pipeline."""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional, Set
from telethon import TelegramClient, events
from telethon.tl.functions.channels import JoinChannelRequest
from telethon.tl.functions.messages import ImportChatInviteRequest, CheckChatInviteRequest
from telethon.errors import (
    UserAlreadyParticipantError,
    InviteHashExpiredError,
    InviteHashInvalidError,
    ChannelPrivateError,
    FloodWaitError,
)
from aiogram import Bot

from app.config import Settings
from app.database.session import get_db_session
from app.repositories.collected_repo import CollectedMessageRepository
from app.services.classification.base import BaseClassifier
from app.services.forwarding import ForwardingService, clean_display_name

logger = logging.getLogger(__name__)


def is_chat_monitored(chat_id: int, settings: Settings, extra_monitored_ids: Optional[Set[int]] = None) -> bool:
    """Check if chat_id matches monitored group IDs, tolerating -100 prefix variances."""
    # Strictly ignore private VIP group itself to avoid forwarding loops
    if chat_id == settings.private_group_id or str(chat_id) == str(settings.private_group_id):
        return False

    monitored_set: Set[int] = set()

    if extra_monitored_ids:
        monitored_set.update(extra_monitored_ids)

    numeric_ids = settings.get_public_group_numeric_ids()
    if numeric_ids:
        monitored_set.update(numeric_ids)

    from app.services.group_manager import get_group_manager
    group_mgr = get_group_manager(settings)
    if group_mgr._active_ids:
        monitored_set.update(group_mgr._active_ids)

    # When NO groups are configured anywhere (neither in DB, settings, nor extra IDs),
    # userbot monitors all groups it has joined (bootstrap mode)
    if not monitored_set:
        return True

    if chat_id in monitored_set:
        return True

    s_id = str(chat_id)
    variants = [chat_id]
    if s_id.startswith("-100"):
        raw_num = s_id[4:]
        variants.extend([int(raw_num), -int(raw_num)])
    elif not s_id.startswith("-"):
        variants.extend([int(f"-100{s_id}"), -int(s_id)])
    elif s_id.startswith("-"):
        raw_num = s_id[1:]
        variants.extend([int(f"-100{raw_num}"), abs(chat_id)])

    return any(v in monitored_set for v in variants)


async def auto_join_and_resolve_targets(client: TelegramClient, settings: Settings) -> Set[int]:
    """
    Automatically join configured groups by invite link or username on startup,
    and resolve all monitored chat IDs into a normalized set.
    """
    targets = settings.get_public_target_strings()
    resolved_ids: Set[int] = set()

    if not targets:
        logger.info("No specific public target groups configured; Userbot will monitor all joined groups.")
        return resolved_ids

    logger.info("Initializing Userbot group membership for %d target(s)...", len(targets))

    for raw_target in targets:
        target = str(raw_target).strip()
        if not target or target == "0":
            continue

        # 1. Direct Numeric Group ID: e.g. -1003726192046 or -5353349446
        if re.match(r"^-?\d+$", target):
            num_id = int(target)
            resolved_ids.add(num_id)
            s = str(num_id)
            if s.startswith("-100"):
                resolved_ids.add(int(s[4:]))
                resolved_ids.add(-int(s[4:]))
            elif s.startswith("-"):
                raw_num = s[1:]
                resolved_ids.add(int(f"-100{raw_num}"))
                resolved_ids.add(abs(num_id))
            else:
                resolved_ids.add(int(f"-100{s}"))
                resolved_ids.add(-num_id)
            logger.info("Userbot monitoring numeric chat ID: %s (and normalized variants)", num_id)
            continue

        # 2. Private Invite Link: e.g. https://t.me/+AbCdEf or t.me/joinchat/AbCdEf
        if "+" in target or "joinchat" in target:
            hash_match = re.search(r"(\+|joinchat/)([a-zA-Z0-9_-]+)", target)
            if hash_match:
                invite_hash = hash_match.group(2)
                try:
                    res = await client(ImportChatInviteRequest(invite_hash))
                    chat = res.chats[0]
                    c_id = chat.id
                    resolved_ids.add(c_id)
                    resolved_ids.add(int(f"-100{abs(c_id)}"))
                    logger.info(
                        "✅ Userbot guruhga muvaffaqiyatli a'zo bo'ldi (Invite link): '%s' (ID: %s)",
                        getattr(chat, "title", ""),
                        c_id,
                    )
                except UserAlreadyParticipantError:
                    try:
                        check = await client(CheckChatInviteRequest(invite_hash))
                        if hasattr(check, "chat"):
                            c_id = check.chat.id
                            resolved_ids.add(c_id)
                            resolved_ids.add(int(f"-100{abs(c_id)}"))
                            logger.info(
                                "Userbot allaqachon guruh a'zosi: '%s' (ID: %s)",
                                getattr(check.chat, "title", ""),
                                c_id,
                            )
                    except Exception as err:
                        logger.debug("Could not verify invite link chat entity: %s", err)
                except (InviteHashExpiredError, InviteHashInvalidError) as err:
                    logger.warning("Taklif havolasi eskirgan yoki noto'g'ri ('%s'): %s", target, err)
                except FloodWaitError as err:
                    logger.warning("Telegram flood wait on join: %s seconds", err.seconds)
                except Exception as exc:
                    logger.warning("Guruhga ulanishda xatolik ('%s'): %s", target, exc)
            continue

        # 3. Public Username or Link: e.g. @viloyat_taksi or https://t.me/viloyat_taksi
        username = re.sub(r"^(https?://)?(t\.me/|telegram\.me/)", "", target).lstrip("@").strip("/")
        if username:
            try:
                entity = await client.get_entity(username)
                try:
                    await client(JoinChannelRequest(entity))
                    logger.info("✅ Userbot ommaviy guruhga qo'shildi: @%s (ID: %s)", username, entity.id)
                except UserAlreadyParticipantError:
                    logger.info("Userbot allaqachon @%s guruhida a'zo (ID: %s)", username, entity.id)
                except Exception as err:
                    logger.warning("Could not join channel @%s: %s", username, err)

                resolved_ids.add(entity.id)
                resolved_ids.add(int(f"-100{abs(entity.id)}"))
            except ChannelPrivateError:
                logger.warning("Guruh yopiq (private) va unga kirish uchun maxsus havola kerak: @%s", username)
            except Exception as exc:
                logger.warning("Guruh nomini aniqlab bo'lmadi ('%s'): %s", username, exc)

    logger.info("Userbot actively monitoring %d resolved group ID(s).", len(resolved_ids))
    return resolved_ids


def setup_userbot_listener(
    client: TelegramClient,
    bot: Bot,
    settings: Settings,
    classifier: BaseClassifier,
    forwarder: ForwardingService,
    monitored_ids: Optional[Set[int]] = None,
    voice_service: Optional[Any] = None,
) -> None:
    """Register NewMessage event handler on the Telethon client."""
    listener_started_at = datetime.now(timezone.utc)
    active_monitored_ids = monitored_ids or set()

    if voice_service is None:
        from app.services.voice_service import VoiceTranscriptionService
        groq_k = settings.groq_api_key.get_secret_value() if settings.groq_api_key else None
        openai_k = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        voice_service = VoiceTranscriptionService(
            groq_api_key=groq_k,
            openai_api_key=openai_k,
            enabled=settings.voice_transcription_enabled,
        )

    logger.info(
        "Userbot message listener registered. Strictly ignoring historical backlog (started at: %s).",
        listener_started_at.isoformat(),
    )

    @client.on(events.NewMessage)
    async def handle_new_userbot_message(event: events.NewMessage.Event) -> None:
        # 1. Strictly ignore historical messages sent before userbot started
        msg_date = getattr(getattr(event, "message", None), "date", None)
        if isinstance(msg_date, datetime):
            if msg_date.tzinfo is None:
                msg_date = msg_date.replace(tzinfo=timezone.utc)
            if msg_date < listener_started_at:
                logger.debug(
                    "Skipping historical message %s (sent at %s before listener started)",
                    event.id,
                    msg_date,
                )
                return

            # Also skip messages that lagged or are older than 2 minutes
            now = datetime.now(timezone.utc)
            if (now - msg_date).total_seconds() > 120:
                logger.debug("Skipping backlog message %s (age > 120s)", event.id)
                return

        # 2. Ignore outgoing messages from user account itself
        if event.out:
            return

        # 3. Ignore private 1-on-1 messages (only listen in monitored groups/supergroups)
        if event.is_private:
            return

        chat_id = event.chat_id
        if not is_chat_monitored(chat_id, settings, active_monitored_ids):
            return

        raw_text = (event.raw_text or "").strip()
        is_voice = False
        voice_bytes = None
        if not raw_text and voice_service and getattr(voice_service, "enabled", True):
            if event.voice or (event.audio and getattr(event.audio, "voice", False)):
                try:
                    audio_bytes = await event.download_media(file=bytes)
                    if audio_bytes:
                        voice_bytes = audio_bytes
                        transcribed = await voice_service.transcribe(audio_bytes, filename="voice.ogg")
                        if transcribed:
                            raw_text = transcribed.strip()
                            is_voice = True
                            logger.info(
                                "🎙️ [USERBOT VOICE TRANSCRIBED] User %s in '%s': '%s'",
                                event.sender_id,
                                event.chat_id,
                                raw_text,
                            )
                except Exception as dl_err:
                    logger.debug("Failed to download or transcribe userbot voice message %s: %s", event.id, dl_err)

        if not raw_text or raw_text.startswith("/") or len(raw_text) < 4:
            return

        sender_id = event.sender_id or 0

        try:
            # 4. Retrieve sender details
            sender = await event.get_sender()
            if sender and getattr(sender, "bot", False):
                return

            sender_name = "Yo'lovchi"
            sender_username = None

            if sender:
                first = getattr(sender, "first_name", "") or ""
                last = getattr(sender, "last_name", "") or ""
                full = f"{first} {last}".strip()
                if full:
                    sender_name = full
                sender_username = getattr(sender, "username", None)

            # 6. Retrieve chat details
            chat = await event.get_chat()
            chat_username = getattr(chat, "username", None)
            chat_title = getattr(chat, "title", str(chat_id))

            # 7. Classify passenger transport intent
            result = await classifier.classify(raw_text)

            sanitized_sender_name = clean_display_name(sender_name)
            user_display = f"@{sender_username}" if sender_username else f"{sanitized_sender_name} ({sender_id})"

            if result.is_passenger_request:
                logger.info(
                    "🎯 [USERBOT PASSENGER DETECTED] Conf: %.2f | Origin: %s | Dest: %s | Seats: %s | From: %s in '%s' (%s)",
                    result.confidence,
                    result.origin,
                    result.destination,
                    result.seats,
                    user_display,
                    chat_title,
                    chat_id,
                )

                # Userbot strictly monitors; dispatching to VIP group is always performed by the Bot
                await forwarder.forward_custom_passenger_request(
                    chat_id=chat_id,
                    message_id=event.id,
                    user_id=sender_id,
                    raw_text=raw_text,
                    classification=result,
                    passenger_name=sanitized_sender_name,
                    passenger_username=sender_username,
                    chat_username=chat_username,
                    is_voice=is_voice,
                    voice_bytes=voice_bytes,
                    userbot_client=client,
                    telethon_message=event.message,
                )
            else:
                # Active learning collection for ambiguous messages
                is_candidate = (
                    (0.30 <= result.confidence < settings.classifier_confidence_threshold)
                    or (
                        result.confidence <= 0.75
                        and any(
                            kw in raw_text.lower()
                            for kw in ("odam", "joy", "mashina", "moshina", "taksi", "bor", "ket", "kishi", "yo'lovchi")
                        )
                    )
                )
                if is_candidate and len(raw_text) >= 5:
                    logger.warning(
                        "⚠️ [USERBOT AMBIGUOUS MSG] Conf: %.2f | Reason: %s | User: %s",
                        result.confidence,
                        result.reason,
                        sender_id,
                    )
                    try:
                        pred_label = "OTHER"
                        if hasattr(classifier, "predict_message"):
                            pred = classifier.predict_message(raw_text)
                            pred_label = pred.get("label", "OTHER")

                        async with get_db_session() as session:
                            await CollectedMessageRepository.save_if_new(
                                session=session,
                                text=raw_text,
                                predicted_label=pred_label,
                                confidence=result.confidence,
                            )
                    except Exception as err:
                        logger.debug("Active learning save failed in userbot: %s", err)
                else:
                    logger.debug(
                        "⏭️ [USERBOT IGNORED] Conf: %.2f | Reason: %s | From: %s",
                        result.confidence,
                        result.reason,
                        sender_id,
                    )

        except Exception as exc:
            logger.error("Userbot error handling message %s in chat %s: %s", event.id, chat_id, exc)
