"""Interactive authentication CLI script for Telegram Userbot."""

import asyncio
import os
import sys
from telethon import TelegramClient
from telethon.errors import (
    SessionPasswordNeededError,
    PhoneCodeInvalidError,
    PasswordHashInvalidError,
    FloodWaitError,
)

from app.config import get_settings


async def authenticate() -> None:
    """Run interactive terminal authentication to generate session file."""
    settings = get_settings()

    if not settings.tg_api_id or not settings.tg_api_hash:
        print("❌ Xatolik: .env faylida TG_API_ID yoki TG_API_HASH ko'rsatilmagan!")
        print("Iltimos, .env faylida TG_API_ID va TG_API_HASH qiymatlarini to'g'rilang.")
        sys.exit(1)

    session_path = settings.userbot_session_name
    dirname = os.path.dirname(session_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    print("==================================================")
    print("📱 Telegram Userbot (Telethon) Autentifikatsiyasi")
    print("==================================================")
    print(f"📁 Sessiya fayli manzili: {session_path}")
    print(f"🔑 API ID: {settings.tg_api_id}")
    print("--------------------------------------------------")

    client = TelegramClient(
        session=session_path,
        api_id=settings.tg_api_id,
        api_hash=settings.tg_api_hash,
    )

    await client.connect()

    if await client.is_user_authorized():
        me = await client.get_me()
        user_name = f"{me.first_name or ''} {me.last_name or ''}".strip()
        print(f"✅ Userbot allaqachon muvaffaqiyatli ulangan!")
        print(f"👤 Ism: {user_name}")
        print(f"🏷 Username: @{me.username}" if me.username else "🏷 Username: Mavjud emas")
        print(f"🆔 Telegram ID: {me.id}")
        await client.disconnect()
        return

    phone = input("\n📞 Telefon raqamingizni xalqaro formatda kiriting (masalan, +998901234567): ").strip()
    if not phone:
        print("❌ Telefon raqami kiritilmadi!")
        await client.disconnect()
        return

    try:
        sent = await client.send_code_request(phone)
        print(f"\n📩 Tasdiqlash kodi Telegram akkauntingizga yuborildi!")
    except FloodWaitError as e:
        print(f"❌ Telegram limitiga tushdingiz. {e.seconds} soniya kuting.")
        await client.disconnect()
        return
    except Exception as e:
        print(f"❌ Kod yuborishda xatolik yuz berdi: {e}")
        await client.disconnect()
        return

    code = input("💬 Telegram orqali kelgan tasdiqlash kodini kiriting: ").strip()
    # Remove any extra spaces or dashes
    code = code.replace(" ", "").replace("-", "")

    try:
        await client.sign_in(phone=phone, code=code, phone_code_hash=sent.phone_code_hash)
    except SessionPasswordNeededError:
        print("\n🔐 Ushbu akkauntda ikki bosqichli tekshiruv (2FA - Two-Step Verification) yoqilgan.")
        pwd = input("🔑 2FA parolingizni kiriting: ").strip()
        try:
            await client.sign_in(password=pwd)
        except PasswordHashInvalidError:
            print("❌ Noto'g'ri 2FA parol kiritildi!")
            await client.disconnect()
            return
    except PhoneCodeInvalidError:
        print("❌ Noto'g'ri tasdiqlash kodi kiritildi!")
        await client.disconnect()
        return
    except Exception as e:
        print(f"❌ Kirishda xatolik yuz berdi: {e}")
        await client.disconnect()
        return

    me = await client.get_me()
    user_name = f"{me.first_name or ''} {me.last_name or ''}".strip()
    print("\n🎉 Tabriklaymiz! Userbot muvaffaqiyatli autentifikatsiyadan o'tdi!")
    print(f"👤 Ulangan akkaunt: {user_name}")
    print(f"🏷 Username: @{me.username}" if me.username else "🏷 Username: Mavjud emas")
    print(f"🆔 Akkaunt ID: {me.id}")
    print(f"📁 Sessiya fayli yaratildi: {session_path}")
    print("\nEndi asosiy botni ishga tushirsangiz, userbot ham avtomatik ishlaydi!")

    await client.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(authenticate())
    except (KeyboardInterrupt, SystemExit):
        print("\nOperatsiya bekor qilindi.")
