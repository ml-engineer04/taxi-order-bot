"""Voice message Speech-to-Text (STT) transcription service using Groq Whisper."""

import logging
import re
from typing import Optional
import aiohttp

logger = logging.getLogger(__name__)

# Specialized Uzbek prompt biasing Whisper decoder to taxi domain terms & geographic names
UZBEK_TAXI_WHISPER_PROMPT = (
    "Viloyatlararo taksi, odam, yo'lovchi, pochta, yuk, haydovchi, ayol kishi, mashina, "
    "bering, olamiz, ketamiz, bormi, turibman, maktab, bozor, vokzal, aeroport. Shaharlar: "
    "Toshkent, Farg'ona, Andijon, Namangan, Samarqand, Buxoro, Xorazm, Urganch, Qarshi, "
    "Termiz, Jizzax, Sirdaryo, Navoiy, Nukus, Qo'qon, Buvayda, Bachqir, Rishton, Bog'dod, "
    "Uchko'prik, Yangiqo'rg'on, Beshariq, Yaypan, Quva, ToshMI, Chorsu, Shtab."
)


def clean_voice_transcription(text: str) -> str:
    """
    Normalize non-standard characters and regional dialect variants commonly
    produced when Whisper transcribes spoken Uzbek.
    """
    if not text:
        return ""

    # Discard hallucinated Whisper prompt echoes (when audio has silence/noise and Whisper repeats prompt words)
    if re.search(r"\b(ayol\s+kishi\s+bor).{0,35}\b(yuk\s+bor).{0,35}\b(boradi[,\s]+ketadi)\b", text, re.IGNORECASE):
        logger.warning("Discarded Whisper hallucinated prompt text: '%s'", text)
        return ""

    # 1. Map foreign/Turkic phonetic characters to standard Uzbek Latin
    char_map = {
        "ə": "a", "Ə": "A",
        "ı": "i", "I": "I",
        "š": "sh", "Š": "Sh",
        "č": "ch", "Č": "Ch",
        "ğ": "g'", "Ğ": "G'",
        "ö": "o'", "Ö": "O'",
        "ü": "u", "Ü": "U",
        "ä": "a", "Ä": "A",
        "ş": "sh", "Ş": "Sh",
        "ç": "ch", "Ç": "Ch",
    }
    for k, v in char_map.items():
        text = text.replace(k, v)

    # 2. Normalize regional dialect pronunciations & phonetic transcriptions
    dialect_patterns = [
        # Phonetic transcription fixes from fast speech
        (r"\bshin\s+umerda\b", "shu nomerda"),
        (r"\bshu\s+umerda\b", "shu nomerda"),
        (r"\b(bira\s*yim|birayim)\s*(enda|menda)\b", "bering menda"),
        (r"\byuanda\s+turib\s*bor\b", "yonida turibman"),
        (r"\bturib\s*bor\b", "turibman"),
        (r"\bmektad\b", "maktab"),
        (r"\b(akka|aka|oka)\s+berin\b", "aka bering"),
        (r"\bmendi\s+kita\b", "menda ikkita"),
        (r"\bkita\s+ayol\b", "ikkita ayol"),
        (r"\byuri\s+apba(\s+oji)?\b", "yuryabman uje"),
        (r"\borkada\b", "orqada"),
        (r"\b(yangi\s+tashmi|tashmi)\b", "Yangi ToshMI"),
        (r"\bbu\s*veydi\b", "Buvayda"),
        (r"\bbachqray\b", "Bachqirga"),
        (r"\bsherehlar\b", "sheriklar"),
        (r"\b(tizimislar|tuzikmislar)\b", "tuzukmisizlar"),
        (r"\bgelad[ae]?\b", "keladi"),
        (r"\b(ereport|ayraport|eroport)(dan)?\b", "aeroportdan"),
        (r"\b(alib|olib)\s+chiqdi(\s+goji|\s+koji|\s+guji|\s+uje)?\b", "olib chiqdik uje"),
        (r"\b(alib|olib)\s+chiqdik\b", "olib chiqdik"),
        (r"\bkelishalmasa\b", "kelisholmasa"),
        (r"\bkelishmasa\b", "kelishmasa"),
        (r"\bddzmi\b", "dedizmi"),
        (r"\bshooir\b", "shopir"),
        # Khorezm/regional: "ayar kisi" / "ayal kishi" -> "ayol kishi"
        (r"\b(ayar|ayal)\s*(kisi|kishi)\b", "ayol kishi"),
        # "odam bar", "kishi bar", "ayol bar", "pochta bar" -> "bor"
        (r"\b(odam|kishi|ayol|pochta|poshta|yuk)\s+bar\b", r"\1 bor"),
        # "yayin qirvan" / phonetic "yangiqorgon" -> "Yangiqo'rg'on"
        (r"\byayin\s*q[i\']?rvan\b", "Yangiqo'rg'on"),
        (r"\byangikorgon\b", "Yangiqo'rg'on"),
        (r"\byangiqo['’]?r\b", "Yangiqo'rg'on"),
        (r"\btshamanzar(dan)?\b", "Chamanzordan"),
        (r"\b(suchko\s*prik|suchkoprik)(ka)?\b", "Uchko'prikka"),
        (r"\bbarma\b", "bormi"),
        # Regional "gevaradi" / "gevarada" / "baredi" -> "boradi"
        (r"\bgevarad[ae]\b", "boradi"),
        (r"\bgevaram[i\']?z\b", "boramiz"),
        # Hub & verb forms
        (r"\bshtab\s*tomand[ae]\b", "Shtab tomonda"),
        (r"\byurg[ae]ll[ae]rg[ae]\b", "yurganlarga"),
        (r"\bodam\s*barm[i\']?k[ae]n\b", "odam bormikin"),
        (r"\bketad[i\']?g[ae]n\b", "ketadigan"),
    ]
    for pattern, repl in dialect_patterns:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    return text.strip()


class VoiceTranscriptionService:
    """Transcribes audio voice messages into Uzbek text using Groq Whisper (whisper-large-v3)."""

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        enabled: bool = True,
    ):
        self.groq_api_key = groq_api_key
        self.openai_api_key = openai_api_key
        self.enabled = enabled

    async def transcribe(self, audio_bytes: bytes, filename: str = "voice.ogg") -> Optional[str]:
        """
        Convert voice audio bytes into text.
        Returns transcribed text if successful, or None otherwise.
        """
        if not self.enabled:
            logger.debug("Voice transcription is disabled in settings.")
            return None

        if not audio_bytes or len(audio_bytes) < 100:
            logger.debug("Audio file too small or empty (%s bytes).", len(audio_bytes) if audio_bytes else 0)
            return None

        # 1. Primary: Groq Whisper (ultra-fast, free-tier flagship whisper-large-v3)
        if self.groq_api_key:
            text = await self._transcribe_groq(audio_bytes, filename)
            if text:
                return text

        # 2. Fallback: OpenAI Whisper API if Groq not configured or failed
        if self.openai_api_key:
            text = await self._transcribe_openai(audio_bytes, filename)
            if text:
                return text

        if not self.groq_api_key and not self.openai_api_key:
            logger.debug("Neither GROQ_API_KEY nor OPENAI_API_KEY configured for voice transcription.")

        return None

    async def _transcribe_groq(self, audio_bytes: bytes, filename: str) -> Optional[str]:
        """Transcribe using Groq Cloud whisper-large-v3 flagship model."""
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "User-Agent": "BotVoiceTranscriber/1.0",
            }
            data = aiohttp.FormData()
            data.add_field("file", audio_bytes, filename=filename, content_type="audio/ogg")
            data.add_field("model", "whisper-large-v3")
            data.add_field("language", "uz")
            data.add_field("prompt", UZBEK_TAXI_WHISPER_PROMPT)
            data.add_field("temperature", "0")
            data.add_field("response_format", "json")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers=headers,
                    data=data,
                    timeout=12,
                ) as response:
                    if response.status == 200:
                        res_json = await response.json()
                        raw_text = (res_json.get("text") or "").strip()
                        cleaned_text = clean_voice_transcription(raw_text)
                        if cleaned_text:
                            logger.info(
                                "🎙️ [GROQ WHISPER] Transcribed (%d bytes) -> '%s'",
                                len(audio_bytes),
                                cleaned_text,
                            )
                            # Intelligent phonetic AI refinement into clean literary Uzbek
                            refined = await self._refine_with_llm(cleaned_text)
                            return refined or cleaned_text
                    else:
                        resp_err = await response.text()
                        logger.warning("Groq Whisper API returned HTTP %s: %s", response.status, resp_err[:120])
        except Exception as exc:
            logger.warning("Groq Whisper transcription failed: %s", exc)

        return None

    async def _refine_with_llm(self, text: str) -> Optional[str]:
        """
        Polish raw STT transcription into clean, natural literary Uzbek text with
        correct place names using Groq's fast LLM (qwen/qwen3.8-27b) in ~0.2 seconds.
        """
        if not self.groq_api_key or not text or len(text) < 4:
            return None

        system_instruction = (
            "Siz O'zbekiston viloyatlararo taksi guruhlaridagi tez gapirilgan ovozli xabarlar bo'yicha mutaxassissiz. "
            "STT (Whisper) xato yoki fonetik eshitgan so'zlarni to'g'ri o'zbek adabiy tiliga tiklang.\n"
            "Misollar:\n"
            "- 'bira yim enda' -> 'bering menda'\n"
            "- 'akka berin mendi kita' -> 'aka bering menda ikkita'\n"
            "- 'yuri apba oji' -> 'yuryapman uje'\n"
            "- 'orkada ponga ayol bor' -> 'orqada bitta ayol bor'\n"
            "- 'torti 2 mektad yuanda turib bor' -> '4-maktab yonida turibman'\n"
            "- 'shin umerda' -> 'shu nomerda'\n"
            "- 'bitta ayar kisi bar yayin qirvan gevaradi' -> 'bitta ayol kishi bor Yangiqo'rg'onga boradi'\n\n"
            "QOIDALAR:\n"
            "1. Faqat va faqat to'g'rilangan yakuniy matnni bitta qatorda qaytaring.\n"
            "2. Hech qanday izoh, tushuntirish, kirish so'z yoki qo'shtirnoq yozmang!"
        )

        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json",
                "User-Agent": "BotVoiceRefiner/1.0",
            }
            payload = {
                "model": "qwen/qwen3.8-27b",
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": text},
                ],
                "temperature": 0.0,
                "max_tokens": 120,
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=3.5,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        refined = (data["choices"][0]["message"]["content"] or "").strip().strip('"\'')
                        # Sanity check: must be single line or clean text
                        refined = refined.split("\n")[0].strip()
                        if refined and len(refined) >= 4:
                            logger.info("✨ [VOICE AI POLISHED] '%s' -> '%s'", text, refined)
                            return refined
        except Exception as exc:
            logger.debug("LLM voice refinement skipped: %s", exc)

        return None

    async def _transcribe_openai(self, audio_bytes: bytes, filename: str) -> Optional[str]:
        """Fallback transcribe using OpenAI Whisper API."""
        try:
            headers = {"Authorization": f"Bearer {self.openai_api_key}"}
            data = aiohttp.FormData()
            data.add_field("file", audio_bytes, filename=filename, content_type="audio/ogg")
            data.add_field("model", "whisper-1")
            data.add_field("language", "uz")
            data.add_field("prompt", UZBEK_TAXI_WHISPER_PROMPT)
            data.add_field("temperature", "0")
            data.add_field("response_format", "json")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers=headers,
                    data=data,
                    timeout=15,
                ) as response:
                    if response.status == 200:
                        res_json = await response.json()
                        raw_text = (res_json.get("text") or "").strip()
                        cleaned_text = clean_voice_transcription(raw_text)
                        if cleaned_text:
                            logger.info(
                                "🎙️ [OPENAI WHISPER] Transcribed (%d bytes) -> '%s'",
                                len(audio_bytes),
                                cleaned_text,
                            )
                            # Polish with Groq LLM if available
                            if self.groq_api_key:
                                refined = await self._refine_with_llm(cleaned_text)
                                return refined or cleaned_text
                            return cleaned_text
                    else:
                        resp_err = await response.text()
                        logger.warning("OpenAI Whisper API returned HTTP %s: %s", response.status, resp_err[:120])
        except Exception as exc:
            logger.warning("OpenAI Whisper transcription failed: %s", exc)

        return None
