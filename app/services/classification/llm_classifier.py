"""Pluggable AI / LLM Classifier Provider using OpenAI (gpt-4o-mini)."""

import json
import logging
from typing import Optional
from app.services.classification.base import BaseClassifier, ClassificationResult
from app.services.classification.rule_based import check_driver_patterns
from app.services.normalization.transliterator import normalize_uzbek_text

logger = logging.getLogger(__name__)


class LLMClassifier(BaseClassifier):
    """
    LLM-powered classifier provider using OpenAI gpt-4o-mini.
    Gracefully falls back to local classifier if API key is invalid, quota exhausted, or network down.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        fallback_classifier: Optional[BaseClassifier] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.fallback_classifier = fallback_classifier

    async def classify(self, raw_text: str) -> ClassificationResult:
        """Analyze message with LLM if API key is provided, or delegate to fallback."""
        if not raw_text or len(raw_text.strip()) < 4:
            return ClassificationResult(
                is_passenger_request=False,
                confidence=0.0,
                reason="Text too short",
                method="pre_filter",
            )

        # 1. Fast guard: Reject definitive driver offers immediately
        norm = normalize_uzbek_text(raw_text)
        driver_reason = check_driver_patterns(norm)
        if driver_reason:
            return ClassificationResult(
                is_passenger_request=False,
                confidence=0.99,
                reason=driver_reason,
                method="rule_driver_guard",
            )

        if not self.api_key:
            if self.fallback_classifier:
                return await self.fallback_classifier.classify(raw_text)
            return ClassificationResult(
                is_passenger_request=False,
                confidence=0.50,
                reason="LLM API key not configured; skipping AI classification",
                method="llm_skipped",
            )

        # Prompt for strict JSON evaluation tailored for Uzbekistan taxi groups
        system_prompt = (
            "You are an expert AI assistant for Uzbekistan intercity taxi groups. "
            "Analyze incoming group messages written in Uzbek (Latin, Cyrillic, or mixed) and classify them:\n\n"
            "CRITICAL UZBEK TAXI DIALECT RULES:\n"
            "1. DRIVER ANNOUNCEMENTS (is_passenger = false):\n"
            "   - 'X ta kam' / 'kam' (e.g., '1 ta kam', '2 ta kam', '1. ТА КАМ', 'bitta kam') ALWAYS means DRIVER has X seats remaining until full. NEVER a passenger request!\n"
            "   - 'joy bor', 'joylar bor', 'salon bo'sh/bosh' = DRIVER has empty seats. Contrast with 'joy kerak'/'joy bormi' = passenger.\n"
            "   - 'odam olamiz', 'klient olamiz', 'pochta olamiz', 'olib ketamiz' = DRIVER picking up passengers or cargo.\n"
            "   - 'mashina bor', 'damas bor', 'kobalt bor', 'yuramiz hozir', 'ketamiz kim bor' = DRIVER.\n"
            "   - Driver posts usually announce departure times, route, remaining seats, and phone number (e.g., 'Azon 3/4 ga Toshkentga 1 ta kam +998...').\n\n"
            "2. PASSENGER REQUESTS (is_passenger = true):\n"
            "   - Real passenger seeking transport: 'odam bor' (we have people waiting), '2 kishi bor', '2 kishimiz', '3 ta odam bor'.\n"
            "   - 'joy kerak', 'mashina kerak', 'taksi kerak', 'taksi bormi', 'ketadigan bormi', 'olib ketadigan bormi'.\n"
            "   - 'Toshkentga ketishim kerak', 'Samarqandga 2 ta joy kerak'.\n"
            "   - CRITICAL: A passenger request MUST clearly specify a destination or departure place. Random chatter, questions ('padashka bormi', 'kim bor'), screen sharing, or non-transport chatter are NOT passenger requests (is_passenger = false)!\n\n"
            "3. OTHER (is_passenger = false):\n"
            "   - Greetings, car selling, currency exchange, news, non-transport chatter.\n\n"
            "Respond ONLY with valid JSON in this exact structure: "
            '{"is_passenger": bool, "confidence": float, "origin": str|null, "destination": str|null, "seats": int|null, "reason": str}'
        )

        try:
            import aiohttp

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": raw_text},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=8,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"]
                        parsed = json.loads(content)
                        is_passenger = bool(parsed.get("is_passenger", False))
                        origin_val = parsed.get("origin")
                        dest_val = parsed.get("destination")
                        if is_passenger and not (origin_val or dest_val):
                            is_passenger = False

                        return ClassificationResult(
                            is_passenger_request=is_passenger,
                            confidence=float(parsed.get("confidence", 0.95)) if is_passenger else 0.20,
                            origin=origin_val.capitalize() if origin_val else None,
                            destination=dest_val.capitalize() if dest_val else None,
                            seats=parsed.get("seats"),
                            reason=parsed.get("reason", "OpenAI decision"),
                            method="openai",
                        )
                    else:
                        resp_body = await response.text()
                        logger.warning(
                            "OpenAI API returned HTTP %s: %s. Falling back to local classifier.",
                            response.status,
                            resp_body[:120],
                        )
        except Exception as exc:
            logger.error("LLM classification failed: %s. Falling back to local classifier.", exc)

        # Fallback to local classifier if available
        if self.fallback_classifier:
            return await self.fallback_classifier.classify(raw_text)

        return ClassificationResult(
            is_passenger_request=False,
            confidence=0.50,
            reason="OpenAI API request error without fallback",
            method="llm_error",
        )
