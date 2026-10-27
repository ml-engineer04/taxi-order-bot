"""Pluggable AI / LLM Classifier Provider using OpenAI (gpt-4o-mini)."""

import json
import logging
from typing import Optional
from app.services.classification.base import BaseClassifier, ClassificationResult

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
            "Analyze incoming group messages written in Uzbek (Latin, Cyrillic, or mixed) and classify them. "
            "1. PASSENGER: A person seeking a ride/taxi (e.g., 'Toshkentga odam bor', 'Samarqandga 2 ta joy kerak', 'Chirchiqqa mashina bormi'). "
            "2. DRIVER: A driver offering transport, seats, or delivery (e.g., 'odam olaman', '3 ta joy bor yuramiz', 'salon bosh'). "
            "3. OTHER: Greetings, news, car sales, non-transport chatter. "
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
                        return ClassificationResult(
                            is_passenger_request=bool(parsed.get("is_passenger", False)),
                            confidence=float(parsed.get("confidence", 0.95)),
                            origin=parsed.get("origin"),
                            destination=parsed.get("destination"),
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
