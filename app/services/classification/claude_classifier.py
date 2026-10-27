"""AI Classifier Provider using Anthropic Claude (claude-haiku)."""

import json
import logging
import re
from typing import Optional
from app.services.classification.base import BaseClassifier, ClassificationResult

logger = logging.getLogger(__name__)


class ClaudeClassifier(BaseClassifier):
    """
    State-of-the-art AI Classifier using Anthropic Claude (Claude Haiku).
    Provides deep contextual semantic understanding of Uzbek taxi messages
    with graceful fallback to local ML classifier.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
        fallback_classifier: Optional[BaseClassifier] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.fallback_classifier = fallback_classifier

    async def classify(self, raw_text: str) -> ClassificationResult:
        """Classify message using Claude API with automatic JSON parsing."""
        if not self.api_key:
            if self.fallback_classifier:
                return await self.fallback_classifier.classify(raw_text)
            return ClassificationResult(
                is_passenger_request=False,
                confidence=0.50,
                reason="Claude API key not configured",
                method="claude_skipped",
            )

        system_prompt = (
            "You are an expert AI classifier for Uzbekistan intercity taxi groups. "
            "Analyze incoming group messages written in Uzbek (Latin, Cyrillic, or mixed) and classify them:\n"
            "1. PASSENGER: A person seeking a ride/taxi (e.g., 'Toshkentga odam bor', 'Samarqandga 2 ta joy kerak', 'Chirchiqqa mashina bormi', '2 kishimiz').\n"
            "2. DRIVER: A driver offering a ride, seats, or delivery (e.g., 'odam olaman', '3 ta joy bor yuramiz', 'salon bosh', 'pochta olamiz').\n"
            "3. OTHER: Greetings, weather, news, car sales, non-transport chatter.\n"
            "Respond ONLY with a JSON object in this exact structure:\n"
            "{\"is_passenger\": bool, \"confidence\": float, \"origin\": str|null, \"destination\": str|null, \"seats\": int|null, \"reason\": str}"
        )

        try:
            import aiohttp

            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload = {
                "model": self.model,
                "max_tokens": 200,
                "system": system_prompt,
                "messages": [{"role": "user", "content": raw_text}],
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=8,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        raw_content = data["content"][0]["text"].strip()

                        # Extract JSON object even if wrapped in markdown code fence
                        json_str = raw_content
                        match = re.search(r"\{.*\}", raw_content, re.DOTALL)
                        if match:
                            json_str = match.group(0)

                        parsed = json.loads(json_str)

                        origin_val = parsed.get("origin")
                        dest_val = parsed.get("destination")
                        seats_val = parsed.get("seats")
                        if seats_val is not None:
                            try:
                                seats_val = int(seats_val)
                            except (ValueError, TypeError):
                                seats_val = None

                        return ClassificationResult(
                            is_passenger_request=bool(parsed.get("is_passenger", False)),
                            confidence=float(parsed.get("confidence", 0.95)),
                            origin=origin_val.capitalize() if origin_val else None,
                            destination=dest_val.capitalize() if dest_val else None,
                            seats=seats_val,
                            reason=parsed.get("reason", "Claude decision"),
                            method="claude",
                        )
                    else:
                        resp_body = await response.text()
                        logger.warning(
                            "Claude API returned HTTP %s: %s. Delegating to fallback.",
                            response.status,
                            resp_body[:120],
                        )
        except Exception as exc:
            logger.error("Claude classification error: %s. Delegating to fallback.", exc)

        # Graceful fallback to local classifier
        if self.fallback_classifier:
            return await self.fallback_classifier.classify(raw_text)

        return ClassificationResult(
            is_passenger_request=False,
            confidence=0.50,
            reason="Claude request error without fallback",
            method="claude_error",
        )
