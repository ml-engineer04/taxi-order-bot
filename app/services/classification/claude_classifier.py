"""AI Classifier Provider using Anthropic Claude (claude-haiku)."""

import json
import logging
import re
from typing import Optional
from app.services.classification.base import BaseClassifier, ClassificationResult
from app.services.classification.rule_based import check_driver_patterns
from app.services.normalization.transliterator import normalize_uzbek_text

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
                reason="Claude API key not configured",
                method="claude_skipped",
            )

        system_prompt = (
            "You are an expert AI classifier for Uzbekistan intercity taxi groups. "
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
            "Extract metadata:\n"
            "- origin: departure city/region (e.g. 'Qo‘qon', 'Toshkent')\n"
            "- destination: arrival city/region (e.g. 'Toshkent', 'Samarqand')\n"
            "- seats: number of passengers/seats requested as integer\n"
            "- time: departure time if mentioned in message (e.g. 'Hozir', 'Kechga', 'Soat 5da', 'Ertalab') or null if not mentioned\n\n"
            "Respond ONLY with a JSON object in this exact structure:\n"
            "{\"is_passenger\": bool, \"confidence\": float, \"origin\": str|null, \"destination\": str|null, \"seats\": int|null, \"time\": str|null, \"reason\": str}"
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

                        is_passenger = bool(parsed.get("is_passenger", False))
                        origin_val = parsed.get("origin")
                        dest_val = parsed.get("destination")
                        seats_val = parsed.get("seats")
                        time_val = parsed.get("time")

                        # Validate that passenger request must have at least origin or destination
                        if is_passenger and not (origin_val or dest_val):
                            is_passenger = False

                        if seats_val is not None:
                            try:
                                seats_val = int(seats_val)
                            except (ValueError, TypeError):
                                seats_val = None

                        return ClassificationResult(
                            is_passenger_request=is_passenger,
                            confidence=float(parsed.get("confidence", 0.95)) if is_passenger else 0.20,
                            origin=origin_val.capitalize() if origin_val else None,
                            destination=dest_val.capitalize() if dest_val else None,
                            seats=seats_val,
                            time=str(time_val).strip() if time_val else None,
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
