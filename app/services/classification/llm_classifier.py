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
            "You are an expert AI assistant for Uzbekistan intercity taxi and parcel/cargo delivery groups. "
            "Analyze incoming group messages written in Uzbek (Latin, Cyrillic, or mixed) and classify them:\n\n"
            "CRITICAL RULES:\n"
            "1. DRIVER ANNOUNCEMENTS (is_order = false):\n"
            "   - 'X ta kam' / 'kam' (e.g., '1 ta kam', '2 ta kam') ALWAYS means DRIVER has remaining seats. NEVER an order!\n"
            "   - 'X ta odam zarur/zarul/kerak' (e.g. '2 ta odam zarul') = DRIVER needs passengers to fill empty car seats. NEVER an order!\n"
            "   - 'poshta olaman' / 'pochta olamiz' / 'poshta olamiz' / 'yuk olamiz' / 'odam olamiz' = DRIVER picking up passengers or cargo. Senders/passengers NEVER say 'olaman' or 'olamiz'!\n"
            "   - 'joy bor', 'joylar bor', 'salon bo'sh/bosh' = DRIVER has empty seats.\n"
            "   - 'mashina bor', 'damas bor', 'kobalt bor', 'yuramiz hozir', 'ketamiz kim bor' = DRIVER.\n"
            "   - Driver posts usually announce departure times, route, vehicle, and phone number.\n\n"
            "2. CUSTOMER ORDERS (is_order = true) - WE FORWARD BOTH PASSENGER AND POCHTA/CARGO REQUESTS:\n"
            "   A. Passenger requests (order_type = 'passenger'):\n"
            "      - 'odam bor' (passengers waiting), '2 kishi bor', '2 kishimiz', '3 ta odam bor'.\n"
            "      - 'joy kerak', 'mashina kerak', 'taksi kerak', 'taksi bormi', 'ketadigan bormi'.\n"
            "      - 'Toshkentga ketishim kerak', 'Samarqandga 2 ta joy kerak'.\n"
            "   B. Parcel / Cargo requests (order_type = 'cargo'):\n"
            "      - Customer has a parcel or cargo to send: 'pochta bor', 'poshta bor', 'yuk bor', 'posilka bor', 'korobka bor'.\n"
            "      - 'pochta berib yuborish kerak', 'poshta olib ketadigan bormi', 'yuk bor mashina kerak', 'bervorgani odam bormi'.\n"
            "   - CRITICAL: A customer order MUST clearly specify a destination or departure place. Non-transport chatter is NOT an order (is_order = false)!\n\n"
            "3. OTHER (is_order = false):\n"
            "   - Greetings, car selling, currency exchange, news, non-transport chatter.\n\n"
            "Respond ONLY with valid JSON in this exact structure: "
            '{"is_order": bool, "order_type": "passenger"|"cargo", "confidence": float, "origin": str|null, "destination": str|null, "seats": int|null, "time": str|null, "reason": str}'
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
                        parsed = None
                        if content:
                            try:
                                parsed = json.loads(content)
                            except json.JSONDecodeError:
                                match = re.search(r"\{.*\}", content, re.DOTALL)
                                if match:
                                    try:
                                        parsed = json.loads(match.group(0))
                                    except json.JSONDecodeError:
                                        pass

                        if not parsed:
                            if self.fallback_classifier:
                                return await self.fallback_classifier.classify(raw_text)
                            return ClassificationResult(
                                is_passenger_request=False,
                                confidence=0.30,
                                reason="OpenAI unparseable JSON",
                                method="llm_unparseable",
                            )
                        is_order = bool(parsed.get("is_order", parsed.get("is_passenger", False)))
                        order_type_val = parsed.get("order_type")
                        if not order_type_val or order_type_val not in ("cargo", "passenger"):
                            order_type_val = (
                                "cargo"
                                if re.search(
                                    r"\b(pochta|poshta|poshda|yuk|posilka|korobka|dastavka|dostavka)\b",
                                    raw_text,
                                    re.IGNORECASE,
                                )
                                else "passenger"
                            )
                        origin_val = parsed.get("origin")
                        dest_val = parsed.get("destination")

                        return ClassificationResult(
                            is_passenger_request=is_order,
                            order_type=order_type_val,
                            confidence=float(parsed.get("confidence", 0.95)) if is_order else 0.20,
                            origin=origin_val.capitalize() if origin_val else None,
                            destination=dest_val.capitalize() if dest_val else None,
                            seats=parsed.get("seats"),
                            time=str(parsed.get("time")).strip() if parsed.get("time") else None,
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
