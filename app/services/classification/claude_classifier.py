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
        if not raw_text or len(raw_text.strip()) < 4:
            return ClassificationResult(
                is_passenger_request=False,
                confidence=0.0,
                reason="Text too short",
                method="pre_filter",
            )

        # 1. Deterministic strict driver check (prevents AI hallucination on obvious taxi ads like 'tom bagaj', 'pochta olmiz', 'ta kam')
        norm = normalize_uzbek_text(raw_text)
        from app.services.classification.rule_based import check_driver_patterns, check_non_transport_patterns

        driver_reason = check_driver_patterns(norm)
        if driver_reason:
            logger.info("🛑 [STRICT DRIVER REJECTED] %s | Text: '%s'", driver_reason, raw_text[:80])
            return ClassificationResult(
                is_passenger_request=False,
                confidence=0.98,
                reason=driver_reason,
                method="rule_based",
            )

        non_transport_reason = check_non_transport_patterns(norm)
        if non_transport_reason:
            logger.info("🛑 [STRICT NON-TRANSPORT REJECTED] %s | Text: '%s'", non_transport_reason, raw_text[:80])
            return ClassificationResult(
                is_passenger_request=False,
                confidence=0.99,
                reason=non_transport_reason,
                method="rule_based",
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
            "You are an expert AI classifier for Uzbekistan intercity taxi and parcel/cargo delivery groups. "
            "Analyze incoming group messages written in Uzbek (Latin, Cyrillic, or mixed) and classify them:\n\n"
            "CRITICAL RULES:\n"
            "1. DRIVER ANNOUNCEMENTS (is_order = false):\n"
            "   - 'Bering' / 'Bering menda ... bor' / 'Berin menda' / 'menda ... bor ... turibman' ALWAYS means a DRIVER asking dispatchers to assign passengers to him (e.g. 'Bering menda ayol kishi bor, 4 maktabda turibman')! Customers NEVER say 'Bering menda'! NEVER an order!\n"
            "   - 'menga berila' / 'manga berila' / 'menga beringlar' / 'manga bering' / 'manga bervorila' / 'manga chiqaringlar' / 'aka bering' / 'akka berin' / 'oka bering' ALWAYS means a DRIVER asking dispatchers or other drivers to give the passenger or cargo to HIM! Customers NEVER say 'menga berila' or 'aka bering'! NEVER an order!\n"
            "   - 'orqada ayol bor' / 'salonda ayol bor' / 'mashinada ayol bor' / 'oldinda ayol bor' ALWAYS means a DRIVER announcing that female passengers are already seated in his car to attract more passengers! NEVER a customer order!\n"
            "   - 'olib chiqdik' / 'olib chiqdim' / 'olib chiqdi' / 'alib chiqdi' / 'olib kettik' / 'olib ketdik' / 'olib ketyapmiz' / 'olib keldik' / 'olib yurdim' ALWAYS means a DRIVER who already picked up or is driving passengers (e.g. 'bizda 2 ta odam bor aeroportdan olib chiqdik uje', 'ikkinchi so\\'rovga ham kelishmasa bizda ikkita odam bor')! Customers NEVER say 'olib chiqdik' or 'olib kettik'! NEVER an order!\n"
            "   - 'so\\'rovga kelishmasa' / 'kelishmasa' / 'kelishilmasa' / 'kelishalmasa' ALWAYS means DRIVERS/DISPATCHERS negotiating orders among themselves. NEVER an order!\n"
            "   - 'dedizmi' / 'ddzmi' / 'dedingizmi' / 'didizmi' / 'aytdizmi' / 'deganmisiz' (e.g. 'Odam bor ddzmi shooir aka', 'Odam bor dedizmi shopir aka', 'pochta bor dedingizmi') ALWAYS means someone asking about another person's previous statement or a driver asking if passengers are available! Customers NEVER ask 'dedizmi'! NEVER an order!\n"
            "   - 'aloqadaman' / 'aloqadamiz' / 'pochtalar bolsa aloqadaman' / 'pochta bolsa aloqadaman' / 'odam bolsa aloqadaman' / 'bolsa aloqaga chiqing' ALWAYS means a DRIVER advertising his route/vehicle and waiting for orders (e.g. 'Xozrga Toshkonga Bogdod Rishton Buvaydadan Zaril pochtalar bolsa aloqadaman')! Senders with parcels say 'pochta bor', they NEVER say 'pochtalar bolsa aloqadaman'! ALWAYS is_order = false!\n"
            "   - Addressing drivers with questions like 'shopir aka dedizmi', 'shooir aka' is conversational driver coordination/chat, NOT an order!\n"
            "   - 'bizda X ta odam bor' when combined with already boarded passengers ('olib chiqdik', 'ketdik', 'yuryapmiz', 'kelishmasa') is a DRIVER announcement. NEVER an order!\n"
            "   - 'yuryapman uje' / 'yuryabman' / 'yurvomman' / 'menda X ta bor yuryapman' ALWAYS means a DRIVER departing right now and filling remaining seats! NEVER an order!\n"
            "   - 'tom bagaj bor' / 'том багаж бор' / 'bagaj bor' / 'tomda bagaj' ALWAYS means DRIVER has a car with roof luggage rack! Customers NEVER say 'tom bagaj bor'! NEVER an order!\n"
            "   - 'olamiz' / 'olaman' / 'olmiz' / 'оламиз' / 'олмиз' / 'olamz' (e.g. '4 odam pochta olmiz', 'pochta olamiz', 'odam olamiz') ALWAYS means DRIVER is accepting/collecting passengers and parcels. Senders/passengers NEVER say 'olamiz' or 'olmiz'! NEVER an order!\n"
            "   - 'X ta kam' / 'kam' (e.g., '1 ta kam', '2 ta kam', '1. ТА КАМ') ALWAYS means DRIVER has remaining seats. NEVER an order!\n"
            "   - 'X ta odam zarur/zarul/kerak' (e.g. '2 ta odam zarul') = DRIVER needs passengers to fill empty car seats. NEVER an order!\n"
            "   - 'poshta olaman' / 'pochta olamiz' / 'poshta olamiz' / 'yuk olamiz' / 'odam olamiz' = DRIVER picking up passengers or cargo. Senders/passengers NEVER say 'olaman' or 'olamiz'!\n"
            "   - 'joy bor', 'joylar bor', 'salon bo'sh/bosh' = DRIVER has empty seats.\n"
            "   - 'mashina bor', 'damas bor', 'kobalt bor', 'yuramiz hozir', 'ketamiz kim bor' = DRIVER.\n"
            "   - Driver posts usually announce departure times, route, vehicle, luggage capacity, and phone number.\n\n"
            "2. CUSTOMER ORDERS (is_order = true) - WE FORWARD BOTH PASSENGER AND POCHTA/CARGO REQUESTS:\n"
            "   A. Passenger requests (order_type = 'passenger'):\n"
            "      - 'odam bor' (passengers waiting), 'bitta odam bor', '2 kishi bor', '2 kishimiz', '3 ta odam bor', 'ayol kishi bor', 'bitta ayol bor'.\n"
            "      - 'joy kerak', 'mashina kerak', 'taksi kerak', 'taksi bormi', 'ketadigan bormi'.\n"
            "      - 'Toshkentga ketishim kerak', 'Samarqandga 2 ta joy kerak', 'shu nomerda bitta odam bor'.\n"
            "   B. Parcel / Cargo requests (order_type = 'cargo'):\n"
            "      - Customer has a parcel or cargo to send: 'pochta bor', 'poshta bor', 'yuk bor', 'posilka bor', 'korobka bor'.\n"
            "      - 'pochta berib yuborish kerak', 'poshta olib ketadigan bormi', 'yuk bor mashina kerak', 'bervorgani odam bormi'.\n\n"
            "3. SPOKEN / VOICE ORDER INTERPRETATION:\n"
            "   Voice messages often have informal speech or transit hubs mentioned:\n"
            "   - 'Yangi ToshMI', 'ToshMI', 'Shtab', 'Skver', 'Chorsu', 'Buvayda', 'Bachqir', 'Uchko'prik', 'Rishton', 'Bog'dod'.\n"
            "   - 'shu nomerda bitta odam bor' (passenger at this phone number), 'bitta odam bor ketadigan', 'pochta bor'.\n"
            "   - When a passenger or parcel order is stated with locations, always classify as is_order = true!\n\n"
            "4. OTHER & NON-TRANSPORT (is_order = false):\n"
            "   - REAL ESTATE & HOUSING: 'kv' (kvartira), 'kv kerak', 'kvartira kerak', 'ijara', 'arenda', 'sheriklikka', 'xona kerak' ALWAYS means apartment/house/room rental (e.g. 'Assalomu aleykum shuhrat chorraxa atrofidan kv kerak 1 kishiga')! Senders looking for apartments are NEVER taxi passengers! 'kv' NEVER means a car! ALWAYS is_order = false!\n"
            "   - JOBS & SERVICES: 'ish kerak', 'ishga olamiz', 'vakansiya', 'usta kerak', 'santexnik'. ALWAYS is_order = false!\n"
            "   - GOODS & SALES: car selling, phone selling, currency exchange, repair. ALWAYS is_order = false!\n"
            "   - Greetings alone, news, chat.\n"
            "   - Vague word lists, prompt echoes, or incomplete fragments without any destination (e.g. 'ayol kishi bor, yuk bor, shtab nomerda, boradi, ketadi' without where they are going) = is_order: false! A genuine transport request MUST specify where the passenger/cargo needs to go!\n\n"
            "Extract metadata:\n"
            "- origin: departure city/region (e.g. 'Qo‘qon', 'Toshkent')\n"
            "- destination: arrival city/region (e.g. 'Toshkent', 'Samarqand')\n"
            "- seats: number of passengers requested as integer (null for cargo)\n"
            "- order_type: 'cargo' if parcel/cargo/pochta, 'passenger' if person/people\n"
            "- time: departure time if mentioned in message (e.g. 'Hozir', 'Kechga', 'Soat 5da') or null\n\n"
            "Respond ONLY with a JSON object in this exact structure:\n"
            '{"is_order": bool, "order_type": "passenger"|"cargo", "confidence": float, "origin": str|null, "destination": str|null, "seats": int|null, "time": str|null, "reason": str}'
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
                "max_tokens": 500,
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
                        raw_content = ""
                        if data.get("content") and len(data["content"]) > 0:
                            raw_content = data["content"][0].get("text", "").strip()

                        # Extract JSON object even if wrapped in markdown code fence
                        parsed = None
                        match = re.search(r"\{.*\}", raw_content, re.DOTALL)
                        if match:
                            try:
                                parsed = json.loads(match.group(0))
                            except json.JSONDecodeError:
                                pass

                        if not parsed and raw_content:
                            try:
                                parsed = json.loads(raw_content)
                            except json.JSONDecodeError:
                                pass

                        if not parsed:
                            logger.warning(
                                "Claude returned unparseable response: '%s'. Falling back to local classifier.",
                                raw_content[:120],
                            )
                            if self.fallback_classifier:
                                return await self.fallback_classifier.classify(raw_text)
                            return ClassificationResult(
                                is_passenger_request=False,
                                confidence=0.30,
                                reason="Claude unparseable JSON",
                                method="claude_unparseable",
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
                        seats_val = parsed.get("seats")
                        time_val = parsed.get("time")

                        if seats_val is not None:
                            try:
                                seats_val = int(seats_val)
                            except (ValueError, TypeError):
                                seats_val = None

                        return ClassificationResult(
                            is_passenger_request=is_order,
                            order_type=order_type_val,
                            confidence=float(parsed.get("confidence", 0.95)) if is_order else 0.20,
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
