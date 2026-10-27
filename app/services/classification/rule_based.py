"""High-performance Rule-Based Classifier for Uzbek Passenger Ride Requests."""

import re
from typing import Optional, Tuple
from app.services.classification.base import BaseClassifier, ClassificationResult
from app.services.classification.data.uzbek_locations import (
    UZBEK_CITIES,
    ORIGIN_SUFFIXES,
    DESTINATION_SUFFIXES,
)
from app.services.normalization.transliterator import normalize_uzbek_text

# Strict Negative Patterns: Definitive driver offers, vehicle capacity, or collection announcements
STRICT_DRIVER_PATTERNS = [
    # Uzbek taxi driver slang: "X ta kam" / "kam" = driver has X seats remaining until full
    re.compile(r"\b(\d+|bitta|ikkita|uchta|to'rtta|tortta)\.?\s*ta\s*kam\b"),
    re.compile(r"\b(\d+)\s*kam\b"),
    re.compile(r"\b(bitta|ikkita)\s+kam\b"),
    # Picking up passengers or cargo
    re.compile(r"\b(odam|yolovchi|yo'lovchi|klient|klent)\s+(olaman|olamiz|olamz|olamiza)\b"),
    re.compile(r"\b(pochta|yuk)\s+(olaman|olamiz|olamz)\b"),
    re.compile(r"\b(salon|bagaj|bagajnik)\s*(bosh|bo'sh)\b"),
]

# Negative Patterns: Driver offers, vehicle announcements, cargo transport
DRIVER_PATTERNS = [
    re.compile(r"\b(olib\s+boraman|olib\s+boramiz|olib\s+boramz|olib\s+ketaman|olib\s+ketamiz|olib\s+ketamz)\b"),
    re.compile(r"\b(pochta|yuk|posilka|dostavka|dastavka)\s+bor\b"),
    re.compile(r"\b(yuk\s+tashish|dostavka|dastavka|dastafka)\b"),
    re.compile(r"\b(mashina|moshina|kobalt|cobalt|gentra|lasetti|lacetti|damas|nexia|neksiya|treker|tracker|onix)\s+bor\b"),
    re.compile(r"\b(joy|joylar)\s+bor\b"),
    re.compile(r"\b(joy\s+bo'sh|joy\s+bosh|bo'sh\s+joy|bosh\s+joy)\b"),
    re.compile(r"\b(yuramiz\s+hozir|yuraman\s+hozir|hozir\s+yuramiz|bosh\s+mashina|bo'sh\s+mashina)\b"),
    re.compile(r"\b(ketamiz\s+kim\s+bor|ketamz\s+kim\s+bor)\b"),
]

def check_driver_patterns(norm: str) -> Optional[str]:
    """Check if normalized text contains definitive driver announcement patterns."""
    for pat in STRICT_DRIVER_PATTERNS:
        if pat.search(norm):
            return "Driver offer detected (e.g. 'ta kam', 'odam olamiz', 'salon bo'sh')"

    has_inquiry = bool(re.search(r"\b(bormi|bormikan|kerak|keremi|kerek|zarur|lozim)\b", norm))
    if not has_inquiry:
        for pat in DRIVER_PATTERNS:
            if pat.search(norm):
                return "Driver offer detected (e.g. 'joy bor', 'mashina bor', 'olib ketamiz')"
    return None

# Irrelevant Non-Transport Keywords
NON_TRANSPORT_PATTERNS = [
    re.compile(r"\b(havo\s+yaxshi|havo\s+sovuq|havo\s+issiq)\b"),
    re.compile(r"\b(zaryadka|zaryadchik|telefon|remont|tamiri|ta'miri|zapchast)\b"),
    re.compile(r"\b(sotiladi|sotaman|olaman\s+narxi|arenda|ijara)\b"),
    re.compile(r"\b(o'tgan\s+hafta\s+bordim|kecha\s+bordim|bordim\b|kordim\b|ko'rdim\b)"),
]

# Passenger Need & Urgency Indicators
PASSENGER_NEED_PATTERNS = [
    re.compile(r"\b(kerak|keremi|kerek|zarur|lozim)\b"),
    re.compile(r"\b(ketaman|ketamiz|boraman|boramiz|borishim|ketishim)\b"),
    re.compile(r"\b(ketishim\s+kerak|ketishimiz\s+kerak|borishim\s+kerak|borishimiz\s+kerak)\b"),
    re.compile(r"\b(ketadigan|boradigan|yuradigan)\b"),
    re.compile(r"\b(taksi\s+bormi|mashina\s+bormi|moshina\s+bormi|joy\s+bormi)\b"),
]

# Passenger Keywords
PASSENGER_KEYWORDS = [
    re.compile(r"\b(taksi|mashina|moshina)\s+(kerak|kerek|zarur|bormi|bormikan)\b"),
    re.compile(r"\b(\d+)\s*(ta)?\s*(joy|odam|kishi|kishimiz|odammiz)\b"),
    re.compile(r"\b(bitta|ikkita|uchta|torta|to'rtta)\s+(joy|odam|kishi)\b"),
    re.compile(r"\b(joy\s+kerak|joy\s+kerek|joy\s+bormi|joy\s+bormikan)\b"),
    re.compile(r"\b(kishimiz|odammiz|kishi\s+bor|odam\s+bor)\b"),
]

# Inquiry expressions (asking for ride availability)
INQUIRY_PATTERNS = [
    re.compile(r"\b(mashina\s+bormi|moshina\s+bormi|taksi\s+bormi|joy\s+bormi)\b"),
    re.compile(r"\b(mashina\s+bormikan|moshina\s+bormikan|taksi\s+bormikan|joy\s+bormikan)\b"),
    re.compile(r"\b(ketadigan\s+bormi|boradigan\s+bormi|yuradigan\s+bormi)\b"),
    re.compile(r"\b(bormi|bormikan)\b"),
]

WORD_TO_NUM = {
    "bitta": 1, "1": 1,
    "ikkita": 2, "2": 2,
    "uchta": 3, "3": 3,
    "torta": 4, "to'rtta": 4, "4": 4,
}


def _match_city_stem(word: str, suffixes: tuple) -> Optional[str]:
    """Extract known city root handling optional possessive infixes (-i, -si)."""
    for sfx in suffixes:
        if word.endswith(sfx) and len(word) > len(sfx) + 2:
            base = word[:-len(sfx)]
            if base in UZBEK_CITIES:
                return base
            # Possessive infixes: aeroport-i-dan, markaz-i-ga
            if base.endswith("si") and base[:-2] in UZBEK_CITIES:
                return base[:-2]
            if base.endswith("i") and base[:-1] in UZBEK_CITIES:
                return base[:-1]
    return None


class RuleBasedClassifier(BaseClassifier):
    """
    Evaluates passenger transportation intent through:
    1. Text cleaning and normalized Latin transliteration.
    2. Negative driver/cargo pattern rejection.
    3. Origin and destination city extraction with agglutinative stem matching.
    4. Passenger need scoring and seat detection.
    """

    def __init__(self, confidence_threshold: float = 0.65):
        self.threshold = confidence_threshold

    def _extract_locations(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Identify departure (origin) and arrival (destination) locations."""
        words = re.findall(r"[a-z']+", text)
        origin = None
        destination = None

        for word in words:
            if not origin:
                orig_match = _match_city_stem(word, ORIGIN_SUFFIXES)
                if orig_match:
                    origin = orig_match
                    continue

            if not destination:
                dest_match = _match_city_stem(word, DESTINATION_SUFFIXES)
                if dest_match:
                    destination = dest_match
                    continue

        if not destination:
            for word in words:
                if word in UZBEK_CITIES and word != origin:
                    destination = word
                    break

        return origin, destination

    def _extract_seats(self, text: str) -> Optional[int]:
        """Extract requested passenger seat count."""
        match = re.search(r"\b(\d+)\s*(ta)?\s*(joy|odam|kishi|kishimiz|odammiz)", text)
        if match:
            try:
                count = int(match.group(1))
                if 1 <= count <= 8:
                    return count
            except ValueError:
                pass

        for word, num in WORD_TO_NUM.items():
            if re.search(rf"\b{word}\s+(joy|odam|kishi|ta\s+joy)", text):
                return num

        return None

    async def classify(self, raw_text: str) -> ClassificationResult:
        """Classify incoming raw message."""
        if not raw_text or len(raw_text.strip()) < 4:
            return ClassificationResult(
                is_passenger_request=False,
                confidence=0.0,
                reason="Text too short",
                method="rule_based",
            )

        # 1. Normalize and transliterate to unified Latin
        norm = normalize_uzbek_text(raw_text)

        # 2. Check Negative Driver Patterns
        driver_reason = check_driver_patterns(norm)
        if driver_reason:
            return ClassificationResult(
                is_passenger_request=False,
                confidence=0.05,
                reason=driver_reason,
                method="rule_based",
            )

        # 3. Check Negative Non-Transport Patterns
        for pat in NON_TRANSPORT_PATTERNS:
            if pat.search(norm):
                return ClassificationResult(
                    is_passenger_request=False,
                    confidence=0.10,
                    reason="Irrelevant non-transport topic or past event",
                    method="rule_based",
                )

        # 4. Extract Locations and Seats
        origin, destination = self._extract_locations(norm)
        seats = self._extract_seats(norm)

        # 5. Check Positive Intent Signals
        has_need_verb = any(pat.search(norm) for pat in PASSENGER_NEED_PATTERNS)
        has_passenger_keyword = any(pat.search(norm) for pat in PASSENGER_KEYWORDS)
        has_inquiry = any(pat.search(norm) for pat in INQUIRY_PATTERNS)

        # 6. Score Calculation
        score = 0.0
        reasons = []

        if destination and has_need_verb:
            score += 0.55
            reasons.append(f"Destination '{destination}' with passenger verb")
        elif destination:
            score += 0.25
            reasons.append(f"Destination '{destination}'")

        if origin:
            score += 0.15
            reasons.append(f"Origin '{origin}'")

        if has_passenger_keyword:
            kw_score = 0.45 if destination else 0.25
            score += kw_score
            reasons.append(f"Passenger signal ({'with destination' if destination else 'keyword'})")

        if seats is not None:
            score += 0.10
            reasons.append(f"Seats: {seats}")

        if has_inquiry:
            score += 0.20
            reasons.append("Inquiry for vehicle/availability")

        score = min(score, 0.98)
        is_passenger = score >= self.threshold

        return ClassificationResult(
            is_passenger_request=is_passenger,
            confidence=round(score, 2),
            origin=origin.capitalize() if origin else None,
            destination=destination.capitalize() if destination else None,
            seats=seats,
            reason="; ".join(reasons) if reasons else "Insufficient passenger signals",
            method="rule_based",
        )
