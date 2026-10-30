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
    # Uzbek taxi driver slang: "X ta kam" / "kam" / "odam kam" = driver has X seats remaining until full
    re.compile(r"\b(\d+|bitta|ikkita|uchta|to'rtta|tortta)?\s*ta?\s*(odam|kishi|joy)?[\s\.\-_]*kam\b"),
    re.compile(r"\b(\d+)\s*[\.\s]*kam\b"),
    re.compile(r"\b(bitta|ikkita|uchta)\s+kam\b"),
    # Driver seeking passengers to fill empty seats ("2 ta odam zarur/kerak")
    re.compile(r"\b(\d+|bitta|ikkita|uchta|to'rtta|tortta)\s*ta?\s*(odam|kishi|klient|klent|yo'lovchi|yolovchi)\s*(zarur|zarul|kerak|kerek|lozim)\b"),
    re.compile(r"\b(odam|kishi|klient|klent|yo'lovchi|yolovchi)\s+(zarur|zarul)\b"),
    # Picking up passengers or cargo ("poshta olaman", "odam olamiz", "pochta olmiz", "4 odam pochta olmiz")
    re.compile(r"\b(odam|yolovchi|yo'lovchi|klient|klent)\b[^\w\n]{0,30}\b(ham\s+)?(olaman|olamiz|olamz|olamiza|olmiz|olimiz)\b"),
    re.compile(r"\b(pochta|poshta|poshda|pochtalar|poshtalar|yuk|posilka)\b[^\w\n]{0,30}\b(ham\s+)?(olaman|olamiz|olamz|olamiza|olmiz|olimiz|opketaman|opketamiz)\b"),
    re.compile(r"\b(olaman|olamiz|olamz|olamiza|olmiz|olimiz|opketaman|opketamiz)\b[^\w\n]{0,30}\b(ham\s+)?(pochta|poshta|poshda|pochtalar|poshtalar|yuk|posilka|odam|yolovchi|yo'lovchi)\b"),
    re.compile(r"\b(\d+|bitta|ikkita|uchta|to'rtta|tortta)?\s*(ta)?\s*(odam|kishi|joy)?[\s\.\-_]*(pochta|poshta|poshda|yuk)?[\s\.\-_]*(olaman|olamiz|olamz|olamiza|olmiz|olimiz)\b"),
    re.compile(r"\b(odam|kishi|joy)\s+(pochta|poshta|poshda|yuk)\s+(olaman|olamiz|olamz|olamiza|olmiz|olimiz)\b"),
    # Driver competing/asking dispatchers for the order: "menga berila", "manga beringlar", "aka bering", "Bering menda...", "manga bervorila"
    re.compile(r"^(aka|oka|akka|brat)?\s*(bering|berin|berila|berilar|beringlar|bervoring|bervorila)\b"),
    re.compile(r"\b(bering|berin|berila|berilar|beringlar|bervoring|bervorila|bira\s*yim)\s+(menda|manda|enda|bizda|manbaqa|manga|menga)\b"),
    re.compile(r"\b(menga|manga|manbaqa|bizga|bizgayam)\b[^\w\n]{0,25}\b(berila|berilar|beringlar|bering|bervorila|bervoringlar|bervoring|chiqaringlar|chiqarila|chiqaring|tashlang|tashavorila|ber)\b"),
    re.compile(r"\b(berila|berilar|beringlar|bering|bervorila|bervoringlar|bervoring|chiqaringlar|chiqarila|chiqaring)\b[^\w\n]{0,25}\b(menga|manga|manbaqa|bizga|bizgayam|ozim|o'zim)\b"),
    re.compile(r"\b(aka|oka|uka|akka|og'a|oga|brat)\b[^\w\n]{0,20}\b(bering|berila|berilar|beringlar|bervorila|bervoring|chiqaring|berin|ber|bervoringlar)\b"),
    re.compile(r"\b(man|men)\s+(olaman|olib\s+ketaman|opketaman|olamiz|olib\s+boraman)\b"),
    # Driver departure announcements: "yuryapman uje", "menda 2 ta bor yuryapman", "menda ayol bor turibman"
    re.compile(r"\b(menda|manda|bizda)\b.{0,35}\b(ayol|odam|kishi|joy)\s*bor\b.{0,35}\b(turibman|turibmiz|turib\s*bor|yuryapman|yuryabman|yurvomman)\b"),
    re.compile(r"\b(menda|manda)\b.{0,35}\b(yuryapman|yuryabman|yurvomman|yurvoman|yuri\s*apba|yuryapmiz|ketvomman|ketvommiz)\b"),
    re.compile(r"\b(yuryapman|yuryabman|yurvomman|yurvoman|yuri\s*apba)\s*(uje|xozir|hozir|shu\s*yerda)?\b"),
    # Driver announcements of already collected/on-board passengers:
    # "olib chiqdik", "alib chiqdi", "olib kettik", "olib keldik", "olib ketyapmiz", "olib yurdim"
    re.compile(r"\b(olib|alib|op)\s*(chiqdik|chiqdim|chiqdi|chiqyapmiz|kettik|ketdik|ketyapmiz|ketyapman|keldik|keldim|yurdim|yuryapmiz|boldik|bo'ldik)\b"),
    # Drivers/dispatchers negotiating an order ("so'rovga kelishmasa", "kelishalmasa", "kelishmasa")
    re.compile(r"\b(so'rovga|sorovga|zakazga)?\s*(kelishmasa|kelishilmasa|kelishalmasa|kelisholmasa)\b"),
    # Clarification questions or quoting previous statements: "odam bor dedizmi", "pochta bor dedingizmi", "ddzmi shopir aka"
    re.compile(r"\b(dedizmi|dedingizmi|dedingmi|didizmi|ddzmi|deganmisiz|deganmidiz|dedilarmi|deganmidingiz|aytdizmi|aytganmidiz)\b"),
    # Driver soliciting parcels/passengers: "pochtalar bo'lsa aloqadaman", "pochta bo'lsa aloqadamiz", "odam bo'lsa aloqadaman", "bo'lsa aloqaga chiqing"
    re.compile(r"\b(aloqadaman|aloqadamiz|aloqadabiz|aloqadala)\b"),
    re.compile(r"\b(pochta|pochtalar|poshta|poshtalar|odam|odamlar|yuk|yuklar|klient|klent)\b.{0,30}\b(bo'lsa|bolsa|bosa)\b.{0,30}\b(aloqadaman|aloqadamiz|aloqada|telefon|tel|chiqing|yozing|ayting|bering|berila)\b"),
    re.compile(r"\b(bo'lsa|bolsa|bosa)\s+(aloqadaman|aloqadamiz|aloqada|aloqaga\s*chiqing)\b"),
    # Driver already has passengers loaded ("bizda X ta odam bor aeroportdan olib chiqdik uje")
    re.compile(r"\b(bizda|menda)\b.{0,30}\b(\d+|bitta|ikkita|uchta|to'rtta|tortta)?\s*(ta)?\s*(odam|kishi|ayol|joy)\s*bor\b.{0,35}\b(olib|alib|chiqdik|chiqdi|kelish|uje)\b"),
    # Driver services & departure announcements
    re.compile(r"\b(pustoy|pustoy\s+mashina|pustoy\s+moshina|bosh\s+mashina|bo'sh\s+mashina)\b"),
    re.compile(r"\b(klient|klent|mijoz)\s+vaqti\s+bilan\b"),
    re.compile(r"\b(pochta|poshta|poshda)\s*(xizmati|xizmatixam|xizmatiham)\b"),
    re.compile(r"\b(yuramiz|yuraman|yuramz)\b"),
    re.compile(r"\b(avto|moshina|mashina)\s+(kobalt|kobilt|koblt|cobalt|gentra|lasetti|lacetti|damas|nexia|neksiya|onix|tracker|treker|matiz|spark)\b"),
    re.compile(r"\b(kobalt|kobilt|koblt|cobalt|gentra|lasetti|lacetti|damas|nexia|neksiya|onix|tracker|treker)\s+bor\b"),
    re.compile(r"\b(tom[\s\.]*bagaj|bagaj[\s\.]*bor|bagajnik[\s\.]*bor|tomda\s*bagaj|bagaj[\s\.]*bo'sh|bagaj[\s\.]*bosh)\b"),
    re.compile(r"\b(orqada|orkada|salonda|mashinada|moshinada|oldinda|ichida)\b.{0,25}\b(ayol|ayollar|ayol\s+kishi)\s+bor\b"),
    re.compile(r"\b(ayol|ayollar|ayol\s+kishi)\s+bor\b.{0,25}\b(orqada|orkada|salonda|mashinada|moshinada|oldinda|ichida)\b"),
    re.compile(r"\b(ayollar\s*bor|ayol\s*ham\s*bor|mashinada\s+ayollar?\s*bor)\b"),
    # Empty vehicle capacity
    re.compile(r"\b(salon|bagaj|bagajnik)\s*(bosh|bo'sh)\b"),
    re.compile(r"\b(bosh|bo'sh)\s*(mashina|moshina|salon|bagaj)\b"),
]

# Negative Patterns: Driver offers, vehicle announcements, cargo transport
DRIVER_PATTERNS = [
    re.compile(r"\b(olib\s+boraman|olib\s+boramiz|olib\s+boramz|olib\s+ketaman|olib\s+ketamiz|olib\s+ketamz)\b"),
    re.compile(r"\b(yuk\s+tashish|dostavka\s+xizmati|dastavka\s+xizmati|dastafka\s+xizmati)\b"),
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
            return "Driver offer detected (e.g. 'ta kam', 'odam olamiz', 'poshta olamiz', 'odam zarur')"

    has_inquiry = bool(re.search(r"\b(bormi|bormikan|kerak|keremi|kerek|zarur|lozim)\b", norm))
    if not has_inquiry:
        for pat in DRIVER_PATTERNS:
            if pat.search(norm):
                return "Driver offer detected (e.g. 'joy bor', 'mashina bor', 'olib ketamiz')"
    return None

# Definitive Non-Transport / Real Estate / Job / Commercial Patterns
STRICT_NON_TRANSPORT_PATTERNS = [
    # Apartment / Room / Real Estate rental & Flatmates ('kv kerak', 'kvartira', 'arenda', 'ijara')
    re.compile(r"\b(kv|kvartira|kvartiraga|xona|xonadon|dom)\s+(kerak|kerek|zarur|bor|qidiryabman|qidiryapman|beriladi|olinadi|izlayapman|arenda|ijara)\b", re.IGNORECASE),
    re.compile(r"\b(arenda|arendaga|ijara|ijaraga)\s*(kvartira|kv|uy|xona|dom|ofis|dokon|do'kon)?\b", re.IGNORECASE),
    re.compile(r"\b(kv|kvartira|xona|uy|ofis|dokon|do'kon)\s+(arenda|arendaga|ijara|ijaraga)\b", re.IGNORECASE),
    re.compile(r"\b(sherik|sheriklikka|sherikka)\s*(kerak|bolalarga|qizlarga|bolalar|qizlar|og'il|qiz|talaba)?\b", re.IGNORECASE),
    re.compile(r"\b(kvga|kvartiraga)\s+(sherik|bolalar|qizlar|ogil|qiz)\b", re.IGNORECASE),
    re.compile(r"\b(\d+|bir|ikki|uch|to'rt|tort)?\s*xona(li)?\s+(kv|kvartira|uy|dom)\b", re.IGNORECASE),
    # Job / Employment / Vacancies
    re.compile(r"\b(ish|vakansiya|ishga|usta|santexnik|elektrik|oshpaz|ofitsiant|sotuvchi)\s+(kerak|bor|olamiz|qidiryapman|taklif)\b", re.IGNORECASE),
    re.compile(r"\b(ishga\s+taklif|ishga\s+olamiz|vakansiya|rezyume|rezume|cv)\b", re.IGNORECASE),
    # Electronics / Phone / Tech sales
    re.compile(r"\b(zaryadka|zaryadchik|remont|tamiri|ta'miri|zapchast|ekran|shina|balon|diska|moy\s+almashtirish)\b", re.IGNORECASE),
    re.compile(r"\b(telefon|iphone|ayfon|samsung|redmi|honor|noutbuk|kompyuter)\s+(sotiladi|sotaman|olaman|kerak|bor)\b", re.IGNORECASE),
    re.compile(r"\b(sotiladi|sotaman)\s+(telefon|iphone|ayfon|samsung|redmi|honor|noutbuk|mashina|uy|kvartira)\b", re.IGNORECASE),
]

def check_non_transport_patterns(norm: str) -> Optional[str]:
    """Check if normalized text contains definitive non-transport (real estate, jobs, commerce) patterns."""
    for pat in STRICT_NON_TRANSPORT_PATTERNS:
        if pat.search(norm):
            return "Non-transport message detected (e.g. apartment rental, jobs, sales)"
    return None

def detect_order_type(norm: str) -> str:
    """Detect whether request is passenger transport or cargo/pochta."""
    if re.search(r"\b(pochta|poshta|poshda|yuk|posilka|korobka|dastavka|dostavka)\b", norm):
        return "cargo"
    return "passenger"

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
    re.compile(r"\b(\d+)\s*(ta)?\s*(joy|odam|kishi|kishimiz|odammiz|ayol)\b"),
    re.compile(r"\b(bitta|ikkita|uchta|torta|to'rtta)\s+(joy|odam|kishi|ayol)\b"),
    re.compile(r"\b(joy\s+kerak|joy\s+kerek|joy\s+bormi|joy\s+bormikan)\b"),
    re.compile(r"\b(kishimiz|odammiz|kishi\s+bor|odam\s+bor|ayol\s+bor|ayol\s+kishi\s+bor)\b"),
    re.compile(r"\b(\d+|bitta|ikkita|uchta|torta|to'rtta)?\s*(ta)?\s*(ayol|ayol\s+kishi)\s*(bor|kerak|ketadi)\b"),
]

# Inquiry expressions (asking for ride availability)
INQUIRY_PATTERNS = [
    re.compile(r"\b(mashina\s+bormi|moshina\s+bormi|taksi\s+bormi|joy\s+bormi)\b"),
    re.compile(r"\b(mashina\s+bormikan|moshina\s+bormikan|taksi\s+bormikan|joy\s+bormikan)\b"),
    re.compile(r"\b(ketadigan\s+bormi|boradigan\s+bormi|yuradigan\s+bormi)\b"),
    re.compile(r"\b(bormi|bormikan)\b"),
]

# Cargo / Parcel Customer Request Patterns (Pochta / Yuk buyurtmalari)
CARGO_PATTERNS = [
    re.compile(r"\b(pochta|poshta|poshda|yuk|posilka|korobka|dastavka|dostavka)\s+bor\b"),
    re.compile(r"\b(pochta|poshta|poshda|yuk|posilka)\s+(berib\s+yuborish|bervorish|yuborish|jonatish|jo'natish|berib\s+yuboraman|bervoraman)\b"),
    re.compile(r"\b(berib\s+yuborishga|bervorishga|yuborishga|jonatishga)\s+(pochta|poshta|poshda|yuk|posilka)\b"),
    re.compile(r"\b(pochta|poshta|poshda|yuk|posilka)\s+(olib\s+ketadigan|opketadigan|boradigan|ketadigan)\s*(bormi|bormikan|taksi|mashina|moshina)?\b"),
    re.compile(r"\b(olib\s+ketadigan|opketadigan|boradigan)\s+(pochta|poshta|poshda|yuk|posilka)\s+bor\b"),
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
        match = re.search(r"\b(\d+)\s*(ta)?\s*(joy|odam|kishi|kishimiz|odammiz|ayol)", text)
        if match:
            try:
                count = int(match.group(1))
                if 1 <= count <= 8:
                    return count
            except ValueError:
                pass

        for word, num in WORD_TO_NUM.items():
            if re.search(rf"\b{word}\s+(joy|odam|kishi|ayol|ayol\s+kishi|ta\s+joy)", text):
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

        # 3. Check Negative Non-Transport Patterns (Apartments, jobs, sales, chatter)
        non_transport_reason = check_non_transport_patterns(norm)
        if non_transport_reason:
            return ClassificationResult(
                is_passenger_request=False,
                confidence=0.05,
                reason=non_transport_reason,
                method="rule_based",
            )

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
        has_cargo_signal = any(pat.search(norm) for pat in CARGO_PATTERNS)

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

        if has_cargo_signal:
            cargo_score = 0.55 if destination else 0.35
            score += cargo_score
            reasons.append(f"Cargo signal ({'with destination' if destination else 'detected'})")

        if seats is not None:
            score += 0.10
            reasons.append(f"Seats: {seats}")

        if has_inquiry:
            score += 0.20
            reasons.append("Inquiry for vehicle/availability")

        score = min(score, 0.98)
        is_passenger = score >= self.threshold
        detected_order_type = detect_order_type(norm)

        return ClassificationResult(
            is_passenger_request=is_passenger,
            confidence=round(score, 2),
            order_type=detected_order_type,
            origin=origin.capitalize() if origin else None,
            destination=destination.capitalize() if destination else None,
            seats=seats,
            reason="; ".join(reasons) if reasons else "Insufficient order signals",
            method="rule_based",
        )
