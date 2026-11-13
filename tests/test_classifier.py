"""Comprehensive Evaluation Benchmark Test Suite for Uzbek Passenger Classifier."""

import pytest
from app.services.classification.rule_based import RuleBasedClassifier
from tests.data.evaluation_dataset import LABELED_DATASET

classifier = RuleBasedClassifier(confidence_threshold=0.65)


@pytest.mark.asyncio
@pytest.mark.parametrize("item", [d for d in LABELED_DATASET if d["is_passenger"]])
async def test_positive_benchmark_cases(item):
    """Verify that genuine passenger ride requests are accurately detected."""
    res = await classifier.classify(item["text"])
    assert res.is_passenger_request is True, (
        f"Failed on [{item['category']}]: '{item['text']}' "
        f"(Confidence: {res.confidence}, Reason: {res.reason})"
    )
    assert res.confidence >= 0.65

    if item.get("origin"):
        assert res.origin == item["origin"], f"Origin mismatch on: '{item['text']}'"
    if item.get("destination"):
        assert res.destination == item["destination"], f"Destination mismatch on: '{item['text']}'"
    if item.get("seats") is not None:
        assert res.seats == item["seats"], f"Seat count mismatch on: '{item['text']}'"


@pytest.mark.asyncio
@pytest.mark.parametrize("item", [d for d in LABELED_DATASET if not d["is_passenger"]])
async def test_negative_benchmark_cases(item):
    """Verify that driver offers, delivery services, and casual chats are rejected."""
    res = await classifier.classify(item["text"])
    assert res.is_passenger_request is False, (
        f"False positive on [{item['category']}]: '{item['text']}' "
        f"(Confidence: {res.confidence}, Reason: {res.reason})"
    )


@pytest.mark.asyncio
async def test_overall_dataset_accuracy():
    """Verify that overall classification accuracy on the 40 benchmark samples exceeds 95%."""
    correct = 0
    total = len(LABELED_DATASET)

    for item in LABELED_DATASET:
        res = await classifier.classify(item["text"])
        if res.is_passenger_request == item["is_passenger"]:
            correct += 1

    accuracy = correct / total
    print(f"\n[BENCHMARK EVALUATION] Tested {total} samples. Accuracy: {accuracy * 100:.1f}% ({correct}/{total})")
    assert accuracy >= 0.95, f"Benchmark accuracy {accuracy * 100:.1f}% is below 95% threshold"


@pytest.mark.asyncio
async def test_driver_slang_rejection():
    """Verify that driver announcements with 'ta kam', 'joy bor' etc are rejected."""
    driver_messages = [
        """АЗОН 3/4  ГА ТОШКЕНТГА

         1. ТА  КАМ

БУВАЙДА ЯНГИКУРГОН

       АТРОФИДАН

+998772574100""",
        "Toshkentga 1 ta kam yuramiz",
        "Samarqandga 2 ta kam cobalt",
        "Qo'qonga bitta kam shoshilinch",
        "Andijonga 3 ta joy bor yuramiz",
        "Odam olamiz Toshkentga salon bosh",
        "Тошкент шахар ичидан риштон богдотга 2 та одам зарул пошта оламан тел +998931579092",
        "Toshkentdan Rishtonga 2 ta odam zarul poshta olaman",
        "2 ta odam zarur toshkentga",
        "poshta olamiz bugun kechga",
        """ТОШКЕНДАН ЮРАМИЗ
РИШТОН - БОГДОД
БУВАЙДА ГА.   
2 ta ОДАМ  ПОЧТА. ОЛАМИЗ. 
ТОМ. БАГАЖ. БОР.
🏎 АВТО  Коблт
☎️ TEЛ:  +998943971716""",
        """Toshkentdan шахар ичидан    
            Rishtonga.      Bogʻdotga yuramiz 
          2та 
odam.кам   pochta xizmatixam 
bor avto kobilt   ayollar bor
+998975557227""",
    ]

    for msg in driver_messages:
        res = await classifier.classify(msg)
        assert res.is_passenger_request is False, f"Expected False for driver message: {msg}"


@pytest.mark.asyncio
async def test_cargo_request_detection():
    """Verify that cargo/pochta customer delivery requests are classified as forwardable orders."""
    cargo_requests = [
        "Toshkentdan Fargonaga pochta bor berib yuborish kerak",
        "Samarqandga poshta bor olib ketadigan bormi",
        "Andijondan Toshkentga 1 ta korobka yuk bor taksi bormi",
        "Buxoroga pochta bor srochni",
    ]

    for msg in cargo_requests:
        res = await classifier.classify(msg)
        assert res.is_passenger_request is True, f"Expected True for cargo order: {msg}"
        assert res.order_type == "cargo", f"Expected cargo order_type for: {msg}"


@pytest.mark.asyncio
async def test_lost_items_and_documents_rejected():
    """Verify that lost items, documents, and rewards announcements are rejected."""
    lost_item_messages = [
        "‼️‼️‼️ Андижондан мехмон булиб келган Багдод Янгикургон Кукон шахрида булган Инобиддинов Абдулходий Иброхимович тегишли права йукотиб куйилган топганга суюнчиси бор +998943802681",
        "Toshkent Samarqand trassasida pasport tushib qolgan topganga suyunchi beriladi",
        "Qo'qonda qora sumka yo'qotib qo'yilgan ichida hujjatlar bor edi iltimos topganlar qaytarsin",
        "Toshkentdan Namanganga ketayotganda mashinada kalit qolib ketgan",
        "Prava topib olindi Alimov Shuxrat nomiga",
    ]

    for msg in lost_item_messages:
        res = await classifier.classify(msg)
        assert res.is_passenger_request is False, f"Expected False for lost item message: {msg}"


@pytest.mark.asyncio
async def test_driver_kam_slang_and_passenger_bittakam_protection():
    """Verify driver '1 odam kam'/'kamdamiz'/'alishtir' are rejected while passenger 'kimga bittakam' is NOT rejected as driver."""
    driver_slang = [
        "Soat 12 larda Bogdod Rishtondan Toshkent shaxarga 1 odam kam",
        "TOSHKENDAN QAYTISHGA 2 TA KAMDAMIZ AYOLLAR BOR",
        "Oldi mestaga 1ta odam bor Alishtiradiganlar bormi",
    ]
    for msg in driver_slang:
        res = await classifier.classify(msg)
        assert res.is_passenger_request is False, f"Driver slang falsely accepted: {msg}"

    # Passenger saying 'kimga bittakam' must NOT be blocked by check_driver_patterns
    from app.services.classification.rule_based import check_driver_patterns
    from app.services.normalization.transliterator import normalize_uzbek_text

    passenger_msg = "Yunusabaddan bir kishi bor, ayol kishi, kimga bittakam, telefon qiling"
    assert check_driver_patterns(normalize_uzbek_text(passenger_msg)) is None


@pytest.mark.asyncio
async def test_colloquial_city_aliases_and_suffixes():
    """Verify colloquial/truncated stems and -tan/-da suffixes extract both cities accurately."""
    res1 = await classifier.classify("Toshkendan Бешарикга 1 ta odam bor kerak")
    assert res1.origin == "Toshkent"
    assert res1.destination == "Beshariq"

    res2 = await classifier.classify("Chilonzorda 1 ta odam bor quqonga")
    assert res2.origin == "Chilonzor"
    assert res2.destination == "Qo'qon"

    res3 = await classifier.classify("Beshariqtan 2 ta odam bor toshkenga")
    assert res3.origin == "Beshariq"
    assert res3.destination == "Toshkent"

    res4 = await classifier.classify("Toshdan beshriqqa 1 kishi bor")
    assert res4.origin == "Toshkent"
    assert res4.destination == "Beshariq"

    res5 = await classifier.classify("Sadoqattan Rishtonga 1 kishi bor")
    assert res5.origin == "Sadoqat"
    assert res5.destination == "Rishton"


@pytest.mark.asyncio
async def test_mountain_pass_and_road_condition_inquiries():
    """Verify road condition / weather questions on Kamchik pass are rejected as non-transport."""
    samples = [
        "pereval ochiqmi akalar",
        "kamchikda qor yog'yaptimi",
        "yo'l ochiqmi shopirlar",
        "postda turibman pereval yopiqmi",
        "dovon qalay yursa bo'ladimi",
    ]
    for s in samples:
        res = await classifier.classify(s)
        assert res.is_passenger_request is False, f"Road condition allowed: {s}"
        assert res.confidence <= 0.10


@pytest.mark.asyncio
async def test_driver_dispatch_begging_rejections():
    """Verify drivers pleading for orders are strictly rejected."""
    samples = [
        "aka bering bitta zakaz",
        "bervoring aka",
        "bitta menga bering",
        "aka chiqaring bitta",
    ]
    for s in samples:
        res = await classifier.classify(s)
        assert res.is_passenger_request is False, f"Driver plea allowed: {s}"
