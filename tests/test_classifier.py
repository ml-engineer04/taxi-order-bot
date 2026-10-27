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
