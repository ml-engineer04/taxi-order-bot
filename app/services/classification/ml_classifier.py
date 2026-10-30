"""Machine Learning NLP Classifier for Uzbek Taxi Group Messages.

Classifies incoming messages into:
1. PASSENGER_REQUEST - genuine passenger seeking transportation.
2. DRIVER_OFFER - driver advertising seats, cargo, or transport.
3. OTHER - greetings, unrelated discussions, spam.
"""

import asyncio
import logging
import os
from typing import Any, Dict, Optional, Tuple

import joblib

from app.services.classification.base import BaseClassifier, ClassificationResult
from app.services.classification.rule_based import RuleBasedClassifier

logger = logging.getLogger(__name__)


class MLClassifier(BaseClassifier):
    """
    Locally running Machine Learning classifier using a hybrid TF-IDF
    (word + char_wb) vectorizer and Calibrated LinearSVC pipeline.

    Operates asynchronously without blocking the Telegram event loop.
    """

    def __init__(
        self,
        model_path: str = "app/ml/models/taxi_classifier.joblib",
        confidence_threshold: float = 0.65,
        fallback_classifier: Optional[BaseClassifier] = None,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.fallback_classifier = fallback_classifier or RuleBasedClassifier(
            confidence_threshold=confidence_threshold
        )
        self.pipeline: Any = None
        self._load_model()
        self._rule_extractor = RuleBasedClassifier(
            confidence_threshold=confidence_threshold
        )

    def _load_model(self) -> None:
        """Load trained scikit-learn pipeline from disk."""
        if not os.path.exists(self.model_path):
            logger.warning(
                "ML model file not found at '%s'. Classification will fallback to rule-based.",
                self.model_path,
            )
            self.pipeline = None
            return

        try:
            self.pipeline = joblib.load(self.model_path)
            logger.info("Successfully loaded ML model from '%s'", self.model_path)
        except Exception as e:
            logger.error("Failed to load ML model from '%s': %s", self.model_path, e)
            self.pipeline = None

    def predict_message(self, text: str) -> Dict[str, Any]:
        """
        Synchronous inference endpoint.
        Returns label, confidence, and per-class probability distribution.

        predict_message(text) -> {"label": str, "confidence": float, "probabilities": dict}
        """
        if not text or len(text.strip()) < 3:
            return {
                "label": "OTHER",
                "confidence": 0.0,
                "probabilities": {
                    "PASSENGER_REQUEST": 0.0,
                    "DRIVER_OFFER": 0.0,
                    "OTHER": 1.0,
                },
            }

        if self.pipeline is None:
            raise RuntimeError(f"ML model pipeline not loaded from '{self.model_path}'")

        probs = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        class_prob_map = {
            str(cls_name): float(prob) for cls_name, prob in zip(classes, probs)
        }

        # Determine highest probability class
        best_label = max(class_prob_map, key=class_prob_map.get)
        confidence = float(class_prob_map[best_label])

        return {
            "label": best_label,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in class_prob_map.items()},
        }

    async def classify(self, raw_text: str) -> ClassificationResult:
        """
        Asynchronously classify message text without blocking the event loop.
        Extracts ride details when classified as PASSENGER_REQUEST.
        """
        if not raw_text or len(raw_text.strip()) < 3:
            return ClassificationResult(
                is_passenger_request=False,
                confidence=0.0,
                reason="Text too short or empty",
                method="ml",
            )

        # Fallback if model is not loaded
        if self.pipeline is None:
            logger.debug("Delegating to fallback classifier due to uninitialized ML pipeline")
            return await self.fallback_classifier.classify(raw_text)

        try:
            # Run CPU-bound ML inference in thread pool
            pred = await asyncio.to_thread(self.predict_message, raw_text)
        except Exception as e:
            logger.error("Error during ML inference for text '%s': %s", raw_text[:50], e)
            return await self.fallback_classifier.classify(raw_text)

        label = pred["label"]
        confidence = pred["confidence"]

        # Extract locations and seats using rule-based stem matching
        from app.services.normalization.transliterator import normalize_uzbek_text
        from app.services.classification.rule_based import detect_order_type

        norm = normalize_uzbek_text(raw_text)
        origin, destination = self._rule_extractor._extract_locations(norm)
        seats = self._rule_extractor._extract_seats(norm)
        detected_order_type = detect_order_type(norm)

        origin_str = origin.capitalize() if origin else None
        destination_str = destination.capitalize() if destination else None

        if label == "PASSENGER_REQUEST":
            is_forwardable = confidence >= self.confidence_threshold
            reason = (
                f"ML PASSENGER_REQUEST (confidence: {confidence:.2f})"
                if is_forwardable
                else f"ML PASSENGER_REQUEST below threshold ({confidence:.2f} < {self.confidence_threshold:.2f})"
            )
            return ClassificationResult(
                is_passenger_request=is_forwardable,
                order_type=detected_order_type,
                confidence=confidence,
                origin=origin_str,
                destination=destination_str,
                seats=seats,
                reason=reason,
                method="ml",
            )
        elif label == "DRIVER_OFFER":
            return ClassificationResult(
                is_passenger_request=False,
                confidence=confidence,
                origin=origin_str,
                destination=destination_str,
                seats=seats,
                reason=f"ML DRIVER_OFFER (confidence: {confidence:.2f})",
                method="ml",
            )
        else:  # OTHER
            return ClassificationResult(
                is_passenger_request=False,
                confidence=confidence,
                origin=origin_str,
                destination=destination_str,
                seats=seats,
                reason=f"ML OTHER/irrelevant (confidence: {confidence:.2f})",
                method="ml",
            )
