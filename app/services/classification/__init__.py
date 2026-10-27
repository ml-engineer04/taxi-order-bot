"""Classification Service Package with Claude, ML, OpenAI, and Hybrid support."""

from typing import Optional
from app.config import get_settings
from app.services.classification.base import BaseClassifier, ClassificationResult
from app.services.classification.rule_based import RuleBasedClassifier
from app.services.classification.llm_classifier import LLMClassifier
from app.services.classification.claude_classifier import ClaudeClassifier
from app.services.classification.ml_classifier import MLClassifier

__all__ = [
    "BaseClassifier",
    "ClassificationResult",
    "MLClassifier",
    "RuleBasedClassifier",
    "LLMClassifier",
    "ClaudeClassifier",
    "HybridClassifier",
    "get_classifier",
]


class HybridClassifier(BaseClassifier):
    """
    Coordinates fast local classifier with AI fallback (Claude or OpenAI)
    when confidence is in the ambiguous zone (0.35 - threshold).
    """

    def __init__(
        self,
        base_classifier: BaseClassifier,
        ai_classifier: Optional[BaseClassifier] = None,
        enable_ai: bool = False,
        threshold: float = 0.65,
    ):
        self.base_classifier = base_classifier
        self.ai_classifier = ai_classifier
        self.enable_ai = enable_ai
        self.threshold = threshold

    async def classify(self, raw_text: str) -> ClassificationResult:
        result = await self.base_classifier.classify(raw_text)

        # If AI is disabled or not provided, return base result
        if not self.enable_ai or not self.ai_classifier:
            return result

        # Check if score falls in the ambiguous evaluation zone
        if 0.35 <= result.confidence < self.threshold:
            ai_res = await self.ai_classifier.classify(raw_text)
            if ai_res.method in ("claude", "openai"):
                return ai_res

        return result


def get_classifier() -> BaseClassifier:
    """Factory to create the configured classifier instance."""
    settings = get_settings()

    rule_classifier = RuleBasedClassifier(
        confidence_threshold=settings.classifier_confidence_threshold
    )

    ml_classifier = MLClassifier(
        model_path=settings.ml_model_path,
        confidence_threshold=settings.classifier_confidence_threshold,
        fallback_classifier=rule_classifier,
    )

    claude_key_str = (
        settings.anthropic_api_key.get_secret_value()
        if settings.anthropic_api_key
        else None
    )

    claude_classifier = ClaudeClassifier(
        api_key=claude_key_str,
        fallback_classifier=None,  # Pure Claude AI without custom model
    )

    openai_key_str = (
        settings.openai_api_key.get_secret_value()
        if settings.openai_api_key
        else None
    )

    llm_classifier = LLMClassifier(
        api_key=openai_key_str,
        fallback_classifier=ml_classifier,
    )

    # 1. Direct Claude Mode (Anthropic)
    if settings.classifier_type in ("claude", "anthropic"):
        return claude_classifier

    # 2. Direct OpenAI Mode
    if settings.classifier_type in ("llm", "openai"):
        return llm_classifier

    # 3. Hybrid Mode: Fast Local ML + Claude/OpenAI for ambiguous cases
    if settings.classifier_type == "hybrid":
        ai_provider = claude_classifier if claude_key_str else llm_classifier
        has_ai_key = bool(claude_key_str or openai_key_str)
        return HybridClassifier(
            base_classifier=ml_classifier,
            ai_classifier=ai_provider,
            enable_ai=has_ai_key,
            threshold=settings.classifier_confidence_threshold,
        )

    # 4. Pure Rule-Based
    if settings.classifier_type == "rule":
        return rule_classifier

    # 5. Pure Local ML (Default)
    return ml_classifier
