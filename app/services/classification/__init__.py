"""Classification Service Package with Claude, ML, OpenAI, and Hybrid support."""

import hashlib
import logging
import time
from typing import Dict, Optional, Tuple
from app.config import get_settings
from app.services.classification.base import BaseClassifier, ClassificationResult
from app.services.classification.rule_based import RuleBasedClassifier
from app.services.classification.llm_classifier import LLMClassifier
from app.services.classification.claude_classifier import ClaudeClassifier
from app.services.classification.deepseek_classifier import DeepSeekClassifier
from app.services.classification.ml_classifier import MLClassifier
from app.services.normalization.transliterator import normalize_uzbek_text

logger = logging.getLogger(__name__)

__all__ = [
    "BaseClassifier",
    "ClassificationResult",
    "MLClassifier",
    "RuleBasedClassifier",
    "LLMClassifier",
    "ClaudeClassifier",
    "DeepSeekClassifier",
    "HybridClassifier",
    "get_classifier",
]


class HybridClassifier(BaseClassifier):
    """
    Coordinates fast local ML classifier (~2ms) with AI fallback (DeepSeek, Claude, or OpenAI),
    normalized-text TTL caching for repeated ads, and asymmetric class thresholds.
    """

    def __init__(
        self,
        base_classifier: BaseClassifier,
        ai_classifier: Optional[BaseClassifier] = None,
        enable_ai: bool = False,
        threshold: float = 0.85,
        driver_reject_threshold: float = 0.72,
        other_reject_threshold: float = 0.82,
        reject_cache_ttl: int = 7200,
        order_cache_ttl: int = 900,
        max_cache_size: int = 5000,
    ):
        self.base_classifier = base_classifier
        self.ai_classifier = ai_classifier
        self.enable_ai = enable_ai
        self.threshold = threshold
        self.driver_reject_threshold = driver_reject_threshold
        self.other_reject_threshold = other_reject_threshold
        self.reject_cache_ttl = reject_cache_ttl
        self.order_cache_ttl = order_cache_ttl
        self.max_cache_size = max_cache_size
        # key -> (expires_at_monotonic, ClassificationResult)
        self._cache: Dict[str, Tuple[float, ClassificationResult]] = {}

    def predict_message(self, text: str):
        """Delegate synchronous ML prediction for active learning labeling when supported."""
        if hasattr(self.base_classifier, "predict_message"):
            return self.base_classifier.predict_message(text)
        return {"label": "OTHER", "confidence": 0.0, "probabilities": {}}

    def _make_cache_key(self, raw_text: str) -> Optional[str]:
        if not raw_text or len(raw_text.strip()) < 4:
            return None
        norm = normalize_uzbek_text(raw_text).strip()
        if len(norm) < 4:
            return None
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    def _get_cached(self, key: str) -> Optional[ClassificationResult]:
        entry = self._cache.get(key)
        if not entry:
            return None
        expires_at, res = entry
        if time.monotonic() > expires_at:
            self._cache.pop(key, None)
            return None
        return res.model_copy()

    def _put_cached(self, key: str, res: ClassificationResult) -> None:
        now = time.monotonic()
        if len(self._cache) >= self.max_cache_size:
            expired = [k for k, (exp, _) in self._cache.items() if now > exp]
            for k in expired:
                self._cache.pop(k, None)
            if len(self._cache) >= self.max_cache_size:
                oldest_key = next(iter(self._cache))
                self._cache.pop(oldest_key, None)
        ttl = self.order_cache_ttl if res.is_passenger_request else self.reject_cache_ttl
        self._cache[key] = (now + ttl, res.model_copy())

    def _needs_ai_review(self, result: ClassificationResult) -> Tuple[bool, bool]:
        """Return (should_query_ai, is_incomplete_route) using safe asymmetric thresholds."""
        lbl = result.predicted_label
        is_incomplete = result.is_passenger_request and not (result.origin and result.destination)

        if lbl == "PASSENGER_REQUEST":
            if result.is_passenger_request and result.origin and result.destination:
                return False, False
            return True, is_incomplete
        if lbl == "DRIVER_OFFER":
            return (0.35 <= result.confidence < self.driver_reject_threshold), False
        if lbl == "OTHER":
            return (0.35 <= result.confidence < self.other_reject_threshold), False

        is_ambiguous = 0.35 <= result.confidence < self.threshold
        return (is_ambiguous or is_incomplete), is_incomplete

    async def classify(self, raw_text: str) -> ClassificationResult:
        cache_key = self._make_cache_key(raw_text)
        if cache_key:
            cached = self._get_cached(cache_key)
            if cached is not None:
                logger.debug("⚡ [HYBRID CACHE HIT] Reused recent decision (order=%s, method=%s)", cached.is_passenger_request, cached.method)
                return cached

        result = await self.base_classifier.classify(raw_text)

        # If AI is disabled or not provided, return base ML result
        if not self.enable_ai or not self.ai_classifier:
            if cache_key:
                self._put_cached(cache_key, result)
            return result

        should_query_ai, is_incomplete_route = self._needs_ai_review(result)
        if should_query_ai:
            logger.info(
                "🤖 [HYBRID AI FALLBACK] Base ML score ambiguous (label=%s, conf=%.2f, incomplete_route=%s). Querying AI...",
                result.predicted_label or result.method,
                result.confidence,
                is_incomplete_route,
            )
            try:
                ai_res = await self.ai_classifier.classify(raw_text)
                if ai_res.method in ("deepseek", "claude", "openai"):
                    if cache_key:
                        self._put_cached(cache_key, ai_res)
                    return ai_res
            except Exception as e:
                logger.error("Hybrid AI classifier error: %s. Returning local ML result.", e)
            # Do not cache transient AI failure fallbacks so subsequent messages can recover
            return result

        if cache_key:
            self._put_cached(cache_key, result)
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

    deepseek_key_str = (
        settings.deepseek_api_key.get_secret_value()
        if settings.deepseek_api_key
        else None
    )

    deepseek_classifier = DeepSeekClassifier(
        api_key=deepseek_key_str,
        model=settings.deepseek_model,
        fallback_classifier=ml_classifier,
    )

    claude_key_str = (
        settings.anthropic_api_key.get_secret_value()
        if settings.anthropic_api_key
        else None
    )

    claude_classifier = ClaudeClassifier(
        api_key=claude_key_str,
        fallback_classifier=rule_classifier,  # Seamless fallback to rule-based engine on any API or parsing hiccup
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

    # 1. Direct DeepSeek Mode (Flagship V3 / Flash)
    if settings.classifier_type in ("deepseek", "deepseek-chat"):
        return deepseek_classifier

    # 2. Direct Claude Mode (Anthropic)
    if settings.classifier_type in ("claude", "anthropic"):
        return claude_classifier

    # 3. Direct OpenAI Mode
    if settings.classifier_type in ("llm", "openai"):
        return llm_classifier

    # 4. Hybrid Mode: Fast Local ML (~2ms) + DeepSeek/Claude/OpenAI for ambiguous cases
    if settings.classifier_type == "hybrid":
        ai_provider = deepseek_classifier if deepseek_key_str else (claude_classifier if claude_key_str else llm_classifier)
        has_ai_key = bool(deepseek_key_str or claude_key_str or openai_key_str)
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
