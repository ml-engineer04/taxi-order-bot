"""Machine Learning Classification Pipeline Architectures."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from app.services.normalization.transliterator import normalize_uzbek_text


class UzbekTextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that augments incoming raw messages
    with normalized Latin transliteration while preserving original script characters.

    This ensures seamless semantic transfer between Cyrillic, Latin, and mixed scripts
    without altering the raw input text for forwarding.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [f"{t} {normalize_uzbek_text(t)}" for t in X]


def build_baseline_pipeline() -> Pipeline:
    """
    Lightweight baseline:
    Augmented text + Simple Word-level TF-IDF with Multinomial Naive Bayes.
    """
    return Pipeline([
        ("preprocessor", UzbekTextPreprocessor()),
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 1),
                sublinear_tf=True,
                min_df=1,
            ),
        ),
        ("clf", MultinomialNB(alpha=0.5)),
    ])


def build_semantic_pipeline() -> Pipeline:
    """
    Advanced Multi-Gram Semantic Pipeline:
    Combines Uzbek transliteration augmentation, word-level n-grams,
    and character-level boundary n-grams (2-5) with a Calibrated Linear Classifier.

    Why this architecture?
    1. Preprocessor: Bridges Latin, Cyrillic, and mixed spellings into a unified feature space.
    2. char_wb (2-5): Captures agglutinative Uzbek suffixes (-dan, -ga, -kerek, -miz)
       and tolerates informal spelling/typos across Latin and Cyrillic.
    3. word (1-2): Captures semantic keyphrases like 'joy kerak', 'odam olaman'.
    4. Calibrated Linear Classifier: Provides accurate posterior probabilities
       P(PASSENGER_REQUEST), P(DRIVER_OFFER), P(OTHER) in < 2ms on CPU.
    """
    feature_union = FeatureUnion([
        (
            "word_features",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
                max_features=4000,
            ),
        ),
        (
            "char_features",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                sublinear_tf=True,
                min_df=1,
                max_features=8000,
            ),
        ),
    ])

    base_svc = LinearSVC(C=1.5, class_weight="balanced", random_state=42, max_iter=2000)
    calibrated_clf = CalibratedClassifierCV(estimator=base_svc, cv=3)

    return Pipeline([
        ("preprocessor", UzbekTextPreprocessor()),
        ("features", feature_union),
        ("clf", calibrated_clf),
    ])
