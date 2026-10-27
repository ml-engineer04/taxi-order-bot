"""Uzbek text cleaning and character normalization."""

import re

# Various apostrophe characters used in Uzbek Latin typing
APOSTROPHE_PATTERN = re.compile(r"[\u02BB\u02BC\u02BD\u2018\u2019\u0060\u00B4\u2032]")

# Multiple spaces or consecutive punctuation
MULTIPLE_WHITESPACE = re.compile(r"\s+")
MULTIPLE_PUNCTUATION = re.compile(r"([!?.,;:])\1+")


def clean_text(text: str) -> str:
    """
    Clean raw input text:
    - Lowercases text.
    - Standardizes all Unicode apostrophes (o', g', etc.) to standard ASCII single quote '.
    - Reduces repeated punctuation (e.g. '???' -> '?').
    - Collapses extra whitespace.
    """
    if not text:
        return ""

    # 1. Lowercase
    cleaned = text.lower()

    # 2. Standardize apostrophes
    cleaned = APOSTROPHE_PATTERN.sub("'", cleaned)

    # 3. Collapse repeated punctuation
    cleaned = MULTIPLE_PUNCTUATION.sub(r"\1", cleaned)

    # 4. Collapse whitespace
    cleaned = MULTIPLE_WHITESPACE.sub(" ", cleaned).strip()

    return cleaned
