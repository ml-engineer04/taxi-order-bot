"""Uzbek Cyrillic to Latin transliterator with word-boundary awareness."""

import re
from app.services.normalization.cleaner import clean_text

# Multi-character mappings
CYRILLIC_MULTI = [
    ("ў", "o'"),
    ("ғ", "g'"),
    ("ш", "sh"),
    ("ч", "ch"),
    ("ё", "yo"),
    ("ю", "yu"),
    ("я", "ya"),
    ("ц", "ts"),
    ("щ", "sh"),
    ("қ", "q"),
    ("ҳ", "h"),
    ("ъ", "'"),
    ("ь", ""),
]

# Single character mappings
CYRILLIC_SINGLE = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "ж": "j",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "x",
    "э": "e",
}

# Vowels in Cyrillic for determining 'ye' vs 'e'
CYRILLIC_VOWELS = set("аеёиоөуүэюя")


def to_latin(text: str) -> str:
    """Convert Uzbek Cyrillic (or mixed Cyrillic/Latin) text to normalized Latin."""
    if not text:
        return ""

    result = []
    text_lower = text.lower()
    i = 0
    n = len(text_lower)

    while i < n:
        char = text_lower[i]

        # Handle 'е' (ye at word start or after vowel; e after consonant)
        if char == "е":
            if i == 0 or text_lower[i - 1] in " \t\n\r-–—.,!?:;\"'([{<" or text_lower[i - 1] in CYRILLIC_VOWELS:
                result.append("ye")
            else:
                result.append("e")
            i += 1
            continue

        # Check multi-char mappings
        matched_multi = False
        for cyr, lat in CYRILLIC_MULTI:
            if text_lower.startswith(cyr, i):
                result.append(lat)
                i += len(cyr)
                matched_multi = True
                break
        if matched_multi:
            continue

        # Check single-char mappings
        if char in CYRILLIC_SINGLE:
            result.append(CYRILLIC_SINGLE[char])
        else:
            result.append(char)
        i += 1

    return "".join(result)


def normalize_uzbek_text(raw_text: str) -> str:
    """
    Complete normalization pipeline:
    1. Clean whitespace and apostrophe variants.
    2. Transliterate any Cyrillic letters to Latin.
    3. Re-clean apostrophes and extra spaces.
    """
    cleaned = clean_text(raw_text)
    latin = to_latin(cleaned)
    return clean_text(latin)
