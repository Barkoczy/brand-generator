"""Similarity algorithms for trademark name comparison.

This module provides various similarity metrics used to compare
candidate names against existing trademarks:
- Levenshtein (edit distance) similarity
- Phonetic similarity (Soundex, Metaphone-like)
- Visual similarity (character shape and trigram overlap)
"""

import unicodedata
from dataclasses import dataclass

from src.scoring.phonetic_models import levenshtein_distance


@dataclass
class NameSimilarityResult:
    """Result of name similarity comparison.

    Attributes:
        name_a: First name (normalized)
        name_b: Second name (normalized)
        edit_distance: Raw Levenshtein edit distance
        string_similarity: String similarity score (0.0-1.0)
        phonetic_similarity: Phonetic similarity score (0.0-1.0)
        visual_similarity: Visual similarity score (0.0-1.0)
        combined_score: Weighted combination of all similarity metrics
    """

    name_a: str
    name_b: str
    edit_distance: int
    string_similarity: float
    phonetic_similarity: float
    visual_similarity: float
    combined_score: float


def normalize_name(name: str) -> str:
    """Normalize a name for comparison.

    - Convert to lowercase
    - Remove accents/diacritics
    - Remove non-alphanumeric characters
    - Strip whitespace

    Args:
        name: Name to normalize

    Returns:
        Normalized name string
    """
    # Convert to lowercase
    name = name.lower().strip()

    # Remove accents/diacritics (NFD normalization)
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")

    # Remove non-alphanumeric characters (keep only letters and digits)
    name = "".join(c for c in name if c.isalnum())

    return name


def string_similarity(name_a: str, name_b: str) -> float:
    """Calculate string similarity based on Levenshtein distance.

    Returns similarity score from 0.0 (completely different) to 1.0 (identical).

    Args:
        name_a: First name
        name_b: Second name

    Returns:
        Similarity score (0.0-1.0)
    """
    na = normalize_name(name_a)
    nb = normalize_name(name_b)

    if not na or not nb:
        return 0.0

    if na == nb:
        return 1.0

    distance = levenshtein_distance(na, nb)
    max_len = max(len(na), len(nb))

    return 1.0 - (distance / max_len)


# Phonetic encoding mappings (simplified Soundex-like approach)
PHONETIC_GROUPS = {
    # Labials (lip sounds)
    "b": "1", "f": "1", "p": "1", "v": "1",
    # Gutturals (throat sounds)
    "c": "2", "g": "2", "j": "2", "k": "2", "q": "2", "s": "2", "x": "2", "z": "2",
    # Dentals (tongue-teeth sounds)
    "d": "3", "t": "3",
    # Liquids
    "l": "4",
    # Nasals
    "m": "5", "n": "5",
    # Vibrants
    "r": "6",
    # Vowels (often dropped in phonetic encoding)
    "a": "0", "e": "0", "i": "0", "o": "0", "u": "0",
    # Semi-vowels
    "h": "", "w": "", "y": "",
}


def get_phonetic_code(name: str) -> str:
    """Generate a phonetic code for a name (Soundex-like).

    This encoding groups phonetically similar sounds together,
    making it useful for finding names that sound alike.

    Args:
        name: Name to encode

    Returns:
        Phonetic code string
    """
    name = normalize_name(name)
    if not name:
        return ""

    # Keep first letter
    code = [name[0].upper()]

    # Encode remaining characters
    prev_digit = PHONETIC_GROUPS.get(name[0], "")

    for char in name[1:]:
        digit = PHONETIC_GROUPS.get(char, "")

        # Skip if same as previous (consecutive similar sounds)
        if digit and digit != prev_digit and digit != "0":
            code.append(digit)
            prev_digit = digit
        elif digit == "0":
            # Vowels reset the previous digit (allow repeated consonant groups)
            prev_digit = ""

    # Pad to minimum length 4 or truncate
    result = "".join(code)
    if len(result) < 4:
        result = result + "0" * (4 - len(result))

    return result[:6]  # Max 6 characters


def phonetic_similarity(name_a: str, name_b: str) -> float:
    """Calculate phonetic similarity between two names.

    Uses a Soundex-like phonetic encoding to compare how names sound,
    regardless of exact spelling.

    Args:
        name_a: First name
        name_b: Second name

    Returns:
        Similarity score (0.0-1.0)
    """
    code_a = get_phonetic_code(name_a)
    code_b = get_phonetic_code(name_b)

    if not code_a or not code_b:
        return 0.0

    if code_a == code_b:
        return 1.0

    # Compare phonetic codes using edit distance
    distance = levenshtein_distance(code_a, code_b)
    max_len = max(len(code_a), len(code_b))

    return 1.0 - (distance / max_len)


def get_trigrams(text: str) -> set[str]:
    """Extract character trigrams from text.

    Trigrams are overlapping 3-character sequences that help
    capture visual/structural similarity.

    Args:
        text: Input text

    Returns:
        Set of trigrams
    """
    text = normalize_name(text)
    if len(text) < 3:
        # For short strings, use bigrams or the string itself
        if len(text) == 2:
            return {text}
        return {text} if text else set()

    return {text[i : i + 3] for i in range(len(text) - 2)}


def visual_similarity(name_a: str, name_b: str) -> float:
    """Calculate visual similarity using trigram overlap.

    This metric captures structural/visual similarity that might
    cause confusion in written form.

    Args:
        name_a: First name
        name_b: Second name

    Returns:
        Similarity score (0.0-1.0) using Jaccard index
    """
    trigrams_a = get_trigrams(name_a)
    trigrams_b = get_trigrams(name_b)

    if not trigrams_a or not trigrams_b:
        # Fall back to string similarity for very short names
        return string_similarity(name_a, name_b)

    # Jaccard similarity: intersection / union
    intersection = len(trigrams_a & trigrams_b)
    union = len(trigrams_a | trigrams_b)

    if union == 0:
        return 0.0

    return intersection / union


def combined_similarity(
    name_a: str,
    name_b: str,
    weights: dict[str, float] | None = None,
) -> NameSimilarityResult:
    """Calculate combined similarity using multiple metrics.

    Args:
        name_a: First name
        name_b: Second name
        weights: Optional weights for each metric. Defaults to:
            - string: 0.4
            - phonetic: 0.35
            - visual: 0.25

    Returns:
        NameSimilarityResult with all similarity scores
    """
    if weights is None:
        weights = {
            "string": 0.4,
            "phonetic": 0.35,
            "visual": 0.25,
        }

    na = normalize_name(name_a)
    nb = normalize_name(name_b)

    str_sim = string_similarity(name_a, name_b)
    phon_sim = phonetic_similarity(name_a, name_b)
    vis_sim = visual_similarity(name_a, name_b)

    combined = (
        str_sim * weights.get("string", 0.4)
        + phon_sim * weights.get("phonetic", 0.35)
        + vis_sim * weights.get("visual", 0.25)
    )

    return NameSimilarityResult(
        name_a=na,
        name_b=nb,
        edit_distance=levenshtein_distance(na, nb),
        string_similarity=str_sim,
        phonetic_similarity=phon_sim,
        visual_similarity=vis_sim,
        combined_score=combined,
    )


def is_confusingly_similar(
    name_a: str,
    name_b: str,
    threshold: float = 0.7,
) -> bool:
    """Check if two names are confusingly similar.

    Uses combined similarity score to determine if names might
    cause confusion in trademark context.

    Args:
        name_a: First name
        name_b: Second name
        threshold: Similarity threshold (default 0.7)

    Returns:
        True if names are confusingly similar
    """
    result = combined_similarity(name_a, name_b)
    return result.combined_score >= threshold
