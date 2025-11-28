"""Advanced phonetic models for brand name generation and scoring.

This module implements mathematical models for evaluating and generating
pronounceable neologisms:
- Markov chains (bigram/trigram)
- Sonority Sequencing Principle (SSP)
- Shannon entropy
- Phonotactic probability
- Zipf's law weighting
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field


# =============================================================================
# PHONETIC DATA - Language-agnostic phonetic properties
# =============================================================================

# Sonority scale (1-10) based on phonetic research
# Higher values = more sonorous (vowel-like)
SONORITY_SCALE: dict[str, int] = {
    # Plosives/Stops (least sonorous)
    "p": 1, "b": 1, "t": 1, "d": 1, "k": 1, "g": 1,
    # Affricates
    "c": 2,
    # Fricatives
    "f": 3, "v": 3, "s": 3, "z": 3, "h": 3,
    # Nasals
    "m": 5, "n": 5,
    # Liquids
    "l": 6, "r": 6,
    # Glides/Semivowels
    "j": 7, "w": 7,
    # Vowels (most sonorous)
    "a": 10, "e": 9, "i": 9, "o": 10, "u": 9,
}

# Phoneme frequency in natural languages (approximation based on multiple languages)
# Used for Zipf's law weighting and entropy calculations
PHONEME_FREQUENCY: dict[str, float] = {
    # Vowels (typically ~40% of text)
    "a": 0.082, "e": 0.127, "i": 0.070, "o": 0.075, "u": 0.028,
    # Common consonants
    "n": 0.067, "r": 0.060, "s": 0.063, "t": 0.091, "l": 0.040,
    "d": 0.043, "c": 0.028, "m": 0.024, "p": 0.019, "b": 0.015,
    # Less common consonants
    "f": 0.022, "g": 0.020, "h": 0.061, "k": 0.008, "v": 0.010,
    "z": 0.007, "j": 0.002,
}

# Bigram frequencies from natural language corpora (normalized)
# These represent common letter pair transitions
BIGRAM_FREQUENCIES: dict[str, float] = {
    # Common CV transitions
    "ta": 0.015, "te": 0.018, "ti": 0.012, "to": 0.014, "tu": 0.006,
    "na": 0.014, "ne": 0.016, "ni": 0.010, "no": 0.012, "nu": 0.004,
    "ra": 0.012, "re": 0.018, "ri": 0.011, "ro": 0.010, "ru": 0.005,
    "la": 0.010, "le": 0.014, "li": 0.009, "lo": 0.008, "lu": 0.004,
    "sa": 0.008, "se": 0.012, "si": 0.008, "so": 0.007, "su": 0.005,
    "ma": 0.009, "me": 0.011, "mi": 0.007, "mo": 0.006, "mu": 0.003,
    "ca": 0.007, "ce": 0.006, "ci": 0.005, "co": 0.008, "cu": 0.004,
    "da": 0.006, "de": 0.010, "di": 0.007, "do": 0.005, "du": 0.003,
    "pa": 0.005, "pe": 0.007, "pi": 0.004, "po": 0.006, "pu": 0.002,
    "ba": 0.004, "be": 0.006, "bi": 0.003, "bo": 0.004, "bu": 0.002,
    "fa": 0.003, "fe": 0.004, "fi": 0.003, "fo": 0.003, "fu": 0.002,
    "va": 0.004, "ve": 0.006, "vi": 0.005, "vo": 0.003, "vu": 0.001,
    "ga": 0.003, "ge": 0.004, "gi": 0.003, "go": 0.003, "gu": 0.002,
    "ha": 0.004, "he": 0.008, "hi": 0.005, "ho": 0.004, "hu": 0.002,
    "ka": 0.002, "ke": 0.003, "ki": 0.002, "ko": 0.002, "ku": 0.001,
    "za": 0.001, "ze": 0.002, "zi": 0.001, "zo": 0.001, "zu": 0.001,
    "ja": 0.001, "je": 0.002, "ji": 0.001, "jo": 0.001, "ju": 0.001,
    # Common VC transitions
    "at": 0.012, "et": 0.010, "it": 0.008, "ot": 0.006, "ut": 0.004,
    "an": 0.018, "en": 0.020, "in": 0.016, "on": 0.014, "un": 0.006,
    "ar": 0.014, "er": 0.022, "ir": 0.008, "or": 0.012, "ur": 0.006,
    "al": 0.012, "el": 0.010, "il": 0.006, "ol": 0.008, "ul": 0.004,
    "as": 0.010, "es": 0.014, "is": 0.010, "os": 0.008, "us": 0.006,
    "am": 0.006, "em": 0.008, "im": 0.004, "om": 0.006, "um": 0.004,
    "ad": 0.004, "ed": 0.008, "id": 0.003, "od": 0.003, "ud": 0.002,
    "ap": 0.003, "ep": 0.004, "ip": 0.002, "op": 0.003, "up": 0.002,
    "ab": 0.002, "eb": 0.002, "ib": 0.002, "ob": 0.002, "ub": 0.001,
    "ac": 0.004, "ec": 0.005, "ic": 0.006, "oc": 0.003, "uc": 0.002,
    "af": 0.002, "ef": 0.003, "if": 0.002, "of": 0.004, "uf": 0.001,
    "av": 0.003, "ev": 0.004, "iv": 0.003, "ov": 0.004, "uv": 0.001,
    "ag": 0.002, "eg": 0.003, "ig": 0.002, "og": 0.002, "ug": 0.001,
    "ah": 0.002, "eh": 0.002, "ih": 0.001, "oh": 0.002, "uh": 0.001,
    "ak": 0.002, "ek": 0.002, "ik": 0.002, "ok": 0.002, "uk": 0.001,
    "az": 0.001, "ez": 0.002, "iz": 0.001, "oz": 0.001, "uz": 0.001,
    # VV transitions (diphthongs)
    "ai": 0.004, "au": 0.003, "ea": 0.006, "ei": 0.004, "eu": 0.002,
    "ia": 0.005, "ie": 0.006, "io": 0.005, "iu": 0.002, "oa": 0.003,
    "oe": 0.002, "oi": 0.002, "ou": 0.006, "ua": 0.003, "ue": 0.003,
    "ui": 0.002, "uo": 0.002,
    # CC transitions (consonant clusters)
    "st": 0.008, "tr": 0.006, "pr": 0.004, "br": 0.003, "cr": 0.003,
    "gr": 0.003, "fr": 0.002, "dr": 0.002, "pl": 0.003, "bl": 0.002,
    "cl": 0.002, "fl": 0.002, "gl": 0.001, "sl": 0.002, "sp": 0.003,
    "sc": 0.002, "sk": 0.002, "sm": 0.002, "sn": 0.001, "sw": 0.001,
    "nt": 0.008, "nd": 0.006, "ng": 0.005, "nk": 0.002, "ns": 0.004,
    "nc": 0.003, "mp": 0.003, "mb": 0.002, "mn": 0.001, "lt": 0.003,
    "ld": 0.003, "lk": 0.001, "ls": 0.002, "lv": 0.001, "rt": 0.004,
    "rd": 0.003, "rk": 0.002, "rs": 0.003, "rv": 0.001, "rn": 0.002,
    "rm": 0.002, "rl": 0.001, "rp": 0.001, "rb": 0.001, "rc": 0.002,
    "ct": 0.003, "pt": 0.002, "ft": 0.002, "ks": 0.002, "ps": 0.001,
}

# Default probability for unseen bigrams (smoothing)
DEFAULT_BIGRAM_PROB = 0.0001


# =============================================================================
# MARKOV CHAIN MODEL
# =============================================================================

@dataclass
class MarkovModel:
    """Markov chain model for phoneme sequence probability.

    Uses bigram and trigram probabilities to estimate how "natural"
    a word sounds based on character transition patterns.
    """

    bigram_probs: dict[str, float] = field(default_factory=lambda: BIGRAM_FREQUENCIES.copy())
    trigram_probs: dict[str, float] = field(default_factory=dict)
    smoothing_factor: float = 0.0001  # Laplace smoothing for unseen n-grams

    def get_bigram_probability(self, bigram: str) -> float:
        """Get probability of a bigram (2-character sequence).

        Args:
            bigram: Two-character string

        Returns:
            Probability value (0-1)
        """
        bigram = bigram.lower()
        return self.bigram_probs.get(bigram, self.smoothing_factor)

    def get_trigram_probability(self, trigram: str) -> float:
        """Get probability of a trigram (3-character sequence).

        Args:
            trigram: Three-character string

        Returns:
            Probability value (0-1)
        """
        trigram = trigram.lower()
        if trigram in self.trigram_probs:
            return self.trigram_probs[trigram]

        # Estimate from bigrams if trigram not available
        if len(trigram) == 3:
            p1 = self.get_bigram_probability(trigram[:2])
            p2 = self.get_bigram_probability(trigram[1:])
            return math.sqrt(p1 * p2)  # Geometric mean

        return self.smoothing_factor

    def calculate_word_probability(self, word: str) -> float:
        """Calculate overall probability of a word based on n-gram model.

        Uses chain rule: P(w1w2w3...wn) = P(w1) * P(w2|w1) * P(w3|w1w2) * ...

        Args:
            word: Word to evaluate

        Returns:
            Log probability (more negative = less likely)
        """
        word = word.lower()
        if len(word) < 2:
            return 0.0

        log_prob = 0.0

        # Bigram probabilities
        for i in range(len(word) - 1):
            bigram = word[i:i+2]
            prob = self.get_bigram_probability(bigram)
            log_prob += math.log(prob + 1e-10)

        return log_prob

    def score_naturalness(self, word: str) -> float:
        """Score how natural a word sounds (0-10 scale).

        Args:
            word: Word to evaluate

        Returns:
            Score from 0 (unnatural) to 10 (very natural)
        """
        log_prob = self.calculate_word_probability(word)

        # Normalize by word length
        if len(word) > 1:
            normalized = log_prob / (len(word) - 1)
        else:
            normalized = log_prob

        # Convert to 0-10 scale
        # Typical range is around -8 to -4 for normalized log prob
        # Map -8 -> 0, -4 -> 10
        score = (normalized + 8) * 2.5
        return max(0.0, min(10.0, score))


# =============================================================================
# SONORITY SEQUENCING PRINCIPLE (SSP)
# =============================================================================

@dataclass
class SonorityModel:
    """Sonority Sequencing Principle model for syllable structure evaluation.

    The SSP states that syllables typically rise in sonority to the nucleus
    (vowel) and fall afterward. Violations indicate harder pronunciation.
    """

    sonority_scale: dict[str, int] = field(default_factory=lambda: SONORITY_SCALE.copy())

    def get_sonority(self, char: str) -> int:
        """Get sonority value for a character.

        Args:
            char: Single character

        Returns:
            Sonority value (1-10)
        """
        return self.sonority_scale.get(char.lower(), 5)

    def get_sonority_profile(self, word: str) -> list[int]:
        """Get sonority profile (list of sonority values) for a word.

        Args:
            word: Word to analyze

        Returns:
            List of sonority values
        """
        return [self.get_sonority(c) for c in word.lower()]

    def count_ssp_violations(self, word: str) -> int:
        """Count violations of the Sonority Sequencing Principle.

        A violation occurs when:
        - Sonority doesn't rise toward a vowel (onset)
        - Sonority doesn't fall after a vowel (coda)

        Args:
            word: Word to analyze

        Returns:
            Number of violations
        """
        profile = self.get_sonority_profile(word)
        if len(profile) < 2:
            return 0

        violations = 0
        vowels = set("aeiou")
        word_lower = word.lower()

        # Find vowel positions (syllable nuclei)
        vowel_positions = [i for i, c in enumerate(word_lower) if c in vowels]

        if not vowel_positions:
            return len(word) - 1  # No vowels = big violation

        # Check onset (before each vowel) - should rise
        for vp in vowel_positions:
            for i in range(max(0, vp - 2), vp):
                if i + 1 < len(profile):
                    if profile[i] >= profile[i + 1] and word_lower[i] not in vowels:
                        violations += 1

        # Check coda (after each vowel) - should fall
        for vp in vowel_positions:
            for i in range(vp, min(len(profile) - 1, vp + 2)):
                if profile[i] < profile[i + 1] and word_lower[i + 1] not in vowels:
                    violations += 1

        return violations

    def calculate_sonority_smoothness(self, word: str) -> float:
        """Calculate how smooth the sonority contour is.

        Smoother contours = easier pronunciation.

        Args:
            word: Word to analyze

        Returns:
            Smoothness score (0-1, higher is smoother)
        """
        profile = self.get_sonority_profile(word)
        if len(profile) < 2:
            return 1.0

        # Calculate sum of absolute differences
        total_diff = sum(abs(profile[i] - profile[i-1]) for i in range(1, len(profile)))
        max_possible_diff = 9 * (len(profile) - 1)  # Max diff is 9 per transition

        # Normalize (lower diff = smoother)
        return 1.0 - (total_diff / max_possible_diff)

    def score_sonority(self, word: str) -> float:
        """Score word based on sonority principles (0-10 scale).

        Args:
            word: Word to evaluate

        Returns:
            Score from 0 (hard to pronounce) to 10 (easy to pronounce)
        """
        violations = self.count_ssp_violations(word)
        smoothness = self.calculate_sonority_smoothness(word)

        # Start with base score
        base_score = 10.0

        # Penalize violations heavily
        base_score -= violations * 1.5

        # Bonus for smoothness
        base_score += (smoothness - 0.5) * 2

        return max(0.0, min(10.0, base_score))


# =============================================================================
# SHANNON ENTROPY
# =============================================================================

@dataclass
class EntropyModel:
    """Shannon entropy model for measuring information content and predictability.

    Lower entropy (more predictable patterns) often correlates with
    easier pronunciation and memorability.
    """

    phoneme_freq: dict[str, float] = field(default_factory=lambda: PHONEME_FREQUENCY.copy())

    def calculate_unigram_entropy(self, word: str) -> float:
        """Calculate Shannon entropy based on character distribution.

        H = -Σ p(x) * log2(p(x))

        Args:
            word: Word to analyze

        Returns:
            Entropy value in bits
        """
        word = word.lower()
        if not word:
            return 0.0

        # Count character frequencies in word
        char_counts: dict[str, int] = defaultdict(int)
        for c in word:
            char_counts[c] += 1

        # Calculate entropy
        entropy = 0.0
        n = len(word)
        for count in char_counts.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def calculate_cross_entropy(self, word: str) -> float:
        """Calculate cross-entropy against natural language distribution.

        Lower cross-entropy = word follows natural language patterns.

        Args:
            word: Word to analyze

        Returns:
            Cross-entropy value
        """
        word = word.lower()
        if not word:
            return 0.0

        cross_entropy = 0.0
        for c in word:
            p = self.phoneme_freq.get(c, 0.001)  # Smoothing for unknown chars
            cross_entropy -= math.log2(p)

        return cross_entropy / len(word)  # Normalize by length

    def score_entropy(self, word: str) -> float:
        """Score word based on entropy (0-10 scale).

        Optimal words have moderate entropy - not too random, not too repetitive.

        Args:
            word: Word to evaluate

        Returns:
            Score from 0 to 10
        """
        unigram_entropy = self.calculate_unigram_entropy(word)
        cross_entropy = self.calculate_cross_entropy(word)

        # Ideal unigram entropy for brand names: 2.0-3.5 bits
        # Too low = repetitive (boring), too high = random (hard to remember)
        if 2.0 <= unigram_entropy <= 3.5:
            entropy_score = 10.0
        elif unigram_entropy < 2.0:
            entropy_score = 5.0 + (unigram_entropy / 2.0) * 5.0
        else:
            entropy_score = max(0, 10.0 - (unigram_entropy - 3.5) * 2.0)

        # Cross-entropy bonus for matching natural language
        # Lower cross-entropy = more natural
        if cross_entropy < 3.0:
            natural_bonus = 1.0
        elif cross_entropy < 4.0:
            natural_bonus = 0.5
        else:
            natural_bonus = 0.0

        return max(0.0, min(10.0, entropy_score + natural_bonus))


# =============================================================================
# PHONOTACTIC PROBABILITY
# =============================================================================

@dataclass
class PhonotacticModel:
    """Phonotactic probability model for evaluating sound sequence legality.

    Phonotactics describes the rules for valid sound combinations in a language.
    """

    position_probs: dict[str, dict[str, float]] = field(default_factory=dict)
    bigram_probs: dict[str, float] = field(default_factory=lambda: BIGRAM_FREQUENCIES.copy())

    def __post_init__(self) -> None:
        """Initialize position-specific probabilities."""
        # Onset (word-initial) probabilities
        self.position_probs["onset"] = {
            "b": 0.08, "c": 0.06, "d": 0.07, "f": 0.05, "g": 0.04,
            "h": 0.04, "j": 0.02, "k": 0.03, "l": 0.06, "m": 0.08,
            "n": 0.07, "p": 0.06, "r": 0.08, "s": 0.10, "t": 0.09,
            "v": 0.04, "z": 0.02,
        }
        # Coda (word-final) probabilities
        self.position_probs["coda"] = {
            "a": 0.12, "e": 0.15, "i": 0.10, "o": 0.12, "u": 0.06,
            "n": 0.10, "r": 0.08, "s": 0.08, "t": 0.06, "l": 0.05,
            "d": 0.03, "k": 0.02, "m": 0.02, "p": 0.01,
        }

    def get_onset_probability(self, char: str) -> float:
        """Get probability of character appearing at word start."""
        return self.position_probs.get("onset", {}).get(char.lower(), 0.01)

    def get_coda_probability(self, char: str) -> float:
        """Get probability of character appearing at word end."""
        return self.position_probs.get("coda", {}).get(char.lower(), 0.01)

    def calculate_positional_probability(self, word: str) -> float:
        """Calculate probability based on character positions.

        Args:
            word: Word to analyze

        Returns:
            Log probability
        """
        if not word:
            return 0.0

        word = word.lower()
        log_prob = 0.0

        # Onset probability
        log_prob += math.log(self.get_onset_probability(word[0]) + 1e-10)

        # Coda probability
        log_prob += math.log(self.get_coda_probability(word[-1]) + 1e-10)

        return log_prob

    def calculate_biphone_probability(self, word: str) -> float:
        """Calculate biphone (bigram) probability.

        Args:
            word: Word to analyze

        Returns:
            Average log probability of bigrams
        """
        if len(word) < 2:
            return 0.0

        word = word.lower()
        log_prob = 0.0

        for i in range(len(word) - 1):
            bigram = word[i:i+2]
            prob = self.bigram_probs.get(bigram, DEFAULT_BIGRAM_PROB)
            log_prob += math.log(prob + 1e-10)

        return log_prob / (len(word) - 1)

    def score_phonotactics(self, word: str) -> float:
        """Score word based on phonotactic probability (0-10 scale).

        Args:
            word: Word to evaluate

        Returns:
            Score from 0 to 10
        """
        pos_prob = self.calculate_positional_probability(word)
        biphone_prob = self.calculate_biphone_probability(word)

        # Combine scores
        combined = (pos_prob + biphone_prob) / 2

        # Map to 0-10 scale (typical range: -10 to -3)
        score = (combined + 10) * (10 / 7)

        return max(0.0, min(10.0, score))


# =============================================================================
# ZIPF'S LAW WEIGHTING
# =============================================================================

@dataclass
class ZipfModel:
    """Zipf's law model for phoneme frequency weighting.

    Zipf's law states that frequency is inversely proportional to rank.
    Words using common phonemes are generally easier to pronounce.
    """

    phoneme_freq: dict[str, float] = field(default_factory=lambda: PHONEME_FREQUENCY.copy())

    def get_phoneme_rank(self, char: str) -> int:
        """Get rank of phoneme by frequency (1 = most common)."""
        sorted_phonemes = sorted(self.phoneme_freq.items(), key=lambda x: -x[1])
        for i, (phoneme, _) in enumerate(sorted_phonemes, 1):
            if phoneme == char.lower():
                return i
        return len(sorted_phonemes) + 1

    def calculate_zipf_score(self, word: str) -> float:
        """Calculate Zipf-weighted score for word.

        Words using more common phonemes score higher.

        Args:
            word: Word to analyze

        Returns:
            Zipf score (higher = uses more common phonemes)
        """
        if not word:
            return 0.0

        word = word.lower()
        total_freq = 0.0

        for char in word:
            total_freq += self.phoneme_freq.get(char, 0.001)

        return total_freq / len(word)

    def calculate_rank_deviation(self, word: str) -> float:
        """Calculate how much word deviates from expected Zipf distribution.

        Args:
            word: Word to analyze

        Returns:
            Deviation score (lower = closer to natural distribution)
        """
        if not word:
            return 0.0

        word = word.lower()
        ranks = [self.get_phoneme_rank(c) for c in word]
        avg_rank = sum(ranks) / len(ranks)

        # Optimal average rank is around 10-15 (common but not too common)
        deviation = abs(avg_rank - 12)

        return deviation

    def score_zipf(self, word: str) -> float:
        """Score word based on Zipf's law (0-10 scale).

        Args:
            word: Word to evaluate

        Returns:
            Score from 0 to 10
        """
        zipf_score = self.calculate_zipf_score(word)
        deviation = self.calculate_rank_deviation(word)

        # Higher frequency = better base score
        base_score = zipf_score * 100  # Scale up

        # Penalize extreme deviations
        penalty = deviation * 0.3

        final_score = base_score - penalty

        # Normalize to 0-10
        return max(0.0, min(10.0, final_score))


# =============================================================================
# LEVENSHTEIN DISTANCE
# =============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein (edit) distance between two strings.

    The minimum number of single-character edits (insertions, deletions,
    or substitutions) required to change one string into the other.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance (integer)
    """
    s1, s2 = s1.lower(), s2.lower()

    if len(s1) < len(s2):
        s1, s2 = s2, s1

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def normalized_levenshtein(s1: str, s2: str) -> float:
    """Calculate normalized Levenshtein distance (0-1 scale).

    Args:
        s1: First string
        s2: Second string

    Returns:
        Normalized distance (0 = identical, 1 = completely different)
    """
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    return levenshtein_distance(s1, s2) / max_len


@dataclass
class UniquenessModel:
    """Model for evaluating uniqueness against existing brands/words."""

    known_brands: list[str] = field(default_factory=list)
    common_words: list[str] = field(default_factory=list)
    min_distance_threshold: int = 3

    def __post_init__(self) -> None:
        """Initialize with common brand-related words to avoid."""
        if not self.common_words:
            self.common_words = [
                # Common tech brands (avoid similarity)
                "google", "apple", "amazon", "microsoft", "meta",
                "adobe", "cisco", "oracle", "intel", "nvidia",
                # Common words
                "hello", "world", "super", "ultra", "mega",
                "power", "smart", "easy", "fast", "best",
                # Document-related (from PRD)
                "document", "invoice", "contract", "archive", "storage",
            ]

    def find_closest_match(self, word: str) -> tuple[str, int]:
        """Find the closest matching known word.

        Args:
            word: Word to check

        Returns:
            Tuple of (closest_word, distance)
        """
        all_words = self.known_brands + self.common_words
        if not all_words:
            return ("", 999)

        closest = ""
        min_dist = 999

        for known in all_words:
            dist = levenshtein_distance(word, known)
            if dist < min_dist:
                min_dist = dist
                closest = known

        return (closest, min_dist)

    def is_unique(self, word: str) -> bool:
        """Check if word is sufficiently unique.

        Args:
            word: Word to check

        Returns:
            True if word is unique enough
        """
        _, distance = self.find_closest_match(word)
        return distance >= self.min_distance_threshold

    def score_uniqueness(self, word: str) -> float:
        """Score word uniqueness (0-10 scale).

        Args:
            word: Word to evaluate

        Returns:
            Score from 0 (too similar) to 10 (very unique)
        """
        closest, distance = self.find_closest_match(word)

        # Score based on distance
        if distance >= 5:
            return 10.0
        elif distance >= self.min_distance_threshold:
            return 7.0 + (distance - self.min_distance_threshold) * 1.5
        else:
            return distance * (7.0 / self.min_distance_threshold)


# =============================================================================
# COMBINED SCORER
# =============================================================================

@dataclass
class PhoneticScoreResult:
    """Result from combined phonetic scoring."""

    word: str
    total_score: float
    markov_score: float
    sonority_score: float
    entropy_score: float
    phonotactic_score: float
    zipf_score: float
    uniqueness_score: float
    details: dict[str, str] = field(default_factory=dict)

    @property
    def is_excellent(self) -> bool:
        """Check if word scores excellent (>= 8)."""
        return self.total_score >= 8.0


class CombinedPhoneticScorer:
    """Combined scorer using all phonetic models.

    Weights can be adjusted to prioritize different aspects:
    - markov: How natural the letter sequences sound
    - sonority: How easy to pronounce (syllable structure)
    - entropy: Balance between predictability and variety
    - phonotactic: Adherence to language sound rules
    - zipf: Use of common vs rare phonemes
    - uniqueness: Distinctiveness from existing words
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        known_brands: list[str] | None = None,
    ):
        """Initialize combined scorer.

        Args:
            weights: Custom weights for each model (should sum to 1.0)
            known_brands: List of existing brands to check uniqueness against
        """
        self.markov = MarkovModel()
        self.sonority = SonorityModel()
        self.entropy = EntropyModel()
        self.phonotactic = PhonotacticModel()
        self.zipf = ZipfModel()
        self.uniqueness = UniquenessModel(known_brands=known_brands or [])

        # Default weights (sum to 1.0)
        self.weights = weights or {
            "markov": 0.20,
            "sonority": 0.20,
            "entropy": 0.10,
            "phonotactic": 0.20,
            "zipf": 0.10,
            "uniqueness": 0.20,
        }

    def score(self, word: str) -> PhoneticScoreResult:
        """Score a word using all phonetic models.

        Args:
            word: Word to evaluate

        Returns:
            PhoneticScoreResult with all scores
        """
        # Calculate individual scores
        markov_score = self.markov.score_naturalness(word)
        sonority_score = self.sonority.score_sonority(word)
        entropy_score = self.entropy.score_entropy(word)
        phonotactic_score = self.phonotactic.score_phonotactics(word)
        zipf_score = self.zipf.score_zipf(word)
        uniqueness_score = self.uniqueness.score_uniqueness(word)

        # Calculate weighted total
        total_score = (
            markov_score * self.weights["markov"]
            + sonority_score * self.weights["sonority"]
            + entropy_score * self.weights["entropy"]
            + phonotactic_score * self.weights["phonotactic"]
            + zipf_score * self.weights["zipf"]
            + uniqueness_score * self.weights["uniqueness"]
        )

        # Collect details
        details = {
            "markov": f"Naturalness: {markov_score:.1f}/10",
            "sonority": f"Sonority (SSP): {sonority_score:.1f}/10",
            "entropy": f"Entropy: {entropy_score:.1f}/10",
            "phonotactic": f"Phonotactics: {phonotactic_score:.1f}/10",
            "zipf": f"Zipf frequency: {zipf_score:.1f}/10",
            "uniqueness": f"Uniqueness: {uniqueness_score:.1f}/10",
        }

        return PhoneticScoreResult(
            word=word,
            total_score=total_score,
            markov_score=markov_score,
            sonority_score=sonority_score,
            entropy_score=entropy_score,
            phonotactic_score=phonotactic_score,
            zipf_score=zipf_score,
            uniqueness_score=uniqueness_score,
            details=details,
        )

    def score_batch(self, words: list[str]) -> list[PhoneticScoreResult]:
        """Score multiple words.

        Args:
            words: List of words to evaluate

        Returns:
            List of results sorted by total score (descending)
        """
        results = [self.score(word) for word in words]
        return sorted(results, key=lambda r: r.total_score, reverse=True)

    def filter_excellent(self, words: list[str], min_score: float = 8.0) -> list[PhoneticScoreResult]:
        """Filter words to only those with excellent scores.

        Args:
            words: List of words to evaluate
            min_score: Minimum score threshold

        Returns:
            List of results above threshold
        """
        results = self.score_batch(words)
        return [r for r in results if r.total_score >= min_score]
