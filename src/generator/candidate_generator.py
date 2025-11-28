"""Candidate name generator using C/V patterns."""

import random
from dataclasses import dataclass

from src.config import Settings


@dataclass
class Candidate:
    """A generated name candidate with metadata."""

    name: str
    pattern: str
    raw_name: str  # lowercase version before capitalization

    def __hash__(self) -> int:
        return hash(self.name.lower())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Candidate):
            return False
        return self.name.lower() == other.name.lower()


class CandidateGenerator:
    """Generator for neologism brand name candidates.

    Uses C/V (consonant/vowel) patterns to generate pronounceable
    invented words suitable for trademark registration.
    """

    def __init__(self, settings: Settings, seed: int | None = None):
        """Initialize generator with settings.

        Args:
            settings: Configuration settings
            seed: Optional random seed for reproducibility
        """
        self.settings = settings
        self.consonants = settings.consonants
        self.vowels = settings.vowels
        self.patterns = settings.patterns
        self.banned_substrings = [s.lower() for s in settings.banned_substrings]
        self.max_consecutive_consonants = settings.max_consecutive_consonants
        self.problematic_combinations = [c.lower() for c in settings.problematic_combinations]

        if seed is not None:
            random.seed(seed)

    def generate_from_pattern(self, pattern: str) -> str:
        """Generate a single word from a C/V pattern.

        Args:
            pattern: String of C (consonant) and V (vowel) characters

        Returns:
            Generated lowercase word
        """
        chars = []
        for symbol in pattern:
            if symbol == "C":
                chars.append(random.choice(self.consonants))
            elif symbol == "V":
                chars.append(random.choice(self.vowels))
            else:
                raise ValueError(f"Unknown pattern symbol: {symbol}")
        return "".join(chars)

    def passes_banned_filter(self, name: str) -> bool:
        """Check if name contains any banned substrings.

        Args:
            name: Lowercase name to check

        Returns:
            True if name passes (contains no banned substrings)
        """
        for banned in self.banned_substrings:
            if banned in name:
                return False
        return True

    def passes_pronunciation_filter(self, name: str) -> bool:
        """Check if name is pronounceable (no excessive consonant clusters).

        Args:
            name: Lowercase name to check

        Returns:
            True if name is pronounceable
        """
        consonant_set = set(self.consonants)
        consecutive_count = 0

        for char in name:
            if char in consonant_set:
                consecutive_count += 1
                if consecutive_count > self.max_consecutive_consonants:
                    return False
            else:
                consecutive_count = 0

        return True

    def passes_combination_filter(self, name: str) -> bool:
        """Check if name contains problematic letter combinations.

        Args:
            name: Lowercase name to check

        Returns:
            True if name passes (no problematic combinations)
        """
        for combo in self.problematic_combinations:
            if combo in name:
                return False
        return True

    def passes_all_filters(self, name: str) -> bool:
        """Check if name passes all filters.

        Args:
            name: Lowercase name to check

        Returns:
            True if name passes all filters
        """
        return (
            self.passes_banned_filter(name)
            and self.passes_pronunciation_filter(name)
            and self.passes_combination_filter(name)
        )

    def generate(self, count: int, max_attempts_multiplier: int = 50) -> set[Candidate]:
        """Generate unique name candidates.

        Args:
            count: Number of candidates to generate
            max_attempts_multiplier: Safety multiplier for max attempts

        Returns:
            Set of unique Candidate objects
        """
        candidates: set[Candidate] = set()
        attempts = 0
        max_attempts = count * max_attempts_multiplier

        while len(candidates) < count and attempts < max_attempts:
            pattern = random.choice(self.patterns)
            raw_name = self.generate_from_pattern(pattern)
            attempts += 1

            # Check length constraints
            if not (self.settings.min_length <= len(raw_name) <= self.settings.max_length):
                continue

            # Apply filters
            if not self.passes_all_filters(raw_name):
                continue

            # Capitalize for display
            capitalized = raw_name.capitalize()
            candidate = Candidate(name=capitalized, pattern=pattern, raw_name=raw_name)
            candidates.add(candidate)

        return candidates

    def generate_exhaustive(self, pattern: str) -> list[Candidate]:
        """Generate all possible combinations for a given pattern.

        Warning: This can generate millions of combinations for longer patterns.

        Args:
            pattern: C/V pattern to enumerate

        Returns:
            List of all valid candidates for the pattern
        """
        from itertools import product

        # Build character lists for each position
        char_options = []
        for symbol in pattern:
            if symbol == "C":
                char_options.append(list(self.consonants))
            elif symbol == "V":
                char_options.append(list(self.vowels))
            else:
                raise ValueError(f"Unknown pattern symbol: {symbol}")

        candidates = []
        for combo in product(*char_options):
            raw_name = "".join(combo)

            # Check length constraints
            if not (self.settings.min_length <= len(raw_name) <= self.settings.max_length):
                continue

            # Apply filters
            if not self.passes_all_filters(raw_name):
                continue

            capitalized = raw_name.capitalize()
            candidate = Candidate(name=capitalized, pattern=pattern, raw_name=raw_name)
            candidates.append(candidate)

        return candidates
