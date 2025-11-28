"""Heuristic scorer for candidate names (no LLM required)."""

from dataclasses import dataclass, field

from src.config import Settings
from src.generator.candidate_generator import Candidate


@dataclass
class HeuristicScore:
    """Heuristic scoring result for a candidate."""

    candidate: Candidate
    score: float  # 0-10
    bonuses: list[str] = field(default_factory=list)
    penalties: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Check if candidate passed minimum score threshold."""
        return self.score >= 5.0


class HeuristicScorer:
    """Heuristic scorer for evaluating name candidates without LLM.

    Applies rule-based scoring to quickly filter candidates before
    more expensive LLM evaluation.
    """

    def __init__(self, settings: Settings):
        """Initialize scorer with settings.

        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self.consonants = set(settings.consonants)
        self.vowels = set(settings.vowels)
        self.preferred_starts = set(settings.preferred_starts)

    def score_cv_alternation(self, name: str) -> tuple[float, str | None]:
        """Score based on consonant-vowel alternation quality.

        Names with good C-V alternation are more pronounceable.

        Returns:
            Tuple of (score_delta, reason or None)
        """
        name_lower = name.lower()
        alternations = 0
        prev_is_consonant = None

        for char in name_lower:
            is_consonant = char in self.consonants
            if prev_is_consonant is not None and is_consonant != prev_is_consonant:
                alternations += 1
            prev_is_consonant = is_consonant

        # Perfect alternation would be len-1 alternations
        max_alternations = len(name_lower) - 1
        if max_alternations == 0:
            return 0.0, None

        ratio = alternations / max_alternations
        if ratio >= 0.8:
            return 1.0, "Vynikající střídání C/V"
        elif ratio >= 0.6:
            return 0.5, "Dobré střídání C/V"
        elif ratio < 0.4:
            return -0.5, "Slabé střídání C/V"
        return 0.0, None

    def score_preferred_start(self, name: str) -> tuple[float, str | None]:
        """Score based on preferred starting letters.

        Returns:
            Tuple of (score_delta, reason or None)
        """
        if not name:
            return 0.0, None

        first_letter = name[0].lower()
        if first_letter in self.preferred_starts:
            return 0.5, f"Preferované začáteční písmeno '{first_letter.upper()}'"
        return 0.0, None

    def score_length(self, name: str) -> tuple[float, str | None]:
        """Score based on name length (optimal is 5-6 characters).

        Returns:
            Tuple of (score_delta, reason or None)
        """
        length = len(name)
        if 5 <= length <= 6:
            return 0.5, "Optimální délka (5-6 znaků)"
        elif length == 4 or length == 7:
            return 0.0, None
        elif length > 8:
            return -0.5, "Příliš dlouhý název"
        return 0.0, None

    def score_ending(self, name: str) -> tuple[float, str | None]:
        """Score based on name ending (vowel endings sound softer).

        Returns:
            Tuple of (score_delta, reason or None)
        """
        if not name:
            return 0.0, None

        last_letter = name[-1].lower()
        if last_letter in self.vowels:
            return 0.5, "Měkký konec (samohláska)"
        # Soft consonants at end
        if last_letter in {"n", "l", "r", "s"}:
            return 0.25, "Měkká koncová souhláska"
        return 0.0, None

    def score_double_letters(self, name: str) -> tuple[float, str | None]:
        """Penalize double letters which can look awkward.

        Returns:
            Tuple of (score_delta, reason or None)
        """
        name_lower = name.lower()
        for i in range(len(name_lower) - 1):
            if name_lower[i] == name_lower[i + 1]:
                return -0.5, f"Dvojité písmeno '{name_lower[i]}'"
        return 0.0, None

    def score_visual_balance(self, name: str) -> tuple[float, str | None]:
        """Score visual balance (mix of ascenders/descenders vs baseline).

        Returns:
            Tuple of (score_delta, reason or None)
        """
        name_lower = name.lower()
        ascenders = set("bdfhklt")  # letters with parts above baseline
        descenders = set("gjpqy")  # letters with parts below baseline

        has_ascender = any(c in ascenders for c in name_lower)
        has_descender = any(c in descenders for c in name_lower)

        # Too many descenders look unbalanced
        descender_count = sum(1 for c in name_lower if c in descenders)
        if descender_count > 2:
            return -0.5, "Příliš mnoho písmen s dolním tahem"

        # Good mix is positive
        if has_ascender and not has_descender:
            return 0.25, "Vyvážené vizuálně"

        return 0.0, None

    def score_memorability(self, name: str) -> tuple[float, str | None]:
        """Score potential memorability based on patterns.

        Returns:
            Tuple of (score_delta, reason or None)
        """
        name_lower = name.lower()

        # Repeated syllable patterns can be catchy
        if len(name_lower) >= 4:
            # Check for repeated 2-char patterns (like "nana", "lala")
            half = len(name_lower) // 2
            if name_lower[:half] == name_lower[half : half * 2]:
                return 0.5, "Opakující se vzor (zapamatovatelné)"

        # Names ending in 'o', 'a', 'i' often sound friendly
        if name_lower.endswith(("o", "a", "i")):
            return 0.25, "Přátelská koncovka"

        return 0.0, None

    def score(self, candidate: Candidate) -> HeuristicScore:
        """Calculate heuristic score for a candidate.

        Args:
            candidate: Candidate to score

        Returns:
            HeuristicScore with detailed breakdown
        """
        name = candidate.name
        base_score = 5.0  # Start at midpoint
        bonuses: list[str] = []
        penalties: list[str] = []

        # Apply all scoring functions
        scoring_functions = [
            self.score_cv_alternation,
            self.score_preferred_start,
            self.score_length,
            self.score_ending,
            self.score_double_letters,
            self.score_visual_balance,
            self.score_memorability,
        ]

        for score_func in scoring_functions:
            delta, reason = score_func(name)
            base_score += delta
            if reason:
                if delta > 0:
                    bonuses.append(reason)
                else:
                    penalties.append(reason)

        # Clamp to 0-10 range
        final_score = max(0.0, min(10.0, base_score))

        return HeuristicScore(
            candidate=candidate,
            score=final_score,
            bonuses=bonuses,
            penalties=penalties,
        )

    def score_batch(self, candidates: list[Candidate]) -> list[HeuristicScore]:
        """Score a batch of candidates.

        Args:
            candidates: List of candidates to score

        Returns:
            List of HeuristicScore objects, sorted by score descending
        """
        scores = [self.score(c) for c in candidates]
        return sorted(scores, key=lambda s: s.score, reverse=True)

    def filter_by_score(
        self, candidates: list[Candidate], min_score: float = 5.0
    ) -> list[HeuristicScore]:
        """Score and filter candidates by minimum score.

        Args:
            candidates: List of candidates to score
            min_score: Minimum score threshold

        Returns:
            List of HeuristicScore objects above threshold, sorted by score
        """
        scores = self.score_batch(candidates)
        return [s for s in scores if s.score >= min_score]
