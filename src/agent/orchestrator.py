"""Autonomous name generation orchestrator (agent)."""

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from src.config import Settings
from src.db.repository import CandidateRepository, StoredCandidate
from src.generator.candidate_generator import Candidate, CandidateGenerator
from src.llm import LLMEvaluation, LLMScorer
from src.scoring.heuristic_scorer import HeuristicScorer


class AgentState(Enum):
    """States of the name generation agent."""

    INIT = "init"
    GENERATE = "generate"
    HEURISTIC_SCORE = "heuristic_score"
    LLM_SCORE = "llm_score"
    EVALUATE = "evaluate"
    ADJUST = "adjust"
    DONE = "done"


@dataclass
class GenerationStats:
    """Statistics from a generation run."""

    candidates_generated: int
    passed_heuristic: int
    llm_evaluated: int
    excellent_count: int
    iteration: int


class NameGenAgent:
    """Autonomous agent for generating and evaluating brand names.

    Implements a state machine that:
    1. Generates candidate names using C/V patterns
    2. Filters with heuristic scoring
    3. Evaluates promising candidates with LLM
    4. Adjusts parameters based on results
    5. Repeats until target is reached
    """

    def __init__(
        self,
        settings: Settings,
        use_llm: bool = True,
        progress_callback: Callable[[str, GenerationStats | None], None] | None = None,
    ):
        """Initialize the agent.

        Args:
            settings: Configuration settings
            use_llm: Whether to use LLM scoring (set False for offline mode)
            progress_callback: Optional callback for progress updates
        """
        self.settings = settings
        self.use_llm = use_llm
        self.progress_callback = progress_callback

        self.generator = CandidateGenerator(settings)
        self.heuristic_scorer = HeuristicScorer(settings)
        self.repository = CandidateRepository(settings)

        if use_llm:
            self.llm_scorer = LLMScorer(settings)
        else:
            self.llm_scorer = None

        self.state = AgentState.INIT
        self.iteration = 0
        self.target_excellent = 10  # Default target for excellent candidates

    def _report_progress(self, message: str, stats: GenerationStats | None = None) -> None:
        """Report progress via callback if available."""
        if self.progress_callback:
            self.progress_callback(message, stats)

    def _generate_batch(self, count: int) -> set[Candidate]:
        """Generate a batch of candidates, excluding already stored names."""
        self._report_progress(f"Generuji {count} kandidátů...")
        candidates = self.generator.generate(count * 2)  # Generate extra for filtering

        # Filter out already stored candidates
        new_candidates = set()
        for candidate in candidates:
            if not self.repository.exists(candidate.name):
                new_candidates.add(candidate)
                if len(new_candidates) >= count:
                    break

        return new_candidates

    def _score_heuristically(self, candidates: set[Candidate]) -> list[Candidate]:
        """Score candidates heuristically and filter.

        Returns candidates that passed the heuristic threshold.
        """
        self._report_progress(f"Heuristické hodnocení {len(candidates)} kandidátů...")

        candidate_list = list(candidates)
        scored = self.heuristic_scorer.filter_by_score(
            candidate_list, min_score=self.settings.scoring.min_acceptable_score
        )

        # Save all candidates with heuristic scores
        for hs in self.heuristic_scorer.score_batch(candidate_list):
            self.repository.save_candidate(hs.candidate, heuristic=hs)

        # Return only those that passed
        return [hs.candidate for hs in scored]

    def _score_with_llm(self, candidates: list[Candidate]) -> list[LLMEvaluation]:
        """Score candidates with LLM."""
        if not self.llm_scorer or not candidates:
            return []

        self._report_progress(f"LLM hodnocení {len(candidates)} kandidátů...")
        evaluations = self.llm_scorer.score_all(candidates)

        # Update repository with LLM scores
        for eval_result in evaluations:
            # Find matching candidate
            matching = next((c for c in candidates if c.name == eval_result.name), None)
            if matching:
                self.repository.update_candidate(matching.name, llm_eval=eval_result)

        return evaluations

    def _evaluate_results(self, evaluations: list[LLMEvaluation]) -> int:
        """Evaluate LLM results and count excellent candidates.

        Returns count of excellent (score >= 8) candidates.
        """
        excellent = [e for e in evaluations if e.score >= self.settings.scoring.excellent_score]
        return len(excellent)

    def run_iteration(self, batch_size: int = 100) -> GenerationStats:
        """Run a single generation iteration.

        Args:
            batch_size: Number of candidates to generate

        Returns:
            Statistics from this iteration
        """
        self.iteration += 1
        self._report_progress(f"\n=== Iterace {self.iteration} ===")

        # Generate
        self.state = AgentState.GENERATE
        candidates = self._generate_batch(batch_size)
        generated_count = len(candidates)

        # Heuristic scoring
        self.state = AgentState.HEURISTIC_SCORE
        passed_heuristic = self._score_heuristically(candidates)
        passed_count = len(passed_heuristic)

        # LLM scoring (if enabled)
        llm_evaluated = 0
        excellent_count = 0
        if self.use_llm and passed_heuristic:
            self.state = AgentState.LLM_SCORE
            evaluations = self._score_with_llm(passed_heuristic)
            llm_evaluated = len(evaluations)

            self.state = AgentState.EVALUATE
            excellent_count = self._evaluate_results(evaluations)

        stats = GenerationStats(
            candidates_generated=generated_count,
            passed_heuristic=passed_count,
            llm_evaluated=llm_evaluated,
            excellent_count=excellent_count,
            iteration=self.iteration,
        )

        self._report_progress(
            f"Iterace {self.iteration}: vygenerováno={generated_count}, "
            f"prošlo heuristikou={passed_count}, LLM hodnoceno={llm_evaluated}, "
            f"vynikajících={excellent_count}",
            stats,
        )

        return stats

    def run(
        self,
        target_excellent: int = 10,
        max_iterations: int = 10,
        batch_size: int = 100,
    ) -> list[StoredCandidate]:
        """Run the agent until target is reached or max iterations.

        Args:
            target_excellent: Target number of excellent candidates (score >= 8)
            max_iterations: Maximum number of iterations
            batch_size: Candidates per iteration

        Returns:
            List of excellent candidates found
        """
        self.target_excellent = target_excellent
        self.state = AgentState.INIT

        total_excellent = self.repository.count_above_score(
            self.settings.scoring.excellent_score
        )

        self._report_progress(
            f"Cíl: {target_excellent} vynikajících kandidátů. "
            f"Aktuálně v databázi: {total_excellent}"
        )

        while total_excellent < target_excellent and self.iteration < max_iterations:
            self.run_iteration(batch_size)

            total_excellent = self.repository.count_above_score(
                self.settings.scoring.excellent_score
            )

            self._report_progress(
                f"Celkem vynikajících: {total_excellent}/{target_excellent}"
            )

        self.state = AgentState.DONE

        if total_excellent >= target_excellent:
            self._report_progress(f"Cíl dosažen! Nalezeno {total_excellent} vynikajících kandidátů.")
        else:
            self._report_progress(
                f"Dosažen limit iterací. Nalezeno {total_excellent} vynikajících kandidátů."
            )

        return self.repository.get_top(limit=target_excellent, min_score=8)

    def run_offline(self, count: int = 500) -> list[StoredCandidate]:
        """Run in offline mode (no LLM, only heuristic scoring).

        Args:
            count: Number of candidates to generate

        Returns:
            List of candidates sorted by heuristic score
        """
        self._report_progress(f"Offline režim: generuji {count} kandidátů...")

        candidates = self._generate_batch(count)
        self._score_heuristically(candidates)

        # Get all new candidates sorted by heuristic score
        top = []
        with self.repository._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM candidates
                WHERE heuristic_score IS NOT NULL
                ORDER BY heuristic_score DESC
                LIMIT ?
                """,
                (count,),
            )
            for row in cursor.fetchall():
                top.append(self.repository._row_to_stored_candidate(row))

        self._report_progress(f"Vygenerováno {len(top)} kandidátů s heuristickým skóre.")
        return top
