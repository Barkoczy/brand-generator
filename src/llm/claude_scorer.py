"""Claude-based scorer for intelligent name evaluation."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

from src.config import Settings
from src.generator.candidate_generator import Candidate


@dataclass
class LLMEvaluation:
    """LLM evaluation result for a candidate."""

    name: str
    score: int
    category: str  # coined, suggestive, descriptive
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    recommendation: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "LLMEvaluation":
        """Create LLMEvaluation from dictionary."""
        return cls(
            name=data.get("name", ""),
            score=data.get("score", 0),
            category=data.get("category", "unknown"),
            pros=data.get("pros", []),
            cons=data.get("cons", []),
            flags=data.get("flags", []),
            recommendation=data.get("recommendation", ""),
        )


class ClaudeScorer:
    """Claude-based scorer for evaluating name candidates.

    Uses Claude API to provide intelligent semantic evaluation
    of brand name candidates.
    """

    def __init__(self, settings: Settings, api_key: str | None = None):
        """Initialize Claude scorer.

        Args:
            settings: Configuration settings
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var
        """
        self.settings = settings
        self.model = settings.llm.model
        self.batch_size = settings.llm.batch_size
        self.max_tokens = settings.llm.max_tokens
        self.temperature = settings.llm.temperature

        # Load system prompt
        prompt_path = Path(__file__).parent.parent.parent / "prompts" / "brand_evaluator_system.txt"
        if prompt_path.exists():
            with open(prompt_path, encoding="utf-8") as f:
                self.system_prompt = f.read()
        else:
            raise FileNotFoundError(f"System prompt not found: {prompt_path}")

        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def _build_user_prompt(self, candidates: list[Candidate]) -> str:
        """Build user prompt for batch evaluation.

        Args:
            candidates: List of candidates to evaluate

        Returns:
            Formatted user prompt
        """
        names_xml = "\n".join(f"    <name>{c.name}</name>" for c in candidates)

        prompt = f"""<task>
Vyhodnoť následující kandidáty na název firmy podle kritérií uvedených v systémovém promptu.
</task>

<candidates>
{names_xml}
</candidates>

<output_format>
Vrať POUZE platný JSON objekt s klíčem "evaluations" obsahujícím pole hodnocení.
Nepiš žádný další text před nebo za JSON.
</output_format>"""

        return prompt

    def _parse_response(self, response_text: str) -> list[LLMEvaluation]:
        """Parse Claude's response into LLMEvaluation objects.

        Args:
            response_text: Raw response text from Claude

        Returns:
            List of LLMEvaluation objects
        """
        # Try to extract JSON from response
        text = response_text.strip()

        # Handle potential markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (```json and ```)
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith("```") and not in_json:
                    in_json = True
                    continue
                elif line.startswith("```") and in_json:
                    break
                elif in_json:
                    json_lines.append(line)
            text = "\n".join(json_lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {text[:500]}")

        evaluations = []
        for item in data.get("evaluations", []):
            evaluations.append(LLMEvaluation.from_dict(item))

        return evaluations

    def score(self, candidates: list[Candidate]) -> list[LLMEvaluation]:
        """Score a batch of candidates using Claude.

        Args:
            candidates: List of candidates to evaluate

        Returns:
            List of LLMEvaluation objects
        """
        if not candidates:
            return []

        user_prompt = self._build_user_prompt(candidates)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        response_text = response.content[0].text
        return self._parse_response(response_text)

    def score_all(self, candidates: list[Candidate]) -> list[LLMEvaluation]:
        """Score all candidates, handling batching automatically.

        Args:
            candidates: List of all candidates to evaluate

        Returns:
            List of all LLMEvaluation objects
        """
        all_evaluations = []

        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i : i + self.batch_size]
            evaluations = self.score(batch)
            all_evaluations.extend(evaluations)

        return all_evaluations
