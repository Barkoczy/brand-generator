"""Universal LLM scorer for intelligent name evaluation."""

import json

from src.config import Settings
from src.generator.candidate_generator import Candidate
from src.llm.base import BaseLLMProvider, LLMConfig, LLMEvaluation
from src.llm.factory import get_provider


class LLMScorer:
    """Universal LLM scorer for evaluating name candidates.

    Uses any supported LLM provider to provide intelligent semantic
    evaluation of brand name candidates.

    Supported providers:
    - anthropic (Claude)
    - openai (GPT)
    - ollama (local models)
    - lmstudio (LM Studio local)
    - gemini (Google Gemini)
    """

    def __init__(self, settings: Settings, provider: BaseLLMProvider | None = None):
        """Initialize LLM scorer.

        Args:
            settings: Configuration settings
            provider: Optional pre-configured provider. If None, creates from settings.
        """
        self.settings = settings
        self.batch_size = settings.llm.batch_size

        if provider is not None:
            self.provider = provider
        else:
            # Create provider from settings
            config = LLMConfig(
                provider=settings.llm.provider,
                model=settings.llm.model,
                api_key=settings.llm.api_key,
                api_base=settings.llm.api_base,
                batch_size=settings.llm.batch_size,
                max_tokens=settings.llm.max_tokens,
                temperature=settings.llm.temperature,
            )
            self.provider = get_provider(config)

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
        """Parse LLM response into LLMEvaluation objects.

        Args:
            response_text: Raw response text from LLM

        Returns:
            List of LLMEvaluation objects
        """
        text = response_text.strip()

        # Handle potential markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
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

        # Try to find JSON object in response
        if not text.startswith("{"):
            # Try to find JSON in the text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                text = text[start:end]

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Nepodařilo se parsovat JSON odpověď: {e}\nOdpověď: {text[:500]}")

        evaluations = []
        for item in data.get("evaluations", []):
            evaluations.append(LLMEvaluation.from_dict(item))

        return evaluations

    def score(self, candidates: list[Candidate]) -> list[LLMEvaluation]:
        """Score a batch of candidates using the LLM provider.

        Args:
            candidates: List of candidates to evaluate

        Returns:
            List of LLMEvaluation objects
        """
        if not candidates:
            return []

        user_prompt = self._build_user_prompt(candidates)
        response_text = self.provider.generate(user_prompt)
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

    def is_available(self) -> bool:
        """Check if the provider is available.

        Returns:
            True if provider can be used
        """
        return self.provider.is_available()

    @property
    def provider_name(self) -> str:
        """Get the name of the current provider."""
        return self.provider.name
