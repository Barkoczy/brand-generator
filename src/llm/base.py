"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


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


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: str | None = None
    api_base: str | None = None  # For Ollama, LM Studio, custom endpoints
    batch_size: int = 20
    max_tokens: int = 4096
    temperature: float = 0.3


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM providers must implement this interface to be used
    with the brand generator's scoring system.
    """

    def __init__(self, config: LLMConfig):
        """Initialize provider with configuration.

        Args:
            config: LLM configuration
        """
        self.config = config
        self._system_prompt: str | None = None

    @property
    def system_prompt(self) -> str:
        """Load and cache system prompt."""
        if self._system_prompt is None:
            prompt_path = (
                Path(__file__).parent.parent.parent / "prompts" / "brand_evaluator_system.txt"
            )
            if prompt_path.exists():
                with open(prompt_path, encoding="utf-8") as f:
                    self._system_prompt = f.read()
            else:
                raise FileNotFoundError(f"System prompt not found: {prompt_path}")
        return self._system_prompt

    @abstractmethod
    def generate(self, user_prompt: str) -> str:
        """Generate a response from the LLM.

        Args:
            user_prompt: The user prompt to send

        Returns:
            The raw text response from the LLM
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured.

        Returns:
            True if provider can be used
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""
        pass
