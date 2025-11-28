"""Settings loader and configuration dataclass."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LLMConfig:
    """LLM configuration.

    Supports multiple providers:
    - anthropic: Claude models (requires ANTHROPIC_API_KEY)
    - openai: GPT models (requires OPENAI_API_KEY)
    - ollama: Local models via Ollama (default: http://localhost:11434)
    - lmstudio: Local models via LM Studio (default: http://localhost:1234/v1)
    - gemini: Google Gemini models (requires GOOGLE_API_KEY)
    """

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: str | None = None  # Can also use env vars
    api_base: str | None = None  # For Ollama/LM Studio custom endpoints
    batch_size: int = 20
    max_tokens: int = 4096
    temperature: float = 0.3


@dataclass
class ScoringConfig:
    """Scoring thresholds configuration."""

    min_acceptable_score: int = 6
    excellent_score: int = 8


@dataclass
class DatabaseConfig:
    """Database configuration."""

    path: str = "data/candidates.db"


@dataclass
class Settings:
    """Main settings container for brand generator."""

    banned_substrings: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=lambda: ["CVCV", "CVCVCV"])
    min_length: int = 4
    max_length: int = 8
    max_consecutive_consonants: int = 2
    consonants: str = "bcdfghjklmnprstvz"
    vowels: str = "aeiou"
    preferred_starts: list[str] = field(default_factory=list)
    problematic_combinations: list[str] = field(default_factory=list)
    llm: LLMConfig = field(default_factory=LLMConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        """Create Settings from dictionary."""
        llm_data = data.pop("llm", {})
        scoring_data = data.pop("scoring", {})
        database_data = data.pop("database", {})

        return cls(
            llm=LLMConfig(**llm_data) if llm_data else LLMConfig(),
            scoring=ScoringConfig(**scoring_data) if scoring_data else ScoringConfig(),
            database=DatabaseConfig(**database_data) if database_data else DatabaseConfig(),
            **data,
        )


def load_settings(config_path: Path | str | None = None) -> Settings:
    """Load settings from YAML configuration file.

    Args:
        config_path: Path to configuration file. If None, uses default config.yaml

    Returns:
        Settings object with loaded configuration
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return Settings()

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return Settings()

    return Settings.from_dict(data)
