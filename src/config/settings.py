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
class EuipoConfig:
    """EUIPO API configuration.

    To obtain credentials:
    1. Create an account at https://dev.euipo.europa.eu
    2. Register your application in the Apps section
    3. Get client_id and client_secret
    """

    client_id: str | None = None
    client_secret: str | None = None
    sandbox: bool = True
    timeout_seconds: int = 30


@dataclass
class TrademarkConfig:
    """Trademark clearance configuration."""

    euipo: EuipoConfig = field(default_factory=EuipoConfig)
    default_nice_classes: list[int] = field(default_factory=lambda: [9, 35, 42])
    similarity_threshold: float = 0.7


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
    trademark: TrademarkConfig = field(default_factory=TrademarkConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        """Create Settings from dictionary."""
        llm_data = data.pop("llm", {})
        scoring_data = data.pop("scoring", {})
        database_data = data.pop("database", {})
        trademark_data = data.pop("trademark", {})

        # Remove keys that are not part of Settings dataclass
        # These are used by other modules (genetic optimizer, phonetic models)
        data.pop("genetic", None)
        data.pop("islands", None)
        data.pop("phonetic_weights", None)
        data.pop("known_brands", None)

        # Parse nested trademark config
        trademark_config = TrademarkConfig()
        if trademark_data:
            euipo_data = trademark_data.pop("euipo", {})
            trademark_config = TrademarkConfig(
                euipo=EuipoConfig(**euipo_data) if euipo_data else EuipoConfig(),
                default_nice_classes=trademark_data.get("default_nice_classes", [9, 35, 42]),
                similarity_threshold=trademark_data.get("similarity_threshold", 0.7),
            )

        return cls(
            llm=LLMConfig(**llm_data) if llm_data else LLMConfig(),
            scoring=ScoringConfig(**scoring_data) if scoring_data else ScoringConfig(),
            database=DatabaseConfig(**database_data) if database_data else DatabaseConfig(),
            trademark=trademark_config,
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
