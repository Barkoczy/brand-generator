"""Factory for creating LLM providers."""

from src.llm.base import BaseLLMProvider, LLMConfig
from src.llm.providers import (
    AnthropicProvider,
    GeminiProvider,
    LMStudioProvider,
    OllamaProvider,
    OpenAIProvider,
)

# Registry of available providers
PROVIDERS: dict[str, type[BaseLLMProvider]] = {
    "anthropic": AnthropicProvider,
    "claude": AnthropicProvider,  # alias
    "openai": OpenAIProvider,
    "gpt": OpenAIProvider,  # alias
    "ollama": OllamaProvider,
    "lmstudio": LMStudioProvider,
    "lm-studio": LMStudioProvider,  # alias
    "gemini": GeminiProvider,
    "google": GeminiProvider,  # alias
}

# Default models for each provider
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "ollama": "llama3.2",
    "lmstudio": "local-model",
    "gemini": "gemini-2.0-flash",
}


def get_provider(config: LLMConfig) -> BaseLLMProvider:
    """Create an LLM provider based on configuration.

    Args:
        config: LLM configuration with provider name

    Returns:
        Initialized LLM provider

    Raises:
        ValueError: If provider is not supported
    """
    provider_name = config.provider.lower()

    if provider_name not in PROVIDERS:
        available = ", ".join(sorted(set(PROVIDERS.keys())))
        raise ValueError(
            f"Neznámý provider: '{provider_name}'. "
            f"Dostupné providery: {available}"
        )

    provider_class = PROVIDERS[provider_name]
    return provider_class(config)


def get_default_model(provider: str) -> str:
    """Get the default model for a provider.

    Args:
        provider: Provider name

    Returns:
        Default model name
    """
    # Normalize provider name
    normalized = provider.lower()
    if normalized in ("claude",):
        normalized = "anthropic"
    elif normalized in ("gpt",):
        normalized = "openai"
    elif normalized in ("lm-studio",):
        normalized = "lmstudio"
    elif normalized in ("google",):
        normalized = "gemini"

    return DEFAULT_MODELS.get(normalized, "default")


def list_providers() -> list[str]:
    """List all available provider names (without aliases).

    Returns:
        List of provider names
    """
    return ["anthropic", "openai", "ollama", "lmstudio", "gemini"]


def check_provider_availability(config: LLMConfig) -> tuple[bool, str]:
    """Check if a provider is available and configured.

    Args:
        config: LLM configuration

    Returns:
        Tuple of (is_available, message)
    """
    try:
        provider = get_provider(config)
        if provider.is_available():
            return True, f"Provider '{config.provider}' je dostupný."
        else:
            return False, f"Provider '{config.provider}' není nakonfigurován (chybí API klíč nebo server neběží)."
    except ValueError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Chyba při kontrole providera: {e}"
