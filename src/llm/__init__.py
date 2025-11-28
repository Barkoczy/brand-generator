"""LLM module for intelligent name evaluation."""

from .base import BaseLLMProvider, LLMConfig, LLMEvaluation
from .factory import (
    check_provider_availability,
    get_default_model,
    get_provider,
    list_providers,
)
from .scorer import LLMScorer

# Backwards compatibility
ClaudeScorer = LLMScorer

__all__ = [
    "BaseLLMProvider",
    "LLMConfig",
    "LLMEvaluation",
    "LLMScorer",
    "ClaudeScorer",
    "get_provider",
    "get_default_model",
    "list_providers",
    "check_provider_availability",
]
