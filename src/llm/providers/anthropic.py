"""Anthropic Claude provider implementation."""

import os

from src.llm.base import BaseLLMProvider, LLMConfig


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider.

    Supports Claude models via the official Anthropic API.
    Requires ANTHROPIC_API_KEY environment variable or api_key in config.
    """

    def __init__(self, config: LLMConfig):
        """Initialize Anthropic provider.

        Args:
            config: LLM configuration
        """
        super().__init__(config)
        self._client = None

    @property
    def name(self) -> str:
        return "anthropic"

    def _get_client(self):
        """Lazy load Anthropic client."""
        if self._client is None:
            import anthropic

            api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def is_available(self) -> bool:
        """Check if Anthropic API is available."""
        try:
            api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
            return api_key is not None and len(api_key) > 0
        except Exception:
            return False

    def generate(self, user_prompt: str) -> str:
        """Generate response using Claude.

        Args:
            user_prompt: The user prompt to send

        Returns:
            The text response from Claude
        """
        client = self._get_client()

        response = client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        return response.content[0].text
