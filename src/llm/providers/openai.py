"""OpenAI provider implementation."""

import os

from src.llm.base import BaseLLMProvider, LLMConfig


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider.

    Supports GPT models via the official OpenAI API.
    Requires OPENAI_API_KEY environment variable or api_key in config.
    """

    def __init__(self, config: LLMConfig):
        """Initialize OpenAI provider.

        Args:
            config: LLM configuration
        """
        super().__init__(config)
        self._client = None

    @property
    def name(self) -> str:
        return "openai"

    def _get_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            from openai import OpenAI

            api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        try:
            api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
            return api_key is not None and len(api_key) > 0
        except Exception:
            return False

    def generate(self, user_prompt: str) -> str:
        """Generate response using OpenAI.

        Args:
            user_prompt: The user prompt to send

        Returns:
            The text response from OpenAI
        """
        client = self._get_client()

        response = client.chat.completions.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return response.choices[0].message.content or ""
