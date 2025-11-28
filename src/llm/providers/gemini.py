"""Google Gemini provider implementation."""

import os

from src.llm.base import BaseLLMProvider, LLMConfig


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider.

    Supports Gemini models via the Google Gen AI SDK.
    Requires GOOGLE_API_KEY environment variable or api_key in config.
    """

    def __init__(self, config: LLMConfig):
        """Initialize Gemini provider.

        Args:
            config: LLM configuration
        """
        super().__init__(config)
        self._client = None

    @property
    def name(self) -> str:
        return "gemini"

    def _get_client(self):
        """Lazy load Google GenAI client."""
        if self._client is None:
            from google import genai

            api_key = self.config.api_key or os.environ.get("GOOGLE_API_KEY")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def is_available(self) -> bool:
        """Check if Google API is available."""
        try:
            api_key = self.config.api_key or os.environ.get("GOOGLE_API_KEY")
            return api_key is not None and len(api_key) > 0
        except Exception:
            return False

    def generate(self, user_prompt: str) -> str:
        """Generate response using Gemini.

        Args:
            user_prompt: The user prompt to send

        Returns:
            The text response from Gemini
        """
        from google.genai import types

        client = self._get_client()

        # Combine system prompt and user prompt
        full_prompt = f"{self.system_prompt}\n\n---\n\n{user_prompt}"

        response = client.models.generate_content(
            model=self.config.model,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            ),
        )

        return response.text or ""
