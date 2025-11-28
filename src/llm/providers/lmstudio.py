"""LM Studio provider implementation."""

import httpx

from src.llm.base import BaseLLMProvider, LLMConfig


class LMStudioProvider(BaseLLMProvider):
    """LM Studio local LLM provider.

    Supports local models via LM Studio's OpenAI-compatible API.
    Default endpoint: http://localhost:1234/v1
    """

    DEFAULT_BASE_URL = "http://localhost:1234/v1"

    def __init__(self, config: LLMConfig):
        """Initialize LM Studio provider.

        Args:
            config: LLM configuration
        """
        super().__init__(config)
        self.base_url = config.api_base or self.DEFAULT_BASE_URL

    @property
    def name(self) -> str:
        return "lmstudio"

    def is_available(self) -> bool:
        """Check if LM Studio server is running."""
        try:
            response = httpx.get(f"{self.base_url}/models", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    def generate(self, user_prompt: str) -> str:
        """Generate response using LM Studio.

        LM Studio provides an OpenAI-compatible API endpoint.

        Args:
            user_prompt: The user prompt to send

        Returns:
            The text response from LM Studio
        """
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            },
            timeout=120.0,
        )

        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
