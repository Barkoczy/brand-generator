"""Ollama provider implementation."""

import httpx

from src.llm.base import BaseLLMProvider, LLMConfig


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider.

    Supports local models via Ollama's HTTP API.
    Default endpoint: http://localhost:11434
    """

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, config: LLMConfig):
        """Initialize Ollama provider.

        Args:
            config: LLM configuration
        """
        super().__init__(config)
        self.base_url = config.api_base or self.DEFAULT_BASE_URL

    @property
    def name(self) -> str:
        return "ollama"

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    def generate(self, user_prompt: str) -> str:
        """Generate response using Ollama.

        Args:
            user_prompt: The user prompt to send

        Returns:
            The text response from Ollama
        """
        # Ollama uses a single prompt with system instruction embedded
        full_prompt = f"""<|system|>
{self.system_prompt}
<|user|>
{user_prompt}
<|assistant|>"""

        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.config.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            },
            timeout=120.0,
        )

        response.raise_for_status()
        return response.json().get("response", "")
