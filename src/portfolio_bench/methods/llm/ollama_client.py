"""HTTP client for Ollama API."""

import json
from typing import Any, Dict, List, Optional, Union

import requests

# JSON schema for structured output - guarantees {"weights": [...]} format
WEIGHTS_SCHEMA = {
    "type": "object",
    "properties": {
        "weights": {
            "type": "array",
            "items": {"type": "number"},
        }
    },
    "required": ["weights"],
}


def build_weights_schema(n_assets: int) -> dict:
    """Build JSON schema with exact array length constraint.

    Note: Schema enforcement is best-effort. Ollama may not honor
    minItems/maxItems depending on version and model. The length
    check in parse_weights_from_response remains authoritative.
    """
    return {
        "type": "object",
        "properties": {
            "weights": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": n_assets,
                "maxItems": n_assets,
            }
        },
        "required": ["weights"],
    }


class OllamaClient:
    """HTTP client for interacting with Ollama API.

    Ollama provides a local LLM server that can run various models.
    Default endpoint is localhost:11434.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:1.5b-instruct",
        timeout: int = 60,
    ):
        """Initialize Ollama client.

        Args:
            base_url: Ollama API base URL.
            model: Model name to use.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        n_assets: Optional[int] = None,
    ) -> str:
        """Generate text completion from the model.

        Args:
            prompt: Input prompt.
            temperature: Sampling temperature (0 = deterministic).
            max_tokens: Maximum tokens to generate (capped at 256).
            n_assets: Number of assets for schema constraint (optional).

        Returns:
            Generated text response.
        """
        url = f"{self.base_url}/api/generate"

        # Cap max_tokens at 256
        max_tokens = min(max_tokens, 256)

        schema = build_weights_schema(n_assets) if n_assets else WEIGHTS_SCHEMA
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": schema,
            "options": {
                "temperature": temperature,
                "top_p": 1,
                "num_predict": max_tokens,
            },
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama at {url}: {e}")

    def chat(
        self,
        messages: list,
        temperature: float = 0.0,
        max_tokens: int = 256,
        n_assets: Optional[int] = None,
    ) -> str:
        """Generate chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate (capped at 256).
            n_assets: Number of assets for schema constraint (optional).

        Returns:
            Assistant's response text.
        """
        url = f"{self.base_url}/api/chat"

        # Cap max_tokens at 256
        max_tokens = min(max_tokens, 256)

        schema = build_weights_schema(n_assets) if n_assets else WEIGHTS_SCHEMA
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": schema,
            "options": {
                "temperature": temperature,
                "top_p": 1,
                "num_predict": max_tokens,
            },
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama at {url}: {e}")

    def is_available(self) -> bool:
        """Check if Ollama server is available.

        Returns:
            True if server is reachable, False otherwise.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False


class FakeOllamaClient:
    """Fake Ollama client for testing without actual LLM server."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:1.5b-instruct",
        timeout: int = 60,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self._n_assets: Optional[int] = None

    def set_n_assets(self, n: int) -> None:
        """Set number of assets for generating fake responses."""
        self._n_assets = n

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        n_assets: Optional[int] = None,
    ) -> str:
        """Generate fake equal-weight response."""
        n = n_assets or self._n_assets or 5
        weights = [round(1.0 / n, 4)] * n
        return json.dumps({"weights": weights})

    def chat(
        self,
        messages: list,
        temperature: float = 0.0,
        max_tokens: int = 256,
        n_assets: Optional[int] = None,
    ) -> str:
        """Generate fake chat response."""
        return self.generate(messages[-1]["content"], temperature, max_tokens, n_assets)

    def is_available(self) -> bool:
        return True
