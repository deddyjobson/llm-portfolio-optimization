"""LLM-based portfolio allocation policies."""

from typing import Union

import numpy as np

from ..base import BasePolicy, project_to_simplex
from .ollama_client import FakeOllamaClient, OllamaClient
from .prompts import (
    build_cot_prompt,
    build_direct_prompt,
    build_few_shot_prompt,
    parse_weights_from_response,
)


class LLMPolicy(BasePolicy):
    """Base class for LLM-based portfolio policies."""

    def __init__(
        self,
        name: str,
        client: Union[OllamaClient, FakeOllamaClient],
        temperature: float = 0.0,
        multiline: bool = False,
    ):
        """Initialize LLM policy.

        Args:
            name: Policy name.
            client: Ollama client instance.
            temperature: Sampling temperature.
            multiline: If True, format data with newlines in prompts.
        """
        super().__init__(name)
        self.client = client
        self.temperature = temperature
        self.multiline = multiline
        self.parse_error_count = 0
        self.total_calls = 0

    def reset(self, n_assets: int) -> None:
        self.n_assets = n_assets
        self.parse_error_count = 0
        self.total_calls = 0
        # Set n_assets on fake client if applicable
        if hasattr(self.client, "set_n_assets"):
            self.client.set_n_assets(n_assets)

    def _build_prompt(self, state: np.ndarray) -> str:
        """Build prompt from state. Override in subclasses."""
        raise NotImplementedError

    def act(self, state: np.ndarray) -> np.ndarray:
        """Get portfolio weights from LLM.

        Args:
            state: Historical log-returns matrix of shape (L, N).

        Returns:
            Portfolio weights of shape (N,).
        """
        prompt = self._build_prompt(state)
        self.total_calls += 1

        try:
            response = self.client.generate(
                prompt=prompt,
                temperature=self.temperature,
                n_assets=self.n_assets,
            )
            weights, parse_error = parse_weights_from_response(response, self.n_assets)
            if parse_error:
                self.parse_error_count += 1
        except (ConnectionError, Exception):
            # Fall back to equal weights on any error
            weights = [1.0 / self.n_assets] * self.n_assets
            self.parse_error_count += 1

        return project_to_simplex(np.array(weights))

    @property
    def parse_error_rate(self) -> float:
        """Return the rate of parse errors encountered."""
        if self.total_calls == 0:
            return 0.0
        return self.parse_error_count / self.total_calls


class DirectPolicy(LLMPolicy):
    """LLM policy using direct prompting."""

    def __init__(
        self,
        client: Union[OllamaClient, FakeOllamaClient],
        temperature: float = 0.0,
        multiline: bool = False,
    ):
        super().__init__("LLM-Direct", client, temperature, multiline)

    def _build_prompt(self, state: np.ndarray) -> str:
        return build_direct_prompt(state, multiline=self.multiline)


class FewShotPolicy(LLMPolicy):
    """LLM policy using few-shot prompting with examples."""

    def __init__(
        self,
        client: Union[OllamaClient, FakeOllamaClient],
        temperature: float = 0.0,
        multiline: bool = False,
    ):
        super().__init__("LLM-FewShot", client, temperature, multiline)

    def _build_prompt(self, state: np.ndarray) -> str:
        return build_few_shot_prompt(state, multiline=self.multiline)


class ChainOfThoughtPolicy(LLMPolicy):
    """LLM policy using chain-of-thought prompting."""

    def __init__(
        self,
        client: Union[OllamaClient, FakeOllamaClient],
        temperature: float = 0.0,
        multiline: bool = False,
    ):
        super().__init__("LLM-CoT", client, temperature, multiline)

    def _build_prompt(self, state: np.ndarray) -> str:
        return build_cot_prompt(state, multiline=self.multiline)


def create_llm_policies(
    base_url: str = "http://localhost:11434",
    model: str = "qwen2.5:1.5b-instruct",
    temperature: float = 0.0,
    use_fake: bool = False,
    multiline: bool = False,
) -> list:
    """Create all three LLM policies.

    Args:
        base_url: Ollama API base URL.
        model: Model name.
        temperature: Sampling temperature.
        use_fake: If True, use fake client for testing.
        multiline: If True, format data with newlines in prompts.

    Returns:
        List of [DirectPolicy, FewShotPolicy, ChainOfThoughtPolicy].
    """
    if use_fake:
        client = FakeOllamaClient(base_url=base_url, model=model)
    else:
        client = OllamaClient(base_url=base_url, model=model)

    return [
        DirectPolicy(client, temperature, multiline),
        FewShotPolicy(client, temperature, multiline),
        ChainOfThoughtPolicy(client, temperature, multiline),
    ]
