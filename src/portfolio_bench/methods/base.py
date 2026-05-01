"""Abstract base class for all portfolio allocation methods."""

from abc import ABC, abstractmethod

import numpy as np


def project_to_simplex(w: np.ndarray) -> np.ndarray:
    """Project weights to the probability simplex.

    Ensures all weights are non-negative and sum to 1.

    Args:
        w: Weight vector of shape (N,).

    Returns:
        Projected weights on the simplex of shape (N,).
    """
    w = np.maximum(w, 0)  # clip negatives
    total = w.sum()
    if total < 1e-12:
        return np.ones(len(w)) / len(w)  # equal weight fallback
    return w / total


class BasePolicy(ABC):
    """Abstract base class for portfolio allocation policies.

    All methods (OR, LLM) must implement this interface to ensure
    fair comparison through the unified backtester.
    """

    def __init__(self, name: str):
        """Initialize the policy.

        Args:
            name: Human-readable name for this policy.
        """
        self.name = name
        self.n_assets: int = 0

    @abstractmethod
    def reset(self, n_assets: int) -> None:
        """Reset the policy for a new backtest.

        Args:
            n_assets: Number of assets in the portfolio.
        """
        ...

    @abstractmethod
    def act(self, state: np.ndarray) -> np.ndarray:
        """Compute portfolio weights given current state.

        Args:
            state: Historical log-returns matrix of shape (L, N) where
                   L is the lookback window and N is the number of assets.

        Returns:
            Portfolio weights of shape (N,) summing to 1.
        """
        ...
