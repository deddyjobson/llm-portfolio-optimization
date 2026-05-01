"""Cell-level ablation analysis for LLM portfolio policies."""

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np


@dataclass
class WindowAnalysis:
    """Results of analyzing a single window."""

    window_idx: int
    state: np.ndarray  # (L, N) input state
    importance: np.ndarray  # (L, N) importance scores
    original_weights: np.ndarray  # (N,) original output
    response: Optional[str] = None  # Raw LLM response if available


def analyze_window(
    policy,
    state: np.ndarray,
    ablation_value: float = 0.0,
) -> np.ndarray:
    """Compute importance scores for each cell via ablation.

    For each cell (i, j) in the state matrix:
    1. Replace state[i, j] with ablation_value
    2. Get new weights from policy
    3. Compute L1 distance from original weights
    4. Store as importance[i, j]

    Args:
        policy: LLM policy with .act(state) method.
        state: Historical log-returns matrix of shape (L, N).
        ablation_value: Value to use when ablating cells (default: 0.0).

    Returns:
        Importance matrix of shape (L, N) where higher values
        indicate cells that caused larger changes when ablated.
    """
    L, N = state.shape
    original_weights = policy.act(state)
    importance = np.zeros((L, N))

    for i in range(L):
        for j in range(N):
            modified = state.copy()
            modified[i, j] = ablation_value
            new_weights = policy.act(modified)
            importance[i, j] = np.abs(new_weights - original_weights).sum()

    return importance


def analyze_windows(
    policy,
    states: List[np.ndarray],
    ablation_value: float = 0.0,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[WindowAnalysis]:
    """Analyze multiple windows and compute importance scores.

    Args:
        policy: LLM policy with .act(state) and .reset(n_assets) methods.
        states: List of state matrices, each of shape (L, N).
        ablation_value: Value to use when ablating cells.
        progress_callback: Optional callback(current, total) for progress updates.

    Returns:
        List of WindowAnalysis results.
    """
    results = []
    n_total = len(states)

    for idx, state in enumerate(states):
        if progress_callback:
            progress_callback(idx + 1, n_total)

        L, N = state.shape
        policy.reset(N)

        # Get original weights
        original_weights = policy.act(state)

        # Compute importance via ablation
        importance = analyze_window(policy, state, ablation_value)

        results.append(
            WindowAnalysis(
                window_idx=idx,
                state=state,
                importance=importance,
                original_weights=original_weights,
            )
        )

    return results


def compute_aggregate_stats(analyses: List[WindowAnalysis]) -> dict:
    """Compute aggregate statistics across all analyzed windows.

    Args:
        analyses: List of WindowAnalysis results.

    Returns:
        Dictionary with aggregate statistics:
        - mean_importance_by_row: (L,) mean importance per time step
        - mean_importance_by_col: (N,) mean importance per asset
        - recency_correlation: Correlation between row index and importance
        - total_mean_importance: Overall mean importance
    """
    if not analyses:
        return {}

    # Stack all importance matrices
    all_importance = np.array([a.importance for a in analyses])  # (W, L, N)

    # Mean importance by row (time step) - averaged across windows and assets
    mean_by_row = all_importance.mean(axis=(0, 2))  # (L,)

    # Mean importance by column (asset) - averaged across windows and time steps
    mean_by_col = all_importance.mean(axis=(0, 1))  # (N,)

    # Recency analysis: correlation between row index and importance
    L = all_importance.shape[1]
    row_indices = np.arange(L)
    # Handle case of zero variance (all same values)
    if np.std(mean_by_row) < 1e-12:
        recency_corr = 0.0
    else:
        recency_corr = np.corrcoef(row_indices, mean_by_row)[0, 1]

    # Ratio of most recent row importance to oldest row
    if mean_by_row[0] > 1e-12:
        recency_ratio = mean_by_row[-1] / mean_by_row[0]
    elif mean_by_row[-1] > 1e-12:
        recency_ratio = float("inf")
    else:
        recency_ratio = 1.0  # Both are ~zero

    return {
        "mean_importance_by_row": mean_by_row.tolist(),
        "mean_importance_by_col": mean_by_col.tolist(),
        "recency_correlation": float(recency_corr) if not np.isnan(recency_corr) else 0.0,
        "recency_ratio": float(recency_ratio),
        "total_mean_importance": float(all_importance.mean()),
        "n_windows": len(analyses),
    }
