"""Prompt templates for LLM-based portfolio allocation."""

import json
from typing import List, Tuple

import numpy as np

# Direct prompt - simple instruction
DIRECT_PROMPT = """You are a portfolio allocation model. Output portfolio weights for exactly {n_assets} assets.

Input: Log-returns matrix with {lookback} periods and {n_assets} assets.
Output: JSON object with exactly {n_assets} weights: {{"weights": [w1, w2, ..., w{n_assets}]}}
Constraints: All weights >= 0, weights sum to 1.

Data: {data}

Output:"""

# Few-shot prompt - exactly 2 examples with N=3, L=3
FEW_SHOT_PROMPT = """You are a portfolio allocation model. Output portfolio weights for exactly {n_assets} assets.
Constraints: weights >= 0, weights sum to 1.
Output JSON only: {{"weights": [w1, w2, ..., w{n_assets}]}}

Example 1 (3 assets):
Input: [[0.02, 0.01, -0.01], [0.03, 0.00, -0.02], [0.01, 0.02, -0.01]]
Output: {{"weights": [0.67, 0.33, 0.00]}}

Example 2 (4 assets):
Input: [[0.0032, 0.0112, 0.0177, 0.0060],[0.0107, 0.0125, 0.0195, -0.0022],[0.0066, 0.0140, 0.0078, 0.0116],[0.0019, 0.0071, 0.0104, 0.0196]]
Output: {{"weights": [0.1664, 0.4364, 0.2863, 0.1109]}}

Now solve ({n_assets} assets):
Input: {data}
Output:"""

# Chain-of-thought prompt - JSON only output
COT_PROMPT = """You are a portfolio allocation model. Output weights for exactly {n_assets} assets.

Steps:
1. There are {n_assets} assets and {lookback} periods of data
2. Calculate mean return for each asset
3. Assess volatility of each asset
4. Allocate weights considering risk-return tradeoff

Data: {data}
Output: {{"weights": [w1, w2, ..., w{n_assets}]}}"""


def format_state_as_json(state: np.ndarray, precision: int = 4, multiline: bool = False) -> str:
    """Format state matrix as JSON string.

    Args:
        state: Log-returns matrix of shape (L, N).
        precision: Decimal precision for rounding.
        multiline: If True, format with newlines for readability.

    Returns:
        JSON string representation of the matrix.
    """
    # Round and convert to list
    data = np.round(state, precision).tolist()
    if multiline:
        return json.dumps(data, indent=2)
    return json.dumps(data)


def build_direct_prompt(state: np.ndarray, multiline: bool = False) -> str:
    """Build direct prompt from state.

    Args:
        state: Log-returns matrix of shape (L, N).
        multiline: If True, format data with newlines.

    Returns:
        Formatted prompt string.
    """
    lookback, n_assets = state.shape
    data_str = format_state_as_json(state, multiline=multiline)
    return DIRECT_PROMPT.format(data=data_str, lookback=lookback, n_assets=n_assets)


def build_few_shot_prompt(state: np.ndarray, multiline: bool = False) -> str:
    """Build few-shot prompt from state.

    Args:
        state: Log-returns matrix of shape (L, N).
        multiline: If True, format data with newlines.

    Returns:
        Formatted prompt string.
    """
    lookback, n_assets = state.shape
    data_str = format_state_as_json(state, multiline=multiline)
    return FEW_SHOT_PROMPT.format(data=data_str, n_assets=n_assets)


def build_cot_prompt(state: np.ndarray, multiline: bool = False) -> str:
    """Build chain-of-thought prompt from state.

    Args:
        state: Log-returns matrix of shape (L, N).
        multiline: If True, format data with newlines.

    Returns:
        Formatted prompt string.
    """
    lookback, n_assets = state.shape
    data_str = format_state_as_json(state, multiline=multiline)
    return COT_PROMPT.format(data=data_str, lookback=lookback, n_assets=n_assets)


def parse_weights_from_response(
    response: str, n_assets: int
) -> Tuple[List[float], bool]:
    """Parse portfolio weights from LLM response.

    Returns normalized weights (non-negative, sum to 1) ready for
    project_to_simplex. Normalization here ensures parse_error only
    reflects structural failures, not invalid weight values.

    Args:
        response: LLM response text (should be valid JSON with structured output).
        n_assets: Expected number of assets.

    Returns:
        Tuple of (weights list, parse_error flag).
        parse_error is True if parsing failed and fallback was used.
    """
    try:
        data = json.loads(response)
        weights = data.get("weights", [])

        # Length check is authoritative (schema enforcement is best-effort)
        if len(weights) != n_assets:
            return [1.0 / n_assets] * n_assets, True

        weights = [float(w) for w in weights]

        # Normalize: clamp negatives to 0, then scale to sum=1
        weights = [max(0.0, w) for w in weights]
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            # All zeros/negatives: use equal weights (valid structure, bad values)
            weights = [1.0 / n_assets] * n_assets

        return weights, False

    except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
        pass

    # Fall back to equal weights - mark as parse error
    return [1.0 / n_assets] * n_assets, True
