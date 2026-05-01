"""Block bootstrap with BCa confidence intervals for portfolio metrics."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class MetricsWithCI:
    """Portfolio metrics with bootstrap confidence intervals."""

    # Point estimates
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    sortino_ratio: float
    calmar_ratio: float
    avg_turnover: float

    # Confidence intervals (lower, upper)
    total_return_ci: Tuple[float, float]
    annualized_return_ci: Tuple[float, float]
    sharpe_ratio_ci: Tuple[float, float]
    max_drawdown_ci: Tuple[float, float]
    sortino_ratio_ci: Tuple[float, float]
    calmar_ratio_ci: Tuple[float, float]
    avg_turnover_ci: Tuple[float, float]

    # Metadata
    n_bootstrap: int
    block_size: int
    confidence_level: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "avg_turnover": self.avg_turnover,
            "total_return_ci": list(self.total_return_ci),
            "annualized_return_ci": list(self.annualized_return_ci),
            "sharpe_ratio_ci": list(self.sharpe_ratio_ci),
            "max_drawdown_ci": list(self.max_drawdown_ci),
            "sortino_ratio_ci": list(self.sortino_ratio_ci),
            "calmar_ratio_ci": list(self.calmar_ratio_ci),
            "avg_turnover_ci": list(self.avg_turnover_ci),
            "n_bootstrap": self.n_bootstrap,
            "block_size": self.block_size,
            "confidence_level": self.confidence_level,
        }


def block_bootstrap_indices(
    n: int, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate circular block bootstrap indices.

    Args:
        n: Length of the original series.
        block_size: Size of each block.
        rng: NumPy random generator.

    Returns:
        Array of bootstrap indices of length n.
    """
    n_blocks = int(np.ceil(n / block_size))
    # Random starting points for each block
    starts = rng.integers(0, n, size=n_blocks)

    indices = []
    for start in starts:
        # Circular block: wraps around if needed
        block = [(start + i) % n for i in range(block_size)]
        indices.extend(block)

    return np.array(indices[:n])


def compute_metrics_from_returns(
    returns: np.ndarray,
    turnovers: np.ndarray,
    trading_days_per_year: int = 252,
) -> Dict[str, float]:
    """Compute metrics from returns and turnover series.

    Args:
        returns: Array of period returns.
        turnovers: Array of turnover values.
        trading_days_per_year: Trading days per year for annualization.

    Returns:
        Dictionary of metric values.
    """
    n_periods = len(returns)

    # Reconstruct equity curve from returns
    equity_curve = np.cumprod(np.concatenate([[1.0], 1.0 + returns]))

    # Total return
    total_return = (equity_curve[-1] - 1.0) * 100

    # Annualized return
    ann_factor = trading_days_per_year / n_periods
    annualized_return = (equity_curve[-1] ** ann_factor - 1) * 100

    # Sharpe ratio
    if np.std(returns) > 1e-10:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(trading_days_per_year)
    else:
        sharpe_ratio = 0.0

    # Max drawdown
    cummax = np.maximum.accumulate(equity_curve)
    drawdowns = (cummax - equity_curve) / cummax
    max_drawdown = np.max(drawdowns) * 100

    # Sortino ratio
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0 and np.std(downside_returns) > 1e-10:
        sortino_ratio = (
            np.mean(returns) / np.std(downside_returns) * np.sqrt(trading_days_per_year)
        )
    else:
        sortino_ratio = sharpe_ratio

    # Calmar ratio
    if max_drawdown > 1e-10:
        calmar_ratio = annualized_return / max_drawdown
    else:
        calmar_ratio = 0.0

    # Average turnover
    if len(turnovers) > 0:
        avg_turnover = np.mean(turnovers) * 100
    else:
        avg_turnover = 0.0

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "avg_turnover": avg_turnover,
    }


def jackknife_block_samples(
    returns: np.ndarray,
    turnovers: np.ndarray,
    block_size: int,
) -> List[Dict[str, float]]:
    """Compute block jackknife samples for BCa acceleration factor.

    Args:
        returns: Array of period returns.
        turnovers: Array of turnover values.
        block_size: Size of each block to leave out.

    Returns:
        List of metric dictionaries, one for each leave-one-block-out sample.
    """
    n = len(returns)
    n_blocks = n // block_size
    jackknife_samples = []

    for i in range(n_blocks):
        # Leave out block i
        start = i * block_size
        end = start + block_size

        jack_returns = np.concatenate([returns[:start], returns[end:]])
        jack_turnovers = np.concatenate([turnovers[:start], turnovers[end:]])

        if len(jack_returns) > 0:
            metrics = compute_metrics_from_returns(jack_returns, jack_turnovers)
            jackknife_samples.append(metrics)

    return jackknife_samples


def bca_interval(
    original: float,
    bootstrap_samples: np.ndarray,
    jackknife_values: np.ndarray,
    alpha: float,
) -> Tuple[float, float]:
    """Compute BCa (bias-corrected and accelerated) confidence interval.

    Args:
        original: Original point estimate.
        bootstrap_samples: Array of bootstrap estimates.
        jackknife_values: Array of jackknife estimates.
        alpha: Significance level (e.g., 0.05 for 95% CI).

    Returns:
        Tuple of (lower, upper) confidence interval bounds.
    """
    n_boot = len(bootstrap_samples)
    n_jack = len(jackknife_values)

    # Bias correction factor z0
    # Proportion of bootstrap samples less than original
    prop_less = np.sum(bootstrap_samples < original) / n_boot
    # Clip to avoid infinite values at boundaries
    prop_less = np.clip(prop_less, 1e-10, 1 - 1e-10)
    z0 = stats.norm.ppf(prop_less)

    # Acceleration factor a
    if n_jack > 1:
        theta_bar = np.mean(jackknife_values)
        diff = theta_bar - jackknife_values
        num = np.sum(diff**3)
        denom = 6 * (np.sum(diff**2)) ** 1.5
        if abs(denom) > 1e-10:
            a = num / denom
        else:
            a = 0.0
    else:
        a = 0.0

    # Adjusted percentiles
    z_alpha_low = stats.norm.ppf(alpha / 2)
    z_alpha_high = stats.norm.ppf(1 - alpha / 2)

    # BCa formula
    def adjusted_percentile(z_alpha):
        numerator = z0 + z_alpha
        denominator = 1 - a * numerator
        if abs(denominator) < 1e-10:
            return z_alpha  # Fall back to standard percentile
        return stats.norm.cdf(z0 + numerator / denominator)

    p_low = adjusted_percentile(z_alpha_low)
    p_high = adjusted_percentile(z_alpha_high)

    # Clip percentiles to valid range
    p_low = np.clip(p_low, 0, 1)
    p_high = np.clip(p_high, 0, 1)

    # Get quantiles from bootstrap distribution
    sorted_samples = np.sort(bootstrap_samples)
    lower = np.percentile(bootstrap_samples, p_low * 100)
    upper = np.percentile(bootstrap_samples, p_high * 100)

    return (lower, upper)


def bootstrap_metrics(
    equity_curve: np.ndarray,
    weights_history: np.ndarray,
    n_bootstrap: int = 1000,
    block_size: Optional[int] = None,
    confidence_level: float = 0.95,
    seed: Optional[int] = None,
    trading_days_per_year: int = 252,
) -> MetricsWithCI:
    """Compute portfolio metrics with BCa bootstrap confidence intervals.

    Args:
        equity_curve: Array of portfolio values over time.
        weights_history: Array of shape (T, N) with portfolio weights.
        n_bootstrap: Number of bootstrap replicates.
        block_size: Block size for block bootstrap (None for auto sqrt(n)).
        confidence_level: Confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed for reproducibility.
        trading_days_per_year: Trading days per year for annualization.

    Returns:
        MetricsWithCI object with point estimates and confidence intervals.
    """
    rng = np.random.default_rng(seed)

    # Compute returns from equity curve
    returns = np.diff(equity_curve) / equity_curve[:-1]
    n = len(returns)

    # Compute turnovers from weights
    if len(weights_history) > 1:
        turnovers = np.sum(np.abs(np.diff(weights_history, axis=0)), axis=1)
    else:
        turnovers = np.array([0.0])

    # Auto block size: sqrt(n)
    if block_size is None:
        block_size = max(1, int(np.sqrt(n)))

    # Original metrics
    original_metrics = compute_metrics_from_returns(returns, turnovers, trading_days_per_year)

    # Bootstrap samples
    metric_names = list(original_metrics.keys())
    bootstrap_estimates = {name: [] for name in metric_names}

    for _ in range(n_bootstrap):
        # Bootstrap indices for returns
        boot_indices = block_bootstrap_indices(n, block_size, rng)
        boot_returns = returns[boot_indices]

        # Bootstrap turnovers with same indices (truncated to turnover length)
        turnover_indices = boot_indices[: len(turnovers)]
        turnover_indices = turnover_indices[turnover_indices < len(turnovers)]
        boot_turnovers = turnovers[turnover_indices] if len(turnover_indices) > 0 else turnovers

        # Compute metrics for bootstrap sample
        boot_metrics = compute_metrics_from_returns(
            boot_returns, boot_turnovers, trading_days_per_year
        )

        for name in metric_names:
            bootstrap_estimates[name].append(boot_metrics[name])

    # Convert to arrays
    for name in metric_names:
        bootstrap_estimates[name] = np.array(bootstrap_estimates[name])

    # Jackknife samples for BCa
    jackknife_samples = jackknife_block_samples(returns, turnovers, block_size)
    jackknife_metrics = {name: [] for name in metric_names}
    for sample in jackknife_samples:
        for name in metric_names:
            jackknife_metrics[name].append(sample[name])
    for name in metric_names:
        jackknife_metrics[name] = np.array(jackknife_metrics[name])

    # Compute BCa intervals
    alpha = 1 - confidence_level
    confidence_intervals = {}
    for name in metric_names:
        ci = bca_interval(
            original_metrics[name],
            bootstrap_estimates[name],
            jackknife_metrics[name],
            alpha,
        )
        confidence_intervals[name] = ci

    return MetricsWithCI(
        # Point estimates
        total_return=original_metrics["total_return"],
        annualized_return=original_metrics["annualized_return"],
        sharpe_ratio=original_metrics["sharpe_ratio"],
        max_drawdown=original_metrics["max_drawdown"],
        sortino_ratio=original_metrics["sortino_ratio"],
        calmar_ratio=original_metrics["calmar_ratio"],
        avg_turnover=original_metrics["avg_turnover"],
        # Confidence intervals
        total_return_ci=confidence_intervals["total_return"],
        annualized_return_ci=confidence_intervals["annualized_return"],
        sharpe_ratio_ci=confidence_intervals["sharpe_ratio"],
        max_drawdown_ci=confidence_intervals["max_drawdown"],
        sortino_ratio_ci=confidence_intervals["sortino_ratio"],
        calmar_ratio_ci=confidence_intervals["calmar_ratio"],
        avg_turnover_ci=confidence_intervals["avg_turnover"],
        # Metadata
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        confidence_level=confidence_level,
    )
