"""Pairwise statistical significance testing for portfolio methods.

Provides HAC (Newey-West) and block bootstrap tests for comparing
daily returns between portfolio methods.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# Try to import statsmodels for HAC test
try:
    from statsmodels.regression.linear_model import OLS
    from statsmodels.stats.sandwich_covariance import cov_hac

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


@dataclass
class PairwiseTestResult:
    """Result from a pairwise statistical test."""

    method_a: str
    method_b: str
    mean_diff: float
    std_error: float
    t_stat: float
    p_value: float
    significant: bool
    test_type: str = "HAC"
    n_observations: int = 0


@dataclass
class BootstrapTestResult:
    """Result from a paired block bootstrap test."""

    method_a: str
    method_b: str
    mean_diff: float
    ci_low: float
    ci_high: float
    significant: bool
    n_bootstrap: int = 1000
    confidence_level: float = 0.95
    n_observations: int = 0


def equity_to_returns(equity_curve: np.ndarray) -> np.ndarray:
    """Convert equity curve to daily returns.

    Args:
        equity_curve: Array of portfolio values over time.

    Returns:
        Array of daily returns (length T-1).
    """
    return np.diff(equity_curve) / equity_curve[:-1]


def paired_hac_test(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    method_a: str = "A",
    method_b: str = "B",
    max_lags: Optional[int] = None,
    alpha: float = 0.05,
) -> PairwiseTestResult:
    """Paired t-test with HAC/Newey-West standard errors.

    Tests H0: mean(returns_a - returns_b) = 0

    Args:
        returns_a: Daily returns for method A.
        returns_b: Daily returns for method B.
        method_a: Name of method A.
        method_b: Name of method B.
        max_lags: Maximum lags for Newey-West. If None, uses rule of thumb.
        alpha: Significance level.

    Returns:
        PairwiseTestResult with test statistics.

    Raises:
        ImportError: If statsmodels is not available.
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError(
            "statsmodels is required for HAC test. "
            "Install with: pip install statsmodels"
        )

    # Compute return differences
    d = returns_a - returns_b
    n = len(d)

    # Newey-West rule of thumb for max lags
    if max_lags is None:
        max_lags = int(np.floor(4 * (n / 100) ** (2 / 9)))

    # OLS regression of d on constant with HAC standard errors
    model = OLS(d, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})

    mean_diff = model.params[0]
    hac_stderr = model.bse[0]
    t_stat = model.tvalues[0]
    p_value = model.pvalues[0]

    return PairwiseTestResult(
        method_a=method_a,
        method_b=method_b,
        mean_diff=mean_diff,
        std_error=hac_stderr,
        t_stat=t_stat,
        p_value=p_value,
        significant=p_value < alpha,
        test_type="HAC",
        n_observations=n,
    )


def _block_bootstrap_mean(
    data: np.ndarray,
    block_size: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute bootstrap means using block resampling.

    Args:
        data: 1D array of observations.
        block_size: Size of blocks for resampling.
        n_bootstrap: Number of bootstrap samples.
        rng: Random number generator.

    Returns:
        Array of bootstrap mean estimates.
    """
    n = len(data)
    n_blocks = int(np.ceil(n / block_size))
    bootstrap_means = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        # Sample block start indices with replacement
        block_starts = rng.integers(0, n - block_size + 1, size=n_blocks)

        # Construct bootstrap sample
        sample = np.concatenate(
            [data[start : start + block_size] for start in block_starts]
        )[:n]

        bootstrap_means[b] = np.mean(sample)

    return bootstrap_means


def paired_block_bootstrap(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    method_a: str = "A",
    method_b: str = "B",
    n_bootstrap: int = 1000,
    block_size: Optional[int] = None,
    confidence_level: float = 0.95,
    seed: Optional[int] = None,
) -> BootstrapTestResult:
    """Paired block bootstrap for return differences.

    Tests whether the mean difference in returns is significantly
    different from zero.

    Args:
        returns_a: Daily returns for method A.
        returns_b: Daily returns for method B.
        method_a: Name of method A.
        method_b: Name of method B.
        n_bootstrap: Number of bootstrap samples.
        block_size: Size of blocks. If None, uses rule of thumb.
        confidence_level: Confidence level for CI.
        seed: Random seed.

    Returns:
        BootstrapTestResult with bootstrap CI.
    """
    # Compute return differences
    d = returns_a - returns_b
    n = len(d)

    # Rule of thumb for block size
    if block_size is None:
        block_size = int(np.ceil(n ** (1 / 3)))

    # Initialize RNG
    rng = np.random.default_rng(seed)

    # Bootstrap the mean difference
    bootstrap_means = _block_bootstrap_mean(d, block_size, n_bootstrap, rng)

    # Compute percentile CI
    alpha = 1 - confidence_level
    ci_low = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    # Significant if CI excludes zero
    significant = (ci_low > 0) or (ci_high < 0)

    return BootstrapTestResult(
        method_a=method_a,
        method_b=method_b,
        mean_diff=np.mean(d),
        ci_low=ci_low,
        ci_high=ci_high,
        significant=significant,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        n_observations=n,
    )


def run_pairwise_tests(
    equity_curves: Dict[str, np.ndarray],
    baseline: str = "MeanVariance",
    alpha: float = 0.05,
    use_bootstrap: bool = False,
    n_bootstrap: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[List[PairwiseTestResult], Optional[PairwiseTestResult]]:
    """Run pairwise tests for all methods vs baseline and #1 vs #2.

    Args:
        equity_curves: Dictionary mapping method name to equity curve.
        baseline: Name of baseline method for comparison.
        alpha: Significance level.
        use_bootstrap: Use block bootstrap instead of HAC.
        n_bootstrap: Number of bootstrap samples (if using bootstrap).
        seed: Random seed for bootstrap.

    Returns:
        Tuple of (baseline_comparisons, top_comparison).
        baseline_comparisons: List of test results vs baseline.
        top_comparison: Test result for #1 vs #2 (or None if < 2 methods).
    """
    # Convert equity curves to returns
    returns_dict = {
        name: equity_to_returns(eq) for name, eq in equity_curves.items()
    }

    # Check baseline exists
    if baseline not in returns_dict:
        raise ValueError(f"Baseline method '{baseline}' not found in equity curves")

    baseline_returns = returns_dict[baseline]

    # Test all methods vs baseline
    baseline_comparisons = []
    for name, returns in returns_dict.items():
        if name == baseline:
            continue

        # Ensure returns have same length
        min_len = min(len(returns), len(baseline_returns))
        if min_len == 0:
            continue

        r_a = returns[:min_len]
        r_b = baseline_returns[:min_len]

        if use_bootstrap:
            result = paired_block_bootstrap(
                r_a,
                r_b,
                method_a=name,
                method_b=baseline,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            # Convert bootstrap result to PairwiseTestResult for consistency
            baseline_comparisons.append(
                PairwiseTestResult(
                    method_a=name,
                    method_b=baseline,
                    mean_diff=result.mean_diff,
                    std_error=(result.ci_high - result.ci_low) / (2 * 1.96),
                    t_stat=result.mean_diff
                    / ((result.ci_high - result.ci_low) / (2 * 1.96) + 1e-10),
                    p_value=0.05 if not result.significant else 0.01,
                    significant=result.significant,
                    test_type="Bootstrap",
                    n_observations=result.n_observations,
                )
            )
        else:
            result = paired_hac_test(
                r_a,
                r_b,
                method_a=name,
                method_b=baseline,
                alpha=alpha,
            )
            baseline_comparisons.append(result)

    # Rank methods by mean return
    mean_returns = {
        name: np.mean(returns) for name, returns in returns_dict.items()
    }
    ranked_methods = sorted(mean_returns.keys(), key=lambda x: mean_returns[x], reverse=True)

    # Test #1 vs #2
    top_comparison = None
    if len(ranked_methods) >= 2:
        top1, top2 = ranked_methods[0], ranked_methods[1]
        r1 = returns_dict[top1]
        r2 = returns_dict[top2]

        min_len = min(len(r1), len(r2))
        if min_len > 0:
            r1 = r1[:min_len]
            r2 = r2[:min_len]

            if use_bootstrap:
                result = paired_block_bootstrap(
                    r1,
                    r2,
                    method_a=top1,
                    method_b=top2,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                )
                top_comparison = PairwiseTestResult(
                    method_a=top1,
                    method_b=top2,
                    mean_diff=result.mean_diff,
                    std_error=(result.ci_high - result.ci_low) / (2 * 1.96),
                    t_stat=result.mean_diff
                    / ((result.ci_high - result.ci_low) / (2 * 1.96) + 1e-10),
                    p_value=0.05 if not result.significant else 0.01,
                    significant=result.significant,
                    test_type="Bootstrap",
                    n_observations=result.n_observations,
                )
            else:
                top_comparison = paired_hac_test(
                    r1,
                    r2,
                    method_a=top1,
                    method_b=top2,
                    alpha=alpha,
                )

    return baseline_comparisons, top_comparison


def format_pairwise_results(
    baseline_comparisons: List[PairwiseTestResult],
    top_comparison: Optional[PairwiseTestResult],
    alpha: float = 0.05,
) -> str:
    """Format pairwise test results as a readable string.

    Args:
        baseline_comparisons: List of test results vs baseline.
        top_comparison: Test result for #1 vs #2.
        alpha: Significance level used.

    Returns:
        Formatted string with results tables.
    """
    lines = []
    lines.append("Pairwise Statistical Significance Tests")
    lines.append("=" * 50)
    lines.append("")

    if baseline_comparisons:
        baseline = baseline_comparisons[0].method_b
        lines.append(f"All Methods vs {baseline} ({baseline_comparisons[0].test_type} test, α={alpha}):")
        lines.append("-" * 80)
        lines.append(
            f"{'Method':<20} | {'Mean Diff':>10} | {'SE':>10} | {'t-stat':>8} | "
            f"{'p-value':>8} | {'Sig?':>5}"
        )
        lines.append("-" * 80)

        for result in baseline_comparisons:
            sig_str = "Yes" if result.significant else "No"
            lines.append(
                f"{result.method_a:<20} | {result.mean_diff:>10.6f} | "
                f"{result.std_error:>10.6f} | {result.t_stat:>8.3f} | "
                f"{result.p_value:>8.4f} | {sig_str:>5}"
            )
        lines.append("")

    if top_comparison:
        lines.append("Top Performer Comparison:")
        lines.append("-" * 60)
        lines.append(
            f"{'Comparison':<30} | {'Mean Diff':>10} | {'p-value':>8} | {'Sig?':>5}"
        )
        lines.append("-" * 60)
        sig_str = "Yes" if top_comparison.significant else "No"
        comparison_str = f"#1 ({top_comparison.method_a}) vs #2 ({top_comparison.method_b})"
        lines.append(
            f"{comparison_str:<30} | {top_comparison.mean_diff:>10.6f} | "
            f"{top_comparison.p_value:>8.4f} | {sig_str:>5}"
        )

    return "\n".join(lines)
