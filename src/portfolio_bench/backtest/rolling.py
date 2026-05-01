"""Rolling window evaluation framework for portfolio methods.

Provides more realistic uncertainty estimates by evaluating across
multiple non-overlapping test windows rather than bootstrap within
a single split.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from ..methods.base import BasePolicy
from .backtester import Backtester, BacktestResult
from .metrics import Metrics


@dataclass
class WindowSpec:
    """Specification for a single evaluation window."""

    window_idx: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def val_size(self) -> int:
        return self.val_end - self.val_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start


@dataclass
class MetricSummary:
    """Summary statistics for a metric across windows."""

    mean: float
    std: float
    ci_low: float
    ci_high: float
    values: List[float] = field(default_factory=list)


@dataclass
class RollingEvalResult:
    """Results from rolling window evaluation for a single policy."""

    policy_name: str
    n_windows: int
    window_results: List[BacktestResult]
    metrics_summary: Dict[str, MetricSummary]
    parse_error_rates: Optional[List[float]] = None
    mean_parse_error_rate: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "policy_name": self.policy_name,
            "n_windows": self.n_windows,
            "metrics_summary": {
                name: {
                    "mean": summary.mean,
                    "std": summary.std,
                    "ci_low": summary.ci_low,
                    "ci_high": summary.ci_high,
                    "values": summary.values,
                }
                for name, summary in self.metrics_summary.items()
            },
        }
        if self.parse_error_rates is not None:
            result["parse_error_rates"] = self.parse_error_rates
            result["mean_parse_error_rate"] = self.mean_parse_error_rate
        return result


class RollingEvaluator:
    """Rolling window evaluator for portfolio allocation methods.

    Evaluates methods across multiple non-overlapping test windows to
    provide more realistic uncertainty estimates than bootstrap within
    a single window.
    """

    def __init__(
        self,
        price_relatives: np.ndarray,
        log_relatives: np.ndarray,
        train_size: int = 100,
        val_size: int = 30,
        test_size: int = 50,
        step_size: Optional[int] = None,
        lookback: int = 10,
        transaction_cost: float = 0.001,
    ):
        """Initialize the rolling evaluator.

        Args:
            price_relatives: Full price relatives matrix of shape (T, N).
            log_relatives: Full log-relatives matrix of shape (T, N).
            train_size: Number of periods for training window.
            val_size: Number of periods for validation window (can be 0).
            test_size: Number of periods for test window.
            step_size: Step between windows (defaults to test_size for non-overlapping tests).
            lookback: Number of historical periods for state.
            transaction_cost: Proportional transaction cost rate.
        """
        self.price_relatives = price_relatives
        self.log_relatives = log_relatives
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.step_size = step_size if step_size is not None else test_size
        self.lookback = lookback
        self.transaction_cost = transaction_cost

        self.T, self.N = price_relatives.shape

    def generate_windows(self) -> List[WindowSpec]:
        """Generate all evaluation windows.

        Returns:
            List of WindowSpec objects defining each window.
        """
        windows = []
        window_idx = 0
        train_start = 0

        while True:
            train_end = train_start + self.train_size
            val_start = train_end
            val_end = val_start + self.val_size
            test_start = val_end
            test_end = test_start + self.test_size

            # Check if we have enough data for this window
            if test_end > self.T:
                # If this is the first window, we don't have enough data at all
                if window_idx == 0:
                    raise ValueError(
                        f"Not enough data for even one window. "
                        f"Need {train_end + self.val_size + self.test_size} periods, "
                        f"have {self.T}"
                    )
                # Otherwise, try to fit a final partial window
                if test_start < self.T:
                    # Adjust test_end to fit remaining data
                    test_end = self.T
                    if test_end - test_start >= self.lookback + 1:
                        windows.append(
                            WindowSpec(
                                window_idx=window_idx,
                                train_start=train_start,
                                train_end=train_end,
                                val_start=val_start,
                                val_end=val_end,
                                test_start=test_start,
                                test_end=test_end,
                            )
                        )
                break

            windows.append(
                WindowSpec(
                    window_idx=window_idx,
                    train_start=train_start,
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )

            train_start += self.step_size
            window_idx += 1

        return windows

    def run_window(
        self,
        window: WindowSpec,
        policies: List[BasePolicy],
    ) -> List[BacktestResult]:
        """Run evaluation on a single window.

        Args:
            window: Window specification.
            policies: List of policies to evaluate. Policies will be reset.

        Returns:
            List of BacktestResult for each policy.
        """
        # Extract window data
        test_pr = self.price_relatives[window.test_start : window.test_end]
        test_lr = self.log_relatives[window.test_start : window.test_end]

        # Run backtest on test set
        backtester = Backtester(
            price_relatives=test_pr,
            log_relatives=test_lr,
            lookback=self.lookback,
            transaction_cost=self.transaction_cost,
        )

        return backtester.run_all(policies)

    def run_all_windows(
        self,
        policies: List[BasePolicy],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, List[BacktestResult]]:
        """Run evaluation across all windows.

        Args:
            policies: List of policies to evaluate.
            progress_callback: Optional callback(current, total) for progress updates.

        Returns:
            Dictionary mapping policy name to list of results per window.
        """
        windows = self.generate_windows()
        n_windows = len(windows)

        # Initialize results storage
        results_by_policy: Dict[str, List[BacktestResult]] = {
            p.name: [] for p in policies
        }

        for i, window in enumerate(windows):
            if progress_callback:
                progress_callback(i + 1, n_windows)

            window_results = self.run_window(window, policies)

            for result in window_results:
                if result.policy_name not in results_by_policy:
                    results_by_policy[result.policy_name] = []
                results_by_policy[result.policy_name].append(result)

        return results_by_policy

    def aggregate_results(
        self,
        results_by_policy: Dict[str, List[BacktestResult]],
        confidence_level: float = 0.95,
    ) -> Dict[str, RollingEvalResult]:
        """Aggregate window results into summary statistics with CIs.

        Uses t-distribution for CI computation.

        Args:
            results_by_policy: Dictionary mapping policy name to results per window.
            confidence_level: Confidence level for intervals (default 0.95).

        Returns:
            Dictionary mapping policy name to RollingEvalResult.
        """
        aggregated = {}

        for policy_name, window_results in results_by_policy.items():
            n_windows = len(window_results)
            if n_windows == 0:
                continue

            # Collect metrics across windows
            metrics_values: Dict[str, List[float]] = {
                "total_return": [],
                "annualized_return": [],
                "sharpe_ratio": [],
                "max_drawdown": [],
                "sortino_ratio": [],
                "calmar_ratio": [],
                "avg_turnover": [],
            }

            parse_error_rates = []

            for result in window_results:
                m = result.metrics
                metrics_values["total_return"].append(m.total_return)
                metrics_values["annualized_return"].append(m.annualized_return)
                metrics_values["sharpe_ratio"].append(m.sharpe_ratio)
                metrics_values["max_drawdown"].append(m.max_drawdown)
                metrics_values["sortino_ratio"].append(m.sortino_ratio)
                metrics_values["calmar_ratio"].append(m.calmar_ratio)
                metrics_values["avg_turnover"].append(m.avg_turnover)

                if result.parse_error_rate is not None:
                    parse_error_rates.append(result.parse_error_rate)

            # Compute summary statistics
            metrics_summary = {}
            alpha = 1 - confidence_level

            for name, values in metrics_values.items():
                values_arr = np.array(values)
                mean = np.mean(values_arr)
                std = np.std(values_arr, ddof=1) if n_windows > 1 else 0.0

                # T-distribution CI
                if n_windows >= 2:
                    t_crit = stats.t.ppf(1 - alpha / 2, df=n_windows - 1)
                    margin = t_crit * std / np.sqrt(n_windows)
                    ci_low = mean - margin
                    ci_high = mean + margin
                else:
                    ci_low = ci_high = mean

                metrics_summary[name] = MetricSummary(
                    mean=mean,
                    std=std,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    values=list(values_arr),
                )

            # Handle parse error rates
            mean_parse_error = None
            if parse_error_rates:
                mean_parse_error = np.mean(parse_error_rates)

            aggregated[policy_name] = RollingEvalResult(
                policy_name=policy_name,
                n_windows=n_windows,
                window_results=window_results,
                metrics_summary=metrics_summary,
                parse_error_rates=parse_error_rates if parse_error_rates else None,
                mean_parse_error_rate=mean_parse_error,
            )

        return aggregated

    def run_evaluation(
        self,
        policies: List[BasePolicy],
        confidence_level: float = 0.95,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, RollingEvalResult]:
        """Run complete rolling window evaluation.

        Convenience method that runs all windows and aggregates results.

        Args:
            policies: List of policies to evaluate.
            confidence_level: Confidence level for intervals.
            progress_callback: Optional callback(current, total) for progress updates.

        Returns:
            Dictionary mapping policy name to RollingEvalResult.
        """
        results_by_policy = self.run_all_windows(policies, progress_callback)
        return self.aggregate_results(results_by_policy, confidence_level)
