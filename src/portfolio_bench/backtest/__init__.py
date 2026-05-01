"""Backtesting engine for portfolio allocation methods."""

from .backtester import Backtester, BacktestResult
from .bootstrap import MetricsWithCI, bootstrap_metrics
from .metrics import compute_metrics
from .pairwise import (
    BootstrapTestResult,
    PairwiseTestResult,
    equity_to_returns,
    format_pairwise_results,
    paired_block_bootstrap,
    paired_hac_test,
    run_pairwise_tests,
)
from .rolling import MetricSummary, RollingEvaluator, RollingEvalResult, WindowSpec

__all__ = [
    "Backtester",
    "BacktestResult",
    "compute_metrics",
    "bootstrap_metrics",
    "MetricsWithCI",
    "RollingEvaluator",
    "RollingEvalResult",
    "WindowSpec",
    "MetricSummary",
    "PairwiseTestResult",
    "BootstrapTestResult",
    "paired_hac_test",
    "paired_block_bootstrap",
    "run_pairwise_tests",
    "format_pairwise_results",
    "equity_to_returns",
]
