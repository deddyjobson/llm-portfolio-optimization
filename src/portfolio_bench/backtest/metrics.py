"""Portfolio performance metrics calculation."""

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class Metrics:
    """Container for portfolio performance metrics."""

    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    sortino_ratio: float
    calmar_ratio: float
    avg_turnover: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "avg_turnover": self.avg_turnover,
        }


def compute_metrics(
    equity_curve: np.ndarray,
    weights_history: np.ndarray,
    trading_days_per_year: int = 252,
    turnovers: np.ndarray | None = None,
) -> Metrics:
    """Compute portfolio performance metrics.

    Args:
        equity_curve: Array of portfolio values over time.
        weights_history: Array of shape (T, N) with portfolio weights at each step.
        trading_days_per_year: Number of trading days per year for annualization.
        turnovers: Optional array of actual turnovers at each step. If provided,
            used instead of computing from weights_history diff.

    Returns:
        Metrics object with computed performance metrics.
    """
    # Returns from equity curve
    returns = np.diff(equity_curve) / equity_curve[:-1]

    # Total return
    total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100

    # Annualized return
    n_periods = len(equity_curve) - 1
    ann_factor = trading_days_per_year / n_periods
    annualized_return = ((equity_curve[-1] / equity_curve[0]) ** ann_factor - 1) * 100

    # Sharpe ratio (assuming 0 risk-free rate)
    if np.std(returns) > 1e-10:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(trading_days_per_year)
    else:
        sharpe_ratio = 0.0

    # Max drawdown
    cummax = np.maximum.accumulate(equity_curve)
    drawdowns = (cummax - equity_curve) / cummax
    max_drawdown = np.max(drawdowns) * 100

    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0 and np.std(downside_returns) > 1e-10:
        sortino_ratio = np.mean(returns) / np.std(downside_returns) * np.sqrt(trading_days_per_year)
    else:
        sortino_ratio = sharpe_ratio  # Fall back to Sharpe if no downside

    # Calmar ratio (annualized return / max drawdown)
    if max_drawdown > 1e-10:
        calmar_ratio = annualized_return / max_drawdown
    else:
        calmar_ratio = 0.0

    # Average turnover
    if turnovers is not None:
        # Use actual turnovers if provided (accounts for weight drift)
        avg_turnover = np.mean(turnovers) * 100
    elif len(weights_history) > 1:
        # Fall back to computing from target weight changes
        computed_turnovers = np.sum(np.abs(np.diff(weights_history, axis=0)), axis=1)
        avg_turnover = np.mean(computed_turnovers) * 100
    else:
        avg_turnover = 0.0

    return Metrics(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        avg_turnover=avg_turnover,
    )
