"""Unified backtester for portfolio allocation methods."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..methods.base import BasePolicy, project_to_simplex
from .metrics import Metrics, compute_metrics


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    policy_name: str
    equity_curve: np.ndarray
    weights_history: np.ndarray
    metrics: Metrics
    parse_error_rate: Optional[float] = None


class Backtester:
    """Unified backtester for evaluating portfolio allocation methods.

    Ensures fair comparison by using identical evaluation logic for all methods.
    """

    def __init__(
        self,
        price_relatives: np.ndarray,
        log_relatives: np.ndarray,
        lookback: int = 10,
        transaction_cost: float = 0.001,
        initial_capital: float = 1.0,
    ):
        """Initialize the backtester.

        Args:
            price_relatives: Price relatives matrix of shape (T, N).
            log_relatives: Log-relatives matrix of shape (T, N).
            lookback: Number of historical periods to provide as state.
            transaction_cost: Proportional transaction cost rate.
            initial_capital: Starting capital.
        """
        self.price_relatives = price_relatives
        self.log_relatives = log_relatives
        self.lookback = lookback
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital

        self.T, self.N = price_relatives.shape

    def run(self, policy: BasePolicy) -> BacktestResult:
        """Run backtest for a single policy.

        Args:
            policy: Portfolio allocation policy to evaluate.

        Returns:
            BacktestResult with equity curve, weights history, and metrics.
        """
        policy.reset(self.N)

        equity = [self.initial_capital]
        weights_history = []
        turnovers = []
        # Track actual holdings (which drift with price movements)
        holdings_weights = np.ones(self.N) / self.N

        # Start from lookback period
        for t in range(self.lookback, self.T):
            # Construct state: lookback window of log-relatives
            state = self.log_relatives[t - self.lookback : t]

            # Get weights from policy and project to simplex
            raw_weights = policy.act(state)
            weights = project_to_simplex(raw_weights)
            weights_history.append(weights)

            # Calculate turnover: trading from current holdings to new target
            # This reflects actual rebalancing needed, not just target weight changes
            turnover = np.sum(np.abs(weights - holdings_weights))
            turnovers.append(turnover)
            tc = self.transaction_cost * turnover

            # Portfolio return for this period
            period_return = np.dot(weights, self.price_relatives[t])

            # Update equity (subtract transaction costs from growth)
            new_equity = equity[-1] * period_return * (1 - tc)
            equity.append(new_equity)

            # Update holdings to reflect drift after price movement
            # After holding 'weights' during this period, prices move and weights drift
            drifted = weights * self.price_relatives[t]
            holdings_weights = drifted / drifted.sum()

        equity_curve = np.array(equity)
        weights_history = np.array(weights_history)
        turnovers = np.array(turnovers)

        # Compute metrics
        metrics = compute_metrics(equity_curve, weights_history, turnovers=turnovers)

        # Capture parse_error_rate from LLM policies if available
        parse_error_rate = None
        if hasattr(policy, "parse_error_rate"):
            parse_error_rate = policy.parse_error_rate

        return BacktestResult(
            policy_name=policy.name,
            equity_curve=equity_curve,
            weights_history=weights_history,
            metrics=metrics,
            parse_error_rate=parse_error_rate,
        )

    def run_all(self, policies: List[BasePolicy]) -> List[BacktestResult]:
        """Run backtest for multiple policies.

        Args:
            policies: List of portfolio allocation policies.

        Returns:
            List of BacktestResult objects.
        """
        results = []
        for policy in policies:
            result = self.run(policy)
            results.append(result)
        return results
