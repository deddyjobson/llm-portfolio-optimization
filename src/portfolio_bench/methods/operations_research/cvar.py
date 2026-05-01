"""Conditional Value-at-Risk (CVaR) portfolio optimization."""

import cvxpy as cp
import numpy as np

from ..base import BasePolicy, project_to_simplex


class CVaRPolicy(BasePolicy):
    """CVaR (Conditional Value-at-Risk) portfolio optimization using CVXPY.

    Minimizes CVaR at a given confidence level, which is the expected loss
    in the worst (1-alpha) fraction of scenarios.

    Solves: minimize CVaR_alpha(portfolio returns)
    subject to: sum(w) = 1, w >= 0
    """

    def __init__(self, alpha: float = 0.95):
        """Initialize CVaR policy.

        Args:
            alpha: Confidence level (e.g., 0.95 for 95% CVaR).
        """
        super().__init__("CVaR")
        self.alpha = alpha

    def reset(self, n_assets: int) -> None:
        self.n_assets = n_assets

    def act(self, state: np.ndarray) -> np.ndarray:
        """Compute optimal weights by minimizing CVaR.

        Args:
            state: Historical log-returns matrix of shape (L, N).

        Returns:
            Portfolio weights of shape (N,).
        """
        L, N = state.shape

        if N == 1:
            return np.array([1.0])

        # Use historical scenarios as possible outcomes
        scenarios = state  # (L, N) - each row is a scenario

        # CVXPY CVaR optimization
        w = cp.Variable(N)
        zeta = cp.Variable()  # VaR threshold
        u = cp.Variable(L)  # auxiliary variables for CVaR

        # Portfolio returns for each scenario
        portfolio_returns = scenarios @ w  # (L,)

        # CVaR formulation:
        # CVaR_alpha = zeta + (1/(1-alpha)) * E[max(-portfolio_return - zeta, 0)]
        # We minimize negative returns (i.e., losses)
        losses = -portfolio_returns

        # CVaR constraint: u_i >= loss_i - zeta, u_i >= 0
        cvar = zeta + (1 / (L * (1 - self.alpha))) * cp.sum(u)

        objective = cp.Minimize(cvar)
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            u >= losses - zeta,
            u >= 0,
        ]

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.ECOS, verbose=False)
            if prob.status in ["optimal", "optimal_inaccurate"]:
                weights = w.value
            else:
                weights = np.ones(N) / N
        except Exception:
            weights = np.ones(N) / N

        return project_to_simplex(weights)
