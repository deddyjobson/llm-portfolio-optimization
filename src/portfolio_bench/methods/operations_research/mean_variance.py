"""Mean-Variance (Markowitz) portfolio optimization."""

import cvxpy as cp
import numpy as np

from ..base import BasePolicy, project_to_simplex


class MeanVariancePolicy(BasePolicy):
    """Markowitz Mean-Variance portfolio optimization using CVXPY.

    Solves: maximize (mu^T w - lambda * w^T Sigma w)
    subject to: sum(w) = 1, w >= 0
    """

    def __init__(self, risk_aversion: float = 10.0):
        """Initialize Mean-Variance policy.

        Args:
            risk_aversion: Risk aversion parameter (lambda). Higher = more conservative.
        """
        super().__init__("MeanVariance")
        self.risk_aversion = risk_aversion

    def reset(self, n_assets: int) -> None:
        self.n_assets = n_assets

    def act(self, state: np.ndarray) -> np.ndarray:
        """Compute optimal weights using quadratic programming.

        Args:
            state: Historical log-returns matrix of shape (L, N).

        Returns:
            Portfolio weights of shape (N,).
        """
        L, N = state.shape

        # Estimate expected returns and covariance
        mu = np.mean(state, axis=0)
        cov = np.cov(state, rowvar=False)

        # Handle case where cov is scalar (N=1)
        if N == 1:
            return np.array([1.0])

        # Regularize covariance matrix for numerical stability
        cov = cov + 1e-6 * np.eye(N)

        # CVXPY optimization
        w = cp.Variable(N)
        ret = mu @ w
        risk = cp.quad_form(w, cov)
        objective = cp.Maximize(ret - self.risk_aversion * risk)
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
        ]

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.OSQP, verbose=False)
            if prob.status in ["optimal", "optimal_inaccurate"]:
                weights = w.value
            else:
                # Fall back to equal weights
                weights = np.ones(N) / N
        except Exception:
            # Fall back to equal weights on solver failure
            weights = np.ones(N) / N

        return project_to_simplex(weights)
