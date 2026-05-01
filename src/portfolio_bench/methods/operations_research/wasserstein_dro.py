"""Distributionally Robust Mean-CVaR with Wasserstein ambiguity set.

Implements the conic reformulation in:
Dizon, Huang, Jeyakumar (2025). Portfolio selection under data uncertainty:
Distributionally robust Mean-CVaR with an ell_2 Wasserstein ambiguity set.
"""

import cvxpy as cp
import numpy as np

from ..base import BasePolicy, project_to_simplex


class WassersteinDROPolicy(BasePolicy):
    """Distributionally Robust Mean-CVaR portfolio optimization.

    Solves the conic Mean-CVaR formulation with a 2-Wasserstein ambiguity set
    and ball support (c = 0) as in Section 5.1 of the reference paper.
    """

    def __init__(
        self,
        eta=None,
        epsilon=None,
        rho=None,
        support_radius=None,
        solver_method: str = "conic",
        solver: str = "ECOS",
        alpha=None,
        epsilon_marginal=None,
        risk_weight=None,
    ):
        """Initialize Wasserstein DRO policy.

        Args:
            eta: CVaR parameter in (0, 1). If None and alpha is provided,
                uses eta = 1 - alpha to match the paper's formula.
            epsilon: Wasserstein radius (epsilon).
            rho: Risk aversion parameter (rho >= 0).
            support_radius: Support set radius (beta). If None, inferred from data.
            solver_method: Kept for compatibility; "conic"/"socp" use SOCP.
            solver: CVXPY solver to use (e.g., "ECOS", "SCS").
            alpha: Optional CVaR confidence; mapped to eta = 1 - alpha.
            epsilon_marginal: Legacy alias for epsilon.
            risk_weight: Legacy alias for rho.
        """
        super().__init__("Wasserstein-DRO")
        if eta is None:
            eta = 1.0 - alpha if alpha is not None else 0.05
        if epsilon is None:
            epsilon = epsilon_marginal if epsilon_marginal is not None else 0.1
        if rho is None:
            rho = risk_weight if risk_weight is not None else 1.0

        self.eta = float(eta)
        self.epsilon = float(epsilon)
        self.rho = float(rho)
        self.support_radius = support_radius
        self.solver_method = solver_method
        self.solver = solver

    def reset(self, n_assets: int) -> None:
        self.n_assets = n_assets

    def act(self, state: np.ndarray) -> np.ndarray:
        """Compute robust optimal weights using the conic DRO formulation.

        Args:
            state: Historical log-returns matrix of shape (L, N).

        Returns:
            Portfolio weights of shape (N,).
        """
        L, N = state.shape

        if N == 1:
            return np.array([1.0])

        if L < 3:
            return np.ones(N) / N

        scenarios = np.asarray(state, dtype=float)
        support_radius = self._resolve_support_radius(scenarios)

        weights = self._solve_conic_mean_cvar(scenarios, support_radius)
        return project_to_simplex(weights)

    def _resolve_support_radius(self, scenarios: np.ndarray) -> float:
        """Infer a support radius if not provided."""
        if self.support_radius is not None:
            return max(float(self.support_radius), 1e-6)

        norms = np.linalg.norm(scenarios, axis=1)
        radius = float(np.nanmax(norms)) if norms.size > 0 else 0.0
        if not np.isfinite(radius) or radius <= 1e-6:
            radius = 1.0
        return radius

    def _solve_conic_mean_cvar(
        self, scenarios: np.ndarray, support_radius: float
    ) -> np.ndarray:
        """Solve the conic Mean-CVaR reformulation (SOCP)."""
        L, N = scenarios.shape

        if not (0.0 < self.eta < 1.0):
            return self._fallback_robust_weights(scenarios)

        if support_radius <= 0.0:
            return self._fallback_robust_weights(scenarios)

        w = cp.Variable(N)
        tau = cp.Variable()
        lam = cp.Variable(nonneg=True)
        alpha = cp.Variable(L)
        v1 = cp.Variable((L, N))
        v2 = cp.Variable((L, N))

        constraints = [cp.sum(w) == 1, w >= 0]

        coeff = 1.0 + self.rho / self.eta
        coeff_tau = self.rho * (1.0 - 1.0 / self.eta)

        for k in range(L):
            constraints.extend([
                cp.norm(v1[k], 2) <= lam,
                cp.norm(v2[k], 2) <= lam,
                cp.norm(w + v1[k], 2)
                <= (-self.rho * tau - scenarios[k] @ v1[k] - alpha[k]) / support_radius,
                cp.norm(coeff * w + v2[k], 2)
                <= (-coeff_tau * tau - scenarios[k] @ v2[k] - alpha[k]) / support_radius,
            ])

        objective = cp.Minimize(lam * self.epsilon - cp.sum(alpha) / L)
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=getattr(cp, self.solver, cp.ECOS), verbose=False)
            if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
                return w.value
        except Exception:
            pass

        try:
            prob.solve(solver=cp.SCS, verbose=False)
            if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
                return w.value
        except Exception:
            pass

        return self._fallback_robust_weights(scenarios)

    def _fallback_robust_weights(self, scenarios: np.ndarray) -> np.ndarray:
        """Fallback: simple robust estimation when optimization fails."""
        L, N = scenarios.shape

        median_returns = np.median(scenarios, axis=0)
        mad = np.median(np.abs(scenarios - median_returns), axis=0)
        robust_std = 1.4826 * mad

        scores = median_returns - self.epsilon * robust_std
        scores_shifted = scores - scores.max()
        weights = np.exp(scores_shifted / (np.abs(scores_shifted).mean() + 1e-8))

        return project_to_simplex(weights)


class RobustMeanCVaRPolicy(BasePolicy):
    """Simplified Robust Mean-CVaR without copula modeling.

    A more tractable version that only considers marginal uncertainty.
    """

    def __init__(
        self,
        alpha: float = 0.95,
        epsilon: float = 0.1,
        risk_weight: float = 0.5,
    ):
        """Initialize Robust Mean-CVaR policy.

        Args:
            alpha: CVaR confidence level.
            epsilon: Uncertainty radius for worst-case analysis.
            risk_weight: Trade-off between mean (1) and CVaR (0).
        """
        super().__init__("Robust-MeanCVaR")
        self.alpha = alpha
        self.epsilon = epsilon
        self.risk_weight = risk_weight

    def reset(self, n_assets: int) -> None:
        self.n_assets = n_assets

    def act(self, state: np.ndarray) -> np.ndarray:
        """Compute robust Mean-CVaR optimal weights.

        Args:
            state: Historical log-returns matrix of shape (L, N).

        Returns:
            Portfolio weights of shape (N,).
        """
        L, N = state.shape

        if N == 1:
            return np.array([1.0])

        if L < 3:
            return np.ones(N) / N

        scenarios = state

        w = cp.Variable(N)
        zeta = cp.Variable()
        u = cp.Variable(L)

        portfolio_returns = scenarios @ w
        losses = -portfolio_returns

        mean_return = cp.sum(portfolio_returns) / L
        cvar = zeta + (1 / (L * (1 - self.alpha))) * cp.sum(u)

        portfolio_variance = cp.quad_form(w, np.cov(scenarios.T) + 1e-6 * np.eye(N))

        objective = cp.Minimize(
            -self.risk_weight * mean_return
            + (1 - self.risk_weight) * cvar
            + self.epsilon * cp.sqrt(portfolio_variance)
        )

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
