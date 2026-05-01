"""Hierarchical Risk Parity (HRP) portfolio optimization."""

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from ..base import BasePolicy, project_to_simplex


class HRPPolicy(BasePolicy):
    """Hierarchical Risk Parity portfolio optimization.

    Uses hierarchical clustering on the correlation matrix to determine
    asset groupings, then allocates risk parity within the hierarchy.

    Reference: Lopez de Prado (2016) - Building Diversified Portfolios that
    Outperform Out of Sample.
    """

    def __init__(self):
        super().__init__("HRP")

    def reset(self, n_assets: int) -> None:
        self.n_assets = n_assets

    def act(self, state: np.ndarray) -> np.ndarray:
        """Compute HRP weights using hierarchical clustering.

        Args:
            state: Historical log-returns matrix of shape (L, N).

        Returns:
            Portfolio weights of shape (N,).
        """
        L, N = state.shape

        if N == 1:
            return np.array([1.0])

        if L < 2:
            return np.ones(N) / N

        # Compute correlation and covariance matrices
        cov = np.cov(state, rowvar=False)
        std = np.sqrt(np.diag(cov))

        # Handle zero std
        std = np.where(std < 1e-10, 1e-10, std)

        corr = cov / np.outer(std, std)
        corr = np.clip(corr, -1, 1)

        # Distance matrix from correlation
        dist = np.sqrt(0.5 * (1 - corr))
        np.fill_diagonal(dist, 0)

        # Hierarchical clustering
        try:
            dist_condensed = squareform(dist, checks=False)
            link = linkage(dist_condensed, method="single")
            sort_idx = leaves_list(link)
        except Exception:
            sort_idx = np.arange(N)

        # Recursive bisection for weight allocation
        weights = self._recursive_bisection(cov, sort_idx)

        return project_to_simplex(weights)

    def _recursive_bisection(
        self, cov: np.ndarray, sort_idx: np.ndarray
    ) -> np.ndarray:
        """Recursively bisect the sorted assets and allocate weights.

        Args:
            cov: Covariance matrix of shape (N, N).
            sort_idx: Sorted asset indices from hierarchical clustering.

        Returns:
            Portfolio weights of shape (N,).
        """
        N = len(sort_idx)
        weights = np.ones(N)

        # Initialize allocation for each asset
        allocations = {i: 1.0 for i in range(N)}

        eps = 1e-10

        # Recursive function to bisect
        def bisect(indices: list, alloc: float):
            if len(indices) <= 1:
                if len(indices) == 1:
                    allocations[indices[0]] = alloc
                return

            # Split into two clusters
            mid = len(indices) // 2
            left_idx = indices[:mid]
            right_idx = indices[mid:]

            # Get original indices
            left_orig = [sort_idx[i] for i in left_idx]
            right_orig = [sort_idx[i] for i in right_idx]

            # Compute cluster variances
            left_var = self._cluster_variance(cov, left_orig)
            right_var = self._cluster_variance(cov, right_orig)

            # Allocate inversely proportional to variance (guard against invalid values)
            left_var = left_var if np.isfinite(left_var) and left_var > eps else eps
            right_var = right_var if np.isfinite(right_var) and right_var > eps else eps
            total_inv_var = 1 / left_var + 1 / right_var
            if not np.isfinite(total_inv_var) or total_inv_var <= 0:
                left_alloc = right_alloc = alloc * 0.5
            else:
                left_alloc = alloc * (1 / left_var) / total_inv_var
                right_alloc = alloc * (1 / right_var) / total_inv_var

            bisect(left_idx, left_alloc)
            bisect(right_idx, right_alloc)

        bisect(list(range(N)), 1.0)

        # Map allocations back to original order
        weights = np.zeros(N)
        for i, orig_idx in enumerate(sort_idx):
            weights[orig_idx] = allocations[i]

        return weights

    def _cluster_variance(self, cov: np.ndarray, indices: list) -> float:
        """Compute variance of an inverse-variance weighted cluster.

        Args:
            cov: Full covariance matrix.
            indices: Asset indices in the cluster.

        Returns:
            Cluster variance.
        """
        eps = 1e-10
        if len(indices) == 1:
            val = cov[indices[0], indices[0]]
            if not np.isfinite(val) or val <= eps:
                return eps
            return val

        # Extract sub-covariance matrix
        sub_cov = cov[np.ix_(indices, indices)]
        if not np.all(np.isfinite(sub_cov)):
            sub_cov = np.nan_to_num(sub_cov, nan=0.0, posinf=0.0, neginf=0.0)
        variances = np.diag(sub_cov)

        # Handle zero variances
        variances = np.where(variances < eps, eps, variances)

        # Inverse variance weights within cluster
        inv_var = 1 / variances
        weights = inv_var / inv_var.sum()

        # Cluster variance
        cluster_var = float(weights @ sub_cov @ weights)
        if not np.isfinite(cluster_var) or cluster_var <= eps:
            return eps
        return cluster_var
