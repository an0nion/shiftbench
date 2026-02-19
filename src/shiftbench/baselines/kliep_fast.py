"""KLIEP Fast Mode: Subsample-based KLIEP for large datasets.

The full KLIEP optimization is O(n_cal * n_basis * n_iter) with SLSQP,
which becomes very slow for large datasets like COMPAS (5000+ samples).

Fast mode:
  1. Subsample both X_cal and X_target to n_subsample points
  2. Fit KLIEP on the subsample
  3. Extrapolate weights to full X_cal via kernel evaluation

This trades accuracy for speed: weights for non-subsampled points
are computed via the same kernel basis, so they're consistent but
estimated from less data.
"""

from __future__ import annotations

from typing import Optional
import warnings

import numpy as np
from scipy.spatial.distance import cdist

from shiftbench.baselines.kliep import KLIEPBaseline


class KLIEPFastBaseline(KLIEPBaseline):
    """KLIEP with subsampling for computational efficiency.

    Fits KLIEP on a subsample of cal/target, then extrapolates
    weights to the full calibration set via kernel evaluation.
    """

    def __init__(
        self,
        n_basis: int = 100,
        sigma: Optional[float] = None,
        max_iter: int = 5000,
        tol: float = 1e-6,
        n_subsample_cal: int = 500,
        n_subsample_target: int = 500,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(
            n_basis=n_basis,
            sigma=sigma,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            **kwargs,
        )
        self.n_subsample_cal = n_subsample_cal
        self.n_subsample_target = n_subsample_target

    def estimate_weights(
        self,
        X_cal: np.ndarray,
        X_target: np.ndarray,
        domain_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Estimate importance weights using subsampled KLIEP.

        If n_cal <= n_subsample_cal, falls back to full KLIEP.
        """
        n_cal = len(X_cal)
        n_target = len(X_target)
        rng = np.random.RandomState(self.hyperparameters["random_state"])

        # If small enough, use full KLIEP
        if n_cal <= self.n_subsample_cal and n_target <= self.n_subsample_target:
            return super().estimate_weights(X_cal, X_target, domain_labels)

        # Subsample
        n_sub_cal = min(self.n_subsample_cal, n_cal)
        n_sub_target = min(self.n_subsample_target, n_target)

        idx_cal = rng.choice(n_cal, size=n_sub_cal, replace=False)
        idx_target = rng.choice(n_target, size=n_sub_target, replace=False)

        X_cal_sub = X_cal[idx_cal]
        X_target_sub = X_target[idx_target]

        # Fit KLIEP on subsample
        n_basis = min(self.hyperparameters["n_basis"], n_sub_cal)
        center_indices = rng.choice(n_sub_cal, size=n_basis, replace=False)
        centers = X_cal_sub[center_indices]

        # Bandwidth
        sigma = self.hyperparameters["sigma"]
        if sigma is None:
            sample_size = min(1000, n_sub_cal)
            sample_idx = rng.choice(n_sub_cal, size=sample_size, replace=False)
            dists = cdist(X_cal_sub[sample_idx], X_cal_sub[sample_idx], metric="euclidean")
            sigma = np.median(dists[dists > 0])
            if sigma == 0:
                sigma = 1.0

        # Kernel matrices on subsample
        K_cal_sub = self._gaussian_kernel(X_cal_sub, centers, sigma)
        K_target_sub = self._gaussian_kernel(X_target_sub, centers, sigma)

        # Optimize on subsample (same as parent class)
        from scipy.optimize import minimize as scipy_minimize

        def objective(alpha):
            K_target_alpha = K_target_sub @ alpha
            K_target_alpha = np.maximum(K_target_alpha, 1e-20)
            return -np.mean(np.log(K_target_alpha))

        def objective_grad(alpha):
            K_target_alpha = K_target_sub @ alpha
            K_target_alpha = np.maximum(K_target_alpha, 1e-20)
            return -K_target_sub.T @ (1.0 / K_target_alpha) / n_sub_target

        def constraint_eq(alpha):
            return np.mean(K_cal_sub @ alpha) - 1.0

        def constraint_eq_jac(alpha):
            return K_cal_sub.mean(axis=0)

        alpha_init = np.ones(n_basis) / n_basis
        alpha_init = alpha_init / np.mean(K_cal_sub @ alpha_init)

        bounds = [(0, None) for _ in range(n_basis)]
        constraints = {'type': 'eq', 'fun': constraint_eq, 'jac': constraint_eq_jac}

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            result = scipy_minimize(
                objective, alpha_init, method='SLSQP',
                jac=objective_grad, bounds=bounds, constraints=constraints,
                options={'maxiter': self.hyperparameters["max_iter"],
                         'ftol': self.hyperparameters["tol"]},
            )

        alpha = np.maximum(result.x, 0)

        # Extrapolate: compute weights for ALL calibration points
        K_cal_full = self._gaussian_kernel(X_cal, centers, sigma)
        weights_raw = K_cal_full @ alpha
        weights_raw = np.maximum(weights_raw, 1e-8)
        weights = weights_raw / weights_raw.mean()

        self._fitted_params = {
            "alpha": alpha,
            "sigma": sigma,
            "centers": centers,
            "n_basis": n_basis,
            "optimization_success": result.success,
            "optimization_nit": result.nit,
            "optimization_objective": result.fun,
            "n_subsample_cal": n_sub_cal,
            "n_subsample_target": n_sub_target,
            "subsampled": n_cal > n_sub_cal or n_target > n_sub_target,
        }

        return weights
