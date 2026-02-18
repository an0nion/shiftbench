"""KLIEP: Kullback-Leibler Importance Estimation Procedure.

Direct density ratio estimation via KL divergence minimization.
Optimization-based approach (SLSQP) vs. uLSIF's closed-form solution.

Key advantages:
- Directly minimizes KL divergence between true and estimated ratios
- Guaranteed non-negative weights via optimization constraints
- Can handle cases where uLSIF's closed-form solution gives negative weights

Key disadvantages:
- Requires iterative optimization (slower than uLSIF)
- Sensitive to initialization and hyperparameters
- May not converge in high-dimensional settings

References:
    Sugiyama et al. 2008. "Direct Importance Estimation with Model Selection
    and Its Application to Covariate Shift Adaptation"
    Neural Information Processing Systems (NIPS) 2008.

    Sugiyama et al. 2012. "Density Ratio Estimation in Machine Learning"
    Cambridge University Press. (Chapter 5)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import warnings

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

from shiftbench.baselines.base import (
    BaselineMethod,
    CohortDecision,
    MethodMetadata,
)


class KLIEPBaseline(BaselineMethod):
    """KLIEP density ratio estimator + Empirical-Bernstein bounds.

    Estimates importance weights w(x) = p_target(x) / p_cal(x) using:
    1. Gaussian kernel basis functions centered on calibration samples
    2. Constrained optimization to maximize KL divergence: max sum(log(K_target @ alpha))
    3. Subject to: alpha >= 0, mean(K_cal @ alpha) = 1
    4. Self-normalization to ensure mean(w) = 1

    Does NOT have stability gating, so never returns NO-GUARANTEE.
    """

    def __init__(
        self,
        n_basis: int = 100,
        sigma: Optional[float] = None,
        max_iter: int = 10000,
        tol: float = 1e-6,
        random_state: int = 42,
        **kwargs,
    ):
        """Initialize KLIEP with hyperparameters.

        Args:
            n_basis: Number of Gaussian kernel centers (subset of calibration samples)
            sigma: Kernel bandwidth (std dev). If None, use median heuristic.
            max_iter: Maximum number of optimization iterations
            tol: Convergence tolerance for optimization
            random_state: Random seed for basis center selection
            **kwargs: Additional hyperparameters
        """
        super().__init__(
            n_basis=n_basis,
            sigma=sigma,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            **kwargs,
        )
        self._fitted_params = None  # Store alpha weights for diagnostics

    def estimate_weights(
        self,
        X_cal: np.ndarray,
        X_target: np.ndarray,
        domain_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Estimate importance weights using KLIEP.

        Args:
            X_cal: Calibration features (n_cal, n_features)
            X_target: Target features (n_target, n_features)
            domain_labels: Not used (KLIEP is direct method)

        Returns:
            weights: Importance weights for calibration samples (n_cal,)
        """
        n_cal = len(X_cal)
        n_target = len(X_target)
        n_basis = min(self.hyperparameters["n_basis"], n_cal)

        # Step 1: Select kernel centers (random subset of calibration samples)
        rng = np.random.RandomState(self.hyperparameters["random_state"])
        center_indices = rng.choice(n_cal, size=n_basis, replace=False)
        centers = X_cal[center_indices]

        # Step 2: Determine kernel bandwidth (sigma)
        sigma = self.hyperparameters["sigma"]
        if sigma is None:
            # Median heuristic: sigma = median of pairwise distances
            sample_size = min(1000, n_cal)
            sample_indices = rng.choice(n_cal, size=sample_size, replace=False)
            dists = cdist(X_cal[sample_indices], X_cal[sample_indices], metric="euclidean")
            sigma = np.median(dists[dists > 0])
            if sigma == 0:
                sigma = 1.0  # Fallback

        # Step 3: Compute kernel matrices
        # K_cal[i, j] = k(x_cal_i, center_j)
        # K_target[i, j] = k(x_target_i, center_j)
        K_cal = self._gaussian_kernel(X_cal, centers, sigma)
        K_target = self._gaussian_kernel(X_target, centers, sigma)

        # Step 4: Solve for alpha via constrained optimization
        # Maximize: J(alpha) = (1/n_target) * sum_i log(K_target[i,:] @ alpha)
        # Subject to: alpha >= 0 (element-wise)
        #             (1/n_cal) * sum_i (K_cal[i,:] @ alpha) = 1

        # Define objective function (negative KL divergence for minimization)
        def objective(alpha):
            """Negative log-likelihood (we minimize, so negate)."""
            K_target_alpha = K_target @ alpha
            # Add small epsilon for numerical stability
            K_target_alpha = np.maximum(K_target_alpha, 1e-20)
            return -np.mean(np.log(K_target_alpha))

        # Define gradient of objective
        def objective_grad(alpha):
            """Gradient of negative log-likelihood."""
            K_target_alpha = K_target @ alpha
            K_target_alpha = np.maximum(K_target_alpha, 1e-20)
            # d/d_alpha [-mean(log(K @ alpha))] = -K^T @ (1 / (K @ alpha)) / n_target
            grad = -K_target.T @ (1.0 / K_target_alpha) / n_target
            return grad

        # Define constraints
        # Constraint: mean(K_cal @ alpha) = 1
        def constraint_eq(alpha):
            return np.mean(K_cal @ alpha) - 1.0

        def constraint_eq_jac(alpha):
            return K_cal.mean(axis=0)

        # Initial guess: uniform weights (satisfies constraint)
        alpha_init = np.ones(n_basis) / n_basis
        # Adjust to satisfy constraint
        mean_K_cal = K_cal.mean(axis=0)
        alpha_init = alpha_init / np.mean(K_cal @ alpha_init)

        # Bounds: alpha >= 0
        bounds = [(0, None) for _ in range(n_basis)]

        # Constraints for scipy.optimize.minimize
        constraints = {
            'type': 'eq',
            'fun': constraint_eq,
            'jac': constraint_eq_jac,
        }

        # Optimize
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            result = minimize(
                objective,
                alpha_init,
                method='SLSQP',
                jac=objective_grad,
                bounds=bounds,
                constraints=constraints,
                options={
                    'maxiter': self.hyperparameters["max_iter"],
                    'ftol': self.hyperparameters["tol"],
                },
            )

        if not result.success:
            warnings.warn(
                f"KLIEP optimization did not converge: {result.message}. "
                f"Using best solution found."
            )

        alpha = result.x

        # Ensure non-negativity (optimization should handle this, but be safe)
        alpha = np.maximum(alpha, 0)

        # Step 5: Compute importance weights w = K_cal @ alpha
        weights_raw = K_cal @ alpha

        # Step 6: Ensure positivity and self-normalize
        weights_raw = np.maximum(weights_raw, 1e-8)
        weights = weights_raw / weights_raw.mean()

        # Store for diagnostics
        self._fitted_params = {
            "alpha": alpha,
            "sigma": sigma,
            "centers": centers,
            "n_basis": n_basis,
            "optimization_success": result.success,
            "optimization_nit": result.nit,
            "optimization_objective": result.fun,
        }

        return weights

    def estimate_bounds(
        self,
        y_cal: np.ndarray,
        predictions_cal: np.ndarray,
        cohort_ids_cal: np.ndarray,
        weights: np.ndarray,
        tau_grid: List[float],
        alpha: float = 0.05,
    ) -> List[CohortDecision]:
        """Estimate PPV lower bounds using Empirical-Bernstein.

        NOTE: KLIEP does NOT have stability gating, so never returns NO-GUARANTEE.
        This may produce unreliable bounds when shift is severe.
        """
        self.validate_inputs(y_cal, predictions_cal, cohort_ids_cal, weights)

        # Import EB bound utilities (assume RAVEL is available)
        from ravel.bounds.empirical_bernstein import eb_lower_bound
        from ravel.bounds.p_value import eb_p_value
        from ravel.bounds.weighted_stats import weighted_stats_01

        all_decisions = []
        unique_cohorts = np.unique(cohort_ids_cal)

        for tau in tau_grid:
            for cohort_id in unique_cohorts:
                # Filter to cohort predicted positives
                cohort_mask = (cohort_ids_cal == cohort_id)
                pos_mask = cohort_mask & (predictions_cal == 1)

                y_cohort = y_cal[pos_mask]
                w_cohort = weights[pos_mask]

                if len(y_cohort) < 5:
                    # Insufficient data
                    decision = CohortDecision(
                        cohort_id=cohort_id,
                        tau=tau,
                        decision="ABSTAIN",
                        mu_hat=np.nan,
                        var_hat=np.nan,
                        n_eff=0,
                        lower_bound=np.nan,
                        p_value=1.0,
                        diagnostics={"reason": "insufficient_positives"},
                    )
                else:
                    # Compute weighted stats
                    stats = weighted_stats_01(y_cohort, w_cohort)

                    # Compute EB lower bound
                    lb = eb_lower_bound(stats.mu, stats.var, stats.n_eff, alpha)

                    # Compute p-value testing H0: PPV < tau
                    pval = eb_p_value(stats.mu, stats.var, stats.n_eff, tau)

                    # Decision: CERTIFY if lower bound >= tau
                    dec = "CERTIFY" if lb >= tau else "ABSTAIN"

                    decision = CohortDecision(
                        cohort_id=cohort_id,
                        tau=tau,
                        decision=dec,
                        mu_hat=stats.mu,
                        var_hat=stats.var,
                        n_eff=stats.n_eff,
                        lower_bound=lb,
                        p_value=pval,
                        diagnostics=self.get_diagnostics(),
                    )

                all_decisions.append(decision)

        return all_decisions

    def get_metadata(self) -> MethodMetadata:
        """Return KLIEP metadata."""
        return MethodMetadata(
            name="kliep",
            version="1.0.0",
            description=(
                "Kullback-Leibler Importance Estimation Procedure. "
                "Direct density ratio estimation via KL divergence optimization."
            ),
            paper_title="Direct Importance Estimation with Model Selection and Its Application to Covariate Shift Adaptation",
            paper_url="https://papers.nips.cc/paper/2007/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract.html",
            code_url="https://github.com/anthropics/shift-bench",
            hyperparameters=self.hyperparameters,
            supports_abstention=False,  # No stability gating
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return KLIEP-specific diagnostics."""
        if self._fitted_params is None:
            return {}

        return {
            "method": "kliep",
            "sigma": self._fitted_params["sigma"],
            "n_basis": self._fitted_params["n_basis"],
            "alpha_min": float(self._fitted_params["alpha"].min()),
            "alpha_max": float(self._fitted_params["alpha"].max()),
            "alpha_std": float(self._fitted_params["alpha"].std()),
            "optimization_success": self._fitted_params["optimization_success"],
            "optimization_nit": self._fitted_params["optimization_nit"],
            "optimization_objective": float(self._fitted_params["optimization_objective"]),
        }

    @staticmethod
    def _gaussian_kernel(X: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
        """Compute Gaussian kernel matrix K[i,j] = exp(-||x_i - c_j||^2 / (2*sigma^2)).

        Args:
            X: Data points (n, d)
            centers: Kernel centers (m, d)
            sigma: Kernel bandwidth

        Returns:
            K: Kernel matrix (n, m)
        """
        dists_sq = cdist(X, centers, metric="sqeuclidean")
        return np.exp(-dists_sq / (2 * sigma ** 2))


def create_kliep_baseline(**kwargs) -> KLIEPBaseline:
    """Create a KLIEP baseline with default or custom hyperparameters.

    Examples:
        >>> # Default KLIEP
        >>> kliep = create_kliep_baseline()

        >>> # More basis functions (higher capacity, slower)
        >>> kliep_large = create_kliep_baseline(n_basis=200)

        >>> # Manual bandwidth selection, longer optimization
        >>> kliep_custom = create_kliep_baseline(sigma=1.0, max_iter=20000)
    """
    return KLIEPBaseline(**kwargs)
