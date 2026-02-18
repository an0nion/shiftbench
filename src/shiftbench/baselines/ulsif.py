"""uLSIF: unconstrained Least-Squares Importance Fitting.

Direct density ratio estimation via squared loss minimization.
More stable than KLIEP (KL minimization) due to closed-form solution.

Key advantages:
- Closed-form solution (no iterative optimization)
- Automatic regularization via ridge penalty
- Numerically stable
- Works well with Gaussian kernels

References:
    Kanamori et al. 2009. "A Least-squares Approach to Direct Importance Estimation"
    Journal of Machine Learning Research 10:1391-1445.

    Sugiyama et al. 2012. "Density Ratio Estimation in Machine Learning"
    Cambridge University Press. (Chapters 3-4)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial.distance import cdist

from shiftbench.baselines.base import (
    BaselineMethod,
    CohortDecision,
    MethodMetadata,
)


class uLSIFBaseline(BaselineMethod):
    """uLSIF density ratio estimator + Empirical-Bernstein bounds.

    Estimates importance weights w(x) = p_target(x) / p_cal(x) using:
    1. Gaussian kernel basis functions centered on calibration samples
    2. Ridge-regularized least-squares to minimize ||w - true_ratio||²
    3. Self-normalization to ensure mean(w) = 1

    Does NOT have stability gating, so never returns NO-GUARANTEE.
    """

    def __init__(
        self,
        n_basis: int = 100,
        sigma: Optional[float] = None,
        lambda_: float = 0.1,
        random_state: int = 42,
        **kwargs,
    ):
        """Initialize uLSIF with hyperparameters.

        Args:
            n_basis: Number of Gaussian kernel centers (subset of calibration samples)
            sigma: Kernel bandwidth (std dev). If None, use median heuristic.
            lambda_: Ridge regularization parameter (higher = smoother, more stable)
            random_state: Random seed for basis center selection
            **kwargs: Additional hyperparameters
        """
        super().__init__(
            n_basis=n_basis,
            sigma=sigma,
            lambda_=lambda_,
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
        """Estimate importance weights using uLSIF.

        Args:
            X_cal: Calibration features (n_cal, n_features)
            X_target: Target features (n_target, n_features)
            domain_labels: Not used (uLSIF is direct method)

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

        # Step 4: Solve for alpha (kernel weights)
        # Minimize: ||K_cal @ alpha - 1||² + λ ||alpha||²
        # Solution: alpha = (K_cal^T K_cal + λI)^{-1} K_target^T 1

        # Compute H = K_cal^T K_cal / n_cal
        H = (K_cal.T @ K_cal) / n_cal

        # Compute h = K_target^T 1 / n_target (mean of target kernel features)
        h = K_target.mean(axis=0)

        # Ridge regression: alpha = (H + λI)^{-1} h
        lambda_ = self.hyperparameters["lambda_"]
        alpha = np.linalg.solve(H + lambda_ * np.eye(n_basis), h)

        # Ensure non-negativity
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

        NOTE: uLSIF does NOT have stability gating, so never returns NO-GUARANTEE.
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
        """Return uLSIF metadata."""
        return MethodMetadata(
            name="ulsif",
            version="1.0.0",
            description=(
                "Unconstrained Least-Squares Importance Fitting. "
                "Direct density ratio estimation with closed-form solution."
            ),
            paper_title="A Least-squares Approach to Direct Importance Estimation",
            paper_url="https://jmlr.org/papers/v10/kanamori09a.html",
            code_url="https://github.com/anthropics/shift-bench",
            hyperparameters=self.hyperparameters,
            supports_abstention=False,  # No stability gating
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return uLSIF-specific diagnostics."""
        if self._fitted_params is None:
            return {}

        return {
            "method": "ulsif",
            "sigma": self._fitted_params["sigma"],
            "n_basis": self._fitted_params["n_basis"],
            "alpha_min": float(self._fitted_params["alpha"].min()),
            "alpha_max": float(self._fitted_params["alpha"].max()),
            "alpha_std": float(self._fitted_params["alpha"].std()),
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


def create_ulsif_baseline(**kwargs) -> uLSIFBaseline:
    """Create a uLSIF baseline with default or custom hyperparameters.

    Examples:
        >>> # Default uLSIF
        >>> ulsif = create_ulsif_baseline()

        >>> # More basis functions (higher capacity, slower)
        >>> ulsif_large = create_ulsif_baseline(n_basis=200)

        >>> # Manual bandwidth selection
        >>> ulsif_fixed = create_ulsif_baseline(sigma=1.0, lambda_=0.01)
    """
    return uLSIFBaseline(**kwargs)
