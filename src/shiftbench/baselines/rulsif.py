"""RULSIF: Relative unconstrained Least-Squares Importance Fitting.

Extension of uLSIF for relative density ratio estimation. More stable than standard
density ratio when distributions differ significantly.

Key advantages over uLSIF:
- More stable when calibration density is close to zero (p_cal ≈ 0)
- Estimates relative ratio: r(x) = p_target(x) / (α*p_target(x) + (1-α)*p_cal(x))
- α parameter controls "relative" vs "absolute" ratio (α=0 → standard uLSIF)
- Still has closed-form solution (no iterative optimization)

References:
    Yamada et al. 2011. "Change-Point Detection in Time-Series Data by Direct
    Density-Ratio Estimation" Neural Networks 24(7):637-649.

    Yamada et al. 2013. "Relative Density-Ratio Estimation for Robust Distribution
    Comparison" Neural Computation 25(5):1324-1370.

    Sugiyama et al. 2012. "Density Ratio Estimation in Machine Learning"
    Cambridge University Press. (Chapter 9)
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


class RULSIFBaseline(BaselineMethod):
    """RULSIF relative density ratio estimator + Empirical-Bernstein bounds.

    Estimates relative importance weights w(x) = p_target(x) / p_α(x) where:
        p_α(x) = α * p_target(x) + (1-α) * p_cal(x)

    This is more stable than standard density ratio (uLSIF) when distributions
    differ significantly because:
    - When α > 0, the denominator never goes to zero even if p_cal(x) ≈ 0
    - α = 0.0 → standard density ratio (equivalent to uLSIF)
    - α = 0.5 → most stable (denominator is average of both distributions)
    - α = 1.0 → all weights = 1 (no reweighting)

    Uses:
    1. Gaussian kernel basis functions centered on calibration samples
    2. Ridge-regularized least-squares (same as uLSIF but with modified target)
    3. Self-normalization to ensure mean(w) = 1

    Does NOT have stability gating, so never returns NO-GUARANTEE.
    """

    def __init__(
        self,
        n_basis: int = 100,
        sigma: Optional[float] = None,
        lambda_: float = 0.1,
        alpha: float = 0.1,
        random_state: int = 42,
        **kwargs,
    ):
        """Initialize RULSIF with hyperparameters.

        Args:
            n_basis: Number of Gaussian kernel centers (subset of calibration samples)
            sigma: Kernel bandwidth (std dev). If None, use median heuristic.
            lambda_: Ridge regularization parameter (higher = smoother, more stable)
            alpha: Relative parameter in [0, 1]. Controls stability vs. standard ratio.
                   - 0.0 = standard density ratio (equivalent to uLSIF)
                   - 0.1 = recommended default (slight stabilization)
                   - 0.5 = maximum stability
                   - 1.0 = no reweighting (all weights = 1)
            random_state: Random seed for basis center selection
            **kwargs: Additional hyperparameters
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        super().__init__(
            n_basis=n_basis,
            sigma=sigma,
            lambda_=lambda_,
            alpha=alpha,
            random_state=random_state,
            **kwargs,
        )
        self._fitted_params = None  # Store theta weights for diagnostics

    def estimate_weights(
        self,
        X_cal: np.ndarray,
        X_target: np.ndarray,
        domain_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Estimate relative importance weights using RULSIF.

        Args:
            X_cal: Calibration features (n_cal, n_features)
            X_target: Target features (n_target, n_features)
            domain_labels: Not used (RULSIF is direct method)

        Returns:
            weights: Relative importance weights for calibration samples (n_cal,)
        """
        n_cal = len(X_cal)
        n_target = len(X_target)
        n_basis = min(self.hyperparameters["n_basis"], n_cal)
        alpha_rel = self.hyperparameters["alpha"]

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

        # Step 4: Solve for theta (kernel weights) using RULSIF objective
        # RULSIF minimizes: ||w_α - 1||² where w_α = K @ theta
        # and w_α should approximate p_target / p_α
        #
        # The key difference from uLSIF:
        # - uLSIF: Minimize E_cal[(K@theta)²] - 2*E_target[K@theta]
        # - RULSIF: Minimize E_α[(K@theta)²] - 2*E_target[K@theta]
        #   where E_α = (1-α)*E_cal + α*E_target
        #
        # This leads to:
        # Minimize: theta^T H_α theta - 2 h^T theta + λ||theta||²
        # where H_α = (1-α)*H_cal + α*H_target
        #       H_cal = K_cal^T K_cal / n_cal
        #       H_target = K_target^T K_target / n_target
        #       h = K_target^T 1 / n_target

        # Compute H_cal = K_cal^T K_cal / n_cal
        H_cal = (K_cal.T @ K_cal) / n_cal

        # Compute H_target = K_target^T K_target / n_target
        H_target = (K_target.T @ K_target) / n_target

        # Compute H_α = (1-α)*H_cal + α*H_target
        H_alpha = (1 - alpha_rel) * H_cal + alpha_rel * H_target

        # Compute h = K_target^T 1 / n_target (mean of target kernel features)
        h = K_target.mean(axis=0)

        # Ridge regression: theta = (H_α + λI)^{-1} h
        lambda_ = self.hyperparameters["lambda_"]
        theta = np.linalg.solve(H_alpha + lambda_ * np.eye(n_basis), h)

        # Ensure non-negativity
        theta = np.maximum(theta, 0)

        # Step 5: Compute relative importance weights w = K_cal @ theta
        weights_raw = K_cal @ theta

        # Step 6: Ensure positivity and self-normalize
        weights_raw = np.maximum(weights_raw, 1e-8)
        weights = weights_raw / weights_raw.mean()

        # Store for diagnostics
        self._fitted_params = {
            "theta": theta,
            "sigma": sigma,
            "centers": centers,
            "n_basis": n_basis,
            "alpha_rel": alpha_rel,
            "weights_raw_std": float(weights_raw.std()),
            "weights_raw_min": float(weights_raw.min()),
            "weights_raw_max": float(weights_raw.max()),
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

        NOTE: RULSIF does NOT have stability gating, so never returns NO-GUARANTEE.
        This may produce unreliable bounds when shift is severe (though RULSIF
        should be more stable than uLSIF in such cases).

        Args:
            y_cal: Binary labels for calibration set (n_cal,)
            predictions_cal: Binary predictions for calibration set (n_cal,)
            cohort_ids_cal: Cohort identifiers for calibration set (n_cal,)
            weights: Importance weights from estimate_weights() (n_cal,)
            tau_grid: List of PPV thresholds to test
            alpha: Significance level (e.g., 0.05 for 95% confidence)

        Returns:
            decisions: List of CohortDecision objects for each (cohort, tau) pair
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
        """Return RULSIF metadata."""
        return MethodMetadata(
            name="rulsif",
            version="1.0.0",
            description=(
                "Relative unconstrained Least-Squares Importance Fitting. "
                "Estimates relative density ratio for stable distribution comparison."
            ),
            paper_title="Relative Density-Ratio Estimation for Robust Distribution Comparison",
            paper_url="https://doi.org/10.1162/NECO_a_00442",
            code_url="https://github.com/anthropics/shift-bench",
            hyperparameters=self.hyperparameters,
            supports_abstention=False,  # No stability gating
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return RULSIF-specific diagnostics."""
        if self._fitted_params is None:
            return {}

        return {
            "method": "rulsif",
            "sigma": self._fitted_params["sigma"],
            "n_basis": self._fitted_params["n_basis"],
            "alpha_rel": self._fitted_params["alpha_rel"],
            "theta_min": float(self._fitted_params["theta"].min()),
            "theta_max": float(self._fitted_params["theta"].max()),
            "theta_std": float(self._fitted_params["theta"].std()),
            "weights_raw_std": self._fitted_params["weights_raw_std"],
            "weights_raw_min": self._fitted_params["weights_raw_min"],
            "weights_raw_max": self._fitted_params["weights_raw_max"],
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


def create_rulsif_baseline(**kwargs) -> RULSIFBaseline:
    """Create a RULSIF baseline with default or custom hyperparameters.

    Examples:
        >>> # Default RULSIF (alpha=0.1 for slight stabilization)
        >>> rulsif = create_rulsif_baseline()

        >>> # More stable RULSIF (alpha=0.5)
        >>> rulsif_stable = create_rulsif_baseline(alpha=0.5)

        >>> # Standard density ratio (alpha=0.0, equivalent to uLSIF)
        >>> rulsif_standard = create_rulsif_baseline(alpha=0.0)

        >>> # More basis functions (higher capacity, slower)
        >>> rulsif_large = create_rulsif_baseline(n_basis=200, alpha=0.1)

        >>> # Manual bandwidth selection
        >>> rulsif_fixed = create_rulsif_baseline(sigma=1.0, lambda_=0.01, alpha=0.1)
    """
    return RULSIFBaseline(**kwargs)
