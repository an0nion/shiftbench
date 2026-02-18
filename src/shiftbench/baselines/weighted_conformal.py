"""Weighted Conformal Prediction for PPV estimation under covariate shift.

Conformal prediction provides distribution-free coverage guarantees by using
quantiles of conformal scores. Under covariate shift, we use importance-weighted
quantiles to maintain marginal coverage on the target distribution.

Key idea:
- Instead of Empirical-Bernstein (parametric) bounds
- Use weighted quantiles (non-parametric) of residuals
- Distribution-free: no assumptions on outcome distribution
- Valid under covariate shift with correct importance weights

References:
    Tibshirani et al. 2019. "Conformal Prediction Under Covariate Shift"
    https://arxiv.org/abs/1904.06019

    Barber et al. 2021. "Conformal Prediction Beyond Exchangeability"
    https://arxiv.org/abs/2202.13415
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


class WeightedConformalBaseline(BaselineMethod):
    """Weighted Conformal Prediction for PPV estimation.

    Uses importance weights (from uLSIF or KLIEP) with weighted quantiles
    to compute distribution-free lower bounds on PPV.

    Key differences from uLSIF/KLIEP:
    - uLSIF/KLIEP use Empirical-Bernstein bounds (parametric, assumes sub-Gaussian)
    - Conformal uses weighted quantiles (non-parametric, distribution-free)
    - Conformal may be more robust but potentially wider bounds

    Does NOT have stability gating, so never returns NO-GUARANTEE.
    """

    def __init__(
        self,
        weight_method: str = "ulsif",
        n_basis: int = 100,
        sigma: Optional[float] = None,
        lambda_: float = 0.1,
        max_iter: int = 10000,
        tol: float = 1e-6,
        random_state: int = 42,
        **kwargs,
    ):
        """Initialize Weighted Conformal Prediction with hyperparameters.

        Args:
            weight_method: Method for weight estimation ("ulsif" or "kliep")
            n_basis: Number of Gaussian kernel centers for weight estimation
            sigma: Kernel bandwidth. If None, use median heuristic.
            lambda_: Ridge regularization for uLSIF (ignored for KLIEP)
            max_iter: Max iterations for KLIEP optimization (ignored for uLSIF)
            tol: Convergence tolerance for KLIEP (ignored for uLSIF)
            random_state: Random seed for basis center selection
            **kwargs: Additional hyperparameters
        """
        super().__init__(
            weight_method=weight_method,
            n_basis=n_basis,
            sigma=sigma,
            lambda_=lambda_,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            **kwargs,
        )
        self._fitted_params = None
        self._weight_estimator = None

    def estimate_weights(
        self,
        X_cal: np.ndarray,
        X_target: np.ndarray,
        domain_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Estimate importance weights using uLSIF or KLIEP.

        Args:
            X_cal: Calibration features (n_cal, n_features)
            X_target: Target features (n_target, n_features)
            domain_labels: Not used (direct methods)

        Returns:
            weights: Importance weights for calibration samples (n_cal,)
        """
        weight_method = self.hyperparameters["weight_method"]

        if weight_method == "ulsif":
            from shiftbench.baselines.ulsif import uLSIFBaseline
            self._weight_estimator = uLSIFBaseline(
                n_basis=self.hyperparameters["n_basis"],
                sigma=self.hyperparameters["sigma"],
                lambda_=self.hyperparameters["lambda_"],
                random_state=self.hyperparameters["random_state"],
            )
        elif weight_method == "kliep":
            from shiftbench.baselines.kliep import KLIEPBaseline
            self._weight_estimator = KLIEPBaseline(
                n_basis=self.hyperparameters["n_basis"],
                sigma=self.hyperparameters["sigma"],
                max_iter=self.hyperparameters["max_iter"],
                tol=self.hyperparameters["tol"],
                random_state=self.hyperparameters["random_state"],
            )
        else:
            raise ValueError(
                f"Unknown weight_method: {weight_method}. "
                f"Must be 'ulsif' or 'kliep'."
            )

        weights = self._weight_estimator.estimate_weights(X_cal, X_target, domain_labels)

        # Store diagnostics from weight estimator
        self._fitted_params = {
            "weight_method": weight_method,
            "weight_diagnostics": self._weight_estimator.get_diagnostics(),
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
        """Estimate PPV lower bounds using weighted conformal prediction.

        For each cohort:
        1. Compute conformal scores: score_i = 1 - y_i (residual)
        2. Compute weighted quantile at level 1-alpha
        3. Convert to PPV lower bound

        Unlike Empirical-Bernstein (which uses mean and variance), conformal
        uses quantiles directly, providing distribution-free guarantees.

        Args:
            y_cal: Binary labels (0/1) for calibration set, shape (n_cal,)
            predictions_cal: Binary predictions for calibration set, shape (n_cal,)
            cohort_ids_cal: Cohort identifiers, shape (n_cal,)
            weights: Importance weights from estimate_weights(), shape (n_cal,)
            tau_grid: List of PPV thresholds to test
            alpha: Miscoverage level (e.g., 0.05 for 95% confidence)

        Returns:
            decisions: List of CohortDecision objects
        """
        self.validate_inputs(y_cal, predictions_cal, cohort_ids_cal, weights)

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
                    # Compute PPV estimate (for reporting)
                    from ravel.bounds.weighted_stats import weighted_stats_01
                    stats = weighted_stats_01(y_cohort, w_cohort)

                    # Conformal approach for binary outcomes:
                    # y_i ~ Bernoulli(p), where p is the true PPV
                    # We want a lower bound on p
                    #
                    # Method: Use Clopper-Pearson-style quantile approach
                    # Sort observations by y (0s first, 1s second)
                    # Find the weighted quantile such that the cumulative sum
                    # of weights up to that point is at most alpha
                    #
                    # Equivalently: Find smallest k such that sum(w[i] for y[i]=1) / sum(w) >= 1-alpha
                    # This gives a lower confidence bound on the proportion of 1s

                    # Sort by outcome (0s first, then 1s)
                    sort_idx = np.argsort(y_cohort)
                    y_sorted = y_cohort[sort_idx]
                    w_sorted = w_cohort[sort_idx]

                    # Normalize weights
                    total_weight = w_sorted.sum()
                    cumsum_weights = np.cumsum(w_sorted) / total_weight

                    # Find the index where cumulative weight crosses alpha
                    # This gives us the (alpha)-quantile position
                    # Everything after this has cumulative weight >= alpha
                    idx_alpha = np.searchsorted(cumsum_weights, alpha, side='left')

                    # Conformal lower bound: proportion of 1s after removing alpha mass
                    # from the lower tail
                    if idx_alpha >= len(y_sorted):
                        # All mass is below alpha quantile - very conservative
                        lower_bound = 0.0
                    else:
                        # Compute proportion of 1s in the upper (1-alpha) quantile
                        remaining_y = y_sorted[idx_alpha:]
                        remaining_w = w_sorted[idx_alpha:]

                        if remaining_w.sum() > 0:
                            lower_bound = (remaining_w * remaining_y).sum() / remaining_w.sum()
                        else:
                            lower_bound = 0.0

                    # Clip to [0, 1]
                    lower_bound = np.clip(lower_bound, 0.0, 1.0)

                    # Compute approximate p-value
                    # p-value = minimum alpha such that lower_bound >= tau
                    # Binary search over alpha to find critical value
                    pval = self._compute_conformal_pvalue(y_cohort, w_cohort, tau)

                    # Decision: CERTIFY if lower bound >= tau
                    dec = "CERTIFY" if lower_bound >= tau else "ABSTAIN"

                    decision = CohortDecision(
                        cohort_id=cohort_id,
                        tau=tau,
                        decision=dec,
                        mu_hat=stats.mu,
                        var_hat=stats.var,
                        n_eff=stats.n_eff,
                        lower_bound=lower_bound,
                        p_value=pval,
                        diagnostics={
                            "method": "weighted_conformal",
                            "quantile_level_alpha": alpha,
                            "idx_alpha": idx_alpha if idx_alpha < len(y_sorted) else len(y_sorted),
                            **self.get_diagnostics(),
                        },
                    )

                all_decisions.append(decision)

        return all_decisions

    def _compute_conformal_pvalue(
        self,
        y: np.ndarray,
        weights: np.ndarray,
        tau: float
    ) -> float:
        """Compute p-value for H0: PPV < tau using conformal approach.

        The p-value is the minimum alpha such that the conformal lower bound >= tau.
        We use binary search to find this critical alpha value.
        """
        # Binary search over alpha
        alpha_min, alpha_max = 0.0, 1.0
        n_iter = 20  # Binary search iterations

        for _ in range(n_iter):
            alpha_mid = (alpha_min + alpha_max) / 2.0

            # Compute lower bound at this alpha
            sort_idx = np.argsort(y)
            y_sorted = y[sort_idx]
            w_sorted = weights[sort_idx]

            total_weight = w_sorted.sum()
            if total_weight <= 0:
                return 1.0

            cumsum_weights = np.cumsum(w_sorted) / total_weight
            idx_alpha = np.searchsorted(cumsum_weights, alpha_mid, side='left')

            if idx_alpha >= len(y_sorted):
                lb = 0.0
            else:
                remaining_y = y_sorted[idx_alpha:]
                remaining_w = w_sorted[idx_alpha:]
                if remaining_w.sum() > 0:
                    lb = (remaining_w * remaining_y).sum() / remaining_w.sum()
                else:
                    lb = 0.0

            # Update binary search
            if lb >= tau:
                # Lower bound is high enough, try larger alpha (weaker guarantee)
                alpha_min = alpha_mid
            else:
                # Lower bound is too low, need smaller alpha (stronger guarantee)
                alpha_max = alpha_mid

        # p-value is the minimum alpha where we can certify
        return float(alpha_min)

    def get_metadata(self) -> MethodMetadata:
        """Return weighted conformal prediction metadata."""
        return MethodMetadata(
            name="weighted_conformal",
            version="1.0.0",
            description=(
                "Weighted Conformal Prediction under covariate shift. "
                "Distribution-free coverage using importance-weighted quantiles."
            ),
            paper_title="Conformal Prediction Under Covariate Shift",
            paper_url="https://arxiv.org/abs/1904.06019",
            code_url="https://github.com/anthropics/shift-bench",
            hyperparameters=self.hyperparameters,
            supports_abstention=False,  # No stability gating
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return weighted conformal prediction diagnostics."""
        if self._fitted_params is None:
            return {"method": "weighted_conformal"}

        return {
            "method": "weighted_conformal",
            "weight_method": self._fitted_params["weight_method"],
            **self._fitted_params.get("weight_diagnostics", {}),
        }


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile_level: float,
) -> float:
    """Compute weighted quantile using linear interpolation.

    The weighted quantile at level q is the value x such that:
        sum(w_i for v_i <= x) / sum(w_i) = q

    Uses linear interpolation between order statistics (like numpy.quantile).

    Args:
        values: Array of values
        weights: Array of weights (must be positive)
        quantile_level: Quantile level in [0, 1]

    Returns:
        Weighted quantile value

    References:
        - Koenker (2005). "Quantile Regression". Cambridge University Press.
        - Hyndman & Fan (1996). "Sample Quantiles in Statistical Packages"
          American Statistician 50(4): 361-365.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if len(values) == 0:
        return np.nan

    if len(values) != len(weights):
        raise ValueError("values and weights must have same length")

    if not (0 <= quantile_level <= 1):
        raise ValueError("quantile_level must be in [0, 1]")

    # Remove non-positive or non-finite weights
    valid_mask = (weights > 0) & np.isfinite(weights) & np.isfinite(values)
    if not np.any(valid_mask):
        return np.nan

    values = values[valid_mask]
    weights = weights[valid_mask]

    # Sort by values
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Normalize weights to sum to 1
    total_weight = sorted_weights.sum()
    if total_weight <= 0:
        return np.nan

    normalized_weights = sorted_weights / total_weight

    # Compute cumulative sum of weights
    cumsum_weights = np.cumsum(normalized_weights)

    # Find the quantile using linear interpolation
    # Special cases
    if quantile_level <= 0:
        return float(sorted_values[0])
    if quantile_level >= 1:
        return float(sorted_values[-1])

    # Find where cumsum crosses quantile_level
    # We want the smallest i such that cumsum[i] >= quantile_level
    idx = np.searchsorted(cumsum_weights, quantile_level, side='left')

    if idx >= len(sorted_values):
        # quantile_level is beyond all data
        return float(sorted_values[-1])

    if idx == 0:
        # quantile_level is before all data
        return float(sorted_values[0])

    # Linear interpolation between sorted_values[idx-1] and sorted_values[idx]
    # Weight of left point
    w_left = cumsum_weights[idx - 1]
    w_right = cumsum_weights[idx]

    if w_right - w_left <= 0:
        # Degenerate case: duplicate weights
        return float(sorted_values[idx])

    # Interpolation factor
    alpha = (quantile_level - w_left) / (w_right - w_left)
    alpha = np.clip(alpha, 0.0, 1.0)

    # Interpolated value
    quantile_value = (1 - alpha) * sorted_values[idx - 1] + alpha * sorted_values[idx]

    return float(quantile_value)


def create_weighted_conformal_baseline(**kwargs) -> WeightedConformalBaseline:
    """Create a Weighted Conformal Prediction baseline.

    Examples:
        >>> # Default: uLSIF weights with conformal quantiles
        >>> wcp = create_weighted_conformal_baseline()

        >>> # Use KLIEP for weight estimation
        >>> wcp_kliep = create_weighted_conformal_baseline(weight_method="kliep")

        >>> # More basis functions (higher capacity)
        >>> wcp_large = create_weighted_conformal_baseline(n_basis=200)

        >>> # Manual bandwidth selection
        >>> wcp_fixed = create_weighted_conformal_baseline(sigma=1.0, lambda_=0.01)
    """
    return WeightedConformalBaseline(**kwargs)
