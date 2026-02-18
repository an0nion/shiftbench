"""Base interface for all shift-aware evaluation methods.

All methods submitted to ShiftBench must implement this interface to ensure
standardized evaluation and reproducible receipts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class CohortDecision:
    """Result of evaluating one (cohort, tau) pair.

    Attributes:
        cohort_id: Identifier for the cohort (e.g., scaffold name, demographic group)
        tau: PPV threshold being tested (e.g., 0.7 = 70% positive predictive value)
        decision: One of "CERTIFY", "ABSTAIN", or "NO-GUARANTEE"
            - CERTIFY: PPV ≥ τ with statistical guarantee (lower_bound >= tau)
            - ABSTAIN: Insufficient evidence to certify (lower_bound < tau)
            - NO-GUARANTEE: Method diagnostics failed; cannot make claims
        mu_hat: Point estimate of PPV (weighted or unweighted mean)
        var_hat: Variance estimate (used for confidence bound)
        n_eff: Effective sample size (may differ from nominal n due to weighting)
        lower_bound: One-sided lower confidence bound on PPV (e.g., 95% lower bound)
        p_value: One-sided p-value testing H0: PPV < τ
        diagnostics: Method-specific diagnostics (e.g., PSIS k-hat, ESS, gate status)
    """
    cohort_id: str
    tau: float
    decision: str  # "CERTIFY", "ABSTAIN", or "NO-GUARANTEE"
    mu_hat: float
    var_hat: float
    n_eff: float
    lower_bound: float
    p_value: float
    diagnostics: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate decision matches bound/tau relationship."""
        if self.decision not in {"CERTIFY", "ABSTAIN", "NO-GUARANTEE"}:
            raise ValueError(f"Invalid decision: {self.decision}")

        # Consistency check: CERTIFY implies lower_bound >= tau
        if self.decision == "CERTIFY" and not np.isnan(self.lower_bound):
            if self.lower_bound < self.tau - 1e-6:  # Allow small numerical tolerance
                raise ValueError(
                    f"Inconsistent CERTIFY: lower_bound={self.lower_bound:.4f} < tau={self.tau:.4f}"
                )


@dataclass
class MethodMetadata:
    """Metadata describing a baseline method.

    Attributes:
        name: Short identifier (e.g., "ravel", "ulsif", "weighted_conformal")
        version: Semantic version string (e.g., "1.0.0")
        description: Brief (1-2 sentence) method description
        paper_title: Full paper title (if published)
        paper_url: Link to paper (arXiv, DOI, or conference proceedings)
        code_url: Link to reference implementation
        hyperparameters: Dictionary of hyperparameters used in this run
        supports_abstention: Whether method can return NO-GUARANTEE decisions
    """
    name: str
    version: str
    description: str
    paper_title: Optional[str] = None
    paper_url: Optional[str] = None
    code_url: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    supports_abstention: bool = True


class BaselineMethod(ABC):
    """Abstract base class for all shift-aware evaluation methods.

    To add a new method to ShiftBench:
    1. Subclass this class
    2. Implement estimate_weights() and estimate_bounds()
    3. Implement get_metadata()
    4. Add unit tests verifying weights are valid and bounds are in [0,1]
    5. Validate against published results (if reproducing existing method)

    See docs/ADDING_METHODS.md for full guide with examples.
    """

    def __init__(self, **hyperparameters):
        """Initialize method with hyperparameters.

        Args:
            **hyperparameters: Method-specific configuration (e.g., kernel bandwidth,
                number of folds, alpha level, temperature scaling).
        """
        self.hyperparameters = hyperparameters

    @abstractmethod
    def estimate_weights(
        self,
        X_cal: np.ndarray,
        X_target: np.ndarray,
        domain_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Estimate importance weights from calibration to target distribution.

        Weights represent the density ratio w(x) = p_target(x) / p_cal(x).

        Args:
            X_cal: Calibration features, shape (n_cal, n_features)
            X_target: Target features, shape (n_target, n_features)
            domain_labels: Optional binary labels (0=cal, 1=target) for discriminative methods.
                If not provided, method should construct them internally.

        Returns:
            weights: Importance weights for calibration samples, shape (n_cal,).
                Must satisfy:
                - All weights > 0
                - All weights finite (no inf, no nan)
                - Mean(weights) ≈ 1.0 (self-normalized)

        Raises:
            ValueError: If inputs are invalid (e.g., mismatched dimensions)
            RuntimeError: If weight estimation fails (e.g., optimization divergence)
        """
        pass

    @abstractmethod
    def estimate_bounds(
        self,
        y_cal: np.ndarray,
        predictions_cal: np.ndarray,
        cohort_ids_cal: np.ndarray,
        weights: np.ndarray,
        tau_grid: List[float],
        alpha: float = 0.05,
    ) -> List[CohortDecision]:
        """Estimate PPV lower bounds for each (cohort, tau) pair.

        Args:
            y_cal: Binary labels for calibration set, shape (n_cal,)
            predictions_cal: Binary predictions for calibration set, shape (n_cal,)
            cohort_ids_cal: Cohort identifiers for calibration set, shape (n_cal,)
            weights: Importance weights from estimate_weights(), shape (n_cal,)
            tau_grid: List of PPV thresholds to test (e.g., [0.5, 0.7, 0.9])
            alpha: Significance level (e.g., 0.05 for 95% confidence)

        Returns:
            decisions: List of CohortDecision objects, one per (cohort, tau) pair.
                Length = len(unique(cohort_ids_cal)) * len(tau_grid).

        Notes:
            - For each cohort, filter to predicted positives: predictions_cal == 1
            - Compute weighted PPV estimate: sum(w * y) / sum(w)
            - Compute confidence bound (method-specific: EB, Hoeffding, CLT, etc.)
            - Return CERTIFY if lower_bound >= tau, else ABSTAIN
            - Return NO-GUARANTEE if diagnostics fail (e.g., ESS too low, k-hat too high)
        """
        pass

    @abstractmethod
    def get_metadata(self) -> MethodMetadata:
        """Return metadata describing this method.

        Returns:
            metadata: MethodMetadata object with name, version, paper URL, etc.
        """
        pass

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return method-specific diagnostic information.

        Diagnostics help users understand when/why a method fails. Examples:
        - PSIS k-hat (Pareto tail diagnostic)
        - ESS (effective sample size)
        - Clip mass (fraction of weights clipped)
        - Convergence status (for optimization-based methods)
        - Kernel bandwidth (for kernel methods)

        Returns:
            diagnostics: Dictionary with method-specific metrics.
        """
        return {}

    def validate_inputs(
        self,
        y_cal: np.ndarray,
        predictions_cal: np.ndarray,
        cohort_ids_cal: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        """Validate inputs to estimate_bounds().

        Raises:
            ValueError: If inputs are invalid (wrong shape, nan values, etc.)
        """
        n = len(y_cal)

        if len(predictions_cal) != n:
            raise ValueError(f"predictions_cal length {len(predictions_cal)} != y_cal length {n}")
        if len(cohort_ids_cal) != n:
            raise ValueError(f"cohort_ids_cal length {len(cohort_ids_cal)} != y_cal length {n}")
        if len(weights) != n:
            raise ValueError(f"weights length {len(weights)} != y_cal length {n}")

        if not np.all(np.isin(y_cal, [0, 1])):
            raise ValueError("y_cal must be binary (0 or 1)")
        if not np.all(np.isin(predictions_cal, [0, 1])):
            raise ValueError("predictions_cal must be binary (0 or 1)")

        if not np.all(weights > 0):
            raise ValueError("All weights must be positive")
        if not np.all(np.isfinite(weights)):
            raise ValueError("All weights must be finite (no inf/nan)")

        if np.any(np.isnan(y_cal)):
            raise ValueError("y_cal contains NaN values")
        if np.any(np.isnan(predictions_cal)):
            raise ValueError("predictions_cal contains NaN values")
