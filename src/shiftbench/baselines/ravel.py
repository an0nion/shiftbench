"""RAVEL: Receipt-Anchored Verifiable Evaluation Ledger.

RAVEL is a shift-aware evaluation method that combines:
1. Cross-fitted density ratio estimation (discriminative classifier)
2. Self-normalized importance sampling (SNIS)
3. Stability gating (PSIS k-hat, ESS, clip mass diagnostics)
4. Empirical-Bernstein confidence bounds
5. Holm step-down for FWER control

Key differentiator: Abstains (NO-GUARANTEE) when importance weights are unstable.

References:
    - Original implementation: https://github.com/anthropics/ravel
    - Mathematical corrections documented in MATH_JUSTIFICATIONS.md
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from shiftbench.baselines.base import (
    BaselineMethod,
    CohortDecision,
    MethodMetadata,
)

# Note: These imports assume RAVEL source is available
# In production, we'd copy the relevant modules or install as dependency
try:
    from ravel.weights.pipeline import run_weights_pipeline
    from ravel.certification.certify import certify_cohorts_at_tau
    from ravel.bounds.holm import holm_reject
    RAVEL_AVAILABLE = True
except ImportError:
    RAVEL_AVAILABLE = False


class RAVELBaseline(BaselineMethod):
    """RAVEL baseline implementation."""

    def __init__(
        self,
        n_folds: int = 5,
        random_state: int = 42,
        logit_temp: float = 1.75,
        psis_k_cap: float = 0.70,
        ess_min_frac: float = 0.30,
        clip_mass_cap: float = 0.10,
        **kwargs,
    ):
        """Initialize RAVEL with hyperparameters.

        Args:
            n_folds: Number of cross-validation folds for density ratio estimation
            random_state: Random seed for reproducibility
            logit_temp: Temperature scaling for classifier logits (higher = smoother)
            psis_k_cap: Maximum allowed PSIS k-hat (diagnostic for heavy tails)
            ess_min_frac: Minimum effective sample size as fraction of nominal n
            clip_mass_cap: Maximum allowed clipped weight mass
            **kwargs: Additional hyperparameters passed to run_weights_pipeline
        """
        if not RAVEL_AVAILABLE:
            raise ImportError(
                "RAVEL source not found. Install with: pip install -e /path/to/ravel"
            )

        super().__init__(
            n_folds=n_folds,
            random_state=random_state,
            logit_temp=logit_temp,
            psis_k_cap=psis_k_cap,
            ess_min_frac=ess_min_frac,
            clip_mass_cap=clip_mass_cap,
            **kwargs,
        )
        self._pipeline_result = None  # Store for diagnostics

    def estimate_weights(
        self,
        X_cal: np.ndarray,
        X_target: np.ndarray,
        domain_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Estimate importance weights using RAVEL's pipeline.

        Args:
            X_cal: Calibration features (n_cal, n_features)
            X_target: Target features (n_target, n_features)
            domain_labels: Optional binary labels (0=cal, 1=target)

        Returns:
            weights: Importance weights for calibration samples (n_cal,)

        Raises:
            RuntimeError: If stability gates fail (NO-GUARANTEE)
        """
        # Construct domain labels if not provided
        if domain_labels is None:
            domain_labels = np.concatenate([
                np.zeros(len(X_cal), dtype=int),
                np.ones(len(X_target), dtype=int)
            ])

        # Concatenate features
        X = np.vstack([X_cal, X_target])

        # Run RAVEL weights pipeline
        result = run_weights_pipeline(
            X=X,
            domain_labels=domain_labels,
            n_folds=self.hyperparameters["n_folds"],
            random_state=self.hyperparameters["random_state"],
            logit_temp=self.hyperparameters["logit_temp"],
            psis_k_cap=self.hyperparameters["psis_k_cap"],
            ess_min_frac=self.hyperparameters["ess_min_frac"],
            clip_mass_cap=self.hyperparameters["clip_mass_cap"],
        )

        self._pipeline_result = result  # Store for diagnostics

        # Check gate status
        if result.state == "NO-GUARANTEE":
            raise RuntimeError(
                f"RAVEL stability gates failed. "
                f"c_final={result.c_final}, "
                f"Try increasing ess_min_frac or clip_mass_cap."
            )

        return result.weights

    def estimate_bounds(
        self,
        y_cal: np.ndarray,
        predictions_cal: np.ndarray,
        cohort_ids_cal: np.ndarray,
        weights: np.ndarray,
        tau_grid: List[float],
        alpha: float = 0.05,
    ) -> List[CohortDecision]:
        """Estimate PPV lower bounds using Empirical-Bernstein + Holm.

        Args:
            y_cal: Binary labels (n_cal,)
            predictions_cal: Binary predictions (n_cal,)
            cohort_ids_cal: Cohort IDs (n_cal,)
            weights: Importance weights (n_cal,)
            tau_grid: PPV thresholds to test
            alpha: Significance level (FWER control)

        Returns:
            decisions: List of CohortDecision objects
        """
        self.validate_inputs(y_cal, predictions_cal, cohort_ids_cal, weights)

        all_decisions = []

        # Process each tau level
        for tau in tau_grid:
            decisions_at_tau = certify_cohorts_at_tau(
                labels=y_cal,
                weights=weights,
                cohort_ids=cohort_ids_cal,
                predictions=predictions_cal,
                tau=tau,
                alpha=alpha,
                eligible_cohorts=None,  # All cohorts eligible if gates passed
                min_eff_positives=5.0,
            )

            # Convert RAVEL's CohortDecision to ShiftBench's CohortDecision
            # (They have the same fields, so we can just copy)
            for d in decisions_at_tau:
                sb_decision = CohortDecision(
                    cohort_id=d.cohort_id,
                    tau=d.tau,
                    decision=d.decision,
                    mu_hat=d.mu_hat,
                    var_hat=d.var_hat,
                    n_eff=d.n_eff,
                    lower_bound=d.lower_bound,
                    p_value=d.p_value,
                    diagnostics=self.get_diagnostics() if hasattr(d, "reason") else None,
                )
                all_decisions.append(sb_decision)

        return all_decisions

    def get_metadata(self) -> MethodMetadata:
        """Return RAVEL metadata."""
        return MethodMetadata(
            name="ravel",
            version="1.0.0",
            description=(
                "Receipt-Anchored Verifiable Evaluation Ledger. "
                "Importance-weighted PPV bounds with stability gating."
            ),
            paper_title="RAVEL: Certify-or-Abstain Evaluation Under Covariate Shift",
            paper_url=None,  # Not yet published
            code_url="https://github.com/anthropics/ravel",
            hyperparameters=self.hyperparameters,
            supports_abstention=True,
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return RAVEL-specific diagnostics."""
        if self._pipeline_result is None:
            return {}

        result = self._pipeline_result
        gate_final = result.gate_path[-1] if result.gate_path else None

        diagnostics = {
            "state": result.state,
            "c_final": result.c_final,
            "smoothing_alpha": result.smoothing_alpha,
            "n_cal": result.n_cal,
            "n_target": result.n_target,
        }

        if gate_final:
            diagnostics.update({
                "psis_k_hat": gate_final.psis_k,
                "ess_fraction": gate_final.ess_frac,
                "clip_mass": gate_final.clip_mass,
                "passed": gate_final.passed,
            })

        return diagnostics


# Factory function for easy instantiation
def create_ravel_baseline(**kwargs) -> RAVELBaseline:
    """Create a RAVEL baseline with default or custom hyperparameters.

    Examples:
        >>> # Default RAVEL
        >>> ravel = create_ravel_baseline()

        >>> # More permissive gates (fewer abstentions)
        >>> ravel_relaxed = create_ravel_baseline(
        ...     psis_k_cap=0.90,
        ...     ess_min_frac=0.20,
        ... )

        >>> # More strict gates (more abstentions, tighter bounds)
        >>> ravel_strict = create_ravel_baseline(
        ...     psis_k_cap=0.50,
        ...     ess_min_frac=0.40,
        ... )
    """
    return RAVELBaseline(**kwargs)
