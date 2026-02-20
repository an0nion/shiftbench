"""Group DRO (Distributionally Robust Optimization) for PPV certification.

Group DRO (Sagawa et al. 2020) finds importance weights that minimise the
WORST-CASE loss over all groups (cohorts).  In our PPV certification context:

  - Group = cohort (scaffold, demographic group, etc.)
  - Loss = PPV deficit: max(0, tau - PPV_c) for cohort c
  - DRO goal: upweight hard cohorts so bounds are robust to worst-case shift

Algorithm (exponential weight update, Exponentiated Gradient):
  1. Initialise: equal weight per cohort, w_c = 1/C
  2. For each iteration:
     a. Compute per-cohort PPV with current sample weights
     b. Compute cohort loss: L_c = max(0, tau - PPV_c)
     c. Update: log_q_c += step_size * L_c
     d. Normalise: q_c = softmax(log_q_c)
  3. Final sample weight: w_i = q_{cohort(i)} / (n_cohort / n_total)

This is equivalent to finding the minimax weight distribution -- the weights
that certify PPV at tau even under the worst-case reweighting of cohorts.

Note: Group DRO upweights entire COHORTS (group-level shift), whereas uLSIF
upweights individual SAMPLES (sample-level shift). They address different
types of distribution shift.

References:
    Sagawa et al. 2020. "Distributionally Robust Neural Networks"
    https://arxiv.org/abs/1911.08731

    Hu et al. 2018. "Does Distributionally Robust Supervised Learning Give
    Robust Classifiers?" https://arxiv.org/abs/1611.02041
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from shiftbench.baselines.base import (
    BaselineMethod,
    CohortDecision,
    MethodMetadata,
)


class GroupDROBaseline(BaselineMethod):
    """Group DRO for shift-aware PPV certification.

    Finds cohort-level importance weights via exponential weight update such
    that the hardest cohorts (lowest PPV) receive the most weight. Provides
    robustness to worst-case cohort-level distribution shifts.
    """

    def __init__(
        self,
        step_size: float = 0.1,
        n_iter: int = 100,
        tau_ref: float = 0.7,
        random_state: int = 42,
        **kwargs,
    ):
        """Initialise Group DRO.

        Args:
            step_size: Learning rate for exponential weight update. Larger values
                converge faster but may overshoot. Default 0.1.
            n_iter: Number of DRO update iterations. Default 100.
            tau_ref: Reference PPV threshold used during weight optimisation.
                Weights will be optimised to certify this threshold. Default 0.7.
                NOTE: bounds are still computed at all tau_grid values.
            random_state: Random seed (currently unused, reserved for future use).
        """
        super().__init__(
            step_size=step_size,
            n_iter=n_iter,
            tau_ref=tau_ref,
            random_state=random_state,
            **kwargs,
        )
        self._cohort_weights: Optional[np.ndarray] = None
        self._cohort_ids: Optional[np.ndarray] = None

    def estimate_weights(
        self,
        X_cal: np.ndarray,
        X_target: np.ndarray,
        domain_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return uniform weights; Group DRO operates on cohort structure.

        Group DRO computes per-cohort weights in estimate_bounds() when the
        cohort assignments are known.  This step returns uniform weights as a
        placeholder.

        Args:
            X_cal: Calibration features (not used -- DRO is cohort-based)
            X_target: Target features (not used)
            domain_labels: Not used

        Returns:
            weights: Uniform weights, shape (n_cal,)
        """
        return np.ones(len(X_cal))

    def estimate_bounds(
        self,
        y_cal: np.ndarray,
        predictions_cal: np.ndarray,
        cohort_ids_cal: np.ndarray,
        weights: np.ndarray,
        tau_grid: List[float],
        alpha: float = 0.05,
    ) -> List[CohortDecision]:
        """Estimate PPV lower bounds using DRO-optimised cohort weights.

        For each tau in tau_grid:
          1. Run DRO to find cohort weights q_c that upweight hard cohorts
          2. Convert to sample weights: w_i = q_{cohort(i)} / freq(cohort)
          3. Compute EB bounds with those weights

        Args:
            y_cal: Binary labels (n_cal,)
            predictions_cal: Binary predictions (n_cal,)
            cohort_ids_cal: Cohort IDs (n_cal,)
            weights: Ignored (DRO computes its own weights internally)
            tau_grid: PPV thresholds to test
            alpha: Significance level

        Returns:
            decisions: List of CohortDecision per (cohort, tau) pair
        """
        self.validate_inputs(y_cal, predictions_cal, cohort_ids_cal, weights)

        from ravel.bounds.empirical_bernstein import eb_lower_bound
        from ravel.bounds.p_value import eb_p_value
        from ravel.bounds.weighted_stats import weighted_stats_01

        all_decisions = []
        unique_cohorts = np.unique(cohort_ids_cal)
        n_cohorts = len(unique_cohorts)
        n_cal = len(y_cal)

        # Cohort index mapping
        cohort_to_idx = {c: i for i, c in enumerate(unique_cohorts)}
        sample_cohort_idx = np.array([cohort_to_idx[c] for c in cohort_ids_cal])

        # Pre-compute cohort frequencies (for per-sample normalisation)
        cohort_counts = np.bincount(sample_cohort_idx, minlength=n_cohorts)
        cohort_freq = cohort_counts / max(n_cal, 1)

        # Pre-compute per-cohort PPV using positive predictions only
        # pos_mask: (n_cal,) bool -- samples that are predicted positive
        pos_mask = (predictions_cal == 1)

        step_size = self.hyperparameters["step_size"]
        n_iter = self.hyperparameters["n_iter"]
        tau_ref = self.hyperparameters["tau_ref"]

        for tau in tau_grid:
            # --- DRO weight optimisation (tau_ref used for weight finding) ---
            effective_tau = tau_ref if tau_ref is not None else tau

            log_q = np.zeros(n_cohorts)  # log cohort weights (unnormalised)

            for _ in range(n_iter):
                # Normalised cohort weights
                q = _softmax(log_q)

                # Compute per-cohort weighted PPV
                # w_i = q[cohort(i)] / (freq[cohort(i)] * n_cal)  -- uniform within cohort
                sample_w = q[sample_cohort_idx] / np.maximum(
                    cohort_freq[sample_cohort_idx] * n_cal, 1e-8
                )
                # Renormalise to mean 1
                sample_w = sample_w / sample_w.mean()

                # Per-cohort PPV under current weights
                cohort_ppv = np.zeros(n_cohorts)
                for ci, cohort_id in enumerate(unique_cohorts):
                    c_pos_mask = pos_mask & (cohort_ids_cal == cohort_id)
                    y_c = y_cal[c_pos_mask]
                    w_c = sample_w[c_pos_mask]
                    if len(y_c) >= 1 and w_c.sum() > 0:
                        cohort_ppv[ci] = float((w_c * y_c).sum() / w_c.sum())
                    else:
                        cohort_ppv[ci] = 0.5  # Neutral for empty cohorts

                # DRO update: upweight cohorts with PPV below threshold
                loss = np.maximum(0.0, effective_tau - cohort_ppv)
                log_q = log_q + step_size * loss

            # Final cohort weights
            q_final = _softmax(log_q)
            sample_w_final = q_final[sample_cohort_idx] / np.maximum(
                cohort_freq[sample_cohort_idx] * n_cal, 1e-8
            )
            sample_w_final = sample_w_final / sample_w_final.mean()

            # --- Compute EB bounds with DRO weights ---
            for cohort_id in unique_cohorts:
                cohort_mask = (cohort_ids_cal == cohort_id)
                c_pos_mask = cohort_mask & pos_mask

                y_cohort = y_cal[c_pos_mask]
                w_cohort = sample_w_final[c_pos_mask]

                if len(y_cohort) < 5:
                    decision = CohortDecision(
                        cohort_id=cohort_id,
                        tau=tau,
                        decision="ABSTAIN",
                        mu_hat=np.nan,
                        var_hat=np.nan,
                        n_eff=0.0,
                        lower_bound=np.nan,
                        p_value=1.0,
                        diagnostics={"reason": "insufficient_positives",
                                     "n": len(y_cohort)},
                    )
                else:
                    stats = weighted_stats_01(y_cohort, w_cohort)
                    lb = eb_lower_bound(stats.mu, stats.var, stats.n_eff, alpha)
                    pval = eb_p_value(stats.mu, stats.var, stats.n_eff, tau)
                    dec = "CERTIFY" if lb >= tau else "ABSTAIN"

                    ci = cohort_to_idx[cohort_id]
                    decision = CohortDecision(
                        cohort_id=cohort_id,
                        tau=tau,
                        decision=dec,
                        mu_hat=stats.mu,
                        var_hat=stats.var,
                        n_eff=stats.n_eff,
                        lower_bound=lb,
                        p_value=pval,
                        diagnostics={
                            "method": "group_dro",
                            "cohort_weight": float(q_final[ci]),
                            "n_iter": n_iter,
                            "step_size": step_size,
                        },
                    )

                all_decisions.append(decision)

            # Store final cohort weights for diagnostics
            self._cohort_weights = q_final
            self._cohort_ids = unique_cohorts

        return all_decisions

    def get_metadata(self) -> MethodMetadata:
        return MethodMetadata(
            name="group_dro",
            version="1.0.0",
            description=(
                "Group DRO: exponential weight update over cohorts to upweight "
                "hardest groups. Finds minimax PPV-certifying weights. "
                "Addresses cohort-level (group) distribution shift."
            ),
            paper_title="Distributionally Robust Neural Networks for Group Shifts",
            paper_url="https://arxiv.org/abs/1911.08731",
            hyperparameters=self.hyperparameters,
            supports_abstention=False,
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        diag: Dict[str, Any] = {"method": "group_dro"}
        if self._cohort_weights is not None:
            diag["cohort_weight_min"] = float(self._cohort_weights.min())
            diag["cohort_weight_max"] = float(self._cohort_weights.max())
            diag["cohort_weight_entropy"] = float(
                -np.sum(self._cohort_weights * np.log(self._cohort_weights + 1e-8))
            )
        return diag


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - x.max())
    return e / e.sum()


def create_group_dro_baseline(**kwargs) -> GroupDROBaseline:
    """Create a Group DRO baseline.

    Examples:
        >>> dro = create_group_dro_baseline()
        >>> dro_strict = create_group_dro_baseline(tau_ref=0.9, n_iter=200)
    """
    return GroupDROBaseline(**kwargs)
