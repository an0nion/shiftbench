"""BBSE: Black-Box Shift Estimation for label shift correction.

BBSE (Lipton et al. 2018) estimates importance weights under LABEL SHIFT:
  p_target(x) = sum_y p_cal(x | y) * p_target(y)
  w(x, y) = p_target(y) / p_cal(y)

The key insight: under label shift, only the marginal label distribution
changes, not the class-conditional feature distribution.  BBSE estimates
target label proportions using soft classifier predictions on the target set,
then reweights by the label proportion ratio.

Algorithm:
  1. Fit classifier on calibration set (X_cal, y_cal) -> p(y|x)
  2. Predict on target set X_target -> get p_target(y=1) = mean(predict_proba)
  3. Compute p_cal(y=1) = mean(y_cal)
  4. Label shift weight: w_i = p_target(y_i) / p_cal(y_i)

Comparison to covariate shift methods (uLSIF, KLIEP):
  - uLSIF/KLIEP: w(x) = p_target(x) / p_cal(x)  [covariate shift]
  - BBSE:        w(y) = p_target(y) / p_cal(y)   [label shift]
  - If BOTH shifts occur, neither alone is sufficient
  - ShiftBench datasets primarily have covariate shift (scaffold, demographic)
    so BBSE may underperform uLSIF -- but provides a useful contrast

References:
    Lipton et al. 2018. "Detecting and Correcting for Label Shift with Black Box Predictors"
    https://arxiv.org/abs/1802.03916
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from shiftbench.baselines.base import (
    BaselineMethod,
    CohortDecision,
    MethodMetadata,
)


class BBSEBaseline(BaselineMethod):
    """BBSE for label-shift-aware PPV certification.

    Estimates label proportions in the target distribution by training a
    classifier on calibration data and applying it to target features.
    Computes label-shift weights w(y) = p_target(y) / p_cal(y) for EB bounds.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
        **kwargs,
    ):
        """Initialise BBSE.

        Args:
            C: Logistic regression regularisation (larger = less regularised).
            max_iter: Max LR iterations.
            random_state: Random seed.
        """
        super().__init__(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs,
        )
        self._X_cal: Optional[np.ndarray] = None
        self._X_target: Optional[np.ndarray] = None
        self._p_target_y1: Optional[float] = None
        self._p_cal_y1: Optional[float] = None

    def estimate_weights(
        self,
        X_cal: np.ndarray,
        X_target: np.ndarray,
        domain_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Store features for BBSE; return uniform weights.

        Label shift weights require y_cal which is only available in
        estimate_bounds().  This step stores the features for use there.

        Args:
            X_cal: Calibration features (n_cal, n_features)
            X_target: Target features (n_target, n_features)
            domain_labels: Not used

        Returns:
            weights: Uniform weights (n_cal,) -- BBSE reweights in estimate_bounds
        """
        self._X_cal = X_cal
        self._X_target = X_target
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
        """Estimate PPV lower bounds using BBSE label-shift weights.

        Steps:
          1. Fit LR on (X_cal, y_cal)
          2. Predict label proportions on X_target -> p_target(y=1)
          3. Compute p_cal(y=1) = mean(y_cal)
          4. Weight sample i: w_i = p_target(y_i) / p_cal(y_i)
          5. Apply EB bounds with label-shift weights

        Args:
            y_cal: Binary labels (n_cal,)
            predictions_cal: Binary predictions (n_cal,)
            cohort_ids_cal: Cohort IDs (n_cal,)
            weights: Uniform weights from estimate_weights() (ignored internally)
            tau_grid: PPV thresholds to test
            alpha: Significance level

        Returns:
            decisions: List of CohortDecision per (cohort, tau) pair
        """
        self.validate_inputs(y_cal, predictions_cal, cohort_ids_cal, weights)

        from ravel.bounds.empirical_bernstein import eb_lower_bound
        from ravel.bounds.p_value import eb_p_value
        from ravel.bounds.weighted_stats import weighted_stats_01

        # --- BBSE label proportion estimation ---
        C = self.hyperparameters["C"]
        max_iter = self.hyperparameters["max_iter"]
        random_state = self.hyperparameters["random_state"]

        p_cal_y1 = float(y_cal.mean())
        p_cal_y0 = 1.0 - p_cal_y1

        # Fallback: if labels are all one class, BBSE degenerates to uniform
        if p_cal_y1 <= 1e-6 or p_cal_y0 <= 1e-6:
            bbse_weights = np.ones(len(y_cal))
            p_target_y1 = p_cal_y1
        elif self._X_cal is None or self._X_target is None:
            bbse_weights = np.ones(len(y_cal))
            p_target_y1 = p_cal_y1
        else:
            try:
                # Step 1: Fit classifier on calibration data
                lr = LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    random_state=random_state,
                    solver="lbfgs"
                )
                lr.fit(self._X_cal, y_cal)

                # Step 2: Estimate label proportions in target
                # p_target(y=1) = mean of predicted positive probability on target
                p_target_y1 = float(
                    lr.predict_proba(self._X_target)[:, 1].mean()
                )
                p_target_y0 = 1.0 - p_target_y1

                # Clip to valid range
                p_target_y1 = float(np.clip(p_target_y1, 1e-6, 1.0 - 1e-6))
                p_target_y0 = 1.0 - p_target_y1

                # Step 3: Label shift weights w_i = p_target(y_i) / p_cal(y_i)
                w_pos = p_target_y1 / p_cal_y1
                w_neg = p_target_y0 / p_cal_y0
                bbse_weights = np.where(y_cal == 1, w_pos, w_neg).astype(float)

                # Normalise to mean 1
                bbse_weights = bbse_weights / bbse_weights.mean()

            except Exception:
                # Fallback: uniform weights
                bbse_weights = np.ones(len(y_cal))
                p_target_y1 = p_cal_y1

        self._p_target_y1 = p_target_y1
        self._p_cal_y1 = p_cal_y1

        # --- Compute EB bounds with BBSE weights ---
        all_decisions = []
        unique_cohorts = np.unique(cohort_ids_cal)

        for tau in tau_grid:
            for cohort_id in unique_cohorts:
                cohort_mask = (cohort_ids_cal == cohort_id)
                pos_mask = cohort_mask & (predictions_cal == 1)

                y_cohort = y_cal[pos_mask]
                w_cohort = bbse_weights[pos_mask]

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
                            "method": "bbse",
                            "p_target_y1": self._p_target_y1,
                            "p_cal_y1": self._p_cal_y1,
                            "label_shift_ratio": (
                                self._p_target_y1 / self._p_cal_y1
                                if self._p_cal_y1 and self._p_cal_y1 > 0
                                else 1.0
                            ),
                        },
                    )

                all_decisions.append(decision)

        return all_decisions

    def get_metadata(self) -> MethodMetadata:
        return MethodMetadata(
            name="bbse",
            version="1.0.0",
            description=(
                "BBSE (Black-Box Shift Estimation). Corrects for label shift "
                "by estimating target label proportions via classifier predictions. "
                "Note: designed for label shift; may underperform on covariate-shift datasets."
            ),
            paper_title="Detecting and Correcting for Label Shift with Black Box Predictors",
            paper_url="https://arxiv.org/abs/1802.03916",
            hyperparameters=self.hyperparameters,
            supports_abstention=False,
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        diag: Dict[str, Any] = {"method": "bbse"}
        if self._p_target_y1 is not None:
            diag["p_target_y1"] = self._p_target_y1
            diag["p_cal_y1"] = self._p_cal_y1
            if self._p_cal_y1 and self._p_cal_y1 > 0:
                diag["label_shift_ratio"] = self._p_target_y1 / self._p_cal_y1
        return diag


def create_bbse_baseline(**kwargs) -> BBSEBaseline:
    """Create a BBSE baseline.

    Examples:
        >>> bbse = create_bbse_baseline()
        >>> bbse_strong = create_bbse_baseline(C=0.1)  # more regularised classifier
    """
    return BBSEBaseline(**kwargs)
