"""CV+ (Cross-Validation Plus) for shift-aware PPV certification.

Cross-validation conformal prediction (Barber et al. 2021) extends split conformal
by using K-fold splitting so that ALL calibration data is used for both weight
estimation and calibration.

Each calibration sample gets a density-ratio weight estimated by LEAVING OUT its
own fold -- avoiding optimistic bias from using the same data for both fitting
and evaluation.

Key improvement over split conformal:
- Split conformal uses ~50% of calibration data for calibration
- CV+ uses ~100% (each sample calibrated using other folds' weight estimates)
- Gives tighter bounds on the same total calibration set

Density ratio is estimated via logistic regression (cal vs target classification),
which naturally generalises to held-out fold samples unlike kernel methods.

References:
    Barber et al. 2021. "Predictive Inference with the Jackknife+"
    https://arxiv.org/abs/1905.02928

    Tibshirani et al. 2019. "Conformal Prediction Under Covariate Shift"
    https://arxiv.org/abs/1904.06019
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from shiftbench.baselines.base import (
    BaselineMethod,
    CohortDecision,
    MethodMetadata,
)


class CVPlusBaseline(BaselineMethod):
    """CV+ for shift-aware PPV certification.

    For each calibration sample i (in fold k):
      1. Estimate density ratio weights using ALL other K-1 folds as cal reference
      2. Apply those weights to sample i
    This gives every sample a weight estimated without itself -- the CV+ principle.

    Then runs standard EB bounds with these cross-validated weights.
    """

    def __init__(
        self,
        n_folds: int = 5,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
        **kwargs,
    ):
        """Initialize CV+ with hyperparameters.

        Args:
            n_folds: Number of cross-validation folds (K). More folds = more stable
                but slower. Default 5.
            C: Logistic regression regularisation (inverse, larger = less regularised).
            max_iter: Max LR iterations per fold.
            random_state: Random seed for fold splitting and LR.
        """
        super().__init__(
            n_folds=n_folds,
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs,
        )
        self._X_target: Optional[np.ndarray] = None
        self._cv_weights: Optional[np.ndarray] = None

    def estimate_weights(
        self,
        X_cal: np.ndarray,
        X_target: np.ndarray,
        domain_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Estimate cross-validated density ratio weights.

        For each fold k, fits logistic regression on non-k calibration samples vs
        target samples, then predicts weights for fold k samples.

        Args:
            X_cal: Calibration features (n_cal, n_features)
            X_target: Target features (n_target, n_features)
            domain_labels: Not used (CV+ uses logistic regression internally)

        Returns:
            weights: Cross-validated importance weights (n_cal,)
        """
        n_cal = len(X_cal)
        n_target = len(X_target)
        self._X_target = X_target

        # Scale factor to account for class imbalance in binary classifier
        # w(x) = P(target|x) / P(cal|x) * (n_cal / n_target)
        scale = n_cal / max(n_target, 1)

        cv_weights = np.ones(n_cal)

        n_folds = self.hyperparameters["n_folds"]
        random_state = self.hyperparameters["random_state"]
        C = self.hyperparameters["C"]
        max_iter = self.hyperparameters["max_iter"]

        # Ensure we don't request more folds than samples
        n_folds = min(n_folds, n_cal)

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        for train_idx, val_idx in kf.split(X_cal):
            X_train = X_cal[train_idx]  # Non-k cal samples
            X_val = X_cal[val_idx]      # Fold k cal samples

            # Build binary classification problem: train_cal=0, target=1
            X_fit = np.vstack([X_train, X_target])
            y_fit = np.concatenate([
                np.zeros(len(X_train), dtype=int),
                np.ones(len(X_target), dtype=int),
            ])

            try:
                lr = LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    random_state=random_state,
                    solver="lbfgs"
                )
                lr.fit(X_fit, y_fit)

                # P(target|x) for val fold samples
                p_target = lr.predict_proba(X_val)[:, 1]
                p_cal = 1.0 - p_target

                # Density ratio: p_target / p_cal, with class imbalance scaling
                raw_w = (p_target / np.maximum(p_cal, 1e-8)) * scale

            except Exception:
                # Fallback: uniform weights if LR fails (e.g. rank-deficient features)
                raw_w = np.ones(len(val_idx))

            # Self-normalise within fold
            fold_mean = raw_w.mean()
            if fold_mean > 0:
                raw_w = raw_w / fold_mean
            cv_weights[val_idx] = raw_w

        # Global renormalise for numerical stability
        global_mean = cv_weights.mean()
        if global_mean > 0:
            cv_weights = cv_weights / global_mean

        self._cv_weights = cv_weights
        return cv_weights

    def estimate_bounds(
        self,
        y_cal: np.ndarray,
        predictions_cal: np.ndarray,
        cohort_ids_cal: np.ndarray,
        weights: np.ndarray,
        tau_grid: List[float],
        alpha: float = 0.05,
    ) -> List[CohortDecision]:
        """Estimate PPV lower bounds using CV+ weights with Empirical-Bernstein.

        Args:
            y_cal: Binary labels (n_cal,)
            predictions_cal: Binary predictions (n_cal,)
            cohort_ids_cal: Cohort IDs (n_cal,)
            weights: CV+ importance weights from estimate_weights()
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

        for tau in tau_grid:
            for cohort_id in unique_cohorts:
                cohort_mask = (cohort_ids_cal == cohort_id)
                pos_mask = cohort_mask & (predictions_cal == 1)

                y_cohort = y_cal[pos_mask]
                w_cohort = weights[pos_mask]

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
                            "method": "cvplus",
                            "n_folds": self.hyperparameters["n_folds"],
                            "n_positives": len(y_cohort),
                        },
                    )

                all_decisions.append(decision)

        return all_decisions

    def get_metadata(self) -> MethodMetadata:
        return MethodMetadata(
            name="cvplus",
            version="1.0.0",
            description=(
                "CV+ (Cross-Validation Plus). K-fold density ratio estimation "
                "via logistic regression; each sample weighted using leave-fold-out "
                "estimator. Tighter than split conformal on same calibration set."
            ),
            paper_title="Predictive Inference with the Jackknife+",
            paper_url="https://arxiv.org/abs/1905.02928",
            hyperparameters=self.hyperparameters,
            supports_abstention=False,
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        diag: Dict[str, Any] = {"method": "cvplus"}
        if self._cv_weights is not None:
            diag["weight_mean"] = float(self._cv_weights.mean())
            diag["weight_std"] = float(self._cv_weights.std())
            diag["weight_max"] = float(self._cv_weights.max())
        return diag


def create_cvplus_baseline(**kwargs) -> CVPlusBaseline:
    """Create a CV+ baseline with default or custom hyperparameters.

    Examples:
        >>> cvplus = create_cvplus_baseline()
        >>> cvplus_10fold = create_cvplus_baseline(n_folds=10)
    """
    return CVPlusBaseline(**kwargs)
