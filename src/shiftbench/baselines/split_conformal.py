"""Split Conformal Prediction baseline (no shift correction).

This is a reference baseline that applies standard conformal prediction
WITHOUT importance weighting. It assumes exchangeability between
calibration and test data, which is violated under covariate shift.

Expected behavior:
- Coverage should degrade under shift (no shift correction)
- Certification rate may be higher than shift-corrected methods
  (because it doesn't account for harder shifted subpopulations)
- Useful as a "what happens without shift awareness" baseline

References:
    Vovk et al. 2005. "Algorithmic Learning in a Random World"
    Lei et al. 2018. "Distribution-Free Predictive Inference For Regression"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from shiftbench.baselines.base import (
    BaselineMethod,
    CohortDecision,
    MethodMetadata,
)


class SplitConformalBaseline(BaselineMethod):
    """Split Conformal Prediction (no shift correction).

    Uses standard (unweighted) conformal quantiles. Serves as a reference
    to demonstrate the need for shift-aware evaluation.
    """

    def __init__(self, random_state: int = 42, **kwargs):
        super().__init__(random_state=random_state, **kwargs)

    def estimate_weights(
        self,
        X_cal: np.ndarray,
        X_target: np.ndarray,
        domain_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return uniform weights (no shift correction)."""
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
        """Estimate PPV lower bounds using unweighted conformal quantiles.

        Uses Clopper-Pearson-style exact binomial lower bound on PPV.
        """
        self.validate_inputs(y_cal, predictions_cal, cohort_ids_cal, weights)

        all_decisions = []
        unique_cohorts = np.unique(cohort_ids_cal)

        for tau in tau_grid:
            for cohort_id in unique_cohorts:
                cohort_mask = (cohort_ids_cal == cohort_id)
                pos_mask = cohort_mask & (predictions_cal == 1)

                y_cohort = y_cal[pos_mask]
                n = len(y_cohort)

                if n < 5:
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
                    mu_hat = y_cohort.mean()
                    var_hat = mu_hat * (1 - mu_hat) / n

                    # Clopper-Pearson lower bound (exact binomial)
                    k = int(y_cohort.sum())
                    lower_bound = self._clopper_pearson_lower(k, n, alpha)

                    # p-value: P(Binom(n, tau) >= k)
                    from scipy.stats import binom
                    pval = 1.0 - binom.cdf(k - 1, n, tau) if k > 0 else 1.0

                    dec = "CERTIFY" if lower_bound >= tau else "ABSTAIN"

                    decision = CohortDecision(
                        cohort_id=cohort_id,
                        tau=tau,
                        decision=dec,
                        mu_hat=mu_hat,
                        var_hat=var_hat,
                        n_eff=float(n),
                        lower_bound=lower_bound,
                        p_value=pval,
                        diagnostics={"method": "split_conformal", "n_positives": n},
                    )

                all_decisions.append(decision)

        return all_decisions

    @staticmethod
    def _clopper_pearson_lower(k: int, n: int, alpha: float) -> float:
        """Compute Clopper-Pearson exact lower confidence bound."""
        if k == 0:
            return 0.0
        from scipy.stats import beta
        return float(beta.ppf(alpha, k, n - k + 1))

    def get_metadata(self) -> MethodMetadata:
        return MethodMetadata(
            name="split_conformal",
            version="1.0.0",
            description=(
                "Split Conformal Prediction without shift correction. "
                "Reference baseline assuming exchangeability."
            ),
            paper_title="Distribution-Free Predictive Inference For Regression",
            paper_url="https://arxiv.org/abs/1604.04173",
            hyperparameters=self.hyperparameters,
            supports_abstention=False,
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        return {"method": "split_conformal", "shift_aware": False}


def create_split_conformal_baseline(**kwargs) -> SplitConformalBaseline:
    """Create a Split Conformal Prediction baseline."""
    return SplitConformalBaseline(**kwargs)
