"""ShiftBench: A Benchmark Suite for Shift-Aware ML Evaluation.

ShiftBench provides:
- 50+ datasets with documented covariate shifts
- 10+ baseline methods spanning density ratio estimation, conformal prediction, and DRO
- Reproducibility infrastructure with hash-chained receipts
- Standardized evaluation protocol with certify-or-abstain framework
"""

__version__ = "1.0.0"

from shiftbench.baselines.base import (
    BaselineMethod,
    CohortDecision,
    MethodMetadata,
)

__all__ = [
    "BaselineMethod",
    "CohortDecision",
    "MethodMetadata",
]
