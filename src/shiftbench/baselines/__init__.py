"""Baseline methods for ShiftBench.

All methods implement the BaselineMethod interface defined in base.py.
"""

from shiftbench.baselines.base import (
    BaselineMethod,
    CohortDecision,
    MethodMetadata,
)

from shiftbench.baselines.ulsif import uLSIFBaseline, create_ulsif_baseline
from shiftbench.baselines.kliep import KLIEPBaseline, create_kliep_baseline
from shiftbench.baselines.kmm import KMMBaseline, create_kmm_baseline
from shiftbench.baselines.rulsif import RULSIFBaseline, create_rulsif_baseline
from shiftbench.baselines.weighted_conformal import (
    WeightedConformalBaseline,
    create_weighted_conformal_baseline,
)
from shiftbench.baselines.split_conformal import (
    SplitConformalBaseline,
    create_split_conformal_baseline,
)

try:
    from shiftbench.baselines.ravel import RAVELBaseline, create_ravel_baseline
    RAVEL_AVAILABLE = True
except ImportError:
    RAVEL_AVAILABLE = False

__all__ = [
    "BaselineMethod",
    "CohortDecision",
    "MethodMetadata",
    "uLSIFBaseline",
    "create_ulsif_baseline",
    "KLIEPBaseline",
    "create_kliep_baseline",
    "KMMBaseline",
    "create_kmm_baseline",
    "RULSIFBaseline",
    "create_rulsif_baseline",
    "WeightedConformalBaseline",
    "create_weighted_conformal_baseline",
    "SplitConformalBaseline",
    "create_split_conformal_baseline",
]

if RAVEL_AVAILABLE:
    __all__.extend(["RAVELBaseline", "create_ravel_baseline"])
