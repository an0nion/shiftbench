# ShiftBench - Phase 0 Implementation Status

**Date**: 2025-02-16
**Phase**: 0 (Foundation)
**Goal**: Transform RAVEL into ShiftBench benchmark suite for NeurIPS D&B 2025

## What We've Built ✅

### 1. Core Infrastructure

**Dataset Registry** (`data/registry.json`):
- ✅ Comprehensive metadata for 11 molecular datasets
- ✅ Fields: name, domain, task_type, n_samples, shift_type, cohort_definition, tau_grid, license, citation
- ✅ Metadata includes expansion plan (30 molecular, 40 text, 30 tabular = 100 total)
- ⚠️ Licenses marked "Unknown - needs audit" (critical for Phase 1)

**Baseline Interface** (`src/shiftbench/baselines/base.py`):
- ✅ `BaselineMethod` abstract class defining standard API
- ✅ `CohortDecision` dataclass for results
- ✅ `MethodMetadata` dataclass for method descriptions
- ✅ Input validation utilities
- ✅ Comprehensive docstrings with usage examples

**Dataset Loader** (`src/shiftbench/data.py`):
- ✅ `DatasetRegistry` class for managing registry.json
- ✅ `load_dataset()` function for loading processed data
- ✅ `get_registry()` helper for easy access
- ✅ Support for filtering by domain
- ⚠️ Assumes preprocessed data exists (need preprocessing scripts)

### 2. Baseline Implementations

**RAVEL Wrapper** (`src/shiftbench/baselines/ravel.py`):
- ✅ Full implementation wrapping existing RAVEL pipeline
- ✅ Implements `estimate_weights()` with cross-fitted density ratio
- ✅ Implements `estimate_bounds()` with EB bounds + Holm
- ✅ Supports stability gating (can return NO-GUARANTEE)
- ✅ Rich diagnostics (PSIS k-hat, ESS, clip mass)
- ⚠️ Depends on RAVEL source being available

**uLSIF Baseline** (`src/shiftbench/baselines/ulsif.py`):
- ✅ Complete implementation of unconstrained Least-Squares Importance Fitting
- ✅ Gaussian kernel basis functions
- ✅ Closed-form solution via ridge regression
- ✅ Median heuristic for automatic bandwidth selection
- ✅ Self-normalized weights
- ✅ Does NOT support abstention (no stability gates)
- ✅ Full diagnostics (kernel params, alpha weights)

### 3. Package Structure

```
shift-bench/
├── data/
│   └── registry.json          ✅ 11 datasets catalogued
├── src/shiftbench/
│   ├── __init__.py            ✅ Package exports
│   ├── data.py                ✅ Registry and loader
│   └── baselines/
│       ├── __init__.py        ✅ Baseline exports
│       ├── base.py            ✅ Abstract interface
│       ├── ravel.py           ✅ RAVEL wrapper
│       └── ulsif.py           ✅ uLSIF implementation
├── scripts/                   🚧 Empty (needs evaluation harness)
├── docs/                      🚧 Empty (needs guides)
├── README.md                  ✅ Project overview
├── setup.py                   ✅ Installation config
└── STATUS.md                  ✅ This file
```

## Critical Gaps (Must Address in Phase 1)

### 1. Missing Preprocessing Infrastructure
**Problem**: Registry references processed data at `data/processed/<dataset>/`, but:
- No preprocessing scripts exist
- No actual processed data (features.npy, labels.npy, cohorts.npy, splits.csv)
- Can't actually load any datasets yet

**Action Required**:
- Create `scripts/preprocess_molecular.py` to convert raw SMILES → features
- Either download or document where to get raw MoleculeNet data
- Generate processed files for all 11 datasets

### 2. License Auditing
**Problem**: 10/11 datasets marked "Unknown - needs audit"
**Risk**: Can't redistribute data without proper licenses

**Action Required**:
- Check MoleculeNet license
- Check individual dataset papers for redistribution rights
- Update registry.json with actual licenses
- Decide on redistribution strategy (host data vs. download scripts)

### 3. No External Baselines Yet (CRITICAL)
**Problem**: Only RAVEL (our method) and uLSIF implemented
**Impact**: Can't claim to be a "benchmark" with only 2 methods

**Action Required** (Priority order):
1. KLIEP (KL importance estimation) - most cited direct method
2. KMM (kernel mean matching) - popular in domain adaptation
3. Weighted Conformal - critical for coverage comparison
4. RULSIF - improved version of uLSIF

### 4. No Evaluation Harness
**Problem**: Can manually evaluate methods, but no automated pipeline

**Action Required**:
- Create `src/shiftbench/evaluate.py` with `run_evaluation()` function
- CLI: `python -m shiftbench.evaluate --method ravel --dataset bace`
- Output: CSV with all decisions + diagnostics

### 5. No Test Data or Tests
**Problem**: Can't verify implementations work correctly

**Action Required**:
- Create `tests/` directory
- Unit tests for base classes
- Integration tests for RAVEL + uLSIF
- Synthetic data tests (known ground truth)

## Next Immediate Steps (Priority Order)

### Step 1: Create Minimal Test Data
**Time**: 1 hour
**Goal**: Verify the infrastructure works

```python
# scripts/create_test_data.py
import numpy as np

# Generate synthetic data for "test_dataset"
n_samples = 1000
n_features = 10
X = np.random.randn(n_samples, n_features)
y = (X[:, 0] > 0).astype(int)
cohorts = np.array([f"cohort_{i % 5}" for i in range(n_samples)])

# Save to data/processed/test_dataset/
np.save("data/processed/test_dataset/features.npy", X)
np.save("data/processed/test_dataset/labels.npy", y)
np.save("data/processed/test_dataset/cohorts.npy", cohorts)

# Create splits
import pandas as pd
splits = pd.DataFrame({
    "uid": range(n_samples),
    "split": ["train"] * 600 + ["cal"] * 200 + ["test"] * 200
})
splits.to_csv("data/processed/test_dataset/splits.csv", index=False)
```

Then test:
```python
from shiftbench.data import load_dataset
X, y, cohorts, splits = load_dataset("test_dataset")
print(f"Loaded {len(X)} samples with {X.shape[1]} features")
```

### Step 2: Test uLSIF on Synthetic Data
**Time**: 30 minutes
**Goal**: Verify uLSIF implementation works

```python
from shiftbench.baselines.ulsif import create_ulsif_baseline

# Load test data
X, y, cohorts, splits = load_dataset("test_dataset")
cal_mask = (splits["split"] == "cal").values
test_mask = (splits["split"] == "test").values

X_cal, y_cal = X[cal_mask], y[cal_mask]
X_test = X[test_mask]

# Estimate weights
ulsif = create_ulsif_baseline(n_basis=50)
weights = ulsif.estimate_weights(X_cal, X_test)

print(f"Weights: mean={weights.mean():.3f}, std={weights.std():.3f}")
print(f"Min={weights.min():.3f}, max={weights.max():.3f}")

# Estimate bounds
predictions_cal = np.ones(len(y_cal), dtype=int)  # All predict positive
decisions = ulsif.estimate_bounds(
    y_cal, predictions_cal, cohorts[cal_mask], weights, tau_grid=[0.5, 0.7, 0.9]
)

for d in decisions:
    print(f"{d.cohort_id} @ τ={d.tau}: {d.decision} (lb={d.lower_bound:.3f})")
```

### Step 3: Implement KLIEP
**Time**: 4-6 hours
**Goal**: Second external baseline

**Template**: Similar structure to uLSIF:
- `src/shiftbench/baselines/kliep.py`
- Use unconstrained optimization (scipy.optimize)
- Maximize log-likelihood: sum(log(K_cal @ alpha))
- Subject to: K_target @ alpha ≥ 0, mean(K_target @ alpha) = 1

### Step 4: License Audit
**Time**: 2-3 hours
**Goal**: Legal compliance

- Check MoleculeNet GitHub for license
- Check individual dataset papers
- Update registry.json
- Document redistribution rights

### Step 5: Preprocessing Scripts
**Time**: 1 day
**Goal**: Can load real datasets

- Download MoleculeNet datasets (or document where to get them)
- Create preprocessing script (SMILES → RDKit fingerprints)
- Generate all 11 processed datasets
- Verify with `load_dataset()`

## Success Metrics (Phase 0 Complete)

- ✅ Dataset registry with 11 molecular datasets
- ✅ Baseline interface (BaselineMethod abstract class)
- ✅ RAVEL wrapper
- ✅ uLSIF implementation
- ⚠️ At least 1 dataset fully processed and loadable (test_dataset counts!)
- ⚠️ At least 1 end-to-end test (load dataset → estimate weights → estimate bounds → decisions)
- 🚧 KLIEP implementation (stretch goal)

## Known Issues / Technical Debt

1. **RAVEL dependency**: `ravel.py` imports from `ravel` package
   - Need to either: (a) copy RAVEL source into shift-bench, or (b) require installation
   - Currently assumes RAVEL is pip installed separately

2. **No input validation in data loader**: `load_dataset()` assumes files exist
   - Should add better error messages
   - Should validate file formats

3. **uLSIF has no FWER control yet**: Uses Holm from RAVEL
   - Should either: (a) copy Holm implementation, or (b) make it a shared utility

4. **No progress tracking**: Long-running methods have no progress bar
   - Consider adding tqdm for large datasets

5. **No caching**: Density ratio estimation is slow, not cached
   - Consider caching weights to disk for repeated evaluations

## Questions for User

1. **RAVEL dependency strategy**: Should we:
   - (A) Copy RAVEL source into shift-bench (cleaner installation)
   - (B) Require separate RAVEL installation (cleaner separation)
   - (C) Extract only the needed modules (middle ground)

2. **Data hosting**: Should we:
   - (A) Host preprocessed data ourselves (easier for users)
   - (B) Provide download scripts only (lower hosting cost)
   - (C) Hybrid (small datasets hosted, large datasets scripted)

3. **Scope prioritization**: Which is more urgent?
   - (A) More baselines (KLIEP, KMM, weighted conformal) with current 11 datasets
   - (B) More datasets (20+ molecular, 10+ text) with current 2 baselines
   - (C) Evaluation infrastructure (harness, aggregation, leaderboard)

4. **License strategy**: If datasets lack clear licenses, should we:
   - (A) Contact original authors for permission
   - (B) Only use clearly-licensed datasets
   - (C) Assume research use is OK (risky)

## Files Created This Session

1. `data/registry.json` - Dataset metadata (11 molecular datasets)
2. `src/shiftbench/__init__.py` - Package exports
3. `src/shiftbench/baselines/__init__.py` - Baseline exports
4. `src/shiftbench/baselines/base.py` - Abstract interface (334 lines)
5. `src/shiftbench/baselines/ravel.py` - RAVEL wrapper (200 lines)
6. `src/shiftbench/baselines/ulsif.py` - uLSIF implementation (254 lines)
7. `src/shiftbench/data.py` - Dataset loading (133 lines)
8. `README.md` - Project documentation
9. `setup.py` - Installation config
10. `STATUS.md` - This file

**Total**: ~1000 lines of production code + documentation
**Time**: ~2-3 hours of focused work

## Conclusion

**What works**: Core abstractions are solid. BaselineMethod interface is clean and extensible. uLSIF implementation is complete and should work (pending testing).

**What's missing**: Can't actually run anything yet because:
1. No processed datasets
2. No test data
3. No evaluation harness

**Recommended next action**: Create test dataset (Step 1) and verify uLSIF works (Step 2). This takes <2 hours and proves the infrastructure is sound. Then proceed to KLIEP implementation and real dataset preprocessing.

**For D&B submission**: Need at minimum:
- 10 baselines (currently have 2 → need 8 more)
- 50 datasets (currently have 11 catalogued → need 39 more)
- Full evaluation results (need harness + compute time)
- Paper (8 pages + appendix)

**Timeline estimate**: With 5-person team working in parallel, 20-24 weeks to full submission-ready benchmark. Without parallelization, ~40 weeks.
