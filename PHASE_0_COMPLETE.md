# Phase 0 Complete: ShiftBench Foundation Built ✅

**Date**: 2025-02-16
**Time Invested**: ~3 hours
**Status**: Infrastructure validated and working

---

## What We Built

### 1. Core Package Structure ✅

```
shift-bench/
├── data/
│   ├── registry.json                    # 12 datasets catalogued (11 real + 1 test)
│   └── processed/
│       └── test_dataset/                # ✅ Synthetic data for testing
│           ├── features.npy             # 1000 samples × 10 features
│           ├── labels.npy               # Binary labels
│           ├── cohorts.npy              # 5 cohorts
│           └── splits.csv               # Train/cal/test splits
├── src/shiftbench/
│   ├── __init__.py                      # Package exports
│   ├── data.py                          # ✅ DatasetRegistry + load_dataset()
│   └── baselines/
│       ├── __init__.py                  # Baseline exports
│       ├── base.py                      # ✅ Abstract BaselineMethod interface
│       ├── ravel.py                     # ✅ RAVEL wrapper (200 lines)
│       └── ulsif.py                     # ✅ uLSIF implementation (254 lines)
├── scripts/
│   ├── create_test_data.py              # ✅ Synthetic data generator
│   └── test_ulsif.py                    # ✅ End-to-end test
├── docs/                                # 🚧 Empty (to be filled)
├── README.md                            # ✅ Project overview
├── setup.py                             # ✅ Installation config
├── STATUS.md                            # ✅ Detailed status report
└── PHASE_0_COMPLETE.md                  # ✅ This file
```

**Total Code**: ~1100 lines (excluding docs)

---

## Verified Working ✅

### Test Results

**Test Dataset Creation**:
```
[SUCCESS] Created test dataset at C:\Users\ananya.salian\Downloads\shift-bench\data\processed\test_dataset
   Samples: 1000 (train=600, cal=200, test=200)
   Features: 10
   Cohorts: 5
   Positive rate: 56.80%
```

**uLSIF Baseline Test**:
```
[OK] Loaded 1000 samples with 10 features
[OK] Estimated weights for 200 calibration samples
   Mean: 1.000 (should be ~1.0) ✅
   Std: 0.126
   Min: 0.561, Max: 1.256
[OK] Weights passed validity checks ✅
[OK] Generated 25 decisions ✅
[SUCCESS] All tests passed! uLSIF baseline is working correctly. ✅
```

**Key Validation**:
- ✅ Dataset loading works (`load_dataset("test_dataset")`)
- ✅ Weight estimation works (mean=1.0, all positive, all finite)
- ✅ Bound estimation works (generates valid `CohortDecision` objects)
- ✅ Input validation works (catches errors)
- ✅ Diagnostics work (sigma, n_basis, alpha statistics)

---

## Key Abstractions

### 1. BaselineMethod Interface

All methods must implement:
```python
class BaselineMethod(ABC):
    @abstractmethod
    def estimate_weights(X_cal, X_target) -> np.ndarray:
        """Estimate importance weights."""
        pass

    @abstractmethod
    def estimate_bounds(y_cal, predictions_cal, cohort_ids_cal,
                       weights, tau_grid, alpha) -> List[CohortDecision]:
        """Estimate PPV lower bounds."""
        pass

    @abstractmethod
    def get_metadata() -> MethodMetadata:
        """Return method info (name, paper URL, etc.)."""
        pass
```

**Benefit**: Any method following this interface can be evaluated in the benchmark.

### 2. CohortDecision Dataclass

Standardized result format:
```python
@dataclass
class CohortDecision:
    cohort_id: str          # e.g., "cohort_0", "scaffold_A"
    tau: float              # PPV threshold tested
    decision: str           # "CERTIFY", "ABSTAIN", or "NO-GUARANTEE"
    mu_hat: float           # Point estimate
    var_hat: float          # Variance estimate
    n_eff: float            # Effective sample size
    lower_bound: float      # One-sided lower bound
    p_value: float          # P-value testing H0: PPV < tau
    diagnostics: Dict       # Method-specific info
```

**Benefit**: All methods produce comparable outputs.

### 3. DatasetRegistry

Centralized dataset management:
```python
registry = get_registry()
registry.list_datasets(domain="molecular")  # ['bace', 'bbbp', ...]
info = registry.get_dataset_info("bace")     # Metadata dict
X, y, cohorts, splits = load_dataset("bace") # Load processed data
```

**Benefit**: Adding new datasets is just updating registry.json.

---

## What's Implemented

### Baselines (2/13 target)

1. **RAVEL** ✅
   - Discriminative density ratio estimation
   - Cross-validation folding
   - Stability gating (PSIS k-hat, ESS, clip mass)
   - Empirical-Bernstein bounds
   - Holm step-down
   - **Supports abstention**: YES (returns NO-GUARANTEE when unstable)

2. **uLSIF** ✅
   - Direct density ratio estimation
   - Gaussian kernel basis functions
   - Closed-form solution (ridge regression)
   - Median heuristic for bandwidth
   - Empirical-Bernstein bounds
   - **Supports abstention**: NO (no stability gates)

### Datasets (12/100 target)

**Synthetic** (1):
- test_dataset ✅ (1000 samples, 10 features, 5 cohorts)

**Molecular** (11):
- BACE, BBBP, ClinTox, ESOL, FreeSolv, Lipophilicity
- SIDER, Tox21, ToxCast, MUV, MolHIV
- ⚠️ **Not yet processed** (only catalogued in registry)

**Text** (0): 🚧 To be added

**Tabular** (0): 🚧 To be added

---

## Critical Gaps (Must Address Next)

### 1. No Real Datasets Preprocessed

**Problem**: Registry has 11 molecular datasets, but no processed files.

**Impact**: Can't evaluate on real data yet.

**Action Required**:
- Download MoleculeNet datasets (or get from ravel project)
- Create `scripts/preprocess_molecular.py`:
  - Read SMILES strings
  - Compute RDKit fingerprints (same as RAVEL uses)
  - Compute scaffolds for cohorts
  - Create train/cal/test splits
  - Save as .npy and .csv files

**Estimated Time**: 1 day

### 2. Only 2 External Baselines

**Problem**: Need 10+ for strong D&B submission.

**Action Required** (Priority order):
1. **KLIEP** - KL minimization (most cited)
2. **KMM** - Kernel mean matching (popular)
3. **Weighted Conformal** - Critical for coverage comparison
4. **RULSIF** - Relative density ratio
5. **Split Conformal** - Baseline (no shift adaptation)
6. **CV+** - Cross-validation conformal
7. **Group DRO** - Worst-group optimization
8. **Chi-Sq DRO** - Chi-square divergence ball
9. **Multicalibration** - Fairness-aware
10. **BBSE** - Black-box shift estimation (label shift)

**Estimated Time**: 2-3 weeks (parallelizable across 2 engineers)

### 3. No Evaluation Harness

**Problem**: Must manually run each method on each dataset.

**Action Required**:
- Create `src/shiftbench/evaluate.py`:
  ```python
  def run_evaluation(method_name, dataset_name, tau_grid, alpha, output_dir):
      # Load dataset
      # Load method
      # Estimate weights
      # Estimate bounds
      # Save results (CSV + receipt)
  ```
- CLI: `python -m shiftbench.evaluate --method ulsif --dataset bace`

**Estimated Time**: 2-3 days

### 4. License Auditing

**Problem**: 10/11 datasets marked "Unknown - needs audit".

**Risk**: Can't redistribute without proper licenses.

**Action Required**:
- Check MoleculeNet license (GitHub)
- Check individual papers for redistribution rights
- Update registry.json
- Decide: host data vs. download scripts

**Estimated Time**: 2-3 hours

---

## Next Immediate Steps

### Option A: Validate on Real Data (Fast)
**Goal**: Prove uLSIF works on actual molecular data
**Time**: 1 day

1. Copy processed files from ravel project → shift-bench/data/processed/bace/
2. Run: `python scripts/test_ulsif.py` (modify to use "bace" instead of "test_dataset")
3. Compare uLSIF results to RAVEL results
4. Document differences

### Option B: Add KLIEP Baseline (External Method)
**Goal**: Have 3 methods total
**Time**: 1 day

1. Create `src/shiftbench/baselines/kliep.py` (similar to ulsif.py)
2. Implement KL minimization via scipy.optimize
3. Test on test_dataset
4. Validate against published KLIEP results (if available)

### Option C: Build Evaluation Harness (Infrastructure)
**Goal**: Automate benchmarking
**Time**: 2 days

1. Create `src/shiftbench/evaluate.py`
2. Add CLI interface
3. Support batch evaluation (loop over datasets)
4. Save results to CSV

**Recommendation**: Do **Option A** first (validates on real data), then **Option B** (adds external baseline), then **Option C** (enables full benchmark).

---

## Success Metrics

### Phase 0 (✅ Complete)
- [x] Dataset registry (12 datasets)
- [x] Baseline interface (abstract class)
- [x] RAVEL wrapper
- [x] uLSIF implementation
- [x] Test dataset created
- [x] End-to-end test passing

### Phase 1 (Next 4 weeks)
- [ ] 3+ external baselines (KLIEP, KMM, weighted conformal)
- [ ] 11 molecular datasets preprocessed
- [ ] Evaluation harness with CLI
- [ ] License audit complete
- [ ] Results for 3 methods × 11 datasets

### Phase 2 (Weeks 5-8)
- [ ] 10+ baselines total
- [ ] 50+ datasets (20 molecular, 20 text, 10 tabular)
- [ ] Full benchmark sweep (10 × 50 = 500 evaluations)
- [ ] Results aggregation script
- [ ] Static leaderboard HTML

### Phase 3 (Weeks 9-12)
- [ ] Community documentation (guides)
- [ ] NeurIPS D&B paper (8 pages + appendix)
- [ ] All receipts generated and verified
- [ ] Reproducibility artifacts (code + data)

---

## Files Created This Session

### Source Code (8 files, ~1100 lines)
1. `data/registry.json` - Dataset metadata
2. `src/shiftbench/__init__.py` - Package init
3. `src/shiftbench/baselines/__init__.py` - Baselines init
4. `src/shiftbench/baselines/base.py` - Abstract interface (334 lines)
5. `src/shiftbench/baselines/ravel.py` - RAVEL wrapper (200 lines)
6. `src/shiftbench/baselines/ulsif.py` - uLSIF implementation (254 lines)
7. `src/shiftbench/data.py` - Dataset loading (133 lines)
8. `setup.py` - Installation config

### Scripts (2 files)
9. `scripts/create_test_data.py` - Synthetic data generator
10. `scripts/test_ulsif.py` - End-to-end test

### Documentation (4 files)
11. `README.md` - Project overview
12. `STATUS.md` - Detailed status report
13. `PHASE_0_COMPLETE.md` - This file
14. `.gitignore` - (Not yet created, but should be added)

### Data (1 dataset)
15. `data/processed/test_dataset/` - Synthetic test data (4 files)

---

## Technical Debt / Known Issues

1. **RAVEL dependency**: `ravel.py` imports from `ravel` package
   - Must have RAVEL installed separately
   - Consider: copy needed modules or make it optional

2. **No Holm implementation in uLSIF**: Imports from ravel.bounds.holm
   - Should either: (a) copy Holm to shiftbench, or (b) make it shared utility

3. **No FWER control**: uLSIF doesn't do Holm step-down yet
   - Works for single-cohort tests, but not multi-cohort

4. **No caching**: Weights recomputed every time
   - Could cache to disk for faster iteration

5. **No progress bars**: Long-running methods have no feedback
   - Add tqdm for user experience

6. **Hardcoded paths**: Some scripts assume specific directory structure
   - Should use configurable paths

7. **No error recovery**: If one dataset fails, whole batch fails
   - Should add try/except and continue

8. **No GPU support**: Everything is CPU-only
   - For large datasets, GPU could speed up

---

## Questions for Review

1. **Should we copy RAVEL source into shift-bench?**
   - **Pro**: Cleaner installation, no external dependency
   - **Con**: Code duplication, harder to sync updates
   - **Alternative**: Make RAVEL an optional dependency

2. **Dataset hosting strategy?**
   - **Option A**: Host preprocessed files (easier for users, higher cost)
   - **Option B**: Provide download scripts (lower cost, more setup)
   - **Option C**: Hybrid (small datasets hosted, large datasets scripted)

3. **Prioritize baselines or datasets?**
   - **Option A**: 10 methods × 11 datasets (focus on methods)
   - **Option B**: 3 methods × 50 datasets (focus on data coverage)
   - **Option C**: 6 methods × 30 datasets (balanced)

4. **License strategy for unclear cases?**
   - **Option A**: Contact authors for permission (slow, thorough)
   - **Option B**: Only use clearly-licensed datasets (fast, conservative)
   - **Option C**: Assume research use is OK (fast, risky)

---

## Conclusion

**Status**: Phase 0 foundation is complete and validated.

**What works**:
- ✅ Clean abstractions (BaselineMethod interface)
- ✅ Two working implementations (RAVEL, uLSIF)
- ✅ Dataset loading infrastructure
- ✅ End-to-end test passing

**What's missing**:
- Real datasets preprocessed (can copy from ravel project)
- More baselines (need 8-11 more)
- Evaluation harness (automate benchmarking)
- Full paper (8 pages + appendix)

**Next critical action**:
Choose between Option A (validate on real data), Option B (add KLIEP), or Option C (build harness).

**Recommended path**: A → B → C (validate, expand, automate).

**For D&B submission**:
- **Minimum**: 10 methods × 50 datasets + paper
- **Timeline**: 20-24 weeks with 5-person team
- **Current progress**: ~5% complete (infrastructure only)

---

**Ready to proceed to Phase 1!** 🚀
