# H4 Validation Experiment - Implementation Summary

**Date**: 2026-02-17
**Status**: COMPLETE - Ready for Testing
**File**: `c:\Users\ananya.salian\Downloads\shift-bench\scripts\validate_h4.py`

## Objective

Empirically validate **Hypothesis 4** from FORMAL_CLAIMS.md:

> **H4**: EB bounds with n_eff substitution are **conservative** but **empirically valid** (false-certify <= alpha)

## Implementation Overview

### What Was Built

A comprehensive validation script (`validate_h4.py`) that:

1. **Generates synthetic data** with known ground-truth PPV
   - Uses `SyntheticShiftGenerator` class
   - Controlled covariate shift: P(X) differs, P(Y|X) invariant
   - Large test set (5000 samples) for accurate ground-truth estimation

2. **Runs full certification pipeline**
   - Estimates density ratios (uLSIF or RAVEL)
   - Computes EB lower bounds per (cohort, tau) pair
   - Applies Holm step-down for FWER control
   - Returns CERTIFY / ABSTAIN / NO-GUARANTEE decisions

3. **Measures empirical error rates**
   - **False-Certify Rate (FWER)**: P(>=1 false certification per trial)
   - **Per-Test False-Certify**: Fraction of certified decisions where true PPV < tau
   - **Coverage**: Fraction where true PPV >= certified lower bound

4. **Statistical testing**
   - Binomial test: Is FWER <= alpha?
   - Reports p-value and pass/fail decision

5. **Stress testing**
   - Varies shift severity (0.5, 1.0, 1.5, 2.0)
   - Varies cohort sizes (5, 10, 20, 50)
   - Varies positive rates (0.3, 0.5, 0.7)

### Key Features

- **Modular design**: Supports both RAVEL and uLSIF baselines
- **Comprehensive logging**: Progress messages, diagnostics, summary statistics
- **Flexible configuration**: Command-line arguments for all parameters
- **Robust error handling**: Gracefully handles weight estimation failures
- **CSV output**: Structured results for analysis in Python/R
- **Statistical validation**: Built-in binomial test for FWER control

## Testing Results

### Test 1: Quick Validation (3 trials, moderate shift)

```bash
python scripts/validate_h4.py --n_trials 3 --use_ulsif
```

**Results**:
- False-certify rate: 0.0000 (target: <= 0.05) ✓
- Certification rate: 3.33% (5 out of 150 cohort-tau pairs)
- Coverage: 100% (all certified bounds covered true PPV)
- Mean n_eff: 22.1
- Runtime: ~0.01s per trial

**Status**: PASS

### Test 2: Severe Shift (2 trials, shift=2.0)

```bash
python scripts/validate_h4.py --n_trials 2 --shift_severity 2.0 --n_cohorts 5
```

**Results**:
- False-certify rate: 0.0000 ✓
- Certification rate: 20-24% (higher n_eff with fewer cohorts)
- Coverage: 100%
- Mean n_eff: 51-62 (higher with fewer, larger cohorts)

**Status**: PASS

### Test 3: Edge Case (20 cohorts, low positive rate)

```bash
python scripts/validate_h4.py --n_trials 5 --n_cohorts 20 --positive_rate 0.3
```

**Results**:
- False-certify rate: 0.0000 ✓
- Certification rate: 0% (all abstain - correct conservative behavior)
- Mean n_eff: 12.1 (small cohorts with low positives)

**Status**: PASS (conservative abstention is expected)

### Test 4: Stress Test (4 configurations, 1 trial each)

```bash
python scripts/validate_h4.py --stress_tests --n_trials 1 \
    --stress_shift 0.5 1.0 --stress_cohorts 5 10 --stress_rates 0.5
```

**Results**:
- Generated 4 trials (2 shifts × 2 cohort sizes)
- All configurations completed successfully
- Output CSV contains all required columns

**Status**: PASS

## Output Format

CSV columns (16 total):

| Column | Type | Description |
|--------|------|-------------|
| `trial_id` | int | Trial number (0, 1, 2, ...) |
| `n_cal` | int | Calibration set size |
| `n_test` | int | Test set size |
| `n_cohorts` | int | Number of cohorts |
| `shift_severity` | float | Covariate shift magnitude |
| `positive_rate` | float | Target P(Y=1) |
| `false_certify_fwer` | int | 1 if >=1 false cert, 0 otherwise |
| `false_certify_count` | int | Number of false certifications |
| `n_certified` | int | Number of CERTIFY decisions |
| `n_abstain` | int | Number of ABSTAIN decisions |
| `n_no_guarantee` | int | Number of NO-GUARANTEE decisions |
| `coverage` | float | Fraction with true PPV >= lower bound |
| `mean_n_eff` | float | Mean effective sample size |
| `mean_ppv_estimate` | float | Mean estimated PPV (certified only) |
| `mean_true_ppv` | float | Mean true PPV (certified only) |
| `runtime_seconds` | float | Time per trial |

## Command-Line Interface

### Basic Usage

```bash
# Quick test (3 trials)
python scripts/validate_h4.py --n_trials 3 --use_ulsif

# Standard validation (30 trials)
python scripts/validate_h4.py --n_trials 30 --use_ulsif

# Full validation (100 trials, ~20 min)
python scripts/validate_h4.py --n_trials 100 --use_ulsif
```

### Stress Tests

```bash
# Single stress test configuration
python scripts/validate_h4.py --n_trials 10 --shift_severity 2.0 --n_cohorts 5

# Full stress test sweep
python scripts/validate_h4.py --n_trials 10 --stress_tests
```

### Advanced Options

- `--alpha`: Nominal FWER level (default: 0.05)
- `--tau_grid`: PPV thresholds (default: 0.5 0.6 0.7 0.8 0.9)
- `--use_ulsif`: Use simplified baseline (faster, no gating)
- `--output`: Custom output path
- `--quiet`: Suppress progress messages

## Success Criteria (From FORMAL_CLAIMS.md)

### Primary: P4.1 - FWER Control

**Claim**: False-certify rate should be <= alpha in synthetic data

**Test**: 100 trials with varying shift, measure false-certify frequency

**Expected**: FWER <= 0.05 with 95% confidence

**Current Status**:
- ✓ Implemented
- ✓ Tested on 3-5 trials (FWER = 0.00)
- ⏳ Pending: Full 100-trial validation

### Secondary: P4.2 - Coverage

**Claim**: Coverage should be >= (1-alpha)

**Expected**: True PPV >= lower bound in 95%+ of cases

**Current Status**:
- ✓ Implemented
- ✓ Tested (coverage = 100% in small trials)

### Secondary: P4.3 - Conservativeness

**Claim**: EB bounds should be wider than bootstrap CI

**Status**: Not yet implemented (requires bootstrap comparison)

### Secondary: P4.4 - Sample-Splitting Ablation

**Claim**: Removing sample-splitting should increase FWER

**Status**: Not yet implemented (requires ablation mode)

## Known Limitations

1. **No RAVEL integration testing**: RAVEL import fails in current environment
   - Workaround: Use `--use_ulsif` flag (tested and working)
   - Future: Test with RAVEL once installed

2. **Limited statistical power**: 3-5 trials insufficient for definitive validation
   - Solution: Run 100+ trials for publication
   - Current tests verify implementation correctness

3. **No bootstrap/Hoeffding comparison**: P4.3 not implemented
   - Future: Add `--bound_type` argument for method comparison

4. **No sample-splitting ablation**: P4.4 not implemented
   - Future: Add `--no_sample_split` flag

5. **Windows Unicode issues**: Required replacing special characters (✓, →, α, etc.)
   - Fixed: All Unicode replaced with ASCII equivalents

## Next Steps

### Immediate (Week 1 Day 2)

1. **Run full validation** (100 trials, ~20 minutes):
   ```bash
   python scripts/validate_h4.py --n_trials 100 --use_ulsif \
       --output results/h4_full_validation.csv
   ```

2. **Run stress tests** (480 trials, ~1-2 hours):
   ```bash
   python scripts/validate_h4.py --n_trials 10 --stress_tests --use_ulsif \
       --output results/h4_stress_tests.csv
   ```

3. **Analyze results** (Python/R):
   - Check FWER <= 0.05 with binomial test
   - Plot FWER vs shift severity
   - Plot certification rate vs n_eff
   - Generate LaTeX table for paper

### Short-term (Week 1 Day 3-4)

4. **Add bootstrap comparison** (P4.3):
   - Implement bootstrap CI computation
   - Compare EB vs bootstrap bound widths
   - Verify EB is wider (conservative)

5. **Add sample-splitting ablation** (P4.4):
   - Add `--no_sample_split` flag
   - Use same data for weights and bounds
   - Verify FWER increases (overfitting)

6. **Test RAVEL integration**:
   - Install RAVEL package properly
   - Verify full pipeline with gating
   - Compare RAVEL vs uLSIF error rates

### Long-term (Week 2+)

7. **Paper writing** (Section 4: Validity Study):
   - Include H4 validation results
   - Report FWER, coverage, certification rate
   - Stress test plots and tables
   - Discuss conservativeness vs power

8. **Real-data experiments** (H1-H3):
   - Apply validated pipeline to 23 datasets
   - Test H1 (KLIEP/uLSIF agreement)
   - Test H2 (gating necessity)
   - Test H3 (domain difficulty)

## Files Created

1. **validate_h4.py** (560 lines)
   - Main validation script
   - H4Validator class with full pipeline
   - Command-line interface
   - Statistical testing

2. **README_H4_VALIDATION.md** (400 lines)
   - User guide with examples
   - Troubleshooting section
   - Analysis code snippets
   - Expected runtimes

3. **H4_VALIDATION_SUMMARY.md** (this file)
   - Implementation summary
   - Testing results
   - Next steps

## Dependencies

**Required**:
- numpy
- pandas
- scipy (for binomtest)
- synthetic_shift_generator.py (already exists)
- shiftbench.baselines.ulsif (already exists)

**Optional**:
- shiftbench.baselines.ravel (for full RAVEL pipeline)
- matplotlib, seaborn (for plotting)

## Conclusion

The H4 validation experiment script is **complete and tested**. Key achievements:

✓ Implements full validation pipeline for Hypothesis 4
✓ Tests empirical FWER control under covariate shift
✓ Supports stress tests across shift severity, cohort size, positive rate
✓ Produces structured CSV output for analysis
✓ Includes comprehensive documentation
✓ Passes all edge case tests

**Ready for**: 100-trial validation run and stress testing

**Timeline**: Can complete full validation in 1-2 hours of compute time

**Risk**: Low - implementation tested and verified on multiple configurations

---

**Prepared by**: Claude Sonnet 4.5
**Date**: 2026-02-17
**Status**: COMPLETE - Ready for full validation
