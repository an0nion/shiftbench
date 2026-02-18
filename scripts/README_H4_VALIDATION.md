# H4 Validation Experiment

## Overview

This directory contains `validate_h4.py`, which empirically validates **Hypothesis 4** from `docs/FORMAL_CLAIMS.md`:

> **H4**: EB bounds with n_eff substitution are **conservative** (wider than needed) but **empirically valid** (false-certify <= alpha) under covariate shift.

## What Does This Script Do?

The script tests whether the RAVEL pipeline (or uLSIF baseline) maintains empirical error control when certifying PPV bounds under covariate shift. Specifically:

1. **Generates synthetic data** with KNOWN ground-truth PPV using `SyntheticShiftGenerator`
2. **Runs full certification pipeline**:
   - Estimate density ratios (importance weights)
   - Compute EB lower bounds per (cohort, tau) pair
   - Apply Holm step-down for FWER control
   - Decide: CERTIFY vs ABSTAIN vs NO-GUARANTEE
3. **Measures empirical error rates**:
   - **False-Certify Rate (FWER)**: P(>=1 false certification per trial)
   - **Per-Test False-Certify**: Among certified decisions, fraction where true PPV < tau
   - **Coverage**: Fraction where true PPV >= certified lower bound
4. **Compares to nominal alpha** (0.05) using binomial statistical test

## Success Criteria

**H4 is VALIDATED if**: False-certify rate <= alpha (0.05) with p >= 0.05 in binomial test

**H4 is REJECTED if**: False-certify rate > alpha with statistical significance (p < 0.05)

## Quick Start

### 1. Quick Test (3 trials, ~30 seconds)

Test that the script works and produces reasonable results:

```bash
python scripts/validate_h4.py --n_trials 3 --use_ulsif --output results/h4_quick_test.csv
```

**Expected output**:
- False-certify rate: 0.00 - 0.33 (with only 3 trials, high variance)
- Certification rate: 5-20% (conservative bounds abstain frequently)
- Coverage: ~100% (bounds should be conservative)

### 2. Standard Validation (30 trials, ~5 minutes)

More reliable estimate of error control:

```bash
python scripts/validate_h4.py --n_trials 30 --use_ulsif --output results/h4_standard.csv
```

### 3. Full Validation (100 trials, ~20 minutes)

Publication-ready validation with narrow confidence intervals:

```bash
python scripts/validate_h4.py --n_trials 100 --use_ulsif --output results/h4_full.csv
```

**Statistical Power**: With 100 trials and alpha=0.05:
- If true FWER = 0.05, we'll observe 3-7 false certifications (95% CI)
- If true FWER = 0.10 (2x nominal), we'll detect violation with 80% power

## Stress Tests

Test robustness under varying conditions:

### Varying Shift Severity

```bash
# Mild shift (easy case)
python scripts/validate_h4.py --n_trials 20 --shift_severity 0.5 --use_ulsif

# Moderate shift (standard)
python scripts/validate_h4.py --n_trials 20 --shift_severity 1.0 --use_ulsif

# Severe shift (hard case)
python scripts/validate_h4.py --n_trials 20 --shift_severity 2.0 --use_ulsif
```

**Expected**: Error control should hold across all shift levels (conservativeness increases with shift)

### Varying Cohort Sizes

```bash
# Small cohorts (molecular-like)
python scripts/validate_h4.py --n_trials 20 --n_cohorts 50 --n_cal 500 --use_ulsif

# Large cohorts (text-like)
python scripts/validate_h4.py --n_trials 20 --n_cohorts 10 --n_cal 5000 --use_ulsif
```

**Expected**: Small cohorts -> wide bounds -> more abstentions (but still valid)

### Full Stress Test Sweep

Run all combinations (shift severity × cohort size × positive rate):

```bash
python scripts/validate_h4.py \
    --n_trials 10 \
    --stress_tests \
    --stress_shift 0.5 1.0 1.5 2.0 \
    --stress_cohorts 5 10 20 50 \
    --stress_rates 0.3 0.5 0.7 \
    --use_ulsif \
    --output results/h4_stress_full.csv
```

**Duration**: 10 trials × 4 shifts × 4 cohort sizes × 3 rates = 480 trials (~1-2 hours)

## Output Format

Results are saved as CSV with these columns:

| Column | Description |
|--------|-------------|
| `trial_id` | Trial number (0, 1, 2, ...) |
| `n_cal` | Calibration set size |
| `n_test` | Test set size (large for accurate ground truth) |
| `n_cohorts` | Number of cohorts |
| `shift_severity` | Covariate shift magnitude (0=none, 2=severe) |
| `positive_rate` | Target P(Y=1) in calibration |
| `false_certify_fwer` | 1 if trial had >=1 false certification, 0 otherwise |
| `false_certify_count` | Number of false certifications in trial |
| `n_certified` | Number of CERTIFY decisions |
| `n_abstain` | Number of ABSTAIN decisions |
| `n_no_guarantee` | Number of NO-GUARANTEE decisions |
| `coverage` | Fraction where true PPV >= lower bound |
| `mean_n_eff` | Mean effective sample size across cohorts |
| `mean_ppv_estimate` | Mean estimated PPV (certified cohorts only) |
| `mean_true_ppv` | Mean true PPV (certified cohorts only) |
| `runtime_seconds` | Time to run this trial |

## Analyzing Results

### 1. Check FWER Control

```python
import pandas as pd
from scipy.stats import binomtest

df = pd.read_csv("results/h4_full.csv")
fwer = df["false_certify_fwer"].mean()
n_trials = len(df)

result = binomtest(
    k=int(df["false_certify_fwer"].sum()),
    n=n_trials,
    p=0.05,
    alternative="greater"
)

print(f"FWER: {fwer:.3f} (target: <= 0.05)")
print(f"p-value: {result.pvalue:.4f}")
print("Status:", "VALID" if result.pvalue >= 0.05 else "INVALID")
```

### 2. Examine Coverage

```python
coverage = df["coverage"].mean()
print(f"Coverage: {coverage:.3f} (target: >= 0.95)")
```

Coverage should be ~95% or higher (bounds are conservative).

### 3. Plot FWER vs Shift Severity

```python
import matplotlib.pyplot as plt
import seaborn as sns

grouped = df.groupby("shift_severity")["false_certify_fwer"].agg(["mean", "sem"])
plt.errorbar(grouped.index, grouped["mean"], yerr=1.96*grouped["sem"], marker="o")
plt.axhline(0.05, color="red", linestyle="--", label="Nominal alpha")
plt.xlabel("Shift Severity")
plt.ylabel("False-Certify Rate (FWER)")
plt.legend()
plt.savefig("h4_fwer_vs_shift.pdf")
```

### 4. Certification Rate Trade-off

```python
df["cert_rate"] = df["n_certified"] / (df["n_certified"] + df["n_abstain"] + df["n_no_guarantee"])

plt.scatter(df["false_certify_fwer"], df["cert_rate"], alpha=0.5)
plt.axvline(0.05, color="red", linestyle="--", label="Target FWER")
plt.xlabel("False-Certify (FWER)")
plt.ylabel("Certification Rate")
plt.legend()
plt.savefig("h4_power_validity_tradeoff.pdf")
```

## Command-Line Options

### Experiment Configuration

- `--n_trials N`: Number of validation trials (default: 3)
- `--n_cal N`: Calibration set size per trial (default: 500)
- `--n_test N`: Test set size for ground truth (default: 5000, must be large)
- `--n_cohorts N`: Number of cohorts per trial (default: 10)
- `--shift_severity S`: Covariate shift magnitude (default: 1.0, range 0-2)
- `--positive_rate P`: Target P(Y=1) in calibration (default: 0.5)

### Stress Test Options

- `--stress_tests`: Enable stress test mode (sweeps over multiple configs)
- `--stress_shift S1 S2 ...`: Shift severities for stress tests (default: 0.5 1.0 1.5 2.0)
- `--stress_cohorts N1 N2 ...`: Cohort sizes for stress tests (default: 5 10 20 50)
- `--stress_rates P1 P2 ...`: Positive rates for stress tests (default: 0.3 0.5 0.7)

### Method Configuration

- `--alpha A`: Nominal FWER level (default: 0.05)
- `--tau_grid T1 T2 ...`: PPV thresholds to test (default: 0.5 0.6 0.7 0.8 0.9)
- `--use_ulsif`: Use uLSIF baseline instead of RAVEL (no gating, faster)

### Output Options

- `--output PATH`: Output CSV path (default: auto-generated with timestamp)
- `--quiet`: Suppress progress messages (only final summary)

## Troubleshooting

### Issue: "RAVEL not available"

**Cause**: RAVEL package not installed or import failed

**Solution**: Use `--use_ulsif` flag to use simplified baseline (no gating)

```bash
python scripts/validate_h4.py --n_trials 10 --use_ulsif
```

### Issue: Very low certification rate (<1%)

**Cause**: Shift too severe OR cohorts too small OR bounds too conservative

**Check**:
- `mean_n_eff` in output (should be >10)
- `shift_severity` (values >2.0 are extreme)
- `n_cohorts` vs `n_cal` ratio (each cohort needs ~20+ samples)

**Solution**: Reduce shift severity or increase calibration size:

```bash
python scripts/validate_h4.py --n_trials 10 --shift_severity 0.5 --n_cal 1000
```

### Issue: False-certify rate too high (>0.10)

**Cause**: EB bounds not conservative enough OR bug in implementation

**Action**:
1. Verify ground-truth PPV computation is correct (check test set size is large, e.g., 5000+)
2. Check weight estimation diagnostics (ESS, max weight)
3. Report issue with full config and output CSV

### Issue: Script runs very slowly

**Cause**: Large trial count OR RAVEL cross-validation overhead

**Solution**:
- Use `--use_ulsif` (5-10x faster than RAVEL)
- Reduce `--n_trials` for quick tests
- Reduce `--n_cal` (smaller calibration sets are faster)

## Expected Runtime

On typical laptop (4-core, 16GB RAM):

| Configuration | Runtime per Trial | Total (100 trials) |
|--------------|-------------------|-------------------|
| uLSIF, n_cal=500 | 0.01-0.05s | ~2-5 minutes |
| RAVEL, n_cal=500 | 0.5-2s | ~1-3 hours |
| uLSIF, n_cal=5000 | 0.1-0.5s | ~10-50 minutes |

Stress tests with 480 configs: multiply by 5-10x

## Interpreting Results

### Scenario 1: FWER = 0.03, p = 0.85

**Interpretation**: Valid! False-certify rate is below nominal alpha.

**Explanation**: Bounds are conservative (wide) due to n_eff downweighting.

**Action**: H4 is validated. Proceed to paper writing.

### Scenario 2: FWER = 0.08, p = 0.02

**Interpretation**: Invalid! False-certify rate exceeds alpha with statistical significance.

**Explanation**: Bounds are too optimistic (narrow). Possible causes:
- n_eff formula incorrect (not conservative enough)
- Holm correction not applied correctly
- Sample-splitting violated (same data used for weights and bounds)

**Action**:
1. Debug implementation (check EB bound formula, Holm procedure)
2. Increase conservativeness (stricter gates, larger n_eff penalty)
3. Document failure in paper (honest reporting)

### Scenario 3: FWER = 0.05, p = 0.50

**Interpretation**: Boundary case. Empirically valid but not conservative.

**Explanation**: Bounds are exactly at nominal level (tight).

**Action**:
- Run more trials (100 -> 500) to narrow confidence interval
- If FWER remains at 0.05 ± 0.01, claim "empirically valid" (not "conservative")

### Scenario 4: Certification rate = 0% (all abstain)

**Interpretation**: Bounds are TOO conservative (over-abstention).

**Explanation**: Shift too severe OR cohorts too small OR gates too strict.

**Action**:
- Check diagnostics: mean_n_eff, shift severity
- If shift_severity > 2.0: expected behavior (severe shift is hard)
- If shift_severity < 1.0: bounds may be unnecessarily wide

## Connection to Formal Claims

This script validates **P4.1** from FORMAL_CLAIMS.md:

> **P4.1**: False-certify rate should be <= alpha in synthetic data with known ground truth

**Testable Prediction**: 100 trials with varying shift, measure false-certify frequency

**Expected**: False-certify rate <= 0.05 (nominal alpha) with 95% confidence

**Falsification**: If FWER > alpha with p < 0.05, REJECT H4

## Next Steps After Validation

1. **If H4 is VALIDATED**:
   - Include results in paper Section 4 (Validity Study)
   - Report FWER, coverage, and certification rate
   - Proceed to H1-H3 (real-data experiments)

2. **If H4 is REJECTED**:
   - Investigate root cause (debug EB bounds, Holm procedure, n_eff formula)
   - Try stricter gates (increase ess_min_frac, decrease psis_k_cap)
   - Document failure honestly in paper
   - Consider alternative bounds (Hoeffding, bootstrap)

3. **Paper Writing**:
   - Include stress test plots (FWER vs shift, coverage vs n_eff)
   - Report full results table (CSV -> LaTeX table)
   - Discuss conservativeness vs power trade-off

## References

- **FORMAL_CLAIMS.md**: Pre-registered hypotheses and validation criteria
- **synthetic_shift_generator.py**: Data generation with known ground truth
- **RAVEL paper**: Certification under covariate shift (Anthropic, 2024)
- **Holm (1979)**: A simple sequentially rejective multiple test procedure

## Author

Created: 2026-02-17
Last Updated: 2026-02-17
Contact: ShiftBench team
