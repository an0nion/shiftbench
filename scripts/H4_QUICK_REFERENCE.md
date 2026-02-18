# H4 Validation - Quick Reference Card

## One-Line Summary

Empirically validate that false-certify rate <= alpha (0.05) under covariate shift using synthetic data with known ground-truth PPV.

## Essential Commands

### Quick Test (30 seconds)
```bash
python scripts/validate_h4.py --n_trials 3 --use_ulsif
```

### Standard Validation (5 minutes)
```bash
python scripts/validate_h4.py --n_trials 30 --use_ulsif --output results/h4_standard.csv
```

### Full Validation (20 minutes)
```bash
python scripts/validate_h4.py --n_trials 100 --use_ulsif --output results/h4_full.csv
```

### Full Stress Tests (1-2 hours)
```bash
python scripts/validate_h4.py --n_trials 10 --stress_tests --use_ulsif --output results/h4_stress.csv
```

## Key Metrics

| Metric | Target | Interpretation |
|--------|--------|----------------|
| **False-Certify Rate (FWER)** | <= 0.05 | P(>=1 false cert per trial) |
| **Coverage** | >= 0.95 | Fraction with true PPV >= lower bound |
| **Certification Rate** | 5-30% | Higher = more power (but may sacrifice validity) |

## Pass/Fail Criteria

**H4 VALIDATED** if:
- FWER <= 0.05 with p >= 0.05 (binomial test)
- Coverage >= 0.95

**H4 REJECTED** if:
- FWER > 0.05 with p < 0.05 (statistically significant violation)

## Output Files

Default location: `results/h4_validation_<timestamp>.csv`

Key columns:
- `false_certify_fwer`: 1 if trial had false cert, 0 otherwise
- `n_certified`, `n_abstain`, `n_no_guarantee`: Decision counts
- `coverage`: Fraction with true PPV >= lower bound
- `mean_n_eff`: Effective sample size (diagnostic)

## Quick Analysis

```python
import pandas as pd
from scipy.stats import binomtest

df = pd.read_csv("results/h4_full.csv")
fwer = df["false_certify_fwer"].mean()
result = binomtest(int(df["false_certify_fwer"].sum()), len(df), 0.05, alternative="greater")

print(f"FWER: {fwer:.3f} (target: <= 0.05)")
print(f"p-value: {result.pvalue:.4f}")
print("Status:", "PASS" if result.pvalue >= 0.05 else "FAIL")
```

## Stress Test Variations

```bash
# Severe shift
python scripts/validate_h4.py --n_trials 20 --shift_severity 2.0 --use_ulsif

# Small cohorts (molecular-like)
python scripts/validate_h4.py --n_trials 20 --n_cohorts 50 --use_ulsif

# Low positive rate
python scripts/validate_h4.py --n_trials 20 --positive_rate 0.2 --use_ulsif
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "RAVEL not available" | Add `--use_ulsif` flag |
| Low cert rate (<1%) | Normal for severe shift or small cohorts |
| High FWER (>0.10) | Check n_test is large (5000+), verify implementation |
| Slow runtime | Use `--use_ulsif` (10x faster than RAVEL) |

## Documentation

- **Full guide**: `README_H4_VALIDATION.md`
- **Implementation summary**: `H4_VALIDATION_SUMMARY.md`
- **Source code**: `validate_h4.py`
- **Theory**: `../docs/FORMAL_CLAIMS.md` (Hypothesis 4)

## Expected Results

Based on 3-trial quick tests:
- FWER: 0.00-0.33 (high variance with few trials)
- Coverage: ~100% (conservative bounds)
- Certification rate: 3-20% (depends on shift severity)
- Runtime: 0.01-0.05s per trial (uLSIF)

With 100 trials:
- FWER: 0.03-0.07 (95% CI if true FWER = 0.05)
- Certification rate: 5-15% (conservative)
- Runtime: ~20 minutes total

## Next Steps After Validation

1. If **PASS**: Include results in paper Section 4, proceed to H1-H3
2. If **FAIL**: Debug EB bounds, increase conservativeness, document honestly

---

**Quick Help**: `python scripts/validate_h4.py --help`
**Created**: 2026-02-17
