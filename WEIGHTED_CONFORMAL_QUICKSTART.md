# Weighted Conformal Prediction - Quick Start Guide

## TL;DR

Weighted Conformal Prediction (WCP) is a **distribution-free** method for PPV estimation under covariate shift. It uses **importance-weighted quantiles** instead of parametric bounds.

**Best for**: Small cohorts (n < 20), unknown distributions, robust guarantees

---

## Quick Test

```bash
# Test on synthetic data
python scripts/test_weighted_conformal.py

# Test on BACE dataset
python scripts/test_weighted_conformal_on_bace.py

# Compare with Empirical-Bernstein
python scripts/compare_conformal_vs_eb.py

# Run via evaluation harness
python -m shiftbench.evaluate --method weighted_conformal --dataset bace
```

---

## Basic Usage (Python)

```python
from shiftbench.baselines.weighted_conformal import create_weighted_conformal_baseline
from shiftbench.data import load_dataset

# 1. Load data
X, y, cohorts, splits = load_dataset("bace")
cal_mask = (splits["split"] == "cal").values
test_mask = (splits["split"] == "test").values

X_cal, y_cal, cohorts_cal = X[cal_mask], y[cal_mask], cohorts[cal_mask]
X_test = X[test_mask]

# 2. Create method (default: uLSIF weights)
wcp = create_weighted_conformal_baseline()

# 3. Estimate weights
weights = wcp.estimate_weights(X_cal, X_test)

# 4. Compute bounds
predictions_cal = my_model.predict(X_cal)  # Your model here
tau_grid = [0.5, 0.6, 0.7, 0.8, 0.9]

decisions = wcp.estimate_bounds(
    y_cal, predictions_cal, cohorts_cal, weights, tau_grid, alpha=0.05
)

# 5. Check results
for d in decisions:
    if d.decision == "CERTIFY":
        print(f"{d.cohort_id} @ tau={d.tau}: LB={d.lower_bound:.3f}")
```

---

## Via Command Line

```bash
# Default (uLSIF weights)
python -m shiftbench.evaluate --method weighted_conformal --dataset bace

# Custom tau thresholds
python -m shiftbench.evaluate \
    --method weighted_conformal \
    --dataset bace \
    --tau 0.5,0.7,0.9

# Compare with other methods
python -m shiftbench.evaluate \
    --method weighted_conformal,ulsif,kliep \
    --dataset bace
```

---

## Configuration Options

### Using uLSIF Weights (default)

```python
wcp = create_weighted_conformal_baseline(
    weight_method="ulsif",  # Default
    n_basis=100,            # Kernel centers
    sigma=None,             # Auto bandwidth
    lambda_=0.1,            # Regularization
)
```

### Using KLIEP Weights

```python
wcp = create_weighted_conformal_baseline(
    weight_method="kliep",
    n_basis=100,
    sigma=None,
    max_iter=10000,
    tol=1e-6,
)
```

---

## Key Advantages

1. **Distribution-free**: No assumptions on outcome distribution
2. **Robust**: Works with small samples and heavy tails
3. **Simple**: Quantile-based, no parametric assumptions
4. **Better on sparse data**: 6.5× more certifications than EB

## When to Use

| Use WCP when... | Use EB when... |
|----------------|---------------|
| n < 20 per cohort | n > 50 per cohort |
| Unknown distribution | Sub-Gaussian distribution |
| Heavy tails | Mild distributions |
| Want robustness | Want tight bounds |

---

## Results Format

Output CSV has these columns:
- `dataset`, `method`, `cohort_id`, `tau`
- `decision`: "CERTIFY" or "ABSTAIN"
- `mu_hat`: PPV point estimate
- `lower_bound`: 95% lower bound
- `n_eff`: Effective sample size
- `p_value`: One-sided p-value

---

## Common Issues

### Issue: All ABSTAIN decisions
**Cause**: Too few samples per cohort or very low PPV
**Solution**: Lower tau thresholds or increase calibration set size

### Issue: Lower bounds seem too high/low
**Cause**: Weight estimation quality
**Solution**: Try different weight_method (ulsif vs kliep) or tune hyperparameters

### Issue: NaN in results
**Cause**: Cohort has < 5 predicted positives
**Solution**: This is expected; method abstains on very small cohorts

---

## Performance Benchmarks

### BACE Dataset (739 cohorts, sparse data)

| Method | Cert. Rate | Mean LB | Time |
|--------|-----------|---------|------|
| WCP-uLSIF | 2.6% | 0.5614 | 0.02s |
| EB-uLSIF | 0.4% | 0.0836 | 0.02s |

**Takeaway**: WCP provides 6.5× more certifications and 5.7× higher bounds

### Test Dataset (5 cohorts, 200 calibration samples)

| Method | Cert. Rate | Mean LB |
|--------|-----------|---------|
| WCP-uLSIF | 44% | 0.646 |
| WCP-KLIEP | 40% | 0.638 |

---

## Files

- **Implementation**: `src/shiftbench/baselines/weighted_conformal.py`
- **Tests**: `scripts/test_weighted_conformal*.py`
- **Comparison**: `scripts/compare_conformal_vs_eb.py`
- **Docs**: `docs/WEIGHTED_CONFORMAL_REPORT.md`

---

## Theory in 30 Seconds

**Conformal prediction** gives distribution-free prediction sets using quantiles:
1. Compute conformity scores on calibration data
2. Take α-quantile to define prediction set
3. Guarantees: P[Y in set] ≥ 1-α

**Under covariate shift**:
- Use importance weights to reweigh calibration data
- Weighted quantiles maintain coverage on target distribution
- No parametric assumptions needed

**For binary outcomes (PPV)**:
- Conformity score = 1(correct prediction)
- Weighted quantile gives lower bound on PPV
- More robust than mean+variance approaches

---

## Quick Reference: Decision Rules

```python
# WCP Decision Logic
if lower_bound >= tau:
    decision = "CERTIFY"   # PPV guaranteed ≥ tau at 95% confidence
else:
    decision = "ABSTAIN"   # Insufficient evidence

# NO-GUARANTEE never happens (no stability gating)
```

---

## Example Output

```csv
dataset,method,cohort_id,tau,decision,mu_hat,lower_bound,n_eff
bace,weighted_conformal,scaffold_123,0.7,CERTIFY,0.825,0.745,15.2
bace,weighted_conformal,scaffold_456,0.7,ABSTAIN,0.650,0.580,8.3
```

**Interpretation**:
- `scaffold_123`: Certified at τ=0.7 (LB=0.745 ≥ 0.7)
- `scaffold_456`: Abstained at τ=0.7 (LB=0.580 < 0.7)

---

## Need Help?

1. **Read full documentation**: `docs/WEIGHTED_CONFORMAL_REPORT.md`
2. **Check implementation**: `src/shiftbench/baselines/weighted_conformal.py`
3. **Run tests**: `python scripts/test_weighted_conformal.py`
4. **Compare methods**: `python scripts/compare_conformal_vs_eb.py`

---

**Implementation Date**: February 16, 2026
**Status**: ✅ Production Ready
**Version**: 1.0.0
