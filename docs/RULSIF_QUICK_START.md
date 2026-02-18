# RULSIF Quick Start Guide

## What is RULSIF?

**RULSIF** (Relative unconstrained Least-Squares Importance Fitting) is a density ratio estimation method that provides **more stable** importance weights than standard uLSIF when distributions differ significantly.

### Key Difference from uLSIF

- **uLSIF**: Estimates `r(x) = p_target(x) / p_cal(x)` (can be unstable when `p_cal(x) ≈ 0`)
- **RULSIF**: Estimates `r(x) = p_target(x) / (α·p_target(x) + (1-α)·p_cal(x))` (more stable)

The parameter `α ∈ [0, 1]` controls the stability-accuracy trade-off:
- `α = 0.0`: Standard density ratio (equivalent to uLSIF)
- `α = 0.1`: Recommended default (slight stabilization)
- `α = 0.5`: Maximum stability
- `α = 1.0`: No reweighting (all weights = 1)

---

## Installation

RULSIF is included in ShiftBench:

```bash
cd shift-bench
pip install -e .
```

---

## Basic Usage

### 1. Import

```python
from shiftbench.baselines import create_rulsif_baseline
```

### 2. Create RULSIF Instance

```python
# Default configuration (alpha=0.1)
rulsif = create_rulsif_baseline()

# Or customize
rulsif = create_rulsif_baseline(
    n_basis=100,    # Number of kernel centers
    alpha=0.1,      # Relative parameter
    lambda_=0.1,    # Ridge regularization
    sigma=None,     # Kernel bandwidth (None = auto)
    random_state=42
)
```

### 3. Estimate Importance Weights

```python
# X_cal: Calibration features (n_cal, n_features)
# X_target: Target features (n_target, n_features)

weights = rulsif.estimate_weights(X_cal, X_target)

# Check weights
print(f"Mean: {weights.mean():.3f}")  # Should be ~1.0
print(f"Std: {weights.std():.3f}")
print(f"Min: {weights.min():.3f}, Max: {weights.max():.3f}")
```

### 4. Estimate PPV Bounds

```python
decisions = rulsif.estimate_bounds(
    y_cal=y_cal,                      # True labels
    predictions_cal=predictions_cal,  # Model predictions
    cohort_ids_cal=cohort_ids_cal,    # Cohort identifiers
    weights=weights,                  # From step 3
    tau_grid=[0.5, 0.7, 0.9],        # PPV thresholds to test
    alpha=0.05                        # Significance level
)

# Analyze results
for d in decisions:
    print(f"Cohort: {d.cohort_id}, Tau: {d.tau}, Decision: {d.decision}")
    print(f"  PPV estimate: {d.mu_hat:.3f}")
    print(f"  Lower bound: {d.lower_bound:.3f}")
```

---

## Complete Example

```python
import numpy as np
from shiftbench.baselines import create_rulsif_baseline

# Create synthetic data with distribution shift
np.random.seed(42)
X_cal = np.random.randn(200, 10)
X_target = np.random.randn(200, 10) + 0.5  # Shift by 0.5

# Create RULSIF
rulsif = create_rulsif_baseline(n_basis=50, alpha=0.1)

# Estimate weights
weights = rulsif.estimate_weights(X_cal, X_target)
print(f"Weights: mean={weights.mean():.3f}, std={weights.std():.3f}")

# Create labels and predictions
y_cal = np.random.binomial(1, 0.7, 200)
predictions_cal = np.ones(200, dtype=int)
cohorts_cal = np.array([f"cohort_{i%5}" for i in range(200)])

# Estimate bounds
decisions = rulsif.estimate_bounds(
    y_cal, predictions_cal, cohorts_cal, weights,
    tau_grid=[0.5, 0.7, 0.9],
    alpha=0.05
)

# Summary
n_certify = sum(1 for d in decisions if d.decision == "CERTIFY")
print(f"Certified {n_certify}/{len(decisions)} (cohort, tau) pairs")

# Diagnostics
diag = rulsif.get_diagnostics()
print(f"Method: {diag['method']}, alpha: {diag['alpha_rel']}")
```

---

## Comparing RULSIF vs. uLSIF

```python
from shiftbench.baselines import create_ulsif_baseline, create_rulsif_baseline

# Create both methods with same settings
ulsif = create_ulsif_baseline(n_basis=100, random_state=42)
rulsif = create_rulsif_baseline(n_basis=100, alpha=0.1, random_state=42)

# Estimate weights
weights_ulsif = ulsif.estimate_weights(X_cal, X_target)
weights_rulsif = rulsif.estimate_weights(X_cal, X_target)

# Compare stability (lower CV = more stable)
cv_ulsif = weights_ulsif.std() / weights_ulsif.mean()
cv_rulsif = weights_rulsif.std() / weights_rulsif.mean()

print(f"uLSIF CV: {cv_ulsif:.4f}")
print(f"RULSIF CV: {cv_rulsif:.4f}")
print(f"Stability improvement: {(cv_ulsif - cv_rulsif) / cv_ulsif * 100:+.1f}%")

# Compare diagnostics
print("\nuLSIF diagnostics:", ulsif.get_diagnostics())
print("\nRULSIF diagnostics:", rulsif.get_diagnostics())
```

---

## Tuning the Alpha Parameter

```python
# Test different alpha values
alphas = [0.0, 0.1, 0.3, 0.5]
results = {}

for alpha in alphas:
    rulsif = create_rulsif_baseline(n_basis=100, alpha=alpha, random_state=42)
    weights = rulsif.estimate_weights(X_cal, X_target)
    cv = weights.std() / weights.mean()
    results[alpha] = {
        'cv': cv,
        'mean': weights.mean(),
        'std': weights.std(),
        'min': weights.min(),
        'max': weights.max()
    }

# Print comparison
print(f"{'Alpha':<8} {'CV':<10} {'Min':<10} {'Max':<10} {'Range':<10}")
print("-" * 50)
for alpha, stats in results.items():
    print(f"{alpha:<8.1f} {stats['cv']:<10.4f} {stats['min']:<10.3f} "
          f"{stats['max']:<10.3f} {stats['max'] - stats['min']:<10.3f}")
```

---

## Testing

### Unit Tests

```bash
python scripts/test_rulsif_unit.py
```

### Synthetic Data Test

```bash
python scripts/create_test_data.py  # Create test dataset
python scripts/test_rulsif.py       # Test RULSIF vs uLSIF
```

### Comprehensive Comparison (NEW!)

```bash
# Compare uLSIF vs RULSIF with multiple alpha values
python scripts/compare_ulsif_vs_rulsif.py --dataset test_dataset

# Features:
# - Tests alpha = 0.0, 0.1, 0.5, 0.9
# - Weight stability analysis (CV, variance reduction)
# - Decision agreement rates
# - Certification rate comparison
# - Generates CSV summaries and visualizations

# Output files (in results/):
# - ulsif_vs_rulsif_{dataset}_summary.csv
# - {method}_{dataset}_results.csv (for each method)
# - ulsif_vs_rulsif_{dataset}_weights.png (histograms)
# - ulsif_vs_rulsif_{dataset}_cv_vs_alpha.png (stability plot)
```

### BACE Dataset Test

```bash
python scripts/preprocess_molecular.py --dataset bace  # Preprocess BACE
python scripts/test_rulsif_on_bace.py                  # Test on BACE
python scripts/compare_ulsif_vs_rulsif.py --dataset bace  # Compare methods
```

---

## When to Use RULSIF

### Use RULSIF when:
- ✅ Distribution shift is **severe** (e.g., scaffold split, domain adaptation)
- ✅ uLSIF produces **extreme weights** (high variance, outliers)
- ✅ You need **more stable** importance weights
- ✅ You can tolerate **slight bias** in density ratio (usually negligible)

### Use uLSIF when:
- ✅ Distribution shift is **moderate**
- ✅ You need **unbiased** density ratio estimation
- ✅ Weight variance is acceptable
- ✅ Computational speed is critical (RULSIF slightly slower)

---

## API Reference

### `create_rulsif_baseline(**kwargs)`

Create a RULSIF baseline instance.

**Parameters**:
- `n_basis` (int, default=100): Number of Gaussian kernel centers
- `sigma` (float or None, default=None): Kernel bandwidth (None = median heuristic)
- `lambda_` (float, default=0.1): Ridge regularization parameter
- `alpha` (float, default=0.1): Relative parameter in [0, 1]
  - 0.0 = standard density ratio (uLSIF)
  - 0.1 = slight stabilization (recommended)
  - 0.5 = maximum stability
  - 1.0 = no reweighting
- `random_state` (int, default=42): Random seed

**Returns**: `RULSIFBaseline` instance

### `RULSIFBaseline.estimate_weights(X_cal, X_target, domain_labels=None)`

Estimate importance weights.

**Parameters**:
- `X_cal` (np.ndarray): Calibration features (n_cal, n_features)
- `X_target` (np.ndarray): Target features (n_target, n_features)
- `domain_labels` (optional): Not used (RULSIF is direct method)

**Returns**: `np.ndarray` of weights (n_cal,)

### `RULSIFBaseline.estimate_bounds(...)`

Estimate PPV lower bounds using Empirical-Bernstein.

**Parameters**:
- `y_cal` (np.ndarray): Binary labels (n_cal,)
- `predictions_cal` (np.ndarray): Binary predictions (n_cal,)
- `cohort_ids_cal` (np.ndarray): Cohort identifiers (n_cal,)
- `weights` (np.ndarray): Importance weights (n_cal,)
- `tau_grid` (List[float]): PPV thresholds to test
- `alpha` (float, default=0.05): Significance level

**Returns**: `List[CohortDecision]`

### `RULSIFBaseline.get_metadata()`

Get method metadata.

**Returns**: `MethodMetadata` with name, version, description, paper info

### `RULSIFBaseline.get_diagnostics()`

Get method-specific diagnostics.

**Returns**: Dict with:
- `method`: "rulsif"
- `sigma`: Kernel bandwidth
- `n_basis`: Number of kernel centers
- `alpha_rel`: Relative parameter value
- `theta_min`, `theta_max`, `theta_std`: Kernel weight statistics
- `weights_raw_std`, `weights_raw_min`, `weights_raw_max`: Raw weight statistics

---

## References

1. **Yamada et al. 2013**. "Relative Density-Ratio Estimation for Robust Distribution Comparison"
   *Neural Computation* 25(5):1324-1370.
   [DOI:10.1162/NECO_a_00442](https://doi.org/10.1162/NECO_a_00442)

2. **Sugiyama et al. 2012**. "Density Ratio Estimation in Machine Learning"
   Cambridge University Press. (Chapter 9)

---

## Support

- **Full documentation**: `docs/RULSIF_IMPLEMENTATION_REPORT.md`
- **Implementation**: `src/shiftbench/baselines/rulsif.py`
- **Tests**: `scripts/test_rulsif*.py`
- **Issues**: Report bugs or ask questions in ShiftBench repository

---

**Quick Links**:
- [Full Implementation Report](RULSIF_IMPLEMENTATION_REPORT.md)
- [ShiftBench README](../README.md)
- [BaselineMethod API](../src/shiftbench/baselines/base.py)
