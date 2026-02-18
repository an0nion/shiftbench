# KMM (Kernel Mean Matching) - Quick Start Guide

## Overview

KMM is a density ratio estimation method that minimizes Maximum Mean Discrepancy (MMD) between weighted calibration and target distributions. It provides explicit weight bounds to prevent extreme reweighting.

**When to use KMM**:
- You need bounded weights (0 <= w_i <= B)
- You prefer MMD-based distribution matching
- Runtime is acceptable (1-2 seconds for 300 samples)
- You want guaranteed convex optimization

## Installation

KMM is included in ShiftBench. No additional dependencies required (cvxpy optional but recommended for better QP solving).

```bash
# Optional: Install cvxpy for better optimization
pip install cvxpy
```

## Basic Usage

```python
from shiftbench.baselines import create_kmm_baseline

# Create KMM baseline with default hyperparameters
kmm = create_kmm_baseline()

# Estimate importance weights
weights = kmm.estimate_weights(X_cal, X_target)

# Estimate PPV bounds
decisions = kmm.estimate_bounds(
    y_cal=y_cal,
    predictions_cal=predictions_cal,
    cohort_ids_cal=cohort_ids_cal,
    weights=weights,
    tau_grid=[0.5, 0.7, 0.9],
    alpha=0.05,
)
```

## Hyperparameters

### Default Configuration
```python
kmm = create_kmm_baseline(
    sigma=None,        # Kernel bandwidth (None = median heuristic)
    lambda_=0.1,       # Ridge regularization
    B=1000.0,          # Box constraint upper bound
    solver="auto",     # QP solver ("cvxpy" or "scipy")
    random_state=42,   # Random seed
)
```

### Tuning Guidelines

1. **sigma (Kernel Bandwidth)**
   - Default: `None` (uses median heuristic)
   - Lower values: More flexible, may overfit
   - Higher values: More regularized, may underfit
   - Typical range: 0.1 to 100 (depends on feature scale)

2. **lambda_ (Ridge Penalty)**
   - Default: `0.1`
   - Higher values: Smoother weights, more stable
   - Lower values: More flexible, may be unstable
   - Typical range: 0.001 to 1.0

3. **B (Box Constraint)**
   - Default: `1000.0`
   - Lower values: Tighter bounds, more conservative
   - Higher values: Less restrictive
   - Monitor clipping fraction in diagnostics
   - Typical range: 10 to 10000

4. **solver**
   - `"auto"`: Use cvxpy if available, else scipy (recommended)
   - `"cvxpy"`: More robust, requires cvxpy package
   - `"scipy"`: Faster for small problems, no extra dependencies

## Examples

### Example 1: Basic Usage
```python
from shiftbench.baselines import create_kmm_baseline
import numpy as np

# Generate synthetic data
np.random.seed(42)
X_cal = np.random.randn(100, 10)
X_target = np.random.randn(100, 10) + 0.5  # Shifted distribution

# Estimate weights
kmm = create_kmm_baseline()
weights = kmm.estimate_weights(X_cal, X_target)

print(f"Weight mean: {weights.mean():.3f}")  # Should be ~1.0
print(f"Weight std: {weights.std():.3f}")
print(f"Weight max: {weights.max():.3f}")

# Check diagnostics
diag = kmm.get_diagnostics()
print(f"Kernel bandwidth: {diag['sigma']:.3f}")
print(f"Weights clipped: {diag['weights_clipped_fraction']:.1%}")
print(f"Optimization success: {diag['optimization_success']}")
```

### Example 2: Custom Hyperparameters
```python
# Tighter bounds for conservative reweighting
kmm_conservative = create_kmm_baseline(
    sigma=1.0,      # Fixed bandwidth
    lambda_=0.5,    # More regularization
    B=100.0,        # Tighter bounds
)

weights = kmm_conservative.estimate_weights(X_cal, X_target)
```

### Example 3: Full Pipeline
```python
from shiftbench.baselines import create_kmm_baseline
from shiftbench.data import load_dataset

# Load dataset
X, y, cohorts, splits = load_dataset("bace")

# Split data
cal_mask = (splits["split"] == "cal").values
test_mask = (splits["split"] == "test").values

X_cal, y_cal, cohorts_cal = X[cal_mask], y[cal_mask], cohorts[cal_mask]
X_test = X[test_mask]

# Estimate weights
kmm = create_kmm_baseline()
weights = kmm.estimate_weights(X_cal, X_test)

# Assume we have predictions
predictions_cal = np.ones(len(y_cal), dtype=int)  # Naive: predict all positive

# Estimate bounds
decisions = kmm.estimate_bounds(
    y_cal=y_cal,
    predictions_cal=predictions_cal,
    cohort_ids_cal=cohorts_cal,
    weights=weights,
    tau_grid=[0.5, 0.7, 0.9],
    alpha=0.05,
)

# Analyze results
n_certify = sum(1 for d in decisions if d.decision == "CERTIFY")
print(f"Certified: {n_certify}/{len(decisions)} cohort-tau pairs")
```

## Diagnostics

KMM provides detailed diagnostics via `get_diagnostics()`:

```python
diag = kmm.get_diagnostics()

print(f"Kernel bandwidth (sigma): {diag['sigma']:.4f}")
print(f"Ridge penalty (lambda): {diag['lambda']:.4f}")
print(f"Box constraint (B): {diag['B']:.1f}")
print(f"Weight min: {diag['weights_min']:.4f}")
print(f"Weight max: {diag['weights_max']:.4f}")
print(f"Weight mean: {diag['weights_mean']:.4f}")
print(f"Weight std: {diag['weights_std']:.4f}")
print(f"Weights clipped (% at B): {diag['weights_clipped_fraction']:.1%}")
print(f"Optimization success: {diag['optimization_success']}")
print(f"Solve time: {diag['solve_time']:.3f}s")
```

**Key Diagnostics**:
- `weights_clipped_fraction`: If high (>10%), consider increasing B
- `optimization_success`: Should be True; if False, try increasing lambda_ or reducing B
- `solve_time`: Increases with sample size (O(n³) complexity)
- `weights_max`: Should be < B (if equals B, many weights clipped)

## Comparison with Other Methods

### KMM vs uLSIF vs KLIEP

| Feature | KMM | uLSIF | KLIEP |
|---------|-----|-------|-------|
| **Objective** | MMD minimization | L2 loss | KL divergence |
| **Optimization** | QP (convex) | Closed-form | Nonlinear |
| **Runtime (300 samples)** | ~1.5s | ~0.08s | ~0.12s |
| **Weight Bounds** | Explicit (0 <= w <= B) | Implicit | Implicit |
| **Weight Variance** | High (1.08) | Low (0.14) | Medium (0.22) |
| **Max Weight** | Bounded (8.0) | Moderate (1.2) | Higher (2.7) |
| **Stability** | High (convex QP) | High (closed-form) | Medium |

**Key Takeaway**: KMM is slower but provides explicit weight bounds, preventing extreme reweighting.

## Common Issues and Solutions

### Issue 1: Optimization Fails
**Symptom**: `optimization_success = False`

**Solutions**:
1. Increase lambda_ (e.g., 0.1 → 1.0) for more regularization
2. Reduce B (e.g., 1000 → 100) for tighter constraints
3. Check for degenerate data (e.g., constant features)
4. Switch solver: `solver="cvxpy"` or `solver="scipy"`

### Issue 2: Many Weights Clipped
**Symptom**: `weights_clipped_fraction > 0.1`

**Solutions**:
1. Increase B (e.g., 1000 → 10000)
2. Increase lambda_ to regularize weights
3. Check for severe distribution shift
4. Consider using RAVEL (has stability gating)

### Issue 3: Runtime Too Slow
**Symptom**: `solve_time > 10s`

**Solutions**:
1. Reduce sample size (subsample calibration set)
2. Use uLSIF instead (closed-form, much faster)
3. Use basis selection (not yet implemented)
4. Check QP solver (cvxpy may be faster than scipy)

### Issue 4: Weights Have High Variance
**Symptom**: `weights_std > 2.0`

**Solutions**:
1. Increase lambda_ for more regularization
2. Reduce B for tighter bounds
3. Check for severe shift (may need RAVEL's gating)
4. Compare with uLSIF/KLIEP (should have lower variance)

## Testing and Validation

Run the provided test scripts to validate KMM:

```bash
# Full comparison study
python scripts/test_kmm.py

# Quick comparison with uLSIF and KLIEP
python scripts/compare_kmm_ulsif_kliep.py --dataset test_dataset
python scripts/compare_kmm_ulsif_kliep.py --dataset bace
```

## References

1. Huang et al. 2007. "Correcting Sample Selection Bias by Unlabeled Data." NIPS 2007.
2. Gretton et al. 2009. "Covariate Shift by Kernel Mean Matching." In "Dataset Shift in Machine Learning", MIT Press.

## API Reference

### `create_kmm_baseline(**kwargs)`
Factory function to create KMM baseline instance.

**Parameters**:
- `sigma` (float, optional): Kernel bandwidth. Default: None (median heuristic)
- `lambda_` (float): Ridge penalty. Default: 0.1
- `B` (float): Box constraint upper bound. Default: 1000.0
- `solver` (str): QP solver ("cvxpy", "scipy", or "auto"). Default: "auto"
- `random_state` (int): Random seed. Default: 42

**Returns**:
- `KMMBaseline` instance

### `KMMBaseline.estimate_weights(X_cal, X_target, domain_labels=None)`
Estimate importance weights from calibration to target distribution.

**Parameters**:
- `X_cal` (ndarray): Calibration features (n_cal, n_features)
- `X_target` (ndarray): Target features (n_target, n_features)
- `domain_labels` (ndarray, optional): Not used (KMM is direct method)

**Returns**:
- `weights` (ndarray): Importance weights (n_cal,)

### `KMMBaseline.estimate_bounds(...)`
Estimate PPV lower bounds using Empirical-Bernstein.

See `BaselineMethod.estimate_bounds()` for full API.

### `KMMBaseline.get_metadata()`
Return method metadata (name, version, description, paper URL, etc.).

### `KMMBaseline.get_diagnostics()`
Return KMM-specific diagnostics (sigma, lambda, B, clipping, solve time, etc.).

## Support

For issues or questions:
1. Check the full report: `docs/KMM_IMPLEMENTATION_REPORT.md`
2. Review test scripts: `scripts/test_kmm.py`, `scripts/compare_kmm_ulsif_kliep.py`
3. Compare with uLSIF/KLIEP implementations in `src/shiftbench/baselines/`
