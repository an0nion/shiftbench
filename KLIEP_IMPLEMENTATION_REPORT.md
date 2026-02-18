# KLIEP Baseline Implementation Report

## Summary

Successfully implemented KLIEP (Kullback-Leibler Importance Estimation Procedure) baseline for ShiftBench. The implementation follows the same interface as uLSIF and RAVEL, enabling direct comparison of density ratio estimation methods.

## Implementation Details

### Location
- **Main implementation**: `c:\Users\ananya.salian\Downloads\shift-bench\src\shiftbench\baselines\kliep.py`
- **Test script**: `c:\Users\ananya.salian\Downloads\shift-bench\scripts\test_kliep.py`
- **Updated module**: `c:\Users\ananya.salian\Downloads\shift-bench\src\shiftbench\baselines\__init__.py`

### Algorithm

KLIEP estimates density ratios by maximizing KL divergence between true and estimated target distributions:

1. **Kernel Basis**: Uses Gaussian kernel basis functions centered on random calibration samples (same as uLSIF)
2. **Objective**: Maximize `sum(log(K_target @ alpha)) / n_target`
3. **Constraints**:
   - `alpha >= 0` (element-wise non-negativity)
   - `mean(K_cal @ alpha) = 1` (normalization constraint)
4. **Optimization**: Uses scipy's SLSQP method for constrained optimization
5. **Weights**: Compute `w = K_cal @ alpha` and self-normalize

### Key Differences from uLSIF

| Aspect | KLIEP | uLSIF |
|--------|-------|-------|
| **Loss Function** | KL divergence (log-likelihood) | Squared loss (L2) |
| **Solution** | Iterative optimization (SLSQP) | Closed-form (ridge regression) |
| **Speed** | Slower (~7-9x slower) | Faster (closed-form) |
| **Optimality** | Statistically optimal for KL | Optimal for L2 loss |
| **Stability** | Requires convergence checks | Always converges |
| **Non-negativity** | Guaranteed by constraints | Via max(alpha, 0) |

### Hyperparameters

```python
KLIEPBaseline(
    n_basis=100,           # Number of Gaussian kernel centers
    sigma=None,            # Kernel bandwidth (None = median heuristic)
    max_iter=10000,        # Maximum optimization iterations
    tol=1e-6,              # Convergence tolerance
    random_state=42,       # Random seed
)
```

## Test Results

### Experiment 1: Synthetic Test Dataset

**Dataset**: 1000 samples, 10 features, 5 cohorts
- Calibration: 200 samples (63% positive)
- Test: 200 samples (69% positive)

**Weight Statistics**:
| Method | Mean | Std | Min | Max | 95th % | Runtime (s) |
|--------|------|-----|-----|-----|--------|-------------|
| KLIEP  | 1.000 | 0.169 | 0.432 | 1.517 | 1.226 | 0.067 |
| uLSIF  | 1.000 | 0.126 | 0.561 | 1.256 | 1.189 | 0.004 |
| RAVEL  | 1.000 | 0.243 | 0.619 | 1.586 | 1.419 | 0.187 |

**Weight Correlations**:
- KLIEP vs uLSIF: 0.861 (high agreement)
- KLIEP vs RAVEL: 0.391 (moderate agreement)
- uLSIF vs RAVEL: 0.363 (moderate agreement)

**Certification Results**:
- All methods: 0/30 (0%) certification rate across tau=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
- Reason: Synthetic data has insufficient signal for certification at these thresholds

### Experiment 2: BACE Molecular Dataset

**Dataset**: 1513 samples, 217 features (Morgan fingerprints), 739 scaffolds
- Calibration: 303 samples (43.56% positive)
- Test: 301 samples (45.18% positive)
- Mode: Oracle predictions (using true labels to isolate weight quality)

**Weight Statistics**:
| Method | Mean | Std | Min | Max | 95th % | Runtime (s) |
|--------|------|-----|-----|-----|--------|-------------|
| KLIEP  | 1.000 | 0.221 | 0.436 | 2.679 | 1.344 | 0.089 |
| uLSIF  | 1.000 | 0.135 | 0.298 | 1.194 | 1.150 | 0.013 |
| RAVEL  | 1.000 | 0.150 | 0.721 | 1.413 | 1.364 | 0.200 |

**Weight Correlations**:
- KLIEP vs uLSIF: 0.377 (moderate agreement)
- KLIEP vs RAVEL: 0.300 (low-moderate agreement)
- uLSIF vs RAVEL: -0.002 (no correlation!)

**Certification Results**:
| Tau | KLIEP | uLSIF |
|-----|-------|-------|
| 0.50 | 1/127 (0.8%) | 1/127 (0.8%) |
| 0.60 | 1/127 (0.8%) | 1/127 (0.8%) |
| 0.70 | 0/127 (0.0%) | 0/127 (0.0%) |
| 0.80 | 0/127 (0.0%) | 0/127 (0.0%) |
| 0.85 | 0/127 (0.0%) | 0/127 (0.0%) |
| 0.90 | 0/127 (0.0%) | 0/127 (0.0%) |
| **Overall** | **2/762 (0.3%)** | **2/762 (0.3%)** |

**Runtime Comparison**:
| Method | Weight Time | Bound Time | Total |
|--------|-------------|------------|-------|
| KLIEP  | 0.089s | 0.016s | 0.105s |
| uLSIF  | 0.013s | 0.017s | 0.030s |
| RAVEL  | 0.200s | N/A* | 0.200s |

*RAVEL had integration issues unrelated to weight estimation

## Key Findings

### 1. KLIEP vs uLSIF: Nearly Identical Performance

- **Certification rates**: Exactly the same on both datasets (0% synthetic, 0.3% BACE)
- **Weight correlations**: High on synthetic (0.86), moderate on BACE (0.38)
- **Weight variance**: KLIEP has higher variance (0.169 vs 0.126 synthetic, 0.221 vs 0.135 BACE)
- **Speed**: uLSIF is 7-17x faster due to closed-form solution

### 2. Weight Quality

Both KLIEP and uLSIF produce valid importance weights:
- ✓ Mean ≈ 1.0 (self-normalized)
- ✓ All positive
- ✓ All finite
- ✓ Reasonable ranges (max < 3.0 even on real data)

KLIEP produces slightly more variable weights:
- **Coefficient of variation**: KLIEP 0.17-0.22 vs uLSIF 0.13-0.14
- This is expected: KL optimization is less regularized than ridge regression

### 3. Optimization Convergence

KLIEP's SLSQP optimization:
- ✓ Converged successfully on all tested datasets
- ✓ Fast convergence (~70-90ms for 100 basis functions)
- ✓ No numerical instabilities observed

### 4. Theoretical vs Empirical

**Theory predicts**: KLIEP should be statistically more efficient (optimal for KL divergence)

**Empirical results**: No meaningful difference in certification rates
- Both methods: 0.3% certification on BACE
- Both methods: Identical decisions on 762 cohort-tau pairs

**Interpretation**:
- Sample sizes may be too small to observe statistical efficiency gains
- Empirical-Bernstein bounds are conservative, limiting certification
- Real-world noisy labels and model errors dominate over density estimation differences

## Validation Checklist

- ✅ Implements `BaselineMethod` interface
- ✅ Has `estimate_weights()` and `estimate_bounds()` methods
- ✅ Uses same EB bounds as uLSIF for fair comparison
- ✅ Includes comprehensive docstrings
- ✅ Has factory function `create_kliep_baseline()`
- ✅ Passes validation on synthetic data
- ✅ Passes validation on real molecular data
- ✅ Produces valid weights (positive, finite, normalized)
- ✅ Produces consistent decisions with uLSIF
- ✅ Includes diagnostics (alpha stats, optimization status)
- ✅ Added to baselines `__init__.py`
- ✅ Comprehensive test script with comparisons
- ✅ Results saved to CSV for leaderboard

## Recommendations

### For ShiftBench Users

1. **Use uLSIF by default**: Faster, more stable, equivalent empirical performance
2. **Use KLIEP when**:
   - You need theoretical KL optimality guarantees
   - You have strong reasons to prefer log-likelihood over squared loss
   - Compute time is not a constraint
3. **Use RAVEL for production**: Only method with stability gating (once integration issues fixed)

### For Method Developers

1. **Closed-form > Optimization**: uLSIF's closed-form solution provides huge speed advantage
2. **Regularization matters**: uLSIF's implicit ridge regularization reduces weight variance
3. **EB bounds are conservative**: Both methods certified <1% on real data, limiting observable differences
4. **Sample size matters**: Small calibration sets (n=303) make it hard to observe statistical efficiency differences

## Future Work

### Potential Improvements

1. **Cross-validation for sigma**: Currently uses median heuristic, could CV
2. **Alternative optimizers**: Try L-BFGS-B, trust-constr, or custom solvers
3. **Warm starting**: Initialize alpha from uLSIF solution
4. **Adaptive basis selection**: Choose centers based on importance, not random
5. **Stability gating**: Add PSIS/ESS diagnostics like RAVEL

### Research Questions

1. **When does KLIEP outperform uLSIF?**
   - Hypothesis: Large samples, severe shift, good-quality features
   - Need experiments on larger datasets (n > 1000)

2. **Is KL optimality worth the compute cost?**
   - Current evidence: No, for typical ShiftBench use cases
   - May change with better optimization or larger samples

3. **Can we combine KLIEP + uLSIF?**
   - Ensemble: Average weights from both methods
   - Fallback: Try KLIEP, use uLSIF if optimization fails

## Files Generated

### Code
- `src/shiftbench/baselines/kliep.py` (358 lines)
- `scripts/test_kliep.py` (472 lines)
- Updated `src/shiftbench/baselines/__init__.py`

### Results
- `results/kliep_test_dataset_results.csv`
- `results/ulsif_test_dataset_results.csv`
- `results/comparison_test_dataset_summary.csv`
- `results/kliep_bace_results.csv`
- `results/ulsif_bace_results.csv`
- `results/comparison_bace_summary.csv`

### Documentation
- This report: `KLIEP_IMPLEMENTATION_REPORT.md`

## Conclusion

KLIEP has been successfully implemented and validated for ShiftBench. The implementation:
- Follows ShiftBench interface standards
- Produces valid, reliable importance weights
- Achieves comparable performance to uLSIF
- Runs efficiently on real molecular data
- Is ready for submission to ShiftBench leaderboard

However, **uLSIF remains the recommended baseline** for most users due to:
- 7-17x faster runtime (closed-form vs optimization)
- Lower weight variance (implicit regularization)
- Identical empirical certification rates
- Greater numerical stability

KLIEP serves as an important methodological comparison point, demonstrating that different loss functions (KL vs L2) can achieve similar practical performance on shift-aware evaluation tasks.
