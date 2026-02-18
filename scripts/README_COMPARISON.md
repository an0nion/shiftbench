# RULSIF vs uLSIF Comparison Tool

**Script**: `compare_ulsif_vs_rulsif.py`
**Purpose**: Comprehensive comparison of uLSIF and RULSIF with multiple alpha values
**Status**: Production Ready

---

## Quick Start

```bash
# Compare on synthetic test data
python scripts/compare_ulsif_vs_rulsif.py --dataset test_dataset

# Compare on BACE molecular data
python scripts/compare_ulsif_vs_rulsif.py --dataset bace

# Compare on any preprocessed dataset
python scripts/compare_ulsif_vs_rulsif.py --dataset {dataset_name}
```

---

## What It Does

This script provides a comprehensive comparison between **uLSIF** (standard density ratio) and **RULSIF** (relative density ratio) with different alpha values.

### Methods Tested:
1. **uLSIF** - Standard density ratio (baseline)
2. **RULSIF(alpha=0.0)** - Should match uLSIF exactly
3. **RULSIF(alpha=0.1)** - Default, slight stabilization
4. **RULSIF(alpha=0.5)** - Maximum stability
5. **RULSIF(alpha=0.9)** - Very high stability

### Analysis Performed:

**1. Weight Stability Analysis**
- Coefficient of Variation (CV = std/mean)
- Variance reduction percentages
- Weight ranges (min, max)
- Improvement metrics vs uLSIF

**2. Decision Agreement**
- Agreement rates between methods
- Number of disagreements
- Types of disagreements (who certifies what)

**3. Certification Performance**
- Certification rates for each method
- Lower bound comparisons
- Effective sample size (n_eff)

**4. Runtime Performance**
- Weight estimation time
- Bound estimation time
- Total runtime comparison

---

## Output Files

All files are saved in `results/` directory:

### 1. Summary CSV
**File**: `ulsif_vs_rulsif_{dataset}_summary.csv`

**Columns**:
```
method              - Method name (uLSIF, RULSIF(alpha=X))
weight_mean         - Mean importance weight (should be ~1.0)
weight_std          - Standard deviation of weights
weight_cv           - Coefficient of variation (std/mean)
weight_min          - Minimum weight value
weight_max          - Maximum weight value
weight_time         - Time to estimate weights (seconds)
bound_time          - Time to estimate bounds (seconds)
total_time          - Total runtime (seconds)
n_certify           - Number of CERTIFY decisions
n_abstain           - Number of ABSTAIN decisions
n_no_guarantee      - Number of NO-GUARANTEE decisions
```

**Example**:
```csv
method,weight_cv,n_certify,n_abstain
uLSIF,0.1255,0,25
RULSIF(alpha=0.0),0.1255,0,25
RULSIF(alpha=0.1),0.1254,0,25
RULSIF(alpha=0.5),0.1244,0,25
RULSIF(alpha=0.9),0.1179,0,25
```

### 2. Decision CSVs (5 files)

**Files**:
- `ulsif_{dataset}_results.csv`
- `rulsifalpha=00_{dataset}_results.csv`
- `rulsifalpha=01_{dataset}_results.csv`
- `rulsifalpha=05_{dataset}_results.csv`
- `rulsifalpha=09_{dataset}_results.csv`

**Columns**:
```
cohort_id      - Cohort identifier
tau            - PPV threshold
decision       - CERTIFY, ABSTAIN, or NO-GUARANTEE
mu_hat         - Point estimate of PPV
lower_bound    - Lower confidence bound
n_eff          - Effective sample size
p_value        - One-sided p-value
```

### 3. Visualizations

**a) Weight Distributions**: `ulsif_vs_rulsif_{dataset}_weights.png`
- 2x2 grid of histograms
- One subplot per method (uLSIF, RULSIF α=0.1, 0.5, 0.9)
- Shows distribution of importance weights
- Red dashed line indicates mean weight
- Title shows CV, mean, and range

**b) CV vs Alpha**: `ulsif_vs_rulsif_{dataset}_cv_vs_alpha.png`
- Line plot showing CV as function of alpha
- Red dashed horizontal line = uLSIF baseline
- Lower CV = more stable weights
- Helps visualize stability-accuracy trade-off

---

## Console Output

The script produces detailed console output:

### Section 1: Loading Dataset
```
[1/5] Loading dataset...
[OK] Loaded 1000 samples with 10 features
   Cohorts: 5
   Positive rate: 56.80%
```

### Section 2: Weight Estimation
```
[2/5] Estimating weights with each method...

  uLSIF...
    Runtime: 0.004s
    Weight mean: 1.0000
    Weight std: 0.1255
    Weight CV: 0.1255
    Weight range: [0.5718, 1.2412]
```

### Section 3: Stability Analysis
```
[3/5] Analyzing weight stability...

  Coefficient of Variation (CV = Std/Mean, lower is more stable):
  Method               CV         Improvement vs uLSIF
  -------------------------------------------------------
  uLSIF                0.1255     baseline
  RULSIF(alpha=0.0)    0.1255     +0.00%
  RULSIF(alpha=0.1)    0.1254     +0.12%
  RULSIF(alpha=0.5)    0.1244     +0.92%
  RULSIF(alpha=0.9)    0.1179     +6.05%

  Weight Variance Reduction:
    RULSIF(alpha=0.1): +0.23% variance reduction
    RULSIF(alpha=0.5): +1.83% variance reduction
    RULSIF(alpha=0.9): +11.74% variance reduction
```

### Section 4: Bound Estimation
```
[4/5] Estimating PPV bounds for each method...

  uLSIF...
    Bound time: 0.003s
    CERTIFY: 0/25 (0.0%)
    ABSTAIN: 25/25 (100.0%)
    NO-GUARANTEE: 0/25 (0.0%)
```

### Section 5: Decision Comparison
```
[5/5] Comparing certification decisions...

  Decision Agreement with uLSIF:
    RULSIF(alpha=0.0): 25/25 (100.0%) agreement
    RULSIF(alpha=0.1): 25/25 (100.0%) agreement
    RULSIF(alpha=0.5): 25/25 (100.0%) agreement
    RULSIF(alpha=0.9): 25/25 (100.0%) agreement
```

### Summary Statistics
```
================================================================================
 SUMMARY STATISTICS
================================================================================

Method               Weight CV    Runtime (s)  Certify Rate
-----------------------------------------------------------
uLSIF                0.1255       0.007        0.0%
RULSIF(alpha=0.1)    0.1254       0.005        0.0%
RULSIF(alpha=0.5)    0.1244       0.004        0.0%
RULSIF(alpha=0.9)    0.1179       0.004        0.0%
```

### Key Findings
```
================================================================================
 KEY FINDINGS
================================================================================

1. Weight Stability:
   - RULSIF(alpha=0.1) provides +0.12% stability improvement over uLSIF
   - RULSIF(alpha=0.5) provides +0.92% stability improvement over uLSIF
   - Higher alpha -> more stable weights (lower CV)

2. Certification Performance:
   - uLSIF: 0 certifications
   - RULSIF(alpha=0.1): 0 certifications
   - RULSIF(alpha=0.5): 0 certifications

3. Recommendation:
   - On this dataset, shift is moderate
   - uLSIF and RULSIF perform similarly
   - Use uLSIF for simplicity (fewer hyperparameters)
   - Consider RULSIF for datasets with more severe shifts
```

---

## Interpreting Results

### Good Signs:
✅ **RULSIF(alpha=0.0) matches uLSIF**: Mean difference < 0.001
✅ **Higher alpha → lower CV**: Stability improvement
✅ **High agreement rate**: Methods agree on decisions
✅ **Similar certification rates**: Both methods effective

### Warning Signs:
⚠️ **No improvement with higher alpha**: Shift may be too mild
⚠️ **Low agreement rate (< 90%)**: Investigate disagreements
⚠️ **RULSIF certifies fewer**: May be too conservative (alpha too high)

### When RULSIF Helps Most:
- **High CV reduction** (> 5%): Significant stability gain
- **Large variance reduction** (> 10%): More stable bounds
- **Extreme weight ranges**: uLSIF has very high max or very low min
- **Severe distribution shifts**: Cross-domain, scaffold splits

---

## Use Cases

### Use Case 1: Quick Comparison
```bash
# Just want to see if RULSIF helps
python scripts/compare_ulsif_vs_rulsif.py --dataset test_dataset

# Look at console output:
# - CV improvement > 5%? → RULSIF helps!
# - CV improvement < 2%? → Stick with uLSIF
```

### Use Case 2: Selecting Optimal Alpha
```bash
# Compare multiple alpha values
python scripts/compare_ulsif_vs_rulsif.py --dataset bace

# Check CV vs Alpha plot:
# - Find alpha with best stability-accuracy balance
# - Usually alpha=0.1 or 0.3 for practical use
```

### Use Case 3: Understanding Disagreements
```bash
# Run comparison
python scripts/compare_ulsif_vs_rulsif.py --dataset dataset_name

# Analyze decision CSVs:
import pandas as pd
ulsif = pd.read_csv("results/ulsif_dataset_name_results.csv")
rulsif = pd.read_csv("results/rulsifalpha=01_dataset_name_results.csv")

# Find disagreements
merged = pd.merge(ulsif, rulsif, on=['cohort_id', 'tau'], suffixes=('_u', '_r'))
disagree = merged[merged['decision_u'] != merged['decision_r']]
print(disagree[['cohort_id', 'tau', 'decision_u', 'decision_r', 'lower_bound_u', 'lower_bound_r']])
```

### Use Case 4: Batch Testing
```bash
# Test on multiple datasets
for dataset in test_dataset bace bbbp clintox sider
do
    echo "Testing $dataset..."
    python scripts/compare_ulsif_vs_rulsif.py --dataset $dataset
done

# Aggregate results
python -c "
import pandas as pd
from pathlib import Path

results_dir = Path('results')
summaries = []

for f in results_dir.glob('ulsif_vs_rulsif_*_summary.csv'):
    df = pd.read_csv(f)
    dataset = f.stem.replace('ulsif_vs_rulsif_', '').replace('_summary', '')
    df['dataset'] = dataset
    summaries.append(df)

all_results = pd.concat(summaries, ignore_index=True)
all_results.to_csv('results/all_comparisons.csv', index=False)
print(all_results.groupby(['dataset', 'method'])['weight_cv'].mean())
"
```

---

## Troubleshooting

### Issue: "Dataset not found"
**Cause**: Dataset not preprocessed

**Solution**:
```bash
# For molecular datasets
python scripts/preprocess_molecular.py --dataset bace

# For tabular datasets
python scripts/preprocess_tabular.py --dataset adult

# For test dataset
python scripts/create_test_data.py
```

### Issue: "No improvement with RULSIF"
**Cause**: Shift is mild, uLSIF already stable

**Solution**: This is expected! Use uLSIF for simplicity.

### Issue: "Cannot encode character in output"
**Cause**: Greek alpha character (α) not supported by console

**Solution**: Already fixed in code - uses ASCII "alpha" instead

### Issue: "Plots not generated"
**Cause**: matplotlib backend issue or insufficient permissions

**Solution**:
```bash
# Check matplotlib
python -c "import matplotlib; print(matplotlib.get_backend())"

# If needed, set backend
export MPLBACKEND=Agg  # Unix/Mac
set MPLBACKEND=Agg     # Windows
```

---

## Advanced Usage

### Custom Alpha Values

Modify script to test different alphas:

```python
# In compare_ulsif_vs_rulsif.py, line ~72
alphas = [0.0, 0.2, 0.4, 0.6, 0.8]  # Custom values

# Or run multiple times
for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
    rulsif = create_rulsif_baseline(alpha=alpha)
    weights = rulsif.estimate_weights(X_cal, X_test)
    print(f"Alpha={alpha}, CV={weights.std()/weights.mean():.4f}")
```

### Programmatic Usage

```python
from scripts.compare_ulsif_vs_rulsif import compare_ulsif_vs_rulsif

# Run comparison programmatically
results = compare_ulsif_vs_rulsif("test_dataset")

# Access results
for name, res in results.items():
    print(f"{name}: CV={res['weights'].std()/res['weights'].mean():.4f}")
```

---

## Performance

### Runtime (test_dataset, 200 calibration samples):
- **Weight estimation**: 2-30ms per method
- **Bound estimation**: 2-3ms per method
- **Total per method**: ~5-35ms
- **Total script**: ~200ms (including visualization)

### Scalability:
- **Small datasets** (< 500 samples): Instant (< 0.1s)
- **Medium datasets** (500-5000 samples): Fast (< 2s)
- **Large datasets** (> 5000 samples): Moderate (< 30s)

*Note: n_basis=100 default works well for all sizes*

---

## Related Scripts

- **`test_rulsif.py`**: Basic RULSIF testing on synthetic data
- **`test_rulsif_on_bace.py`**: RULSIF testing on real molecular data
- **`compare_kliep_ulsif.py`**: KLIEP vs uLSIF comparison
- **`compare_kmm_ulsif_kliep.py`**: Three-way method comparison

---

## FAQ

**Q: Why compare multiple alpha values?**

A: To understand the stability-accuracy trade-off and find optimal alpha for your dataset.

**Q: Should I always use the alpha with lowest CV?**

A: Not necessarily. Very high alpha (> 0.7) may introduce too much bias. Balance stability and accuracy.

**Q: What if RULSIF certifies fewer cohorts than uLSIF?**

A: This can happen if alpha is too high, making weights too conservative. Try lower alpha (0.1-0.3).

**Q: Can I use this script with my own dataset?**

A: Yes! Just preprocess your dataset into ShiftBench format and pass the dataset name.

**Q: How do I interpret "agreement rate"?**

A: Percentage of (cohort, tau) pairs where both methods make the same decision. 100% = perfect agreement, < 90% = investigate differences.

---

## Citation

If you use this comparison tool in your research, please cite:

```bibtex
@misc{shiftbench2026rulsif,
  title={RULSIF Implementation and Comparison Tool for ShiftBench},
  author={ShiftBench Contributors},
  year={2026},
  url={https://github.com/anthropics/shift-bench}
}
```

---

## Support

- **Documentation**: See `docs/RULSIF_IMPLEMENTATION_REPORT.md` and `docs/RULSIF_QUICK_START.md`
- **Issues**: Report bugs in ShiftBench repository
- **Questions**: Check FAQ above or review console output for diagnostic info

---

**Quick Reference**:

```bash
# Run comparison
python scripts/compare_ulsif_vs_rulsif.py --dataset test_dataset

# Check results
cat results/ulsif_vs_rulsif_test_dataset_summary.csv

# View plots
open results/ulsif_vs_rulsif_test_dataset_weights.png
open results/ulsif_vs_rulsif_test_dataset_cv_vs_alpha.png
```

**Status**: ✅ Production Ready
**Version**: 1.0.0
**Date**: 2026-02-16
