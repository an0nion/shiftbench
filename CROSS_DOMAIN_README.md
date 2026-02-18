# Cross-Domain Benchmark - Quick Start Guide

This guide explains how to run the comprehensive cross-domain evaluation to compare importance weighting methods across molecular, text, and tabular domains.

---

## Quick Start

### 1. Run Full Benchmark (Recommended)

Evaluate all methods on all domains:

```bash
cd shift-bench/

# Full benchmark (may take 1-2 hours)
python scripts/run_cross_domain_benchmark.py \
    --methods ulsif,kliep,kmm,rulsif,ravel,weighted_conformal \
    --domains molecular,text,tabular \
    --output results/cross_domain/ \
    --tau 0.5,0.6,0.7,0.8,0.85,0.9 \
    --alpha 0.05
```

**Expected Output**:
- `cross_domain_raw_results.csv` - All decisions (dataset, method, cohort, tau, decision, etc.)
- `cross_domain_summary.csv` - Aggregated by domain and method
- `cross_domain_by_dataset.csv` - Per-dataset breakdown
- `cross_domain_by_method.csv` - Per-method breakdown
- `cross_domain_statistical_analysis.csv` - Statistical tests
- `cross_domain_difficulty.csv` - Domain difficulty ranking
- `benchmark.log` - Detailed execution log

---

### 2. Generate Visualizations

Create publication-quality figures:

```bash
# Generate plots
python scripts/plot_cross_domain.py \
    --input results/cross_domain/ \
    --output results/cross_domain/plots/ \
    --format pdf
```

**Figures Generated**:
1. `cert_rate_by_domain.pdf` - Bar chart of certification rates
2. `method_ranking_heatmap.pdf` - Heatmap showing method × domain interaction
3. `runtime_by_domain.pdf` - Runtime comparison
4. `decision_distribution.pdf` - Stacked bar of CERTIFY/ABSTAIN/NO-GUARANTEE
5. `domain_difficulty.pdf` - Domain difficulty ranking
6. `effective_sample_size.pdf` - ESS distribution by domain
7. `method_comparison_scatter.pdf` - Point estimate vs lower bound

---

### 3. Quick Demo (5-10 minutes)

Test on subset of datasets:

```bash
# Run demo
bash DEMO_CROSS_DOMAIN.sh
```

This runs on:
- Molecular: BACE, BBBP
- Text: IMDB, Yelp
- Tabular: Adult, Bank

---

## Available Options

### Methods

Choose which methods to evaluate:

```bash
--methods ulsif,kliep,kmm,rulsif,ravel,weighted_conformal
```

**Available Methods**:
- `ulsif` - Unconstrained Least-Squares Importance Fitting (fastest)
- `kliep` - KL Importance Estimation Procedure
- `kmm` - Kernel Mean Matching
- `rulsif` - Relative uLSIF
- `ravel` - RAVEL with stability gates (requires RAVEL package)
- `weighted_conformal` - Weighted split conformal prediction

**Recommendation**: Start with `ulsif,kliep` for quick tests, then add others.

---

### Domains

Choose which domains to include:

```bash
--domains molecular,text,tabular
```

**Domain Details**:

**Molecular** (6-9 datasets):
- BACE, BBBP, ClinTox, ESOL, FreeSolv, Lipophilicity, SIDER, Tox21, ToxCast
- Shift type: Scaffold-based covariate shift
- Expected difficulty: Hard (many cohorts, high-D features)

**Text** (5 datasets):
- IMDB, Yelp, Amazon, Civil Comments, Twitter
- Shift type: Temporal, geographic, demographic
- Expected difficulty: Easy (few cohorts, large N)

**Tabular** (6 datasets):
- Adult, COMPAS, Bank, German Credit, Heart Disease, Diabetes
- Shift type: Demographic, temporal
- Expected difficulty: Variable (depends on dataset)

---

### Tau Grid

Specify PPV thresholds to test:

```bash
--tau 0.5,0.6,0.7,0.8,0.85,0.9
```

**Interpretation**:
- `tau=0.5` - At least 50% precision required (lenient)
- `tau=0.8` - At least 80% precision required (moderate)
- `tau=0.9` - At least 90% precision required (strict)

**Default**: `0.5, 0.6, 0.7, 0.8, 0.85, 0.9` (6 thresholds)

---

### Alpha (FWER)

Set family-wise error rate:

```bash
--alpha 0.05
```

**Default**: 0.05 (5% FWER across all cohorts)

---

## Output Files

### 1. Raw Results (`cross_domain_raw_results.csv`)

**Columns**:
- `dataset` - Dataset name (e.g., "bace", "imdb")
- `domain` - Domain name (molecular, text, tabular)
- `method` - Method name (e.g., "ulsif")
- `cohort_id` - Cohort identifier
- `tau` - PPV threshold tested
- `decision` - CERTIFY, ABSTAIN, or NO-GUARANTEE
- `mu_hat` - Point estimate of PPV
- `lower_bound` - 95% lower confidence bound
- `p_value` - One-sided p-value
- `n_eff` - Effective sample size
- `elapsed_sec` - Runtime in seconds
- `diagnostics` - JSON string with method-specific diagnostics

**Size**: ~10K-100K rows (depends on datasets × methods × cohorts × tau values)

---

### 2. Summary Tables

#### `cross_domain_summary.csv`
Aggregated by (domain, method):
- `cert_rate_%` - Certification rate (%)
- `mean_mu_hat` - Average point estimate
- `mean_lower_bound` - Average lower bound
- `mean_n_eff` - Average effective sample size
- `mean_runtime_sec` - Average runtime per dataset
- `n_datasets` - Number of datasets
- `n_cohorts` - Total number of cohorts

#### `cross_domain_by_dataset.csv`
Per-dataset breakdown (dataset, domain, method):
- `cert_rate_%` - Certification rate for this dataset
- `total_runtime_sec` - Total runtime
- `n_cohorts` - Number of cohorts

#### `cross_domain_by_method.csv`
Per-method breakdown (method, domain):
- `cert_rate_%` - Certification rate
- `n_datasets` - Number of datasets evaluated

---

### 3. Statistical Analysis

#### `cross_domain_statistical_analysis.csv`
- **Test 1**: Kruskal-Wallis test (do domains differ?)
- **Test 2**: Chi-square test (method × domain interaction?)
- **Test 3**: Domain difficulty ranking

**Interpretation**:
- `p_value < 0.05` → Significant difference
- `significant = True` → Reject null hypothesis

#### `cross_domain_difficulty.csv`
Domains ranked by mean certification rate:
- `cert_rate` - Mean certification rate (0-1)
- `std` - Standard deviation
- `n_decisions` - Total number of decisions

**Lower cert rate = harder domain**

---

## Example Workflows

### Workflow 1: Molecular-Only Benchmark

Focus on molecular domain to compare methods:

```bash
python scripts/run_cross_domain_benchmark.py \
    --methods ulsif,kliep,ravel \
    --domains molecular \
    --output results/molecular_only/ \
    --tau 0.5,0.7,0.9
```

---

### Workflow 2: Quick RAVEL vs Baselines

Compare RAVEL to simple baselines:

```bash
python scripts/run_cross_domain_benchmark.py \
    --methods ulsif,ravel \
    --domains molecular,text \
    --output results/ravel_comparison/
```

---

### Workflow 3: Text Domain Deep Dive

Evaluate all methods on text datasets:

```bash
# Run benchmark
python scripts/run_cross_domain_benchmark.py \
    --methods ulsif,kliep,kmm,rulsif,ravel \
    --domains text \
    --output results/text_deep_dive/

# Generate plots
python scripts/plot_cross_domain.py \
    --input results/text_deep_dive/ \
    --format pdf
```

---

### Workflow 4: Full Benchmark for Paper

Complete evaluation for publication:

```bash
# Full benchmark
python scripts/run_cross_domain_benchmark.py \
    --methods ulsif,kliep,kmm,rulsif,ravel,weighted_conformal \
    --domains molecular,text,tabular \
    --output results/full_benchmark/ \
    --tau 0.5,0.6,0.7,0.8,0.85,0.9 \
    --alpha 0.05

# Generate plots (PDF for paper)
python scripts/plot_cross_domain.py \
    --input results/full_benchmark/ \
    --format pdf

# View results
cat results/full_benchmark/cross_domain_summary.csv
cat results/full_benchmark/cross_domain_statistical_analysis.csv
```

---

## Interpreting Results

### Key Questions to Ask

1. **Which domain is hardest?**
   - Look at `cross_domain_difficulty.csv`
   - Lower cert rate = harder

2. **Does method ranking change by domain?**
   - Check p-value in `cross_domain_statistical_analysis.csv` (Test 2)
   - If p < 0.05, method ranking varies by domain
   - Look at `method_ranking_heatmap.pdf` for visual

3. **Does RAVEL's gating help more for molecular vs text?**
   - Compare RAVEL certification rate across domains in `cross_domain_summary.csv`
   - Check abstention rates in `decision_distribution.pdf`
   - If RAVEL abstains more in molecular → gating triggered by instability

4. **Which datasets have most shift?**
   - Look at mean `n_eff` (effective sample size) in `cross_domain_by_dataset.csv`
   - Lower ESS = more severe shift

5. **Are text datasets easier?**
   - Compare molecular vs text cert rates in `cross_domain_summary.csv`
   - Look at `cert_rate_by_domain.pdf`

---

### Expected Patterns

Based on our predictions (see `docs/CROSS_DOMAIN_ANALYSIS.md`):

**Domain Difficulty**:
```
Text (easiest) > Tabular > Molecular (hardest)
```

**Method Ranking**:
- **Molecular**: RAVEL > KLIEP > uLSIF (gating helps)
- **Text**: uLSIF ≈ KLIEP > RAVEL (gating unnecessary)
- **Tabular**: Mixed (depends on dataset)

**Shift Severity** (ESS/N):
- **Text**: 0.6-0.9 (mild shift)
- **Tabular**: 0.3-0.6 (moderate shift)
- **Molecular**: 0.2-0.4 (severe shift)

---

## Troubleshooting

### Issue 1: "Dataset not found"

**Solution**: Preprocess dataset first:

```bash
# Molecular
python scripts/preprocess_molecular.py --dataset bace

# Text
python scripts/preprocess_text.py --dataset imdb

# Tabular
python scripts/preprocess_tabular.py --dataset adult
```

---

### Issue 2: "Method not available"

**Solution**: Check available methods:

```python
from shiftbench.baselines import *

# List available
print("Available methods:")
print("- ulsif")
print("- kliep")
print("- kmm")
print("- rulsif")
print("- weighted_conformal")

# RAVEL requires separate package
try:
    from shiftbench.baselines import create_ravel_baseline
    print("- ravel (available)")
except ImportError:
    print("- ravel (not available - install RAVEL package)")
```

---

### Issue 3: Runtime too long

**Solution**: Use subset of datasets or methods:

```bash
# Quick test (5 minutes)
python scripts/run_cross_domain_benchmark.py \
    --methods ulsif,kliep \
    --domains text \
    --output results/quick_test/

# Medium test (20 minutes)
python scripts/run_cross_domain_benchmark.py \
    --methods ulsif,kliep,ravel \
    --domains molecular,text \
    --tau 0.5,0.8,0.9
```

---

### Issue 4: Out of memory

**Solution**: Process domains sequentially:

```bash
# Run one domain at a time
for domain in molecular text tabular; do
    python scripts/run_cross_domain_benchmark.py \
        --methods ulsif,kliep \
        --domains $domain \
        --output results/${domain}_only/
done

# Combine results manually
cat results/molecular_only/cross_domain_raw_results.csv \
    results/text_only/cross_domain_raw_results.csv \
    results/tabular_only/cross_domain_raw_results.csv \
    > results/combined_results.csv
```

---

## Performance Tips

### Parallel Evaluation

The benchmark script processes datasets sequentially. For faster evaluation, run multiple instances:

```bash
# Terminal 1: Molecular
python scripts/run_cross_domain_benchmark.py \
    --domains molecular \
    --output results/molecular/ &

# Terminal 2: Text
python scripts/run_cross_domain_benchmark.py \
    --domains text \
    --output results/text/ &

# Terminal 3: Tabular
python scripts/run_cross_domain_benchmark.py \
    --domains tabular \
    --output results/tabular/ &
```

---

### Subset Evaluation

Test on representative subset first:

```bash
# Fast subset (3 datasets, 2 methods)
python scripts/run_cross_domain_benchmark.py \
    --methods ulsif,kliep \
    --domains molecular,text,tabular \
    --tau 0.5,0.8 \
    --output results/subset/
```

---

## Next Steps

After running the benchmark:

1. **Fill in findings** in `docs/CROSS_DOMAIN_ANALYSIS.md`
2. **Update tables** with actual results (replace `[TBD]`)
3. **Interpret patterns** - do they match predictions?
4. **Generate publication figures** using plotting script
5. **Write discussion** for paper/report

---

## Citation

If you use ShiftBench in your research, please cite:

```bibtex
@software{shiftbench2025,
  title = {ShiftBench: A Benchmark Suite for Distribution Shift Evaluation},
  author = {[Authors]},
  year = {2025},
  url = {https://github.com/anthropics/shift-bench}
}
```

---

## Contact

For questions or issues:
- See `docs/CROSS_DOMAIN_ANALYSIS.md` for analysis details
- See `README.md` for general ShiftBench documentation
- Check logs in `results/cross_domain/benchmark.log`

---

**Last Updated**: 2026-02-16
**Status**: Ready for production use
**Estimated Runtime**: 1-2 hours (full benchmark), 5-10 minutes (demo)
