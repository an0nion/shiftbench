# Cross-Domain Analysis: ShiftBench Evaluation Results

**Date**: 2026-02-16
**Benchmark Version**: 1.0
**Status**: Analysis Framework Complete

---

## Executive Summary

This document presents the comprehensive cross-domain analysis of shift-aware evaluation methods on ShiftBench, comparing performance across **molecular**, **text**, and **tabular** domains. The analysis reveals critical domain-specific insights about when importance weighting succeeds, fails, or requires abstention.

### Key Questions Addressed

1. **Does certification rate differ by domain?** (Statistical testing via ANOVA/Kruskal-Wallis)
2. **Does method ranking change by domain?** (Interaction effects via Chi-square)
3. **Which domain is hardest for shift adaptation?** (Mean certification rates)
4. **Does RAVEL's gating help more for molecular vs text?** (Comparative analysis)
5. **Which datasets have the most/least shift?** (ESS, PSIS diagnostics)

---

## Methods Evaluated

We compare the following density ratio estimation methods:

| Method | Type | Key Features | Abstention? |
|--------|------|--------------|-------------|
| **uLSIF** | Direct ratio | Closed-form, Gaussian kernels | No |
| **KLIEP** | Direct ratio | KL minimization, log-linear | No |
| **KMM** | Kernel matching | Max-mean discrepancy | No |
| **RULSIF** | Relative ratio | Numerator-denominator split | No |
| **RAVEL** | Discriminative + gates | PSIS-k, ESS, clip-mass gates | **Yes** |
| **Weighted Conformal** | Coverage-based | Split conformal + weights | Partial |

---

## Datasets Evaluated

### Molecular Domain (6-9 datasets)
**Shift Type**: Scaffold-based covariate shift

| Dataset | Samples | Cohorts | Positive Rate | Expected Difficulty |
|---------|---------|---------|---------------|---------------------|
| BACE | 1,513 | 127 | 52.1% | Hard (many cohorts) |
| BBBP | 1,975 | 127 | 83.7% | Moderate |
| ClinTox | 1,458 | 127 | 20.4% | Hard (low pos rate) |
| ESOL | 1,117 | 127 | - | Moderate (regression) |
| FreeSolv | 642 | 127 | - | Hard (small N) |
| Lipophilicity | 4,200 | 127 | - | Easier (large N) |

**Hypothesis**: Molecular datasets will be the hardest domain due to:
- High-dimensional molecular fingerprints (2048-4096 dims)
- Many small cohorts (127 scaffolds)
- Severe scaffold shift (structural dissimilarity)
- Low positive rates in some tasks

---

### Text Domain (5 datasets)
**Shift Type**: Temporal, geographic, or demographic shift

| Dataset | Samples | Cohorts | Positive Rate | Expected Difficulty |
|---------|---------|---------|---------------|---------------------|
| IMDB | 50,000 | 10 | 50.0% | Easy (balanced, few cohorts) |
| Yelp | 60,000 | 10 | 50.0% | Easy (balanced, few cohorts) |
| Amazon | 30,000 | 3 | 50.0% | Very easy (3 cohorts) |
| Civil Comments | 30,000 | 5 | 8.2% | Moderate (low pos rate) |
| Twitter | 30,000 | 10 | 50.0% | Easy (balanced) |

**Hypothesis**: Text datasets will be easier due to:
- Fewer cohorts (3-10 vs 127)
- Large sample sizes (30K-60K)
- TF-IDF features capture clear shift patterns
- Balanced positive rates (except Civil Comments)

---

### Tabular Domain (6-7 datasets)
**Shift Type**: Demographic or temporal shift

| Dataset | Samples | Cohorts | Positive Rate | Expected Difficulty |
|---------|---------|---------|---------------|---------------------|
| Adult | 48,842 | 50 | 23.9% | Moderate |
| COMPAS | 6,172 | 44 | 45.5% | Moderate |
| Bank | 41,188 | 10 | 11.3% | Hard (low pos rate) |
| German Credit | 1,000 | 16 | 30.0% | Hard (small N) |
| Heart Disease | 303 | 8 | 45.9% | Very hard (tiny N) |
| Diabetes | 768 | 4 | 34.9% | Moderate |

**Hypothesis**: Tabular datasets will vary widely due to:
- Mixed feature types (numeric + categorical)
- Highly variable sample sizes (300-50K)
- Demographic shifts may be subtle or severe
- Class imbalance in some datasets

---

## Expected Cross-Domain Insights

### 1. Domain Difficulty Ranking

**Prediction**:
```
Hardest → Easiest:
Molecular > Tabular > Text
```

**Rationale**:
- **Molecular**: Many cohorts + high-D features + structural discontinuities
- **Tabular**: Variable quality, demographic shifts often subtle
- **Text**: Few cohorts + large N + clear distributional shifts

**Metric**: Mean certification rate at τ=0.8 (higher = easier)

---

### 2. Method × Domain Interactions

**Prediction**: Method ranking will change by domain

#### Expected Winners by Domain

| Domain | Best Method | Reason |
|--------|-------------|--------|
| Molecular | RAVEL | PSIS gating filters unstable scaffolds |
| Text | uLSIF/KLIEP | Shift is smooth, gates unnecessary |
| Tabular | RAVEL | Demographic shifts trigger abstention |

**Test**: Chi-square test for method × domain interaction

---

### 3. RAVEL's Gating: When Does It Help?

**Key Question**: Does RAVEL's abstention mechanism (PSIS-k, ESS, clip-mass gates) improve reliability more in some domains?

**Predictions**:

| Domain | RAVEL Advantage | Mechanism |
|--------|-----------------|-----------|
| Molecular | **High** | Many scaffolds fail ESS/PSIS gates → abstain |
| Text | **Low** | Few cohorts, large N → weights stable → gates pass |
| Tabular | **Medium** | Demographic shifts hit gates inconsistently |

**Metric**: Compare RAVEL certification rate vs baseline methods:
- If RAVEL certs **fewer** but **more reliable** → gates working
- If RAVEL certs **same** → gates not triggered
- If RAVEL certs **zero** → shift too severe (abstention correct)

---

### 4. Shift Severity by Domain

**Metrics**:
1. **ESS/N ratio**: Lower = more severe shift
2. **PSIS k-hat**: Higher = heavier tails (unstable)
3. **Certification rate**: Lower = harder shift

**Predictions**:

| Domain | ESS/N | PSIS k-hat | Cert Rate |
|--------|-------|------------|-----------|
| Molecular | 0.2-0.4 | 0.5-0.7 | 5-15% |
| Text | 0.6-0.9 | 0.1-0.3 | 60-90% |
| Tabular | 0.3-0.6 | 0.3-0.5 | 20-50% |

---

### 5. Dataset-Specific Insights

**Easiest Datasets** (Predicted):
1. **Amazon** (3 cohorts, 30K samples, balanced)
2. **Yelp** (10 cohorts, 60K samples, balanced)
3. **IMDB** (10 cohorts, 50K samples, balanced)

**Hardest Datasets** (Predicted):
1. **ClinTox** (127 cohorts, low pos rate)
2. **Heart Disease** (303 samples, 8 cohorts)
3. **German Credit** (1K samples, 16 cohorts)
4. **BACE** (127 cohorts, molecular shift)

**Metric**: Certification rate at τ=0.8, averaged across methods

---

## Analysis Pipeline

### 1. Statistical Tests

#### Test 1: Domain Difficulty (Kruskal-Wallis)
**Null Hypothesis**: Certification rates are equal across domains
**Alternative**: At least one domain differs significantly

```python
from scipy.stats import kruskal

molecular_certs = df[df.domain == "molecular"]["certified"]
text_certs = df[df.domain == "text"]["certified"]
tabular_certs = df[df.domain == "tabular"]["certified"]

H, p = kruskal(molecular_certs, text_certs, tabular_certs)
```

**Interpretation**:
- p < 0.05 → Domains differ significantly in difficulty
- Report effect size (eta-squared)

---

#### Test 2: Method × Domain Interaction (Chi-Square)
**Null Hypothesis**: Method performance is independent of domain
**Alternative**: Method ranking changes by domain

```python
from scipy.stats import chi2_contingency

contingency = pd.crosstab(df.domain, df.method, values=df.certified, aggfunc="sum")
chi2, p, dof, expected = chi2_contingency(contingency)
```

**Interpretation**:
- p < 0.05 → Method ranking changes by domain
- Examine residuals to find which (method, domain) pairs deviate

---

#### Test 3: Pairwise Domain Comparisons (Mann-Whitney U)
**Post-hoc tests** after significant Kruskal-Wallis:

```python
from scipy.stats import mannwhitneyu

# Bonferroni correction for 3 comparisons: α = 0.05/3 ≈ 0.017
comparisons = [
    ("molecular", "text"),
    ("molecular", "tabular"),
    ("text", "tabular")
]

for d1, d2 in comparisons:
    U, p = mannwhitneyu(df[df.domain == d1]["certified"],
                        df[df.domain == d2]["certified"])
    print(f"{d1} vs {d2}: U={U}, p={p}")
```

---

### 2. Effect Sizes

Beyond p-values, report **practical significance**:

- **Certification rate difference**: Δ = (rate_A - rate_B)
- **Cohen's d**: Standardized mean difference
- **Cliff's delta**: Non-parametric effect size

**Thresholds** (Cohen's d):
- Small: 0.2
- Medium: 0.5
- Large: 0.8

---

### 3. Visualizations

#### Figure 1: Certification Rate by Domain
**Type**: Grouped bar chart
**X-axis**: Domain (molecular, text, tabular)
**Y-axis**: Certification rate (%)
**Bars**: Methods (uLSIF, KLIEP, RAVEL, etc.)

**Key Insight**: Does RAVEL's advantage vary by domain?

---

#### Figure 2: Method Ranking Heatmap
**Type**: Heatmap
**Rows**: Methods
**Columns**: Domains
**Color**: Certification rate (%)

**Key Insight**: Visual confirmation of method × domain interaction

---

#### Figure 3: Runtime by Domain
**Type**: Scatter plot
**X-axis**: Dataset size (samples)
**Y-axis**: Runtime (seconds)
**Color**: Domain
**Shape**: Method

**Key Insight**: Does molecular domain have higher compute cost?

---

#### Figure 4: Decision Distribution
**Type**: Stacked bar chart
**X-axis**: Domain
**Y-axis**: Percentage of decisions
**Stacks**: CERTIFY (green), ABSTAIN (orange), NO-GUARANTEE (red)

**Key Insight**: Which domain triggers most abstentions (RAVEL)?

---

#### Figure 5: Domain Difficulty Ranking
**Type**: Bar chart (sorted)
**X-axis**: Domain (sorted by cert rate)
**Y-axis**: Mean certification rate
**Error bars**: 95% CI

**Key Insight**: Quantify relative difficulty

---

## Key Findings (To Be Filled After Running Benchmark)

### Finding 1: Domain Difficulty Hierarchy

**Result**:
```
[To be filled after running benchmark]

Example:
Text (78.2% certified) > Tabular (34.5%) > Molecular (12.7%)
```

**Interpretation**:
- [Explain why this ordering occurred]
- [Compare to predictions]

---

### Finding 2: RAVEL's Gating Advantage

**Result**:
```
[To be filled]

Example:
RAVEL abstained in 45% of molecular cohorts but only 8% of text cohorts.
Certification rate (when not abstaining): Molecular 85%, Text 92%.
```

**Interpretation**:
- RAVEL's gates are triggered more frequently in [domain]
- This is evidence that [domain] has more severe/unstable shift
- Abstention is **protective**, not a bug

---

### Finding 3: Method Ranking Changes by Domain

**Result**:
```
[To be filled]

Example:
Molecular: RAVEL > KLIEP > uLSIF
Text: uLSIF > KLIEP > RAVEL
Tabular: RAVEL > uLSIF > KLIEP
```

**Interpretation**:
- No single "best" method across all domains
- RAVEL excels when shift is severe (molecular, some tabular)
- Simple methods (uLSIF) sufficient for clean shifts (text)

---

### Finding 4: Shift Severity Indicators

**Result**:
```
[To be filled]

Example ESS/N by domain:
Molecular: 0.28 ± 0.15
Text: 0.82 ± 0.09
Tabular: 0.45 ± 0.22

PSIS k-hat by domain:
Molecular: 0.61 ± 0.18
Text: 0.15 ± 0.08
Tabular: 0.38 ± 0.20
```

**Interpretation**:
- Text has highest ESS (least shift)
- Molecular has highest k-hat (heaviest tails)
- Diagnostics align with certification rates

---

### Finding 5: Hardest/Easiest Datasets

**Hardest** (Lowest cert rate):
```
[To be filled]

Example:
1. ClinTox (2.3% certified) - 127 cohorts, 20% positive rate
2. Heart Disease (4.1%) - 303 samples, 8 cohorts
3. BACE (8.7%) - 127 cohorts, scaffold shift
```

**Easiest** (Highest cert rate):
```
[To be filled]

Example:
1. Amazon (96.4% certified) - 3 cohorts, 30K samples
2. Yelp (91.2%) - 10 cohorts, 60K samples
3. IMDB (84.5%) - 10 cohorts, 50K samples
```

**Pattern**: Few cohorts + large N + balanced labels → high cert rates

---

## Practical Recommendations

### For Practitioners

1. **Molecular domain**: Use RAVEL with gating; expect ~10-20% certification
2. **Text domain**: Simple methods (uLSIF, KLIEP) often sufficient; expect 70-90% certification
3. **Tabular domain**: Test multiple methods; performance varies by dataset
4. **Small datasets** (<1K): Be skeptical of all methods; consider gathering more data
5. **Many cohorts** (>50): Expect many abstentions; prioritize high-information cohorts

---

### For Researchers

1. **Benchmark on multiple domains**: Single-domain results may not generalize
2. **Report abstention rates**: Not just certification rate
3. **Include diagnostics**: ESS, PSIS k-hat, clip-mass
4. **Test statistical significance**: Not just point estimates
5. **Document failure modes**: When does your method abstain/fail?

---

## Future Work

### Dataset Expansion
- [ ] Add 10+ molecular datasets (MolHIV, MUV, etc.)
- [ ] Add 5+ text datasets (sentiment, NLI, QA)
- [ ] Add 5+ tabular datasets (fairness benchmarks)

### Method Expansion
- [ ] Add Group DRO baseline
- [ ] Add Chi-Sq DRO baseline
- [ ] Add adversarial validation baseline

### Analysis Extensions
- [ ] Multi-level modeling (dataset nested in domain)
- [ ] Learning curves (performance vs sample size)
- [ ] Calibration analysis (are bounds valid?)
- [ ] Computational cost analysis (FLOPs, memory)

---

## Reproducibility

### Run Complete Benchmark
```bash
cd shift-bench/

# Full benchmark (all methods, all domains)
python scripts/run_cross_domain_benchmark.py \
    --methods ulsif,kliep,kmm,rulsif,ravel,weighted_conformal \
    --domains molecular,text,tabular \
    --output results/cross_domain/ \
    --tau 0.5,0.6,0.7,0.8,0.85,0.9 \
    --alpha 0.05
```

### Generate Plots
```bash
# Generate all visualizations
python scripts/plot_cross_domain.py \
    --input results/cross_domain/ \
    --output results/cross_domain/plots/ \
    --format pdf
```

### Analyze Results
```bash
# Raw results
cat results/cross_domain/cross_domain_raw_results.csv

# Summary tables
cat results/cross_domain/cross_domain_summary.csv
cat results/cross_domain/cross_domain_by_dataset.csv
cat results/cross_domain/cross_domain_by_method.csv

# Statistical analysis
cat results/cross_domain/cross_domain_statistical_analysis.csv
cat results/cross_domain/cross_domain_difficulty.csv
```

---

## References

### Dataset Sources
- **MoleculeNet**: Wu et al. (2018), *ChemRxiv*
- **IMDB/Yelp**: Sentiment analysis benchmarks
- **Adult/COMPAS**: Fairness ML benchmarks
- **Civil Comments**: Toxicity detection (Jigsaw)

### Method Papers
- **uLSIF**: Kanamori et al. (2009), *JMLR*
- **KLIEP**: Sugiyama et al. (2008), *NIPS*
- **KMM**: Huang et al. (2007), *NIPS*
- **RAVEL**: [This work], *NeurIPS D&B 2025*
- **Weighted Conformal**: Tibshirani et al. (2019), *JRSS-B*

### Statistical Methods
- **Kruskal-Wallis**: Non-parametric ANOVA
- **Chi-Square**: Contingency table independence test
- **Mann-Whitney U**: Non-parametric pairwise comparison

---

## Appendix: Result Tables

### Table 1: Certification Rate by Domain and Method

| Method | Molecular | Text | Tabular | Mean |
|--------|-----------|------|---------|------|
| uLSIF | [TBD] | [TBD] | [TBD] | [TBD] |
| KLIEP | [TBD] | [TBD] | [TBD] | [TBD] |
| KMM | [TBD] | [TBD] | [TBD] | [TBD] |
| RULSIF | [TBD] | [TBD] | [TBD] | [TBD] |
| RAVEL | [TBD] | [TBD] | [TBD] | [TBD] |
| Weighted Conformal | [TBD] | [TBD] | [TBD] | [TBD] |

---

### Table 2: Mean Effective Sample Size by Domain

| Domain | Mean ESS | Median ESS | Std ESS | Mean ESS/N |
|--------|----------|------------|---------|------------|
| Molecular | [TBD] | [TBD] | [TBD] | [TBD] |
| Text | [TBD] | [TBD] | [TBD] | [TBD] |
| Tabular | [TBD] | [TBD] | [TBD] | [TBD] |

---

### Table 3: Runtime by Domain and Method (seconds)

| Method | Molecular | Text | Tabular | Mean |
|--------|-----------|------|---------|------|
| uLSIF | [TBD] | [TBD] | [TBD] | [TBD] |
| KLIEP | [TBD] | [TBD] | [TBD] | [TBD] |
| KMM | [TBD] | [TBD] | [TBD] | [TBD] |
| RULSIF | [TBD] | [TBD] | [TBD] | [TBD] |
| RAVEL | [TBD] | [TBD] | [TBD] | [TBD] |
| Weighted Conformal | [TBD] | [TBD] | [TBD] | [TBD] |

---

## Contact

For questions about this analysis or to contribute additional datasets/methods, please see the ShiftBench documentation.

---

**Last Updated**: 2026-02-16
**Status**: Framework complete, awaiting benchmark results
**Next Step**: Run `scripts/run_cross_domain_benchmark.py` and fill in findings
