# Cross-Domain Evaluation - Quick Reference

**One-page guide for running cross-domain benchmarks**

---

## 🚀 Quick Commands

### Full Benchmark (1-2 hours)
```bash
cd shift-bench/

# Run benchmark
python scripts/run_cross_domain_benchmark.py \
    --methods ulsif,kliep,ravel \
    --domains molecular,text,tabular \
    --output results/cross_domain/

# Generate plots
python scripts/plot_cross_domain.py \
    --input results/cross_domain/ \
    --format pdf

# Analyze results
python scripts/analyze_cross_domain_results.py results/cross_domain/
```

### Quick Demo (5 min)
```bash
bash DEMO_CROSS_DOMAIN.sh
```

---

## 📊 Output Files

| File | Description |
|------|-------------|
| `cross_domain_raw_results.csv` | All decisions (10K-100K rows) |
| `cross_domain_summary.csv` | By domain and method |
| `cross_domain_by_dataset.csv` | Per-dataset breakdown |
| `cross_domain_by_method.csv` | Per-method breakdown |
| `cross_domain_statistical_analysis.csv` | Statistical tests |
| `cross_domain_difficulty.csv` | Domain difficulty ranking |
| `plots/cert_rate_by_domain.pdf` | Main figure for paper |
| `plots/method_ranking_heatmap.pdf` | Method × domain interaction |

---

## 🎯 Key Questions Answered

1. **Which domain is hardest?** → Check `cross_domain_difficulty.csv`
2. **Does method ranking change by domain?** → Check statistical analysis (Chi-square test)
3. **Does RAVEL's gating help more for molecular?** → Compare abstention rates
4. **Which datasets have most shift?** → Check ESS values in raw results
5. **Are text datasets easier?** → Compare cert rates across domains

---

## 🔬 Available Methods

| Method | Type | Speed | Abstention? |
|--------|------|-------|-------------|
| `ulsif` | Direct ratio | Fast | No |
| `kliep` | KL minimization | Medium | No |
| `kmm` | Kernel matching | Medium | No |
| `rulsif` | Relative ratio | Fast | No |
| `ravel` | Discriminative + gates | Slow | **Yes** |
| `weighted_conformal` | Coverage-based | Medium | Partial |

---

## 🗂️ Available Datasets

### Molecular (6-9 datasets)
BACE, BBBP, ClinTox, ESOL, FreeSolv, Lipophilicity, SIDER, Tox21, ToxCast

**Expected**: Hardest domain (many cohorts, severe shift)

### Text (5 datasets)
IMDB, Yelp, Amazon, Civil Comments, Twitter

**Expected**: Easiest domain (few cohorts, large N)

### Tabular (6 datasets)
Adult, COMPAS, Bank, German Credit, Heart Disease, Diabetes

**Expected**: Variable difficulty

---

## ⚙️ Common Options

### Subset of Methods
```bash
--methods ulsif,kliep  # Fast baseline comparison
--methods ravel  # RAVEL only
```

### Subset of Domains
```bash
--domains molecular  # Molecular only
--domains text,tabular  # No molecular
```

### Subset of Tau Values
```bash
--tau 0.5,0.8  # Quick test (2 thresholds)
--tau 0.5,0.6,0.7,0.8,0.85,0.9  # Full grid (6 thresholds)
```

### Custom Output Directory
```bash
--output results/my_benchmark/
```

---

## 📈 Expected Results

### Domain Difficulty
```
Text (easiest) > Tabular > Molecular (hardest)
```

### Certification Rates (τ=0.8)
```
Text:      60-90%
Tabular:   20-50%
Molecular:  5-15%
```

### Method Rankings
```
Molecular:  RAVEL > KLIEP > uLSIF  (gating helps)
Text:       uLSIF ≈ KLIEP > RAVEL  (gating unnecessary)
Tabular:    Mixed (dataset-dependent)
```

---

## 🛠️ Troubleshooting

### Dataset not found
```bash
python scripts/preprocess_molecular.py --dataset bace
```

### Method not available
```bash
--methods ulsif,kliep  # Skip unavailable methods
```

### Runtime too long
```bash
--domains text --tau 0.5,0.8  # Subset
```

### Out of memory
```bash
# Run domains separately
python scripts/run_cross_domain_benchmark.py --domains molecular
python scripts/run_cross_domain_benchmark.py --domains text
python scripts/run_cross_domain_benchmark.py --domains tabular
```

---

## 📚 Documentation

- **Quick Start**: `CROSS_DOMAIN_README.md` (700+ lines)
- **Analysis Framework**: `docs/CROSS_DOMAIN_ANALYSIS.md` (840+ lines)
- **Complete Summary**: `CROSS_DOMAIN_COMPLETE.md`
- **This Reference**: `CROSS_DOMAIN_QUICK_REFERENCE.md`

---

## 🎓 Example Workflows

### Workflow 1: Molecular-Only
```bash
python scripts/run_cross_domain_benchmark.py \
    --domains molecular \
    --methods ulsif,kliep,ravel \
    --output results/molecular_only/
```

### Workflow 2: RAVEL vs Baselines
```bash
python scripts/run_cross_domain_benchmark.py \
    --methods ulsif,ravel \
    --domains molecular,text \
    --output results/ravel_comparison/
```

### Workflow 3: Text Deep Dive
```bash
python scripts/run_cross_domain_benchmark.py \
    --methods ulsif,kliep,kmm,rulsif,ravel \
    --domains text \
    --output results/text_deep_dive/
```

### Workflow 4: Paper Results
```bash
# Full benchmark
python scripts/run_cross_domain_benchmark.py \
    --methods ulsif,kliep,kmm,rulsif,ravel,weighted_conformal \
    --domains molecular,text,tabular \
    --output results/paper/

# Generate plots
python scripts/plot_cross_domain.py --input results/paper/ --format pdf

# Analyze
python scripts/analyze_cross_domain_results.py results/paper/
```

---

## 📊 Reading Results

### View Summary
```bash
cat results/cross_domain/cross_domain_summary.csv
```

### View Statistical Tests
```bash
cat results/cross_domain/cross_domain_statistical_analysis.csv
```

### View Domain Difficulty
```bash
cat results/cross_domain/cross_domain_difficulty.csv
```

### View Runtime
```bash
cat results/cross_domain/cross_domain_runtime.csv
```

---

## 🎨 Visualizations

All figures saved to `results/cross_domain/plots/`:

1. `cert_rate_by_domain.pdf` ← **Use in paper**
2. `method_ranking_heatmap.pdf` ← **Use in paper**
3. `runtime_by_domain.pdf`
4. `decision_distribution.pdf`
5. `domain_difficulty.pdf`
6. `effective_sample_size.pdf`
7. `method_comparison_scatter.pdf`

---

## ✅ Validation Checklist

Before submitting results:

- [ ] Run full benchmark on all domains
- [ ] Generate all plots
- [ ] Run analysis script
- [ ] Check for failed datasets (in log)
- [ ] Verify statistical tests (p-values)
- [ ] Compare to predictions in `docs/CROSS_DOMAIN_ANALYSIS.md`
- [ ] Update findings in analysis document
- [ ] Check plot quality (300 DPI for paper)
- [ ] Archive results and logs

---

## 🚨 Common Pitfalls

1. **Not preprocessing datasets first** → Run preprocessing scripts
2. **Using too many tau values** → Increases runtime linearly
3. **Forgetting to generate plots** → Run plotting script after benchmark
4. **Ignoring statistical tests** → Check significance in analysis
5. **Not documenting findings** → Update `docs/CROSS_DOMAIN_ANALYSIS.md`

---

## 📞 Support

- **Quick questions**: Check `CROSS_DOMAIN_README.md`
- **Analysis help**: See `docs/CROSS_DOMAIN_ANALYSIS.md`
- **Troubleshooting**: See "Troubleshooting" section above
- **Bugs**: Check `results/cross_domain/benchmark.log`

---

**Last Updated**: 2026-02-16
**Status**: Ready for production use
**Estimated Time**: 1-2 hours (full), 5-10 min (demo)
