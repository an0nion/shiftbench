# Tabular Datasets - Quick Start Guide

## 🚀 Quick Commands

### Preprocess All Tabular Datasets
```bash
cd shift-bench
python scripts/preprocess_tabular.py --all
```

### Preprocess Single Dataset
```bash
python scripts/preprocess_tabular.py --dataset adult
python scripts/preprocess_tabular.py --dataset bank
python scripts/preprocess_tabular.py --dataset german_credit
```

### Run Evaluation
```bash
# Single dataset
python -m shiftbench.evaluate --method ulsif --dataset adult --output results/

# Multiple datasets
python -m shiftbench.evaluate --method ulsif \
    --dataset adult,bank,german_credit \
    --output results/tabular/

# All tabular datasets
python -m shiftbench.evaluate --method all \
    --dataset adult,compas,bank,german_credit,diabetes,heart_disease \
    --output results/tabular_full/
```

---

## 📊 Dataset Overview

| Dataset | Size | Features | Cohorts | Use Case |
|---------|------|----------|---------|----------|
| **Adult** | 48K | 113 | 50 | Income prediction (fairness) |
| **COMPAS** | 6K | 48K* | 44 | Recidivism (justice) |
| **Bank** | 41K | 63 | 10 | Marketing (temporal) |
| **German Credit** | 1K | 65 | 16 | Credit risk (fairness) |
| **Diabetes** | 768 | 12 | 4 | Medical diagnosis |
| **Heart Disease** | 303 | 17 | 8 | Medical diagnosis |

*COMPAS has 48,716 features due to one-hot encoding explosion

---

## 🎯 Recommended Test Sequence

### 1. Start Small (Fast Testing)
```bash
# Diabetes: smallest, 4 cohorts
python -m shiftbench.evaluate --method ulsif --dataset diabetes --output results/test/

# Heart Disease: small, 8 cohorts
python -m shiftbench.evaluate --method ulsif --dataset heart_disease --output results/test/
```

### 2. Medium Complexity
```bash
# Bank: temporal shift, 10 cohorts, high cert rate
python -m shiftbench.evaluate --method ulsif --dataset bank --output results/test/

# German Credit: 16 cohorts
python -m shiftbench.evaluate --method ulsif --dataset german_credit --output results/test/
```

### 3. Large Scale
```bash
# Adult: 50 cohorts, demographic shift
python -m shiftbench.evaluate --method ulsif --dataset adult --output results/test/

# COMPAS: 44 cohorts, high-dim (WARNING: 2.3 GB features)
python -m shiftbench.evaluate --method ulsif --dataset compas --output results/test/
```

---

## ⚡ Expected Performance

### Evaluation Times (uLSIF)
- Diabetes: ~0.3s
- Heart Disease: ~0.4s
- German Credit: ~0.5s
- Bank Marketing: ~0.6s
- Adult: ~0.9s
- COMPAS: ~2-5s (high-dim)

### Certification Rates (τ=0.7)
- **Bank**: 90% (9/10 cohorts) - Best performance
- **Diabetes**: ~75% (3/4 cohorts)
- **German Credit**: ~60% (10/16 cohorts)
- **Adult**: ~18% (9/49 cohorts) - Fine-grained cohorts
- **Heart Disease**: ~50% (4/8 cohorts)
- **COMPAS**: TBD (high-dim, may need tuning)

---

## 📁 Output Structure

After preprocessing:
```
data/
├── processed/
│   ├── adult/
│   │   ├── features.npy      # (48842, 113)
│   │   ├── labels.npy         # (48842,)
│   │   ├── cohorts.npy        # (48842,)
│   │   ├── splits.csv         # uid, split
│   │   └── metadata.json
│   ├── bank/
│   │   └── ...
│   └── ...
└── raw_tabular/               # Cached downloads (not in git)
    ├── adult/
    ├── bank/
    └── ...
```

After evaluation:
```
results/
├── ulsif_adult_results.csv    # Per-cohort results
├── ulsif_bank_results.csv
├── all_results.csv             # Combined results
├── all_metadata.csv            # Run metadata
└── aggregated_summary.csv      # Summary by (method, dataset, tau)
```

---

## 🔍 Inspecting Results

### View Summary
```bash
cat results/aggregated_summary.csv
```

### View Detailed Results
```bash
cat results/ulsif_adult_results.csv | head -20
```

### Quick Python Analysis
```python
import pandas as pd

# Load results
df = pd.read_csv('results/all_results.csv')

# Filter by dataset
adult = df[df['dataset'] == 'adult']

# Certification rate by tau
cert_rate = adult.groupby('tau').apply(
    lambda x: (x['decision'] == 'CERTIFY').mean()
)
print(cert_rate)
```

---

## ⚙️ Common Options

### Custom Tau Grid
```bash
python -m shiftbench.evaluate \
    --method ulsif \
    --dataset adult \
    --tau 0.5,0.6,0.7,0.8,0.9 \
    --output results/
```

### Verbose Output
```bash
python -m shiftbench.evaluate \
    --method ulsif \
    --dataset adult \
    --verbose \
    --output results/
```

### Fail Fast (Stop on Error)
```bash
python -m shiftbench.evaluate \
    --method ulsif \
    --dataset adult,bank,compas \
    --fail-fast \
    --output results/
```

---

## 🐛 Troubleshooting

### Issue: Dataset Not Found
```
FileNotFoundError: Dataset directory not found
```
**Solution**: Run preprocessing first
```bash
python scripts/preprocess_tabular.py --dataset adult
```

### Issue: Module Not Found
```
ModuleNotFoundError: No module named 'shiftbench'
```
**Solution**: Install package
```bash
pip install -e .
```

### Issue: COMPAS Takes Too Long
**Problem**: 48,716 features, 2.3 GB
**Solution**: Reduce n_basis or use dimensionality reduction
```python
# Edit evaluate.py AVAILABLE_METHODS
"ulsif": {
    "default_params": {
        "n_basis": 50,  # Reduced from 100
        ...
    }
}
```

### Issue: Low Certification Rates
**Cause**: Fine-grained cohorts (e.g., Adult: 50 cohorts)
**Solution**: This is expected! See analysis in `TABULAR_DATASETS_COMPLETE.md`

---

## 📚 Documentation

- **Detailed Guide**: `docs/tabular_datasets_summary.md`
- **Complete Report**: `TABULAR_DATASETS_COMPLETE.md`
- **This Guide**: `TABULAR_QUICK_START.md`

---

## 🎓 Example Workflow

### Complete Evaluation Pipeline
```bash
# 1. Preprocess all tabular datasets
python scripts/preprocess_tabular.py --all

# 2. Run evaluation (uLSIF)
python -m shiftbench.evaluate \
    --method ulsif \
    --dataset adult,compas,bank,german_credit,diabetes,heart_disease \
    --output results/tabular_ulsif/

# 3. Run evaluation (KLIEP)
python -m shiftbench.evaluate \
    --method kliep \
    --dataset adult,compas,bank,german_credit,diabetes,heart_disease \
    --output results/tabular_kliep/

# 4. Compare results
python scripts/compare_methods.py \
    results/tabular_ulsif/all_results.csv \
    results/tabular_kliep/all_results.csv
```

---

## 💡 Tips

1. **Start with Bank Marketing**: High certification rates, good for validation
2. **Avoid COMPAS initially**: 48K features, memory intensive
3. **Use --verbose**: Helpful for debugging
4. **Check aggregated_summary.csv**: Quick overview of results
5. **Compare domains**: Tabular vs Molecular performance patterns

---

## 🔗 Quick Links

- Registry: `data/registry.json`
- Preprocessing: `scripts/preprocess_tabular.py`
- Evaluation: `src/shiftbench/evaluate.py`
- Results: `results/`

---

**Need Help?** Check `TABULAR_DATASETS_COMPLETE.md` for detailed analysis and troubleshooting.
