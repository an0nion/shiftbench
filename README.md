# ShiftBench

**A Benchmark Suite for Distribution Shift Evaluation**

ShiftBench provides a standardized benchmark for evaluating machine learning models under covariate shift, with a focus on reproducibility and honest abstention when estimates are unreliable.

## What ShiftBench Does

ShiftBench answers the question: *"How do different shift-aware evaluation methods compare across diverse datasets with documented covariate shifts?"*

**Key Features**:
- **50+ datasets** across molecular, text, and tabular domains with documented shifts
- **10+ baseline methods** spanning density ratio estimation, conformal prediction, and robust optimization
- **Reproducibility-first**: Hash-chained receipts make every result independently verifiable
- **Honest abstention**: Methods can declare NO-GUARANTEE when weights are unstable
- **Standardized protocol**: Certify-or-abstain framework with FWER control

## Quick Start

### Installation

```bash
# Basic installation
pip install -e .

# With development dependencies
pip install -e ".[dev]"

# With RAVEL support
pip install -e ".[ravel]"
```

### Load a Dataset

```python
from shiftbench.data import load_dataset, get_registry

# List available datasets
registry = get_registry()
print(registry.list_datasets(domain="molecular"))

# Load a dataset
X, y, cohorts, splits = load_dataset("bace")

# Split into calibration and target
cal_mask = (splits["split"] == "cal").values
test_mask = (splits["split"] == "test").values

X_cal, y_cal = X[cal_mask], y[cal_mask]
X_test, y_test = X[test_mask], y[test_mask]
```

### Evaluate a Method

```python
from shiftbench.baselines.ravel import create_ravel_baseline
from shiftbench.baselines.ulsif import create_ulsif_baseline

# Create method instances
ravel = create_ravel_baseline()
ulsif = create_ulsif_baseline()

# Estimate importance weights
weights_ravel = ravel.estimate_weights(X_cal, X_test)
weights_ulsif = ulsif.estimate_weights(X_cal, X_test)

# Get predictions (from your model)
predictions_cal = my_model.predict(X_cal)

# Estimate PPV bounds
tau_grid = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
decisions_ravel = ravel.estimate_bounds(
    y_cal, predictions_cal, cohorts[cal_mask], weights_ravel, tau_grid
)
decisions_ulsif = ulsif.estimate_bounds(
    y_cal, predictions_cal, cohorts[cal_mask], weights_ulsif, tau_grid
)

# Compare results
for d in decisions_ravel:
    print(f"RAVEL: {d.cohort_id} @ τ={d.tau}: {d.decision} (lb={d.lower_bound:.3f})")
for d in decisions_ulsif:
    print(f"uLSIF: {d.cohort_id} @ τ={d.tau}: {d.decision} (lb={d.lower_bound:.3f})")
```

## Current Status (Phase 0)

### Implemented ✅

**Infrastructure**:
- [x] Dataset registry (`data/registry.json`)
- [x] Baseline interface (`BaselineMethod` abstract class)
- [x] Dataset loader (`load_dataset()`)
- [x] RAVEL baseline wrapper
- [x] uLSIF baseline implementation

**Datasets** (11 molecular):
- [x] BACE, BBBP, ClinTox, ESOL, FreeSolv, Lipophilicity
- [x] SIDER, Tox21, ToxCast, MUV, MolHIV

### In Progress 🚧

**Baselines** (Priority 1):
- [ ] KLIEP (KL importance estimation)
- [ ] KMM (kernel mean matching)
- [ ] RULSIF (relative uLSIF)
- [ ] Weighted conformal prediction

**Datasets** (Expansion):
- [ ] Text datasets (AG News, IMDB, Civil Comments)
- [ ] Tabular datasets (Adult, OpenML)

**Infrastructure**:
- [ ] Evaluation harness (`python -m shiftbench.evaluate`)
- [ ] Results aggregation script
- [ ] Receipt generation for all methods

## Baseline Methods

### Tier 1: Density Ratio Estimation
1. **RAVEL** ✅ - Discriminative classifier + stability gates
2. **uLSIF** ✅ - Least-squares fitting (closed-form)
3. **KLIEP** 🚧 - KL minimization
4. **KMM** 🚧 - Kernel mean matching
5. **RULSIF** 🚧 - Relative density ratio

### Tier 2: Conformal Prediction
6. **Weighted Conformal** 🚧 - Tibshirani et al. 2019
7. **Split Conformal** 🚧 - Baseline (no shift adaptation)
8. **CV+** 🚧 - Cross-validation conformal

### Tier 3: Robust Optimization
9. **Group DRO** 🚧 - Sagawa et al. 2020
10. **Chi-Sq DRO** 🚧 - Duchi & Namkoong 2019

## Datasets

### Molecular (11/30 target)
All use scaffold-based splits to create covariate shift:
- **BACE** (1513 samples) - BACE inhibition
- **BBBP** (1975 samples) - Blood-brain barrier penetration
- **ClinTox** (1458 samples) - Clinical trial toxicity
- **ESOL** (1117 samples) - Aqueous solubility
- **FreeSolv** (642 samples) - Hydration free energy
- **Lipophilicity** (4200 samples) - Octanol/water distribution
- **SIDER** (1427 samples) - Side effects
- **Tox21** (7831 samples) - Nuclear receptor toxicity
- **ToxCast** (8576 samples) - High-throughput toxicity
- **MUV** (93087 samples) - Virtual screening
- **MolHIV** (41120 samples) - HIV inhibition

### Text (0/40 target)
🚧 Coming soon

### Tabular (0/30 target)
🚧 Coming soon

## Project Structure

```
shift-bench/
├── data/
│   ├── registry.json          # Dataset metadata
│   └── processed/             # Preprocessed datasets
│       └── <dataset>/
│           ├── features.npy
│           ├── labels.npy
│           ├── cohorts.npy
│           └── splits.csv
├── src/shiftbench/
│   ├── __init__.py
│   ├── data.py                # Dataset loading
│   └── baselines/
│       ├── base.py            # Abstract interface
│       ├── ravel.py           # RAVEL implementation ✅
│       └── ulsif.py           # uLSIF implementation ✅
├── scripts/
│   └── aggregate_results.py  # Results aggregation (TBD)
├── docs/
│   ├── SUBMISSION_GUIDE.md   # How to submit methods (TBD)
│   ├── ADDING_DATASETS.md    # How to add datasets (TBD)
│   └── ADDING_METHODS.md     # How to implement baselines (TBD)
└── README.md                  # This file
```

## Contributing

See [docs/ADDING_METHODS.md](docs/ADDING_METHODS.md) for how to implement a new baseline method.

See [docs/ADDING_DATASETS.md](docs/ADDING_DATASETS.md) for how to contribute datasets.

## Roadmap

**Phase 1** (Weeks 1-4): Foundation + External Baselines
- ✅ Dataset registry and loader
- ✅ Baseline interface
- ✅ RAVEL + uLSIF implementations
- 🚧 KLIEP, KMM, RULSIF implementations
- 🚧 Weighted conformal prediction

**Phase 2** (Weeks 5-8): Infrastructure + Full Benchmark
- 🚧 Evaluation harness
- 🚧 Full benchmark sweep (10 methods × 50 datasets)
- 🚧 Results aggregation
- 🚧 Static leaderboard

**Phase 3** (Weeks 9-12): Documentation + Paper
- 🚧 Community documentation
- 🚧 NeurIPS D&B paper
- 🚧 Reproducibility artifacts

## Citation

```bibtex
@software{shiftbench2025,
  title = {ShiftBench: A Benchmark Suite for Distribution Shift Evaluation},
  author = {[Authors]},
  year = {2025},
  url = {https://github.com/anthropics/shift-bench}
}
```

## License

MIT

## Acknowledgments

- MoleculeNet for molecular datasets
- RAVEL project for baseline implementation
- All baseline method authors (see individual method papers)
