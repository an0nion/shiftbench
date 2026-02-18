# ShiftBench Evaluation Harness - Quick Start Guide

## Installation

The evaluation harness is already installed at:
```
c:\Users\ananya.salian\Downloads\shift-bench\src\shiftbench\evaluate.py
```

No additional installation required beyond existing ShiftBench dependencies.

## Basic Usage

### 1. List Available Datasets

```bash
cd c:\Users\ananya.salian\Downloads\shift-bench\src
python -m shiftbench.evaluate --method ulsif --dataset list
```

Output:
```
Available datasets:
  - test_dataset         (synthetic ,   1000 samples)
  - bace                 (molecular ,   1513 samples)
  - bbbp                 (molecular ,   1975 samples)
  - clintox              (molecular ,   1458 samples)
  ...
```

### 2. Run Single Evaluation

```bash
# Evaluate uLSIF on test dataset
python -m shiftbench.evaluate --method ulsif --dataset test_dataset --output ../results/

# Evaluate uLSIF on BACE dataset
python -m shiftbench.evaluate --method ulsif --dataset bace --output ../results/
```

### 3. View Results

Results are saved to CSV files in the output directory:

```bash
# View main results
cat ../results/ulsif_bace_results.csv

# View aggregated summary
cat ../results/aggregated_summary.csv

# View metadata
cat ../results/all_metadata.csv
```

## Advanced Usage

### Custom Tau Grid

```bash
python -m shiftbench.evaluate \
  --method ulsif \
  --dataset bace \
  --tau 0.5,0.7,0.9 \
  --output ../results/
```

### Custom Alpha Level

```bash
python -m shiftbench.evaluate \
  --method ulsif \
  --dataset bace \
  --alpha 0.1 \
  --output ../results/
```

### Verbose Logging

```bash
python -m shiftbench.evaluate \
  --method ulsif \
  --dataset bace \
  --verbose \
  --output ../results/
```

## Batch Processing (Python API)

For evaluating multiple datasets or methods, use the Python API:

```python
import sys
sys.path.insert(0, 'c:/Users/ananya.salian/Downloads/shift-bench/src')

from shiftbench.evaluate import evaluate_batch
from pathlib import Path

# Evaluate uLSIF on multiple datasets
results, metadata = evaluate_batch(
    dataset_names=['test_dataset', 'bace', 'bbbp'],
    method_names=['ulsif'],
    output_dir=Path('../results/batch_run'),
    continue_on_error=True
)

print(f"Total decisions: {len(results)}")
print(f"Successful runs: {len(metadata)}")
```

## Output Files

Each evaluation creates the following files:

1. **`{method}_{dataset}_results.csv`** - Per-cohort results
   - Columns: dataset, method, cohort_id, tau, decision, mu_hat, var_hat, n_eff, lower_bound, p_value, elapsed_sec

2. **`all_results.csv`** - Combined results from all runs
   - Same schema as individual results

3. **`all_metadata.csv`** - Run metadata
   - Columns: dataset, method, n_samples_total, n_calibration, n_test, n_features, n_cohorts, n_decisions, n_certify, n_abstain, n_no_guarantee, weight_elapsed_sec, bound_elapsed_sec, total_elapsed_sec, alpha, use_oracle, diagnostics...

4. **`aggregated_summary.csv`** - Summary statistics by (method, dataset, tau)
   - Columns: method, dataset, tau, n_cohorts, n_certify, n_abstain, n_no_guarantee, mean_mu_hat, mean_lower_bound, mean_n_eff, elapsed_sec, cert_rate

## Understanding Results

### Decision Types

- **CERTIFY**: Lower bound ≥ tau (empirical evidence supports PPV ≥ tau at level alpha)
- **ABSTAIN**: Lower bound < tau (insufficient evidence)
- **NO-GUARANTEE**: Method diagnostics failed (only for methods with stability gating)

### Key Metrics

- **cert_rate**: Fraction of cohorts certified at each tau
- **mean_mu_hat**: Average point estimate of PPV
- **mean_lower_bound**: Average 95% lower confidence bound
- **mean_n_eff**: Average effective sample size (after weighting)

## Example Output

### Console Output
```
================================================================================
SUMMARY: Certification Rates by (Method, Dataset, Tau)
================================================================================
method dataset  tau  n_cohorts  n_certify  cert_rate
 ulsif    bace 0.50        127          1   0.007874
 ulsif    bace 0.60        127          1   0.007874
 ulsif    bace 0.70        127          0   0.000000
 ulsif    bace 0.80        127          0   0.000000
 ulsif    bace 0.85        127          0   0.000000
 ulsif    bace 0.90        127          0   0.000000
================================================================================
```

### CSV Output (aggregated_summary.csv)
```csv
method,dataset,tau,n_cohorts,n_certify,n_abstain,n_no_guarantee,mean_mu_hat,mean_lower_bound,mean_n_eff,elapsed_sec,cert_rate
ulsif,bace,0.5,127,1,126,0,1.0,0.113,10.09,0.058,0.0079
ulsif,bace,0.6,127,1,126,0,1.0,0.113,10.09,0.058,0.0079
```

## Troubleshooting

### ModuleNotFoundError: No module named 'shiftbench'

**Solution**: Always run from the `src/` directory:
```bash
cd c:\Users\ananya.salian\Downloads\shift-bench\src
python -m shiftbench.evaluate ...
```

### FileNotFoundError: Dataset not found

**Solution**: Preprocess the dataset first:
```bash
cd c:\Users\ananya.salian\Downloads\shift-bench
python scripts/preprocess_molecular.py --dataset bace
```

### ImportError: RAVEL not found

**Solution**: RAVEL is optional. Use uLSIF or install RAVEL:
```bash
pip install -e /path/to/ravel
```

## Available Methods

Currently supported:
- **ulsif**: Unconstrained Least-Squares Importance Fitting
- **ravel**: Receipt-Anchored Verifiable Evaluation Ledger (if installed)

To add a new method, see the main documentation.

## Available Datasets

Processed and ready to use:
- `test_dataset` - Synthetic data for testing (1000 samples)
- `bace` - BACE inhibition (1513 samples, 217 features)
- `bbbp` - Blood-brain barrier penetration (1975 samples)
- `clintox` - Clinical trial toxicity (1458 samples)
- `esol` - Aqueous solubility (1117 samples)
- `freesolv` - Hydration free energy (642 samples)
- `lipophilicity` - Octanol/water distribution (4200 samples)

## Performance

Typical runtimes:
- test_dataset: ~20ms
- bace: ~60ms
- bbbp: ~55ms
- Larger datasets: 100-500ms

## Next Steps

1. Run on test_dataset to verify setup
2. Run on bace to evaluate on real molecular data
3. Examine results in CSV files
4. Use batch API for systematic evaluation
5. Add new methods or datasets as needed

## Full Documentation

For complete documentation, see:
- `EVALUATION_HARNESS_SUMMARY.md` - Comprehensive implementation details
- `src/shiftbench/evaluate.py` - Source code with docstrings
- `src/shiftbench/baselines/base.py` - Method interface

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the full documentation
3. Examine the source code
4. Test with synthetic data first (test_dataset)
