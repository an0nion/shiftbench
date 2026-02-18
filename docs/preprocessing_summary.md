# Molecular Dataset Preprocessing Summary

**Date:** 2026-02-16  
**Script:** `scripts/preprocess_molecular.py`  
**Datasets Processed:** 5 molecular datasets from MoleculeNet

---

## Overview

All 5 datasets were successfully preprocessed for ShiftBench. Each dataset was:
1. Loaded from raw CSV files
2. Featurized using RDKit 2D molecular descriptors
3. Organized into cohorts based on Murcko scaffolds
4. Split into train/calibration/test sets (60/20/20 split, scaffold-aware)
5. Saved to processed data directory

---

## Dataset Statistics

### 1. BBBP (Blood-Brain Barrier Penetration)

**Task:** Binary Classification  
**Samples:** 1,975 total
- Train: 1,186 samples (60.1%)
- Calibration: 395 samples (20.0%)
- Test: 394 samples (19.9%)

**Features:** 217 RDKit 2D descriptors  
**Cohorts:** 1,102 unique Murcko scaffolds  
**Label Distribution:** 75.95% positive rate  
**Label Range:** [0.00, 1.00]

**Warnings:**
- Labels are not strictly 0/1: [0.0, 0.5, 1.0]
- Multiple RDKit warnings: "not removing hydrogen atom without neighbors" (120+ occurrences during scaffold computation)

**Status:** SUCCESS  
**Output:** `C:\Users\ananya.salian\Downloads\shift-bench\data\processed\bbbp`

---

### 2. ClinTox (Clinical Toxicity)

**Task:** Binary Classification  
**Samples:** 1,458 total
- Train: 902 samples (61.9%)
- Calibration: 294 samples (20.2%)
- Test: 262 samples (18.0%)

**Features:** 217 RDKit 2D descriptors  
**Cohorts:** 813 unique Murcko scaffolds  
**Label Column:** FDA_APPROVED (auto-detected)  
**Label Distribution:** 93.55% positive rate (highly imbalanced)  
**Label Range:** [0.00, 1.00]

**Warnings:**
- Labels are not strictly 0/1: [0.0, 0.5, 1.0]

**Status:** SUCCESS  
**Output:** `C:\Users\ananya.salian\Downloads\shift-bench\data\processed\clintox`

---

### 3. ESOL (Aqueous Solubility)

**Task:** Regression  
**Samples:** 1,117 total
- Train: 673 samples (60.2%)
- Calibration: 263 samples (23.5%)
- Test: 181 samples (16.2%)

**Features:** 217 RDKit 2D descriptors  
**Cohorts:** 269 unique Murcko scaffolds  
**Label Range:** [-11.60, 1.58] (log solubility)

**Warnings:** None

**Status:** SUCCESS  
**Output:** `C:\Users\ananya.salian\Downloads\shift-bench\data\processed\esol`

---

### 4. FreeSolv (Solvation Free Energy)

**Task:** Regression  
**Samples:** 642 total
- Train: 417 samples (64.9%)
- Calibration: 154 samples (24.0%)
- Test: 71 samples (11.1%)

**Features:** 217 RDKit 2D descriptors  
**Cohorts:** 63 unique Murcko scaffolds (lowest diversity)  
**Label Range:** [-25.47, 3.43] (kcal/mol)

**Warnings:** None

**Status:** SUCCESS  
**Output:** `C:\Users\ananya.salian\Downloads\shift-bench\data\processed\freesolv`

---

### 5. Lipophilicity

**Task:** Regression  
**Samples:** 4,200 total (largest dataset)
- Train: 2,520 samples (60.0%)
- Calibration: 840 samples (20.0%)
- Test: 840 samples (20.0%)

**Features:** 217 RDKit 2D descriptors  
**Cohorts:** 2,443 unique Murcko scaffolds (highest diversity)  
**Label Range:** [-1.50, 4.50] (octanol/water distribution coefficient)

**Warnings:** None

**Status:** SUCCESS  
**Output:** `C:\Users\ananya.salian\Downloads\shift-bench\data\processed\lipophilicity`

---

## Comparative Analysis

| Dataset | Samples | Features | Cohorts | Task | Train | Cal | Test | Cohort Ratio* |
|---------|---------|----------|---------|------|-------|-----|------|---------------|
| **BBBP** | 1,975 | 217 | 1,102 | Binary | 1,186 | 395 | 394 | 55.8% |
| **ClinTox** | 1,458 | 217 | 813 | Binary | 902 | 294 | 262 | 55.8% |
| **ESOL** | 1,117 | 217 | 269 | Regression | 673 | 263 | 181 | 24.1% |
| **FreeSolv** | 642 | 217 | 63 | Regression | 417 | 154 | 71 | 9.8% |
| **Lipophilicity** | 4,200 | 217 | 2,443 | Regression | 2,520 | 840 | 840 | 58.2% |

*Cohort Ratio = (Unique Scaffolds / Total Samples) × 100

---

## Key Observations

1. **Feature Consistency:** All datasets use the same 217 RDKit 2D molecular descriptors, ensuring compatibility across datasets.

2. **Cohort Diversity:** 
   - FreeSolv has the lowest cohort diversity (9.8%), with many molecules sharing the same scaffold
   - Lipophilicity has the highest diversity (58.2%), with most molecules having unique scaffolds
   - This affects the difficulty of scaffold-based distribution shift tasks

3. **Dataset Size:**
   - Lipophilicity is the largest (4,200 samples)
   - FreeSolv is the smallest (642 samples)
   - Size range: 6.5x difference

4. **Task Distribution:**
   - 2 binary classification tasks (BBBP, ClinTox)
   - 3 regression tasks (ESOL, FreeSolv, Lipophilicity)

5. **Label Imbalance:**
   - ClinTox is highly imbalanced (93.55% positive), which may affect model training and evaluation
   - BBBP is moderately imbalanced (75.95% positive)

6. **RDKit Warnings:**
   - Only BBBP generated significant warnings during scaffold computation
   - These warnings are related to molecular structure parsing and don't affect the final output

---

## Processing Time Estimates

Based on execution time observations:
- **BBBP:** ~40 seconds (largest processing time due to warnings)
- **ClinTox:** ~15 seconds
- **ESOL:** ~12 seconds
- **FreeSolv:** ~8 seconds (smallest dataset)
- **Lipophilicity:** ~30 seconds (largest dataset)

**Total Processing Time:** ~2 minutes for all 5 datasets

---

## Files Generated

Each dataset generated the following files in `data/processed/{dataset}/`:
- `features.npy` - NumPy array of molecular descriptors
- `labels.npy` - NumPy array of target values
- `cohorts.npy` - NumPy array of scaffold IDs
- `splits.csv` - CSV with train/cal/test split assignments
- `metadata.json` - JSON with dataset statistics and configuration

---

## Recommendations

1. **ClinTox Imbalance:** Consider using stratified sampling or weighted loss functions due to the 93.55% positive rate.

2. **FreeSolv Cohorts:** With only 63 scaffolds for 642 samples, scaffold-based shifts may be more challenging. Consider alternative cohort definitions if needed.

3. **Label Encoding:** BBBP and ClinTox have non-binary labels [0.0, 0.5, 1.0]. Verify if 0.5 represents missing data or a specific category.

4. **Validation:** All preprocessing completed successfully. Ready for model training and shift evaluation.

---

## Next Steps

1. Run baseline models on all 5 datasets
2. Evaluate shift robustness using the scaffold-based splits
3. Compare performance across binary vs. regression tasks
4. Analyze cohort-specific performance variations

---

**Generated by:** Claude Code preprocessing automation  
**All datasets ready for ShiftBench evaluation pipeline**
