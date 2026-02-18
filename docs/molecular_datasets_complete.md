# Molecular Datasets - Complete Coverage

## Processing Summary (Feb 16, 2026)

All 5 remaining molecular datasets processed successfully\!

### SIDER
- Samples: 1,427 (train=921, cal=286, test=220)
- Features: 217
- Cohorts: 868
- Positive rate: 52.07%
- Processing time: ~30 seconds

### Tox21
- Samples: 7,831 (train=4,765, cal=1,567, test=1,499)
- Features: 217
- Cohorts: 2,405
- Positive rate: 4.25% (92.8% valid)
- Processing time: ~40 seconds

### ToxCast
- Samples: 8,576 (train=5,214, cal=1,715, test=1,647)
- Features: 217
- Cohorts: 2,508
- Positive rate: 25.59% (20.2% valid)
- Processing time: ~2.5 minutes

### MUV
- Samples: 93,087 (train=55,854, cal=18,617, test=18,616)
- Features: 217
- Cohorts: 42,930
- Positive rate: 0.18% (15.9% valid)
- Processing time: ~3 minutes
- Note: LARGEST molecular dataset

### MolHIV
- Samples: 41,120 (train=24,672, cal=8,245, test=8,203)
- Features: 217
- Cohorts: 19,082
- Positive rate: 3.51%
- Processing time: ~2 minutes

## Total Coverage: 11/11 Molecular Datasets

ShiftBench now includes:
1. BACE (1,513)
2. BBBP (2,039)
3. ClinTox (1,478)
4. ESOL (1,128)
5. FreeSolv (642)
6. Lipophilicity (4,200)
7. SIDER (1,427) NEW
8. Tox21 (7,831) NEW
9. ToxCast (8,576) NEW
10. MUV (93,087) NEW
11. MolHIV (41,120) NEW

## Verification: All Passed

- Preprocessing: SUCCESS
- Loading: SUCCESS
- Evaluation: SUCCESS (tested with ULSIF on SIDER)

## Key Statistics

- Total samples across all molecular datasets: ~162,000
- Feature dimension: 217 (RDKit 2D descriptors)
- Scaffold-based cohorts for realistic covariate shift
- Dataset sizes: 642 to 93,087 samples
- Class balance: 0.18% to 52% positive rate

## Next Steps

With complete molecular coverage, you can now:
1. Run full benchmarks across all 11 datasets
2. Compare small vs large dataset performance
3. Study impact of class imbalance
4. Analyze scaffold diversity effects
5. Investigate multilabel handling
