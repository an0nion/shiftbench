# ShiftBench: Remaining Work to 100% D&B Ready

**Current Status**: 70% Complete
**Target**: 100% (NeurIPS D&B Submission Ready)
**Timeline**: 8-10 weeks
**Last Updated**: 2026-02-16

---

## Overview

This document provides a detailed, prioritized task list for completing ShiftBench and reaching 100% D&B readiness. Each task includes estimated hours, dependencies, and assignable owners (if team > 1 person).

**Total Estimated Hours**: ~320 hours (8 weeks full-time)
**With 2-3 person team**: 4-6 weeks calendar time

---

## Priority 1: Baseline Methods (4 Remaining)

### TASK 1.1: Implement Split Conformal Prediction

**Description**: Implement split conformal prediction for distribution-free coverage guarantees

**Priority**: HIGH
**Estimated Hours**: 12 hours
**Dependencies**: None (can start immediately)

**Subtasks**:
1. Research split conformal methodology (2 hours)
   - Read Vovk et al. 2005, Lei et al. 2018
   - Understand calibration set split
   - Understand conformal scores for binary classification

2. Implement `SplitConformalBaseline` class (6 hours)
   - Inherit from `BaselineMethod`
   - Implement `estimate_weights()` (identity weights for split conformal)
   - Implement `estimate_bounds()` using conformal quantiles
   - Add documentation and type hints

3. Test on existing datasets (3 hours)
   - Test on test_dataset (synthetic)
   - Test on BACE (molecular)
   - Test on Adult (tabular)
   - Verify coverage guarantees

4. Document implementation (1 hour)
   - Create `docs/SPLIT_CONFORMAL_REPORT.md`
   - Add usage examples
   - Add to `src/shiftbench/baselines/__init__.py`

**Deliverable**: `src/shiftbench/baselines/split_conformal.py` (200-300 lines)

**Test Command**:
```bash
python -m shiftbench.evaluate --method split_conformal --dataset bace
```

---

### TASK 1.2: Implement CV+ (Cross-Validation+)

**Description**: Implement CV+ for nested cross-validation with coverage guarantees

**Priority**: HIGH
**Estimated Hours**: 14 hours
**Dependencies**: Split Conformal complete (shares conformal methodology)

**Subtasks**:
1. Research CV+ methodology (2 hours)
   - Read Barber et al. 2021 (CV+ paper)
   - Understand K-fold nested structure
   - Understand aggregation across folds

2. Implement `CVPlusBaseline` class (8 hours)
   - Inherit from `BaselineMethod`
   - Implement K-fold splitting logic
   - Implement per-fold conformal prediction
   - Implement fold aggregation (union operation)
   - Add documentation

3. Test on existing datasets (3 hours)
   - Test with K=5, K=10
   - Verify coverage tighter than split conformal
   - Compare to split conformal results

4. Document implementation (1 hour)
   - Create `docs/CVPLUS_REPORT.md`
   - Add usage examples
   - Document computational cost (K times slower than split)

**Deliverable**: `src/shiftbench/baselines/cvplus.py` (300-400 lines)

**Test Command**:
```bash
python -m shiftbench.evaluate --method cvplus --dataset bace --folds 5
```

---

### TASK 1.3: Implement Group DRO (Distributionally Robust Optimization)

**Description**: Implement Group DRO for worst-case group performance optimization

**Priority**: MEDIUM
**Estimated Hours**: 20 hours (complex)
**Dependencies**: None (independent)

**Subtasks**:
1. Research Group DRO methodology (3 hours)
   - Read Sagawa et al. 2020 (DRO paper)
   - Understand robust loss optimization
   - Understand group reweighting strategy

2. Implement `GroupDROBaseline` class (10 hours)
   - Inherit from `BaselineMethod`
   - Implement group-wise loss computation
   - Implement robust reweighting (exponential weights update)
   - Implement iterative optimization loop
   - Add stopping criteria

3. Test on existing datasets (4 hours)
   - Test on Adult (multiple demographic groups)
   - Test on COMPAS (fairness-critical)
   - Verify worst-group performance improvement
   - Compare to uLSIF/KLIEP

4. Document implementation (2 hours)
   - Create `docs/GROUP_DRO_REPORT.md`
   - Add usage examples
   - Document hyperparameters (learning rate, iterations)

5. Debugging and refinement (1 hour)
   - Group DRO convergence can be tricky
   - May need to tune hyperparameters per dataset

**Deliverable**: `src/shiftbench/baselines/group_dro.py` (400-500 lines)

**Test Command**:
```bash
python -m shiftbench.evaluate --method group_dro --dataset adult
```

**Risk**: HIGH COMPLEXITY - May take longer than estimated
**Mitigation**: Use Wilds repository as reference implementation

---

### TASK 1.4: Implement BBSE (Black-Box Shift Estimation)

**Description**: Implement BBSE for label shift correction without target labels

**Priority**: MEDIUM
**Estimated Hours**: 16 hours
**Dependencies**: None (independent)

**Subtasks**:
1. Research BBSE methodology (2 hours)
   - Read Lipton et al. 2018 (BBSE paper)
   - Understand confusion matrix estimation
   - Understand EM-based label proportion estimation

2. Implement `BBSEBaseline` class (10 hours)
   - Inherit from `BaselineMethod`
   - Implement confusion matrix construction
   - Implement EM algorithm for label shift
   - Implement importance weight derivation from label proportions
   - Add convergence checks

3. Test on existing datasets (3 hours)
   - Test on imbalanced datasets (Bank, ClinTox)
   - Compare to covariate shift methods (uLSIF, KLIEP)
   - Verify label shift correction

4. Document implementation (1 hour)
   - Create `docs/BBSE_REPORT.md`
   - Explain when to use BBSE (label shift vs covariate shift)
   - Add usage examples

**Deliverable**: `src/shiftbench/baselines/bbse.py` (300-400 lines)

**Test Command**:
```bash
python -m shiftbench.evaluate --method bbse --dataset bank
```

**Note**: BBSE addresses label shift, not covariate shift. May have lower performance on ShiftBench (most datasets have covariate shift). Still valuable for completeness.

---

## Priority 2: Datasets (27 Remaining)

### TASK 2.1: Process Remaining Molecular Datasets

**Priority**: MEDIUM
**Estimated Hours**: 24 hours
**Dependencies**: Preprocessing script already exists

**Datasets to Add** (19 total):

**High Priority** (well-established):
1. MolHIV (41,120 samples) - 2 hours
2. HIV (41,127 samples) - 2 hours
3. PCBA (439,863 samples) - 3 hours (large!)
4. QM7 (7,165 samples) - 1 hour
5. QM8 (21,786 samples) - 2 hours
6. QM9 (133,885 samples) - 3 hours
7. MolMUV (93,087 samples) - Already processed ✅

**Medium Priority**:
8. SIDER (1,427 samples) - Already processed ✅
9. Tox21 (7,831 samples) - Already processed ✅
10. ToxCast (8,576 samples) - Already processed ✅
11. Delaney (regression, 1,128 samples) - 1 hour
12. Freesolv (regression, 642 samples) - Already processed ✅

**Low Priority** (if time permits):
13-19. Additional OGB molecular datasets - 5 hours total

**Subtasks**:
1. Download raw SMILES data (2 hours)
   - Use MoleculeNet download scripts
   - Cache in `data/raw_molecular/`

2. Run preprocessing script (8 hours compute)
   ```bash
   python scripts/preprocess_molecular.py --dataset molhiv
   python scripts/preprocess_molecular.py --dataset hiv
   # etc.
   ```

3. Validate processed files (2 hours)
   - Check features.npy dimensions
   - Check cohort assignments
   - Verify splits sum to 100%

4. Update registry.json (1 hour)
   - Add all 19 new datasets
   - Update metadata counts

**Deliverable**: 19 new molecular datasets in `data/processed/`

---

### TASK 2.2: Process Additional Tabular Datasets

**Priority**: MEDIUM
**Estimated Hours**: 16 hours
**Dependencies**: Preprocessing script already exists

**Datasets to Add** (10 total):

**High Priority** (fairness-critical):
1. ACS PUMS (American Community Survey) - 3 hours
   - Large census dataset
   - Geographic and demographic shifts
2. Law School Admissions - 1 hour
   - Education fairness
3. FICO Credit Score - 2 hours
   - Financial fairness

**Medium Priority** (diverse shifts):
4. Communities & Crime - 2 hours
   - Geographic shift across US communities
5. UCI Wine Quality - 1 hour
   - Regional shift (red vs white, Portuguese wines)
6. Credit Default (Taiwan) - 2 hours
   - Temporal shift (monthly)
7. Census Income (KDD) - 2 hours
   - Different from Adult, larger

**Low Priority**:
8. Cardiotocography - 1 hour
9. Occupancy Detection - 1 hour
10. Bike Sharing - 1 hour

**Subtasks**:
1. Download raw data (2 hours)
   - UCI, Kaggle, ProPublica sources

2. Run preprocessing script (6 hours)
   ```bash
   python scripts/preprocess_tabular.py --dataset acs_pums
   python scripts/preprocess_tabular.py --dataset law_school
   # etc.
   ```

3. Handle dataset-specific issues (4 hours)
   - Large datasets may need subsampling
   - Mixed feature types need special handling

4. Update registry (1 hour)

**Deliverable**: 10 new tabular datasets

---

### TASK 2.3: Process Additional Text Datasets

**Priority**: MEDIUM
**Estimated Hours**: 20 hours
**Dependencies**: Preprocessing script already exists

**Datasets to Add** (10 total):

**High Priority** (diverse shifts):
1. AG News - 2 hours
   - 120K news articles, topic shift
2. SST-2 (Stanford Sentiment) - 1 hour
   - Movie reviews, temporal shift
3. Reddit - 3 hours
   - Subreddit shift (cross-community)
4. Twitter Hate Speech - 2 hours
   - Demographic shift (protected groups)

**Medium Priority**:
5. TREC - 1 hour
   - Question classification
6. Subjectivity - 1 hour
   - Subjective vs objective
7. DBpedia - 2 hours
   - Entity classification
8. Yahoo Answers - 2 hours
   - Topic shift

**Low Priority**:
9. WikiText - 2 hours
   - Language modeling (temporal)
10. OpenWebText subset - 3 hours
   - Web data (domain shift)

**Subtasks**:
1. Download raw text data (3 hours)
   - HuggingFace datasets
   - Kaggle
   - Academic sources

2. Run preprocessing script (10 hours)
   ```bash
   python scripts/preprocess_text.py --dataset ag_news
   python scripts/preprocess_text.py --dataset sst2
   # etc.
   ```

3. Handle text-specific issues (4 hours)
   - TF-IDF may be slow on large datasets
   - Consider subsampling for OpenWebText

4. Update registry (1 hour)

**Deliverable**: 10 new text datasets

---

## Priority 3: Full Benchmark Evaluation

### TASK 3.1: Run Complete Benchmark

**Priority**: HIGH (after baselines + datasets complete)
**Estimated Hours**: 16 hours (mostly compute time)
**Dependencies**: 10 baselines + 50 datasets

**Subtasks**:
1. Set up batch evaluation (2 hours)
   ```bash
   python scripts/run_full_benchmark.py \
       --methods all \
       --datasets all \
       --output results/full_benchmark/
   ```

2. Run evaluations (8 hours compute)
   - 10 methods × 50 datasets = 500 evaluations
   - Estimated 1-2 minutes per evaluation
   - Total: 8-16 hours (can parallelize)

3. Quality checks (3 hours)
   - Verify all 500 results exist
   - Check for NaN or invalid values
   - Identify failed evaluations
   - Re-run failures

4. Generate aggregated statistics (3 hours)
   - Certification rates by method, dataset, domain
   - Runtime statistics
   - Weight diagnostics
   - Bound quality metrics

**Deliverable**: `results/full_benchmark/` with 500+ CSV files + aggregated summaries

---

### TASK 3.2: Cross-Domain Analysis

**Priority**: HIGH
**Estimated Hours**: 12 hours
**Dependencies**: Full benchmark complete

**Subtasks**:
1. Method ranking by domain (4 hours)
   - Which methods work best for molecular?
   - Which methods work best for tabular?
   - Which methods work best for text?
   - Statistical significance tests (paired t-tests)

2. Dataset difficulty ranking (3 hours)
   - Certification rate by dataset
   - Correlate with dataset characteristics (sample size, cohort count, positive rate)
   - Identify hardest/easiest datasets

3. Failure mode analysis (4 hours)
   - Which cohorts fail most often?
   - Which tau thresholds are hardest?
   - Predict failure from diagnostics (PSIS k, ESS, etc.)

4. Create comparison figures (1 hour)
   - Method ranking heatmap
   - Dataset difficulty scatter plot
   - Failure mode distributions

**Deliverable**: `results/analysis/` with statistical tests and figures

---

### TASK 3.3: Generate All Figures and Tables

**Priority**: HIGH (for paper)
**Estimated Hours**: 16 hours
**Dependencies**: Analysis complete

**Figures Needed** (6-8 total):

1. **Figure 1: ShiftBench Overview** (2 hours)
   - Illustration of shift types
   - Example cohort structure
   - Evaluation pipeline flowchart

2. **Figure 2: Method Comparison** (3 hours)
   - Certification rate by method (bar chart)
   - Speed vs tightness tradeoff (scatter plot)
   - Cross-domain performance (heatmap)

3. **Figure 3: KLIEP-uLSIF Agreement** (2 hours)
   - Scatter plot of lower bounds
   - Agreement matrix across tau values

4. **Figure 4: Stability Gating Impact** (2 hours)
   - Certification rate with/without gating
   - Diagnostic distributions (PSIS k, ESS)

5. **Figure 5: Dataset Characteristics** (2 hours)
   - Sample size distribution
   - Cohort count distribution
   - Shift type coverage

6. **Figure 6: Failure Mode Analysis** (3 hours)
   - Failure predictors (regression coefficients)
   - Example failed vs certified cohorts

**Tables Needed** (4-6 total):

1. **Table 1: Dataset Summary** (1 hour)
   - All 50 datasets with key statistics

2. **Table 2: Method Comparison** (1 hour)
   - All 10 methods with performance metrics

3. **Table 3: Cross-Domain Results** (1 hour)
   - Method rankings by domain

4. **Table 4: Computational Cost** (1 hour)
   - Runtime by method and dataset size

**Deliverable**: `figures/` and `tables/` directories with publication-ready graphics

---

## Priority 4: Paper Writing

### TASK 4.1: Write Paper Outline

**Priority**: URGENT (start immediately)
**Estimated Hours**: 4 hours
**Dependencies**: None

**Subtasks**:
1. Draft detailed outline (2 hours)
   - Section structure
   - Paragraph-level bullet points
   - Figure/table placements
   - Page budget allocation

2. Identify key messages (1 hour)
   - Main contributions (3-5 bullets)
   - Novel findings (3-5 bullets)
   - Practical implications

3. Create figure mockups (1 hour)
   - Sketch 6 figures (hand-drawn or whiteboard)
   - Determine which results to highlight

**Deliverable**: `docs/PAPER_OUTLINE.md` (this session)

---

### TASK 4.2: Write Introduction (1 page)

**Priority**: HIGH
**Estimated Hours**: 8 hours
**Dependencies**: Outline complete

**Subtasks**:
1. Motivation paragraph (2 hours)
   - Why shift-aware evaluation matters
   - Failure stories under shift

2. Problem statement (2 hours)
   - Existing methods not systematically compared
   - Lack of cross-domain benchmarks
   - Need for reproducible infrastructure

3. Contributions paragraph (2 hours)
   - ShiftBench benchmark (10 methods, 50 datasets, 3 domains)
   - Novel findings (KLIEP-uLSIF equivalence, gating necessity)
   - Reproducible infrastructure (hash-chained receipts)

4. Paper structure paragraph (1 hour)
   - Brief overview of sections

5. Polish and refine (1 hour)

**Deliverable**: Introduction section (~600 words)

---

### TASK 4.3: Write Related Work (0.5 pages)

**Priority**: HIGH
**Estimated Hours**: 6 hours
**Dependencies**: None (can parallelize with intro)

**Subtasks**:
1. Domain adaptation benchmarks (2 hours)
   - DomainBed, WILDS, Shifts
   - How ShiftBench differs (evaluation focus, not training)

2. Covariate shift methods (2 hours)
   - Importance weighting (uLSIF, KLIEP, KMM)
   - Conformal prediction (weighted, split, CV+)
   - Distributionally robust optimization

3. Evaluation under shift (1 hour)
   - Calibration under shift
   - Fairness under shift

4. Polish and integrate (1 hour)

**Deliverable**: Related Work section (~400 words)

---

### TASK 4.4: Write ShiftBench Design (1.5 pages)

**Priority**: HIGH
**Estimated Hours**: 10 hours
**Dependencies**: Outline complete

**Subtasks**:
1. Dataset selection criteria (2 hours)
   - Diversity (domains, shift types, sizes)
   - Accessibility (licenses, redistribution)
   - Relevance (real-world applications)

2. Shift types covered (2 hours)
   - Scaffold, demographic, temporal, geographic, category
   - How cohorts are defined per domain

3. Evaluation protocol (3 hours)
   - Train/cal/test splits
   - Oracle predictions (labels as predictions)
   - Tau grids
   - FWER control (Holm's step-down)
   - Certification vs abstention

4. Receipt system (2 hours)
   - Hash-chained receipts
   - Reproducibility guarantees
   - Auditability

5. Polish (1 hour)

**Deliverable**: ShiftBench Design section (~900 words)

---

### TASK 4.5: Write Datasets Section (1.5 pages)

**Priority**: HIGH
**Estimated Hours**: 10 hours
**Dependencies**: All datasets processed

**Subtasks**:
1. Molecular datasets (3 hours)
   - MoleculeNet overview
   - Scaffold shift explanation
   - Statistics (Table 1)

2. Tabular datasets (3 hours)
   - UCI + fairness datasets
   - Demographic and temporal shifts
   - Protected attributes

3. Text datasets (3 hours)
   - NLP datasets overview
   - Various shift types
   - Preprocessing (TF-IDF)

4. Dataset statistics table (1 hour)
   - Create Table 1 (all 50 datasets)

**Deliverable**: Datasets section (~900 words)

---

### TASK 4.6: Write Baseline Methods Section (1.5 pages)

**Priority**: HIGH
**Estimated Hours**: 10 hours
**Dependencies**: All baselines implemented

**Subtasks**:
1. Density ratio methods (3 hours)
   - uLSIF, KLIEP, KMM, RULSIF
   - Mathematical formulations
   - Key differences

2. Conformal methods (3 hours)
   - Weighted, Split, CV+
   - Distribution-free guarantees

3. Other methods (2 hours)
   - RAVEL (gating), Group DRO, BBSE
   - When to use each

4. Method comparison table (2 hours)
   - Create Table 2 (all 10 methods)

**Deliverable**: Baseline Methods section (~900 words)

---

### TASK 4.7: Write Results & Analysis Section (1.5 pages)

**Priority**: HIGH
**Estimated Hours**: 16 hours
**Dependencies**: Full benchmark complete, analysis done

**Subtasks**:
1. Overall results (4 hours)
   - Certification rates by method (Figure 2)
   - Runtime comparisons (Table 4)
   - Key takeaways

2. Key findings (6 hours)
   - Finding 1: KLIEP-uLSIF equivalence (Figure 3)
   - Finding 2: Stability gating necessity (Figure 4)
   - Finding 3: Cross-domain insights (Table 3)
   - Statistical significance tests

3. Failure mode analysis (4 hours)
   - When methods fail (Figure 6)
   - Predictive factors
   - Practical guidance

4. Polish (2 hours)

**Deliverable**: Results & Analysis section (~900 words)

---

### TASK 4.8: Write Conclusion (0.5 pages)

**Priority**: MEDIUM
**Estimated Hours**: 4 hours
**Dependencies**: All sections complete

**Subtasks**:
1. Summary of contributions (1 hour)
2. Impact statement (1 hour)
3. Limitations (1 hour)
4. Future work (1 hour)

**Deliverable**: Conclusion section (~300 words)

---

### TASK 4.9: Write Appendix

**Priority**: MEDIUM
**Estimated Hours**: 12 hours
**Dependencies**: Main paper complete

**Sections Needed**:
1. Full results tables (3 hours)
   - All 500 evaluations
   - Per-cohort results for key datasets

2. Dataset details (3 hours)
   - Preprocessing procedures
   - Cohort definitions
   - Licenses and citations

3. Method implementation details (3 hours)
   - Hyperparameters
   - Algorithmic details
   - Computational complexity

4. Ablation studies (2 hours)
   - Effect of hyperparameter choices
   - Sensitivity analysis

5. Code availability statement (1 hour)

**Deliverable**: Appendix (~unlimited pages)

---

### TASK 4.10: Create All Figures (Final Versions)

**Priority**: HIGH
**Estimated Hours**: 16 hours (already estimated in TASK 3.3)
**Dependencies**: Analysis complete

Covered in TASK 3.3 above.

---

### TASK 4.11: Proofread and Polish

**Priority**: HIGH
**Estimated Hours**: 12 hours
**Dependencies**: Complete draft

**Subtasks**:
1. Self-review (4 hours)
   - Check for clarity
   - Fix typos and grammar
   - Verify citations
   - Check figure/table references

2. Co-author review (4 hours)
   - Incorporate feedback
   - Revise sections

3. External review (4 hours, if available)
   - Get fresh perspective
   - Address blind spots

**Deliverable**: Polished paper ready for submission

---

## Priority 5: Submission Materials

### TASK 5.1: Prepare Code Release

**Priority**: HIGH
**Estimated Hours**: 8 hours
**Dependencies**: All code complete

**Subtasks**:
1. Clean up code (2 hours)
   - Remove debug prints
   - Add missing docstrings
   - Fix linting issues

2. Create README (2 hours)
   - Installation instructions
   - Quick start guide
   - Example commands

3. Create CONTRIBUTING guide (1 hour)
   - How to add new methods
   - How to add new datasets

4. License file (1 hour)
   - Choose license (MIT or Apache 2.0)

5. Test installation (2 hours)
   - Fresh virtual environment
   - Follow README instructions
   - Verify all examples work

**Deliverable**: Public GitHub repository

---

### TASK 5.2: Prepare Data Release

**Priority**: HIGH
**Estimated Hours**: 6 hours
**Dependencies**: All datasets processed

**Subtasks**:
1. Upload processed datasets (3 hours)
   - Zenodo or similar
   - Generate DOI
   - Update registry with URLs

2. Create dataset README (2 hours)
   - Describe each dataset
   - Licenses and citations
   - Download instructions

3. Verify downloads work (1 hour)
   - Test from clean machine

**Deliverable**: Hosted datasets with DOI

---

### TASK 5.3: Create Submission Package

**Priority**: HIGH
**Estimated Hours**: 4 hours
**Dependencies**: Paper complete, code/data released

**Subtasks**:
1. Compile submission materials (2 hours)
   - Paper PDF
   - Supplementary material PDF (appendix)
   - Code link
   - Data link

2. Verify NeurIPS D&B requirements (1 hour)
   - Check formatting
   - Check page limits
   - Check anonymization (if required)

3. Submit (1 hour)
   - Upload to CMT
   - Verify submission

**Deliverable**: Submitted to NeurIPS D&B

---

## Summary: Total Effort Estimation

### By Priority

| Priority | Tasks | Hours |
|----------|-------|-------|
| Priority 1: Baselines | 4 tasks | 62 hours |
| Priority 2: Datasets | 3 tasks | 60 hours |
| Priority 3: Evaluation | 3 tasks | 44 hours |
| Priority 4: Paper | 11 tasks | 112 hours |
| Priority 5: Submission | 3 tasks | 18 hours |
| **Total** | **24 tasks** | **296 hours** |

### By Week (10-week plan)

| Week | Focus | Hours |
|------|-------|-------|
| 1 | Baselines (Split Conformal, CV+) | 26 hours |
| 2 | Baselines (Group DRO, BBSE) | 36 hours |
| 3 | Datasets (Molecular) | 24 hours |
| 4 | Datasets (Tabular + Text) | 36 hours |
| 5 | Full Benchmark Run | 28 hours |
| 6 | Analysis + Figures | 32 hours |
| 7 | Paper (Intro, Related, Design, Datasets) | 38 hours |
| 8 | Paper (Methods, Results, Conclusion) | 42 hours |
| 9 | Appendix + Polish | 24 hours |
| 10 | Submission Materials | 10 hours |
| **Total** | **10 weeks** | **296 hours** |

**Average**: 29.6 hours/week (~4 days/week full-time)

---

## Critical Path

Tasks that MUST be completed in sequence (cannot parallelize):

1. Complete all baselines (Weeks 1-2)
2. Complete all datasets (Weeks 3-4)
3. Run full benchmark (Week 5)
4. Analyze results (Week 6)
5. Write results section (Week 8)
6. Polish and submit (Weeks 9-10)

**Paper writing** can start in parallel:
- Write intro/related/design during Weeks 3-6 (while datasets/eval running)

---

## Parallelization Opportunities

If working with a team:

**Person A** (Weeks 1-4): Baseline implementations
**Person B** (Weeks 1-4): Dataset processing
**Person C** (Weeks 1-6): Paper writing (intro, related, design, datasets, methods)

**Weeks 5-6**: All focus on evaluation and analysis
**Weeks 7-8**: All focus on results section and figures
**Weeks 9-10**: All focus on polish and submission

**With 3-person team**: 6-8 weeks instead of 10 weeks

---

## High-Risk Items (Require Extra Attention)

1. **Group DRO implementation** (20 hours estimated, may be 30)
   - Complex optimization
   - May need hyperparameter tuning

2. **Paper writing** (112 hours estimated, may be 140)
   - Writing always takes longer than expected
   - Multiple revision rounds

3. **Large dataset processing** (PCBA: 439K samples, OpenWebText)
   - May hit memory limits
   - May need subsampling or special handling

**Recommendation**: Build in 1-2 week buffer for these

---

## Next Actions (This Week)

**Today** (Session 4):
- [x] Create this REMAINING_WORK.md document
- [ ] Create D&B_SUBMISSION_CHECKLIST.md
- [ ] Create KEY_FINDINGS_FOR_PAPER.md
- [ ] Create PAPER_OUTLINE.md

**This Week** (Days 1-7):
- [ ] Start Split Conformal implementation (TASK 1.1)
- [ ] Start paper outline (TASK 4.1)
- [ ] License audit (1 day task, not listed above)

---

## Conclusion

**Total Work Remaining**: ~296 hours (8 weeks full-time)
**Timeline**: 8-10 weeks to submission
**Status**: ON TRACK

**Key to Success**:
1. Start paper outline immediately (don't wait for all results)
2. Parallelize where possible (baselines + datasets + paper)
3. Build in buffer time for high-risk items (Group DRO, paper writing)
4. Maintain momentum with weekly milestones

**Confidence**: HIGH (85%) - Clear task breakdown, proven velocity, no major blockers

---

**Document Prepared By**: Claude Sonnet 4.5
**Last Updated**: 2026-02-16
**Next Update**: Weekly (after each major milestone)
