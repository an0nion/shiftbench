# ShiftBench Research Logbook

---

## Session 10 -- 2026-02-23

### Work Completed This Session

**H3 Regression Expanded (27 datasets -- MAJOR REVISION):**
Script: scripts/analysis_h3_regression_40.py. Results: results/h3_regression_40/.

Combined RAVEL tabular/text (20 datasets, Session 9) + uLSIF molecular
(7 real-pred datasets from cross_domain_extended). Computed shift metrics
for 12 new datasets via two-sample AUC classifier.

KEY FINDING REVISION: n_eff (R2=0.645) now outperforms domain alone (R2=0.504)
as a cross-domain predictor. Partial R2(n_eff | domain) = 0.406 (was 0.002 in
6-dataset analysis). The old "domain R2=0.994" was an overfitting artifact of
6 maximally-separated data points.

New claim: "n_eff is the primary mechanistic driver; domain partially mediates
n_eff (scaffold shift -> low n_eff; text overlap -> high n_eff) but does not
subsume it."

Within-domain n_eff coefficients all positive:
- Molecular: R2(neff)=0.525, neff_coef=+0.058
- Tabular:   R2(neff)=0.386, neff_coef=+0.087
- Text:      R2(neff)=0.438, neff_coef=+0.108

**H2 Wilson CI finding documented in SECTION_5_REAL_DATA.md (Section 5.4.4):**
neff_ess_gated FWER=5.24%, CI=[0.048, 0.057] -- criterion met.
naive_* variants FWER~6.5%, CI_upper~0.070 -- NOT MET (anti-conservative ~1.3pp).
Conclusion: n_eff correction necessary for calibration at boundary null.

**SECTION_5_REAL_DATA.md updated:**
- Section 5.1: experimental setup expanded to 27 datasets
- Section 5.2: Finding 1 revised (n_eff > domain as predictor)
- Section 5.4.4: added H2 Wilson CI calibration finding (new section)
- Section 5.5.1: H3 regression table updated with 27-dataset results
- Section 5.6: RAVEL behavior expanded with tabular/text per-dataset tables

**PI Issues Audit -- Updated:**

| # | Priority | Status (Session 10) |
|---|----------|--------------------|
| 1 | WCP validation on all datasets | DONE (Session 7) |
| 2 | Full method matrix (6 methods x 40 datasets) | PARTIAL -- uLSIF/KLIEP/WCP on 6; RAVEL on 20 tabular/text |
| 3 | Add 27 more datasets (target 40) | DONE (Session 9) -- 40 in prediction_mapping |
| 4 | H3 regression (cert_rate ~ domain + n_eff + shift) | DONE -- revised with 27 datasets |
| 5 | 8-page paper draft | PARTIAL -- Sections 4+5 complete; Intro/Method/Related Work/Conclusion missing |

---

## Session 9 -- 2026-02-23

### Work Completed This Session

**Dataset expansion to 40 (Priority 3 -- COMPLETE):**
Added 15 new datasets (credit_default, hate_speech, rotten_tomatoes + others).
prediction_mapping.json expanded from 25 to 40 datasets.

Bug fixes applied before running new datasets:
1. tox21/toxcast/muv: NaN mismatch fixed (nan_to_num instead of drop_nan_labels)
2. amazon/civil_comments/twitter: replaced sparse TF-IDF (15-21 features,
   AUC ~0.5) with real HuggingFace TF-IDF. New AUC: 0.83-0.96.

**RAVEL on tabular (11 datasets, Priority 2 -- DONE for tabular):**
Script: scripts/run_cross_domain_benchmark.py with method=ravel.
Results: results/ravel_tabular_text/
Domain summary: cert_rate=4.95%, mean n_eff=37 (11 datasets)

**RAVEL on text (9 datasets, Priority 2 -- DONE for text):**
Results: results/ravel_tabular_text/
Domain summary: cert_rate=40.82%, mean n_eff=485 (9 datasets)

**H4 per-tau null test (extended):**
0 false certifications across 6000 trials (200 x 6 datasets x 5 taus). PASS.

**H2 Wilson CI 10k:**
neff_ess_gated: FWER=5.24%, CI=[0.048, 0.057]. Criterion (CI_upper < 0.06) MET.
naive_* variants: FWER~6.5%, CI_upper~0.070. NOT MET.
Results: results/h2_wilson_10k/h2_wilson_10k_proper.csv

**Committed and pushed to an0nion/shiftbench** (2 commits: Session 9 + Wilson CI).

---

## Session 8 — 2026-02-20

### Work Completed This Session

**PI Priority 3 -- Dataset Expansion (PARTIAL PROGRESS):**
Expanded benchmark from 10 datasets with model predictions to 25 datasets.
Scripts: scripts/train_new_datasets.py, scripts/run_extended_benchmark.py.
Results: results/cross_domain_extended/

Approach:
- All 26 datasets in data/processed/ were already fully preprocessed (features.npy,
  labels.npy, cohorts.npy, splits.csv). The gap was missing trained model predictions.
- Trained RF models (molecular, 217-dim RDKit features) and LR models (tabular/text)
  for 13 datasets: freesolv, lipophilicity, sider, tox21, toxcast, muv, molhiv,
  diabetes, heart_disease, student_performance, amazon, civil_comments, twitter.
- Added camelyon17, waterbirds to prediction_mapping.json (predictions already existed).
- prediction_mapping.json expanded from 10 to 25 datasets.

Model training results (test AUC):

| Dataset           | Domain    | AUC   | Notes                            |
|-------------------|-----------|-------|----------------------------------|
| freesolv          | molecular | 0.939 | median-split binarization        |
| lipophilicity     | molecular | 0.849 | median-split binarization        |
| sider             | molecular | 0.692 |                                  |
| tox21             | molecular | ~0.5  | NaN->0; very low pred_pos        |
| toxcast           | molecular | ~0.5  | NaN->0; very low pred_pos        |
| muv               | molecular | ~0.5  | NaN->0; very low pred_pos        |
| molhiv            | molecular | 0.775 | very imbalanced                  |
| diabetes          | tabular   | ~0.75 |                                  |
| heart_disease     | tabular   | ~0.80 |                                  |
| student_performance | tabular | ~0.85 |                                  |
| amazon            | text      | ~0.5  | only 21 TF-IDF features (sparse) |
| civil_comments    | text      | ~0.5  | only 15 TF-IDF features (sparse) |
| twitter           | text      | ~0.5  | only 21 TF-IDF features (sparse) |

Extended benchmark (23 datasets, uLSIF, results/cross_domain_extended/):

| Dataset             | Domain    | cert_%  | n_eff   |
|---------------------|-----------|---------|---------|
| yelp                | text      | 62.00   | 424.8   |
| imdb                | text      | 60.00   | 538.4   |
| student_performance | tabular   | 18.18   | 15.2    |
| freesolv            | molecular | 26.67   | 15.1    |
| compas              | tabular   | 8.42    | 15.2    |
| esol                | molecular | 6.67    | 24.4    |
| toxcast             | molecular | 3.81    | 4.1     |
| adult               | tabular   | 2.45    | 37.3    |
| tox21               | molecular | 0.75    | 0.4     |
| clintox             | molecular | 0.73    | 1.4     |
| bbbp                | molecular | 0.63    | 1.8     |
| bace                | molecular | 0.31    | 0.6     |
| molhiv/muv/amazon/  | --        | 0.00    | ~0      |
| civil_comments/twitter/... | -- | 0.00   | --      |

KNOWN ISSUES:
1. tox21/toxcast/muv: prediction length mismatch (NaN->0 in training vs drop-NaN
   in benchmark). Falls back to oracle predictions (y > 0.5). cert_rate is oracle-based.
2. amazon/civil_comments/twitter: sparse TF-IDF (15-21 features) gives AUC ~0.5.
   Models are near-random; 0% cert rate reflects model quality, not dataset behaviour.
3. Still need 15+ more datasets to reach PI's 40-dataset target. All 26 preprocessed
   datasets are now exhausted; new raw data must be downloaded for further expansion.

STATUS: 25/40 datasets registered. Priority 3 remains PARTIAL.

---

## Session 7 — 2026-02-19 (continuation)

### Work Completed This Session

**PI Priority 1 -- WCP vs EB on All Real Datasets (RESOLVED):**
Ran Weighted Conformal Prediction vs Empirical-Bernstein comparison across all
6 Tier-A datasets with real model predictions. Script: scripts/wcp_vs_eb_all_datasets.py.
Results: results/wcp_vs_eb_all_datasets/

Key results (raw cert rate; no Holm correction; per-cohort decisions):

| Dataset | Domain     | WCP cert% | EB cert% | Ratio | WCP tighter% |
|---------|-----------|-----------|---------|-------|--------------|
| BACE    | molecular | 8.0%      | 0.0%    | 2.0x  | 100%         |
| BBBP    | molecular | 76.0%     | 44.0%   | 1.7x  | 100%         |
| Adult   | tabular   | 1.8%      | 0.0%    | 4.0x  | 86%          |
| COMPAS  | tabular   | 4.5%      | 0.6%    | 7.0x  | 94%          |
| IMDB    | text      | 40.0%     | 40.0%   | 1.0x  | 60%          |
| Yelp    | text      | 0.0%      | 0.0%    | --    | 100%         |

n_eff breakdown: WCP advantage largest at n_eff < 300 (92-93% tighter).
At n_eff 300+, advantage drops to 90% tighter but cert ratios converge for text.

CONCLUSION: Finding 3 is NOT cherry-picked. WCP generalises to tabular
(COMPAS 7x, Adult 4x) and molecular (BACE/BBBP). Text at high n_eff is the
one domain where WCP and EB converge. The advantage is n_eff-mediated: when
n_eff is very large (IMDB: 39,996), quantile and EB bounds both become very
tight and agree. At n_eff < 300, WCP consistently provides higher lower bounds.

CAVEAT (preserved from PI report): WCP and EB have different guarantees.
WCP provides marginal coverage (distribution-free); EB provides concentration
bounds (sub-Gaussian assumption). The finding is practically meaningful but
the comparison is not apples-to-apples theoretically.

Updated docs: SECTION_5_REAL_DATA.md (Section 5.7), FORMAL_CLAIMS.md

---

## PI ISSUES AUDIT UPDATE -- Session 7

| # | Priority | Status (Session 7) |
|---|----------|--------------------|
| 1 | WCP validation on all datasets | **DONE** -- see above |
| 2 | Full method matrix (6 methods x 23 datasets) | PARTIAL -- uLSIF/KLIEP/WCP all 6; RAVEL molecular only |
| 3 | Add 27 more datasets | NOT DONE |
| 4 | Regression analysis | DONE (Session 5-6) |
| 5 | 8-page paper draft | PARTIAL (Sections 4+5 complete; Intro/Method/Related Work/Conclusion missing) |

Q1 RESOLVED: Finding 3 generalises -- WCP advantage is real and systematic
Q2 RESOLVED: KLIEP-uLSIF agreement mechanism (near-identical weights)
Q3 UNRESOLVED: scope (27+ datasets not yet added)
Q4 UNRESOLVED: venue (blocked by Q3)

### What Remains After Session 7

**Critical (must do before submission):**
1. Add 27+ datasets (Priority 3, scope minimum for D&B submission)
2. Run RAVEL on text/tabular (Priority 2 completion)
3. Paper: Intro, Method, Related Work, Conclusion sections

**Should do:**
4. H4 stronger semi-synthetic: set true_ppv below each tau independently
5. H2 Wilson CI: run 10,000 trials to satisfy CI_upper < 0.06 criterion
6. Finding 5 (method ranking by domain): needs RAVEL + WCP on all domains

**Complete (as of Session 7):**
- All 5 CHANGES.md PI feedback deliverables
- H1-H4 ablation suite with mechanistic explanations
- WCP vs EB across all 6 Tier-A datasets (Priority 1 DONE)
- SECTION_4_VALIDITY.md (complete)
- SECTION_5_REAL_DATA.md (complete, Section 5.7 added)
- FORMAL_CLAIMS.md (fully updated)
- KLIEPFast implementation
- Binarization sensitivity

---

## Session 6 -- 2026-02-19

### Work Completed This Session

**H1 Mechanistic Ablation (P1.1-P1.5):** Post-hoc re-analysis of 4,350 stored
(cohort, tau) pairs from experiment_c_real. KLIEP not re-run (cost prohibitive).

- P1.1 alpha sweep: 5/6 datasets 100% agreement at ALL alpha (0.001-0.20).
  BBBP worsens at stricter alpha (77.3% at a=0.001 vs 88.4% at a=0.05).
  KLIEP is consistently more conservative than uLSIF.
- P1.2 tau density: BBBP disagreements cluster at tau=0.6 (16.5%) and
  tau=0.7 (61.1%). At tau=0.8+, no active BBBP pairs.
- Driver analysis: lb_gap (r=+0.31) and mu_gap (r=+0.31) are top predictors.
  neff_ratio (r=+0.08) is NOT a driver. Mean neff_ratio=1.003 across all pairs.
- MECHANISM: Agreement is driven by near-identical weight estimates. KLIEP
  (KL minimization) and uLSIF (L2 minimization) converge to near-identical
  solutions on these datasets. BBBP exception: scaffold shift causes KL/L2
  divergence at tau=0.6-0.7 borderline.
- Script: scripts/h1_posthoc_analysis.py | Results: results/h1_ablation/

**H2 Gate Isolation:** 8 configs x 6 sigma x 300 trials = 14,400 trials.
- k-hat is DOMINANT gate: removing it inflates cert_rate 3.4x at sigma=1.5.
- ESS is secondary (1.3x inflation). Clip-mass tertiary.
- In Gaussian-shift regime, FWER controlled by EB regardless of gating.
  Gates reduce spurious certs; they don't prevent FWERs in this regime.
- The 78% FWER improvement (H2 adversarial) is driven primarily by k-hat.
- Results: results/h2_gate_isolation/

**H3 Subsampling (P3.4):** IMDB cert_rate drops 40pp (40% -> 0% at
cohort_size=20). Pre-registered 60pp threshold NOT MET. Honestly reported.
Mechanism confirmed (n_eff monotonically drives cert_rate within IMDB).
Yelp: 0% cert throughout due to PPV < 0.5 in this 10-cohort structure.
Results: results/h3_subsampling/

**H4 Real-Data FWER:** 0/3,600 false certs (6 datasets x 3 offsets x 200 trials).
KNOWN LIMITATION: null only at tau=0.9; a stronger test would set true_ppv
below each tau independently. Documented in SECTION_4_VALIDITY.md.

**H4 Bootstrap (P4.3):** EB wider in 84.6% overall. 4/6 datasets meet 70%
target individually (adult 61.8%, imdb 60.0% slightly below).

**Binarization:** Clinical thresholds within +-7pp of median-split. Not a
confound. Lipophilicity 0% cert regardless of threshold.

**KLIEPFast:** Implemented src/shiftbench/baselines/kliep_fast.py.
Subsample 500 from cal+target, SLSQP on subsample, extrapolate weights to
full cal via kernel evaluation. Used in H1 ablation to handle COMPAS (4937 samples).

**Paper sections written:**
- docs/SECTION_4_VALIDITY.md (complete, all placeholders filled)
- docs/SECTION_5_REAL_DATA.md (complete, all placeholders filled)
- docs/FORMAL_CLAIMS.md (Section 6 updated, all IN PROGRESS -> COMPLETED)

**Committed and pushed to an0nion/shiftbench** (2 commits this session).

---

## PI ISSUES AUDIT -- Full Reconciliation (Session 6)

Sources: CHANGES.md (PI feedback fixes), PI_REPORT_FOR_REVIEW.md (priority actions).

### CHANGES.md: 5 Deliverables

| # | Issue | Status | Evidence |
|---|-------|--------|----------|
| 1 | H1 tabular active pairs (experiment_c used n_cohorts=5, no active pairs) | DONE (Session 5) | results/h1_disagreement/; COMPAS 100% (17 pairs), Adult 85.7% (7 pairs) |
| 2 | H2 Wilson CI power analysis (500 trials insufficient, CI_upper=0.073 > 0.06) | DONE (Session 5) | results/h2_good_weights_500/h2_power_analysis.csv; need ~10,000 trials for 90% power |
| 3 | H3 regression rewrite (oracle filter, partial R2, within-domain) | DONE (Session 4-5) | results/h3_regression/; Domain R2=0.994, n_eff R2=0.708, partial R2=0.002 |
| 4 | H4 slack by n_eff + minimum n_eff table | DONE (Session 4-5) | results/h4_slack/; min n_eff table in SECTION_4_VALIDITY.md |
| 5 | RAVEL failure documentation (esol/freesolv/tox21/toxcast -> c_final=0.0) | DONE (Session 4-5) | scripts/diagnose_ravel_failures.py; scaffold shift -> extreme weights -> abstain |

**All 5 CHANGES.md deliverables: DONE.**

---

### PI_REPORT_FOR_REVIEW.md: 5 Priority Actions

| # | Priority | Action Required | Status |
|---|----------|----------------|--------|
| 1 | **Priority 1 (High risk)** | Validate Finding 3 (WCP vs EB) on all datasets | **DONE (Session 7)** -- generalises; COMPAS 7x, Adult 4x, text converges at high n_eff |
| 2 | **Priority 2 (Medium risk)** | Run ALL 6 methods on ALL 23 datasets | **PARTIAL** -- uLSIF/KLIEP/WCP all 6; RAVEL molecular only |
| 3 | **Priority 3 (Low risk)** | Add 27 more datasets (target 40+) | **NOT DONE** |
| 4 | **Priority 4 (Medium risk)** | Regression cert_rate ~ domain + cohort_size + shift | **DONE** -- H3 regression (R2 decomposition done) |
| 5 | **Priority 5 (High risk)** | Write 8-page paper draft | **PARTIAL** -- Sections 4 and 5 written. Intro, Method, Related Work, Conclusion missing. |

---
