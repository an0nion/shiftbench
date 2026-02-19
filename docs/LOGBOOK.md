# ShiftBench Research Logbook

---

## Session 6 — 2026-02-19

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

## PI ISSUES AUDIT — Full Reconciliation

Sources: CHANGES.md (PI feedback fixes), PI_REPORT_FOR_REVIEW.md (priority actions).

### CHANGES.md: 5 Deliverables

| # | Issue | Status | Evidence |
|---|-------|--------|----------|
| 1 | H1 tabular active pairs (experiment_c used n_cohorts=5, no active pairs) | DONE (Session 5) | results/h1_disagreement/; COMPAS 100% (17 pairs), Adult 85.7% (7 pairs) |
| 2 | H2 Wilson CI power analysis (500 trials insufficient, CI_upper=0.073 > 0.06) | DONE (Session 5) | results/h2_good_weights_500/h2_power_analysis.csv; need ~10,000 trials for 90% power |
| 3 | H3 regression rewrite (oracle filter, partial R², within-domain) | DONE (Session 4-5) | results/h3_regression/; Domain R2=0.994, n_eff R2=0.708, partial R2=0.002 |
| 4 | H4 slack by n_eff + minimum n_eff table | DONE (Session 4-5) | results/h4_slack/; min n_eff table in SECTION_4_VALIDITY.md |
| 5 | RAVEL failure documentation (esol/freesolv/tox21/toxcast -> c_final=0.0) | DONE (Session 4-5) | scripts/diagnose_ravel_failures.py; scaffold shift -> extreme weights -> abstain |

**All 5 CHANGES.md deliverables: DONE.**

---

### PI_REPORT_FOR_REVIEW.md: 5 Priority Actions

| # | Priority | Action Required | Status |
|---|----------|----------------|--------|
| 1 | **Priority 1 (High risk)** | Validate Finding 3 (WCP vs EB) on all 23 datasets. Determine if 6.5x improvement generalises or is cherry-picked. | **NOT DONE** |
| 2 | **Priority 2 (Medium risk)** | Run ALL 6 methods on ALL 23 datasets (full benchmark completion) | **PARTIALLY DONE** — uLSIF done on all; KLIEP done on 6 (+ H1); RAVEL done on molecular only. Text/tabular RAVEL pending. |
| 3 | **Priority 3 (Low risk)** | Add 27 more datasets (target 40+ total) | **NOT DONE** — still at ~10 Tier-A datasets |
| 4 | **Priority 4 (Medium risk)** | Regression analysis of cert_rate ~ domain + cohort_size + shift_magnitude | **DONE** — H3 regression complete (R2 decomposition done) |
| 5 | **Priority 5 (High risk)** | Write 8-page paper draft | **PARTIAL** — Sections 4 and 5 written. Intro, Method, Related Work, Conclusion missing. |

### PI Critical Questions (PI_REPORT)

| Q | Question | Resolution |
|---|----------|-----------|
| Q1 | Is Finding 3 (WCP 6.5x) oversold? | UNRESOLVED — WCP not tested beyond original BACE comparison |
| Q2 | Is 100% KLIEP-uLSIF agreement too good? | RESOLVED — mechanism found: near-identical weights (neff_ratio=1.003, mu_gap=0.010). BBBP exception explained. |
| Q3 | Minimum scope to submit? | UNRESOLVED — still at ~10 datasets, need 40 |
| Q4 | NeurIPS D&B or elsewhere? | UNRESOLVED — scope decision blocked by Q3 |

---

### Summary: What Remains Unresolved from PI

**Critical (must do before submission):**
1. WCP vs EB validation on all datasets (Priority 1)
2. Add 27+ datasets to reach 40 (Priority 3)
3. Run RAVEL on text/tabular datasets (Priority 2 partial)
4. Paper sections: Intro, Method, Related Work, Conclusion

**Important (should do):**
5. Finding 5 (method ranking by domain) — requires RAVEL + WCP on all domains
6. H4 stronger semi-synthetic: set true_ppv below each tau independently
7. H2 Wilson CI: run 10,000 trials to satisfy CI criterion

**Complete:**
- All CHANGES.md deliverables (5/5)
- All Session 6 user requests (9/9)
- H1-H4 ablation suite (ablation_h1-h4 scripts)
- KLIEPFast implementation
- SECTION_4 and SECTION_5 paper drafts
- Binarization sensitivity
- FORMAL_CLAIMS Section 6

---
