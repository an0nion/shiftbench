# ShiftBench — PI Feedback Fixes

Addresses all five deliverables raised in PI feedback session.

---

## H1 — Tabular Active Pairs

**Problem:** `experiment_c` used `n_cohorts=5` for adult/COMPAS, yielding 0 active
(cohort, tau) pairs and no H1 evidence for the tabular domain.

**Fix:** New script `scripts/analysis_h1_supplement_crossdomain.py` extracts H1
evidence from the existing `cross_domain_real` run, which uses the natural cohort
structure (38 cohorts for COMPAS, 49 for adult). KLIEP and uLSIF are run on
identical data, so cert-rate agreement is a valid H1 test.

**Results (from `results/h1_disagreement/`):**
- `h1_crossdomain_supplement.csv` — per-dataset cert-rate agreement
- `h1_crossdomain_by_neff.csv` — binned by n_eff
- `h1_combined_by_neff.csv` — experiment_c + cross_domain merged

| Dataset | n_eff bin | Active pairs | Disagree (LB) | Agree (LB) |
|---------|-----------|-------------|---------------|------------|
| compas  | 5–25      | 17          | 0             | 100%       |
| adult   | 25–100    | 7           | 1             | 85.7%      |

Note: `experiment_c_tabular_regime.py` is ready to run but requires `data/processed/`
(not tracked). When available, `analysis_h1_disagreement.py` will automatically
prefer the regime data over the cross_domain supplement.

---

## H2 — Wilson CI Power Analysis

**Problem:** PI criterion: Wilson CI upper ≤ α+0.01 = 0.06. The 500-trial
`good_weights` run gives CI upper = 0.0728 — marked `ci_upper_within_tolerance=False`.
The existing analysis didn't explain *why*.

**Fix:** New script `scripts/analysis_h2_ci_power.py` simulates the distribution
of Wilson CI upper bounds as a function of n_trials.

**Results (from `results/h2_good_weights_500/`):**
- `h2_power_analysis.csv` — E[CI_upper] and P(CI_upper < 0.06) by n_trials
- `h2_ci_power_analysis.csv` — per-variant interpretation

| n_trials | E[CI_upper] | P(CI_upper < 0.06) |
|----------|-------------|---------------------|
| 500      | 0.073       | 12.6%               |
| 2000     | 0.060       | 49.1%               |
| 5000     | 0.056       | 87.4%               |
| ~10 000  | —           | ~90% (criterion)    |

Root cause: statistical power, not calibration. 500 trials is insufficient to
satisfy the CI criterion in the majority of runs when true FWER = alpha = 5%.

---

## H3 — Regression (Oracle Filter + Partial R²)

**Problem:** Full regression had `log_neff` coefficient = −0.03 (sign-flipped).
Root cause: amazon, civil_comments, twitter all used oracle predictions (cert_rate=1.0)
and dominated the text domain, making n_eff and domain collinear.

**Fix:** `scripts/analysis_h3_regression.py` rewritten.
- Added `REAL_PRED_DATASETS` allowlist (10 datasets, no oracle fallbacks)
- Loads from `cross_domain_tier_a/cross_domain_by_dataset.csv` (present) instead
  of raw results (gitignored)
- Added incremental/partial R² computation
- Added within-domain regression

**Results (from `results/h3_regression/`):**
- `h3_partial_r2.csv` — global partial R² table
- `h3_within_domain_regression.csv` — per-domain n_eff regression
- Updated `regression_results.csv`, `regression_data.csv`, `scatter_data.csv`

| Model | R² |
|-------|----|
| Domain alone | 0.994 |
| n_eff alone | 0.708 |
| Partial R²(n_eff \| domain) | 0.002 |
| Within-molecular R²(n_eff) | 0.669 (coef = +0.011, correct sign) |

Narrative correction: domain is the primary predictor; n_eff explains
within-domain variation, not cross-domain.

---

## H4 — Slack by n_eff Bin + Minimum n_eff Table

**Problem:** [1–5] and [5–25] n_eff bins were empty with no explanation.
`300+` bin had `false_cert_rate=0.026` with no context.

**Fix:** `scripts/analysis_h4_slack.py` updated.
- Added 3 new low-n_cal simulation configs to populate low-n_eff range
- Added `compute_minimum_neff_for_certification()` analytical function
- Fixed `groupby(observed=False)` to emit all bins (even empty)
- Added note distinguishing per-decision `false_cert_rate` vs per-trial FWER

**Results (from `results/h4_slack/`):**
- `h4_minimum_neff_required.csv` — minimum n_eff for certification by (tau, PPV)
- Updated `h4_slack_by_neff.csv`, `h4_slack_by_config.csv`

| n_eff bin | Certified | false_cert_rate | Explanation |
|-----------|-----------|-----------------|-------------|
| 1–5       | 0         | —               | Analytical min n_eff ≈ 4 at PPV=0.9/tau=0.5; actual EB more conservative |
| 5–25      | 0         | —               | Same; EB conservatism requires PPV very close to 1.0 |
| 25–100    | 957       | 0.000           | |
| 100–300   | 3591      | 0.001           | |
| 300+      | 349       | 0.026           | Per-decision rate; per-trial FWER still ≤ alpha by construction |

---

## RAVEL Failures — esol / freesolv / tox21 / toxcast

**Problem:** RAVEL returns `c_final=0.0` on these molecular datasets. Needed
documentation of root cause.

**Fix:** New script `scripts/diagnose_ravel_failures.py` documents the failure mode.

Root cause: scaffold-based covariate shift is extreme (n_eff ≈ 1–2 after reweighting).
RAVEL's discriminative classifier achieves high AUC on scaffold split → extreme weights
→ all weight mass must be clipped → `c_final=0.0`. This is the intended
**certify-or-abstain** guarantee: RAVEL refuses to certify when it cannot produce
reliable weights. Correct behavior, not a bug.

---

## Files Changed

### Scripts (modified)
- `scripts/analysis_h1_disagreement.py` — calls supplement; handles missing tabular regime data
- `scripts/analysis_h3_regression.py` — full rewrite (oracle filter, partial R², within-domain)
- `scripts/analysis_h4_slack.py` — low-n_cal configs, min-n_eff table, empty bin fix

### Scripts (new)
- `scripts/analysis_h1_supplement_crossdomain.py`
- `scripts/analysis_h2_ci_power.py`
- `scripts/diagnose_ravel_failures.py`

### Results (new)
- `results/h1_disagreement/h1_crossdomain_supplement.csv`
- `results/h1_disagreement/h1_crossdomain_by_neff.csv`
- `results/h1_disagreement/h1_combined_by_neff.csv`
- `results/h2_good_weights_500/h2_power_analysis.csv`
- `results/h2_good_weights_500/h2_ci_power_analysis.csv`
- `results/h3_regression/h3_partial_r2.csv`
- `results/h3_regression/h3_within_domain_regression.csv`
- `results/h4_slack/h4_minimum_neff_required.csv`

### Results (updated)
- `results/h3_regression/regression_results.csv`
- `results/h3_regression/regression_data.csv`
- `results/h3_regression/scatter_data.csv`
- `results/h4_slack/h4_slack_by_neff.csv`
