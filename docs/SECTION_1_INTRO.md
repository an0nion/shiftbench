# Section 1: Introduction

## 1.1 The Evaluation Problem Under Distribution Shift

Machine learning models are increasingly deployed in high-stakes settings where the
test distribution differs from training: drug discovery models evaluated on new
chemical scaffolds, recidivism predictors applied across demographic groups not
seen during calibration, and sentiment classifiers deployed across temporal periods.
In these settings, standard IID evaluation—held-out test performance computed under
the same distribution as training—provides a misleading picture of deployment behavior.

The standard response to distribution shift is importance weighting (IW): reweight
calibration examples by the estimated density ratio w(x) = p_test(x)/p_cal(x), then
re-estimate performance metrics under the reweighted distribution. A second line of
work uses conformal prediction to obtain distribution-free coverage guarantees.
Both families have been extensively studied in theory, but practitioners face an
unanswered question: *which method should I use, and does the choice matter?*

## 1.2 The Gap: No Systematic Benchmark Exists

Existing shift benchmarks focus on *training* robustness—DomainBed [Gulrajani & Lopez-Paz,
2021] and WILDS [Koh et al., 2021] compare algorithms for learning under shift, not
evaluation methods applied to a fixed model. The specific problem of *certifying
model performance under covariate shift*, with rigorous error control, has not been
benchmarked systematically across domains, shift types, and scales.

This gap matters for three reasons. First, IW methods are known to be unstable under
heavy-tailed weights, but the severity of instability across real datasets is
uncharacterized. Second, stability diagnostics (PSIS k-hat, ESS) have been proposed
but their necessity—and the relative importance of each gate—has not been empirically
validated. Third, conformal methods provide different guarantees (marginal coverage vs.
concentration bounds) and their practical tradeoffs versus IW methods are unknown.

## 1.3 ShiftBench

We introduce **ShiftBench**, a benchmark for shift-aware model evaluation. ShiftBench
provides:

1. **40 curated datasets** spanning molecular (Murcko scaffold shift), tabular
   (demographic and temporal shift), and text (temporal, geographic, and category
   shift) domains, covering six distinct shift types and sample sizes ranging from
   300 to 93,000.

2. **10 baseline methods** implementing the major families: density ratio estimation
   (uLSIF, KLIEP, KMM, RULSIF, RAVEL), weighted conformal prediction (WCP, Split
   Conformal, CV+), and distributionally robust approaches (Group DRO, BBSE).

3. **A certify-or-abstain evaluation harness** with Holm step-down FWER control,
   empirical Bernstein bounds, and stability gating (PSIS k-hat, ESS). The harness
   emits reproducible audit receipts linking decisions to weights and bounds.

4. **Pre-registered hypotheses and falsification criteria** (FORMAL_CLAIMS.md),
   enabling transparent reporting of both confirmed and rejected predictions.

## 1.4 Key Findings

Our benchmark reveals four findings that challenge or refine conventional wisdom:

**Finding 1 — Density ratio choice is practically irrelevant for certification.**
Under empirical Bernstein bounds with Holm correction, KLIEP (KL minimization)
and uLSIF (L2 minimization) produce identical certify/abstain decisions on 5 of 6
datasets (100% agreement). On the 6th (BBBP, scaffold shift), 88.4% agreement with
disagreements confined to the tau=0.6–0.7 borderline. The mechanism is weight
convergence: mean n_eff ratio = 1.003 across all 4,350 (cohort, tau) pairs.

**Finding 2 — Stability gating is necessary, and k-hat is the critical gate.**
Without n_eff correction, FWER escalates to 42% at log-normal sigma=3.0 (8x nominal).
Among the three stability gates (k-hat, ESS, clip-mass), k-hat is dominant: removing
it inflates certification rate 3.4x at sigma=1.5 by allowing high-mu_hat certifications
with unreliable weights.

**Finding 3 — WCP dominates IW methods at n_eff < 300.**
Weighted conformal prediction provides 1.7–7x more certifications than EB bounds
across molecular and tabular datasets (COMPAS: 7x, BBBP: 1.7x). The advantage
disappears at very high n_eff (IMDB: n_eff=40,000), where both bound families converge.

**Finding 4 — n_eff is the primary mechanistic predictor of certification difficulty.**
Expanded regression on 27 datasets shows n_eff (R²=0.645) outperforms domain alone
(R²=0.504) as a cross-domain predictor. The earlier finding of "domain R²=0.994"
was an artifact of only 6 maximally-separated data points. With 27 datasets, partial
R²(n_eff | domain) = 0.406, confirming n_eff retains strong independent explanatory
power within every domain (molecular: R²=0.525, tabular: R²=0.386, text: R²=0.438).
Domain structure mediates n_eff (scaffold shift → extreme weights → low n_eff) but
does not subsume it.

## 1.5 Paper Organization

Section 2 reviews related work on distribution shift evaluation and benchmarking.
Section 3 describes ShiftBench's design and evaluation protocol. Section 4 presents
the validity study (H4: empirical FWER control). Section 5 presents the real-data
evaluation (H1–H3 findings and the WCP comparison). Section 6 discusses limitations
and future work.
