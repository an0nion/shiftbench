# Section 2: Related Work

## 2.1 Distribution Shift Benchmarks

Prior shift benchmarks focus on *training* under shift, not *evaluation* of a fixed
model. DomainBed [Gulrajani & Lopez-Paz, 2021] benchmarks domain adaptation algorithms
(ERM, CORAL, IRM) across 7 image datasets with domain labels. WILDS [Koh et al., 2021]
provides 8 real-world shift datasets for training robust models, measuring in-distribution
and out-of-distribution accuracy. Shifts [Malinin et al., 2021] targets weather, vehicle,
and clinical tabular shifts for robustness evaluation. These benchmarks answer: *which
training algorithm is most robust?* ShiftBench answers a complementary question: *how
should we evaluate a fixed trained model when test distribution differs from calibration?*

The closest related work is WILDS [Koh et al., 2021], which provides real covariate
shift across multiple domains. However, WILDS does not provide an evaluation harness
for certifying performance metrics post-hoc, nor does it compare IW estimation methods
systematically.

## 2.2 Importance Weighting Methods

Covariate shift was formalized by Shimodaira [2000], who showed that IID-trained models
can be corrected by reweighting. Kernel Mean Matching (KMM) [Huang et al., 2007]
minimizes an RKHS discrepancy. KLIEP [Sugiyama et al., 2008] minimizes KL divergence.
uLSIF [Kanamori et al., 2009] minimizes squared loss—faster and often more stable.
RULSIF [Yamada et al., 2013] estimates relative density ratios, providing robustness
to extreme ratios. RAVEL [Federici et al., 2021] uses a discriminative classifier with
PSIS k-hat [Vehtari et al., 2015, 2017] and ESS stability gating.

Our benchmark is the first to compare all five methods head-to-head across 25 datasets
with a unified evaluation protocol and explicit stability diagnostics.

## 2.3 Conformal Prediction Under Shift

Conformal prediction [Vovk et al., 2005] provides distribution-free coverage guarantees.
Weighted conformal prediction (WCP) [Tibshirani et al., 2019] extends coverage to
covariate shift via importance weights. CV+ [Barber et al., 2021] improves efficiency
via cross-conformal aggregation. Split conformal [Lei et al., 2018] is a computationally
efficient baseline. These methods provide marginal coverage (P(Y in C(X)) >= 1-alpha)
rather than concentration bounds on specific metrics like PPV.

Our benchmark is the first to compare conformal methods (WCP, CV+, Split Conformal)
to IW-based concentration bounds (EB+n_eff) for the specific task of PPV certification
under covariate shift.

## 2.4 Empirical Bernstein Bounds and FWER Control

Empirical Bernstein bounds [Maurer & Pontil, 2009] provide tight concentration
inequalities for bounded random variables that exploit observed variance. Holm's
step-down procedure [Holm, 1979] controls FWER for multiple simultaneous hypotheses.
The combination of n_eff substitution and Holm correction in ShiftBench follows the
heuristic proposed in RAVEL [Federici et al., 2021], which we validate empirically
across 13,600 trials.

## 2.5 Certification and Performance Verification

Recent work on performance verification under shift [Garg et al., 2022; Ginart et al.,
2022] studies when model performance can be reliably estimated. Our approach differs in
providing *cohort-level* certificates (certify/abstain per group) rather than
global performance bounds, enabling fairness-critical applications where per-group PPV
matters independently.
