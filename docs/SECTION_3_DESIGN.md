# Section 3: ShiftBench Design and Evaluation Protocol

## 3.1 Design Principles

ShiftBench is designed around three principles: (1) **ecological validity** — datasets
reflect real deployment shifts, not synthetic or contrived scenarios; (2)
**reproducibility** — fixed seeds, deterministic splits, and hash-chained audit
receipts; and (3) **domain coverage** — molecular, tabular, and text domains span
the major application areas where covariate shift is practically consequential.

We focus on **covariate shift** (p(X) changes, p(Y|X) stable), the most
well-studied and practically actionable shift type. The certify-or-abstain framing
is central: methods must either provide a statistically guaranteed lower bound on
PPV for a cohort, or explicitly abstain — preventing metric hallucination when
weights are unreliable.

## 3.2 Dataset Selection and Cohort Definitions

We select 40 datasets across three domains based on: (1) availability of a natural
shift structure (scaffold, demographic, temporal, geographic, or category cohorts);
(2) binary classification task (regression datasets are binarized via training-set
median); and (3) public license permitting redistribution.

Cohort definitions vary by domain:
- **Molecular**: Murcko scaffold groups — each scaffold is a cohort. Scaffold shift
  is severe; test scaffolds are chemically novel relative to calibration.
- **Tabular**: Demographic bins (race × sex × age for fairness datasets; temporal
  bins for bank/shoppers). Shift is moderate.
- **Text**: Topic, category, or temporal bins (decades for IMDB; cities for Yelp;
  topic categories for AG News, DBpedia). Shift is mild; overlapping vocabulary
  produces high n_eff.

## 3.3 Evaluation Protocol

**Splits.** Each dataset is partitioned 60/20/20 (train/cal/test) with seed=42.
The train split is reserved for future training-under-shift experiments; ShiftBench
evaluates **post-hoc** using calibration and test only.

**Models.** Random forest (RF) for molecular datasets (217-dim RDKit features);
logistic regression (LR) with TF-IDF features (5,000-dim) for text; LR for tabular.
All models are pre-trained on the train split; predictions on cal and test are saved
to `models/predictions/` and loaded at evaluation time.

**Metrics.** Primary metric: **certification rate** — the fraction of
(cohort, τ) pairs where a lower bound on PPV ≥ τ is returned with a statistical
guarantee. Secondary metrics: mean lower bound, mean n_eff, runtime.

**Thresholds.** τ ∈ {0.5, 0.6, 0.7, 0.8, 0.9} evaluated simultaneously.

**Error control.** FWER controlled at α=0.05 via Holm's step-down procedure
across all (cohort, τ) pairs per dataset. Each method either certifies
(lower bound ≥ τ with FWER guarantee), abstains (insufficient evidence), or
returns NO-GUARANTEE (stability gates fail — weights too unreliable to use).

## 3.4 Certify-or-Abstain with Empirical Bernstein Bounds

For IW-based methods, we estimate a lower confidence bound on cohort PPV via
empirical Bernstein (EB) bounds [Maurer & Pontil, 2009]. Given importance weights
w_i and binary outcomes z_i = 1[f(x_i) = y_i]:

    μ̂ = (1/N) Σ w_i z_i
    LB_EB(μ̂, σ̂², n_eff, α) = μ̂ - sqrt(2σ̂² log(2/α)/n_eff) - 7 log(2/α)/(3 n_eff)

where n_eff = (Σ w_i)² / Σ w_i² is the effective sample size substituted for
the raw sample count. This substitution is the key stability mechanism: under
heavy-tailed weights, n_eff ≪ N, automatically widening the bound and preventing
false certifications. We validate FWER control empirically across 13,600 trials
(Section 4).

## 3.5 Stability Gating

RAVEL [Federici et al., 2021] applies three diagnostic gates before computing bounds:
- **PSIS k-hat ≤ 0.7**: Pareto tail index of the weight distribution. k > 0.7
  indicates the weighted estimator has infinite variance; certification is blocked.
- **ESS/N ≥ 0.3**: If fewer than 30% of calibration samples are effective,
  bounds are too wide to certify at any practical τ.
- **Clip mass ≤ 0.1**: Weight mass beyond the 99th percentile clip point.

Gate failure returns NO-GUARANTEE rather than a potentially invalid bound. We
show in Section 4 that k-hat is the dominant gate: removing it inflates
certification rate 3.4× at moderate shift severity.

## 3.6 Reproducibility: Hash-Chained Receipts

Each evaluation run emits a **receipt** binding dataset identity, method config,
weight vector hash, bound values, and the final certify/abstain/no-guarantee
decision. Receipts are chained: hash(receipt_t) depends on hash(receipt_{t-1}),
preventing selective reporting. All results in this paper are accompanied by
verified receipts in `results/`.
