# Section 7: Conclusion

## 7.1 Summary

We introduced ShiftBench, a benchmark for shift-aware model evaluation comprising
40 datasets across molecular, tabular, and text domains; 10 baseline methods
spanning density ratio estimation, conformal prediction, and distributionally
robust optimisation; and a reproducible evaluation harness with empirical
Bernstein bounds, Holm FWER control, stability gating, and hash-chained audit
receipts.

Our systematic evaluation yields four findings:

1. **Density ratio estimator choice is practically irrelevant.** KLIEP and uLSIF
   produce identical certify/abstain decisions on 5/6 Tier-A datasets. Practitioners
   can use uLSIF (7–16× faster) without sacrificing decision quality.

2. **Stability gating is necessary; k-hat is the critical gate.** Without n_eff
   correction, FWER reaches 42% under heavy-tailed weights (8× nominal). Removing
   only k-hat inflates certification 3.4× at moderate shift severity.

3. **WCP dominates EB at n_eff < 300; the advantage is n_eff-mediated.**
   WCP certifies 2–7× more cohorts in molecular and tabular domains, where
   small effective sample sizes make the sub-Gaussian variance penalty of EB
   prohibitive. At high n_eff (text), both methods converge.

4. **n_eff is the primary mechanistic predictor of certification difficulty.**
   Across 27 datasets, n_eff (R²=0.645) outperforms domain (R²=0.504) as a
   predictor. Within all three domains, the n_eff coefficient is positive (+0.058
   to +0.108), confirming a consistent causal mechanism.

## 7.2 Limitations

**Oracle predictions are not used.** All results use real trained model predictions
(RF for molecular, LR for tabular/text), but model quality varies. Low-AUC models
(Tox21, ToxCast: AUC≈0.5) dominate zero-certification outcomes regardless of shift.
A limitation of our design is that model quality and shift severity are confounded
for some datasets.

**Method comparison is not always apples-to-apples.** WCP and EB have different
statistical guarantees (marginal coverage vs. concentration bounds). The practical
comparison is informative but theoretically heterogeneous.

**H4 semi-synthetic is conservative.** The null evaluation fixes true_ppv below
tau=0.9 only. A stronger test would set true_ppv below each τ independently for
each (cohort, τ) pair. This is documented as a known limitation and left for future
validation.

**RAVEL on molecular abstains fully.** On all tested molecular datasets, RAVEL's
k-hat gate fires and returns NO-GUARANTEE. This is the correct certify-or-abstain
behaviour under extreme scaffold shift — but it means RAVEL provides no power
data for the molecular domain.

## 7.3 Future Work

1. **Extended method matrix.** Running all 10 methods on all 40 datasets will enable
   Finding 5 (method rankings by domain) to be fully supported with data.
2. **H4 stronger null.** Setting true_ppv below each τ independently for each cohort
   will provide more rigorous FWER validation at high tau thresholds.
3. **Real model diversity.** Including neural networks, gradient-boosted trees,
   and pre-trained transformers will test whether findings generalise beyond RF/LR.
4. **Interactive leaderboard.** A public leaderboard accepting community-contributed
   methods and datasets would accelerate research on shift-aware evaluation.

## 7.4 Ethical Considerations

ShiftBench includes datasets with protected-attribute cohorts (race, sex, age):
Adult, COMPAS, German Credit, Civil Comments. These enable fairness-critical
evaluation but carry dual-use risk: a benchmark certifying high PPV for some
demographic groups may inadvertently legitimise models that harm others. We
emphasise that statistical certification (lower bound on PPV ≥ τ) is a necessary
but not sufficient condition for deployment — practitioners must assess disparate
impact separately and ensure compliance with applicable law.
