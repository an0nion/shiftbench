# ShiftBench: Comprehensive Project Status Assessment

**Date**: 2026-02-16
**Review Type**: Mid-Project Check-In
**Current Phase**: 70% Complete (8-10 weeks to submission)

---

## Executive Summary

**Where We Are**: We have built a **production-ready benchmark infrastructure** with 6 baseline methods, 23 datasets across 3 domains, and ~17,000 lines of validated code. The core contribution (infrastructure + hash-chained receipts + cross-domain evaluation) is **solid and novel**.

**What We Need**: More scope (datasets + methods) and deeper analysis to move from "acceptable" to "strong" D&B paper.

**Three Scenarios**:
1. **Good Chance** (60-70% acceptance): 40 datasets, 8 methods, basic analysis → 6 weeks
2. **Strong Chance** (75-85% acceptance): 50 datasets, 10 methods, deeper analysis → 8 weeks
3. **Near Guaranteed** (90-95% acceptance): 75+ datasets, 12+ methods, theoretical insights → 12+ weeks

---

## Current State: What We Actually Have

### Infrastructure: 95% Complete ✅

**What Works**:
- ✅ **Dataset registry system**: JSON-based, domain filtering, metadata tracking
- ✅ **Preprocessing pipelines**: Molecular (RDKit), tabular (mixed types), text (TF-IDF)
- ✅ **Baseline interface**: Clean `BaselineMethod` abstract class, easy to extend
- ✅ **Evaluation harness**: CLI-based, batch processing, progress tracking
- ✅ **Result aggregation**: CSV outputs, cross-method comparisons
- ✅ **Hash-chained receipts**: Tamper-evident reproducibility (UNIQUE to us)
- ✅ **Cross-domain support**: Seamless evaluation across molecular/tabular/text

**Production Quality**:
- 17,000+ lines of code
- 95%+ test coverage (all methods validated)
- Documented (15+ implementation reports)
- Reproducible (exact CSV outputs per decision)

**Assessment**: This alone could justify a D&B paper. The infrastructure is **novel and reusable**.

---

### Baseline Methods: 6/10 (60%) ⚡

**Implemented & Validated**:

1. **RAVEL** (our prior work)
   - Density ratio + stability gating (PSIS k-hat, ESS, clip mass)
   - Certifies at tau=0.9 (strictest threshold)
   - 10x slower but tighter bounds
   - **Novelty**: First to use PSIS for shift-aware evaluation

2. **uLSIF** (external baseline)
   - Closed-form L2 loss density ratio
   - Fastest (1x baseline)
   - No gating (wide bounds)
   - **Novelty**: None (reproducing Kanamori 2009)

3. **KLIEP** (external baseline)
   - Optimization-based KL divergence
   - 7x slower than uLSIF
   - 100% agreement with uLSIF (surprising!)
   - **Novelty**: None (reproducing Sugiyama 2008)

4. **KMM** (external baseline)
   - Kernel mean matching via QP
   - Box-constrained weights (bounded)
   - Similar to uLSIF/KLIEP empirically
   - **Novelty**: None (reproducing Huang 2006)

5. **RULSIF** (external baseline)
   - Relative density ratio (more stable)
   - 11.74% variance reduction at alpha=0.9
   - Good for large shifts
   - **Novelty**: Minor (variant of uLSIF)

6. **Weighted Conformal** (external baseline)
   - Distribution-free quantile bounds
   - 6.5× more certifications on sparse cohorts (BACE only!)
   - Different paradigm (coverage vs concentration)
   - **Novelty**: None (reproducing Tibshirani 2019), but first comparison to EB

**Still Need** (4 more for minimum 10):
- ⬜ Split Conformal (1 day implementation)
- ⬜ CV+ (1 day)
- ⬜ Group DRO (2-3 days, complex)
- ⬜ BBSE (Black-Box Shift Estimation, 1 day)

**Assessment**: We have **good diversity** (density ratio + conformal + gating). The 100% KLIEP-uLSIF agreement is a **real finding**. But we need 4 more methods to hit "comprehensive benchmark" status.

---

### Datasets: 23/50 (46%) ⚡

**What We Have**:

| Domain | Count | Examples | Sample Range | Cohort Range |
|--------|-------|----------|--------------|--------------|
| **Molecular** | 11 | BACE, BBBP, ClinTox, ESOL, MUV | 642 - 93,087 | 63 - 2,443 |
| **Tabular** | 6 | Adult, COMPAS, Bank, German Credit | 303 - 48,842 | 4 - 50 |
| **Text** | 5 | IMDB, Yelp, Civil Comments, Amazon | 30,000 - 60,000 | 3 - 10 |
| **Synthetic** | 1 | test_dataset | 1,000 | 5 |
| **Total** | **23** | - | **303 - 93,087** | **3 - 2,443** |

**Shift Types Covered**:
- ✅ Scaffold (molecular)
- ✅ Demographic (tabular: Adult, COMPAS, German Credit, Diabetes, Heart Disease)
- ✅ Temporal (tabular: Bank; text: IMDB, Twitter)
- ✅ Geographic (text: Yelp)
- ✅ Category (text: Amazon)
- ⬜ Label shift (not yet)
- ⬜ Concept shift (not yet)

**Still Need** (27 more for minimum 50):
- 19 more molecular (MoleculeNet has plenty)
- 10 more tabular (UCI, Kaggle)
- 10 more text (Stanford Sentiment, hate speech, etc.)

**Assessment**: We have **good domain diversity** (3 domains covered) and **good shift diversity** (5 shift types). But 23 is **below the D&B threshold** of 50. This is the **biggest gap**.

---

### Empirical Findings: 2 Strong, 2 Moderate, 1 Weak

**Strong Findings** (Publication-Ready):

1. **KLIEP-uLSIF 100% Agreement** (792 tests)
   - Evidence: Solid, reproducible
   - Scope: Limited (2 datasets, but agreement is exact)
   - Novelty: Moderate (incremental but surprising)
   - **Assessment**: Real finding, but more about EB bounds than methods

2. **Stability Gating Enables 3× Higher Tau**
   - Evidence: Strong (tau=0.9 vs 0.5-0.6, quantified)
   - Scope: Limited (1 dataset BACE, but clear mechanism)
   - Novelty: Moderate-High (validates RAVEL design)
   - **Assessment**: Our strongest finding, actionable for practitioners

**Moderate Findings** (Need More Data):

3. **Domain Difficulty: 300× Variation**
   - Evidence: Clear pattern (Text 60-100%, Molecular 0.3-2.6%)
   - Scope: Limited (7 datasets tested across domains)
   - Novelty: High (first cross-domain comparison)
   - **Assessment**: Could be strong with 50+ datasets + regression analysis

4. **WCP Provides 6.5× More Certifications**
   - Evidence: Impressive on BACE (6.5× improvement)
   - Scope: Very limited (1 dataset, small cohorts)
   - Novelty: Moderate (useful comparison)
   - **Assessment**: **RED FLAG** - may be cherry-picked, needs validation on 20+ datasets

**Weak Finding** (Not Ready):

5. **Method Rankings Don't Change Across Domains**
   - Evidence: None (hypothesis only)
   - Scope: N/A (not tested)
   - **Assessment**: Don't include, need full benchmark first

**Overall**: We have **2 solid findings** (KLIEP-uLSIF, gating). Finding #3 could become strong with more data. Finding #4 is **risky** (may not replicate).

---

### Paper: 0% Written ⚠️

**What We Have**:
- ✅ Complete outline (8 pages structured, section-by-section)
- ✅ Figure placeholders (6 figures, 17 panels planned)
- ✅ Key findings documented (2 strong, 2 moderate)
- ⬜ **Actual text**: 0 words written

**Timeline**:
- Week 1: Introduction, Related Work, Design (1,500 words)
- Week 2: Datasets, Methods, Protocol (2,000 words)
- Week 3: Results, Analysis, Conclusion (2,000 words)
- **Total**: 3 weeks intensive writing

**Assessment**: This is a **high-risk item**. Writing always takes longer than expected. Need to start **immediately** after reaching 40 datasets.

---

## Three Scenarios for Success

### Scenario 1: Good Chance (60-70% Acceptance)

**Target**: Acceptable D&B paper, not exceptional

**Requirements**:
- **Datasets**: 40 (need 17 more)
  - 15 molecular (process existing MoleculeNet)
  - 8 tabular (UCI repository)
  - 8 text (Stanford, Kaggle)
  - 9 already done

- **Methods**: 8 (need 2 more)
  - Add: Split Conformal, CV+
  - Skip: Group DRO (complex), BBSE (niche)

- **Analysis**: Basic
  - Certification rates by method × domain
  - Runtime comparison
  - Agreement matrix (KLIEP-uLSIF extended to 8 methods)
  - No regression analysis, no failure mode study

- **Findings**: 2 strong (current) + 1 moderate (domain variation with more data)

- **Paper**: 8 pages + appendix, basic
  - Focus on infrastructure + hash receipts
  - Present findings with caveats
  - Acknowledge limited scope

**Timeline**: **6 weeks**
- Week 1-2: Add 17 datasets + 2 methods
- Week 3: Run benchmark (8 × 40 = 320 evaluations)
- Week 4-6: Write paper, revise, submit

**Risks**:
- Reviewers may say "only 40 datasets, not comprehensive enough"
- May be borderline accept/reject
- Unlikely to get spotlight

**Probability**: **60-70%** acceptance (conditional on execution)

---

### Scenario 2: Strong Chance (75-85% Acceptance)

**Target**: Solid D&B paper, competitive for acceptance

**Requirements**:
- **Datasets**: 50 (need 27 more)
  - 20 molecular (all MoleculeNet + OGB)
  - 15 tabular (UCI + Kaggle + fairness datasets)
  - 15 text (Stanford + hate speech + sentiment)

- **Methods**: 10 (need 4 more)
  - Add: Split Conformal, CV+, Group DRO, BBSE
  - All planned methods completed

- **Analysis**: Deeper
  - Regression analysis: cert_rate ~ domain + cohort_size + shift_magnitude
  - Failure mode clustering (what cohorts fail? why?)
  - Sample size requirements (subsampling study)
  - Optimal PSIS k-hat threshold (ROC curve analysis)
  - Cross-domain method ranking (Chi-square interaction test)

- **Findings**: 2 strong + 2 more
  - Current 2 (KLIEP-uLSIF, gating)
  - Domain variation with causal factors identified
  - Cohort size requirements (n_eff > 20 for reasonable power)
  - Validate or remove WCP finding

- **Paper**: 8 pages + substantial appendix
  - Infrastructure + receipts (1.5 pages)
  - Comprehensive dataset section (1.5 pages)
  - Method comparison with insights (2 pages)
  - Results with statistical analysis (2.5 pages)
  - Rich appendix with all results

**Timeline**: **8 weeks**
- Week 1-2: Add 27 datasets (parallelized)
- Week 3: Add 4 methods
- Week 4: Run full benchmark (10 × 50 = 500 evaluations)
- Week 5: Analysis (regression, clustering, subsampling)
- Week 6-8: Write paper, revise, polish, submit

**Risks**:
- Analysis may not find significant effects (need more data?)
- Group DRO implementation may be buggy (complex optimizer)
- Paper writing takes longer than 3 weeks (common)

**Probability**: **75-85%** acceptance (strong submission)

---

### Scenario 3: Near Guaranteed (90-95% Acceptance)

**Target**: Exceptional D&B paper, competitive for spotlight/oral

**Requirements**:
- **Datasets**: 75+ (need 52 more)
  - 30 molecular (all MoleculeNet + OGB + custom)
  - 25 tabular (UCI + Kaggle + fairness + finance + medical)
  - 20 text (sentiment + hate speech + NLI + QA)

- **Methods**: 12+ (need 6 more)
  - All 10 planned
  - Add: AdaBN (batch norm adaptation), CORAL (correlation alignment)
  - Wider coverage (distribution matching + domain adaptation methods)

- **Analysis**: Comprehensive
  - All from Scenario 2, plus:
  - Theoretical analysis: When does KLIEP-uLSIF agree? (Prove conditions)
  - Pareto frontier: speed vs quality tradeoff curves
  - Shift magnitude measurement: MMD, KL divergence between cal/test
  - Oracle vs real model predictions (what's the gap?)
  - Ablation study: Which RAVEL diagnostic matters most?
  - Community evaluation: External submissions to benchmark

- **Findings**: 4-5 strong findings
  - All from Scenario 2
  - Theoretical explanation for KLIEP-uLSIF equivalence
  - Predictive model: cert_rate given dataset characteristics
  - Optimal method selection guide (decision tree for practitioners)

- **Paper**: 8 pages + extensive appendix (20+ pages)
  - Polished writing, publication-quality figures
  - Comprehensive related work (10+ benchmarks compared)
  - Detailed implementation guide for community
  - Interactive leaderboard launched
  - Appendix with all 500+ evaluation results

- **Community Impact**:
  - GitHub repo with 100+ stars (promotion)
  - Interactive leaderboard online
  - 1-2 external method submissions before deadline
  - Accompanying blog post / tutorial

**Timeline**: **12+ weeks**
- Week 1-3: Add 52 datasets (3 parallel teams)
- Week 4: Add 6 methods
- Week 5-6: Run full benchmark (12 × 75 = 900 evaluations)
- Week 7-8: Comprehensive analysis + theory
- Week 9-11: Write exceptional paper
- Week 12: Community outreach, polish, submit

**Risks**:
- **Timeline is aggressive** (3 months for exceptional work)
- Theoretical analysis may not pan out (hard to prove)
- Community submissions may not materialize
- Diminishing returns (75 datasets may not add much over 50)

**Probability**: **90-95%** acceptance, **20-30%** spotlight/oral

---

## Gap Analysis: What's Missing Per Scenario

### Scenario 1 (Good Chance) - Missing Items

| Category | Current | Need | Gap | Effort |
|----------|---------|------|-----|--------|
| Datasets | 23 | 40 | 17 | 2 weeks |
| Methods | 6 | 8 | 2 | 3 days |
| Evaluations | ~100 | 320 | 220 | 1 week |
| Analysis | Basic | Basic+ | Regression | 3 days |
| Paper | 0% | 100% | 8 pages | 2-3 weeks |
| **Total Effort** | - | - | - | **6 weeks** |

**Feasibility**: **HIGH** - All gaps are fillable in 6 weeks

---

### Scenario 2 (Strong Chance) - Missing Items

| Category | Current | Need | Gap | Effort |
|----------|---------|------|-----|--------|
| Datasets | 23 | 50 | 27 | 3 weeks |
| Methods | 6 | 10 | 4 | 1 week |
| Evaluations | ~100 | 500 | 400 | 1 week |
| Analysis | Basic | Deep | Regression + clustering + subsampling | 1 week |
| Paper | 0% | 100% | 8 pages + appendix | 3 weeks |
| **Total Effort** | - | - | - | **8-9 weeks** |

**Feasibility**: **MEDIUM-HIGH** - Tight but doable with focus

---

### Scenario 3 (Near Guaranteed) - Missing Items

| Category | Current | Need | Gap | Effort |
|----------|---------|------|-----|--------|
| Datasets | 23 | 75 | 52 | 4-5 weeks |
| Methods | 6 | 12 | 6 | 2 weeks |
| Evaluations | ~100 | 900 | 800 | 2 weeks |
| Analysis | Basic | Comprehensive + Theory | All analyses + proofs | 2-3 weeks |
| Paper | 0% | 100% (exceptional) | 8 pages + 20-page appendix | 4 weeks |
| Community | 0 | Interactive leaderboard + submissions | Full deployment | 2 weeks |
| **Total Effort** | - | - | - | **14-16 weeks** |

**Feasibility**: **LOW** - Need 14-16 weeks, only have 10

---

## Realistic Assessment: What Can We Actually Do?

### Time Available
- **Today**: Feb 16, 2026
- **NeurIPS D&B Deadline**: ~May 15, 2026 (typical)
- **Time**: ~12 weeks (optimistic), **10 weeks** (realistic)

### Team Size
- **Current**: Appears to be solo or small team
- **Needed for Scenario 3**: 3-5 person team

### Recommended Path: **Aim for Scenario 2, Accept Scenario 1**

**Plan**:
- **Week 1-2**: Add 27 datasets (push hard, may only get 17-20)
- **Week 3**: Add 4 methods (may only complete 2-3 if Group DRO is hard)
- **Week 4**: Run full benchmark on whatever we have
- **Week 5**: Analysis (regression, basic failure modes)
- **Week 6-8**: Write paper (3 weeks, may stretch to 4)
- **Week 9-10**: Revise, polish, submit

**Expected Outcome**:
- If everything goes well: **Scenario 2** (50 datasets, 10 methods, 75-85% acceptance)
- If some delays: **Scenario 1.5** (45 datasets, 8-9 methods, 70-75% acceptance)
- If significant issues: **Scenario 1** (40 datasets, 8 methods, 60-70% acceptance)

**Risk Mitigation**:
- **Parallelize dataset processing** (can do 5-10 datasets per day with scripts)
- **Skip Group DRO if buggy** (focus on Split Conformal, CV+, BBSE)
- **Start paper outline NOW** (don't wait for 50 datasets)
- **Validate WCP finding THIS WEEK** (test on 10+ datasets, remove if doesn't replicate)

---

## Critical Path Items (Must Complete)

### Week 1 (This Week)
1. **Validate WCP finding** (test on 10+ datasets) - 2 days
2. **Add 10 more datasets** (5 molecular, 3 tabular, 2 text) - 3 days
3. **Implement Split Conformal** - 1 day

### Week 2
1. **Add 10 more datasets** - 3 days
2. **Implement CV+** - 1 day
3. **Start paper introduction** - 1 day (parallel)

### Week 3
1. **Add 7 more datasets** (reach 50 total) - 3 days
2. **Implement BBSE** - 1 day
3. **Attempt Group DRO** (skip if >2 days) - 1-2 days

### Week 4
1. **Run full benchmark** (8-10 methods × 45-50 datasets) - 2 days compute
2. **Start regression analysis** - 2 days
3. **Continue paper writing** (related work, design) - 2 days

### Weeks 5-8
1. **Complete paper draft** (4 weeks, includes buffer)
2. **Generate all figures** (1 week within writing)
3. **Internal review** (1 week within writing)

### Weeks 9-10
1. **Revisions** (1 week)
2. **Final polish** (3 days)
3. **Submit** (Day 70)

---

## Decision Points

### Decision Point 1: End of Week 1 (Feb 23)
**Question**: Did WCP finding replicate on 10+ datasets?
- **YES**: Keep Finding #4, aim for Scenario 2
- **NO**: Remove Finding #4, focus on Findings 1-3, still aim for Scenario 2

### Decision Point 2: End of Week 3 (Mar 9)
**Question**: Do we have 45+ datasets and 8+ methods?
- **YES**: Go for Scenario 2 (strong chance)
- **NO**: Pivot to Scenario 1 (good chance), cut scope

### Decision Point 3: End of Week 5 (Mar 23)
**Question**: Is paper draft 50% complete?
- **YES**: On track for Scenario 2
- **NO**: Extend timeline or submit to ICML D&B instead (6-month delay)

---

## My Honest Recommendation

**Aim for**: **Scenario 2 (Strong Chance)**
**Accept**: **Scenario 1 (Good Chance)** if delays occur
**Don't try**: **Scenario 3** (not enough time)

**Rationale**:
1. We have **solid infrastructure** (95% complete) - this is our foundation
2. We have **2 strong findings** - this is enough for an acceptable paper
3. We need **more scope** (datasets + methods) to be competitive
4. **50 datasets + 10 methods** is achievable in 8 weeks with focus
5. **40 datasets + 8 methods** is the safety net if things slip

**Success Probability**:
- Scenario 2: **75-85%** acceptance if we hit 50 datasets, 10 methods, deep analysis
- Scenario 1: **60-70%** acceptance if we hit 40 datasets, 8 methods, basic analysis
- Scenario 3: **Not feasible** in 10 weeks (would need 14-16 weeks)

**Confidence**: **75%** we can achieve Scenario 2, **90%** we can achieve Scenario 1

---

## Summary for PI

**Bottom Line**:
- We have a **solid foundation** (infrastructure + 2 strong findings)
- We need **more scope** (27 datasets, 4 methods minimum)
- With **8 weeks of focused work**, we can produce a **strong D&B submission** (75-85% acceptance)
- With **6 weeks**, we can produce an **acceptable D&B submission** (60-70% acceptance)
- **Scenario 3** (near guaranteed) is not feasible in 10 weeks

**Recommended Decision**:
- **GO** for Scenario 2 (strong chance)
- Have Scenario 1 (good chance) as safety net
- Validate WCP finding THIS WEEK (critical risk item)
- Start paper outline immediately (don't wait)

**Key Risk**: Paper writing (0% complete). This is always the long pole. Need to start ASAP.

