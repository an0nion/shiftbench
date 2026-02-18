# Dataset Best Practices Research for ShiftBench

**Research Date:** 2026-02-16
**Purpose:** Identify commonly used datasets in distribution shift, domain adaptation, and covariate shift evaluation to strengthen ShiftBench for NeurIPS D&B submission

## Executive Summary

This research surveyed 5 major benchmark papers (DomainBed, WILDS, Shifts, OpenOOD, RobustBench) and multiple foundational papers in domain adaptation, covariate shift, conformal prediction, and fairness to identify dataset best practices. We identified **35+ commonly used datasets** across vision, NLP, and tabular domains.

**Key Findings:**
- ShiftBench currently has: 11 molecular + 7 tabular + 5 text = **23 datasets**
- High-priority missing datasets appearing in 3+ papers: ImageNet-C, Office-31, MNIST/SVHN/USPS, PACS, CivilComments (partial)
- Strong coverage of molecular and fairness/tabular domains
- Gap: Classic vision benchmarks (Office-31, PACS, VisDA) and ImageNet variants

---

## 1. Major Benchmark Papers Analysis

### 1.1 DomainBed (Gulrajani & Lopez-Paz, ICLR 2021)

**Paper:** "In Search of Lost Domain Generalization"
**Citation:** [Gulrajani & Lopez-Paz 2021](https://arxiv.org/abs/2007.01434)
**Focus:** Domain generalization for vision tasks

#### Datasets (5 core)
1. **VLCS** (Fang et al. 2013)
   - 4 domains: PASCAL VOC2007, LabelMe, Caltech, Sun
   - 5 classes: bird, car, chair, dog, person
   - ~10,000 images
   - **Status:** ❌ Missing from ShiftBench

2. **PACS** (Li et al. 2017)
   - 4 domains: Photo, Art, Cartoon, Sketch
   - 7 classes: dog, elephant, giraffe, guitar, horse, house, person
   - ~9,991 images
   - **Status:** ❌ Missing from ShiftBench
   - **Priority:** HIGH (appears in 5+ papers)

3. **Office-Home** (Venkateswara et al. 2017)
   - 4 domains: Art, Clipart, Product, Real-World
   - 65 classes
   - ~15,500 images
   - **Status:** ❌ Missing from ShiftBench
   - **Priority:** MEDIUM

4. **TerraIncognita** (Beery et al. 2018)
   - Wildlife camera trap images
   - 4 domains (locations)
   - 10 classes of animals
   - ~24,788 images
   - **Status:** ❌ Missing from ShiftBench
   - **Priority:** LOW (specialized)

5. **DomainNet** (Peng et al. 2019)
   - 6 domains: Clipart, Infograph, Painting, Quickdraw, Real, Sketch
   - 345 classes
   - ~600,000 images
   - **Status:** ❌ Missing from ShiftBench
   - **Priority:** MEDIUM (very large, comprehensive)

**Key Insight:** DomainBed focuses on vision-only benchmarks with synthetic domain shifts (artistic styles, sketch vs photo). These are classic benchmarks for domain generalization.

---

### 1.2 WILDS (Koh et al., ICML 2021)

**Paper:** "WILDS: A Benchmark of in-the-Wild Distribution Shifts"
**Citation:** [Koh et al. 2021](https://arxiv.org/abs/2012.07421)
**Focus:** Real-world distribution shifts across multiple modalities

#### Datasets (10 core)

**Vision (5):**

1. **iWildCam**
   - Animal detection in camera trap images
   - Shift: Geographic location (camera trap sites)
   - 243 classes, ~260,000 images
   - **Status:** ❌ Missing
   - **Priority:** LOW (specialized, large)

2. **Camelyon17**
   - Tumor tissue detection in histopathology
   - Shift: Different hospitals
   - 2 classes, ~450,000 patches
   - **Status:** ❌ Missing
   - **Priority:** MEDIUM (medical, real shift)

3. **FMoW** (Functional Map of the World)
   - Satellite imagery classification
   - Shift: Geographic region + temporal
   - 62 classes, ~350,000 images (large: 50 GB)
   - **Status:** ❌ Missing
   - **Priority:** LOW (very large)

4. **RxRx1**
   - Cell microscopy with genetic perturbations
   - Shift: Experimental batch effects
   - 1,139 classes, ~125,514 images
   - **Status:** ❌ Missing
   - **Priority:** MEDIUM (biological shift, similar to molecular)

5. **GlobalWheat**
   - Wheat head detection in field images
   - Shift: Geographic location (12 countries)
   - Object detection task, ~3,422 images
   - **Status:** ❌ Missing
   - **Priority:** LOW (specialized, small)

**NLP (2):**

6. **CivilComments**
   - Toxicity detection in online comments
   - Shift: Demographic identity groups
   - 2 classes, ~450,000 comments
   - **Status:** ✅ PARTIAL - We have civil_comments (30k samples)
   - **Priority:** LOW (already have partial coverage)

7. **Amazon-WILDS**
   - Product review sentiment
   - Shift: Reviewer demographics
   - 5 classes (rating), ~3M reviews
   - **Status:** ✅ PARTIAL - We have amazon (30k samples, 2 classes)
   - **Priority:** LOW (already have partial coverage)

**Biology (2):**

8. **OGB-MolPCBA**
   - Molecular property prediction
   - Shift: Scaffold split (structural dissimilarity)
   - 128 tasks, ~437,929 molecules
   - **Status:** ✅ SIMILAR - We have 11 MoleculeNet datasets with scaffold splits
   - **Priority:** LOW (redundant with existing molecular datasets)

9. **Poverty Map**
   - Poverty estimation from satellite imagery
   - Shift: Geographic (country + urban/rural)
   - Regression task, ~19,669 images
   - **Status:** ❌ Missing
   - **Priority:** LOW (specialized domain)

**Code (1):**

10. **Py150**
    - Python code completion
    - Shift: Different GitHub repositories
    - Token prediction, ~100k files
    - **Status:** ❌ Missing
    - **Priority:** LOW (specialized, code domain)

**Key Insight:** WILDS emphasizes real-world shifts with large-scale datasets. ShiftBench already covers molecular (overlaps with OGB-MolPCBA) and partial text coverage.

---

### 1.3 Shifts (Malinin et al., 2021)

**Paper:** "Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks"
**Citation:** [Malinin et al. 2021](https://arxiv.org/abs/2107.07455)
**Focus:** Large-scale production datasets with uncertainty quantification

#### Datasets (3 core)

1. **Weather Prediction**
   - Tabular meteorological data
   - Shift: Geographic location
   - Regression task
   - **Status:** ❌ Missing
   - **Priority:** LOW (specialized, very large production dataset)

2. **Machine Translation**
   - English-Russian translation
   - Shift: Domain/topic of text
   - Sequence-to-sequence task
   - **Status:** ❌ Missing
   - **Priority:** LOW (translation is out of scope for binary/regression focus)

3. **Vehicle Motion Prediction**
   - Autonomous driving trajectory prediction
   - Shift: Geographic location, weather
   - Continuous sequence modeling
   - **Status:** ❌ Missing
   - **Priority:** LOW (specialized, large)

**Key Insight:** Shifts focuses on production-scale datasets with complex shifts. Most are too specialized/large for ShiftBench's current scope.

---

### 1.4 OpenOOD (Yang et al., NeurIPS 2022)

**Paper:** "OpenOOD: Benchmarking Generalized Out-of-Distribution Detection"
**Citation:** [Yang et al. 2022](https://arxiv.org/abs/2210.07242)
**Focus:** OOD detection across 9 benchmarks

#### Datasets (Near-OOD and Far-OOD variants)

**In-Distribution Datasets:**
1. **CIFAR-10 / CIFAR-100**
   - Standard image classification
   - 10/100 classes, 60k images each
   - **Status:** ❌ Missing
   - **Priority:** HIGH (foundational benchmark, small)

2. **ImageNet-1k**
   - Large-scale image classification
   - 1000 classes, ~1.2M images
   - **Status:** ❌ Missing
   - **Priority:** LOW (very large, infrastructure intensive)

**OOD Test Datasets:**
3. **MNIST**
   - Handwritten digits
   - 10 classes, 70k images
   - **Status:** ❌ Missing
   - **Priority:** HIGH (foundational, used in 10+ papers)

4. **SVHN** (Street View House Numbers)
   - Real-world digit recognition
   - 10 classes, ~600k images
   - **Status:** ❌ Missing
   - **Priority:** HIGH (classic domain adaptation benchmark)

5. **Textures (DTD)**
   - Describable Textures Dataset
   - 47 texture categories
   - **Status:** ❌ Missing
   - **Priority:** LOW (specialized)

6. **Places365**
   - Scene recognition
   - 365 classes, large-scale
   - **Status:** ❌ Missing
   - **Priority:** LOW (very large)

**Key Insight:** OpenOOD relies heavily on vision datasets. MNIST and SVHN are foundational and appear in many domain adaptation papers.

---

### 1.5 RobustBench (Croce et al., NeurIPS 2021)

**Paper:** "RobustBench: A Standardized Adversarial Robustness Benchmark"
**Citation:** [Croce et al. 2021](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/a3c65c2974270fd093ee8a9bf8ae7d0b-Paper-round2.pdf)
**Focus:** Adversarial robustness evaluation

#### Datasets (3 core threat models)

1. **CIFAR-10** (ℓ∞, ℓ2 threats)
   - Standard corruptions and adversarial perturbations
   - **Status:** ❌ Missing
   - **Priority:** HIGH

2. **ImageNet** (ℓ∞, ℓ2 threats)
   - Large-scale robustness
   - **Status:** ❌ Missing
   - **Priority:** LOW (very large)

3. **ImageNet-C** (Common Corruptions)
   - 15 corruption types × 5 severity levels
   - Weather, blur, noise, digital corruptions
   - **Status:** ❌ Missing
   - **Priority:** HIGH (standard robustness benchmark, appears in 5+ papers)

**Key Insight:** RobustBench focuses on adversarial robustness, which is tangentially related to covariate shift. ImageNet-C is more relevant as a natural distribution shift benchmark.

---

## 2. Domain Adaptation Standard Datasets

### 2.1 Classic Vision Benchmarks

#### Office-31 (Saenko et al. 2010)
- **Description:** 4,652 images, 31 classes
- **Domains:** Amazon (web), Webcam, DSLR, Caltech
- **Shift Type:** Camera/acquisition domain
- **Status:** ❌ Missing from ShiftBench
- **Priority:** HIGH
- **Appears in:** 20+ domain adaptation papers
- **Effort:** Low (small dataset, publicly available)

#### VisDA (Peng et al. 2017)
- **Description:** Synthetic-to-real visual domain adaptation
- **Domains:** Synthetic (3D renderings, 152k images) → Real (MS COCO, 55k images)
- **Classes:** 12 object categories
- **Status:** ❌ Missing
- **Priority:** MEDIUM
- **Appears in:** 15+ papers
- **Effort:** Medium (larger dataset, ~200k images)

#### MNIST → SVHN → USPS
- **Description:** Classic digit recognition domain adaptation
- **MNIST:** 70k handwritten digits (28×28 grayscale)
- **SVHN:** ~600k street view house numbers (32×32 color)
- **USPS:** 9,298 handwritten digits (16×16 grayscale)
- **Status:** ❌ Missing all three
- **Priority:** HIGH (foundational benchmark trio)
- **Appears in:** 30+ papers
- **Effort:** Low (all publicly available, small-to-medium)

### 2.2 NLP Domain Adaptation

#### Multi-Domain Sentiment (Blitzer et al. 2007)
- **Description:** Amazon product reviews across domains
- **Domains:** Books, DVDs, Electronics, Kitchen (4 core)
- **Full dataset:** 25 domains, 340k+ reviews
- **Status:** ✅ PARTIAL - We have amazon (3 categories)
- **Priority:** LOW (already covered)
- **Appears in:** 25+ NLP domain adaptation papers

#### AG News
- **Description:** Topic classification
- **Classes:** World, Sports, Business, Sci/Tech
- **Samples:** 120k train, 7.6k test
- **Status:** ❌ Missing
- **Priority:** MEDIUM (standard NLP benchmark)
- **Appears in:** 10+ papers
- **Effort:** Low (publicly available via HuggingFace)

---

## 3. Covariate Shift & Importance Weighting Papers

### 3.1 Kernel Mean Matching (Sugiyama et al. 2008)

**Paper:** "Covariate Shift by Kernel Mean Matching"
**Citation:** [Gretton et al. 2009, Dataset Shift in ML book]

**Datasets Used:**
- Synthetic Gaussian mixture datasets (not applicable)
- **Microarray gene expression** (specialized biomedical)
- **UCI datasets:** Wine, Breast Cancer, Ionosphere

**Key Insight:** KMM papers typically use small UCI datasets. We already cover similar scale with tabular datasets.

### 3.2 Shimodaira (2000) - Importance Weighting

**Paper:** "Improving Predictive Inference Under Covariate Shift"

**Datasets Used:**
- Small synthetic datasets
- Time-series financial data (specialized)

**Key Insight:** Early theoretical work; datasets not broadly adopted.

---

## 4. Conformal Prediction Under Shift

### 4.1 Weighted Conformal Prediction (Tibshirani et al. 2019)

**Paper:** "Conformal Prediction Under Covariate Shift"
**Citation:** [Tibshirani et al. 2019](https://arxiv.org/abs/1904.06019)

**Datasets Used:**
1. **Airfoil** (UCI ML Repository)
   - N=1,503, d=5 (scaled sound pressure level)
   - Regression task
   - **Status:** ❌ Missing
   - **Priority:** LOW (small, specialized)

2. **Communities & Crime** (UCI)
   - N=1,994, d=100+ (crime prediction)
   - **Status:** ❌ Missing
   - **Priority:** LOW

### 4.2 Conformal Prediction Beyond Exchangeability (Barber et al. 2023)

**Paper:** "Conformal Prediction Beyond Exchangeability" (Annals of Statistics)
**Citation:** [Barber et al. 2023](https://arxiv.org/abs/2202.13415)

**Datasets Used:**
1. **ELEC2** (Electricity demand in Australia)
   - Time-series with distribution drift
   - **Status:** ❌ Missing
   - **Priority:** LOW (time-series focus)

2. **Election Forecasting** (US presidential elections)
   - Specialized political dataset
   - **Status:** ❌ Missing
   - **Priority:** LOW

**Key Insight:** Conformal papers use specialized regression datasets. ShiftBench has molecular regression datasets that can serve similar purposes.

---

## 5. Fairness & Demographic Shift Datasets

### 5.1 Adult / Census Income
- **Description:** Predict income >$50K from demographics
- **Samples:** 48,842
- **Shift:** Demographic groups (race, sex, age)
- **Status:** ✅ HAVE - adult dataset
- **Priority:** N/A (already included)

### 5.2 COMPAS
- **Description:** Recidivism prediction
- **Samples:** 6,172
- **Shift:** Demographic groups
- **Status:** ✅ HAVE - compas dataset
- **Priority:** N/A (already included)

### 5.3 Folktables / ACS PUMS
- **Description:** US Census American Community Survey
- **Tasks:** Income, employment, health insurance, etc.
- **Shift:** State-level distribution shifts (51 states × 5 years = 255 datasets)
- **Status:** ❌ Missing
- **Priority:** HIGH for fairness research
- **Appears in:** 10+ fairness/shift papers since 2021
- **Effort:** Medium (requires folktables Python package)

**Paper:** "Retiring Adult: New Datasets for Fair Machine Learning" (Ding et al., NeurIPS 2021)
**Citation:** [Ding et al. 2021](https://arxiv.org/abs/2108.04884)

### 5.4 Other Fairness Datasets We Have
- ✅ Bank Marketing (temporal shift)
- ✅ German Credit (demographic shift)
- ✅ Diabetes (age shift)
- ✅ Heart Disease (demographic shift)
- ✅ Student Performance (school/demographic shift)

**Key Insight:** ShiftBench has excellent coverage of fairness/demographic shift datasets. Folktables would be a valuable addition for state-level shifts.

---

## 6. Robustness to Natural Distribution Shifts

### ImageNet Variants (Hendrycks et al.)

#### ImageNet-C (Corruption Robustness)
- **Description:** 15 corruption types × 5 severity levels
- **Corruptions:** Gaussian noise, shot noise, impulse noise, defocus blur, glass blur, motion blur, zoom blur, snow, frost, fog, brightness, contrast, elastic transform, pixelate, JPEG
- **Status:** ❌ Missing
- **Priority:** HIGH
- **Appears in:** 20+ robustness papers
- **Effort:** Medium (derived from ImageNet, requires preprocessing)

**Paper:** "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations" (Hendrycks & Dietterich, ICLR 2019)

#### ImageNet-R (Renditions)
- **Description:** 30k images with different artistic renditions
- **Classes:** 200 ImageNet classes
- **Styles:** Art, cartoons, origami, graffiti, embroidery, graphics, etc.
- **Status:** ❌ Missing
- **Priority:** MEDIUM
- **Appears in:** 10+ papers
- **Effort:** Low (pre-processed dataset available)

#### ImageNet-A (Adversarial)
- **Description:** Natural adversarial examples
- **Classes:** 200 ImageNet classes
- **Samples:** 7,500 real-world images that fool models
- **Status:** ❌ Missing
- **Priority:** LOW (specialized)
- **Effort:** Low

**Key Insight:** ImageNet-C is the gold standard for corruption robustness. However, all ImageNet variants are large and require significant infrastructure (pre-trained models, feature extractors).

---

## 7. TableShift Benchmark (Gardner et al., NeurIPS 2023)

**Paper:** "Benchmarking Distribution Shift in Tabular Data with TableShift"
**Citation:** [Gardner et al. 2023](https://arxiv.org/abs/2312.07577)

**Datasets (15 total):**

**Healthcare:**
1. Diabetes (hospital readmission)
2. ICU Length-of-Stay
3. ICU Mortality

**Public Policy:**
4. ACS Employment (Folktables)
5. ACS Income (Folktables)
6. ACS Public Coverage (Folktables)
7. ACS Mobility (Folktables)

**Education:**
8. College Scorecard

**Finance:**
9. Voting (US voter turnout)

**Other:**
10. Food Stamps
11. Assistive Technology
12. Hospital Infections
13. ANES (voter survey)
14. ACSFHEO (fair housing)
15. BRFSS (health risk behaviors)

**Shift Types:** Domain, subpopulation, temporal

**Status:** ❌ Most missing (we have diabetes, some ACS-like via Adult/COMPAS)
**Priority:** LOW-MEDIUM (comprehensive but overlaps with our tabular + adds new medical/policy datasets)
**Effort:** Medium-High (requires tableshift package + preprocessing)

**Key Insight:** TableShift provides systematic tabular benchmarks. ShiftBench has partial coverage (Adult, Diabetes, Bank, COMPAS). Could add ACS PUMS tasks for completeness.

---

## 8. Molecular Property Prediction

### QM9 (Quantum Chemistry)
- **Description:** 134k small organic molecules with DFT properties
- **Properties:** 13 quantum properties (energy, HOMO-LUMO gap, etc.)
- **Shift:** Out-of-distribution molecule sizes
- **Status:** ❌ Missing
- **Priority:** LOW (redundant with MoleculeNet datasets)
- **Effort:** Low (available via MoleculeNet)

**Key Insight:** ShiftBench already has 11 MoleculeNet datasets with scaffold splits, which cover molecular property prediction under covariate shift well.

---

## 9. Summary Tables

### 9.1 Datasets Appearing in 3+ Papers (HIGH PRIORITY)

| Dataset | Domain | Papers | ShiftBench Status | Priority |
|---------|--------|--------|-------------------|----------|
| **MNIST** | Vision | 30+ | ❌ Missing | HIGH |
| **SVHN** | Vision | 25+ | ❌ Missing | HIGH |
| **Office-31** | Vision | 20+ | ❌ Missing | HIGH |
| **ImageNet-C** | Vision | 20+ | ❌ Missing | HIGH |
| **Multi-Domain Sentiment** | NLP | 25+ | ✅ Partial (Amazon) | LOW |
| **Adult/Census** | Tabular | 20+ | ✅ HAVE | N/A |
| **COMPAS** | Tabular | 15+ | ✅ HAVE | N/A |
| **PACS** | Vision | 15+ | ❌ Missing | HIGH |
| **VisDA** | Vision | 15+ | ❌ Missing | MEDIUM |
| **CIFAR-10** | Vision | 15+ | ❌ Missing | HIGH |
| **Folktables** | Tabular | 10+ | ❌ Missing | HIGH |
| **AG News** | NLP | 10+ | ❌ Missing | MEDIUM |
| **USPS** | Vision | 10+ | ❌ Missing | MEDIUM |

### 9.2 ShiftBench Current Coverage

| Domain | Current Datasets | Status |
|--------|------------------|--------|
| **Molecular** | 11 (BACE, BBBP, ClinTox, ESOL, FreeSolv, Lipophilicity, SIDER, Tox21, ToxCast, MUV, MolHIV) | ✅ Strong |
| **Tabular** | 7 (Adult, COMPAS, Bank, German Credit, Diabetes, Heart Disease, Student) | ✅ Good |
| **Text** | 5 (IMDB, Yelp, Amazon, CivilComments, Twitter) | ✅ Good |
| **Vision** | 0 | ❌ **GAP** |
| **TOTAL** | 23 | Target: 50-100 |

### 9.3 Recommendations by Priority

#### TIER 1: Essential for Credibility (Add These)

1. **MNIST + SVHN + USPS** (vision trio)
   - Why: Foundational domain adaptation benchmarks, cited in 30+ papers
   - Effort: LOW (all small, <100 MB total)
   - Impact: Establishes vision credibility

2. **Office-31** (vision domain adaptation)
   - Why: Standard benchmark for DA methods, 20+ citations
   - Effort: LOW (~300 MB)
   - Impact: Enables comparison with DA baselines

3. **CIFAR-10** (vision robustness)
   - Why: Standard benchmark for robustness, 15+ citations
   - Effort: LOW (180 MB)
   - Impact: OOD detection baseline

4. **Folktables (ACS PUMS)** (tabular fairness)
   - Why: Modern fairness benchmark, 10+ recent papers
   - Effort: MEDIUM (requires folktables package)
   - Impact: State-of-the-art fairness evaluation

#### TIER 2: Strengthens Benchmark (Consider Adding)

5. **PACS** (vision domain generalization)
   - Why: Standard DG benchmark, 15+ citations
   - Effort: MEDIUM (~1 GB)
   - Impact: Artistic domain shifts

6. **ImageNet-C** (vision corruption robustness)
   - Why: Gold standard for corruption robustness, 20+ citations
   - Effort: HIGH (requires ImageNet infrastructure)
   - Impact: Natural distribution shift evaluation

7. **AG News** (NLP topic classification)
   - Why: Standard NLP benchmark, 10+ citations
   - Effort: LOW (~30 MB)
   - Impact: Non-sentiment NLP task

8. **VisDA** (vision synthetic-to-real)
   - Why: Large-scale DA benchmark, 15+ citations
   - Effort: MEDIUM-HIGH (~2 GB)
   - Impact: Realistic sim-to-real shifts

#### TIER 3: Nice-to-Have (Lower Priority)

9. **Camelyon17** (medical imaging)
   - Why: Real-world medical shift (hospital-level)
   - Effort: HIGH (large medical images)
   - Impact: Medical domain coverage

10. **RxRx1** (cell microscopy)
    - Why: Biological batch effects (similar to molecular)
    - Effort: HIGH (large image dataset)
    - Impact: Extends biological coverage

11. **TableShift datasets** (various tabular)
    - Why: Comprehensive tabular benchmark
    - Effort: MEDIUM-HIGH (15 datasets)
    - Impact: Systematic tabular coverage

---

## 10. Gap Analysis: What's Missing from ShiftBench?

### Critical Gaps

1. **Vision Domain:** Zero vision datasets currently
   - Impact: Cannot compare with DomainBed, OpenOOD, RobustBench baselines
   - Solution: Add MNIST/SVHN/USPS + Office-31 + CIFAR-10 (TIER 1)

2. **State-Level Fairness:** No geographic shift datasets
   - Impact: Missing modern fairness benchmarks (Folktables)
   - Solution: Add Folktables ACS tasks (TIER 1)

3. **Image Corruption Robustness:** No corruption benchmarks
   - Impact: Cannot evaluate natural distribution shift robustness
   - Solution: Add ImageNet-C (TIER 2, but requires infrastructure)

### Strengths to Leverage

1. **Molecular Domain:** 11 datasets with scaffold splits
   - Unique contribution: Most benchmarks don't cover molecular
   - Advantage: Real-world drug discovery shifts

2. **Fairness/Tabular:** 7 diverse fairness datasets
   - Strong coverage: Adult, COMPAS, Bank, German Credit, etc.
   - Advantage: Demographic and temporal shifts

3. **Text/NLP:** 5 sentiment + toxicity datasets
   - Good coverage: IMDB, Yelp, Amazon, CivilComments, Twitter
   - Advantage: Multiple shift types (temporal, geographic, demographic)

---

## 11. Effort Estimation for TIER 1 Additions

| Dataset | Download Size | Preprocessing | Integration | Total Effort |
|---------|---------------|---------------|-------------|--------------|
| MNIST | ~50 MB | Featurization (flatten or CNN) | 2-4 hours | Low |
| SVHN | ~200 MB | Featurization | 2-4 hours | Low |
| USPS | ~5 MB | Featurization | 1-2 hours | Low |
| Office-31 | ~300 MB | Feature extraction (ResNet) | 4-6 hours | Low-Medium |
| CIFAR-10 | ~180 MB | Featurization | 2-4 hours | Low |
| Folktables | ~200 MB | Python API integration | 4-8 hours | Medium |

**Total TIER 1 Effort:** ~15-30 hours of preprocessing + integration

---

## 12. Final Recommendations for NeurIPS D&B Submission

### Must-Have (TIER 1) - Add Before Submission

1. ✅ **MNIST, SVHN, USPS** - Establish vision credibility with foundational benchmarks
2. ✅ **Office-31** - Standard domain adaptation comparison point
3. ✅ **CIFAR-10** - OOD detection baseline
4. ⚠️ **Folktables (ACS)** - Modern fairness benchmark (consider if time permits)

**Rationale:** These 4-5 additions would give ShiftBench:
- **Total: 27-28 datasets** (respectable for initial benchmark)
- **Vision coverage:** 5 foundational datasets
- **Enables comparison** with DomainBed, OpenOOD, WILDS
- **Low effort:** ~20-30 hours total

### Defer to Future Work (TIER 2-3)

- PACS, VisDA, DomainNet (larger vision benchmarks)
- ImageNet-C (requires infrastructure)
- TableShift datasets (15 additional datasets)
- Specialized datasets (Camelyon17, RxRx1, Poverty Map)

### Positioning for Paper

**Current Strengths to Emphasize:**
1. "Unique molecular property prediction coverage (11 scaffold-split datasets)"
2. "Comprehensive fairness evaluation across demographic shifts (7 datasets)"
3. "Multi-domain text datasets with temporal/geographic/demographic shifts (5 datasets)"

**New Additions for Credibility:**
4. "Classic vision domain adaptation benchmarks (Office-31, MNIST/SVHN/USPS)"
5. "Standard OOD detection baselines (CIFAR-10)"
6. "Modern fairness benchmarks (Folktables)" [if added]

---

## 13. References & Sources

### Major Benchmarks
- [DomainBed](https://github.com/facebookresearch/DomainBed) - Gulrajani & Lopez-Paz, ICLR 2021
- [WILDS](https://wilds.stanford.edu/) - Koh et al., ICML 2021
- [Shifts](https://github.com/Shifts-Project/shifts) - Malinin et al., 2021
- [OpenOOD](https://github.com/Jingkang50/OpenOOD) - Yang et al., NeurIPS 2022
- [RobustBench](https://robustbench.github.io/) - Croce et al., NeurIPS 2021
- [TableShift](https://tableshift.org/) - Gardner et al., NeurIPS 2023

### Domain Adaptation
- [Multi-Domain Sentiment Dataset](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/) - Blitzer et al., ACL 2007
- [Office-31](http://ai.bu.edu/adaptation.html) - Saenko et al., 2010
- [VisDA](https://ai.bu.edu/visda-2017/) - Peng et al., 2017

### Conformal Prediction
- [Tibshirani et al. 2019](https://arxiv.org/abs/1904.06019) - Weighted Conformal Prediction
- [Barber et al. 2023](https://arxiv.org/abs/2202.13415) - Conformal Beyond Exchangeability

### Fairness
- [Folktables](https://github.com/socialfoundations/folktables) - Ding et al., NeurIPS 2021

### Robustness
- [ImageNet-C](https://github.com/hendrycks/robustness) - Hendrycks & Dietterich, ICLR 2019

### Repositories
- [UCI ML Repository](https://archive.ics.uci.edu/)
- [OpenML](https://www.openml.org/)
- [HuggingFace Datasets](https://huggingface.co/datasets)

---

## Appendix: Full Dataset Catalog (35+ Datasets Surveyed)

### Vision (18)
- MNIST, SVHN, USPS, CIFAR-10, CIFAR-100
- Office-31, Office-Home, VisDA, PACS, VLCS, TerraIncognita, DomainNet
- ImageNet, ImageNet-C, ImageNet-R, ImageNet-A
- iWildCam, Camelyon17, FMoW, RxRx1, GlobalWheat, Poverty Map

### NLP (7)
- Multi-Domain Sentiment (Amazon), AG News, IMDB, Yelp, CivilComments, Twitter Sentiment, Py150

### Tabular (10+)
- Adult, COMPAS, Bank Marketing, German Credit, Diabetes, Heart Disease, Student Performance
- Folktables (ACS Income, Employment, Public Coverage, Mobility)
- TableShift (15 datasets including healthcare, policy, education)

### Molecular (11)
- MoleculeNet: BACE, BBBP, ClinTox, ESOL, FreeSolv, Lipophilicity, SIDER, Tox21, ToxCast, MUV, MolHIV
- QM9 (alternative)

---

**Document Compiled:** 2026-02-16
**Next Steps:** Review TIER 1 recommendations and prioritize for NeurIPS D&B submission timeline.
