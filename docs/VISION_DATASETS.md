# Vision Datasets for ShiftBench

## Overview

ShiftBench evaluates certification protocols, not model training. For vision
datasets, we use pre-extracted feature embeddings (CLIP ViT-B/16 or ResNet)
stored as numpy arrays -- identical to how text datasets are pre-vectorized
as TF-IDF. This means the certification pipeline sees (n_samples, d_features)
arrays, not raw images.

## Selected Datasets (4 vision benchmarks)

### 1. Camelyon17-WILDS

| Property | Value |
|----------|-------|
| Task | Binary (tumor vs normal) |
| Shift type | Hospital/institutional (staining, scanning equipment) |
| Cohorts | 5 hospitals |
| Size | ~456K patches |
| License | CC0 (public domain) |
| Features | CLIP ViT-B/16 embeddings (512-dim) |

**Why**: Clean binary task, institutional shift mirrors real deployment
scenarios (train at hospital A, deploy at hospital B). CC0 license is ideal.

**Source**: https://wilds.stanford.edu/datasets/#camelyon17

### 2. Waterbirds (CUB + Places)

| Property | Value |
|----------|-------|
| Task | Binary (waterbird vs landbird) |
| Shift type | Spurious correlation (background) |
| Cohorts | 4 groups: (bird_type x background_type) |
| Size | 4,795 images |
| License | Research use |
| Features | ResNet-18 penultimate layer (512-dim), pre-extracted |

**Why**: Standard spurious correlation benchmark. The 4-group structure
tests certification under subpopulation shift. Pre-computed features
are widely available (group_DRO repository).

**Source**: https://github.com/kohpangwei/group_DRO

### 3. PACS

| Property | Value |
|----------|-------|
| Task | 7-class (bird, car, chair, dog, guitar, house, person) |
| Shift type | Visual domain/style (photo, art, cartoon, sketch) |
| Cohorts | 4 domains |
| Size | 9,991 images |
| License | Research use |
| Features | CLIP ViT-B/16 embeddings (512-dim) |

**Why**: Classic domain generalization benchmark. 4 clean domains enable
leave-one-domain-out evaluation. Small enough for rapid iteration.

**Source**: https://dali-dl.github.io/project_iccv2017.html

### 4. Terra Incognita

| Property | Value |
|----------|-------|
| Task | 10-class wildlife classification |
| Shift type | Camera-trap geographic shift |
| Cohorts | 4 camera locations |
| Size | 24,788 images |
| License | Research use |
| Features | CLIP ViT-B/16 embeddings (512-dim) |

**Why**: Natural geographic shift with wildlife categories. Camera-trap
locations provide well-defined cohorts with real distributional differences.

**Source**: https://beerys.github.io/CaltechCameraTraps/

## Feature Extraction Strategy

For all datasets, a one-time feature extraction pass using CLIP ViT-B/16
produces 512-dimensional embedding vectors stored as numpy arrays:

```
data/processed/{dataset}/
    features.npy      # (n_samples, 512)  CLIP embeddings
    labels.npy        # (n_samples,)      integer labels
    cohorts.npy       # (n_samples,)      cohort/domain IDs
    splits.csv        # uid, split (train/cal/test)
    metadata.json     # dataset statistics
```

This is directly analogous to molecular (fingerprint vectors) and text
(TF-IDF vectors) datasets already in ShiftBench.

### For multi-class datasets (PACS, Terra Incognita)

ShiftBench currently evaluates PPV for binary predictions. For multi-class:
- Option A: One-vs-rest binarization (certify per-class PPV)
- Option B: Certify top-1 accuracy per cohort
- Option C: Focus on binary datasets only (Camelyon17, Waterbirds)

Recommendation: Start with Camelyon17 + Waterbirds (binary), add PACS/Terra
later with one-vs-rest binarization.

## Integration Priority

1. **Camelyon17-WILDS** (highest - CC0, binary, large, medical relevance)
2. **Waterbirds** (high - binary, pre-extracted features, standard benchmark)
3. **PACS** (medium - needs binarization, but very well-known)
4. **Terra Incognita** (medium - needs binarization, geographic shift)

## Data Preparation Scripts

See `scripts/prepare_vision_datasets.py` for downloading and processing.

## Domain Coverage After Addition

| Domain | Datasets | Example |
|--------|----------|---------|
| Molecular | BACE, BBBP, ClinTox, ESOL, ... | Scaffold split |
| Tabular | Adult, COMPAS, Bank, German Credit | Demographic cohorts |
| Text | IMDB, Yelp, Civil Comments, Amazon | Temporal/category cohorts |
| **Vision** | **Camelyon17, Waterbirds, PACS, Terra Incognita** | **Hospital/style/geographic** |
