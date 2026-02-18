"""
Prepare Vision Datasets for ShiftBench
========================================

Downloads and processes vision datasets into ShiftBench's standard format:
    features.npy, labels.npy, cohorts.npy, splits.csv, metadata.json

For vision datasets, features are pre-extracted embeddings (CLIP or ResNet),
NOT raw images. ShiftBench evaluates certification protocols, not models.

Supported datasets:
    - waterbirds: Binary (waterbird vs landbird), 4 spurious correlation groups
    - camelyon17: Binary (tumor vs normal), 5 hospital cohorts (WILDS)

Usage:
    python scripts/prepare_vision_datasets.py --dataset waterbirds
    python scripts/prepare_vision_datasets.py --dataset camelyon17
    python scripts/prepare_vision_datasets.py --all

Requirements:
    pip install torch torchvision  (for feature extraction)
    pip install wilds              (for Camelyon17)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def prepare_waterbirds(output_dir: str, data_root: str = "data/raw/waterbirds"):
    """Prepare Waterbirds dataset with pre-extracted features.

    Waterbirds (Sagawa et al., 2020):
    - Task: Binary (waterbird=1 vs landbird=0)
    - Shift: Spurious correlation between bird type and background
    - Cohorts: 4 groups (bird_type x background_type)
    - Pre-extracted ResNet-50 features from group_DRO repo

    If raw data is not available, generates a synthetic placeholder that
    mirrors the dataset structure for pipeline testing.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check if pre-extracted features exist
    feature_path = os.path.join(data_root, "features.npy")

    if os.path.exists(feature_path):
        print("Loading pre-extracted Waterbirds features...")
        features = np.load(feature_path)
        labels = np.load(os.path.join(data_root, "labels.npy"))
        groups = np.load(os.path.join(data_root, "groups.npy"))
        splits_raw = np.load(os.path.join(data_root, "splits.npy"))
    else:
        print("Pre-extracted Waterbirds features not found.")
        print(f"Expected at: {feature_path}")
        print()
        print("To use real Waterbirds data:")
        print("  1. Download from https://github.com/kohpangwei/group_DRO")
        print("  2. Run their feature extraction script")
        print("  3. Place features.npy, labels.npy, groups.npy, splits.npy")
        print(f"     in {data_root}/")
        print()
        print("Generating synthetic Waterbirds placeholder for pipeline testing...")
        features, labels, groups, splits_raw = _generate_waterbirds_placeholder()

    # Map splits: 0=train, 1=val(cal), 2=test
    n_samples = len(labels)
    split_names = []
    for s in splits_raw:
        if s == 0:
            split_names.append("train")
        elif s == 1:
            split_names.append("cal")
        else:
            split_names.append("test")

    # Save in ShiftBench format
    np.save(os.path.join(output_dir, "features.npy"), features)
    np.save(os.path.join(output_dir, "labels.npy"), labels.astype(int))
    np.save(os.path.join(output_dir, "cohorts.npy"), groups.astype(int))

    splits_df = pd.DataFrame({
        "uid": range(n_samples),
        "split": split_names,
    })
    splits_df.to_csv(os.path.join(output_dir, "splits.csv"), index=False)

    # Metadata
    train_mask = np.array(split_names) == "train"
    cal_mask = np.array(split_names) == "cal"
    test_mask = np.array(split_names) == "test"

    metadata = {
        "dataset": "waterbirds",
        "domain": "vision",
        "task": "binary",
        "shift_type": "spurious_correlation",
        "cohort_definition": "bird_type x background_type",
        "n_samples": int(n_samples),
        "n_features": int(features.shape[1]),
        "n_cohorts": int(len(np.unique(groups))),
        "n_train": int(train_mask.sum()),
        "n_cal": int(cal_mask.sum()),
        "n_test": int(test_mask.sum()),
        "positive_rate": float(labels.mean()),
        "feature_type": "resnet50_penultimate" if os.path.exists(feature_path)
                        else "synthetic_placeholder",
        "source": "https://github.com/kohpangwei/group_DRO",
        "license": "research_use",
        "citation": "Sagawa et al. (2020). Distributionally Robust Neural Networks.",
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Waterbirds saved to {output_dir}")
    print(f"  Samples: {n_samples} (train={train_mask.sum()}, "
          f"cal={cal_mask.sum()}, test={test_mask.sum()})")
    print(f"  Features: {features.shape[1]}-dim")
    print(f"  Cohorts: {len(np.unique(groups))}")
    print(f"  Positive rate: {labels.mean():.3f}")


def _generate_waterbirds_placeholder():
    """Generate synthetic data mimicking Waterbirds structure.

    4 groups with spurious correlation:
    - Group 0: waterbird + water background (majority, easy)
    - Group 1: waterbird + land background (minority, hard)
    - Group 2: landbird + water background (minority, hard)
    - Group 3: landbird + land background (majority, easy)
    """
    rng = np.random.RandomState(42)

    # Group sizes (from real dataset)
    group_sizes = {
        0: (1057, "train"),   # waterbird + water (large)
        1: (56, "train"),     # waterbird + land (small)
        2: (184, "train"),    # landbird + water (small)
        3: (3498, "train"),   # landbird + land (large)
    }

    # Train/val/test proportions (60/20/20)
    features_list = []
    labels_list = []
    groups_list = []
    splits_list = []

    for group_id in range(4):
        n_total = sum([1057, 56, 184, 3498])[group_id] if False else \
                  [1057, 56, 184, 3498][group_id]
        label = 1 if group_id <= 1 else 0  # waterbird=1, landbird=0

        # Generate features with group-specific mean
        mean_shift = rng.randn(512) * 0.5
        X = rng.randn(n_total, 512) + mean_shift

        # Add spurious feature (background)
        bg_signal = rng.randn(512) * 0.3
        if group_id in [0, 3]:  # majority (spurious correlated)
            X += bg_signal
        else:  # minority (spurious anti-correlated)
            X -= bg_signal

        features_list.append(X)
        labels_list.append(np.full(n_total, label))
        groups_list.append(np.full(n_total, group_id))

        # Split: 60/20/20
        n_train = int(n_total * 0.6)
        n_cal = int(n_total * 0.2)
        split_arr = (["train"] * n_train +
                     ["cal"] * n_cal +
                     ["test"] * (n_total - n_train - n_cal))
        splits_list.extend(split_arr)

    features = np.vstack(features_list).astype(np.float32)
    labels = np.concatenate(labels_list)
    groups = np.concatenate(groups_list)
    splits_raw = np.array([{"train": 0, "cal": 1, "test": 2}[s]
                           for s in splits_list])

    # Shuffle
    perm = rng.permutation(len(features))
    return features[perm], labels[perm], groups[perm], splits_raw[perm]


def prepare_camelyon17(output_dir: str, data_root: str = "data/raw/camelyon17"):
    """Prepare Camelyon17-WILDS dataset.

    Camelyon17 (Bandi et al., 2019, WILDS version):
    - Task: Binary (tumor=1 vs normal=0)
    - Shift: Hospital/institutional (5 hospitals)
    - Cohorts: Hospital ID (0-4)

    Requires WILDS package: pip install wilds
    Feature extraction via CLIP ViT-B/16 or pre-trained DenseNet-121.

    If WILDS data is not available, generates a synthetic placeholder.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check for pre-extracted features
    feature_path = os.path.join(data_root, "features.npy")

    if os.path.exists(feature_path):
        print("Loading pre-extracted Camelyon17 features...")
        features = np.load(feature_path)
        labels = np.load(os.path.join(data_root, "labels.npy"))
        hospitals = np.load(os.path.join(data_root, "hospitals.npy"))
        splits_raw = np.load(os.path.join(data_root, "splits.npy"))
    else:
        # Try WILDS
        try:
            from wilds import get_dataset
            print("Downloading Camelyon17-WILDS (this may take a while)...")
            dataset = get_dataset(dataset="camelyon17", root_dir=data_root,
                                  download=True)
            # Would need to extract features here -- complex
            print("WILDS download complete. Feature extraction needed.")
            print("Use CLIP or DenseNet-121 to extract embeddings.")
            print("Generating placeholder for now...")
            features, labels, hospitals, splits_raw = \
                _generate_camelyon17_placeholder()
        except ImportError:
            print("WILDS package not installed and no pre-extracted features.")
            print("Install: pip install wilds")
            print("Or place pre-extracted features in:")
            print(f"  {data_root}/features.npy")
            print()
            print("Generating synthetic Camelyon17 placeholder...")
            features, labels, hospitals, splits_raw = \
                _generate_camelyon17_placeholder()

    # Map splits
    n_samples = len(labels)
    split_names = []
    for s in splits_raw:
        if s == 0:
            split_names.append("train")
        elif s == 1:
            split_names.append("cal")
        else:
            split_names.append("test")

    # Save
    np.save(os.path.join(output_dir, "features.npy"), features)
    np.save(os.path.join(output_dir, "labels.npy"), labels.astype(int))
    np.save(os.path.join(output_dir, "cohorts.npy"), hospitals.astype(int))

    splits_df = pd.DataFrame({
        "uid": range(n_samples),
        "split": split_names,
    })
    splits_df.to_csv(os.path.join(output_dir, "splits.csv"), index=False)

    # Metadata
    train_mask = np.array(split_names) == "train"
    cal_mask = np.array(split_names) == "cal"
    test_mask = np.array(split_names) == "test"

    metadata = {
        "dataset": "camelyon17",
        "domain": "vision",
        "task": "binary",
        "shift_type": "hospital_institutional",
        "cohort_definition": "hospital_id",
        "n_samples": int(n_samples),
        "n_features": int(features.shape[1]),
        "n_cohorts": int(len(np.unique(hospitals))),
        "n_train": int(train_mask.sum()),
        "n_cal": int(cal_mask.sum()),
        "n_test": int(test_mask.sum()),
        "positive_rate": float(labels.mean()),
        "feature_type": "clip_vitb16" if os.path.exists(feature_path)
                        else "synthetic_placeholder",
        "source": "https://wilds.stanford.edu/datasets/#camelyon17",
        "license": "CC0",
        "citation": "Bandi et al. (2019). From Detection of Individual Metastases "
                    "to Classification of Lymph Node Status at the Patient Level.",
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Camelyon17 saved to {output_dir}")
    print(f"  Samples: {n_samples} (train={train_mask.sum()}, "
          f"cal={cal_mask.sum()}, test={test_mask.sum()})")
    print(f"  Features: {features.shape[1]}-dim")
    print(f"  Cohorts (hospitals): {len(np.unique(hospitals))}")
    print(f"  Positive rate: {labels.mean():.3f}")


def _generate_camelyon17_placeholder():
    """Generate synthetic data mimicking Camelyon17 structure.

    5 hospitals with varying tumor prevalence and feature distributions.
    """
    rng = np.random.RandomState(42)

    # Hospital configurations (mimicking real dataset characteristics)
    hospitals_config = [
        {"n": 5000, "tumor_rate": 0.30, "shift": 0.0},   # Hospital 0 (reference)
        {"n": 4000, "tumor_rate": 0.25, "shift": 0.5},   # Hospital 1
        {"n": 3000, "tumor_rate": 0.35, "shift": 0.8},   # Hospital 2
        {"n": 3500, "tumor_rate": 0.28, "shift": 1.0},   # Hospital 3 (OOD test)
        {"n": 2500, "tumor_rate": 0.32, "shift": 1.2},   # Hospital 4 (OOD test)
    ]

    features_list = []
    labels_list = []
    hospital_list = []
    splits_list = []

    # Shared coefficients for label generation
    beta = rng.randn(512)
    beta = beta / np.linalg.norm(beta) * np.sqrt(512)

    for h_id, config in enumerate(hospitals_config):
        n = config["n"]
        shift = config["shift"]

        # Hospital-specific mean shift
        shift_dir = rng.randn(512)
        shift_dir = shift_dir / np.linalg.norm(shift_dir)

        X = rng.randn(n, 512).astype(np.float32) + shift * shift_dir

        # Labels
        logits = X @ beta / np.sqrt(512) + rng.randn(n) * 0.5
        probs = 1 / (1 + np.exp(-logits))
        # Adjust to match target tumor rate
        threshold = np.quantile(probs, 1 - config["tumor_rate"])
        y = (probs >= threshold).astype(int)

        features_list.append(X)
        labels_list.append(y)
        hospital_list.append(np.full(n, h_id))

        # Split: Hospitals 0-2 for train/cal, 3-4 for test (mimics WILDS)
        if h_id <= 2:
            n_train = int(n * 0.7)
            n_cal = int(n * 0.3)
            split_arr = (["train"] * n_train +
                         ["cal"] * n_cal)
        else:
            split_arr = ["test"] * n

        # Pad if needed
        while len(split_arr) < n:
            split_arr.append(split_arr[-1])
        split_arr = split_arr[:n]
        splits_list.extend(split_arr)

    features = np.vstack(features_list)
    labels = np.concatenate(labels_list)
    hospitals = np.concatenate(hospital_list)
    splits_raw = np.array([{"train": 0, "cal": 1, "test": 2}[s]
                           for s in splits_list])

    # Shuffle within splits
    perm = rng.permutation(len(features))
    return features[perm], labels[perm], hospitals[perm], splits_raw[perm]


def main():
    parser = argparse.ArgumentParser(
        description="Prepare vision datasets for ShiftBench"
    )
    parser.add_argument("--dataset", choices=["waterbirds", "camelyon17"],
                        help="Dataset to prepare")
    parser.add_argument("--all", action="store_true",
                        help="Prepare all vision datasets")
    parser.add_argument("--output_dir", default="data/processed",
                        help="Output base directory")
    parser.add_argument("--data_root", default="data/raw",
                        help="Raw data root directory")
    args = parser.parse_args()

    if args.all or args.dataset == "waterbirds":
        prepare_waterbirds(
            output_dir=os.path.join(args.output_dir, "waterbirds"),
            data_root=os.path.join(args.data_root, "waterbirds"),
        )
        print()

    if args.all or args.dataset == "camelyon17":
        prepare_camelyon17(
            output_dir=os.path.join(args.output_dir, "camelyon17"),
            data_root=os.path.join(args.data_root, "camelyon17"),
        )
        print()

    if not args.all and args.dataset is None:
        parser.print_help()


if __name__ == "__main__":
    main()
