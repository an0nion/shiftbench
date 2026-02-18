"""Preprocess molecular datasets from ravel format to shift-bench format.

This script converts raw SMILES + labels into standardized shift-bench format:
- features.npy: RDKit 2D descriptors (standardized)
- labels.npy: Binary/regression labels
- cohorts.npy: Scaffold-based cohort assignments
- splits.csv: Train/cal/test splits

Usage:
    python scripts/preprocess_molecular.py --dataset bace
    python scripts/preprocess_molecular.py --all  # Process all molecular datasets
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add ravel to path to use its featurizers
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ravel" / "src"))

try:
    from ravel.chem.smiles import murcko_scaffold
    from ravel.data.splits import scaffoldwise_split
    from ravel.featurizers.rdkit2d import featurize_rdkit2d
    RAVEL_AVAILABLE = True
except ImportError:
    RAVEL_AVAILABLE = False
    print("[WARNING] RAVEL not found. Install with: pip install -e /path/to/ravel")


DATASET_CONFIG = {
    "bace": {
        "source_file": "bace_train.csv",
        "task_type": "binary",
        "label_col": "label",
    },
    "bbbp": {
        "source_file": "bbbp_train.csv",
        "task_type": "binary",
        "label_col": "label",
    },
    "clintox": {
        "source_file": "clintox_train.csv",
        "task_type": "binary",
        "label_col": None,  # Auto-detect
    },
    "esol": {
        "source_file": "esol_train.csv",
        "task_type": "regression",
        "label_col": "label",
    },
    "freesolv": {
        "source_file": "freesolv_train.csv",
        "task_type": "regression",
        "label_col": "label",
    },
    "lipophilicity": {
        "source_file": "lipophilicity_train.csv",
        "task_type": "regression",
        "label_col": "label",
    },
    "sider": {
        "source_file": "sider_train.csv",
        "task_type": "multilabel",
        "label_col": None,  # Auto-detect
    },
    "tox21": {
        "source_file": "tox21_train.csv",
        "task_type": "multilabel",
        "label_col": None,  # Auto-detect
    },
    "toxcast": {
        "source_file": "toxcast_train.csv",
        "task_type": "multilabel",
        "label_col": None,  # Auto-detect
    },
    "muv": {
        "source_file": "muv_train.csv",
        "task_type": "multilabel",
        "label_col": None,  # Auto-detect
    },
    "molhiv": {
        "source_file": "molhiv_train.csv",
        "task_type": "binary",
        "label_col": None,  # Auto-detect
    },
}


def preprocess_molecular_dataset(
    dataset_name: str,
    ravel_data_dir: Path,
    output_dir: Path,
    seed: int = 42,
):
    """Preprocess a single molecular dataset.

    Args:
        dataset_name: Name of dataset (e.g., "bace")
        ravel_data_dir: Path to ravel/data/raw/ directory
        output_dir: Path to shift-bench/data/processed/<dataset_name>/
        seed: Random seed for splits
    """
    if not RAVEL_AVAILABLE:
        raise ImportError("RAVEL is required for molecular preprocessing")

    print(f"\n{'='*80}")
    print(f"Processing {dataset_name.upper()}")
    print(f"{'='*80}")

    # Get configuration
    config = DATASET_CONFIG[dataset_name]
    source_path = ravel_data_dir / config["source_file"]

    # Step 1: Load raw data
    print(f"\n[1/6] Loading raw data from {source_path}...")
    df = pd.read_csv(source_path)

    # Standardize SMILES column
    smiles_col = None
    for c in ["can_smiles", "smiles", "smi"]:
        if c in df.columns:
            smiles_col = c
            break
    if smiles_col is None:
        raise ValueError(f"No SMILES column found in {source_path}")

    if smiles_col != "can_smiles":
        df = df.rename(columns={smiles_col: "can_smiles"})

    # Ensure UID column
    if "uid" not in df.columns:
        df["uid"] = [f"{dataset_name}:{i:09d}" for i in range(len(df))]

    print(f"   Loaded {len(df)} samples")

    # Step 2: Featurize
    print(f"\n[2/6] Computing RDKit 2D descriptors...")
    smiles = df["can_smiles"].tolist()
    try:
        X = featurize_rdkit2d(smiles)
        print(f"   Computed {X.shape[1]} features")
    except Exception as e:
        print(f"   [ERROR] Featurization failed: {e}")
        raise

    # Step 3: Extract labels
    print(f"\n[3/6] Extracting labels...")
    label_col = config["label_col"]
    if label_col is None:
        # Auto-detect: first numeric column after reserved names
        reserved = {"uid", "can_smiles", "smiles", "scaffold"}
        numeric_cols = [
            c for c in df.columns
            if c.lower() not in reserved
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not numeric_cols:
            raise ValueError(f"No numeric label column found in {source_path}")
        label_col = numeric_cols[0]  # Use first numeric column
        print(f"   Auto-detected label column: {label_col}")

    labels = df[label_col].values

    # For binary classification, ensure 0/1 labels
    if config["task_type"] == "binary":
        unique_labels = np.unique(labels[~np.isnan(labels)])
        if set(unique_labels) != {0, 1} and set(unique_labels) != {0.0, 1.0}:
            print(f"   [WARNING] Labels are not 0/1: {unique_labels}")
            # Try to convert
            labels = (labels > 0.5).astype(int)

    print(f"   Task: {config['task_type']}")
    print(f"   Labels: {labels.shape}, range=[{np.nanmin(labels):.2f}, {np.nanmax(labels):.2f}]")
    if config["task_type"] == "binary":
        print(f"   Positive rate: {np.nanmean(labels):.2%}")

    # Step 4: Compute scaffolds for cohorts
    print(f"\n[4/6] Computing Murcko scaffolds for cohorts...")
    scaffolds = []
    for smi in smiles:
        try:
            scaf = murcko_scaffold(smi)
            scaffolds.append(scaf if scaf else "no_scaffold")
        except Exception:
            scaffolds.append("no_scaffold")

    df["scaffold"] = scaffolds
    unique_scaffolds = df["scaffold"].nunique()
    print(f"   Found {unique_scaffolds} unique scaffolds")

    # Step 5: Create scaffold-aware splits
    print(f"\n[5/6] Creating scaffold-aware train/cal/test splits...")
    splits_df = scaffoldwise_split(
        df,
        scaffold_col="scaffold",
        uid_col="uid",
        frac_train=0.60,
        frac_cal=0.20,
        frac_test=0.20,
        seed=seed,
    )

    # Merge splits back to original dataframe
    df = df.merge(splits_df, on="uid", how="left")

    # Count splits
    split_counts = df["split"].value_counts()
    print(f"   Train: {split_counts.get('train', 0)} samples")
    print(f"   Cal: {split_counts.get('cal', 0)} samples")
    print(f"   Test: {split_counts.get('test', 0)} samples")

    # Step 6: Save processed data
    print(f"\n[6/6] Saving processed data to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save features
    np.save(output_dir / "features.npy", X)
    print(f"   Saved features.npy ({X.shape})")

    # Save labels
    np.save(output_dir / "labels.npy", labels)
    print(f"   Saved labels.npy ({labels.shape})")

    # Save cohorts (scaffold names)
    cohorts = np.array(scaffolds, dtype=object)
    np.save(output_dir / "cohorts.npy", cohorts)
    print(f"   Saved cohorts.npy ({cohorts.shape})")

    # Save splits
    splits_df.to_csv(output_dir / "splits.csv", index=False)
    print(f"   Saved splits.csv ({len(splits_df)} rows)")

    # Save metadata
    metadata = {
        "dataset": dataset_name,
        "task_type": config["task_type"],
        "n_samples": len(df),
        "n_features": X.shape[1],
        "n_cohorts": unique_scaffolds,
        "split_counts": {
            "train": int(split_counts.get("train", 0)),
            "cal": int(split_counts.get("cal", 0)),
            "test": int(split_counts.get("test", 0)),
        },
        "seed": seed,
    }

    import json
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"   Saved metadata.json")

    print(f"\n[SUCCESS] {dataset_name.upper()} preprocessing complete!")
    print(f"   Output: {output_dir}")

    return metadata


def main():
    """Main CLI."""
    parser = argparse.ArgumentParser(
        description="Preprocess molecular datasets for ShiftBench"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIG.keys()) + ["all"],
        help="Dataset to preprocess (or 'all' for all datasets)",
    )
    parser.add_argument(
        "--ravel-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "ravel",
        help="Path to ravel directory (default: ../ravel)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "processed",
        help="Output directory (default: data/processed)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits (default: 42)",
    )

    args = parser.parse_args()

    # Check ravel data directory
    ravel_data_dir = args.ravel_dir / "data" / "raw"
    if not ravel_data_dir.exists():
        print(f"[ERROR] RAVEL data directory not found: {ravel_data_dir}")
        print("        Ensure --ravel-dir points to the ravel project root")
        return 1

    # Process datasets
    datasets_to_process = (
        list(DATASET_CONFIG.keys()) if args.dataset == "all" else [args.dataset]
    )

    results = []
    for dataset_name in datasets_to_process:
        try:
            output_dir = args.output_dir / dataset_name
            metadata = preprocess_molecular_dataset(
                dataset_name,
                ravel_data_dir,
                output_dir,
                seed=args.seed,
            )
            results.append((dataset_name, "SUCCESS", metadata))
        except Exception as e:
            print(f"\n[ERROR] Failed to process {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((dataset_name, "FAILED", str(e)))

    # Print summary
    print("\n" + "="*80)
    print("PREPROCESSING SUMMARY")
    print("="*80)
    for dataset_name, status, info in results:
        print(f"  {dataset_name:<20} {status}")
        if status == "SUCCESS" and isinstance(info, dict):
            print(f"      {info['n_samples']} samples, {info['n_features']} features, {info['n_cohorts']} cohorts")

    n_success = sum(1 for _, status, _ in results if status == "SUCCESS")
    print(f"\nTotal: {n_success}/{len(results)} datasets processed successfully")

    return 0 if n_success == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
