"""
Replace Oracle Predictions with Real Model Predictions

This script:
1. Loads trained models from models/models/{dataset}/{model}.pkl
2. Generates predictions on BOTH calibration and test sets
3. Saves predictions in format compatible with existing RAVEL pipeline
4. Creates mapping file for easy reference

Critical: All current findings (H1, H2, H3) may be artifacts of oracle setup.
This script enables re-running experiments with real model predictions.

Usage:
    python scripts/replace_oracle_predictions.py
    python scripts/replace_oracle_predictions.py --datasets bace,bbbp
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# Core 10 strong models (excluding civil_comments and amazon per user request)
CORE_10_STRONG = [
    # Molecular (4)
    ("bace", "molecular", "rf"),
    ("bbbp", "molecular", "rf"),
    ("clintox", "molecular", "rf"),
    ("esol", "molecular", "rf"),
    # Tabular (4)
    ("adult", "tabular", "lr"),
    ("compas", "tabular", "lr"),
    ("bank", "tabular", "lr"),
    ("german_credit", "tabular", "lr"),
    # Text (2)
    ("imdb", "text", "lr"),
    ("yelp", "text", "lr"),
]


def load_dataset(dataset_name: str, data_dir: str = "data/processed") -> Tuple:
    """
    Load processed dataset.

    Returns:
        X, y, cohorts, splits (DataFrame with 'split' column)
    """
    dataset_path = Path(data_dir) / dataset_name

    # Load features
    if (dataset_path / "features.npy").exists():
        X = np.load(dataset_path / "features.npy", allow_pickle=True)
    else:
        raise FileNotFoundError(f"Features not found: {dataset_path / 'features.npy'}")

    # Load labels
    if (dataset_path / "labels.npy").exists():
        y = np.load(dataset_path / "labels.npy")
    else:
        raise FileNotFoundError(f"Labels not found: {dataset_path / 'labels.npy'}")

    # Load cohorts
    if (dataset_path / "cohorts.npy").exists():
        cohorts = np.load(dataset_path / "cohorts.npy", allow_pickle=True)
    else:
        raise FileNotFoundError(f"Cohorts not found: {dataset_path / 'cohorts.npy'}")

    # Load splits
    if (dataset_path / "splits.csv").exists():
        splits = pd.read_csv(dataset_path / "splits.csv")
    else:
        raise FileNotFoundError(f"Splits not found: {dataset_path / 'splits.csv'}")

    # Handle ESOL binarization (consistent with training)
    if dataset_name == "esol":
        print(f"  ESOL: Binarizing using median split on training data")
        train_mask = (splits["split"] == "train")
        median_threshold = np.median(y[train_mask])
        y = (y > median_threshold).astype(int)
        print(f"    Threshold: {median_threshold:.3f}")

    return X, y, cohorts, splits


def generate_predictions_for_dataset(
    dataset_name: str,
    domain: str,
    model_name: str,
    models_dir: str = "models/models",
    predictions_dir: str = "models/predictions",
    data_dir: str = "data/processed"
) -> bool:
    """
    Generate predictions for calibration and test sets using trained model.

    Args:
        dataset_name: Dataset name (e.g., 'bace')
        domain: Domain type (molecular, tabular, text)
        model_name: Model name (e.g., 'rf', 'lr')
        models_dir: Directory containing trained models
        predictions_dir: Directory to save predictions
        data_dir: Directory containing processed datasets

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Generating predictions: {dataset_name} ({domain}) - {model_name}")
    print(f"{'='*80}")

    # Load trained model
    model_path = Path(models_dir) / dataset_name / f"{model_name}.pkl"
    if not model_path.exists():
        print(f"  ERROR: Model not found at {model_path}")
        return False

    try:
        model = joblib.load(model_path)
        print(f"  Loaded model from {model_path}")
    except Exception as e:
        print(f"  ERROR: Failed to load model: {e}")
        return False

    # Load data
    try:
        X, y, cohorts, splits = load_dataset(dataset_name, data_dir)
    except Exception as e:
        print(f"  ERROR: Failed to load dataset: {e}")
        return False

    # Create masks
    cal_mask = (splits["split"] == "cal").values
    test_mask = (splits["split"] == "test").values

    print(f"  Samples: cal={cal_mask.sum()}, test={test_mask.sum()}")

    # Generate predictions for calibration set
    try:
        if hasattr(model, "predict_proba"):
            preds_proba_cal = model.predict_proba(X[cal_mask])[:, 1]
        else:
            preds_proba_cal = model.decision_function(X[cal_mask])
            # Normalize to [0, 1]
            preds_proba_cal = (preds_proba_cal - preds_proba_cal.min()) / \
                              (preds_proba_cal.max() - preds_proba_cal.min())

        preds_binary_cal = (preds_proba_cal > 0.5).astype(int)

        print(f"  Calibration predictions:")
        print(f"    Binary accuracy: {(preds_binary_cal == y[cal_mask]).mean():.3f}")
        print(f"    Mean probability: {preds_proba_cal.mean():.3f}")
        print(f"    Positive rate (pred): {preds_binary_cal.mean():.3f}")
        print(f"    Positive rate (true): {y[cal_mask].mean():.3f}")

    except Exception as e:
        print(f"  ERROR: Failed to generate calibration predictions: {e}")
        return False

    # Generate predictions for test set
    try:
        if hasattr(model, "predict_proba"):
            preds_proba_test = model.predict_proba(X[test_mask])[:, 1]
        else:
            preds_proba_test = model.decision_function(X[test_mask])
            # Normalize to [0, 1]
            preds_proba_test = (preds_proba_test - preds_proba_test.min()) / \
                               (preds_proba_test.max() - preds_proba_test.min())

        preds_binary_test = (preds_proba_test > 0.5).astype(int)

        print(f"  Test predictions:")
        print(f"    Binary accuracy: {(preds_binary_test == y[test_mask]).mean():.3f}")
        print(f"    Mean probability: {preds_proba_test.mean():.3f}")
        print(f"    Positive rate (pred): {preds_binary_test.mean():.3f}")
        print(f"    Positive rate (true): {y[test_mask].mean():.3f}")

    except Exception as e:
        print(f"  ERROR: Failed to generate test predictions: {e}")
        return False

    # Save predictions
    pred_dir = Path(predictions_dir) / dataset_name
    pred_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Calibration set
        np.save(pred_dir / f"{model_name}_cal_preds_binary.npy", preds_binary_cal)
        np.save(pred_dir / f"{model_name}_cal_preds_proba.npy", preds_proba_cal)

        # Test set
        np.save(pred_dir / f"{model_name}_test_preds_binary.npy", preds_binary_test)
        np.save(pred_dir / f"{model_name}_test_preds_proba.npy", preds_proba_test)

        print(f"  Saved predictions to {pred_dir}")
        print(f"    - {model_name}_cal_preds_binary.npy")
        print(f"    - {model_name}_cal_preds_proba.npy")
        print(f"    - {model_name}_test_preds_binary.npy")
        print(f"    - {model_name}_test_preds_proba.npy")

    except Exception as e:
        print(f"  ERROR: Failed to save predictions: {e}")
        return False

    return True


def create_mapping_file(
    datasets: List[Tuple[str, str, str]],
    predictions_dir: str = "models/predictions",
    output_path: str = "models/prediction_mapping.json"
) -> None:
    """
    Create mapping file for easy reference.

    Args:
        datasets: List of (dataset_name, domain, model_name) tuples
        predictions_dir: Directory containing predictions
        output_path: Path to save mapping JSON
    """
    mapping = {}

    for dataset_name, domain, model_name in datasets:
        pred_path = Path(predictions_dir) / dataset_name / f"{model_name}_cal_preds_binary.npy"
        if pred_path.exists():
            mapping[dataset_name] = {
                "model": model_name,
                "domain": domain,
                "path": str(pred_path),
                "cal_binary": str(Path(predictions_dir) / dataset_name / f"{model_name}_cal_preds_binary.npy"),
                "cal_proba": str(Path(predictions_dir) / dataset_name / f"{model_name}_cal_preds_proba.npy"),
                "test_binary": str(Path(predictions_dir) / dataset_name / f"{model_name}_test_preds_binary.npy"),
                "test_proba": str(Path(predictions_dir) / dataset_name / f"{model_name}_test_preds_proba.npy"),
            }

    # Save mapping
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"\nMapping file saved to: {output_path}")


def replace_all_oracle_predictions(
    datasets: Optional[List[str]] = None,
    models_dir: str = "models/models",
    predictions_dir: str = "models/predictions",
    data_dir: str = "data/processed"
) -> Dict[str, bool]:
    """
    Replace oracle predictions with real model predictions for all datasets.

    Args:
        datasets: List of dataset names to process (None = all core 10)
        models_dir: Directory containing trained models
        predictions_dir: Directory to save predictions
        data_dir: Directory containing processed datasets

    Returns:
        Dictionary mapping dataset name to success status
    """
    print("=" * 80)
    print("REPLACING ORACLE PREDICTIONS WITH REAL MODEL PREDICTIONS")
    print("=" * 80)
    print(f"\nModels directory: {models_dir}")
    print(f"Predictions directory: {predictions_dir}")
    print(f"Data directory: {data_dir}")

    # Filter datasets if specified
    if datasets is not None:
        datasets_to_run = [(name, domain, model) for name, domain, model in CORE_10_STRONG if name in datasets]
    else:
        datasets_to_run = CORE_10_STRONG

    print(f"\nDatasets to process: {len(datasets_to_run)}")
    for name, domain, model in datasets_to_run:
        print(f"  - {name} ({domain}) using {model}")

    # Process all datasets
    results = {}

    for dataset_name, domain, model_name in datasets_to_run:
        success = generate_predictions_for_dataset(
            dataset_name=dataset_name,
            domain=domain,
            model_name=model_name,
            models_dir=models_dir,
            predictions_dir=predictions_dir,
            data_dir=data_dir
        )
        results[dataset_name] = success

    # Create mapping file
    create_mapping_file(
        datasets=datasets_to_run,
        predictions_dir=predictions_dir,
        output_path=Path(predictions_dir).parent / "prediction_mapping.json"
    )

    # Print summary
    print("\n" + "=" * 80)
    print("REPLACEMENT COMPLETE")
    print("=" * 80)

    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]

    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    if successful:
        print(f"  Replaced oracle predictions for {len(successful)} datasets:")
        print(f"  {', '.join(successful)}")

    if failed:
        print(f"\nFailed: {len(failed)}/{len(results)}")
        print(f"  {', '.join(failed)}")

    print("\n[SUCCESS] Oracle predictions replaced with real model predictions!")
    print("\n[NEXT] Next Steps:")
    print("  1. Re-run H1 experiments with real model predictions")
    print("  2. Re-run H2 experiments (distribution shift)")
    print("  3. Re-run H3 experiments (test set validation)")
    print("  4. Compare results with oracle baseline")
    print("  5. Update formal claims if findings differ")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replace oracle predictions with real model predictions"
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models/models",
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default="models/predictions",
        help="Directory to save predictions"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing processed datasets"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of datasets (default: all core 10)"
    )

    args = parser.parse_args()

    # Parse datasets
    datasets = None
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]

    # Replace oracle predictions
    results = replace_all_oracle_predictions(
        datasets=datasets,
        models_dir=args.models_dir,
        predictions_dir=args.predictions_dir,
        data_dir=args.data_dir
    )
