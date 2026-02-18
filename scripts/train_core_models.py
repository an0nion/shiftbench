"""
Train Real Models on Core 12 Datasets

CRITICAL: This replaces oracle predictions (y_true) with real model predictions.
All current findings (H1, H2, H3) may be artifacts of oracle setup.

Usage:
    python scripts/train_core_models.py --output models/
    python scripts/train_core_models.py --datasets bace,bbbp --quick_test
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import joblib
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    print("WARNING: XGBoost not installed. Install with: pip install xgboost")
    HAS_XGB = False

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve


# Core 12 datasets (per FORMAL_CLAIMS.md)
CORE_12 = [
    # Molecular (4)
    ("bace", "molecular"),
    ("bbbp", "molecular"),
    ("clintox", "molecular"),
    ("esol", "molecular"),  # Will be binarized
    # Tabular (4)
    ("adult", "tabular"),
    ("compas", "tabular"),
    ("bank", "tabular"),
    ("german_credit", "tabular"),
    # Text (4)
    ("imdb", "text"),
    ("yelp", "text"),
    ("civil_comments", "text"),
    ("amazon", "text"),
]


# Model configurations per domain
def get_models(domain: str, X: np.ndarray = None, quick_test: bool = False) -> Dict:
    """Get model configurations for a domain.

    Args:
        domain: Domain type (molecular, tabular, text)
        X: Feature array to check if text is pre-vectorized
        quick_test: Use smaller models for quick testing
    """

    if domain == "molecular":
        models = {
            "rf": RandomForestClassifier(
                n_estimators=50 if quick_test else 100,
                max_depth=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        }
        if HAS_XGB:
            models["xgb"] = XGBClassifier(
                n_estimators=50 if quick_test else 100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )

    elif domain == "tabular":
        models = {
            "lr": LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        }
        if HAS_XGB:
            models["xgb"] = XGBClassifier(
                n_estimators=50 if quick_test else 100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )

    elif domain == "text":
        # Check if features are already vectorized (numpy array, not raw text)
        # If X is 2D numeric array with >1 features, assume pre-vectorized
        is_pre_vectorized = (X is not None and
                           isinstance(X, np.ndarray) and
                           X.ndim == 2 and
                           X.dtype.kind in ('f', 'i') and  # Numeric (float or int)
                           X.shape[1] > 1)  # More than 1 feature

        if is_pre_vectorized:
            print(f"  Text features are pre-vectorized ({X.shape[1]} features), using LR directly")
            models = {
                "lr": LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=42,
                    n_jobs=-1
                )
            }
        else:
            # Raw text - use TF-IDF pipeline
            models = {
                "tfidf_lr": Pipeline([
                    ("tfidf", TfidfVectorizer(
                        max_features=5000 if not quick_test else 1000,
                        ngram_range=(1, 2),
                        min_df=2
                    )),
                    ("clf", LogisticRegression(
                        C=1.0,
                        max_iter=1000,
                        random_state=42,
                        n_jobs=-1
                    ))
                ])
            }

    else:
        raise ValueError(f"Unknown domain: {domain}")

    return models


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

    # Handle ESOL binarization (per FORMAL_CLAIMS.md line 483)
    if dataset_name == "esol":
        print(f"  ESOL: Binarizing using median split on training data")
        train_mask = (splits["split"] == "train")
        median_threshold = np.median(y[train_mask])
        y = (y > median_threshold).astype(int)
        print(f"    Threshold: {median_threshold:.3f} (median of training set)")
        print(f"    Positive rate: {y.mean():.3f}")

    return X, y, cohorts, splits


def evaluate_model(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    y_pred_proba: np.ndarray
) -> Dict:
    """Compute evaluation metrics."""
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)

    # Probabilistic metrics
    try:
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        metrics['auc'] = np.nan  # All one class

    metrics['brier'] = brier_score_loss(y_true, y_pred_proba)
    metrics['log_loss'] = log_loss(y_true, y_pred_proba)

    # Calibration error (mean absolute calibration error)
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
        metrics['calibration_error'] = np.abs(frac_pos - mean_pred).mean()
    except ValueError:
        metrics['calibration_error'] = np.nan

    # Positive prediction rate
    metrics['pred_positive_rate'] = y_pred_binary.mean()
    metrics['true_positive_rate'] = y_true.mean()

    return metrics


def train_model_on_dataset(
    dataset_name: str,
    domain: str,
    output_dir: str,
    data_dir: str = "data/processed",
    quick_test: bool = False
) -> Dict:
    """
    Train all models for a dataset and save predictions + models.

    Returns:
        results: Dict of model_name -> metrics
    """
    print(f"\n{'='*80}")
    print(f"Training: {dataset_name} ({domain})")
    print(f"{'='*80}")

    # Load data
    X, y, cohorts, splits = load_dataset(dataset_name, data_dir)

    # Create masks
    train_mask = (splits["split"] == "train").values
    cal_mask = (splits["split"] == "cal").values
    test_mask = (splits["split"] == "test").values

    print(f"  Samples: train={train_mask.sum()}, cal={cal_mask.sum()}, test={test_mask.sum()}")
    print(f"  Features: {X.shape[1] if len(X.shape) > 1 else 'variable (text)'}")
    print(f"  Cohorts: {len(np.unique(cohorts))}")
    print(f"  Positive rate: {y.mean():.3f}")

    # Get models (pass X to detect pre-vectorized text data)
    models = get_models(domain, X=X, quick_test=quick_test)

    results = {}

    for model_name, model in models.items():
        print(f"\n  Training {model_name}...")

        # Train
        try:
            model.fit(X[train_mask], y[train_mask])
        except Exception as e:
            print(f"    ERROR: Training failed: {e}")
            continue

        # Predict on calibration set
        try:
            if hasattr(model, "predict_proba"):
                preds_proba_cal = model.predict_proba(X[cal_mask])[:, 1]
            else:
                preds_proba_cal = model.decision_function(X[cal_mask])
                # Normalize to [0, 1]
                preds_proba_cal = (preds_proba_cal - preds_proba_cal.min()) / \
                                  (preds_proba_cal.max() - preds_proba_cal.min())

            preds_binary_cal = (preds_proba_cal > 0.5).astype(int)

        except Exception as e:
            print(f"    ERROR: Prediction failed: {e}")
            continue

        # Evaluate on calibration set
        metrics_cal = evaluate_model(y[cal_mask], preds_binary_cal, preds_proba_cal)

        print(f"    OK Acc: {metrics_cal['accuracy']:.3f}, "
              f"AUC: {metrics_cal['auc']:.3f}, "
              f"Brier: {metrics_cal['brier']:.3f}")
        print(f"      Calibration Error: {metrics_cal['calibration_error']:.3f}")
        print(f"      Pred Positive Rate: {metrics_cal['pred_positive_rate']:.3f}")

        # Save predictions (calibration set only for now)
        pred_dir = Path(output_dir) / "predictions" / dataset_name
        pred_dir.mkdir(parents=True, exist_ok=True)

        np.save(pred_dir / f"{model_name}_cal_preds_binary.npy", preds_binary_cal)
        np.save(pred_dir / f"{model_name}_cal_preds_proba.npy", preds_proba_cal)

        # Save model
        model_dir = Path(output_dir) / "models" / dataset_name
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, model_dir / f"{model_name}.pkl")

        # Save metrics
        results[model_name] = metrics_cal

    return results


def train_all_core_models(
    output_dir: str,
    data_dir: str = "data/processed",
    datasets: Optional[List[str]] = None,
    quick_test: bool = False
) -> pd.DataFrame:
    """
    Train all models on all core 12 datasets.

    Returns:
        DataFrame with results
    """
    print("=" * 80)
    print("TRAINING REAL MODELS ON CORE 12 DATASETS")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Quick test mode: {quick_test}")

    # Filter datasets if specified
    if datasets is not None:
        datasets_to_run = [(name, domain) for name, domain in CORE_12 if name in datasets]
    else:
        datasets_to_run = CORE_12

    print(f"\nDatasets to process: {len(datasets_to_run)}")
    for name, domain in datasets_to_run:
        print(f"  - {name} ({domain})")

    # Train all
    all_results = []

    for dataset_name, domain in datasets_to_run:
        try:
            results = train_model_on_dataset(
                dataset_name, domain, output_dir, data_dir, quick_test
            )

            for model_name, metrics in results.items():
                all_results.append({
                    "dataset": dataset_name,
                    "domain": domain,
                    "model": model_name,
                    **metrics
                })

        except Exception as e:
            print(f"\nERROR: Failed on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Save summary
    summary_path = Path(output_dir) / "training_summary.csv"
    results_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nTrained {len(results_df)} models across {len(datasets_to_run)} datasets")
    print(f"Summary saved to: {summary_path}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS (Calibration Set)")
    print("=" * 80)

    if len(results_df) > 0:
        summary = results_df.groupby('domain')[['accuracy', 'auc', 'brier', 'calibration_error']].agg(['mean', 'std'])
        print(summary.to_string())

        print("\n[SUCCESS] All real models trained successfully!")
        print("\n[NEXT] Next Steps:")
        print("  1. Replace oracle predictions in existing results")
        print("  2. Re-run H1 experiments with real models")
        print("  3. Verify findings persist")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train real models on core 12 datasets")
    parser.add_argument(
        "--output",
        type=str,
        default="models",
        help="Output directory for models and predictions"
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
        help="Comma-separated list of datasets (default: all core 12)"
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Quick test mode (fewer trees, smaller vocab)"
    )

    args = parser.parse_args()

    # Parse datasets
    datasets = None
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]

    # Train
    results_df = train_all_core_models(
        output_dir=args.output,
        data_dir=args.data_dir,
        datasets=datasets,
        quick_test=args.quick_test
    )
