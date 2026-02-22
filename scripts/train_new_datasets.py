"""
Train models and generate predictions for all processed datasets
that do not yet have cal+test predictions.

Handles:
  - Regression labels (esol-style): median-split binarization
  - Multi-label with NaN: replace NaN with 0 (conservative)
  - Vision datasets (512-feature embeddings): LR directly
  - Partial predictions (amazon, civil_comments): generate test/train preds

After running, updates models/prediction_mapping.json with all new datasets.

Usage:
    python scripts/train_new_datasets.py
    python scripts/train_new_datasets.py --datasets twitter,diabetes
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

shift_bench_dir = Path(__file__).resolve().parent.parent
DATA_DIR   = shift_bench_dir / "data" / "processed"
MODEL_DIR  = shift_bench_dir / "models"
PRED_DIR   = MODEL_DIR / "predictions"
MAP_FILE   = shift_bench_dir / "models" / "prediction_mapping.json"

# -------------------------------------------------------------------------
# Dataset configs for everything not yet in prediction_mapping.json
# -------------------------------------------------------------------------
NEW_DATASETS = {
    # --- molecular (RF, 217 features, scaffold cohorts) ---
    "freesolv":         {"domain": "molecular", "model": "rf", "binarize": "median"},
    "lipophilicity":    {"domain": "molecular", "model": "rf", "binarize": "median"},
    "sider":            {"domain": "molecular", "model": "rf", "binarize": None},
    "tox21":            {"domain": "molecular", "model": "rf", "binarize": "nan_zero"},
    "toxcast":          {"domain": "molecular", "model": "rf", "binarize": "nan_zero"},
    "muv":              {"domain": "molecular", "model": "rf", "binarize": "nan_zero"},
    "molhiv":           {"domain": "molecular", "model": "rf", "binarize": None},
    # --- tabular (LR, demographic cohorts) ---
    "diabetes":         {"domain": "tabular",   "model": "lr", "binarize": None},
    "heart_disease":    {"domain": "tabular",   "model": "lr", "binarize": None},
    "student_performance": {"domain": "tabular","model": "lr", "binarize": None},
    # --- text (LR, pre-vectorized TF-IDF) ---
    "amazon":           {"domain": "text",      "model": "lr", "binarize": None},
    "civil_comments":   {"domain": "text",      "model": "lr", "binarize": None},
    "twitter":          {"domain": "text",      "model": "lr", "binarize": None},
    # --- vision (LR, 512-dim CNN embeddings, already have full predictions) ---
    # camelyon17, waterbirds: just need mapping update, skip training

    # ---- NEW batch (Session 9): 12 more datasets ----
    # molecular
    "hiv":              {"domain": "molecular", "model": "rf", "binarize": None},
    "qm7":              {"domain": "molecular", "model": "rf", "binarize": None},
    "delaney":          {"domain": "molecular", "model": "rf", "binarize": None},
    "sampl":            {"domain": "molecular", "model": "rf", "binarize": None},
    # tabular
    "wine_quality":     {"domain": "tabular",   "model": "lr", "binarize": None},
    "online_shoppers":  {"domain": "tabular",   "model": "lr", "binarize": None},
    "communities_crime":{"domain": "tabular",   "model": "lr", "binarize": None},
    "mushroom":         {"domain": "tabular",   "model": "lr", "binarize": None},
    # text (pre-vectorized TF-IDF in features.npy)
    "ag_news":          {"domain": "text",      "model": "lr", "binarize": None},
    "dbpedia":          {"domain": "text",      "model": "lr", "binarize": None},
    "imdb_genre":       {"domain": "text",      "model": "lr", "binarize": None},
    "sst2":             {"domain": "text",      "model": "lr", "binarize": None},
}

# Datasets that already have full predictions — just add to mapping
MAPPING_ONLY = {
    "camelyon17": {"domain": "vision",  "model": "lr"},
    "waterbirds":  {"domain": "vision",  "model": "lr"},
}


def load_dataset(name):
    """Load features, labels (binarized), splits."""
    base = DATA_DIR / name
    X       = np.load(base / "features.npy",  allow_pickle=True)
    y_raw   = np.load(base / "labels.npy",    allow_pickle=True)
    cohorts = np.load(base / "cohorts.npy",   allow_pickle=True)
    splits  = pd.read_csv(base / "splits.csv")

    cfg = NEW_DATASETS.get(name, {})
    binarize = cfg.get("binarize")

    # --- binarize labels ---
    if binarize == "median":
        # Continuous regression target: median-split on training data
        train_mask = splits["split"].values == "train"
        threshold = np.nanmedian(y_raw[train_mask])
        y = (y_raw > threshold).astype(int)
        print(f"  median-split binarize: threshold={threshold:.4f}, pos_rate={y.mean():.3f}")
    elif binarize == "nan_zero":
        # Multi-label with NaN: replace NaN with 0 (conservative)
        y = np.nan_to_num(y_raw, nan=0.0).astype(int)
        nan_frac = np.isnan(y_raw).mean()
        print(f"  NaN->0 binarize: nan_frac={nan_frac:.3f}, pos_rate={y.mean():.3f}")
    else:
        y = y_raw.astype(int)
        print(f"  labels as-is: pos_rate={y.mean():.3f}")

    return X, y, cohorts, splits


def build_model(cfg):
    if cfg["model"] == "rf":
        return RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        )
    else:
        return LogisticRegression(C=1.0, max_iter=1000, random_state=42, n_jobs=-1)


def train_and_save(name, cfg):
    print(f"\n{'='*60}")
    print(f"  {name.upper()} ({cfg['domain']})")
    print(f"{'='*60}")

    X, y, cohorts, splits = load_dataset(name)
    split_arr = splits["split"].values
    train_mask = split_arr == "train"
    cal_mask   = split_arr == "cal"
    test_mask  = split_arr == "test"
    print(f"  train={train_mask.sum()} cal={cal_mask.sum()} test={test_mask.sum()}")
    print(f"  features={X.shape[1]} cohorts={len(np.unique(cohorts))}")

    # Check if test preds already exist (partial case: amazon, civil_comments)
    model_name = cfg["model"]
    out_dir = PRED_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    test_bin_path = out_dir / f"{model_name}_test_preds_binary.npy"

    if test_bin_path.exists():
        print(f"  Test predictions already exist at {test_bin_path.name} -- re-training to fill train split")

    # Train
    model = build_model(cfg)
    X_train, y_train = X[train_mask], y[train_mask]
    if len(X_train) == 0:
        print("  ERROR: no training samples. Skipping.")
        return None

    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"  ERROR fitting: {e}")
        return None

    auc_scores = {}
    for split_name, mask in [("train", train_mask), ("cal", cal_mask), ("test", test_mask)]:
        if mask.sum() == 0:
            continue
        proba = model.predict_proba(X[mask])[:, 1]
        binary = (proba >= 0.5).astype(int)
        np.save(out_dir / f"{model_name}_{split_name}_preds_binary.npy", binary)
        np.save(out_dir / f"{model_name}_{split_name}_preds_proba.npy", proba)
        y_split = y[mask]
        if len(np.unique(y_split)) == 2:
            try:
                auc = roc_auc_score(y_split, proba)
                auc_scores[split_name] = auc
            except Exception:
                pass
        print(f"  {split_name}: n={mask.sum()} pos={y_split.mean():.3f} "
              f"pred_pos={binary.mean():.3f}"
              + (f" AUC={auc_scores.get(split_name, float('nan')):.3f}" if split_name in auc_scores else ""))

    return {
        "model": model_name,
        "cal_binary":  str(out_dir / f"{model_name}_cal_preds_binary.npy"),
        "cal_proba":   str(out_dir / f"{model_name}_cal_preds_proba.npy"),
        "test_binary": str(out_dir / f"{model_name}_test_preds_binary.npy"),
        "test_proba":  str(out_dir / f"{model_name}_test_preds_proba.npy"),
    }


def update_mapping(new_entries):
    """Add new datasets to prediction_mapping.json."""
    if MAP_FILE.exists():
        with open(MAP_FILE) as f:
            mapping = json.load(f)
    else:
        mapping = {}

    for name, entry in new_entries.items():
        cfg = {**NEW_DATASETS.get(name, {}), **MAPPING_ONLY.get(name, {})}
        model_name = cfg.get("model", "lr")
        out_dir = PRED_DIR / name
        mapping[name] = {
            "model": model_name,
            "domain": cfg.get("domain", "unknown"),
            "path": str(out_dir / f"{model_name}_cal_preds_binary.npy"),
            "cal_binary":  str(out_dir / f"{model_name}_cal_preds_binary.npy"),
            "cal_proba":   str(out_dir / f"{model_name}_cal_preds_proba.npy"),
            "test_binary": str(out_dir / f"{model_name}_test_preds_binary.npy"),
            "test_proba":  str(out_dir / f"{model_name}_test_preds_proba.npy"),
        }

    with open(MAP_FILE, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\nUpdated {MAP_FILE} -> {len(mapping)} datasets total")
    return mapping


def main(datasets_filter=None):
    to_train = NEW_DATASETS
    if datasets_filter:
        to_train = {k: v for k, v in to_train.items() if k in datasets_filter}

    print(f"Training {len(to_train)} datasets: {list(to_train.keys())}")

    new_entries = {}
    failed = []

    for name, cfg in to_train.items():
        try:
            result = train_and_save(name, cfg)
            if result:
                new_entries[name] = result
        except Exception as e:
            import traceback
            print(f"\nERROR on {name}: {e}")
            traceback.print_exc()
            failed.append(name)

    # Add mapping-only datasets (camelyon17, waterbirds) if they have predictions
    for name, cfg in MAPPING_ONLY.items():
        model_name = cfg["model"]
        test_path = PRED_DIR / name / f"{model_name}_test_preds_binary.npy"
        if test_path.exists():
            new_entries[name] = {"model": model_name}
            print(f"\n{name}: adding to mapping (predictions exist)")
        else:
            print(f"\n{name}: skipping (no test predictions found)")

    mapping = update_mapping(new_entries)

    print("\n" + "="*60)
    print(f"DONE: {len(new_entries)} datasets added to mapping")
    print(f"Total in mapping: {len(mapping)}")
    if failed:
        print(f"Failed: {failed}")
    print("\nNext: run cross-domain benchmark on all mapped datasets")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated list (default: all new datasets)")
    args = parser.parse_args()
    datasets_filter = [d.strip() for d in args.datasets.split(",")] if args.datasets else None
    main(datasets_filter)
