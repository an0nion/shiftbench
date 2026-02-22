"""
Preprocess 3 new datasets to push total from 37 -> 40:
  1. credit_default  (tabular)  - Taiwan credit card defaults, temporal shift
  2. hate_speech     (text)     - tweet_eval/hate, binary hate detection
  3. rotten_tomatoes (text)     - movie review sentiment, category cohorts

Usage:
    python scripts/preprocess_new_datasets_v2.py
    python scripts/preprocess_new_datasets_v2.py --dataset credit_default
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ── shared helpers ───────────────────────────────────────────────────────────

def make_splits(n, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    train_end = int(0.6 * n)
    cal_end   = int(0.8 * n)
    split = np.empty(n, dtype=object)
    split[idx[:train_end]]        = "train"
    split[idx[train_end:cal_end]] = "cal"
    split[idx[cal_end:]]          = "test"
    return split


def save_dataset(name, X, y, cohorts, splits_arr, metadata, out_root):
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "features.npy", X)
    np.save(out / "labels.npy", y.astype(float))
    np.save(out / "cohorts.npy", cohorts)
    pd.DataFrame({"split": splits_arr}).to_csv(out / "splits.csv", index=False)
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved {name}: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(np.unique(cohorts))} cohorts, pos_rate={y.mean():.3f}")


def train_and_save_preds(name, X, y, splits_arr, pred_root, model_type="lr"):
    train_mask = splits_arr == "train"
    cal_mask   = splits_arr == "cal"
    test_mask  = splits_arr == "test"

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                              random_state=42)
    clf.fit(X[train_mask], y[train_mask])

    cal_proba   = clf.predict_proba(X[cal_mask])[:, 1]
    cal_binary  = clf.predict(X[cal_mask])
    test_proba  = clf.predict_proba(X[test_mask])[:, 1]
    test_binary = clf.predict(X[test_mask])

    auc_cal  = roc_auc_score(y[cal_mask],  cal_proba)
    auc_test = roc_auc_score(y[test_mask], test_proba)
    print(f"  AUC: cal={auc_cal:.3f}, test={auc_test:.3f}")

    out = pred_root / name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "lr_cal_preds_binary.npy",  cal_binary)
    np.save(out / "lr_cal_preds_proba.npy",   cal_proba)
    np.save(out / "lr_test_preds_binary.npy", test_binary)
    np.save(out / "lr_test_preds_proba.npy",  test_proba)
    return auc_cal, auc_test


# ── dataset builders ─────────────────────────────────────────────────────────

def build_credit_default(out_root, pred_root, seed=42):
    """Taiwan credit card default dataset: demographic shift.

    Cohorts: education x marriage cross (4-6 groups).
    """
    print("\n=== Building credit_default ===")
    zip_path = ROOT / "data" / "raw" / "credit_default.zip"
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("default of credit card clients.xls") as f:
            df = pd.read_excel(f, header=1)  # row 0 is extra header

    print(f"  Raw shape: {df.shape}, columns: {df.columns[:8].tolist()}...")

    # Target column
    target_col = [c for c in df.columns if "default" in c.lower()][0]
    y = df[target_col].values.astype(int)

    # Feature columns: all numeric except ID and target
    drop_cols = {target_col, "ID"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X_raw = df[feature_cols].fillna(0).values.astype(float)

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Cohorts: EDUCATION (1=grad, 2=university, 3=high school, 4=other)
    # x MARRIAGE (1=married, 2=single, 3=other) -> 6 main groups
    edu_col = [c for c in feature_cols if "EDUC" in c.upper()]
    mar_col = [c for c in feature_cols if "MARR" in c.upper()]
    edu_idx = feature_cols.index(edu_col[0]) if edu_col else None
    mar_idx = feature_cols.index(mar_col[0]) if mar_col else None

    if edu_idx is not None and mar_idx is not None:
        edu = df[edu_col[0]].clip(1, 4).astype(int).values
        mar = df[mar_col[0]].clip(1, 3).astype(int).values
        cohorts = np.array([f"edu{e}_mar{m}" for e, m in zip(edu, mar)],
                           dtype=object)
    else:
        # fallback: quartile-based age cohorts
        age_col = [c for c in feature_cols if "AGE" in c.upper()]
        if age_col:
            ages = df[age_col[0]].values
            quartiles = np.percentile(ages, [25, 50, 75])
            q_labels = np.digitize(ages, quartiles)
            cohorts = np.array([f"age_q{q}" for q in q_labels], dtype=object)
        else:
            cohorts = np.zeros(len(y), dtype=object)

    splits = make_splits(len(y), seed)
    unique_c = np.unique(cohorts)
    cohort_counts = {c: int((cohorts == c).sum()) for c in unique_c}
    split_counts  = {s: int((splits == s).sum()) for s in ["train", "cal", "test"]}

    meta = {
        "dataset": "credit_default", "task_type": "binary",
        "shift_type": "demographic", "cohort_definition": "education_marriage",
        "n_samples": len(y), "n_features": X.shape[1],
        "n_cohorts": len(unique_c), "featurizer": "standardscaler",
        "split_counts": split_counts,
        "cohort_info": {"type": "demographic", "n_cohorts": len(unique_c),
                        "cohort_counts": cohort_counts},
        "seed": seed, "source": "UCI/Kaggle Taiwan Credit Card Default"
    }
    save_dataset("credit_default", X, y, cohorts, splits, meta, out_root)
    train_and_save_preds("credit_default", X, y, splits, pred_root)


def build_hate_speech(out_root, pred_root, seed=42):
    """tweet_eval/hate - binary hate speech detection, temporal cohorts."""
    print("\n=== Building hate_speech ===")
    from datasets import load_dataset
    train_ds = load_dataset("tweet_eval", "hate", split="train")
    test_ds  = load_dataset("tweet_eval", "hate", split="test")

    texts  = list(train_ds["text"])  + list(test_ds["text"])
    labels = np.array(list(train_ds["label"]) + list(test_ds["label"]), dtype=int)
    # label: 0=non-hate, 1=hate (binary)

    print(f"  Texts: {len(texts)}, pos_rate={labels.mean():.3f}")
    vec = TfidfVectorizer(max_features=5000, sublinear_tf=True,
                          min_df=2, ngram_range=(1, 2))
    X = vec.fit_transform(texts).toarray()
    print(f"  TF-IDF shape: {X.shape}")

    # Temporal cohorts: 5 equal buckets
    n_buckets = 5
    cohorts = np.array([str(i * n_buckets // len(texts))
                        for i in range(len(texts))], dtype=object)

    splits = make_splits(len(texts), seed)
    cohort_counts = {str(i): int((cohorts == str(i)).sum()) for i in range(n_buckets)}
    split_counts  = {s: int((splits == s).sum()) for s in ["train", "cal", "test"]}

    meta = {
        "dataset": "hate_speech", "task_type": "binary",
        "shift_type": "temporal", "cohort_definition": "temporal_buckets",
        "n_samples": len(texts), "n_features": X.shape[1],
        "n_cohorts": n_buckets, "featurizer": "tfidf",
        "max_features": 5000, "split_counts": split_counts,
        "cohort_info": {"type": "temporal", "n_cohorts": n_buckets,
                        "cohort_counts": cohort_counts},
        "seed": seed, "source": "tweet_eval/hate (HuggingFace)"
    }
    save_dataset("hate_speech", X, labels, cohorts, splits, meta, out_root)
    train_and_save_preds("hate_speech", X, labels, splits, pred_root)


def build_rotten_tomatoes(out_root, pred_root, seed=42):
    """Rotten Tomatoes sentiment (movie reviews), category cohorts."""
    print("\n=== Building rotten_tomatoes ===")
    from datasets import load_dataset
    train_ds = load_dataset("rotten_tomatoes", split="train")
    val_ds   = load_dataset("rotten_tomatoes", split="validation")
    test_ds  = load_dataset("rotten_tomatoes", split="test")

    texts  = (list(train_ds["text"]) + list(val_ds["text"])
              + list(test_ds["text"]))
    labels = np.array(
        list(train_ds["label"]) + list(val_ds["label"]) + list(test_ds["label"]),
        dtype=int
    )

    print(f"  Texts: {len(texts)}, pos_rate={labels.mean():.3f}")
    vec = TfidfVectorizer(max_features=5000, sublinear_tf=True,
                          min_df=2, ngram_range=(1, 2))
    X = vec.fit_transform(texts).toarray()
    print(f"  TF-IDF shape: {X.shape}")

    # Cohorts: 5 buckets based on text length (short/medium/long reviews)
    lengths = np.array([len(t.split()) for t in texts])
    quartiles = np.percentile(lengths, [20, 40, 60, 80])
    bucket_ids = np.digitize(lengths, quartiles)  # 0-4
    cohorts = np.array([f"len_q{b}" for b in bucket_ids], dtype=object)

    splits = make_splits(len(texts), seed)
    unique_c = [f"len_q{i}" for i in range(5)]
    cohort_counts = {c: int((cohorts == c).sum()) for c in unique_c}
    split_counts  = {s: int((splits == s).sum()) for s in ["train", "cal", "test"]}

    meta = {
        "dataset": "rotten_tomatoes", "task_type": "binary",
        "shift_type": "category", "cohort_definition": "review_length_quintile",
        "n_samples": len(texts), "n_features": X.shape[1],
        "n_cohorts": len(unique_c), "featurizer": "tfidf",
        "max_features": 5000, "split_counts": split_counts,
        "cohort_info": {"type": "category", "n_cohorts": len(unique_c),
                        "cohort_counts": cohort_counts},
        "seed": seed, "source": "rotten_tomatoes (HuggingFace)"
    }
    save_dataset("rotten_tomatoes", X, labels, cohorts, splits, meta, out_root)
    train_and_save_preds("rotten_tomatoes", X, labels, splits, pred_root)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["credit_default", "hate_speech", "rotten_tomatoes", "all"],
        default="all"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_root  = ROOT / "data" / "processed"
    pred_root = ROOT / "models" / "predictions"

    to_run = (["credit_default", "hate_speech", "rotten_tomatoes"]
              if args.dataset == "all" else [args.dataset])

    for ds in to_run:
        if ds == "credit_default":
            build_credit_default(out_root, pred_root, seed=args.seed)
        elif ds == "hate_speech":
            build_hate_speech(out_root, pred_root, seed=args.seed)
        elif ds == "rotten_tomatoes":
            build_rotten_tomatoes(out_root, pred_root, seed=args.seed)

    print("\n=== Updating prediction_mapping.json ===")
    mapping_path = ROOT / "models" / "prediction_mapping.json"
    with open(mapping_path) as f:
        mapping = json.load(f)

    domain_map = {
        "credit_default": "tabular",
        "hate_speech": "text",
        "rotten_tomatoes": "text",
    }
    for ds in to_run:
        pred_dir = pred_root / ds
        mapping[ds] = {
            "model": "lr",
            "domain": domain_map[ds],
            "path":         str(pred_dir / "lr_cal_preds_binary.npy"),
            "cal_binary":   str(pred_dir / "lr_cal_preds_binary.npy"),
            "cal_proba":    str(pred_dir / "lr_cal_preds_proba.npy"),
            "test_binary":  str(pred_dir / "lr_test_preds_binary.npy"),
            "test_proba":   str(pred_dir / "lr_test_preds_proba.npy"),
        }
        print(f"  Added {ds} ({domain_map[ds]})")

    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)

    total = len(mapping)
    print(f"\nTotal datasets in mapping: {total}")
    print("Done.")


if __name__ == "__main__":
    main()
