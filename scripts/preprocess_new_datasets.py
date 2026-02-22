"""Download and preprocess new datasets to reach PI's 40-dataset target.

This script downloads and preprocesses 12+ new datasets from three domains:

TEXT (HuggingFace):
  ag_news       - 120K news articles, 4-topic category shift
  dbpedia       - 560K (subsampled 40K) entity classification, 14-class cohorts
  trec          - 6K question classification, 6-type cohorts

TABULAR (UCI / public HTTP):
  wine_quality      - 6K red+white wine, red/white + quality cohorts
  online_shoppers   - 12K e-commerce sessions, temporal (month) cohorts
  credit_default    - 30K Taiwan credit card, age-group cohorts
  communities_crime - 2K US communities, region cohorts

MOLECULAR (DeepChem S3 SMILES):
  hiv        - 41K HIV inhibition, scaffold cohorts
  qm7        - 7K quantum chemistry (regression->binarize), scaffold cohorts
  muv_refix  - skip (already have muv)

Usage:
    python scripts/preprocess_new_datasets.py --datasets all
    python scripts/preprocess_new_datasets.py --datasets ag_news,dbpedia
    python scripts/preprocess_new_datasets.py --domain text
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple
from urllib.request import urlretrieve
from urllib.error import URLError
import io

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT.parent / "ravel" / "src"))


# ------------------------------------------------------------------ helpers ---

def make_splits(cohorts: np.ndarray, seed: int = 42,
                train_frac: float = 0.6, cal_frac: float = 0.2
                ) -> pd.DataFrame:
    """Assign each sample to train/cal/test within each cohort."""
    n = len(cohorts)
    split_arr = np.array(["train"] * n, dtype=object)

    rng = np.random.RandomState(seed)
    unique_cohorts = np.unique(cohorts)
    for c in unique_cohorts:
        idx = np.where(cohorts == c)[0]
        rng.shuffle(idx)
        n_c = len(idx)
        n_train = max(1, int(n_c * train_frac))
        n_cal = max(1, int(n_c * cal_frac))
        split_arr[idx[:n_train]] = "train"
        split_arr[idx[n_train:n_train + n_cal]] = "cal"
        split_arr[idx[n_train + n_cal:]] = "test"

    uids = [f"sample_{i:09d}" for i in range(n)]
    return pd.DataFrame({"uid": uids, "split": split_arr})


def save_dataset(name: str, features: np.ndarray, labels: np.ndarray,
                 cohorts: np.ndarray, splits_df: pd.DataFrame) -> None:
    """Save a processed dataset to data/processed/<name>/."""
    out_dir = DATA_PROCESSED / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "features.npy", features.astype(np.float32))
    np.save(out_dir / "labels.npy", labels.astype(int))
    np.save(out_dir / "cohorts.npy", cohorts.astype(str))
    splits_df.to_csv(out_dir / "splits.csv", index=False)
    meta = {
        "n_samples": int(len(features)),
        "n_features": int(features.shape[1]),
        "n_cohorts": int(len(set(cohorts))),
        "label_pos_rate": float(labels.mean()),
        "split_counts": splits_df["split"].value_counts().to_dict(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"  Saved {name}: n={meta['n_samples']}, d={meta['n_features']}, "
          f"cohorts={meta['n_cohorts']}, pos_rate={meta['label_pos_rate']:.3f}")


def _download(url: str, dest: Path) -> bool:
    """Download file if not already present. Returns True on success."""
    if dest.exists():
        return True
    print(f"  Downloading {url} ...")
    try:
        urlretrieve(url, dest)
        return True
    except (URLError, Exception) as e:
        print(f"  [FAILED] download error: {e}")
        return False


# ========================================================================== #
#  TEXT DATASETS
# ========================================================================== #

def process_ag_news() -> bool:
    """AG News: 4-class news topic classification, category shift."""
    name = "ag_news"
    print(f"\n--- {name} ---")
    try:
        from datasets import load_dataset
        ds = load_dataset("ag_news")
    except Exception as e:
        print(f"  [SKIP] HuggingFace load failed: {e}")
        return False

    # Combine train + test
    records = []
    for split_name in ("train", "test"):
        for row in ds[split_name]:
            records.append({"text": row["text"], "label": row["label"]})
    df = pd.DataFrame(records)

    # Classes: 0=World, 1=Sports, 2=Business, 3=Sci-Tech
    class_names = {0: "world", 1: "sports", 2: "business", 3: "sci_tech"}
    df["cohort"] = df["label"].map(class_names)

    # Binary label: 1 if Science/Technology (class 3), 0 otherwise
    df["y"] = (df["label"] == 3).astype(int)

    # TF-IDF features
    tfidf = TfidfVectorizer(max_features=5000, sublinear_tf=True, min_df=3)
    X = tfidf.fit_transform(df["text"]).toarray().astype(np.float32)
    y = df["y"].values
    cohorts = df["cohort"].values
    splits_df = make_splits(cohorts, seed=42)

    save_dataset(name, X, y, cohorts, splits_df)
    return True


def process_dbpedia() -> bool:
    """DBpedia: 14-class entity type classification, category shift (subsampled)."""
    name = "dbpedia"
    print(f"\n--- {name} ---")
    try:
        from datasets import load_dataset
        ds = load_dataset("dbpedia_14")
    except Exception as e:
        print(f"  [SKIP] HuggingFace load failed: {e}")
        return False

    class_names = {
        0: "company", 1: "educational", 2: "artist", 3: "athlete",
        4: "office_holder", 5: "transport", 6: "building", 7: "natural_place",
        8: "village", 9: "animal", 10: "plant", 11: "album",
        12: "film", 13: "written_work"
    }

    records = []
    for split_name in ("train", "test"):
        for row in ds[split_name]:
            records.append({"text": row["content"], "label": row["label"]})
    df = pd.DataFrame(records)

    # Subsample: 3000 per class (42K total) to keep manageable
    rng = np.random.RandomState(42)
    dfs = []
    for cls in range(14):
        subset = df[df["label"] == cls]
        n_take = min(3000, len(subset))
        dfs.append(subset.sample(n_take, random_state=rng.randint(10000)))
    df = pd.concat(dfs).reset_index(drop=True)

    df["cohort"] = df["label"].map(class_names)
    # Binary: 1 if abstract entity (company/office/transport/building = classes 0,4,5,6)
    abstract_classes = {0, 4, 5, 6}
    df["y"] = df["label"].apply(lambda x: int(x in abstract_classes))

    tfidf = TfidfVectorizer(max_features=5000, sublinear_tf=True, min_df=3)
    X = tfidf.fit_transform(df["text"]).toarray().astype(np.float32)
    y = df["y"].values
    cohorts = df["cohort"].values
    splits_df = make_splits(cohorts, seed=42)

    save_dataset(name, X, y, cohorts, splits_df)
    return True


def process_trec() -> bool:
    """TREC: question type classification, 6-type cohorts."""
    name = "trec"
    print(f"\n--- {name} ---")
    try:
        from datasets import load_dataset
        ds = load_dataset("trec")
    except Exception as e:
        print(f"  [SKIP] HuggingFace load failed: {e}")
        return False

    # coarse labels: 0=ABBR, 1=ENTY, 2=DESC, 3=HUM, 4=LOC, 5=NUM
    coarse_names = {0: "abbr", 1: "entity", 2: "desc", 3: "human", 4: "loc", 5: "num"}
    records = []
    for split_name in ("train", "test"):
        for row in ds[split_name]:
            records.append({"text": row["text"], "label": row["coarse_label"]})
    df = pd.DataFrame(records)

    df["cohort"] = df["label"].map(coarse_names)
    # Binary: 1 if factoid (ENTY, LOC, NUM = classes 1,4,5)
    df["y"] = df["label"].apply(lambda x: int(x in {1, 4, 5}))

    tfidf = TfidfVectorizer(max_features=2000, sublinear_tf=True, min_df=2)
    X = tfidf.fit_transform(df["text"]).toarray().astype(np.float32)
    y = df["y"].values
    cohorts = df["cohort"].values
    splits_df = make_splits(cohorts, seed=42)

    save_dataset(name, X, y, cohorts, splits_df)
    return True


def process_sst2() -> bool:
    """SST-2: Stanford Sentiment Treebank, fine-grained sentiment as cohorts."""
    name = "sst2"
    print(f"\n--- {name} ---")
    try:
        from datasets import load_dataset
        ds = load_dataset("sst2")
    except Exception as e:
        print(f"  [SKIP] HuggingFace load failed: {e}")
        return False

    records = []
    for row in ds["train"]:
        # tokens: sentence tokens; label: 0=neg, 1=pos
        # fine-grained sentiment from tree: use label + sentence length as cohort
        records.append({"text": row["sentence"], "label": int(row["label"])})

    df = pd.DataFrame(records).dropna()
    df = df[df["label"].isin([0, 1])]

    # Cohorts by sentence length quartile x sentiment
    lengths = df["text"].str.split().str.len()
    quartiles = pd.qcut(lengths, 4, labels=["short", "medium", "long", "very_long"])
    df["cohort"] = quartiles.astype(str) + "_" + df["label"].map({0: "neg", 1: "pos"})
    df["y"] = df["label"]

    tfidf = TfidfVectorizer(max_features=3000, sublinear_tf=True, min_df=3)
    X = tfidf.fit_transform(df["text"]).toarray().astype(np.float32)
    y = df["y"].values
    cohorts = df["cohort"].values
    splits_df = make_splits(cohorts, seed=42)

    save_dataset(name, X, y, cohorts, splits_df)
    return True


def process_imdb_genre() -> bool:
    """IMDb genre classification (multi-label -> binary) with genre cohorts."""
    name = "imdb_genre"
    print(f"\n--- {name} ---")
    try:
        from datasets import load_dataset
        # Use the standard IMDb sentiment but add genre cohorts from year
        ds = load_dataset("imdb")
    except Exception as e:
        print(f"  [SKIP] HuggingFace load failed: {e}")
        return False

    records = []
    for split_name in ("train", "test"):
        for row in ds[split_name]:
            records.append({"text": row["text"], "label": row["label"]})
    df = pd.DataFrame(records)

    # Cohorts: review length decile (proxy for writing style shift)
    lengths = df["text"].str.split().str.len()
    deciles = pd.qcut(lengths, 8, labels=[f"len_{i}" for i in range(8)], duplicates="drop")
    df["cohort"] = deciles.astype(str)
    df["y"] = df["label"]

    # Subsample to 20K to keep fast
    df = df.sample(20000, random_state=42).reset_index(drop=True)

    tfidf = TfidfVectorizer(max_features=5000, sublinear_tf=True, min_df=5)
    X = tfidf.fit_transform(df["text"]).toarray().astype(np.float32)
    y = df["y"].values
    cohorts = df["cohort"].values
    splits_df = make_splits(cohorts, seed=42)

    save_dataset(name, X, y, cohorts, splits_df)
    return True


# ========================================================================== #
#  TABULAR DATASETS
# ========================================================================== #

def process_wine_quality() -> bool:
    """Wine Quality: red+white wine, quality classification, red/white+quality cohorts."""
    name = "wine_quality"
    print(f"\n--- {name} ---")

    urls = {
        "red": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        "white": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
    }

    dfs = []
    for wtype, url in urls.items():
        dest = DATA_RAW / f"winequality-{wtype}.csv"
        if not _download(url, dest):
            print(f"  [SKIP] Could not download {wtype} wine data")
            return False
        df = pd.read_csv(dest, sep=";")
        df["wine_type"] = wtype
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Binary label: quality >= 6 is "good"
    df["y"] = (df["quality"] >= 6).astype(int)

    # Cohorts: wine_type x quality_band (low<5, medium5-6, high>6)
    def quality_band(q):
        if q <= 4: return "low"
        elif q <= 6: return "medium"
        else: return "high"
    df["cohort"] = df["wine_type"] + "_" + df["quality"].apply(quality_band)

    feature_cols = [c for c in df.columns if c not in ("quality", "y", "cohort", "wine_type")]
    X = df[feature_cols].values.astype(np.float32)
    # Add wine_type as binary feature
    X = np.hstack([X, (df["wine_type"] == "white").values.reshape(-1, 1).astype(np.float32)])

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    y = df["y"].values
    cohorts = df["cohort"].values
    splits_df = make_splits(cohorts, seed=42)

    save_dataset(name, X, y, cohorts, splits_df)
    return True


def process_online_shoppers() -> bool:
    """Online Shoppers: e-commerce session data, temporal (month) cohorts."""
    name = "online_shoppers"
    print(f"\n--- {name} ---")

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
    dest = DATA_RAW / "online_shoppers_intention.csv"
    if not _download(url, dest):
        print("  [SKIP] Could not download online shoppers data")
        return False

    df = pd.read_csv(dest)
    df = df.dropna()

    # Binary: Revenue (True/False already)
    df["y"] = df["Revenue"].astype(int)

    # Cohorts: Month (temporal shift between months)
    df["cohort"] = df["Month"]

    # Encode categorical features
    cat_cols = ["Month", "VisitorType", "Weekend"]
    for c in cat_cols:
        df[c] = df[c].astype("category").cat.codes

    feature_cols = [c for c in df.columns if c not in ("Revenue", "y", "cohort")]
    X = df[feature_cols].values.astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    y = df["y"].values
    cohorts = df["cohort"].values
    splits_df = make_splits(cohorts, seed=42)

    save_dataset(name, X, y, cohorts, splits_df)
    return True


def process_credit_default() -> bool:
    """Credit Default (Taiwan): default prediction, age-group x education cohorts."""
    name = "credit_default"
    print(f"\n--- {name} ---")

    # Try UCI direct download (XLS file)
    # Alternative: use a CSV mirror
    url = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
    dest_zip = DATA_RAW / "credit_default.zip"
    dest_csv = DATA_RAW / "credit_default.csv"

    if not dest_csv.exists():
        if not _download(url, dest_zip):
            print("  [SKIP] Could not download credit default data")
            return False
        import zipfile
        try:
            with zipfile.ZipFile(dest_zip) as zf:
                names = zf.namelist()
                # Find the XLS or CSV file
                data_file = next((n for n in names if n.endswith(('.xls', '.xlsx', '.csv'))), None)
                if data_file is None:
                    print(f"  [SKIP] No data file found in zip: {names}")
                    return False
                zf.extract(data_file, DATA_RAW)
                extracted = DATA_RAW / data_file
                if data_file.endswith(('.xls', '.xlsx')):
                    df_raw = pd.read_excel(extracted, header=1)
                    df_raw.to_csv(dest_csv, index=False)
                else:
                    (extracted).rename(dest_csv)
        except Exception as e:
            print(f"  [SKIP] Extract failed: {e}")
            return False

    try:
        df = pd.read_csv(dest_csv)
    except Exception:
        try:
            df = pd.read_excel(dest_csv, header=1)
        except Exception as e:
            print(f"  [SKIP] Read failed: {e}")
            return False

    # Target: default.payment.next.month
    target_col = next((c for c in df.columns if "default" in c.lower()), None)
    if target_col is None:
        print(f"  [SKIP] No default column found. Columns: {df.columns.tolist()[:10]}")
        return False

    df = df.dropna(subset=[target_col])
    df["y"] = df[target_col].astype(int)

    # Age groups
    if "AGE" in df.columns:
        df["age_group"] = pd.cut(df["AGE"], bins=[0, 30, 40, 50, 100],
                                  labels=["young", "mid", "senior", "elder"])
    elif "age" in df.columns:
        df["age_group"] = pd.cut(df["age"], bins=[0, 30, 40, 50, 100],
                                  labels=["young", "mid", "senior", "elder"])
    else:
        df["age_group"] = "all"

    # Education groups
    edu_col = next((c for c in df.columns if "EDUC" in c.upper()), None)
    if edu_col:
        df["edu_group"] = df[edu_col].clip(1, 4).astype(str)
    else:
        df["edu_group"] = "0"

    df["cohort"] = df["age_group"].astype(str) + "_edu" + df["edu_group"].astype(str)

    # Features: all numeric columns except target + ID
    exclude = {target_col, "ID", "y", "cohort", "age_group", "edu_group"}
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    y = df["y"].values
    cohorts = df["cohort"].values
    splits_df = make_splits(cohorts, seed=42)

    save_dataset(name, X, y, cohorts, splits_df)
    return True


def process_communities_crime() -> bool:
    """Communities & Crime: violent crime prediction, US region cohorts."""
    name = "communities_crime"
    print(f"\n--- {name} ---")

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data"
    names_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names"
    dest = DATA_RAW / "communities.data"
    if not _download(url, dest):
        print("  [SKIP] Could not download communities data")
        return False

    # Parse: 128 attributes, ? for missing values
    df = pd.read_csv(dest, header=None, na_values="?")

    # Columns: [community_name, state, county_code, community_code, fold, 5-127=features, 128=target]
    # State is column 1 (0-indexed)
    n_cols = df.shape[1]
    if n_cols < 10:
        print(f"  [SKIP] Unexpected shape: {df.shape}")
        return False

    # Target: last column (ViolentCrimesPerPop)
    target_col = n_cols - 1
    df["y"] = (df[target_col] >= df[target_col].median()).astype(int)

    # Cohorts: US state (column 1) -> grouped into 9 geographic regions
    state_col = 1
    state_to_region = {
        # New England
        9: "northeast", 23: "northeast", 25: "northeast",
        33: "northeast", 44: "northeast", 50: "northeast",
        # Mid-Atlantic
        34: "mid_atlantic", 36: "mid_atlantic", 42: "mid_atlantic",
        # South Atlantic
        10: "south_atlantic", 11: "south_atlantic", 12: "south_atlantic",
        13: "south_atlantic", 24: "south_atlantic", 37: "south_atlantic",
        45: "south_atlantic", 51: "south_atlantic", 54: "south_atlantic",
        # East North Central
        17: "east_north", 18: "east_north", 26: "east_north",
        39: "east_north", 55: "east_north",
        # West North Central
        19: "west_north", 20: "west_north", 27: "west_north",
        29: "west_north", 31: "west_north", 38: "west_north", 46: "west_north",
        # East South Central
        1: "east_south", 21: "east_south", 28: "east_south", 47: "east_south",
        # West South Central
        5: "west_south", 22: "west_south", 40: "west_south", 48: "west_south",
        # Mountain
        4: "mountain", 8: "mountain", 16: "mountain", 30: "mountain",
        32: "mountain", 35: "mountain", 49: "mountain", 56: "mountain",
        # Pacific
        2: "pacific", 6: "pacific", 15: "pacific", 41: "pacific", 53: "pacific",
    }
    df["cohort"] = df[state_col].map(state_to_region).fillna("other")

    # Features: columns 5 to target-1 (skip identifiers in 0-4)
    feature_cols = list(range(5, target_col))
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    y = df["y"].values
    cohorts = df["cohort"].values
    splits_df = make_splits(cohorts, seed=42)

    # Require at least 5 cohorts with >= 5 samples
    cohort_counts = pd.Series(cohorts).value_counts()
    if (cohort_counts >= 5).sum() < 4:
        print(f"  [SKIP] Too few valid cohorts: {cohort_counts.head()}")
        return False

    save_dataset(name, X, y, cohorts, splits_df)
    return True


def process_mushroom() -> bool:
    """UCI Mushroom: edible/poisonous, habitat cohorts."""
    name = "mushroom"
    print(f"\n--- {name} ---")

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    dest = DATA_RAW / "mushroom.data"
    if not _download(url, dest):
        print("  [SKIP] Could not download mushroom data")
        return False

    col_names = [
        "class", "cap_shape", "cap_surface", "cap_color", "bruises", "odor",
        "gill_attachment", "gill_spacing", "gill_size", "gill_color",
        "stalk_shape", "stalk_root", "stalk_surface_above", "stalk_surface_below",
        "stalk_color_above", "stalk_color_below", "veil_type", "veil_color",
        "ring_number", "ring_type", "spore_print_color", "population", "habitat"
    ]
    df = pd.read_csv(dest, header=None, names=col_names, na_values="?")
    df = df.dropna()

    # Binary: p=poisonous=1, e=edible=0
    df["y"] = (df["class"] == "p").astype(int)

    # Cohorts: habitat (7 types: d=woods, g=grasses, l=leaves, m=meadows, p=paths, u=urban, w=waste)
    habitat_names = {"d": "woods", "g": "grasses", "l": "leaves",
                     "m": "meadows", "p": "paths", "u": "urban", "w": "waste"}
    df["cohort"] = df["habitat"].map(habitat_names).fillna("other")

    # Encode all categorical features as ordinal
    feature_cols = [c for c in col_names if c not in ("class", "habitat")]
    for c in feature_cols:
        df[c] = df[c].astype("category").cat.codes.astype(np.float32)

    X = df[feature_cols].values.astype(np.float32)
    y = df["y"].values
    cohorts = df["cohort"].values
    splits_df = make_splits(cohorts, seed=42)

    save_dataset(name, X, y, cohorts, splits_df)
    return True


def process_law_school() -> bool:
    """Law School admissions: bar passage, race x gender cohorts."""
    name = "law_school"
    print(f"\n--- {name} ---")

    # LSAC dataset available on GitHub (fairness ML)
    url = "https://raw.githubusercontent.com/joshualoftus/lawschool_fairness/main/lawschool_race.csv"
    dest = DATA_RAW / "law_school.csv"
    if not _download(url, dest):
        # Try alternative
        url2 = "https://raw.githubusercontent.com/propublica/compas-analysis/master/law_school.csv"
        if not _download(url2, dest):
            print("  [SKIP] Could not download law school data")
            return False

    try:
        df = pd.read_csv(dest)
    except Exception as e:
        print(f"  [SKIP] Read error: {e}")
        return False

    # Find bar passage column
    bar_col = next((c for c in df.columns if "bar" in c.lower() or "pass" in c.lower()), None)
    if bar_col is None:
        print(f"  [SKIP] No bar passage col found. Cols: {df.columns.tolist()[:15]}")
        return False

    df = df.dropna(subset=[bar_col])
    df["y"] = (df[bar_col] >= 1).astype(int) if df[bar_col].max() > 1 else df[bar_col].astype(int)

    # Cohorts: race (if available) else gender
    race_col = next((c for c in df.columns if "race" in c.lower()), None)
    gender_col = next((c for c in df.columns if "sex" in c.lower() or "gender" in c.lower()), None)
    if race_col:
        df["cohort"] = df[race_col].astype(str)
    elif gender_col:
        df["cohort"] = df[gender_col].astype(str)
    else:
        print(f"  [SKIP] No cohort column found. Cols: {df.columns.tolist()[:15]}")
        return False

    # Features: LSAT, GPA, and other numeric predictors
    exclude = {bar_col, "y", "cohort", race_col, gender_col}
    feature_cols = [c for c in df.columns
                    if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if len(feature_cols) < 2:
        print(f"  [SKIP] Too few features: {feature_cols}")
        return False

    X = df[feature_cols].fillna(df[feature_cols].median()).values.astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    y = df["y"].values
    cohorts = df["cohort"].values
    splits_df = make_splits(cohorts, seed=42)

    save_dataset(name, X, y, cohorts, splits_df)
    return True


# ========================================================================== #
#  MOLECULAR DATASETS
# ========================================================================== #

def _process_molecular_smiles(name: str, url: str, smiles_col: str,
                                label_col: str, binarize: Optional[str] = None) -> bool:
    """Generic molecular dataset processor from DeepChem SMILES CSV."""
    print(f"\n--- {name} ---")

    dest = DATA_RAW / f"{name}.csv"
    if not _download(url, dest):
        print(f"  [SKIP] Could not download {name}")
        return False

    try:
        df = pd.read_csv(dest)
    except Exception as e:
        print(f"  [SKIP] Read error: {e}")
        return False

    if smiles_col not in df.columns:
        # Try to find SMILES column
        smiles_col = next((c for c in df.columns if "smiles" in c.lower()), None)
        if smiles_col is None:
            print(f"  [SKIP] No SMILES column found. Cols: {df.columns.tolist()[:10]}")
            return False

    if label_col not in df.columns:
        label_col = next((c for c in df.columns
                          if c.lower() not in ("smiles", smiles_col.lower(), "mol_id", "id")), None)
        if label_col is None:
            print(f"  [SKIP] No label column found")
            return False

    # Drop NaN labels
    df = df.dropna(subset=[label_col]).reset_index(drop=True)

    # Labels
    if binarize == "median":
        threshold = df[label_col].median()
        labels = (df[label_col] > threshold).astype(int).values
    else:
        labels = df[label_col].astype(int).values

    # RDKit features
    try:
        from ravel.featurizers.rdkit2d import featurize_rdkit2d
    except ImportError:
        print("  [SKIP] ravel not available for RDKit featurization")
        return False

    # Pre-filter invalid SMILES so batch featurizer doesn't crash
    from rdkit import Chem as _Chem
    df["_smi_valid"] = df[smiles_col].apply(
        lambda s: _Chem.MolFromSmiles(s) is not None if isinstance(s, str) else False
    )
    n_before = len(df)
    df = df[df["_smi_valid"]].drop(columns=["_smi_valid"]).reset_index(drop=True)
    labels = (df[label_col] > df[label_col].median()).astype(int).values if binarize == "median" \
             else df[label_col].astype(int).values
    print(f"  Valid SMILES: {len(df)} / {n_before}")

    print(f"  Featurizing {len(df)} molecules...")
    smiles_list = df[smiles_col].tolist()
    features = featurize_rdkit2d(smiles_list)  # (n, 217)

    # Filter out failed featurizations (NaN rows)
    valid_mask = ~np.any(np.isnan(features), axis=1)
    features = features[valid_mask]
    labels = labels[valid_mask]
    df_valid = df[valid_mask].reset_index(drop=True)
    print(f"  Valid molecules: {valid_mask.sum()} / {len(valid_mask)}")

    if len(features) < 100:
        print(f"  [SKIP] Too few valid molecules: {len(features)}")
        return False

    # Scaffold cohorts
    try:
        from ravel.chem.smiles import murcko_scaffold
        from ravel.data.splits import scaffoldwise_split
    except ImportError:
        print("  [SKIP] ravel chem not available")
        return False

    # Use existing rebinning logic (copy from train_new_datasets.py approach)
    valid_smiles = df_valid[smiles_col].tolist()
    scaffolds = []
    for smi in valid_smiles:
        try:
            sc = murcko_scaffold(smi)
            scaffolds.append(sc if sc else "no_scaffold")
        except Exception:
            scaffolds.append("no_scaffold")
    cohorts_raw = np.array(scaffolds)

    # Rebin to max 20 groups (for benchmark tractability)
    scaffold_counts = pd.Series(cohorts_raw).value_counts()
    top_scaffolds = scaffold_counts.head(19).index.tolist()
    cohorts = np.where(np.isin(cohorts_raw, top_scaffolds),
                       cohorts_raw, "other_scaffolds")

    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features).astype(np.float32)
    splits_df = make_splits(cohorts, seed=42)

    save_dataset(name, features, labels, cohorts, splits_df)
    return True


def process_hiv() -> bool:
    """HIV inhibition dataset from DeepChem."""
    return _process_molecular_smiles(
        name="hiv",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
        smiles_col="smiles",
        label_col="HIV_active",
        binarize=None,
    )


def process_qm7() -> bool:
    """QM7 quantum chemistry regression (binarized by median)."""
    return _process_molecular_smiles(
        name="qm7",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7.csv",
        smiles_col="smiles",
        label_col="u0_atom",
        binarize="median",
    )


def process_delaney() -> bool:
    """Delaney (ESOL) aqueous solubility -- independent reprocess with better cohorts."""
    # Note: 'esol' already exists but may have sparse features
    # This is an independent reprocess as 'delaney'
    return _process_molecular_smiles(
        name="delaney",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        smiles_col="smiles",
        label_col="measured log solubility in mols per litre",
        binarize="median",
    )


def process_sampl() -> bool:
    """SAMPL hydration free energy dataset."""
    return _process_molecular_smiles(
        name="sampl",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
        smiles_col="smiles",
        label_col="expt",
        binarize="median",
    )


# ========================================================================== #
#  MAIN
# ========================================================================== #

ALL_DATASETS = {
    # Text
    "ag_news": process_ag_news,
    "dbpedia": process_dbpedia,
    "trec": process_trec,
    "sst2": process_sst2,
    "imdb_genre": process_imdb_genre,
    # Tabular
    "wine_quality": process_wine_quality,
    "online_shoppers": process_online_shoppers,
    "credit_default": process_credit_default,
    "communities_crime": process_communities_crime,
    "mushroom": process_mushroom,
    "law_school": process_law_school,
    # Molecular
    "hiv": process_hiv,
    "qm7": process_qm7,
    "delaney": process_delaney,
    "sampl": process_sampl,
}

DOMAIN_MAP = {
    "text": ["ag_news", "dbpedia", "trec", "sst2", "imdb_genre"],
    "tabular": ["wine_quality", "online_shoppers", "credit_default",
                "communities_crime", "mushroom", "law_school"],
    "molecular": ["hiv", "qm7", "delaney", "sampl"],
}


def main():
    parser = argparse.ArgumentParser(description="Preprocess new ShiftBench datasets")
    parser.add_argument("--datasets", default="all",
                        help="Comma-separated dataset names, 'all', or domain name")
    parser.add_argument("--domain", default=None,
                        help="Process all datasets in a domain (text/tabular/molecular)")
    args = parser.parse_args()

    if args.domain:
        targets = DOMAIN_MAP.get(args.domain, [])
        if not targets:
            print(f"Unknown domain: {args.domain}. Valid: {list(DOMAIN_MAP.keys())}")
            sys.exit(1)
    elif args.datasets == "all":
        targets = list(ALL_DATASETS.keys())
    else:
        targets = [d.strip() for d in args.datasets.split(",")]

    print(f"Processing {len(targets)} datasets: {targets}")
    print(f"Output: {DATA_PROCESSED}\n")

    results = {}
    for ds_name in targets:
        if ds_name not in ALL_DATASETS:
            print(f"[UNKNOWN] {ds_name} - skipping")
            results[ds_name] = "unknown"
            continue
        try:
            ok = ALL_DATASETS[ds_name]()
            results[ds_name] = "OK" if ok else "SKIPPED"
        except Exception as e:
            print(f"  [ERROR] {ds_name}: {e}")
            results[ds_name] = f"ERROR: {e}"

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for ds, status in results.items():
        flag = "+" if status == "OK" else "-"
        print(f"  [{flag}] {ds}: {status}")
    n_ok = sum(1 for s in results.values() if s == "OK")
    print(f"\n{n_ok}/{len(results)} datasets successfully processed.")

    # List all processed datasets now
    all_processed = sorted([d.name for d in DATA_PROCESSED.iterdir() if d.is_dir()])
    print(f"\nTotal processed datasets: {len(all_processed)}")
    for d in all_processed:
        print(f"  {d}")


if __name__ == "__main__":
    main()
