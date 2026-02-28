"""Preprocess tabular datasets for shift-bench format.

This script converts raw tabular datasets with demographic/covariate shifts
into standardized shift-bench format:
- features.npy: Standardized numeric + one-hot encoded categorical features
- labels.npy: Binary classification labels
- cohorts.npy: Cohort assignments based on protected attributes/temporal/geographic groups
- splits.csv: Train/cal/test splits
- metadata.json: Dataset metadata

Datasets included:
1. Adult/Census Income - Income prediction, demographic shift (race, gender, age)
2. COMPAS Recidivism - Recidivism prediction, demographic shift (race)
3. Bank Marketing - Marketing response, temporal shift
4. Credit Default - Default prediction, demographic shift
5. German Credit - Credit risk, demographic shift

Usage:
    python scripts/preprocess_tabular.py --dataset adult
    python scripts/preprocess_tabular.py --all  # Process all tabular datasets
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


DATASET_CONFIG = {
    "adult": {
        "description": "Adult Census Income dataset - predict income >50K",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "test_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
        "task_type": "binary",
        "label_col": "income",
        "cohort_definition": "demographic",  # race, gender, age groups
        "shift_type": "demographic_shift",
        "protected_attributes": ["race", "sex", "age"],
    },
    "compas": {
        "description": "COMPAS Recidivism - predict recidivism within 2 years",
        "url": "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
        "task_type": "binary",
        "label_col": "two_year_recid",
        "cohort_definition": "demographic",  # race, sex, age
        "shift_type": "demographic_shift",
        "protected_attributes": ["race", "sex", "age"],
    },
    "bank": {
        "description": "Bank Marketing - predict term deposit subscription",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip",
        "task_type": "binary",
        "label_col": "y",
        "cohort_definition": "temporal",  # month-based cohorts
        "shift_type": "temporal_shift",
        "protected_attributes": ["month"],
    },
    "credit_default": {
        "description": "Default of Credit Card Clients - predict default payment",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
        "task_type": "binary",
        "label_col": "default.payment.next.month",
        "cohort_definition": "demographic",  # education, marriage, age
        "shift_type": "demographic_shift",
        "protected_attributes": ["SEX", "EDUCATION", "MARRIAGE", "AGE"],
    },
    "german_credit": {
        "description": "German Credit - predict credit risk",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
        "task_type": "binary",
        "label_col": "class",
        "cohort_definition": "demographic",  # age, sex (inferred)
        "shift_type": "demographic_shift",
        "protected_attributes": ["age", "personal_status"],
    },
    "diabetes": {
        "description": "Pima Indians Diabetes - predict diabetes onset",
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        "task_type": "binary",
        "label_col": "outcome",
        "cohort_definition": "demographic",  # age groups
        "shift_type": "demographic_shift",
        "protected_attributes": ["age"],
    },
    "heart_disease": {
        "description": "Heart Disease - predict presence of heart disease",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "task_type": "binary",
        "label_col": "num",
        "cohort_definition": "demographic",  # age, sex
        "shift_type": "demographic_shift",
        "protected_attributes": ["age", "sex"],
    },
    "student_performance": {
        "description": "Student Performance - predict student grade with demographic/school shifts",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip",
        "task_type": "binary",
        "label_col": "pass",
        "cohort_definition": "demographic",  # school, sex, age
        "shift_type": "demographic_shift",
        "protected_attributes": ["school", "sex", "age"],
    },
}


def download_dataset(dataset_name: str, cache_dir: Path) -> Path:
    """Download dataset if not already cached.

    Args:
        dataset_name: Name of dataset
        cache_dir: Directory to cache raw data

    Returns:
        Path to downloaded file
    """
    config = DATASET_CONFIG[dataset_name]
    cache_dir = cache_dir / dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    if dataset_name == "adult":
        # Adult has both train and test files
        train_file = cache_dir / "adult.data"
        test_file = cache_dir / "adult.test"

        if not train_file.exists():
            print(f"   Downloading training data from {config['url']}...")
            urlretrieve(config['url'], train_file)

        if not test_file.exists():
            print(f"   Downloading test data from {config['test_url']}...")
            urlretrieve(config['test_url'], test_file)

        return train_file

    elif dataset_name == "bank":
        # Bank is a zip file
        zip_file = cache_dir / "bank-additional.zip"
        if not zip_file.exists():
            print(f"   Downloading from {config['url']}...")
            urlretrieve(config['url'], zip_file)

        # Extract if needed
        csv_file = cache_dir / "bank-additional" / "bank-additional-full.csv"
        if not csv_file.exists():
            import zipfile
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(cache_dir)

        return csv_file

    elif dataset_name == "credit_default":
        # Credit default is an Excel file
        xls_file = cache_dir / "default_credit.xls"
        if not xls_file.exists():
            print(f"   Downloading from {config['url']}...")
            urlretrieve(config['url'], xls_file)
        return xls_file

    elif dataset_name == "student_performance":
        # Student performance is a zip file with multiple CSV files
        zip_file = cache_dir / "student.zip"
        if not zip_file.exists():
            print(f"   Downloading from {config['url']}...")
            urlretrieve(config['url'], zip_file)

        # Extract if needed
        mat_file = cache_dir / "student-mat.csv"
        if not mat_file.exists():
            import zipfile
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(cache_dir)

        return mat_file

    else:
        # Simple CSV download
        file_name = cache_dir / f"{dataset_name}.csv"
        if not file_name.exists():
            print(f"   Downloading from {config['url']}...")
            urlretrieve(config['url'], file_name)
        return file_name


def load_adult_dataset(cache_dir: Path) -> pd.DataFrame:
    """Load and combine Adult dataset train/test files."""
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    train_file = cache_dir / "adult" / "adult.data"
    test_file = cache_dir / "adult" / "adult.test"

    # Load train and test
    df_train = pd.read_csv(train_file, names=columns, skipinitialspace=True, na_values="?")
    df_test = pd.read_csv(test_file, names=columns, skipinitialspace=True, skiprows=1, na_values="?")

    # Combine
    df = pd.concat([df_train, df_test], ignore_index=True)

    # Clean income labels (remove periods from test set)
    df["income"] = df["income"].str.replace(".", "", regex=False).str.strip()

    # Binary label: >50K = 1, <=50K = 0
    df["income"] = (df["income"] == ">50K").astype(int)

    return df


def load_compas_dataset(cache_dir: Path) -> pd.DataFrame:
    """Load COMPAS dataset."""
    file_path = cache_dir / "compas" / "compas.csv"
    df = pd.read_csv(file_path)

    # Filter to relevant rows (remove missing data)
    df = df[df["days_b_screening_arrest"] <= 30]
    df = df[df["days_b_screening_arrest"] >= -30]
    df = df[df["is_recid"] != -1]
    df = df[df["c_charge_degree"] != "O"]
    df = df[df["score_text"] != "N/A"]

    # Use two_year_recid as label
    df = df[df["two_year_recid"].notna()]

    return df


def load_bank_dataset(cache_dir: Path) -> pd.DataFrame:
    """Load Bank Marketing dataset."""
    file_path = cache_dir / "bank" / "bank-additional" / "bank-additional-full.csv"
    df = pd.read_csv(file_path, sep=";")

    # Binary label: yes = 1, no = 0
    df["y"] = (df["y"] == "yes").astype(int)

    return df


def load_credit_default_dataset(cache_dir: Path) -> pd.DataFrame:
    """Load Credit Default dataset.

    Note: This dataset requires openpyxl to read the Excel file.
    Install with: pip install openpyxl
    """
    file_path = cache_dir / "credit_default" / "default_credit.xls"

    # Try to read Excel file with openpyxl engine
    try:
        import openpyxl
        df = pd.read_excel(file_path, engine='openpyxl', header=1)  # Skip first row
    except ImportError:
        print(f"   [ERROR] openpyxl not installed. Install with: pip install openpyxl")
        raise
    except Exception as e:
        # Try with different engine
        try:
            df = pd.read_excel(file_path, engine='xlrd', header=1)
        except Exception as e2:
            print(f"   [ERROR] Failed to read Excel file: {e2}")
            raise

    # Rename default column if needed
    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": "default.payment.next.month"})
    elif "default.payment.next.month" not in df.columns:
        # Find the last column as the label
        df = df.rename(columns={df.columns[-1]: "default.payment.next.month"})

    return df


def load_german_credit_dataset(cache_dir: Path) -> pd.DataFrame:
    """Load German Credit dataset."""
    file_path = cache_dir / "german_credit" / "german_credit.csv"

    # German credit has no header, space-separated
    # https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc
    columns = [
        "status", "duration", "credit_history", "purpose", "credit_amount",
        "savings", "employment", "installment_rate", "personal_status",
        "other_debtors", "residence_since", "property", "age",
        "other_installment_plans", "housing", "num_credits", "job",
        "num_dependents", "telephone", "foreign_worker", "class"
    ]

    df = pd.read_csv(file_path, sep=" ", names=columns)

    # Binary label: 1 = good credit, 2 = bad credit
    # Convert to: 1 = good (0), 2 = bad (1) for consistency
    df["class"] = (df["class"] == 2).astype(int)

    return df


def load_diabetes_dataset(cache_dir: Path) -> pd.DataFrame:
    """Load Pima Indians Diabetes dataset."""
    file_path = cache_dir / "diabetes" / "diabetes.csv"

    columns = [
        "pregnancies", "glucose", "blood_pressure", "skin_thickness",
        "insulin", "bmi", "diabetes_pedigree", "age", "outcome"
    ]

    df = pd.read_csv(file_path, names=columns)

    # outcome is already 0/1
    return df


def load_heart_disease_dataset(cache_dir: Path) -> pd.DataFrame:
    """Load Heart Disease dataset."""
    file_path = cache_dir / "heart_disease" / "heart_disease.csv"

    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
    ]

    df = pd.read_csv(file_path, names=columns, na_values="?")

    # Binary label: 0 = no disease, >0 = disease
    df["num"] = (df["num"] > 0).astype(int)

    return df


def load_student_performance_dataset(cache_dir: Path) -> pd.DataFrame:
    """Load Student Performance dataset.

    Combines student-mat.csv (Math) and student-por.csv (Portuguese) datasets.
    """
    mat_file = cache_dir / "student_performance" / "student-mat.csv"
    por_file = cache_dir / "student_performance" / "student-por.csv"

    # Load both files
    df_mat = pd.read_csv(mat_file, sep=';')
    df_mat['course'] = 'math'

    df_por = pd.read_csv(por_file, sep=';')
    df_por['course'] = 'portuguese'

    # Combine datasets
    df = pd.concat([df_mat, df_por], ignore_index=True)

    # Create binary pass/fail label based on final grade (G3)
    # Pass: G3 >= 10 (out of 20), Fail: G3 < 10
    df['pass'] = (df['G3'] >= 10).astype(int)

    return df


def create_cohorts(df: pd.DataFrame, dataset_name: str) -> np.ndarray:
    """Create cohort assignments based on protected attributes.

    Args:
        df: DataFrame with all features
        dataset_name: Name of dataset

    Returns:
        Array of cohort identifiers (strings)
    """
    config = DATASET_CONFIG[dataset_name]
    cohort_def = config["cohort_definition"]

    if dataset_name == "adult":
        # Combine race, sex, and age groups
        age_bins = [0, 25, 35, 45, 55, 100]
        age_labels = ["<25", "25-35", "35-45", "45-55", "55+"]
        df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels)

        # Create cohort string
        cohorts = df.apply(
            lambda row: f"{row['race']}_{row['sex']}_{row['age_group']}",
            axis=1
        ).values

    elif dataset_name == "compas":
        # Combine race, sex, age
        age_bins = [0, 25, 35, 45, 100]
        age_labels = ["<25", "25-35", "35-45", "45+"]
        df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels)

        cohorts = df.apply(
            lambda row: f"{row['race']}_{row['sex']}_{row['age_group']}",
            axis=1
        ).values

    elif dataset_name == "bank":
        # Temporal cohorts based on month
        cohorts = df["month"].values

    elif dataset_name == "credit_default":
        # Demographic: sex, education, marriage status
        cohorts = df.apply(
            lambda row: f"SEX{row['SEX']}_EDU{row['EDUCATION']}_MAR{row['MARRIAGE']}",
            axis=1
        ).values

    elif dataset_name == "german_credit":
        # Age groups and personal status
        age_bins = [0, 25, 35, 45, 100]
        age_labels = ["<25", "25-35", "35-45", "45+"]
        df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels)

        cohorts = df.apply(
            lambda row: f"{row['personal_status']}_{row['age_group']}",
            axis=1
        ).values

    elif dataset_name == "diabetes":
        # Age groups
        age_bins = [0, 30, 40, 50, 100]
        age_labels = ["<30", "30-40", "40-50", "50+"]
        df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels)

        cohorts = df["age_group"].values.astype(str)

    elif dataset_name == "heart_disease":
        # Age and sex cohorts
        age_bins = [0, 45, 55, 65, 100]
        age_labels = ["<45", "45-55", "55-65", "65+"]
        df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels)

        cohorts = df.apply(
            lambda row: f"SEX{row['sex']}_{row['age_group']}",
            axis=1
        ).values

    elif dataset_name == "student_performance":
        # School, sex, age cohorts
        age_bins = [0, 16, 18, 20, 100]
        age_labels = ["<16", "16-18", "18-20", "20+"]
        df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels)

        cohorts = df.apply(
            lambda row: f"{row['school']}_{row['sex']}_{row['age_group']}",
            axis=1
        ).values

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return cohorts


def preprocess_features(df: pd.DataFrame, label_col: str) -> np.ndarray:
    """Preprocess features: one-hot encode categoricals, standardize numerics.

    Args:
        df: DataFrame with features and labels
        label_col: Name of label column

    Returns:
        Feature matrix (n_samples, n_features)
    """
    # Separate features from label
    feature_cols = [c for c in df.columns if c != label_col]
    df_features = df[feature_cols].copy()

    # Identify numeric and categorical columns
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_features.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"   Numeric features: {len(numeric_cols)}")
    print(f"   Categorical features: {len(categorical_cols)}")

    # Process numeric features
    numeric_data = df_features[numeric_cols].values

    # Fill missing values with median
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    numeric_data = imputer.fit_transform(numeric_data)

    # Standardize
    scaler = StandardScaler()
    numeric_data = scaler.fit_transform(numeric_data)

    # Process categorical features (one-hot encode)
    categorical_data = None
    if categorical_cols:
        # Fill missing values with "missing"
        # Convert to object first so pandas Categorical columns accept new values
        df_cat = df_features[categorical_cols].astype(object).fillna("missing")

        # One-hot encode
        df_encoded = pd.get_dummies(df_cat, drop_first=False)
        categorical_data = df_encoded.values

        print(f"   One-hot encoded features: {categorical_data.shape[1]}")

    # Combine numeric and categorical
    if categorical_data is not None:
        X = np.hstack([numeric_data, categorical_data])
    else:
        X = numeric_data

    print(f"   Total features: {X.shape[1]}")

    return X


def create_splits(n_samples: int, seed: int = 42) -> pd.DataFrame:
    """Create train/cal/test splits.

    Args:
        n_samples: Number of samples
        seed: Random seed

    Returns:
        DataFrame with columns [uid, split]
    """
    rng = np.random.RandomState(seed)

    # 60% train, 20% cal, 20% test
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_train = int(0.60 * n_samples)
    n_cal = int(0.20 * n_samples)

    splits = np.array(["train"] * n_samples, dtype=object)
    splits[indices[n_train:n_train + n_cal]] = "cal"
    splits[indices[n_train + n_cal:]] = "test"

    # Create DataFrame
    uids = [f"sample_{i:09d}" for i in range(n_samples)]
    splits_df = pd.DataFrame({"uid": uids, "split": splits})

    return splits_df


def preprocess_tabular_dataset(
    dataset_name: str,
    cache_dir: Path,
    output_dir: Path,
    seed: int = 42,
):
    """Preprocess a single tabular dataset.

    Args:
        dataset_name: Name of dataset
        cache_dir: Directory for cached raw data
        output_dir: Output directory for processed data
        seed: Random seed for splits
    """
    print(f"\n{'='*80}")
    print(f"Processing {dataset_name.upper()}")
    print(f"{'='*80}")

    config = DATASET_CONFIG[dataset_name]

    # Step 1: Download dataset
    print(f"\n[1/6] Downloading dataset...")
    try:
        file_path = download_dataset(dataset_name, cache_dir)
        print(f"   Cached at: {file_path}")
    except Exception as e:
        print(f"   [ERROR] Download failed: {e}")
        raise

    # Step 2: Load dataset
    print(f"\n[2/6] Loading dataset...")
    try:
        if dataset_name == "adult":
            df = load_adult_dataset(cache_dir)
        elif dataset_name == "compas":
            df = load_compas_dataset(cache_dir)
        elif dataset_name == "bank":
            df = load_bank_dataset(cache_dir)
        elif dataset_name == "credit_default":
            df = load_credit_default_dataset(cache_dir)
        elif dataset_name == "german_credit":
            df = load_german_credit_dataset(cache_dir)
        elif dataset_name == "diabetes":
            df = load_diabetes_dataset(cache_dir)
        elif dataset_name == "heart_disease":
            df = load_heart_disease_dataset(cache_dir)
        elif dataset_name == "student_performance":
            df = load_student_performance_dataset(cache_dir)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        print(f"   Loaded {len(df)} samples")
        print(f"   Columns: {len(df.columns)}")
    except Exception as e:
        print(f"   [ERROR] Loading failed: {e}")
        raise

    # Step 3: Clean missing values
    print(f"\n[3/6] Cleaning data...")
    initial_size = len(df)

    # Drop rows with missing labels
    label_col = config["label_col"]
    df = df.dropna(subset=[label_col])

    print(f"   Removed {initial_size - len(df)} rows with missing labels")
    print(f"   Remaining: {len(df)} samples")

    # Step 4: Create cohorts
    print(f"\n[4/6] Creating cohorts ({config['cohort_definition']})...")
    try:
        cohorts = create_cohorts(df, dataset_name)
        unique_cohorts = np.unique(cohorts)
        print(f"   Found {len(unique_cohorts)} unique cohorts")

        # Show cohort distribution
        cohort_counts = pd.Series(cohorts).value_counts()
        print(f"   Largest cohort: {cohort_counts.iloc[0]} samples")
        print(f"   Smallest cohort: {cohort_counts.iloc[-1]} samples")
    except Exception as e:
        print(f"   [ERROR] Cohort creation failed: {e}")
        raise

    # Step 5: Preprocess features
    print(f"\n[5/6] Preprocessing features...")
    try:
        X = preprocess_features(df, label_col)
        y = df[label_col].values.astype(int)

        print(f"   Feature matrix: {X.shape}")
        print(f"   Labels: {y.shape}, positive rate: {y.mean():.2%}")
    except Exception as e:
        print(f"   [ERROR] Feature preprocessing failed: {e}")
        raise

    # Step 6: Create splits and save
    print(f"\n[6/6] Creating splits and saving...")
    output_dir = output_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create splits
    splits_df = create_splits(len(X), seed=seed)

    # Count splits
    split_counts = splits_df["split"].value_counts()
    print(f"   Train: {split_counts.get('train', 0)} samples")
    print(f"   Cal: {split_counts.get('cal', 0)} samples")
    print(f"   Test: {split_counts.get('test', 0)} samples")

    # Save files
    np.save(output_dir / "features.npy", X)
    print(f"   Saved features.npy ({X.shape})")

    np.save(output_dir / "labels.npy", y)
    print(f"   Saved labels.npy ({y.shape})")

    np.save(output_dir / "cohorts.npy", cohorts)
    print(f"   Saved cohorts.npy ({cohorts.shape})")

    splits_df.to_csv(output_dir / "splits.csv", index=False)
    print(f"   Saved splits.csv ({len(splits_df)} rows)")

    # Save metadata
    metadata = {
        "dataset": dataset_name,
        "task_type": config["task_type"],
        "description": config["description"],
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_cohorts": len(unique_cohorts),
        "cohort_definition": config["cohort_definition"],
        "shift_type": config["shift_type"],
        "split_counts": {
            "train": int(split_counts.get("train", 0)),
            "cal": int(split_counts.get("cal", 0)),
            "test": int(split_counts.get("test", 0)),
        },
        "positive_rate": float(y.mean()),
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
        description="Preprocess tabular datasets for ShiftBench"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIG.keys()) + ["all"],
        help="Dataset to preprocess (or 'all' for all datasets)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "raw_tabular",
        help="Directory to cache raw data (default: data/raw_tabular)",
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

    # Create cache directory
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Process datasets
    datasets_to_process = (
        list(DATASET_CONFIG.keys()) if args.dataset == "all" else [args.dataset]
    )

    results = []
    for dataset_name in datasets_to_process:
        try:
            metadata = preprocess_tabular_dataset(
                dataset_name,
                args.cache_dir,
                args.output_dir,
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
