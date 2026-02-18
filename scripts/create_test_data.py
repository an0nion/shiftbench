"""Create synthetic test dataset to verify infrastructure.

This script generates a small synthetic dataset that follows the ShiftBench
format. Use this to test that the data loading and baseline evaluation work
before processing real datasets.

Usage:
    python scripts/create_test_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


def create_test_dataset(
    n_samples: int = 1000,
    n_features: int = 10,
    n_cohorts: int = 5,
    output_dir: Path = None,
    random_state: int = 42,
):
    """Generate synthetic test dataset.

    Creates a binary classification task where:
    - Positive class if first feature > 0
    - Covariate shift via shifting mean of features
    - Cohorts based on feature clustering

    Args:
        n_samples: Total samples to generate
        n_features: Number of features
        n_cohorts: Number of cohorts
        output_dir: Where to save (default: data/processed/test_dataset/)
        random_state: Random seed
    """
    rng = np.random.RandomState(random_state)

    # Generate features with shift
    # Train: mean=0, Cal: mean=0.2, Test: mean=0.5 (increasing shift)
    n_train = int(0.6 * n_samples)
    n_cal = int(0.2 * n_samples)
    n_test = n_samples - n_train - n_cal

    X_train = rng.randn(n_train, n_features)
    X_cal = rng.randn(n_cal, n_features) + 0.2  # Slight shift
    X_test = rng.randn(n_test, n_features) + 0.5  # Moderate shift

    X = np.vstack([X_train, X_cal, X_test])

    # Labels: positive if first feature > 0
    y = (X[:, 0] > 0).astype(int)

    # Cohorts: based on second feature (quintiles)
    cohort_boundaries = np.percentile(X[:, 1], np.linspace(0, 100, n_cohorts + 1))
    cohort_ids = np.digitize(X[:, 1], cohort_boundaries[1:-1])
    cohorts = np.array([f"cohort_{i}" for i in cohort_ids])

    # Splits
    splits_labels = (
        ["train"] * n_train + ["cal"] * n_cal + ["test"] * n_test
    )

    # Create output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "processed" / "test_dataset"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save arrays
    np.save(output_dir / "features.npy", X)
    np.save(output_dir / "labels.npy", y)
    np.save(output_dir / "cohorts.npy", cohorts)

    # Save splits DataFrame
    splits_df = pd.DataFrame({
        "uid": [f"sample_{i}" for i in range(n_samples)],
        "split": splits_labels,
    })
    splits_df.to_csv(output_dir / "splits.csv", index=False)

    # Print summary
    print(f"[SUCCESS] Created test dataset at {output_dir}")
    print(f"   Samples: {n_samples} (train={n_train}, cal={n_cal}, test={n_test})")
    print(f"   Features: {n_features}")
    print(f"   Cohorts: {n_cohorts}")
    print(f"   Positive rate: {y.mean():.2%}")
    print(f"   Cohort distribution:")
    for cohort in np.unique(cohorts):
        count = (cohorts == cohort).sum()
        print(f"      {cohort}: {count} samples")

    # Verify files exist
    assert (output_dir / "features.npy").exists()
    assert (output_dir / "labels.npy").exists()
    assert (output_dir / "cohorts.npy").exists()
    assert (output_dir / "splits.csv").exists()

    print("\n[SUCCESS] All files saved successfully!")
    return output_dir


if __name__ == "__main__":
    create_test_dataset()
