"""Test script to verify all text datasets work with ShiftBench baselines.

This script loads each text dataset and performs basic sanity checks:
- Dataset loads successfully
- Data shapes are correct
- Splits are properly defined
- Features are numeric and finite
- Cohorts are properly assigned

Usage:
    python scripts/test_text_datasets.py
    python scripts/test_text_datasets.py --verbose
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from shiftbench.data import DatasetRegistry, load_dataset


def test_dataset(dataset_name: str, verbose: bool = False) -> dict:
    """Test loading and basic properties of a dataset.

    Args:
        dataset_name: Name of dataset to test
        verbose: Whether to print detailed information

    Returns:
        dict with test results
    """
    results = {
        "dataset": dataset_name,
        "status": "UNKNOWN",
        "errors": [],
    }

    try:
        # Load dataset
        if verbose:
            print(f"\n{'='*80}")
            print(f"Testing {dataset_name.upper()}")
            print(f"{'='*80}")

        X, y, cohorts, splits = load_dataset(dataset_name)

        # Get dataset info
        registry = DatasetRegistry()
        info = registry.get_dataset_info(dataset_name)

        if verbose:
            print(f"\nDataset info:")
            print(f"  Domain: {info['domain']}")
            print(f"  Task: {info['task_type']}")
            print(f"  Shift type: {info['shift_type']}")

        # Test 1: Shape consistency
        n_samples = len(X)
        if len(y) != n_samples or len(cohorts) != n_samples or len(splits) != n_samples:
            results["errors"].append(
                f"Shape mismatch: X={len(X)}, y={len(y)}, cohorts={len(cohorts)}, splits={len(splits)}"
            )

        if verbose:
            print(f"\nData shapes:")
            print(f"  X: {X.shape}")
            print(f"  y: {y.shape}")
            print(f"  cohorts: {cohorts.shape}")
            print(f"  splits: {len(splits)} rows")

        # Test 2: Feature values
        if not np.all(np.isfinite(X)):
            results["errors"].append("Non-finite values in features")

        if X.dtype not in [np.float32, np.float64, np.int32, np.int64]:
            results["errors"].append(f"Unexpected feature dtype: {X.dtype}")

        if verbose:
            print(f"\nFeature statistics:")
            print(f"  Mean: {X.mean():.4f}")
            print(f"  Std: {X.std():.4f}")
            print(f"  Min: {X.min():.4f}")
            print(f"  Max: {X.max():.4f}")
            print(f"  Non-zero: {np.count_nonzero(X)} / {X.size} ({np.count_nonzero(X)/X.size:.2%})")

        # Test 3: Labels
        unique_labels = np.unique(y[~np.isnan(y)])
        if info["task_type"] == "binary":
            if not np.array_equal(unique_labels, [0, 1]) and not np.array_equal(unique_labels, [0.0, 1.0]):
                results["errors"].append(f"Binary labels not 0/1: {unique_labels}")

        if verbose:
            print(f"\nLabel statistics:")
            print(f"  Unique values: {unique_labels}")
            print(f"  Positive rate: {y.mean():.2%}")
            print(f"  Missing values: {np.sum(np.isnan(y))}")

        # Test 4: Cohorts
        unique_cohorts = np.unique(cohorts)
        n_cohorts = len(unique_cohorts)

        if n_cohorts < 2:
            results["errors"].append(f"Too few cohorts: {n_cohorts}")

        if verbose:
            print(f"\nCohort statistics:")
            print(f"  Unique cohorts: {n_cohorts}")
            print(f"  Cohort sizes:")
            for cohort in unique_cohorts[:10]:  # Show first 10
                count = np.sum(cohorts == cohort)
                print(f"    {cohort}: {count} samples ({count/n_samples:.1%})")
            if n_cohorts > 10:
                print(f"    ... and {n_cohorts - 10} more cohorts")

        # Test 5: Splits
        required_splits = {"train", "cal", "test"}
        actual_splits = set(splits["split"].unique())

        if not required_splits.issubset(actual_splits):
            results["errors"].append(
                f"Missing splits: {required_splits - actual_splits}"
            )

        if verbose:
            print(f"\nSplit statistics:")
            for split in ["train", "cal", "test"]:
                if split in actual_splits:
                    count = (splits["split"] == split).sum()
                    print(f"  {split}: {count} samples ({count/n_samples:.1%})")

        # Test 6: Registry metadata
        expected_samples = info.get("n_samples")
        expected_features = info.get("n_features")

        if expected_samples and abs(n_samples - expected_samples) > 10:
            results["errors"].append(
                f"Sample count mismatch: got {n_samples}, expected {expected_samples}"
            )

        if expected_features and abs(X.shape[1] - expected_features) > 1:
            results["errors"].append(
                f"Feature count mismatch: got {X.shape[1]}, expected {expected_features}"
            )

        # Final status
        if len(results["errors"]) == 0:
            results["status"] = "PASS"
            if verbose:
                print(f"\n[SUCCESS] All tests passed for {dataset_name}")
        else:
            results["status"] = "FAIL"
            if verbose:
                print(f"\n[FAIL] {len(results['errors'])} error(s) found:")
                for error in results["errors"]:
                    print(f"  - {error}")

    except Exception as e:
        results["status"] = "ERROR"
        results["errors"].append(str(e))
        if verbose:
            print(f"\n[ERROR] Failed to test {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test all text datasets in ShiftBench"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Test specific dataset (default: all text datasets)",
    )

    args = parser.parse_args()

    # Get text datasets
    registry = DatasetRegistry()

    if args.dataset:
        dataset_names = [args.dataset]
    else:
        dataset_names = registry.list_datasets(domain="text")

    print("\n" + "="*80)
    print("ShiftBench Text Dataset Testing")
    print("="*80)
    print(f"\nTesting {len(dataset_names)} text dataset(s)...")

    # Run tests
    all_results = []
    for dataset_name in dataset_names:
        result = test_dataset(dataset_name, verbose=args.verbose)
        all_results.append(result)

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)

    n_pass = sum(1 for r in all_results if r["status"] == "PASS")
    n_fail = sum(1 for r in all_results if r["status"] == "FAIL")
    n_error = sum(1 for r in all_results if r["status"] == "ERROR")

    for result in all_results:
        status_icon = {
            "PASS": "[PASS]",
            "FAIL": "[FAIL]",
            "ERROR": "[ERROR]",
        }.get(result["status"], "[?]")

        print(f"{status_icon} {result['dataset']:20s} {result['status']}")

        if result["errors"] and not args.verbose:
            for error in result["errors"]:
                print(f"    - {error}")

    print(f"\nResults:")
    print(f"  PASS: {n_pass}/{len(all_results)}")
    print(f"  FAIL: {n_fail}/{len(all_results)}")
    print(f"  ERROR: {n_error}/{len(all_results)}")

    if n_fail + n_error > 0:
        print("\n[FAIL] Some tests failed. Run with --verbose for details.")
        return 1
    else:
        print("\n[SUCCESS] All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
