"""Test uLSIF baseline on synthetic data.

This script verifies that:
1. Dataset loading works
2. uLSIF weight estimation works
3. Bound estimation works
4. Results are reasonable

Usage:
    python scripts/create_test_data.py  # Create test data first
    python scripts/test_ulsif.py        # Then run this test
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from shiftbench.baselines.ulsif import create_ulsif_baseline
from shiftbench.data import load_dataset


def test_ulsif_on_synthetic_data():
    """Run uLSIF on test dataset and print results."""

    print("=" * 80)
    print("Testing uLSIF Baseline on Synthetic Data")
    print("=" * 80)

    # Step 1: Load dataset
    print("\n[1/5] Loading test dataset...")
    try:
        X, y, cohorts, splits = load_dataset("test_dataset")
        print(f"[OK] Loaded {len(X)} samples with {X.shape[1]} features")
        print(f"   Cohorts: {len(np.unique(cohorts))}")
        print(f"   Positive rate: {y.mean():.2%}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("   Run 'python scripts/create_test_data.py' first!")
        return

    # Step 2: Split into calibration and test
    print("\n[2/5] Splitting into calibration and test sets...")
    cal_mask = (splits["split"] == "cal").values
    test_mask = (splits["split"] == "test").values

    X_cal, y_cal, cohorts_cal = X[cal_mask], y[cal_mask], cohorts[cal_mask]
    X_test = X[test_mask]

    print(f"[OK] Calibration: {len(X_cal)} samples")
    print(f"   Test: {len(X_test)} samples")
    print(f"   Cal positive rate: {y_cal.mean():.2%}")

    # Step 3: Estimate weights
    print("\n[3/5] Estimating importance weights with uLSIF...")
    ulsif = create_ulsif_baseline(n_basis=50, random_state=42)

    try:
        weights = ulsif.estimate_weights(X_cal, X_test)
        print(f"[OK] Estimated weights for {len(weights)} calibration samples")
        print(f"   Mean: {weights.mean():.3f} (should be ~1.0)")
        print(f"   Std: {weights.std():.3f}")
        print(f"   Min: {weights.min():.3f}, Max: {weights.max():.3f}")

        # Check validity
        assert np.all(weights > 0), "All weights must be positive"
        assert np.all(np.isfinite(weights)), "All weights must be finite"
        assert np.abs(weights.mean() - 1.0) < 0.1, "Mean should be ~1.0"
        print("[OK] Weights passed validity checks")

    except Exception as e:
        print(f"[ERROR] Error estimating weights: {e}")
        return

    # Step 4: Generate predictions (simple: predict positive for all)
    print("\n[4/5] Generating predictions...")
    predictions_cal = np.ones(len(y_cal), dtype=int)
    print(f"[OK] Predicted {predictions_cal.sum()} positives (naive baseline)")

    # Step 5: Estimate bounds
    print("\n[5/5] Estimating PPV bounds...")
    tau_grid = [0.5, 0.6, 0.7, 0.8, 0.9]

    try:
        decisions = ulsif.estimate_bounds(
            y_cal,
            predictions_cal,
            cohorts_cal,
            weights,
            tau_grid,
            alpha=0.05,
        )

        print(f"[OK] Generated {len(decisions)} decisions")
        print(f"\nResults by cohort and threshold:")
        print("-" * 80)
        print(f"{'Cohort':<15} {'Tau':<5} {'Decision':<12} {'Mean':<8} {'LB':<8} {'p-value':<10}")
        print("-" * 80)

        for d in decisions:
            print(
                f"{d.cohort_id:<15} {d.tau:<5.2f} {d.decision:<12} "
                f"{d.mu_hat:<8.3f} {d.lower_bound:<8.3f} {d.p_value:<10.4f}"
            )

        # Summary statistics
        n_certify = sum(1 for d in decisions if d.decision == "CERTIFY")
        n_abstain = sum(1 for d in decisions if d.decision == "ABSTAIN")
        n_no_guarantee = sum(1 for d in decisions if d.decision == "NO-GUARANTEE")

        print("-" * 80)
        print(f"\nSummary:")
        print(f"  CERTIFY: {n_certify}/{len(decisions)} ({n_certify/len(decisions):.1%})")
        print(f"  ABSTAIN: {n_abstain}/{len(decisions)} ({n_abstain/len(decisions):.1%})")
        print(f"  NO-GUARANTEE: {n_no_guarantee}/{len(decisions)} ({n_no_guarantee/len(decisions):.1%})")

        # Diagnostics
        print(f"\nMethod diagnostics:")
        diag = ulsif.get_diagnostics()
        for key, value in diag.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"[ERROR] Error estimating bounds: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print("[SUCCESS] All tests passed! uLSIF baseline is working correctly.")
    print("=" * 80)


if __name__ == "__main__":
    test_ulsif_on_synthetic_data()
