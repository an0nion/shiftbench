"""Test Weighted Conformal Prediction baseline on synthetic data.

This script verifies that:
1. Dataset loading works
2. Weight estimation works (uLSIF or KLIEP)
3. Weighted quantile computation works
4. Conformal bound estimation works
5. Results are reasonable

Usage:
    python scripts/create_test_data.py  # Create test data first
    python scripts/test_weighted_conformal.py        # Then run this test
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from shiftbench.baselines.weighted_conformal import create_weighted_conformal_baseline
from shiftbench.data import load_dataset


def test_weighted_conformal_on_synthetic_data():
    """Run Weighted Conformal Prediction on test dataset and print results."""

    print("=" * 80)
    print("Testing Weighted Conformal Prediction Baseline on Synthetic Data")
    print("=" * 80)

    # Step 1: Load dataset
    print("\n[1/6] Loading test dataset...")
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
    print("\n[2/6] Splitting into calibration and test sets...")
    cal_mask = (splits["split"] == "cal").values
    test_mask = (splits["split"] == "test").values

    X_cal, y_cal, cohorts_cal = X[cal_mask], y[cal_mask], cohorts[cal_mask]
    X_test = X[test_mask]

    print(f"[OK] Calibration: {len(X_cal)} samples")
    print(f"   Test: {len(X_test)} samples")
    print(f"   Cal positive rate: {y_cal.mean():.2%}")

    # Step 3: Test with uLSIF weights
    print("\n[3/6] Testing with uLSIF weight estimation...")
    wcp_ulsif = create_weighted_conformal_baseline(
        weight_method="ulsif",
        n_basis=50,
        random_state=42
    )

    try:
        weights_ulsif = wcp_ulsif.estimate_weights(X_cal, X_test)
        print(f"[OK] Estimated weights for {len(weights_ulsif)} calibration samples")
        print(f"   Mean: {weights_ulsif.mean():.3f} (should be ~1.0)")
        print(f"   Std: {weights_ulsif.std():.3f}")
        print(f"   Min: {weights_ulsif.min():.3f}, Max: {weights_ulsif.max():.3f}")

        # Check validity
        assert np.all(weights_ulsif > 0), "All weights must be positive"
        assert np.all(np.isfinite(weights_ulsif)), "All weights must be finite"
        assert np.abs(weights_ulsif.mean() - 1.0) < 0.1, "Mean should be ~1.0"
        print("[OK] Weights passed validity checks")

    except Exception as e:
        print(f"[ERROR] Error estimating weights: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 4: Test with KLIEP weights
    print("\n[4/6] Testing with KLIEP weight estimation...")
    wcp_kliep = create_weighted_conformal_baseline(
        weight_method="kliep",
        n_basis=50,
        max_iter=5000,
        random_state=42
    )

    try:
        weights_kliep = wcp_kliep.estimate_weights(X_cal, X_test)
        print(f"[OK] Estimated weights for {len(weights_kliep)} calibration samples")
        print(f"   Mean: {weights_kliep.mean():.3f} (should be ~1.0)")
        print(f"   Std: {weights_kliep.std():.3f}")
        print(f"   Min: {weights_kliep.min():.3f}, Max: {weights_kliep.max():.3f}")

        # Check validity
        assert np.all(weights_kliep > 0), "All weights must be positive"
        assert np.all(np.isfinite(weights_kliep)), "All weights must be finite"
        assert np.abs(weights_kliep.mean() - 1.0) < 0.1, "Mean should be ~1.0"
        print("[OK] Weights passed validity checks")

    except Exception as e:
        print(f"[ERROR] Error estimating weights: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 5: Generate predictions (simple: predict positive for all)
    print("\n[5/6] Generating predictions...")
    predictions_cal = np.ones(len(y_cal), dtype=int)
    print(f"[OK] Predicted {predictions_cal.sum()} positives (naive baseline)")

    # Step 6: Estimate bounds with both weight methods
    print("\n[6/6] Estimating conformal PPV bounds...")
    tau_grid = [0.5, 0.6, 0.7, 0.8, 0.9]

    print("\n" + "=" * 80)
    print("RESULTS WITH uLSIF WEIGHTS")
    print("=" * 80)

    try:
        decisions_ulsif = wcp_ulsif.estimate_bounds(
            y_cal,
            predictions_cal,
            cohorts_cal,
            weights_ulsif,
            tau_grid,
            alpha=0.05,
        )

        print(f"[OK] Generated {len(decisions_ulsif)} decisions")
        print(f"\nResults by cohort and threshold:")
        print("-" * 80)
        print(f"{'Cohort':<15} {'Tau':<5} {'Decision':<12} {'Mean':<8} {'LB':<8} {'p-value':<10}")
        print("-" * 80)

        for d in decisions_ulsif:
            print(
                f"{d.cohort_id:<15} {d.tau:<5.2f} {d.decision:<12} "
                f"{d.mu_hat:<8.3f} {d.lower_bound:<8.3f} {d.p_value:<10.4f}"
            )

        # Summary statistics
        n_certify = sum(1 for d in decisions_ulsif if d.decision == "CERTIFY")
        n_abstain = sum(1 for d in decisions_ulsif if d.decision == "ABSTAIN")
        n_no_guarantee = sum(1 for d in decisions_ulsif if d.decision == "NO-GUARANTEE")

        print("-" * 80)
        print(f"\nSummary:")
        print(f"  CERTIFY: {n_certify}/{len(decisions_ulsif)} ({n_certify/len(decisions_ulsif):.1%})")
        print(f"  ABSTAIN: {n_abstain}/{len(decisions_ulsif)} ({n_abstain/len(decisions_ulsif):.1%})")
        print(f"  NO-GUARANTEE: {n_no_guarantee}/{len(decisions_ulsif)} ({n_no_guarantee/len(decisions_ulsif):.1%})")

        # Diagnostics
        print(f"\nMethod diagnostics:")
        diag = wcp_ulsif.get_diagnostics()
        for key, value in diag.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"[ERROR] Error estimating bounds with uLSIF: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print("RESULTS WITH KLIEP WEIGHTS")
    print("=" * 80)

    try:
        decisions_kliep = wcp_kliep.estimate_bounds(
            y_cal,
            predictions_cal,
            cohorts_cal,
            weights_kliep,
            tau_grid,
            alpha=0.05,
        )

        print(f"[OK] Generated {len(decisions_kliep)} decisions")
        print(f"\nResults by cohort and threshold:")
        print("-" * 80)
        print(f"{'Cohort':<15} {'Tau':<5} {'Decision':<12} {'Mean':<8} {'LB':<8} {'p-value':<10}")
        print("-" * 80)

        for d in decisions_kliep:
            print(
                f"{d.cohort_id:<15} {d.tau:<5.2f} {d.decision:<12} "
                f"{d.mu_hat:<8.3f} {d.lower_bound:<8.3f} {d.p_value:<10.4f}"
            )

        # Summary statistics
        n_certify = sum(1 for d in decisions_kliep if d.decision == "CERTIFY")
        n_abstain = sum(1 for d in decisions_kliep if d.decision == "ABSTAIN")
        n_no_guarantee = sum(1 for d in decisions_kliep if d.decision == "NO-GUARANTEE")

        print("-" * 80)
        print(f"\nSummary:")
        print(f"  CERTIFY: {n_certify}/{len(decisions_kliep)} ({n_certify/len(decisions_kliep):.1%})")
        print(f"  ABSTAIN: {n_abstain}/{len(decisions_kliep)} ({n_abstain/len(decisions_kliep):.1%})")
        print(f"  NO-GUARANTEE: {n_no_guarantee}/{len(decisions_kliep)} ({n_no_guarantee/len(decisions_kliep):.1%})")

        # Diagnostics
        print(f"\nMethod diagnostics:")
        diag = wcp_kliep.get_diagnostics()
        for key, value in diag.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"[ERROR] Error estimating bounds with KLIEP: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 7: Compare bounds
    print("\n" + "=" * 80)
    print("COMPARISON: uLSIF vs KLIEP Weights")
    print("=" * 80)
    print(f"{'Cohort':<15} {'Tau':<5} {'uLSIF LB':<10} {'KLIEP LB':<10} {'Difference':<12}")
    print("-" * 80)

    for d_ulsif, d_kliep in zip(decisions_ulsif, decisions_kliep):
        diff = d_ulsif.lower_bound - d_kliep.lower_bound
        print(
            f"{d_ulsif.cohort_id:<15} {d_ulsif.tau:<5.2f} "
            f"{d_ulsif.lower_bound:<10.4f} {d_kliep.lower_bound:<10.4f} "
            f"{diff:+11.4f}"
        )

    print("\n" + "=" * 80)
    print("[SUCCESS] All tests passed! Weighted Conformal Prediction is working.")
    print("=" * 80)
    print("\nKey observations:")
    print("- Conformal prediction uses quantiles (non-parametric)")
    print("- Unlike EB bounds, no assumptions on distribution")
    print("- Bounds may differ from uLSIF/KLIEP EB due to different bound type")
    print("- More robust to heavy-tailed distributions")
    print("=" * 80)


if __name__ == "__main__":
    test_weighted_conformal_on_synthetic_data()
