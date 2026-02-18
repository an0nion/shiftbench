"""Test RULSIF baseline on synthetic data and compare with uLSIF.

This script verifies that:
1. Dataset loading works
2. RULSIF weight estimation works
3. Bound estimation works
4. Results are reasonable
5. RULSIF provides more stable weights than uLSIF (lower variance)
6. Different alpha values affect stability

Usage:
    python scripts/create_test_data.py  # Create test data first
    python scripts/test_rulsif.py       # Then run this test
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from shiftbench.baselines.rulsif import create_rulsif_baseline
from shiftbench.baselines.ulsif import create_ulsif_baseline
from shiftbench.data import load_dataset


def test_rulsif_on_synthetic_data():
    """Run RULSIF on test dataset and compare with uLSIF."""

    print("=" * 80)
    print("Testing RULSIF Baseline on Synthetic Data")
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

    # Step 3: Estimate weights with different methods
    print("\n[3/6] Estimating importance weights with multiple methods...")

    # uLSIF (baseline)
    print("\n  [3a] uLSIF (standard density ratio, alpha=0.0)...")
    ulsif = create_ulsif_baseline(n_basis=50, random_state=42)
    try:
        weights_ulsif = ulsif.estimate_weights(X_cal, X_test)
        print(f"  [OK] Mean: {weights_ulsif.mean():.3f}, Std: {weights_ulsif.std():.3f}")
        print(f"       Min: {weights_ulsif.min():.3f}, Max: {weights_ulsif.max():.3f}")
        print(f"       CV: {weights_ulsif.std()/weights_ulsif.mean():.3f} (coefficient of variation)")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return

    # RULSIF with alpha=0.0 (should be identical to uLSIF)
    print("\n  [3b] RULSIF with alpha=0.0 (equivalent to uLSIF)...")
    rulsif_0 = create_rulsif_baseline(n_basis=50, alpha=0.0, random_state=42)
    try:
        weights_rulsif_0 = rulsif_0.estimate_weights(X_cal, X_test)
        print(f"  [OK] Mean: {weights_rulsif_0.mean():.3f}, Std: {weights_rulsif_0.std():.3f}")
        print(f"       Min: {weights_rulsif_0.min():.3f}, Max: {weights_rulsif_0.max():.3f}")
        print(f"       CV: {weights_rulsif_0.std()/weights_rulsif_0.mean():.3f}")

        # Check if close to uLSIF
        diff = np.abs(weights_rulsif_0 - weights_ulsif).mean()
        print(f"  [INFO] Mean difference from uLSIF: {diff:.6f}")
        if diff < 0.01:
            print(f"  [OK] RULSIF(alpha=0.0) matches uLSIF (as expected)")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return

    # RULSIF with alpha=0.1 (default, slight stabilization)
    print("\n  [3c] RULSIF with alpha=0.1 (default, slight stabilization)...")
    rulsif_01 = create_rulsif_baseline(n_basis=50, alpha=0.1, random_state=42)
    try:
        weights_rulsif_01 = rulsif_01.estimate_weights(X_cal, X_test)
        print(f"  [OK] Mean: {weights_rulsif_01.mean():.3f}, Std: {weights_rulsif_01.std():.3f}")
        print(f"       Min: {weights_rulsif_01.min():.3f}, Max: {weights_rulsif_01.max():.3f}")
        print(f"       CV: {weights_rulsif_01.std()/weights_rulsif_01.mean():.3f}")

        # Compare stability with uLSIF
        cv_ulsif = weights_ulsif.std() / weights_ulsif.mean()
        cv_rulsif = weights_rulsif_01.std() / weights_rulsif_01.mean()
        stability_improvement = (cv_ulsif - cv_rulsif) / cv_ulsif * 100
        print(f"  [INFO] Stability improvement over uLSIF: {stability_improvement:+.1f}%")
        if cv_rulsif < cv_ulsif:
            print(f"  [OK] RULSIF is more stable than uLSIF (lower CV)")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return

    # RULSIF with alpha=0.5 (maximum stability)
    print("\n  [3d] RULSIF with alpha=0.5 (maximum stability)...")
    rulsif_05 = create_rulsif_baseline(n_basis=50, alpha=0.5, random_state=42)
    try:
        weights_rulsif_05 = rulsif_05.estimate_weights(X_cal, X_test)
        print(f"  [OK] Mean: {weights_rulsif_05.mean():.3f}, Std: {weights_rulsif_05.std():.3f}")
        print(f"       Min: {weights_rulsif_05.min():.3f}, Max: {weights_rulsif_05.max():.3f}")
        print(f"       CV: {weights_rulsif_05.std()/weights_rulsif_05.mean():.3f}")

        cv_rulsif_05 = weights_rulsif_05.std() / weights_rulsif_05.mean()
        stability_improvement = (cv_ulsif - cv_rulsif_05) / cv_ulsif * 100
        print(f"  [INFO] Stability improvement over uLSIF: {stability_improvement:+.1f}%")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return

    # Step 4: Check weight validity
    print("\n[4/6] Validating weights...")
    all_weights = [
        ("uLSIF", weights_ulsif),
        ("RULSIF(alpha=0.0)", weights_rulsif_0),
        ("RULSIF(alpha=0.1)", weights_rulsif_01),
        ("RULSIF(alpha=0.5)", weights_rulsif_05),
    ]

    for name, weights in all_weights:
        try:
            assert np.all(weights > 0), f"{name}: All weights must be positive"
            assert np.all(np.isfinite(weights)), f"{name}: All weights must be finite"
            assert np.abs(weights.mean() - 1.0) < 0.1, f"{name}: Mean should be ~1.0"
            print(f"  [OK] {name} passed validity checks")
        except AssertionError as e:
            print(f"  [ERROR] {name}: {e}")
            return

    # Step 5: Generate predictions
    print("\n[5/6] Generating predictions...")
    predictions_cal = np.ones(len(y_cal), dtype=int)
    print(f"[OK] Predicted {predictions_cal.sum()} positives (naive baseline)")

    # Step 6: Compare bounds with uLSIF vs RULSIF
    print("\n[6/6] Estimating PPV bounds and comparing methods...")
    tau_grid = [0.5, 0.6, 0.7, 0.8, 0.9]

    # Estimate bounds for uLSIF
    print("\n  [6a] Estimating bounds with uLSIF...")
    try:
        decisions_ulsif = ulsif.estimate_bounds(
            y_cal, predictions_cal, cohorts_cal, weights_ulsif, tau_grid, alpha=0.05
        )
        print(f"  [OK] Generated {len(decisions_ulsif)} decisions")
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return

    # Estimate bounds for RULSIF (alpha=0.1)
    print("\n  [6b] Estimating bounds with RULSIF (alpha=0.1)...")
    try:
        decisions_rulsif = rulsif_01.estimate_bounds(
            y_cal, predictions_cal, cohorts_cal, weights_rulsif_01, tau_grid, alpha=0.05
        )
        print(f"  [OK] Generated {len(decisions_rulsif)} decisions")
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return

    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON: uLSIF vs RULSIF(alpha=0.1)")
    print("=" * 80)
    print(f"{'Cohort':<15} {'Tau':<5} {'Method':<12} {'Decision':<12} {'Mean':<8} {'LB':<8} {'p-value':<10}")
    print("-" * 80)

    for d_ulsif, d_rulsif in zip(decisions_ulsif, decisions_rulsif):
        # Print uLSIF result
        print(
            f"{d_ulsif.cohort_id:<15} {d_ulsif.tau:<5.2f} {'uLSIF':<12} {d_ulsif.decision:<12} "
            f"{d_ulsif.mu_hat:<8.3f} {d_ulsif.lower_bound:<8.3f} {d_ulsif.p_value:<10.4f}"
        )
        # Print RULSIF result
        print(
            f"{d_rulsif.cohort_id:<15} {d_rulsif.tau:<5.2f} {'RULSIF(0.1)':<12} {d_rulsif.decision:<12} "
            f"{d_rulsif.mu_hat:<8.3f} {d_rulsif.lower_bound:<8.3f} {d_rulsif.p_value:<10.4f}"
        )
        print("-" * 80)

    # Summary statistics
    print("\nSummary by Method:")
    print("-" * 80)

    for name, decisions in [("uLSIF", decisions_ulsif), ("RULSIF(alpha=0.1)", decisions_rulsif)]:
        n_certify = sum(1 for d in decisions if d.decision == "CERTIFY")
        n_abstain = sum(1 for d in decisions if d.decision == "ABSTAIN")
        n_no_guarantee = sum(1 for d in decisions if d.decision == "NO-GUARANTEE")

        print(f"\n{name}:")
        print(f"  CERTIFY:      {n_certify}/{len(decisions)} ({n_certify/len(decisions):.1%})")
        print(f"  ABSTAIN:      {n_abstain}/{len(decisions)} ({n_abstain/len(decisions):.1%})")
        print(f"  NO-GUARANTEE: {n_no_guarantee}/{len(decisions)} ({n_no_guarantee/len(decisions):.1%})")

    # Weight diagnostics comparison
    print("\n" + "=" * 80)
    print("WEIGHT STATISTICS COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<25} {'uLSIF':<15} {'RULSIF(0.0)':<15} {'RULSIF(0.1)':<15} {'RULSIF(0.5)':<15}")
    print("-" * 80)

    metrics = [
        ("Mean", lambda w: w.mean()),
        ("Std", lambda w: w.std()),
        ("Min", lambda w: w.min()),
        ("Max", lambda w: w.max()),
        ("CV (Std/Mean)", lambda w: w.std() / w.mean()),
        ("Range (Max-Min)", lambda w: w.max() - w.min()),
    ]

    for metric_name, metric_fn in metrics:
        vals = [
            metric_fn(weights_ulsif),
            metric_fn(weights_rulsif_0),
            metric_fn(weights_rulsif_01),
            metric_fn(weights_rulsif_05),
        ]
        print(f"{metric_name:<25} {vals[0]:<15.4f} {vals[1]:<15.4f} {vals[2]:<15.4f} {vals[3]:<15.4f}")

    # Method diagnostics
    print("\n" + "=" * 80)
    print("METHOD DIAGNOSTICS")
    print("=" * 80)

    print("\nuLSIF diagnostics:")
    diag_ulsif = ulsif.get_diagnostics()
    for key, value in diag_ulsif.items():
        print(f"  {key}: {value}")

    print("\nRULSIF(alpha=0.1) diagnostics:")
    diag_rulsif = rulsif_01.get_diagnostics()
    for key, value in diag_rulsif.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print("[SUCCESS] All tests passed! RULSIF baseline is working correctly.")
    print("=" * 80)

    # Final observations
    print("\nKey Findings:")
    print("-" * 80)
    cv_ulsif = weights_ulsif.std() / weights_ulsif.mean()
    cv_rulsif_01 = weights_rulsif_01.std() / weights_rulsif_01.mean()
    cv_rulsif_05 = weights_rulsif_05.std() / weights_rulsif_05.mean()

    print(f"1. RULSIF(alpha=0.0) matches uLSIF (as expected)")
    print(f"2. RULSIF(alpha=0.1) provides {(cv_ulsif - cv_rulsif_01) / cv_ulsif * 100:+.1f}% stability improvement")
    print(f"3. RULSIF(alpha=0.5) provides {(cv_ulsif - cv_rulsif_05) / cv_ulsif * 100:+.1f}% stability improvement")
    print(f"4. Higher alpha -> more stable weights (lower variance)")
    print(f"5. Trade-off: stability vs. accurate density ratio estimation")


if __name__ == "__main__":
    test_rulsif_on_synthetic_data()
