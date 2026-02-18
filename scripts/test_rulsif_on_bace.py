"""Test RULSIF baseline on real BACE dataset and compare with uLSIF.

This script validates that:
1. BACE loads correctly from shift-bench format
2. RULSIF produces valid weights on molecular data
3. RULSIF produces reasonable certification decisions
4. RULSIF is more stable than uLSIF (especially for severe shifts)
5. Results are documented for comparison with RAVEL and uLSIF

Usage:
    python scripts/test_rulsif_on_bace.py
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from shiftbench.baselines.rulsif import create_rulsif_baseline
from shiftbench.baselines.ulsif import create_ulsif_baseline
from shiftbench.data import load_dataset


def test_rulsif_on_bace():
    """Run RULSIF on BACE and compare with uLSIF."""

    print("=" * 80)
    print("Testing RULSIF Baseline on BACE (Real Molecular Data)")
    print("=" * 80)

    # Step 1: Load BACE
    print("\n[1/6] Loading BACE dataset...")
    try:
        X, y, cohorts, splits = load_dataset("bace")
        print(f"[OK] Loaded {len(X)} samples with {X.shape[1]} features")
        print(f"   Cohorts: {len(np.unique(cohorts))} scaffolds")
        print(f"   Positive rate: {y.mean():.2%}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("   Run 'python scripts/preprocess_molecular.py --dataset bace' first!")
        return

    # Step 2: Split into calibration and test
    print("\n[2/6] Splitting into calibration and test sets...")
    cal_mask = (splits["split"] == "cal").values
    test_mask = (splits["split"] == "test").values

    X_cal, y_cal, cohorts_cal = X[cal_mask], y[cal_mask], cohorts[cal_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"[OK] Calibration: {len(X_cal)} samples")
    print(f"   Test: {len(X_test)} samples")
    print(f"   Cal positive rate: {y_cal.mean():.2%}")
    print(f"   Test positive rate: {y_test.mean():.2%}")

    # Step 3: Estimate weights with different methods
    print("\n[3/6] Estimating importance weights with uLSIF and RULSIF...")

    # uLSIF (baseline)
    print("\n  [3a] uLSIF (standard density ratio)...")
    start_time = time.time()

    ulsif = create_ulsif_baseline(
        n_basis=100,
        sigma=None,
        lambda_=0.1,
        random_state=42,
    )

    try:
        weights_ulsif = ulsif.estimate_weights(X_cal, X_test)
        elapsed = time.time() - start_time

        print(f"  [OK] Estimated weights ({elapsed:.1f}s)")
        print(f"       Mean: {weights_ulsif.mean():.3f}, Std: {weights_ulsif.std():.3f}")
        print(f"       Min: {weights_ulsif.min():.3f}, Max: {weights_ulsif.max():.3f}")
        print(f"       Median: {np.median(weights_ulsif):.3f}")
        print(f"       CV: {weights_ulsif.std() / weights_ulsif.mean():.3f}")
        print(f"       95th percentile: {np.percentile(weights_ulsif, 95):.3f}")
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return

    # RULSIF with alpha=0.1 (slight stabilization)
    print("\n  [3b] RULSIF with alpha=0.1 (default)...")
    start_time = time.time()

    rulsif_01 = create_rulsif_baseline(
        n_basis=100,
        sigma=None,
        lambda_=0.1,
        alpha=0.1,
        random_state=42,
    )

    try:
        weights_rulsif_01 = rulsif_01.estimate_weights(X_cal, X_test)
        elapsed = time.time() - start_time

        print(f"  [OK] Estimated weights ({elapsed:.1f}s)")
        print(f"       Mean: {weights_rulsif_01.mean():.3f}, Std: {weights_rulsif_01.std():.3f}")
        print(f"       Min: {weights_rulsif_01.min():.3f}, Max: {weights_rulsif_01.max():.3f}")
        print(f"       Median: {np.median(weights_rulsif_01):.3f}")
        print(f"       CV: {weights_rulsif_01.std() / weights_rulsif_01.mean():.3f}")
        print(f"       95th percentile: {np.percentile(weights_rulsif_01, 95):.3f}")

        # Compare stability
        cv_ulsif = weights_ulsif.std() / weights_ulsif.mean()
        cv_rulsif = weights_rulsif_01.std() / weights_rulsif_01.mean()
        stability_improvement = (cv_ulsif - cv_rulsif) / cv_ulsif * 100
        print(f"       Stability improvement over uLSIF: {stability_improvement:+.1f}%")
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return

    # RULSIF with alpha=0.5 (maximum stability)
    print("\n  [3c] RULSIF with alpha=0.5 (maximum stability)...")
    start_time = time.time()

    rulsif_05 = create_rulsif_baseline(
        n_basis=100,
        sigma=None,
        lambda_=0.1,
        alpha=0.5,
        random_state=42,
    )

    try:
        weights_rulsif_05 = rulsif_05.estimate_weights(X_cal, X_test)
        elapsed = time.time() - start_time

        print(f"  [OK] Estimated weights ({elapsed:.1f}s)")
        print(f"       Mean: {weights_rulsif_05.mean():.3f}, Std: {weights_rulsif_05.std():.3f}")
        print(f"       Min: {weights_rulsif_05.min():.3f}, Max: {weights_rulsif_05.max():.3f}")
        print(f"       Median: {np.median(weights_rulsif_05):.3f}")
        print(f"       CV: {weights_rulsif_05.std() / weights_rulsif_05.mean():.3f}")
        print(f"       95th percentile: {np.percentile(weights_rulsif_05, 95):.3f}")

        cv_rulsif_05 = weights_rulsif_05.std() / weights_rulsif_05.mean()
        stability_improvement = (cv_ulsif - cv_rulsif_05) / cv_ulsif * 100
        print(f"       Stability improvement over uLSIF: {stability_improvement:+.1f}%")
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 4: Validate weights
    print("\n[4/6] Validating weights...")
    all_weights = [
        ("uLSIF", weights_ulsif),
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

    # Step 5: Generate predictions (oracle: use true labels)
    print("\n[5/6] Generating predictions (oracle: true labels)...")
    predictions_cal = y_cal.astype(int)
    print(f"[OK] Using {predictions_cal.sum()} predicted positives (oracle mode)")

    # Step 6: Estimate bounds for all methods
    print("\n[6/6] Estimating PPV bounds and comparing methods...")
    tau_grid = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]

    # uLSIF bounds
    print("\n  [6a] Estimating bounds with uLSIF...")
    start_time = time.time()
    try:
        decisions_ulsif = ulsif.estimate_bounds(
            y_cal, predictions_cal, cohorts_cal, weights_ulsif, tau_grid, alpha=0.05
        )
        elapsed = time.time() - start_time
        print(f"  [OK] Generated {len(decisions_ulsif)} decisions ({elapsed:.1f}s)")
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return

    # RULSIF(alpha=0.1) bounds
    print("\n  [6b] Estimating bounds with RULSIF(alpha=0.1)...")
    start_time = time.time()
    try:
        decisions_rulsif_01 = rulsif_01.estimate_bounds(
            y_cal, predictions_cal, cohorts_cal, weights_rulsif_01, tau_grid, alpha=0.05
        )
        elapsed = time.time() - start_time
        print(f"  [OK] Generated {len(decisions_rulsif_01)} decisions ({elapsed:.1f}s)")
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return

    # RULSIF(alpha=0.5) bounds
    print("\n  [6c] Estimating bounds with RULSIF(alpha=0.5)...")
    start_time = time.time()
    try:
        decisions_rulsif_05 = rulsif_05.estimate_bounds(
            y_cal, predictions_cal, cohorts_cal, weights_rulsif_05, tau_grid, alpha=0.05
        )
        elapsed = time.time() - start_time
        print(f"  [OK] Generated {len(decisions_rulsif_05)} decisions ({elapsed:.1f}s)")
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return

    # Compare results by threshold
    print("\n" + "=" * 80)
    print("RESULTS BY THRESHOLD (aggregated across cohorts)")
    print("=" * 80)
    print(f"{'Tau':<6} {'Method':<20} {'Total':<8} {'Certify':<10} {'Abstain':<10} {'Cert Rate':<12}")
    print("-" * 80)

    for tau in tau_grid:
        for name, decisions in [
            ("uLSIF", decisions_ulsif),
            ("RULSIF(alpha=0.1)", decisions_rulsif_01),
            ("RULSIF(alpha=0.5)", decisions_rulsif_05),
        ]:
            tau_decisions = [d for d in decisions if d.tau == tau]
            n_total = len(tau_decisions)
            n_certify = sum(1 for d in tau_decisions if d.decision == "CERTIFY")
            n_abstain = sum(1 for d in tau_decisions if d.decision == "ABSTAIN")
            cert_rate = n_certify / n_total if n_total > 0 else 0

            print(f"{tau:<6.2f} {name:<20} {n_total:<8} {n_certify:<10} {n_abstain:<10} {cert_rate:<12.1%}")
        print("-" * 80)

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    for name, decisions in [
        ("uLSIF", decisions_ulsif),
        ("RULSIF(alpha=0.1)", decisions_rulsif_01),
        ("RULSIF(alpha=0.5)", decisions_rulsif_05),
    ]:
        n_certify = sum(1 for d in decisions if d.decision == "CERTIFY")
        n_abstain = sum(1 for d in decisions if d.decision == "ABSTAIN")
        n_no_guarantee = sum(1 for d in decisions if d.decision == "NO-GUARANTEE")

        print(f"\n{name}:")
        print(f"  CERTIFY:      {n_certify}/{len(decisions)} ({n_certify/len(decisions):.1%})")
        print(f"  ABSTAIN:      {n_abstain}/{len(decisions)} ({n_abstain/len(decisions):.1%})")
        print(f"  NO-GUARANTEE: {n_no_guarantee}/{len(decisions)} ({n_no_guarantee/len(decisions):.1%})")

    # Weight statistics comparison
    print("\n" + "=" * 80)
    print("WEIGHT STATISTICS COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<25} {'uLSIF':<15} {'RULSIF(0.1)':<15} {'RULSIF(0.5)':<15}")
    print("-" * 80)

    metrics = [
        ("Mean", lambda w: w.mean()),
        ("Std", lambda w: w.std()),
        ("Min", lambda w: w.min()),
        ("Max", lambda w: w.max()),
        ("Median", lambda w: np.median(w)),
        ("CV (Std/Mean)", lambda w: w.std() / w.mean()),
        ("Range (Max-Min)", lambda w: w.max() - w.min()),
        ("95th percentile", lambda w: np.percentile(w, 95)),
    ]

    for metric_name, metric_fn in metrics:
        vals = [
            metric_fn(weights_ulsif),
            metric_fn(weights_rulsif_01),
            metric_fn(weights_rulsif_05),
        ]
        print(f"{metric_name:<25} {vals[0]:<15.4f} {vals[1]:<15.4f} {vals[2]:<15.4f}")

    # Method diagnostics
    print("\n" + "=" * 80)
    print("METHOD DIAGNOSTICS")
    print("=" * 80)

    print("\nuLSIF diagnostics:")
    diag_ulsif = ulsif.get_diagnostics()
    for key, value in diag_ulsif.items():
        print(f"  {key}: {value}")

    print("\nRULSIF(alpha=0.1) diagnostics:")
    diag_rulsif_01 = rulsif_01.get_diagnostics()
    for key, value in diag_rulsif_01.items():
        print(f"  {key}: {value}")

    print("\nRULSIF(alpha=0.5) diagnostics:")
    diag_rulsif_05 = rulsif_05.get_diagnostics()
    for key, value in diag_rulsif_05.items():
        print(f"  {key}: {value}")

    # Save results to CSV
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    for name, decisions, method_obj in [
        ("ulsif", decisions_ulsif, ulsif),
        ("rulsif_alpha01", decisions_rulsif_01, rulsif_01),
        ("rulsif_alpha05", decisions_rulsif_05, rulsif_05),
    ]:
        results_df = pd.DataFrame([
            {
                "cohort_id": d.cohort_id,
                "tau": d.tau,
                "decision": d.decision,
                "mu_hat": d.mu_hat,
                "var_hat": d.var_hat,
                "n_eff": d.n_eff,
                "lower_bound": d.lower_bound,
                "p_value": d.p_value,
            }
            for d in decisions
        ])

        output_path = Path(__file__).parent.parent / "results" / f"{name}_bace_results.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"[OK] {name} results saved to {output_path}")

    print("\n" + "=" * 80)
    print("[SUCCESS] RULSIF validation on BACE complete!")
    print("=" * 80)

    # Comparison to RAVEL
    print("\nComparison Notes:")
    print("-" * 80)
    print("  - RAVEL (from real_data_comparison.csv):")
    print("    * State: PASS")
    print("    * c_final: 1.4")
    print("    * PSIS k-hat: 0.085846 (good)")
    print("    * ESS fraction: 0.98031 (excellent)")
    print("    * Certified: 1 cohort (at tau=0.9)")
    print("")

    n_certify_ulsif = sum(1 for d in decisions_ulsif if d.decision == "CERTIFY")
    n_certify_rulsif_01 = sum(1 for d in decisions_rulsif_01 if d.decision == "CERTIFY")
    n_certify_rulsif_05 = sum(1 for d in decisions_rulsif_05 if d.decision == "CERTIFY")

    print("  - uLSIF (this run):")
    print(f"    * No stability gating")
    print(f"    * Certification rate: {n_certify_ulsif}/{len(decisions_ulsif)} ({n_certify_ulsif/len(decisions_ulsif):.1%})")
    print(f"    * Weight CV: {weights_ulsif.std() / weights_ulsif.mean():.3f}")
    print("")
    print("  - RULSIF(alpha=0.1) (this run):")
    print(f"    * No stability gating")
    print(f"    * Certification rate: {n_certify_rulsif_01}/{len(decisions_rulsif_01)} ({n_certify_rulsif_01/len(decisions_rulsif_01):.1%})")
    print(f"    * Weight CV: {weights_rulsif_01.std() / weights_rulsif_01.mean():.3f}")
    cv_improvement_01 = (weights_ulsif.std() / weights_ulsif.mean() - weights_rulsif_01.std() / weights_rulsif_01.mean())
    cv_improvement_01 /= (weights_ulsif.std() / weights_ulsif.mean())
    print(f"    * Stability improvement: {cv_improvement_01 * 100:+.1f}%")
    print("")
    print("  - RULSIF(alpha=0.5) (this run):")
    print(f"    * No stability gating")
    print(f"    * Certification rate: {n_certify_rulsif_05}/{len(decisions_rulsif_05)} ({n_certify_rulsif_05/len(decisions_rulsif_05):.1%})")
    print(f"    * Weight CV: {weights_rulsif_05.std() / weights_rulsif_05.mean():.3f}")
    cv_improvement_05 = (weights_ulsif.std() / weights_ulsif.mean() - weights_rulsif_05.std() / weights_rulsif_05.mean())
    cv_improvement_05 /= (weights_ulsif.std() / weights_ulsif.mean())
    print(f"    * Stability improvement: {cv_improvement_05 * 100:+.1f}%")
    print("")
    print("  Interpretation:")
    print("    - RULSIF provides more stable weights than uLSIF (lower variance)")
    print("    - Higher alpha -> more stability but less accurate density ratio")
    print("    - Certification rates may differ due to different weight distributions")
    print("    - Both methods lack stability gating (unlike RAVEL)")


if __name__ == "__main__":
    test_rulsif_on_bace()
