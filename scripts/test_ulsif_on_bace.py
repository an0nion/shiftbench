"""Test uLSIF baseline on real BACE dataset.

This script validates that:
1. BACE loads correctly from shift-bench format
2. uLSIF produces valid weights on molecular data
3. uLSIF produces reasonable certification decisions
4. Results are documented for comparison with RAVEL

Usage:
    python scripts/test_ulsif_on_bace.py
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from shiftbench.baselines.ulsif import create_ulsif_baseline
from shiftbench.data import load_dataset


def test_ulsif_on_bace():
    """Run uLSIF on BACE and compare to expectations."""

    print("=" * 80)
    print("Testing uLSIF Baseline on BACE (Real Molecular Data)")
    print("=" * 80)

    # Step 1: Load BACE
    print("\n[1/5] Loading BACE dataset...")
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
    print("\n[2/5] Splitting into calibration and test sets...")
    cal_mask = (splits["split"] == "cal").values
    test_mask = (splits["split"] == "test").values

    X_cal, y_cal, cohorts_cal = X[cal_mask], y[cal_mask], cohorts[cal_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"[OK] Calibration: {len(X_cal)} samples")
    print(f"   Test: {len(X_test)} samples")
    print(f"   Cal positive rate: {y_cal.mean():.2%}")
    print(f"   Test positive rate: {y_test.mean():.2%}")

    # Step 3: Estimate weights
    print("\n[3/5] Estimating importance weights with uLSIF...")
    start_time = time.time()

    ulsif = create_ulsif_baseline(
        n_basis=100,  # More basis functions for complex molecular data
        sigma=None,   # Use median heuristic
        lambda_=0.1,
        random_state=42,
    )

    try:
        weights = ulsif.estimate_weights(X_cal, X_test)
        elapsed = time.time() - start_time

        print(f"[OK] Estimated weights for {len(weights)} calibration samples ({elapsed:.1f}s)")
        print(f"   Mean: {weights.mean():.3f} (should be ~1.0)")
        print(f"   Std: {weights.std():.3f}")
        print(f"   Min: {weights.min():.3f}, Max: {weights.max():.3f}")
        print(f"   Median: {np.median(weights):.3f}")
        print(f"   95th percentile: {np.percentile(weights, 95):.3f}")

        # Check validity
        assert np.all(weights > 0), "All weights must be positive"
        assert np.all(np.isfinite(weights)), "All weights must be finite"
        assert np.abs(weights.mean() - 1.0) < 0.1, "Mean should be ~1.0"
        print("[OK] Weights passed validity checks")

    except Exception as e:
        print(f"[ERROR] Error estimating weights: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 4: Generate predictions (oracle: use true labels)
    print("\n[4/5] Generating predictions (oracle: true labels)...")
    # In real use, these would be model predictions
    # For testing, we use true labels to isolate weight estimation quality
    predictions_cal = y_cal.astype(int)
    print(f"[OK] Using {predictions_cal.sum()} predicted positives (oracle mode)")

    # Step 5: Estimate bounds
    print("\n[5/5] Estimating PPV bounds...")
    tau_grid = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]

    start_time = time.time()
    try:
        decisions = ulsif.estimate_bounds(
            y_cal,
            predictions_cal,
            cohorts_cal,
            weights,
            tau_grid,
            alpha=0.05,
        )
        elapsed = time.time() - start_time

        print(f"[OK] Generated {len(decisions)} decisions ({elapsed:.1f}s)")

        # Group by tau for easier analysis
        print(f"\nResults by threshold (aggregated across cohorts):")
        print("-" * 80)
        print(f"{'Tau':<6} {'Total':<8} {'Certify':<10} {'Abstain':<10} {'Cert Rate':<12}")
        print("-" * 80)

        for tau in tau_grid:
            tau_decisions = [d for d in decisions if d.tau == tau]
            n_total = len(tau_decisions)
            n_certify = sum(1 for d in tau_decisions if d.decision == "CERTIFY")
            n_abstain = sum(1 for d in tau_decisions if d.decision == "ABSTAIN")
            cert_rate = n_certify / n_total if n_total > 0 else 0

            print(f"{tau:<6.2f} {n_total:<8} {n_certify:<10} {n_abstain:<10} {cert_rate:<12.1%}")

        # Overall summary
        n_certify = sum(1 for d in decisions if d.decision == "CERTIFY")
        n_abstain = sum(1 for d in decisions if d.decision == "ABSTAIN")
        n_no_guarantee = sum(1 for d in decisions if d.decision == "NO-GUARANTEE")

        print("-" * 80)
        print(f"\nOverall Summary:")
        print(f"  CERTIFY: {n_certify}/{len(decisions)} ({n_certify/len(decisions):.1%})")
        print(f"  ABSTAIN: {n_abstain}/{len(decisions)} ({n_abstain/len(decisions):.1%})")
        print(f"  NO-GUARANTEE: {n_no_guarantee}/{len(decisions)} ({n_no_guarantee/len(decisions):.1%})")

        # Analyze cohorts with highest certification rates
        cohort_cert_rates = {}
        for cohort_id in np.unique(cohorts_cal):
            cohort_decisions = [d for d in decisions if d.cohort_id == cohort_id]
            n_cert = sum(1 for d in cohort_decisions if d.decision == "CERTIFY")
            cohort_cert_rates[cohort_id] = n_cert / len(cohort_decisions) if cohort_decisions else 0

        # Top 10 cohorts
        top_cohorts = sorted(cohort_cert_rates.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\nTop 10 Cohorts by Certification Rate:")
        for cohort_id, rate in top_cohorts:
            n_samples = (cohorts_cal == cohort_id).sum()
            print(f"  {cohort_id[:40]:<40} {rate:.1%} (n={n_samples})")

        # Method diagnostics
        print(f"\nMethod Diagnostics:")
        diag = ulsif.get_diagnostics()
        for key, value in diag.items():
            print(f"  {key}: {value}")

        # Save results to CSV
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

        output_path = Path(__file__).parent.parent / "results" / "ulsif_bace_results.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\n[OK] Results saved to {output_path}")

    except Exception as e:
        print(f"[ERROR] Error estimating bounds: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print("[SUCCESS] uLSIF validation on BACE complete!")
    print("=" * 80)

    # Compare to RAVEL expectations
    print("\nComparison Notes:")
    print("  - RAVEL (from real_data_comparison.csv):")
    print("    * State: PASS")
    print("    * c_final: 1.4")
    print("    * PSIS k-hat: 0.085846 (good)")
    print("    * ESS fraction: 0.98031 (excellent)")
    print("    * Certified: 1 cohort (at tau=0.9)")
    print("")
    print("  - uLSIF (this run):")
    print(f"    * No stability gating (no NO-GUARANTEE)")
    print(f"    * Certification rate: {n_certify}/{len(decisions)} ({n_certify/len(decisions):.1%})")
    print(f"    * Expected: Lower cert rate than RAVEL (no gating means wider bounds)")
    print("")
    print("  Interpretation:")
    print("    - uLSIF should certify FEWER cohorts than RAVEL")
    print("    - This is expected: no stability diagnostics means less confident bounds")
    print("    - Key validation: weights are valid and bounds are conservative")


if __name__ == "__main__":
    test_ulsif_on_bace()
