"""Test Weighted Conformal Prediction baseline on BACE dataset.

This script compares Weighted Conformal Prediction against uLSIF and KLIEP
baselines on the BACE molecular property prediction dataset.

Key comparisons:
- Conformal (quantile-based) vs EB (parametric) bounds
- Distribution-free guarantees vs sub-Gaussian assumptions
- Robustness to heavy-tailed distributions

Usage:
    python scripts/test_weighted_conformal_on_bace.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from shiftbench.baselines.weighted_conformal import create_weighted_conformal_baseline
from shiftbench.baselines.ulsif import create_ulsif_baseline
from shiftbench.baselines.kliep import create_kliep_baseline
from shiftbench.data import load_dataset


def test_weighted_conformal_on_bace():
    """Run Weighted Conformal Prediction on BACE and compare with EB methods."""

    print("=" * 80)
    print("Testing Weighted Conformal Prediction on BACE Dataset")
    print("=" * 80)

    # Step 1: Load BACE dataset
    print("\n[1/7] Loading BACE dataset...")
    try:
        X, y, cohorts, splits = load_dataset("bace")
        print(f"[OK] Loaded {len(X)} samples with {X.shape[1]} features")
        print(f"   Unique cohorts: {len(np.unique(cohorts))}")
        print(f"   Positive rate: {y.mean():.2%}")
        print(f"   Cohort distribution:")
        for cohort in np.unique(cohorts):
            n_cohort = (cohorts == cohort).sum()
            print(f"      {cohort}: {n_cohort} samples ({n_cohort/len(cohorts):.1%})")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 2: Split into calibration and test
    print("\n[2/7] Splitting into calibration and test sets...")
    cal_mask = (splits["split"] == "cal").values
    test_mask = (splits["split"] == "test").values

    X_cal, y_cal, cohorts_cal = X[cal_mask], y[cal_mask], cohorts[cal_mask]
    X_test = X[test_mask]

    print(f"[OK] Calibration: {len(X_cal)} samples")
    print(f"   Test: {len(X_test)} samples")
    print(f"   Cal positive rate: {y_cal.mean():.2%}")

    # Step 3: Initialize methods
    print("\n[3/7] Initializing methods...")
    wcp_ulsif = create_weighted_conformal_baseline(
        weight_method="ulsif",
        n_basis=100,
        random_state=42
    )
    wcp_kliep = create_weighted_conformal_baseline(
        weight_method="kliep",
        n_basis=100,
        max_iter=10000,
        random_state=42
    )
    ulsif_eb = create_ulsif_baseline(n_basis=100, random_state=42)
    kliep_eb = create_kliep_baseline(n_basis=100, max_iter=10000, random_state=42)
    print("[OK] Initialized 4 methods: WCP-uLSIF, WCP-KLIEP, uLSIF-EB, KLIEP-EB")

    # Step 4: Estimate weights
    print("\n[4/7] Estimating importance weights...")

    print("   Estimating uLSIF weights...")
    weights_ulsif = wcp_ulsif.estimate_weights(X_cal, X_test)
    print(f"   [OK] uLSIF: mean={weights_ulsif.mean():.3f}, "
          f"std={weights_ulsif.std():.3f}, max={weights_ulsif.max():.3f}")

    print("   Estimating KLIEP weights...")
    weights_kliep = wcp_kliep.estimate_weights(X_cal, X_test)
    print(f"   [OK] KLIEP: mean={weights_kliep.mean():.3f}, "
          f"std={weights_kliep.std():.3f}, max={weights_kliep.max():.3f}")

    # Step 5: Generate predictions (using a simple threshold model)
    print("\n[5/7] Generating predictions...")
    # Simple model: use first principal component
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1, random_state=42)
    X_cal_pc = pca.fit_transform(X_cal).ravel()

    # Predict positive if PC1 > threshold (tune for ~50% positive rate)
    threshold = np.median(X_cal_pc)
    predictions_cal = (X_cal_pc > threshold).astype(int)

    n_pred_pos = predictions_cal.sum()
    print(f"[OK] Predicted {n_pred_pos} positives ({n_pred_pos/len(predictions_cal):.1%})")
    print(f"   True positive rate among predictions: "
          f"{y_cal[predictions_cal == 1].mean():.2%}")

    # Step 6: Estimate bounds for all methods
    print("\n[6/7] Estimating PPV bounds...")
    tau_grid = [0.5, 0.6, 0.7, 0.8]

    print("   Computing WCP-uLSIF bounds...")
    decisions_wcp_ulsif = wcp_ulsif.estimate_bounds(
        y_cal, predictions_cal, cohorts_cal, weights_ulsif, tau_grid, alpha=0.05
    )

    print("   Computing WCP-KLIEP bounds...")
    decisions_wcp_kliep = wcp_kliep.estimate_bounds(
        y_cal, predictions_cal, cohorts_cal, weights_kliep, tau_grid, alpha=0.05
    )

    print("   Computing uLSIF-EB bounds...")
    decisions_ulsif_eb = ulsif_eb.estimate_bounds(
        y_cal, predictions_cal, cohorts_cal, weights_ulsif, tau_grid, alpha=0.05
    )

    print("   Computing KLIEP-EB bounds...")
    decisions_kliep_eb = kliep_eb.estimate_bounds(
        y_cal, predictions_cal, cohorts_cal, weights_kliep, tau_grid, alpha=0.05
    )

    print("[OK] All bounds computed")

    # Step 7: Compare results
    print("\n[7/7] Comparing results...")
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON: Conformal vs Empirical-Bernstein Bounds")
    print("=" * 80)

    # Create comparison table
    results = []
    for i in range(len(decisions_wcp_ulsif)):
        d_wcp_u = decisions_wcp_ulsif[i]
        d_wcp_k = decisions_wcp_kliep[i]
        d_eb_u = decisions_ulsif_eb[i]
        d_eb_k = decisions_kliep_eb[i]

        results.append({
            "cohort": d_wcp_u.cohort_id,
            "tau": d_wcp_u.tau,
            "wcp_ulsif_lb": d_wcp_u.lower_bound,
            "wcp_kliep_lb": d_wcp_k.lower_bound,
            "eb_ulsif_lb": d_eb_u.lower_bound,
            "eb_kliep_lb": d_eb_k.lower_bound,
            "wcp_ulsif_cert": d_wcp_u.decision == "CERTIFY",
            "eb_ulsif_cert": d_eb_u.decision == "CERTIFY",
            "mu_hat": d_wcp_u.mu_hat,
            "n_eff": d_wcp_u.n_eff,
        })

    df_results = pd.DataFrame(results)

    # Print results by cohort
    for cohort in df_results["cohort"].unique():
        df_cohort = df_results[df_results["cohort"] == cohort]

        print(f"\n{'-' * 80}")
        print(f"Cohort: {cohort}")
        print(f"{'-' * 80}")
        print(f"{'Tau':<6} {'PPV Est':<9} {'n_eff':<8} {'WCP-uLSIF':<11} {'WCP-KLIEP':<11} "
              f"{'EB-uLSIF':<11} {'EB-KLIEP':<11}")
        print(f"{'':6} {'':9} {'':8} {'LB':>11} {'LB':>11} {'LB':>11} {'LB':>11}")
        print("-" * 80)

        for _, row in df_cohort.iterrows():
            cert_wcp = "*" if row["wcp_ulsif_cert"] else " "
            cert_eb = "*" if row["eb_ulsif_cert"] else " "
            print(
                f"{row['tau']:<6.2f} {row['mu_hat']:<9.3f} {row['n_eff']:<8.1f} "
                f"{row['wcp_ulsif_lb']:>9.4f}{cert_wcp:<2} "
                f"{row['wcp_kliep_lb']:>9.4f}  "
                f"{row['eb_ulsif_lb']:>9.4f}{cert_eb:<2} "
                f"{row['eb_kliep_lb']:>9.4f}  "
            )

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print("\nCertification rates (proportion of (cohort, tau) pairs certified):")
    print(f"  WCP-uLSIF:  {df_results['wcp_ulsif_cert'].mean():.1%}")
    print(f"  EB-uLSIF:   {df_results['eb_ulsif_cert'].mean():.1%}")

    print("\nMean lower bound by method:")
    print(f"  WCP-uLSIF:  {df_results['wcp_ulsif_lb'].mean():.4f}")
    print(f"  WCP-KLIEP:  {df_results['wcp_kliep_lb'].mean():.4f}")
    print(f"  EB-uLSIF:   {df_results['eb_ulsif_lb'].mean():.4f}")
    print(f"  EB-KLIEP:   {df_results['eb_kliep_lb'].mean():.4f}")

    print("\nDifference in lower bounds (WCP - EB):")
    df_results["diff_ulsif"] = df_results["wcp_ulsif_lb"] - df_results["eb_ulsif_lb"]
    df_results["diff_kliep"] = df_results["wcp_kliep_lb"] - df_results["eb_kliep_lb"]
    print(f"  uLSIF: mean={df_results['diff_ulsif'].mean():+.4f}, "
          f"std={df_results['diff_ulsif'].std():.4f}")
    print(f"  KLIEP: mean={df_results['diff_kliep'].mean():+.4f}, "
          f"std={df_results['diff_kliep'].std():.4f}")

    # Agreement analysis
    print("\nAgreement on certification decisions:")
    agreement_ulsif = (df_results["wcp_ulsif_cert"] == df_results["eb_ulsif_cert"]).mean()
    print(f"  uLSIF: {agreement_ulsif:.1%} agreement")

    # Disagreement cases
    disagreement = df_results[df_results["wcp_ulsif_cert"] != df_results["eb_ulsif_cert"]]
    if len(disagreement) > 0:
        print(f"\n  Disagreement cases ({len(disagreement)} total):")
        wcp_only = disagreement[disagreement["wcp_ulsif_cert"]]
        eb_only = disagreement[disagreement["eb_ulsif_cert"]]
        print(f"    WCP certified but EB did not: {len(wcp_only)}")
        print(f"    EB certified but WCP did not: {len(eb_only)}")

    print("\n" + "=" * 80)
    print("[SUCCESS] BACE evaluation complete!")
    print("=" * 80)
    print("\nKey insights:")
    print("1. Conformal uses quantiles (distribution-free)")
    print("2. EB uses mean + variance (assumes sub-Gaussian)")
    print("3. Different guarantees: marginal coverage vs concentration")
    print("4. Conformal may be more robust to heavy-tailed distributions")
    print("5. EB may be tighter when assumptions hold")
    print("=" * 80)


if __name__ == "__main__":
    test_weighted_conformal_on_bace()
