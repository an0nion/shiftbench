"""Compare Weighted Conformal Prediction vs Empirical-Bernstein bounds.

This script provides a detailed comparison of two approaches to PPV bounds:
1. Weighted Conformal: Quantile-based, distribution-free
2. Empirical-Bernstein: Mean+variance, assumes sub-Gaussian

Key questions:
- Which is tighter?
- Which certifies more?
- When do they disagree?
- How do bounds differ across cohort sizes?

Usage:
    python scripts/compare_conformal_vs_eb.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from shiftbench.baselines.weighted_conformal import create_weighted_conformal_baseline
from shiftbench.baselines.ulsif import create_ulsif_baseline
from shiftbench.data import load_dataset


def compare_on_dataset(dataset_name: str, use_pca_model: bool = True):
    """Run comparison on a single dataset."""

    print("=" * 80)
    print(f"Comparing Weighted Conformal vs Empirical-Bernstein on {dataset_name.upper()}")
    print("=" * 80)

    # Load dataset
    print(f"\n[1/5] Loading {dataset_name} dataset...")
    try:
        X, y, cohorts, splits = load_dataset(dataset_name)
        print(f"[OK] Loaded {len(X)} samples with {X.shape[1]} features")
        print(f"   Unique cohorts: {len(np.unique(cohorts))}")
        print(f"   Positive rate: {y.mean():.2%}")
    except Exception as e:
        print(f"[ERROR] Failed to load {dataset_name}: {e}")
        return None

    # Split data
    print(f"\n[2/5] Splitting into calibration and test sets...")
    cal_mask = (splits["split"] == "cal").values
    test_mask = (splits["split"] == "test").values

    X_cal, y_cal, cohorts_cal = X[cal_mask], y[cal_mask], cohorts[cal_mask]
    X_test = X[test_mask]

    print(f"[OK] Calibration: {len(X_cal)} samples")
    print(f"   Test: {len(X_test)} samples")

    # Generate predictions
    print(f"\n[3/5] Generating predictions...")
    if use_pca_model:
        # PCA-based model
        pca = PCA(n_components=1, random_state=42)
        X_cal_pc = pca.fit_transform(X_cal).ravel()
        threshold = np.median(X_cal_pc)
        predictions_cal = (X_cal_pc > threshold).astype(int)
    else:
        # Naive: predict all positive
        predictions_cal = np.ones(len(y_cal), dtype=int)

    n_pred_pos = predictions_cal.sum()
    ppv_empirical = y_cal[predictions_cal == 1].mean()
    print(f"[OK] Predicted {n_pred_pos} positives ({n_pred_pos/len(predictions_cal):.1%})")
    print(f"   Empirical PPV: {ppv_empirical:.3f}")

    # Initialize methods
    print(f"\n[4/5] Initializing methods and estimating weights...")
    wcp = create_weighted_conformal_baseline(
        weight_method="ulsif",
        n_basis=100,
        random_state=42
    )
    eb = create_ulsif_baseline(
        n_basis=100,
        random_state=42
    )

    # Estimate weights (both use same weights)
    weights = wcp.estimate_weights(X_cal, X_test)
    print(f"[OK] Estimated weights: mean={weights.mean():.3f}, "
          f"std={weights.std():.3f}, max={weights.max():.3f}")

    # Compute bounds
    print(f"\n[5/5] Computing bounds...")
    tau_grid = [0.5, 0.6, 0.7, 0.8, 0.9]

    print("   Computing Weighted Conformal bounds...")
    decisions_wcp = wcp.estimate_bounds(
        y_cal, predictions_cal, cohorts_cal, weights, tau_grid, alpha=0.05
    )

    print("   Computing Empirical-Bernstein bounds...")
    decisions_eb = eb.estimate_bounds(
        y_cal, predictions_cal, cohorts_cal, weights, tau_grid, alpha=0.05
    )

    print("[OK] All bounds computed")

    # Analyze results
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Create DataFrame for comparison
    results = []
    for d_wcp, d_eb in zip(decisions_wcp, decisions_eb):
        results.append({
            "cohort": d_wcp.cohort_id,
            "tau": d_wcp.tau,
            "wcp_lb": d_wcp.lower_bound,
            "eb_lb": d_eb.lower_bound,
            "wcp_cert": d_wcp.decision == "CERTIFY",
            "eb_cert": d_eb.decision == "CERTIFY",
            "mu_hat": d_wcp.mu_hat,
            "n_eff": d_wcp.n_eff,
        })

    df = pd.DataFrame(results)

    # Remove rows with invalid bounds
    valid_mask = (
        np.isfinite(df["wcp_lb"]) &
        np.isfinite(df["eb_lb"]) &
        (df["n_eff"] > 0)
    )
    df_valid = df[valid_mask].copy()

    print(f"\nValid comparisons: {len(df_valid)} / {len(df)}")

    if len(df_valid) == 0:
        print("[WARNING] No valid comparisons possible (all cohorts too small)")
        return None

    # Compute statistics
    print("\n" + "-" * 80)
    print("OVERALL STATISTICS")
    print("-" * 80)

    print(f"\nCertification Rates:")
    print(f"  Weighted Conformal: {df_valid['wcp_cert'].mean():.1%} "
          f"({df_valid['wcp_cert'].sum()} / {len(df_valid)})")
    print(f"  Empirical-Bernstein: {df_valid['eb_cert'].mean():.1%} "
          f"({df_valid['eb_cert'].sum()} / {len(df_valid)})")

    if df_valid['eb_cert'].sum() > 0:
        ratio = df_valid['wcp_cert'].sum() / df_valid['eb_cert'].sum()
        print(f"  Ratio (WCP/EB): {ratio:.1f}x")

    print(f"\nMean Lower Bounds:")
    print(f"  Weighted Conformal: {df_valid['wcp_lb'].mean():.4f}")
    print(f"  Empirical-Bernstein: {df_valid['eb_lb'].mean():.4f}")
    print(f"  Difference (WCP - EB): {(df_valid['wcp_lb'] - df_valid['eb_lb']).mean():+.4f}")

    print(f"\nLower Bound Comparison:")
    df_valid["diff"] = df_valid["wcp_lb"] - df_valid["eb_lb"]
    print(f"  Mean: {df_valid['diff'].mean():+.4f}")
    print(f"  Std:  {df_valid['diff'].std():.4f}")
    print(f"  Min:  {df_valid['diff'].min():+.4f}")
    print(f"  Max:  {df_valid['diff'].max():+.4f}")
    print(f"  WCP tighter: {(df_valid['diff'] > 0).sum()} / {len(df_valid)} "
          f"({(df_valid['diff'] > 0).mean():.1%})")

    # Decision agreement
    print(f"\nDecision Agreement:")
    agreement = (df_valid["wcp_cert"] == df_valid["eb_cert"]).mean()
    print(f"  Agreement: {agreement:.1%}")

    # Disagreement analysis
    disagree = df_valid[df_valid["wcp_cert"] != df_valid["eb_cert"]]
    if len(disagree) > 0:
        wcp_only = disagree[disagree["wcp_cert"]]
        eb_only = disagree[disagree["eb_cert"]]
        print(f"  Disagreements: {len(disagree)}")
        print(f"    WCP certifies but EB does not: {len(wcp_only)}")
        print(f"    EB certifies but WCP does not: {len(eb_only)}")
    else:
        print("  No disagreements - perfect agreement!")

    # Cohort size analysis
    print("\n" + "-" * 80)
    print("ANALYSIS BY EFFECTIVE SAMPLE SIZE")
    print("-" * 80)

    # Bin by n_eff
    df_valid["n_eff_bin"] = pd.cut(
        df_valid["n_eff"],
        bins=[0, 10, 20, 50, 100, np.inf],
        labels=["<10", "10-20", "20-50", "50-100", "100+"]
    )

    for bin_name in ["<10", "10-20", "20-50", "50-100", "100+"]:
        df_bin = df_valid[df_valid["n_eff_bin"] == bin_name]
        if len(df_bin) == 0:
            continue

        print(f"\nn_eff {bin_name}: {len(df_bin)} comparisons")
        print(f"  WCP cert rate: {df_bin['wcp_cert'].mean():.1%}")
        print(f"  EB cert rate:  {df_bin['eb_cert'].mean():.1%}")
        print(f"  WCP mean LB:   {df_bin['wcp_lb'].mean():.4f}")
        print(f"  EB mean LB:    {df_bin['eb_lb'].mean():.4f}")
        print(f"  Difference:    {(df_bin['wcp_lb'] - df_bin['eb_lb']).mean():+.4f}")

    # Per-threshold analysis
    print("\n" + "-" * 80)
    print("ANALYSIS BY THRESHOLD (TAU)")
    print("-" * 80)

    for tau_val in sorted(df_valid["tau"].unique()):
        df_tau = df_valid[df_valid["tau"] == tau_val]
        print(f"\nTau = {tau_val:.1f}: {len(df_tau)} comparisons")
        print(f"  WCP cert rate: {df_tau['wcp_cert'].mean():.1%} "
              f"({df_tau['wcp_cert'].sum()})")
        print(f"  EB cert rate:  {df_tau['eb_cert'].mean():.1%} "
              f"({df_tau['eb_cert'].sum()})")

    # Show example disagreements
    if len(disagree) > 0:
        print("\n" + "-" * 80)
        print("EXAMPLE DISAGREEMENTS (first 10)")
        print("-" * 80)
        print(f"{'Cohort':<20} {'Tau':<5} {'n_eff':<8} {'PPV':<8} "
              f"{'WCP LB':<8} {'EB LB':<8} {'WCP':<8} {'EB':<8}")
        print("-" * 80)

        for _, row in disagree.head(10).iterrows():
            wcp_dec = "CERT" if row["wcp_cert"] else "ABS"
            eb_dec = "CERT" if row["eb_cert"] else "ABS"
            cohort_short = row["cohort"][:18] if len(row["cohort"]) > 18 else row["cohort"]
            print(
                f"{cohort_short:<20} {row['tau']:<5.2f} {row['n_eff']:<8.1f} "
                f"{row['mu_hat']:<8.3f} {row['wcp_lb']:<8.4f} {row['eb_lb']:<8.4f} "
                f"{wcp_dec:<8} {eb_dec:<8}"
            )

    return df_valid


def main():
    """Run comparison on multiple datasets."""

    print("=" * 80)
    print("WEIGHTED CONFORMAL vs EMPIRICAL-BERNSTEIN COMPARISON")
    print("=" * 80)
    print("\nThis script compares two bound types:")
    print("  1. Weighted Conformal: Quantile-based, distribution-free")
    print("  2. Empirical-Bernstein: Mean+variance, assumes sub-Gaussian")
    print("")
    print("Key questions:")
    print("  - Which provides tighter bounds?")
    print("  - Which certifies more cohorts?")
    print("  - How do they differ across cohort sizes?")
    print("=" * 80)

    # Test on available datasets
    datasets_to_test = ["test_dataset", "bace"]
    all_results = {}

    for dataset_name in datasets_to_test:
        print("\n\n")
        try:
            result = compare_on_dataset(dataset_name, use_pca_model=True)
            if result is not None:
                all_results[dataset_name] = result
        except Exception as e:
            print(f"[ERROR] Failed on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    # Overall summary
    print("\n\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    if len(all_results) == 0:
        print("[ERROR] No successful comparisons")
        return

    for dataset_name, df in all_results.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"  Comparisons: {len(df)}")
        print(f"  WCP cert rate: {df['wcp_cert'].mean():.1%}")
        print(f"  EB cert rate:  {df['eb_cert'].mean():.1%}")
        print(f"  WCP mean LB:   {df['wcp_lb'].mean():.4f}")
        print(f"  EB mean LB:    {df['eb_lb'].mean():.4f}")
        print(f"  Difference:    {(df['wcp_lb'] - df['eb_lb']).mean():+.4f}")

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print("""
1. COVERAGE TYPE:
   - Weighted Conformal: Marginal coverage (distribution-free)
   - Empirical-Bernstein: Concentration inequality (sub-Gaussian)

2. SMALL SAMPLE BEHAVIOR:
   - WCP: Uses quantiles, less conservative with small n
   - EB: Very conservative with small n (includes variance penalty)

3. TIGHTNESS:
   - WCP: Often provides higher lower bounds on sparse data
   - EB: Can be tighter when n is large and assumptions hold

4. CERTIFICATIONS:
   - WCP: Typically certifies more on small-sample problems
   - EB: May certify more on large-sample, homogeneous problems

5. ROBUSTNESS:
   - WCP: Distribution-free, robust to heavy tails
   - EB: Assumes sub-Gaussian, sensitive to outliers

RECOMMENDATION:
- Use WCP for small cohorts (n < 20) or unknown distributions
- Use EB for large cohorts (n > 50) with mild distributions
- Compare both and take intersection for highest confidence
    """)
    print("=" * 80)
    print("[SUCCESS] Comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
