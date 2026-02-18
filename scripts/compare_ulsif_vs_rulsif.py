"""Comprehensive comparison of uLSIF vs RULSIF with different alpha values.

This script compares:
- uLSIF (standard density ratio, alpha=0.0)
- RULSIF with alpha=0.1 (default, slight stabilization)
- RULSIF with alpha=0.5 (maximum stability)
- RULSIF with alpha=0.9 (very high stability)

Analyzes:
1. Weight variance comparison (stability)
2. Agreement rates (certification decisions)
3. Certification rate comparison
4. When does RULSIF help vs. hurt?

Usage:
    python scripts/create_test_data.py  # Create test data first (if needed)
    python scripts/compare_ulsif_vs_rulsif.py [--dataset test_dataset|bace]
"""

import sys
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shiftbench.baselines.ulsif import create_ulsif_baseline
from shiftbench.baselines.rulsif import create_rulsif_baseline
from shiftbench.data import load_dataset


def compare_ulsif_vs_rulsif(dataset_name="test_dataset"):
    """Compare uLSIF vs RULSIF with different alpha values."""

    print("=" * 80)
    print(f" uLSIF vs RULSIF Comparison on {dataset_name.upper()}")
    print("=" * 80)

    # Load dataset
    print("\n[1/5] Loading dataset...")
    try:
        X, y, cohorts, splits = load_dataset(dataset_name)
        print(f"[OK] Loaded {len(X)} samples with {X.shape[1]} features")
        print(f"   Cohorts: {len(np.unique(cohorts))}")
        print(f"   Positive rate: {y.mean():.2%}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        if dataset_name == "test_dataset":
            print("   Run 'python scripts/create_test_data.py' first!")
        else:
            print(f"   Run 'python scripts/preprocess_molecular.py --dataset {dataset_name}' first!")
        return None

    # Split data
    cal_mask = (splits["split"] == "cal").values
    test_mask = (splits["split"] == "test").values
    X_cal, y_cal, cohorts_cal = X[cal_mask], y[cal_mask], cohorts[cal_mask]
    X_test = X[test_mask]

    print(f"   Calibration: {len(X_cal)} samples (positive rate: {y_cal.mean():.2%})")
    print(f"   Test: {len(X_test)} samples")

    # Initialize methods
    print("\n[2/5] Estimating weights with each method...")

    # Test multiple alpha values
    alphas = [0.0, 0.1, 0.5, 0.9]
    methods = {
        "uLSIF": create_ulsif_baseline(n_basis=100, sigma=None, lambda_=0.1, random_state=42),
    }
    for alpha in alphas:
        name = f"RULSIF(alpha={alpha})"
        methods[name] = create_rulsif_baseline(n_basis=100, sigma=None, lambda_=0.1, alpha=alpha, random_state=42)

    results = {}

    for name, method in methods.items():
        print(f"\n  {name}...")
        start = time.time()
        weights = method.estimate_weights(X_cal, X_test)
        elapsed = time.time() - start

        results[name] = {
            "weights": weights,
            "runtime": elapsed,
            "diagnostics": method.get_diagnostics(),
            "method": method,
        }

        # Weight statistics
        cv = weights.std() / weights.mean()
        print(f"    Runtime: {elapsed:.3f}s")
        print(f"    Weight mean: {weights.mean():.4f}")
        print(f"    Weight std: {weights.std():.4f}")
        print(f"    Weight CV: {cv:.4f}")
        print(f"    Weight range: [{weights.min():.4f}, {weights.max():.4f}]")

    # Analyze weight stability
    print("\n[3/5] Analyzing weight stability...")

    # Compute stability improvements vs uLSIF
    ulsif_cv = results["uLSIF"]["weights"].std() / results["uLSIF"]["weights"].mean()

    print("\n  Coefficient of Variation (CV = Std/Mean, lower is more stable):")
    print(f"  {'Method':<20} {'CV':<10} {'Improvement vs uLSIF':<25}")
    print("  " + "-" * 55)

    for name, res in results.items():
        w = res["weights"]
        cv = w.std() / w.mean()
        if name == "uLSIF":
            improvement = "baseline"
        else:
            improvement_pct = (ulsif_cv - cv) / ulsif_cv * 100
            improvement = f"{improvement_pct:+.2f}%"
        print(f"  {name:<20} {cv:<10.4f} {improvement:<25}")

    # Weight variance comparison
    print("\n  Weight Variance Reduction:")
    ulsif_var = results["uLSIF"]["weights"].var()
    for name, res in results.items():
        if name == "uLSIF":
            continue
        var = res["weights"].var()
        reduction_pct = (ulsif_var - var) / ulsif_var * 100
        print(f"    {name}: {reduction_pct:+.2f}% variance reduction")

    # Check if RULSIF(alpha=0.0) matches uLSIF
    rulsif_0_weights = results["RULSIF(alpha=0.0)"]["weights"]
    ulsif_weights = results["uLSIF"]["weights"]
    diff = np.abs(rulsif_0_weights - ulsif_weights).mean()
    max_diff = np.abs(rulsif_0_weights - ulsif_weights).max()
    print(f"\n  Verification: RULSIF(alpha=0.0) vs uLSIF")
    print(f"    Mean absolute difference: {diff:.6f}")
    print(f"    Max absolute difference: {max_diff:.6f}")
    if diff < 0.01:
        print(f"    [OK] RULSIF(alpha=0.0) matches uLSIF (as expected)")
    else:
        print(f"    [WARNING] RULSIF(alpha=0.0) differs from uLSIF (unexpected!)")

    # Estimate bounds for all methods
    print("\n[4/5] Estimating PPV bounds for each method...")

    tau_grid = [0.5, 0.6, 0.7, 0.8, 0.9]
    predictions_cal = np.ones(len(y_cal), dtype=int)  # Naive baseline: predict all positive

    decisions_all = {}

    for name, res in results.items():
        print(f"\n  {name}...")
        method = res["method"]
        weights = res["weights"]

        start = time.time()
        decisions = method.estimate_bounds(
            y_cal, predictions_cal, cohorts_cal, weights, tau_grid, alpha=0.05
        )
        elapsed = time.time() - start

        decisions_all[name] = decisions
        res["bound_time"] = elapsed

        n_certify = sum(1 for d in decisions if d.decision == "CERTIFY")
        n_abstain = sum(1 for d in decisions if d.decision == "ABSTAIN")
        n_no_guarantee = sum(1 for d in decisions if d.decision == "NO-GUARANTEE")

        print(f"    Bound time: {elapsed:.3f}s")
        print(f"    CERTIFY: {n_certify}/{len(decisions)} ({n_certify/len(decisions):.1%})")
        print(f"    ABSTAIN: {n_abstain}/{len(decisions)} ({n_abstain/len(decisions):.1%})")
        print(f"    NO-GUARANTEE: {n_no_guarantee}/{len(decisions)} ({n_no_guarantee/len(decisions):.1%})")

    # Compare certification decisions
    print("\n[5/5] Comparing certification decisions...")

    # Convert to DataFrames for easier comparison
    dfs = {}
    for name, decisions in decisions_all.items():
        df = pd.DataFrame([
            {
                "cohort_id": d.cohort_id,
                "tau": d.tau,
                "decision": d.decision,
                "mu_hat": d.mu_hat,
                "lower_bound": d.lower_bound,
                "n_eff": d.n_eff,
                "p_value": d.p_value,
            }
            for d in decisions
        ])
        dfs[name] = df

    # Agreement analysis
    print("\n  Decision Agreement with uLSIF:")
    ulsif_df = dfs["uLSIF"]

    for name, df in dfs.items():
        if name == "uLSIF":
            continue

        # Merge on cohort_id and tau
        merged = pd.merge(
            ulsif_df, df,
            on=["cohort_id", "tau"],
            suffixes=("_ulsif", f"_{name}")
        )

        total_pairs = len(merged)
        agree = (merged["decision_ulsif"] == merged[f"decision_{name}"]).sum()
        agreement_rate = agree / total_pairs

        print(f"    {name}: {agree}/{total_pairs} ({agreement_rate:.1%}) agreement")

        # Analyze disagreements
        disagree = merged[merged["decision_ulsif"] != merged[f"decision_{name}"]]
        if len(disagree) > 0:
            # Count types of disagreements
            ulsif_cert = (disagree["decision_ulsif"] == "CERTIFY").sum()
            rulsif_cert = (disagree[f"decision_{name}"] == "CERTIFY").sum()
            print(f"      Disagreements ({len(disagree)}): uLSIF certifies {ulsif_cert}, {name} certifies {rulsif_cert}")

    # Bound comparison
    print("\n  Lower Bound Comparison:")
    print(f"  {'Method':<20} {'Mean LB':<10} {'Std LB':<10} {'Mean n_eff':<12}")
    print("  " + "-" * 52)

    for name, df in dfs.items():
        valid = df[df["n_eff"] > 0]
        if len(valid) > 0:
            mean_lb = valid["lower_bound"].mean()
            std_lb = valid["lower_bound"].std()
            mean_neff = valid["n_eff"].mean()
            print(f"  {name:<20} {mean_lb:<10.4f} {std_lb:<10.4f} {mean_neff:<12.2f}")

    # Summary table
    print("\n" + "=" * 80)
    print(" SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\n{'Method':<20} {'Weight CV':<12} {'Runtime (s)':<12} {'Certify Rate':<15}")
    print("-" * 59)

    for name, res in results.items():
        w = res["weights"]
        cv = w.std() / w.mean()
        runtime = res["runtime"] + res["bound_time"]
        decisions = decisions_all[name]
        cert_rate = sum(1 for d in decisions if d.decision == "CERTIFY") / len(decisions)
        print(f"{name:<20} {cv:<12.4f} {runtime:<12.3f} {cert_rate:<15.1%}")

    # Key findings
    print("\n" + "=" * 80)
    print(" KEY FINDINGS")
    print("=" * 80)

    print("\n1. Weight Stability:")
    rulsif_01_cv = results["RULSIF(alpha=0.1)"]["weights"].std() / results["RULSIF(alpha=0.1)"]["weights"].mean()
    rulsif_05_cv = results["RULSIF(alpha=0.5)"]["weights"].std() / results["RULSIF(alpha=0.5)"]["weights"].mean()
    improvement_01 = (ulsif_cv - rulsif_01_cv) / ulsif_cv * 100
    improvement_05 = (ulsif_cv - rulsif_05_cv) / ulsif_cv * 100

    print(f"   - RULSIF(alpha=0.1) provides {improvement_01:+.2f}% stability improvement over uLSIF")
    print(f"   - RULSIF(alpha=0.5) provides {improvement_05:+.2f}% stability improvement over uLSIF")
    print(f"   - Higher alpha -> more stable weights (lower CV)")

    print("\n2. Certification Performance:")
    ulsif_cert = sum(1 for d in decisions_all["uLSIF"] if d.decision == "CERTIFY")
    rulsif_01_cert = sum(1 for d in decisions_all["RULSIF(alpha=0.1)"] if d.decision == "CERTIFY")
    rulsif_05_cert = sum(1 for d in decisions_all["RULSIF(alpha=0.5)"] if d.decision == "CERTIFY")

    print(f"   - uLSIF: {ulsif_cert} certifications")
    print(f"   - RULSIF(alpha=0.1): {rulsif_01_cert} certifications")
    print(f"   - RULSIF(alpha=0.5): {rulsif_05_cert} certifications")

    # Determine when RULSIF helps
    if improvement_05 > 5:
        print("\n3. When to Use RULSIF:")
        print("   [RECOMMENDED] Use RULSIF when:")
        print("      - Distribution shift is severe")
        print("      - uLSIF produces high weight variance")
        print("      - You need more stable importance weights")
        print("   [TRADE-OFF]")
        print("      - More stable weights may lead to different certification decisions")
        print("      - alpha=0.1 provides good balance between stability and accuracy")
        print("      - alpha=0.5 provides maximum stability but may be too conservative")
    else:
        print("\n3. Recommendation:")
        print("   - On this dataset, shift is moderate")
        print("   - uLSIF and RULSIF perform similarly")
        print("   - Use uLSIF for simplicity (fewer hyperparameters)")
        print("   - Consider RULSIF for datasets with more severe shifts")

    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save comparison summary
    summary_data = []
    for name, res in results.items():
        w = res["weights"]
        decisions = decisions_all[name]
        summary_data.append({
            "method": name,
            "weight_mean": w.mean(),
            "weight_std": w.std(),
            "weight_cv": w.std() / w.mean(),
            "weight_min": w.min(),
            "weight_max": w.max(),
            "weight_time": res["runtime"],
            "bound_time": res["bound_time"],
            "total_time": res["runtime"] + res["bound_time"],
            "n_certify": sum(1 for d in decisions if d.decision == "CERTIFY"),
            "n_abstain": sum(1 for d in decisions if d.decision == "ABSTAIN"),
            "n_no_guarantee": sum(1 for d in decisions if d.decision == "NO-GUARANTEE"),
        })

    summary_df = pd.DataFrame(summary_data)
    summary_file = results_dir / f"ulsif_vs_rulsif_{dataset_name}_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\n[OK] Summary saved to {summary_file}")

    # Save detailed decisions for each method
    for name, df in dfs.items():
        safe_name = name.replace("α=", "alpha").replace("(", "").replace(")", "").replace(".", "")
        decisions_file = results_dir / f"{safe_name.lower()}_{dataset_name}_results.csv"
        df.to_csv(decisions_file, index=False)
        print(f"[OK] Decisions saved to {decisions_file}")

    # Create visualization
    try:
        print("\n[Optional] Creating weight distribution plots...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        plot_methods = ["uLSIF", "RULSIF(alpha=0.1)", "RULSIF(alpha=0.5)", "RULSIF(alpha=0.9)"]

        for ax, name in zip(axes, plot_methods):
            weights = results[name]["weights"]
            cv = weights.std() / weights.mean()

            ax.hist(weights, bins=40, alpha=0.7, edgecolor="black", color="steelblue")
            ax.axvline(weights.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
            ax.set_xlabel("Weight", fontsize=11)
            ax.set_ylabel("Frequency", fontsize=11)
            ax.set_title(f"{name}\nCV={cv:.4f}, Mean={weights.mean():.3f}, Range=[{weights.min():.3f}, {weights.max():.3f}]", fontsize=10)
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plot_file = results_dir / f"ulsif_vs_rulsif_{dataset_name}_weights.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"[OK] Weight distribution plot saved to {plot_file}")
        plt.close()

        # Create CV comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))

        alphas_vals = [0.0, 0.1, 0.5, 0.9]
        cvs = []
        for alpha in alphas_vals:
            name = f"RULSIF(alpha={alpha})"
            cvs.append(results[name]["weights"].std() / results[name]["weights"].mean())

        ax.plot(alphas_vals, cvs, marker='o', linewidth=2, markersize=8, color='steelblue')
        ax.axhline(ulsif_cv, color='red', linestyle='--', linewidth=2, label='uLSIF')
        ax.set_xlabel("Alpha (a)", fontsize=12)
        ax.set_ylabel("Coefficient of Variation (CV)", fontsize=12)
        ax.set_title("Weight Stability vs Alpha Parameter", fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=11)

        plt.tight_layout()
        cv_plot_file = results_dir / f"ulsif_vs_rulsif_{dataset_name}_cv_vs_alpha.png"
        plt.savefig(cv_plot_file, dpi=150, bbox_inches='tight')
        print(f"[OK] CV vs Alpha plot saved to {cv_plot_file}")
        plt.close()

    except Exception as e:
        print(f"[WARNING] Could not create plots: {e}")

    print("\n" + "=" * 80)
    print(" COMPARISON COMPLETE")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare uLSIF vs RULSIF")
    parser.add_argument(
        "--dataset",
        type=str,
        default="test_dataset",
        help="Dataset to use (default: test_dataset)",
    )
    args = parser.parse_args()

    compare_ulsif_vs_rulsif(args.dataset)


if __name__ == "__main__":
    main()
