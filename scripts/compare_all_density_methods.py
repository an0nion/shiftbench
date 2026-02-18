"""Comprehensive comparison of all density ratio methods: RAVEL, uLSIF, KLIEP, KMM.

This script provides a complete analysis comparing all four density ratio estimation methods
available in ShiftBench:

1. RAVEL: Cross-fitted discriminative + stability gating
2. uLSIF: Unconstrained Least-Squares Importance Fitting (closed-form)
3. KLIEP: Kullback-Leibler Importance Estimation (KL optimization)
4. KMM: Kernel Mean Matching (MMD minimization, QP optimization)

Comparisons include:
- Weight distribution statistics
- Weight correlation matrix (agreement analysis)
- Certification rate comparison
- Runtime comparison
- Method-specific diagnostics
- Agreement heatmaps and visualizations

Usage:
    python scripts/compare_all_density_methods.py [--dataset test_dataset|bace]
    python scripts/compare_all_density_methods.py --dataset bace --save-plots
"""

import sys
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from shiftbench.baselines.kmm import create_kmm_baseline
from shiftbench.baselines.ulsif import create_ulsif_baseline
from shiftbench.baselines.kliep import create_kliep_baseline

# Try to load RAVEL if available
try:
    from shiftbench.baselines.ravel import create_ravel_baseline
    RAVEL_AVAILABLE = True
except ImportError:
    RAVEL_AVAILABLE = False
    print("[WARNING] RAVEL not available. Will only compare KMM vs uLSIF vs KLIEP.")

from shiftbench.data import load_dataset


def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 100)
    print(f" {title}")
    print("=" * 100)


def print_subheader(title):
    """Print formatted subsection header."""
    print(f"\n{title}")
    print("-" * 100)


def analyze_weights(weights, method_name):
    """Compute weight distribution statistics."""
    return {
        "method": method_name,
        "mean": weights.mean(),
        "std": weights.std(),
        "min": weights.min(),
        "max": weights.max(),
        "median": np.median(weights),
        "q25": np.percentile(weights, 25),
        "q75": np.percentile(weights, 75),
        "q95": np.percentile(weights, 95),
        "q99": np.percentile(weights, 99),
        "coef_var": weights.std() / weights.mean(),
        "ess": (weights.sum() ** 2) / (weights ** 2).sum(),  # Effective sample size
        "ess_ratio": (weights.sum() ** 2) / (weights ** 2).sum() / len(weights),
    }


def compute_agreement_matrix(decisions_dict, tau_grid):
    """Compute pairwise agreement matrix for certification decisions.

    Agreement is defined as the fraction of (cohort, tau) pairs where
    both methods make the same decision (CERTIFY vs not CERTIFY).
    """
    methods = list(decisions_dict.keys())
    n_methods = len(methods)

    agreement_matrix = np.zeros((n_methods, n_methods))

    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i == j:
                agreement_matrix[i, j] = 1.0
                continue

            # Extract decisions for both methods
            decisions1 = decisions_dict[method1]
            decisions2 = decisions_dict[method2]

            # Create (cohort_id, tau) -> decision mapping
            dec1_map = {(d.cohort_id, d.tau): d.decision for d in decisions1}
            dec2_map = {(d.cohort_id, d.tau): d.decision for d in decisions2}

            # Compute agreement
            n_agree = 0
            n_total = 0
            for key in dec1_map:
                if key in dec2_map:
                    n_total += 1
                    # Both certify or both don't certify
                    cert1 = dec1_map[key] == "CERTIFY"
                    cert2 = dec2_map[key] == "CERTIFY"
                    if cert1 == cert2:
                        n_agree += 1

            agreement_matrix[i, j] = n_agree / n_total if n_total > 0 else 0.0

    return agreement_matrix, methods


def compare_methods(dataset_name="test_dataset", save_plots=False):
    """Compare all density ratio estimation methods on a dataset."""

    print_header(f"Comprehensive Density Ratio Method Comparison on {dataset_name.upper()}")

    # Load dataset
    print("\n[1/6] Loading dataset...")
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
            print(f"   Run preprocessing script for {dataset_name} first!")
        return None

    # Split data
    print("\n[2/6] Splitting into calibration and test sets...")
    cal_mask = (splits["split"] == "cal").values
    test_mask = (splits["split"] == "test").values

    X_cal, y_cal, cohorts_cal = X[cal_mask], y[cal_mask], cohorts[cal_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"[OK] Calibration: {len(X_cal)} samples ({y_cal.mean():.2%} positive)")
    print(f"   Test: {len(X_test)} samples ({y_test.mean():.2%} positive)")

    # Initialize all methods
    print("\n[3/6] Initializing all methods...")
    methods = {}

    # Determine n_basis based on dataset size
    n_basis = 100 if len(X_cal) > 200 else 50

    # KMM: MMD minimization via QP
    methods['KMM'] = create_kmm_baseline(
        sigma=None,  # Median heuristic
        lambda_=0.1,
        B=1000.0,
        random_state=42,
        solver="auto",
    )

    # uLSIF: L2 loss minimization (closed-form)
    methods['uLSIF'] = create_ulsif_baseline(
        n_basis=n_basis,
        sigma=None,
        lambda_=0.1,
        random_state=42,
    )

    # KLIEP: KL divergence minimization
    methods['KLIEP'] = create_kliep_baseline(
        n_basis=n_basis,
        sigma=None,
        max_iter=10000,
        random_state=42,
    )

    # RAVEL: Cross-fitted discriminative (if available)
    if RAVEL_AVAILABLE:
        methods['RAVEL'] = create_ravel_baseline(
            n_folds=5,
            random_state=42,
        )

    print(f"[OK] Initialized {len(methods)} methods: {', '.join(methods.keys())}")

    # Estimate weights
    print("\n[4/6] Estimating importance weights...")
    weights_dict = {}
    runtimes = {}
    diagnostics_dict = {}

    for method_name, method in methods.items():
        print(f"\n  {method_name}...")
        start_time = time.time()
        try:
            weights = method.estimate_weights(X_cal, X_test)
            elapsed = time.time() - start_time

            # Validate
            assert np.all(weights > 0), "All weights must be positive"
            assert np.all(np.isfinite(weights)), "All weights must be finite"
            assert np.abs(weights.mean() - 1.0) < 0.2, f"Mean should be ~1.0, got {weights.mean():.4f}"

            weights_dict[method_name] = weights
            runtimes[method_name] = elapsed
            diagnostics_dict[method_name] = method.get_diagnostics()

            print(f"  [OK] {method_name}: {elapsed:.3f}s")
            print(f"       Mean: {weights.mean():.4f}, Std: {weights.std():.4f}")
            print(f"       Range: [{weights.min():.4f}, {weights.max():.4f}]")

        except Exception as e:
            print(f"  [ERROR] {method_name} failed: {e}")
            if method_name == "RAVEL":
                print("  (This may be expected if shift is too severe)")

    if not weights_dict:
        print("[ERROR] No methods succeeded!")
        return None

    # Analyze weight distributions
    print_subheader("Weight Distribution Statistics")
    weight_stats = []
    for method_name, weights in weights_dict.items():
        stats = analyze_weights(weights, method_name)
        weight_stats.append(stats)

    weight_stats_df = pd.DataFrame(weight_stats)
    print(weight_stats_df.to_string(index=False))

    # Weight correlation analysis
    print_subheader("Weight Correlation Matrix")
    method_names = list(weights_dict.keys())
    n_methods = len(method_names)

    corr_matrix = np.zeros((n_methods, n_methods))
    for i, method1 in enumerate(method_names):
        for j, method2 in enumerate(method_names):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                w1 = weights_dict[method1]
                w2 = weights_dict[method2]
                corr_matrix[i, j] = np.corrcoef(w1, w2)[0, 1]

    # Print correlation matrix
    print(f"\n{'':15s}", end="")
    for name in method_names:
        print(f"{name:>12s}", end="")
    print()
    print("-" * (15 + 12 * n_methods))
    for i, method1 in enumerate(method_names):
        print(f"{method1:15s}", end="")
        for j in range(n_methods):
            print(f"{corr_matrix[i, j]:12.4f}", end="")
        print()

    # Generate predictions (oracle mode)
    print("\n[5/6] Generating predictions and estimating bounds...")
    predictions_cal = y_cal.astype(int)
    print(f"[OK] Using oracle predictions ({predictions_cal.sum()} positives)")

    # Estimate bounds
    tau_grid = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    alpha = 0.05

    decisions_dict = {}
    bound_runtimes = {}

    for method_name, method in methods.items():
        if method_name not in weights_dict:
            continue

        print(f"\n  {method_name}...")
        start_time = time.time()
        try:
            decisions = method.estimate_bounds(
                y_cal,
                predictions_cal,
                cohorts_cal,
                weights_dict[method_name],
                tau_grid,
                alpha=alpha,
            )
            elapsed = time.time() - start_time

            decisions_dict[method_name] = decisions
            bound_runtimes[method_name] = elapsed

            n_certify = sum(1 for d in decisions if d.decision == "CERTIFY")
            n_abstain = sum(1 for d in decisions if d.decision == "ABSTAIN")
            n_no_guarantee = sum(1 for d in decisions if d.decision == "NO-GUARANTEE")

            print(f"  [OK] {method_name}: {elapsed:.3f}s")
            print(f"       CERTIFY: {n_certify}/{len(decisions)} ({n_certify/len(decisions):.1%})")
            print(f"       ABSTAIN: {n_abstain}/{len(decisions)} ({n_abstain/len(decisions):.1%})")
            if n_no_guarantee > 0:
                print(f"       NO-GUARANTEE: {n_no_guarantee}/{len(decisions)} ({n_no_guarantee/len(decisions):.1%})")

        except Exception as e:
            print(f"  [ERROR] {method_name} bounds failed: {e}")

    # Compare certification rates
    print_subheader("Certification Rate Comparison")

    cert_comparison = []
    for tau in tau_grid:
        row = {"tau": tau}
        for method_name, decisions in decisions_dict.items():
            tau_decisions = [d for d in decisions if d.tau == tau]
            n_certify = sum(1 for d in tau_decisions if d.decision == "CERTIFY")
            n_total = len(tau_decisions)
            cert_rate = n_certify / n_total if n_total > 0 else 0
            row[f"{method_name}_certify"] = n_certify
            row[f"{method_name}_total"] = n_total
            row[f"{method_name}_rate"] = cert_rate
        cert_comparison.append(row)

    cert_df = pd.DataFrame(cert_comparison)
    print("\nCertification counts by tau:")
    for tau in tau_grid:
        print(f"\nTau = {tau:.2f}:")
        print(f"  {'Method':<15} {'Certify':<10} {'Total':<10} {'Rate':<10}")
        print("  " + "-" * 45)
        for method_name in decisions_dict.keys():
            row = cert_df[cert_df['tau'] == tau].iloc[0]
            n_cert = row[f"{method_name}_certify"]
            n_total = row[f"{method_name}_total"]
            rate = row[f"{method_name}_rate"]
            print(f"  {method_name:<15} {n_cert:<10.0f} {n_total:<10.0f} {rate:<10.1%}")

    # Agreement matrix
    print_subheader("Decision Agreement Matrix")
    agreement_matrix, method_order = compute_agreement_matrix(decisions_dict, tau_grid)

    print("\nPairwise agreement on certification decisions:")
    print(f"{'':15s}", end="")
    for name in method_order:
        print(f"{name:>12s}", end="")
    print()
    print("-" * (15 + 12 * len(method_order)))
    for i, method1 in enumerate(method_order):
        print(f"{method1:15s}", end="")
        for j in range(len(method_order)):
            print(f"{agreement_matrix[i, j]:12.1%}", end="")
        print()

    # Runtime comparison
    print_subheader("Runtime Comparison")
    runtime_df = pd.DataFrame([
        {
            "method": method_name,
            "weight_time_sec": runtimes.get(method_name, np.nan),
            "bound_time_sec": bound_runtimes.get(method_name, np.nan),
            "total_time_sec": runtimes.get(method_name, 0) + bound_runtimes.get(method_name, 0),
        }
        for method_name in weights_dict.keys()
    ])
    print(runtime_df.to_string(index=False))

    # Method-specific diagnostics
    print_subheader("Method-Specific Diagnostics")
    for method_name, diag in diagnostics_dict.items():
        print(f"\n{method_name}:")
        if method_name == "KMM":
            print(f"  Sigma (bandwidth): {diag.get('sigma', 'N/A'):.4f}")
            print(f"  Lambda (ridge): {diag.get('lambda', 'N/A'):.4f}")
            print(f"  B (box constraint): {diag.get('B', 'N/A'):.1f}")
            print(f"  Weights clipped: {diag.get('weights_clipped_fraction', 0):.1%}")
            print(f"  Optimization success: {diag.get('optimization_success', 'N/A')}")
            print(f"  Solve time: {diag.get('solve_time', 'N/A'):.3f}s")
        elif method_name == "uLSIF":
            print(f"  Sigma (bandwidth): {diag.get('sigma', 'N/A'):.4f}")
            print(f"  N basis: {diag.get('n_basis', 'N/A')}")
            print(f"  Alpha min/max: {diag.get('alpha_min', 'N/A'):.4f} / {diag.get('alpha_max', 'N/A'):.4f}")
        elif method_name == "KLIEP":
            print(f"  Sigma (bandwidth): {diag.get('sigma', 'N/A'):.4f}")
            print(f"  N basis: {diag.get('n_basis', 'N/A')}")
            print(f"  Optimization success: {diag.get('optimization_success', 'N/A')}")
            print(f"  Iterations: {diag.get('optimization_nit', 'N/A')}")
        elif method_name == "RAVEL":
            print(f"  N folds: {diag.get('n_folds', 'N/A')}")
            if 'mean_ess_ratio' in diag:
                print(f"  Mean ESS ratio: {diag.get('mean_ess_ratio', 'N/A'):.3f}")

    # Save results
    print("\n[6/6] Saving results...")
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save weight statistics
    weight_stats_path = results_dir / f"all_methods_{dataset_name}_weight_stats.csv"
    weight_stats_df.to_csv(weight_stats_path, index=False)
    print(f"[OK] Weight statistics saved to {weight_stats_path}")

    # Save correlation matrix
    corr_df = pd.DataFrame(
        corr_matrix,
        index=method_names,
        columns=method_names,
    )
    corr_path = results_dir / f"all_methods_{dataset_name}_weight_correlation.csv"
    corr_df.to_csv(corr_path)
    print(f"[OK] Weight correlation matrix saved to {corr_path}")

    # Save certification comparison
    cert_path = results_dir / f"all_methods_{dataset_name}_certification_comparison.csv"
    cert_df.to_csv(cert_path, index=False)
    print(f"[OK] Certification comparison saved to {cert_path}")

    # Save agreement matrix
    agreement_df = pd.DataFrame(
        agreement_matrix,
        index=method_order,
        columns=method_order,
    )
    agreement_path = results_dir / f"all_methods_{dataset_name}_decision_agreement.csv"
    agreement_df.to_csv(agreement_path)
    print(f"[OK] Decision agreement matrix saved to {agreement_path}")

    # Save runtime comparison
    runtime_path = results_dir / f"all_methods_{dataset_name}_runtime_comparison.csv"
    runtime_df.to_csv(runtime_path, index=False)
    print(f"[OK] Runtime comparison saved to {runtime_path}")

    # Create visualizations if requested
    if save_plots:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            print("\n[Optional] Creating visualizations...")

            # 1. Weight distributions
            fig, axes = plt.subplots(1, len(weights_dict), figsize=(5 * len(weights_dict), 4))
            if len(weights_dict) == 1:
                axes = [axes]

            for ax, (method_name, weights) in zip(axes, weights_dict.items()):
                ax.hist(weights, bins=30, alpha=0.7, edgecolor="black")
                ax.set_xlabel("Weight")
                ax.set_ylabel("Frequency")
                ax.set_title(f"{method_name}\nMean={weights.mean():.3f}, Std={weights.std():.3f}")
                ax.axvline(weights.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
                ax.legend()

            plt.tight_layout()
            weights_plot_path = results_dir / f"all_methods_{dataset_name}_weight_distributions.png"
            plt.savefig(weights_plot_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Weight distributions plot saved to {weights_plot_path}")
            plt.close()

            # 2. Weight correlation heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".3f",
                cmap="YlOrRd",
                xticklabels=method_names,
                yticklabels=method_names,
                vmin=0,
                vmax=1,
                ax=ax,
            )
            ax.set_title(f"Weight Correlation Matrix - {dataset_name.upper()}")
            plt.tight_layout()
            corr_plot_path = results_dir / f"all_methods_{dataset_name}_weight_correlation_heatmap.png"
            plt.savefig(corr_plot_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Weight correlation heatmap saved to {corr_plot_path}")
            plt.close()

            # 3. Agreement heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                agreement_matrix,
                annot=True,
                fmt=".1%",
                cmap="YlGnBu",
                xticklabels=method_order,
                yticklabels=method_order,
                vmin=0,
                vmax=1,
                ax=ax,
            )
            ax.set_title(f"Decision Agreement Matrix - {dataset_name.upper()}")
            plt.tight_layout()
            agreement_plot_path = results_dir / f"all_methods_{dataset_name}_decision_agreement_heatmap.png"
            plt.savefig(agreement_plot_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Decision agreement heatmap saved to {agreement_plot_path}")
            plt.close()

            # 4. Certification rate by tau
            fig, ax = plt.subplots(figsize=(10, 6))
            for method_name in decisions_dict.keys():
                rates = [cert_df[cert_df['tau'] == tau][f"{method_name}_rate"].iloc[0] for tau in tau_grid]
                ax.plot(tau_grid, rates, marker='o', label=method_name, linewidth=2)

            ax.set_xlabel("Tau (PPV Threshold)")
            ax.set_ylabel("Certification Rate")
            ax.set_title(f"Certification Rate vs. Tau - {dataset_name.upper()}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            cert_plot_path = results_dir / f"all_methods_{dataset_name}_certification_by_tau.png"
            plt.savefig(cert_plot_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Certification rate plot saved to {cert_plot_path}")
            plt.close()

        except Exception as e:
            print(f"[WARNING] Could not create plots: {e}")

    # Summary
    print_header("SUMMARY AND KEY FINDINGS")

    print("\nTheoretical Comparison:")
    print("  KMM (Kernel Mean Matching):")
    print("    - Objective: Minimize MMD between weighted calibration and test distributions")
    print("    - Optimization: Quadratic Programming with box constraints (0 <= w_i <= B)")
    print("    - Pros: Bounded weights prevent extreme values, direct distribution matching")
    print("    - Cons: Requires QP solver (slower), sensitive to kernel bandwidth")
    print("")
    print("  uLSIF (Unconstrained Least-Squares Importance Fitting):")
    print("    - Objective: Minimize squared loss between density ratio and basis functions")
    print("    - Optimization: Closed-form solution (ridge regression)")
    print("    - Pros: Fast, stable, no convergence issues")
    print("    - Cons: Squared loss not as principled, may have extreme weights")
    print("")
    print("  KLIEP (Kullback-Leibler Importance Estimation Procedure):")
    print("    - Objective: Minimize KL divergence from test to weighted calibration")
    print("    - Optimization: Iterative optimization with non-negativity constraints")
    print("    - Pros: KL optimality, guaranteed non-negative weights")
    print("    - Cons: Iterative (slower), sensitive to initialization")
    print("")
    if RAVEL_AVAILABLE:
        print("  RAVEL (Risk-Aware Validation with Embedded Localization):")
        print("    - Objective: Discriminative density ratio with cross-validation")
        print("    - Optimization: Logistic regression with stability diagnostics")
        print("    - Pros: Stability gating (abstains when unreliable), FWER control")
        print("    - Cons: May abstain more, requires cross-validation")

    print("\nEmpirical Findings:")
    print(f"  Weight Correlation (average off-diagonal): {np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]):.3f}")
    print(f"  Decision Agreement (average off-diagonal): {np.mean(agreement_matrix[np.triu_indices_from(agreement_matrix, k=1)]):.1%}")

    if len(weights_dict) >= 3:
        # Find most similar pair
        corr_triu = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        max_corr_idx = np.argmax(corr_triu)
        i, j = np.triu_indices_from(corr_matrix, k=1)
        max_corr_i, max_corr_j = i[max_corr_idx], j[max_corr_idx]
        print(f"  Most similar weights: {method_names[max_corr_i]} vs {method_names[max_corr_j]} (r={corr_matrix[max_corr_i, max_corr_j]:.3f})")

        # Find least similar pair
        min_corr_idx = np.argmin(corr_triu)
        min_corr_i, min_corr_j = i[min_corr_idx], j[min_corr_idx]
        print(f"  Least similar weights: {method_names[min_corr_i]} vs {method_names[min_corr_j]} (r={corr_matrix[min_corr_i, min_corr_j]:.3f})")

    print("\nRuntime Summary:")
    fastest = runtime_df.loc[runtime_df['weight_time_sec'].idxmin(), 'method']
    print(f"  Fastest: {fastest} ({runtime_df[runtime_df['method'] == fastest]['weight_time_sec'].iloc[0]:.3f}s)")
    slowest = runtime_df.loc[runtime_df['weight_time_sec'].idxmax(), 'method']
    print(f"  Slowest: {slowest} ({runtime_df[runtime_df['method'] == slowest]['weight_time_sec'].iloc[0]:.3f}s)")

    print("\nRecommendations:")
    print("  1. Use RAVEL for production (best safety via stability gating)")
    print("  2. Use uLSIF for fast prototyping (closed-form, stable)")
    print("  3. Use KMM when weight bounds are critical (prevents extreme weights)")
    print("  4. Use KLIEP when KL optimality is needed")
    print("  5. Always check weight diagnostics and cross-validate on your data")

    return {
        "weights": weights_dict,
        "decisions": decisions_dict,
        "weight_stats": weight_stats_df,
        "correlation_matrix": corr_df,
        "agreement_matrix": agreement_df,
        "runtime": runtime_df,
        "certification": cert_df,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare all density ratio estimation methods (RAVEL, uLSIF, KLIEP, KMM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="test_dataset",
        help="Dataset to use (default: test_dataset)",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save visualization plots (requires matplotlib and seaborn)",
    )

    args = parser.parse_args()

    print_header("ShiftBench: Comprehensive Density Ratio Method Comparison")
    print("\nThis script compares four density ratio estimation methods:")
    print("  1. KMM: Kernel Mean Matching (MMD minimization)")
    print("  2. uLSIF: Unconstrained Least-Squares Importance Fitting (L2 loss)")
    print("  3. KLIEP: Kullback-Leibler Importance Estimation (KL divergence)")
    print("  4. RAVEL: Risk-Aware Validation with stability gating (if available)")
    print("\nAnalysis includes:")
    print("  - Weight distribution statistics")
    print("  - Weight correlation matrix (method agreement)")
    print("  - Certification rate comparison")
    print("  - Runtime comparison")
    print("  - Method-specific diagnostics")

    results = compare_methods(args.dataset, args.save_plots)

    if results:
        print_header("COMPARISON COMPLETE")
        print(f"\nResults saved to: results/all_methods_{args.dataset}_*.csv")
        if args.save_plots:
            print(f"Plots saved to: results/all_methods_{args.dataset}_*.png")
        print("\nNext steps:")
        print("  1. Review correlation and agreement matrices")
        print("  2. Check if methods agree on high-stakes decisions")
        print("  3. Validate weight distributions are reasonable")
        print("  4. Compare to leaderboard baselines")


if __name__ == "__main__":
    main()
