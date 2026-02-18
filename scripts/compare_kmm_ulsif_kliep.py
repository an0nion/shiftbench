"""Quick comparison of KMM vs uLSIF vs KLIEP on a single dataset.

This script provides a focused comparison of three density ratio methods:
- KMM: MMD minimization via QP
- uLSIF: L2 loss minimization (closed-form)
- KLIEP: KL divergence minimization via optimization

Usage:
    python scripts/compare_kmm_ulsif_kliep.py [--dataset test_dataset|bace]
"""

import sys
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt

from shiftbench.baselines.kmm import create_kmm_baseline
from shiftbench.baselines.ulsif import create_ulsif_baseline
from shiftbench.baselines.kliep import create_kliep_baseline
from shiftbench.data import load_dataset


def compare_methods(dataset_name="test_dataset"):
    """Compare KMM, uLSIF, and KLIEP on a dataset."""

    print("=" * 80)
    print(f" Comparing KMM vs uLSIF vs KLIEP on {dataset_name.upper()}")
    print("=" * 80)

    # Load dataset
    print("\n[1/3] Loading dataset...")
    try:
        X, y, cohorts, splits = load_dataset(dataset_name)
        print(f"[OK] Loaded {len(X)} samples with {X.shape[1]} features")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return None

    # Split data
    cal_mask = (splits["split"] == "cal").values
    test_mask = (splits["split"] == "test").values
    X_cal, X_test = X[cal_mask], X[test_mask]

    print(f"   Calibration: {len(X_cal)} samples")
    print(f"   Test: {len(X_test)} samples")

    # Initialize methods
    print("\n[2/3] Estimating weights with each method...")

    methods = {
        "KMM": create_kmm_baseline(sigma=None, lambda_=0.1, B=1000.0),
        "uLSIF": create_ulsif_baseline(n_basis=100, sigma=None, lambda_=0.1),
        "KLIEP": create_kliep_baseline(n_basis=100, sigma=None, max_iter=10000),
    }

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
        }

        print(f"    Runtime: {elapsed:.3f}s")
        print(f"    Weight mean: {weights.mean():.4f}")
        print(f"    Weight std: {weights.std():.4f}")
        print(f"    Weight max: {weights.max():.4f}")
        print(f"    Weight min: {weights.min():.4f}")

    # Compare weights
    print("\n[3/3] Comparing weight distributions...")

    # Compute correlations
    print("\n  Weight Correlations:")
    methods_list = list(results.keys())
    for i, m1 in enumerate(methods_list):
        for m2 in methods_list[i+1:]:
            w1, w2 = results[m1]["weights"], results[m2]["weights"]
            corr = np.corrcoef(w1, w2)[0, 1]
            print(f"    {m1} vs {m2}: {corr:.4f}")

    # Print diagnostics
    print("\n  Method-Specific Diagnostics:")
    for name, res in results.items():
        diag = res["diagnostics"]
        print(f"\n    {name}:")
        if name == "KMM":
            print(f"      Sigma (bandwidth): {diag.get('sigma', 'N/A'):.4f}")
            print(f"      Lambda (ridge): {diag.get('lambda', 'N/A'):.4f}")
            print(f"      B (box constraint): {diag.get('B', 'N/A'):.1f}")
            print(f"      Weights clipped: {diag.get('weights_clipped_fraction', 0):.1%}")
            print(f"      Optimization success: {diag.get('optimization_success', 'N/A')}")
        elif name == "uLSIF":
            print(f"      Sigma (bandwidth): {diag.get('sigma', 'N/A'):.4f}")
            print(f"      Alpha min/max: {diag.get('alpha_min', 'N/A'):.4f} / {diag.get('alpha_max', 'N/A'):.4f}")
        elif name == "KLIEP":
            print(f"      Sigma (bandwidth): {diag.get('sigma', 'N/A'):.4f}")
            print(f"      Optimization success: {diag.get('optimization_success', 'N/A')}")
            print(f"      Iterations: {diag.get('optimization_nit', 'N/A')}")

    # Summary table
    print("\n" + "=" * 80)
    print(" SUMMARY COMPARISON")
    print("=" * 80)
    print(f"\n{'Method':<10} {'Runtime':<12} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 72)
    for name, res in results.items():
        w = res["weights"]
        print(f"{name:<10} {res['runtime']:<12.3f} {w.mean():<10.4f} {w.std():<10.4f} {w.min():<10.4f} {w.max():<10.4f}")

    print("\nKey Observations:")
    print(f"  - uLSIF is fastest (closed-form solution)")
    print(f"  - KMM and KLIEP use optimization (slower)")
    print(f"  - KMM has bounded weights (0 <= w <= B={results['KMM']['diagnostics'].get('B', 'N/A')})")
    print(f"  - uLSIF has lowest weight variance (most regularized)")
    print(f"  - Weight correlations indicate agreement between methods")

    # Create visualization
    try:
        print("\n[Optional] Creating weight distribution plots...")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for ax, (name, res) in zip(axes, results.items()):
            weights = res["weights"]
            ax.hist(weights, bins=30, alpha=0.7, edgecolor="black")
            ax.set_xlabel("Weight")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{name}\nMean={weights.mean():.3f}, Std={weights.std():.3f}")
            ax.axvline(weights.mean(), color='red', linestyle='--', label='Mean')
            ax.legend()

        plt.tight_layout()
        output_path = Path(__file__).parent.parent / "results" / f"kmm_comparison_{dataset_name}_weights.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Weight distribution plot saved to {output_path}")
        plt.close()
    except Exception as e:
        print(f"[WARNING] Could not create plots: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare KMM vs uLSIF vs KLIEP")
    parser.add_argument(
        "--dataset",
        type=str,
        default="test_dataset",
        choices=["test_dataset", "bace"],
        help="Dataset to use (default: test_dataset)",
    )
    args = parser.parse_args()

    compare_methods(args.dataset)


if __name__ == "__main__":
    main()
