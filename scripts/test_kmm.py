"""Test KMM baseline and compare with uLSIF and KLIEP.

This script validates KMM (Kernel Mean Matching) implementation and provides comprehensive comparison:
1. Test on synthetic test_dataset
2. Test on real BACE molecular dataset
3. Compare weight statistics across methods (KMM, uLSIF, KLIEP)
4. Compare certification rates and runtime
5. Document results for ShiftBench leaderboard

Usage:
    python scripts/test_kmm.py
"""

import sys
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
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_subheader(title):
    """Print formatted subsection header."""
    print(f"\n{title}")
    print("-" * 80)


def analyze_weights(weights, method_name):
    """Print weight distribution statistics."""
    print(f"\n{method_name} Weight Statistics:")
    print(f"  Mean:       {weights.mean():.4f} (should be ~1.0)")
    print(f"  Std:        {weights.std():.4f}")
    print(f"  Min:        {weights.min():.4f}")
    print(f"  Max:        {weights.max():.4f}")
    print(f"  Median:     {np.median(weights):.4f}")
    print(f"  Q25-Q75:    [{np.percentile(weights, 25):.4f}, {np.percentile(weights, 75):.4f}]")
    print(f"  95th pct:   {np.percentile(weights, 95):.4f}")
    print(f"  99th pct:   {np.percentile(weights, 99):.4f}")
    print(f"  Coef. Var:  {weights.std() / weights.mean():.4f}")


def compare_decisions(decisions_dict, tau_grid):
    """Compare certification decisions across methods."""
    print_subheader("Certification Rate Comparison")
    print(f"{'Tau':<8} ", end="")
    for method in decisions_dict.keys():
        print(f"{method:<15} ", end="")
    print()
    print("-" * (8 + 15 * len(decisions_dict)))

    for tau in tau_grid:
        print(f"{tau:<8.2f} ", end="")
        for method, decisions in decisions_dict.items():
            tau_decisions = [d for d in decisions if d.tau == tau]
            n_certify = sum(1 for d in tau_decisions if d.decision == "CERTIFY")
            n_total = len(tau_decisions)
            cert_rate = n_certify / n_total if n_total > 0 else 0
            print(f"{n_certify}/{n_total} ({cert_rate:.1%})  ", end="")
        print()

    # Overall statistics
    print("-" * (8 + 15 * len(decisions_dict)))
    print(f"{'Overall':<8} ", end="")
    for method, decisions in decisions_dict.items():
        n_certify = sum(1 for d in decisions if d.decision == "CERTIFY")
        n_total = len(decisions)
        cert_rate = n_certify / n_total if n_total > 0 else 0
        print(f"{n_certify}/{n_total} ({cert_rate:.1%})  ", end="")
    print()


def test_on_dataset(dataset_name, use_oracle=False):
    """Test KMM vs uLSIF vs KLIEP vs RAVEL on a dataset.

    Args:
        dataset_name: Name of dataset ('test_dataset' or 'bace')
        use_oracle: If True, use true labels as predictions (isolates weight quality)
    """
    print_header(f"Testing on {dataset_name.upper()}")

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
            print(f"   Run 'python scripts/preprocess_molecular.py --dataset {dataset_name}' first!")
        return None

    # Split data
    print("\n[2/6] Splitting into calibration and test sets...")
    cal_mask = (splits["split"] == "cal").values
    test_mask = (splits["split"] == "test").values

    X_cal, y_cal, cohorts_cal = X[cal_mask], y[cal_mask], cohorts[cal_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"[OK] Calibration: {len(X_cal)} samples ({y_cal.mean():.2%} positive)")
    print(f"   Test: {len(X_test)} samples ({y_test.mean():.2%} positive)")

    # Initialize methods
    print("\n[3/6] Initializing methods...")
    methods = {}

    # KMM (new - the star of the show!)
    methods['KMM'] = create_kmm_baseline(
        sigma=None,  # Use median heuristic
        lambda_=0.1,
        B=1000.0,
        random_state=42,
        solver="auto",  # Use cvxpy if available, else scipy
    )

    # uLSIF (closed-form L2)
    methods['uLSIF'] = create_ulsif_baseline(
        n_basis=100 if dataset_name == "bace" else 50,
        sigma=None,
        lambda_=0.1,
        random_state=42,
    )

    # KLIEP (KL optimization)
    methods['KLIEP'] = create_kliep_baseline(
        n_basis=100 if dataset_name == "bace" else 50,
        sigma=None,
        max_iter=10000,
        random_state=42,
    )

    # RAVEL (if available)
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

    for method_name, method in methods.items():
        print(f"\n  {method_name}...")
        start_time = time.time()
        try:
            weights = method.estimate_weights(X_cal, X_test)
            elapsed = time.time() - start_time

            # Validate
            assert np.all(weights > 0), "All weights must be positive"
            assert np.all(np.isfinite(weights)), "All weights must be finite"
            assert np.abs(weights.mean() - 1.0) < 0.1, f"Mean should be ~1.0, got {weights.mean():.4f}"

            weights_dict[method_name] = weights
            runtimes[method_name] = elapsed

            print(f"  [OK] {method_name}: {elapsed:.2f}s")

            # Print diagnostics for KMM
            if method_name == "KMM":
                diag = method.get_diagnostics()
                print(f"       Sigma (bandwidth): {diag.get('sigma', 'N/A'):.4f}")
                print(f"       Lambda (ridge): {diag.get('lambda', 'N/A'):.4f}")
                print(f"       Box constraint B: {diag.get('B', 'N/A'):.1f}")
                print(f"       Optimization success: {diag.get('optimization_success', 'N/A')}")
                print(f"       Solve time: {diag.get('solve_time', 'N/A'):.3f}s")
                clipped_frac = diag.get('weights_clipped_fraction', 0)
                if clipped_frac > 0:
                    print(f"       WARNING: {clipped_frac:.1%} of weights hit upper bound B={diag.get('B', 'N/A')}")

        except Exception as e:
            print(f"  [ERROR] {method_name} failed: {e}")
            if method_name == "RAVEL":
                print("  (This may be expected if shift is too severe)")
            else:
                import traceback
                traceback.print_exc()

    if not weights_dict:
        print("[ERROR] No methods succeeded!")
        return None

    # Analyze weight distributions
    print_subheader("Weight Distribution Analysis")
    for method_name, weights in weights_dict.items():
        analyze_weights(weights, method_name)

    # Compare weights
    if len(weights_dict) > 1:
        print_subheader("Weight Correlation Analysis")
        method_names = list(weights_dict.keys())
        for i, method1 in enumerate(method_names):
            for method2 in method_names[i+1:]:
                w1, w2 = weights_dict[method1], weights_dict[method2]
                corr = np.corrcoef(w1, w2)[0, 1]
                print(f"  {method1} vs {method2}: {corr:.4f}")

    # Generate predictions
    print("\n[5/6] Generating predictions...")
    if use_oracle:
        predictions_cal = y_cal.astype(int)
        print(f"[OK] Using oracle predictions ({predictions_cal.sum()} positives)")
    else:
        # Naive baseline: predict all positive
        predictions_cal = np.ones(len(y_cal), dtype=int)
        print(f"[OK] Using naive predictions ({predictions_cal.sum()} positives)")

    # Estimate bounds
    print("\n[6/6] Estimating PPV bounds...")
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

            print(f"  [OK] {method_name}: {elapsed:.2f}s")
            print(f"       CERTIFY: {n_certify}/{len(decisions)} ({n_certify/len(decisions):.1%})")
            print(f"       ABSTAIN: {n_abstain}/{len(decisions)} ({n_abstain/len(decisions):.1%})")
            if n_no_guarantee > 0:
                print(f"       NO-GUARANTEE: {n_no_guarantee}/{len(decisions)} ({n_no_guarantee/len(decisions):.1%})")

        except Exception as e:
            print(f"  [ERROR] {method_name} bounds failed: {e}")
            import traceback
            traceback.print_exc()

    # Compare results
    if decisions_dict:
        compare_decisions(decisions_dict, tau_grid)

    # Runtime comparison
    print_subheader("Runtime Comparison")
    print(f"{'Method':<15} {'Weights':<15} {'Bounds':<15} {'Total':<15}")
    print("-" * 60)
    for method_name in weights_dict.keys():
        w_time = runtimes.get(method_name, 0)
        b_time = bound_runtimes.get(method_name, 0)
        total = w_time + b_time
        print(f"{method_name:<15} {w_time:<15.2f} {b_time:<15.2f} {total:<15.2f}")

    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results for each method
    for method_name, decisions in decisions_dict.items():
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

        output_path = results_dir / f"{method_name.lower()}_{dataset_name}_results.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\n[OK] {method_name} results saved to {output_path}")

    # Save summary comparison
    summary_rows = []
    for method_name, decisions in decisions_dict.items():
        for tau in tau_grid:
            tau_decisions = [d for d in decisions if d.tau == tau]
            n_certify = sum(1 for d in tau_decisions if d.decision == "CERTIFY")
            n_total = len(tau_decisions)
            summary_rows.append({
                "method": method_name,
                "dataset": dataset_name,
                "tau": tau,
                "n_certify": n_certify,
                "n_total": n_total,
                "cert_rate": n_certify / n_total if n_total > 0 else 0,
                "weight_time": runtimes.get(method_name, np.nan),
                "bound_time": bound_runtimes.get(method_name, np.nan),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = results_dir / f"kmm_comparison_{dataset_name}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[OK] Summary comparison saved to {summary_path}")

    return {
        "methods": methods,
        "weights": weights_dict,
        "decisions": decisions_dict,
        "runtimes": runtimes,
        "summary": summary_df,
    }


def main():
    """Run all tests and comparisons."""
    print_header("KMM Baseline Validation and Comparison Study")
    print("\nThis script compares four density ratio estimation methods:")
    print("  1. KMM: Kernel Mean Matching (MMD minimization, QP optimization)")
    print("  2. uLSIF: Unconstrained Least-Squares Importance Fitting (L2 loss, closed-form)")
    print("  3. KLIEP: Kullback-Leibler Importance Estimation (KL divergence, optimization)")
    print("  4. RAVEL: Cross-fitted discriminative + stability gating")

    # Test 1: Synthetic data
    print_header("EXPERIMENT 1: Synthetic Test Dataset")
    print("Purpose: Validate implementation on controlled synthetic distribution shift")

    results_synthetic = test_on_dataset("test_dataset", use_oracle=False)

    if results_synthetic is not None:
        print_header("EXPERIMENT 1: SUCCESS")
        print("\nKey Findings (Synthetic Data):")
        if 'KMM' in results_synthetic['weights']:
            kmm_weights = results_synthetic['weights']['KMM']
            print(f"  - KMM weights: mean={kmm_weights.mean():.3f}, std={kmm_weights.std():.3f}")
        if 'uLSIF' in results_synthetic['weights']:
            ulsif_weights = results_synthetic['weights']['uLSIF']
            print(f"  - uLSIF weights: mean={ulsif_weights.mean():.3f}, std={ulsif_weights.std():.3f}")
        if 'KLIEP' in results_synthetic['weights']:
            kliep_weights = results_synthetic['weights']['KLIEP']
            print(f"  - KLIEP weights: mean={kliep_weights.mean():.3f}, std={kliep_weights.std():.3f}")
        if 'KMM' in results_synthetic['decisions'] and 'uLSIF' in results_synthetic['decisions']:
            kmm_cert = sum(1 for d in results_synthetic['decisions']['KMM'] if d.decision == "CERTIFY")
            ulsif_cert = sum(1 for d in results_synthetic['decisions']['uLSIF'] if d.decision == "CERTIFY")
            print(f"  - KMM certified: {kmm_cert} cohort-tau pairs")
            print(f"  - uLSIF certified: {ulsif_cert} cohort-tau pairs")

    # Test 2: BACE molecular data
    print_header("EXPERIMENT 2: BACE Molecular Dataset")
    print("Purpose: Test on real-world molecular scaffold shift")

    results_bace = test_on_dataset("bace", use_oracle=True)

    if results_bace is not None:
        print_header("EXPERIMENT 2: SUCCESS")
        print("\nKey Findings (BACE Data):")
        if 'KMM' in results_bace['weights']:
            kmm_weights = results_bace['weights']['KMM']
            print(f"  - KMM weights: mean={kmm_weights.mean():.3f}, std={kmm_weights.std():.3f}")
        if 'uLSIF' in results_bace['weights']:
            ulsif_weights = results_bace['weights']['uLSIF']
            print(f"  - uLSIF weights: mean={ulsif_weights.mean():.3f}, std={ulsif_weights.std():.3f}")
        if 'KLIEP' in results_bace['weights']:
            kliep_weights = results_bace['weights']['KLIEP']
            print(f"  - KLIEP weights: mean={kliep_weights.mean():.3f}, std={kliep_weights.std():.3f}")
        if 'KMM' in results_bace['decisions'] and 'uLSIF' in results_bace['decisions']:
            kmm_cert = sum(1 for d in results_bace['decisions']['KMM'] if d.decision == "CERTIFY")
            ulsif_cert = sum(1 for d in results_bace['decisions']['uLSIF'] if d.decision == "CERTIFY")
            kliep_cert = sum(1 for d in results_bace['decisions']['KLIEP'] if d.decision == "CERTIFY")
            print(f"  - KMM certified: {kmm_cert} cohort-tau pairs")
            print(f"  - uLSIF certified: {ulsif_cert} cohort-tau pairs")
            print(f"  - KLIEP certified: {kliep_cert} cohort-tau pairs")

    # Final summary
    print_header("FINAL SUMMARY AND RECOMMENDATIONS")

    print("\nTheoretical Comparison:")
    print("  KMM:")
    print("    + Directly minimizes MMD (distribution matching distance)")
    print("    + Bounded weights (0 <= w_i <= B) prevent extreme values")
    print("    + Well-studied theoretical properties (Huang et al. 2007)")
    print("    - Requires QP solver (slower than uLSIF, similar to KLIEP)")
    print("    - Sensitive to kernel bandwidth selection")
    print("    - No stability diagnostics")
    print("")
    print("  uLSIF:")
    print("    + Closed-form solution (fast, no convergence issues)")
    print("    + Automatic regularization via ridge penalty")
    print("    + Numerically stable")
    print("    - Squared loss not as principled as MMD or KL")
    print("    - No explicit weight bounds (can get extreme values)")
    print("    - No stability diagnostics")
    print("")
    print("  KLIEP:")
    print("    + Directly minimizes KL divergence (statistically optimal)")
    print("    + Guaranteed non-negative weights via constrained optimization")
    print("    - Requires iterative optimization (slower)")
    print("    - Sensitive to initialization and hyperparameters")
    print("    - No stability diagnostics")
    print("")
    print("  RAVEL:")
    print("    + Cross-fitted to avoid overfitting")
    print("    + Stability gating (abstains when weights unreliable)")
    print("    + FWER control via Holm step-down")
    print("    - May abstain on cohorts where other methods certify")
    print("    - Requires cross-validation (slower)")

    if results_synthetic and results_bace:
        print("\nEmpirical Findings:")
        print("  (Based on results saved in results/ directory)")
        print("")
        print("  Weight Quality:")
        print("    - Compare weight distributions in detailed logs above")
        print("    - Check weight correlations between methods")
        print("    - KMM should have bounded weights (max <= B=1000)")
        print("    - uLSIF may have extreme weights (no explicit bounds)")
        print("    - Look for weight variance differences due to different objectives")
        print("")
        print("  Certification Rates:")
        print("    - See comparison tables above")
        print("    - RAVEL should have highest certification rate (uses gating)")
        print("    - KMM, uLSIF, KLIEP should be similar (all direct, no gating)")
        print("    - Differences indicate statistical efficiency tradeoffs")
        print("")
        print("  Runtime:")
        print("    - uLSIF fastest (closed-form)")
        print("    - KMM and KLIEP similar (both use optimization)")
        print("    - RAVEL slowest (cross-validation)")
        print("")
        print("  Agreement:")
        print("    - Check weight correlations (should be high, e.g., >0.8)")
        print("    - All methods estimate same density ratio, different approaches")
        print("    - Disagreement indicates numerical issues or severe shift")

    print("\nRecommendations for ShiftBench Users:")
    print("  1. Use RAVEL for production (best safety via stability gating)")
    print("  2. Use uLSIF for fast prototyping (closed-form, stable)")
    print("  3. Use KMM when weight bounds are critical (prevents extreme weights)")
    print("  4. Use KLIEP when KL optimality is critical and compute allows")
    print("  5. Always check weight diagnostics (ESS, max weight, clipping)")
    print("  6. Compare multiple methods on your specific dataset")

    print_header("VALIDATION COMPLETE")
    print("\nNext steps:")
    print("  1. Review results in results/ directory")
    print("  2. Compare certification rates across methods")
    print("  3. Validate weight distributions look reasonable")
    print("  4. Check KMM-specific diagnostics (clipping, solve time)")
    print("  5. Submit KMM to ShiftBench leaderboard if passing all checks")


if __name__ == "__main__":
    main()
