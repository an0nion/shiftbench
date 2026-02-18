"""
Experiment D: Gating Ablation
================================

Shows that stability diagnostics (ESS gating) are necessary by
constructing scenarios with pathological importance weights where
ungated methods produce false certifications.

Design:
    Key insight: uLSIF's regularization prevents truly pathological weights,
    so we DIRECTLY INJECT synthetic weights with controlled pathology to
    isolate the gating mechanism.

    1. Generate synthetic data with known ground-truth PPV
    2. Create weight variants with controlled tail heaviness:
       - "well_behaved": log-normal, low variance (n_eff/n ~ 0.8)
       - "moderate_tails": log-normal, medium variance (n_eff/n ~ 0.3)
       - "heavy_tails": Pareto-distributed (n_eff/n ~ 0.05-0.10)
       - "adversarial": Extreme Pareto (single weight dominates)
    3. For each weight type, run EB+Holm certification:
       - "Ungated": raw weights used directly
       - "ESS-gated": cohorts with n_eff/n < threshold -> abstain
       - "Clipped-99": weights clipped at 99th percentile
    4. Measure FWER (false certifications), cert rate, coverage

Expected findings:
    - Well-behaved weights: All variants agree (gating doesn't trigger)
    - Heavy-tailed weights: Ungated has FWER > alpha, gated maintains control
    - Adversarial weights: Ungated fails badly, gated abstains correctly

Usage:
    python scripts/experiment_d_gating_ablation.py --quick
    python scripts/experiment_d_gating_ablation.py --n_trials 100
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project paths
shift_bench_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(shift_bench_dir / "src"))
sys.path.insert(0, str(shift_bench_dir.parent / "ravel" / "src"))
sys.path.insert(0, str(shift_bench_dir / "scripts"))

from synthetic_shift_generator import SyntheticShiftGenerator


def generate_synthetic_weights(
    n: int,
    pathology: str,
    seed: int,
) -> np.ndarray:
    """Generate importance weights with controlled tail heaviness.

    Args:
        n: Number of samples
        pathology: one of "well_behaved", "moderate_tails", "heavy_tails", "adversarial"
        seed: Random seed

    Returns:
        weights array of shape (n,), mean-normalized to ~1.0
    """
    rng = np.random.RandomState(seed)

    if pathology == "well_behaved":
        # Log-normal with low variance -> n_eff/n ~ 0.7-0.9
        log_w = rng.normal(0, 0.3, size=n)
        weights = np.exp(log_w)

    elif pathology == "moderate_tails":
        # Log-normal with medium variance -> n_eff/n ~ 0.2-0.4
        log_w = rng.normal(0, 1.0, size=n)
        weights = np.exp(log_w)

    elif pathology == "heavy_tails":
        # Pareto-distributed -> n_eff/n ~ 0.05-0.15
        # Pareto(alpha=1.5) has heavy tails
        weights = (rng.pareto(1.5, size=n) + 1.0)

    elif pathology == "adversarial":
        # Extreme: one or two samples dominate
        weights = np.ones(n) * 0.01
        n_dominant = max(1, n // 50)  # ~2% of samples get huge weights
        dominant_idx = rng.choice(n, size=n_dominant, replace=False)
        weights[dominant_idx] = rng.pareto(0.8, size=n_dominant) + 10.0

    else:
        raise ValueError(f"Unknown pathology: {pathology}")

    # Normalize to mean 1
    weights = weights / weights.mean()
    return weights


def apply_ess_gate(weights: np.ndarray, min_ess_frac: float = 0.15) -> dict:
    """Apply ESS gate. Returns dict with gated weights and diagnostics."""
    n = len(weights)
    if n == 0:
        return {"weights": weights, "gated": False, "n_eff": 0, "ess_frac": 0}

    n_eff = (weights.sum() ** 2) / (weights ** 2).sum()
    ess_frac = n_eff / n

    if ess_frac < min_ess_frac:
        return {
            "weights": np.ones(n),
            "gated": True,
            "n_eff": n_eff,
            "ess_frac": ess_frac,
        }

    return {"weights": weights, "gated": False, "n_eff": n_eff, "ess_frac": ess_frac}


def clip_weights(weights: np.ndarray, quantile: float = 0.99) -> np.ndarray:
    """Clip weights at the given quantile and renormalize."""
    if len(weights) == 0:
        return weights

    threshold = np.quantile(weights, quantile)
    clipped = np.minimum(weights, threshold)

    if clipped.sum() > 0:
        clipped = clipped / clipped.mean()

    return clipped


def run_gating_trial(
    n_cal: int,
    n_test: int,
    n_cohorts: int,
    shift_severity: float,
    positive_rate: float,
    pathology: str,
    alpha: float,
    tau_grid: np.ndarray,
    trial_id: int,
    seed: int,
) -> list:
    """Run one trial comparing gated, ungated, and clipped pipelines
    with synthetic pathological weights.

    Returns list of dicts, one per (variant, trial).
    """
    from ravel.bounds.empirical_bernstein import eb_lower_bound
    from ravel.bounds.p_value import eb_p_value
    from ravel.bounds.weighted_stats import weighted_stats_01
    from ravel.bounds.holm import holm_reject

    # 1. Generate synthetic data
    gen = SyntheticShiftGenerator(
        n_cal=n_cal, n_test=n_test, n_cohorts=n_cohorts,
        d_features=10, shift_severity=shift_severity,
        positive_rate=positive_rate, seed=seed,
    )
    data = gen.generate(tau_grid=tau_grid)

    # 2. Generate synthetic weights with controlled pathology
    raw_weights = generate_synthetic_weights(n_cal, pathology, seed + 99999)

    # 3. Define weight processing variants
    # Key insight: EB with n_eff provides implicit weight safety.
    # We test what happens with naive n (no n_eff correction) to show
    # why explicit gating OR n_eff correction is necessary.
    variant_configs = {
        # Naive: EB uses raw n, no weight correction in bound width
        "naive_ungated": {"process": "none", "use_neff": False},
        "naive_clipped_99": {"process": "clip", "quantile": 0.99, "use_neff": False},
        "naive_ess_gated": {"process": "ess_gate", "min_ess_frac": 0.15, "use_neff": False},
        # n_eff: EB uses n_eff (current RAVEL approach)
        "neff_ungated": {"process": "none", "use_neff": True},
        "neff_ess_gated": {"process": "ess_gate", "min_ess_frac": 0.15, "use_neff": True},
    }

    # 4. Run certification for each variant
    results = []
    cohort_ids = np.unique(data.cohorts_cal)

    for variant_name, config in variant_configs.items():
        t0 = time.time()

        pvals = []
        test_info = []
        n_eff_list = []
        gated_cohorts = 0
        total_ess_frac = []

        for cohort_id in cohort_ids:
            cohort_mask = data.cohorts_cal == cohort_id
            pos_mask = cohort_mask & (data.preds_cal == 1)

            y_pos = data.y_cal[pos_mask]
            w_pos = raw_weights[pos_mask].copy()

            if len(y_pos) < 2 or w_pos.sum() == 0:
                for tau in tau_grid:
                    pvals.append(1.0)
                    test_info.append((cohort_id, tau, np.nan, np.nan, np.nan))
                continue

            # Apply weight processing
            if config["process"] == "clip":
                w_pos = clip_weights(w_pos, config["quantile"])
            elif config["process"] == "ess_gate":
                gate_result = apply_ess_gate(w_pos, config["min_ess_frac"])
                total_ess_frac.append(gate_result["ess_frac"])
                if gate_result["gated"]:
                    gated_cohorts += 1
                    for tau in tau_grid:
                        pvals.append(1.0)  # Abstain
                        test_info.append((cohort_id, tau, np.nan, np.nan, np.nan))
                    continue
                w_pos = gate_result["weights"]

            # Compute weighted stats
            result = weighted_stats_01(y_pos, w_pos)
            n_eff_list.append(result.n_eff)

            # Choose sample size for EB bound
            use_neff = config.get("use_neff", True)
            bound_n = result.n_eff if use_neff else float(len(y_pos))

            for tau in tau_grid:
                pval = eb_p_value(result.mu, result.var, bound_n, tau)
                lb = eb_lower_bound(result.mu, result.var, bound_n, alpha)
                pvals.append(pval)
                test_info.append((cohort_id, tau, result.mu, bound_n, lb))

        # Holm step-down
        n_certified = 0
        n_false_certify = 0
        n_tests = 0
        coverage_list = []

        if len(pvals) > 0:
            pvals_series = pd.Series(pvals)
            rejected = holm_reject(pvals_series, alpha)

            for i, (cohort_id, tau, mu, n_eff, lb) in enumerate(test_info):
                n_tests += 1
                if rejected.iloc[i]:
                    n_certified += 1
                    true_ppv = data.true_ppv.get(cohort_id, {}).get(tau, np.nan)
                    if not np.isnan(true_ppv) and true_ppv < tau:
                        n_false_certify += 1
                    if not np.isnan(true_ppv) and not np.isnan(lb):
                        coverage_list.append(float(true_ppv >= lb))

        runtime = time.time() - t0

        # Weight diagnostics
        w_diagnostic = raw_weights.copy()
        w_n_eff = (w_diagnostic.sum() ** 2) / (w_diagnostic ** 2).sum()
        w_max_ratio = w_diagnostic.max() / w_diagnostic.mean() if w_diagnostic.mean() > 0 else 0
        w_cv = w_diagnostic.std() / w_diagnostic.mean() if w_diagnostic.mean() > 0 else 0

        results.append({
            "variant": variant_name,
            "pathology": pathology,
            "trial_id": trial_id,
            "n_cal": n_cal,
            "n_cohorts": n_cohorts,
            "shift_severity": shift_severity,
            "alpha": alpha,
            "n_tests": n_tests,
            "n_certified": n_certified,
            "cert_rate": n_certified / n_tests if n_tests > 0 else 0.0,
            "n_false_certify": n_false_certify,
            "false_certify_fwer": int(n_false_certify > 0),
            "mean_n_eff": np.mean(n_eff_list) if n_eff_list else np.nan,
            "coverage": np.mean(coverage_list) if coverage_list else np.nan,
            "gated_cohorts": gated_cohorts,
            "global_ess_frac": w_n_eff / n_cal,
            "weight_cv": w_cv,
            "weight_max_ratio": w_max_ratio,
            "runtime_sec": runtime,
        })

    return results


def run_experiment_d(
    n_trials: int = 100,
    alpha: float = 0.05,
    tau_grid: np.ndarray = None,
    seed: int = 42,
    output_dir: str = "results/experiment_d",
):
    """Run Experiment D: gating ablation with synthetic pathological weights."""
    if tau_grid is None:
        tau_grid = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

    os.makedirs(output_dir, exist_ok=True)

    # Test configurations: (n_cal, n_cohorts, shift_sev, pathology, label)
    configs = [
        # Well-behaved weights: All variants should agree
        (2000, 5, 1.0, "well_behaved", "usable_good_weights"),
        # Moderate tails: some differences
        (2000, 5, 1.0, "moderate_tails", "usable_moderate_tails"),
        # Heavy tails: ungated should fail
        (2000, 5, 1.0, "heavy_tails", "usable_heavy_tails"),
        # Adversarial weights: ungated should fail badly
        (2000, 5, 1.0, "adversarial", "usable_adversarial"),
        # Heavy tails + small n: worst case
        (500, 10, 1.5, "heavy_tails", "small_n_heavy_tails"),
        # Adversarial + small n: extreme worst case
        (500, 10, 1.5, "adversarial", "small_n_adversarial"),
    ]

    all_results = []

    for n_cal, n_cohorts, shift_sev, pathology, config_label in configs:
        print(f"\n{'='*70}")
        print(f"Config: {config_label}")
        print(f"  n_cal={n_cal}, n_cohorts={n_cohorts}, shift={shift_sev}, "
              f"pathology={pathology}")
        print(f"{'='*70}")

        for trial in range(n_trials):
            if trial % max(1, n_trials // 5) == 0:
                print(f"  Trial {trial+1}/{n_trials}...")

            trial_seed = seed + trial * 137
            trial_results = run_gating_trial(
                n_cal=n_cal, n_test=5000, n_cohorts=n_cohorts,
                shift_severity=shift_sev, positive_rate=0.5,
                pathology=pathology,
                alpha=alpha, tau_grid=tau_grid,
                trial_id=trial, seed=trial_seed % (2**31),
            )
            for r in trial_results:
                r["config"] = config_label
            all_results.extend(trial_results)

    # Save raw
    raw_df = pd.DataFrame(all_results)
    raw_path = os.path.join(output_dir, "experiment_d_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"\nSaved raw: {raw_path}")

    # Aggregate by (config, variant)
    summary = raw_df.groupby(["config", "pathology", "variant"]).agg(
        mean_cert_rate=("cert_rate", "mean"),
        fwer_violations=("false_certify_fwer", "sum"),
        n_trials=("trial_id", "count"),
        total_certified=("n_certified", "sum"),
        total_false=("n_false_certify", "sum"),
        mean_n_eff=("mean_n_eff", "mean"),
        mean_coverage=("coverage", "mean"),
        mean_gated=("gated_cohorts", "mean"),
        mean_ess_frac=("global_ess_frac", "mean"),
        mean_weight_cv=("weight_cv", "mean"),
    ).reset_index()

    summary["observed_fwer"] = summary["fwer_violations"] / summary["n_trials"]

    summary_path = os.path.join(output_dir, "experiment_d_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    # Print results
    print(f"\n{'='*70}")
    print("EXPERIMENT D: GATING ABLATION (SYNTHETIC PATHOLOGICAL WEIGHTS)")
    print(f"{'='*70}")

    for config_label in summary["config"].unique():
        config_data = summary[summary["config"] == config_label]
        pathology = config_data.iloc[0]["pathology"]
        print(f"\n--- {config_label} (weights: {pathology}) ---")
        print(f"{'Variant':<15} {'FWER':<18} {'Cert%':<8} {'n_eff':<8} "
              f"{'Gated':<8} {'Cover':<8}")
        print("-" * 65)
        for _, row in config_data.iterrows():
            fwer_str = (f"{row['fwer_violations']:.0f}/{row['n_trials']:.0f} "
                        f"({row['observed_fwer']:.1%})")
            cov = (f"{row['mean_coverage']:.3f}"
                   if not np.isnan(row["mean_coverage"]) else "N/A")
            gated = f"{row['mean_gated']:.1f}"
            print(f"{row['variant']:<15} {fwer_str:<18} "
                  f"{row['mean_cert_rate']:<8.1%} "
                  f"{row['mean_n_eff']:<8.1f} {gated:<8} {cov:<8}")

    # Key findings
    print(f"\n{'='*70}")
    print("KEY FINDINGS:")

    # Compare ungated vs gated under heavy tails
    for label_check in ["usable_heavy_tails", "usable_adversarial",
                        "small_n_heavy_tails", "small_n_adversarial"]:
        sub = summary[summary["config"] == label_check]
        if len(sub) == 0:
            continue
        ungated = sub[sub["variant"] == "ungated"]
        gated = sub[sub["variant"] == "ess_gated_15"]

        if len(ungated) > 0 and len(gated) > 0:
            u_fwer = ungated.iloc[0]["observed_fwer"]
            g_fwer = gated.iloc[0]["observed_fwer"]
            u_cert = ungated.iloc[0]["mean_cert_rate"]
            g_cert = gated.iloc[0]["mean_cert_rate"]

            status = ""
            if u_fwer > alpha and g_fwer <= alpha:
                status = "GATING NECESSARY"
            elif u_fwer <= alpha and g_fwer <= alpha:
                status = "Both valid (EB conservative enough)"
            elif u_fwer > alpha and g_fwer > alpha:
                status = "Both fail (need stronger gating)"
            else:
                status = "Unexpected (gated worse than ungated)"

            print(f"\n  {label_check}:")
            print(f"    Ungated: FWER={u_fwer:.1%}, cert={u_cert:.1%}")
            print(f"    ESS-gated: FWER={g_fwer:.1%}, cert={g_cert:.1%}")
            print(f"    -> {status}")

    print(f"\n{'='*70}")

    # Save text summary
    text = format_gating_summary(summary, n_trials, alpha)
    text_path = os.path.join(output_dir, "experiment_d_summary.txt")
    with open(text_path, "w") as f:
        f.write(text)
    print(f"Saved text summary: {text_path}")

    return raw_df, summary


def format_gating_summary(summary, n_trials, alpha):
    lines = []
    lines.append("=" * 70)
    lines.append("EXPERIMENT D: GATING ABLATION (SYNTHETIC PATHOLOGICAL WEIGHTS)")
    lines.append("=" * 70)
    lines.append(f"Design: {n_trials} trials per config, alpha={alpha}")
    lines.append("Weights injected with controlled pathology levels.")
    lines.append("Variants: ungated, clipped_99, clipped_95, ess_gated_15, ess_gated_25")
    lines.append("")

    for config in summary["config"].unique():
        config_data = summary[summary["config"] == config]
        pathology = config_data.iloc[0]["pathology"]
        lines.append(f"--- {config} (weights: {pathology}) ---")
        for _, row in config_data.iterrows():
            status = "PASS" if row["observed_fwer"] <= alpha else "FAIL"
            cov = (f"cov={row['mean_coverage']:.3f}"
                   if not np.isnan(row["mean_coverage"]) else "cov=N/A")
            lines.append(
                f"  {row['variant']}: FWER={row['observed_fwer']:.1%}, "
                f"cert={row['mean_cert_rate']:.1%}, "
                f"gated={row['mean_gated']:.1f}, "
                f"{cov} [{status}]"
            )
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Experiment D: Gating ablation with synthetic pathological weights"
    )
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/experiment_d")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 20 trials")
    args = parser.parse_args()

    if args.quick:
        args.n_trials = 20

    run_experiment_d(
        n_trials=args.n_trials,
        alpha=args.alpha,
        seed=args.seed,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
