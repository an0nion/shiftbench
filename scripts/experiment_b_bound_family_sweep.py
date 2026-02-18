"""
Experiment B: Bound-Family Sweep
==================================

Compares three bound families on the Pareto frontier of
validity (FWER) vs power (cert rate) vs runtime:

1. Empirical-Bernstein (EB) -- current method
2. Hoeffding -- distribution-free, more conservative
3. Bootstrap percentile -- data-driven, less conservative

Uses synthetic data with known ground-truth PPV for accurate
false-certify measurement.

Design:
    For each bound family:
    1. Run 50 trials in the usable regime (n_cal=2000, n_cohorts=5)
    2. Measure: FWER, certification rate, coverage, runtime
    3. Plot Pareto frontier: (FWER vs cert_rate) for each family

Expected findings:
    - EB: moderate power, valid (FWER <= alpha)
    - Hoeffding: low power, very conservative (FWER << alpha)
    - Bootstrap: high power, may violate FWER (coverage < 1-alpha)

Usage:
    python scripts/experiment_b_bound_family_sweep.py --quick
    python scripts/experiment_b_bound_family_sweep.py --n_trials 50
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


# ---------------------------------------------------------------------------
# Bound implementations
# ---------------------------------------------------------------------------

def hoeffding_lower_bound(
    mu_hat: float, n_eff: float, delta: float
) -> float:
    """Hoeffding's inequality lower bound for [0,1]-bounded mean.

    LB = mu_hat - sqrt(log(1/delta) / (2*n_eff))

    More conservative than EB because it ignores variance information.
    """
    if not math.isfinite(n_eff) or n_eff <= 0:
        return float("nan")
    z = math.log(1.0 / max(delta, 1e-300))
    return max(0.0, min(1.0, mu_hat - math.sqrt(z / (2.0 * n_eff))))


def bootstrap_lower_bound(
    y: np.ndarray,
    weights: np.ndarray,
    delta: float,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> float:
    """Weighted bootstrap percentile lower bound.

    Resamples with replacement proportional to weights,
    returns the (delta)-quantile of bootstrap means.

    Less conservative than EB but no formal guarantee.
    """
    if len(y) < 2 or weights.sum() == 0:
        return float("nan")

    rng = np.random.RandomState(seed)

    # Normalize weights to probabilities
    probs = weights / weights.sum()

    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(len(y), size=len(y), replace=True, p=probs)
        boot_means[b] = y[idx].mean()

    # Lower bound = delta-th percentile
    lb = np.quantile(boot_means, delta)
    return max(0.0, min(1.0, lb))


# ---------------------------------------------------------------------------
# Hoeffding p-value (analogous to EB p-value)
# ---------------------------------------------------------------------------

def hoeffding_p_value(
    mu_hat: float, n_eff: float, tau: float
) -> float:
    """P-value for testing H0: mu < tau using Hoeffding bound inversion.

    p = exp(-2 * n_eff * max(mu_hat - tau, 0)^2)
    """
    if not math.isfinite(n_eff) or n_eff <= 0:
        return 1.0
    margin = max(mu_hat - tau, 0.0)
    return math.exp(-2.0 * n_eff * margin * margin)


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def run_bound_sweep_trial(
    n_cal: int,
    n_test: int,
    n_cohorts: int,
    shift_severity: float,
    positive_rate: float,
    alpha: float,
    tau_grid: np.ndarray,
    trial_id: int,
    seed: int,
    n_bootstrap: int = 500,
) -> list:
    """Run one trial comparing all three bound families on same data.

    Returns list of dicts, one per (bound_family, trial) pair.
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

    # 2. Estimate importance weights
    try:
        from shiftbench.baselines.ulsif import uLSIFBaseline
        ulsif = uLSIFBaseline(n_basis=min(100, n_cal), sigma=None,
                              lambda_=0.1, random_state=seed)
        weights = ulsif.estimate_weights(data.X_cal, data.X_test)
    except Exception:
        weights = np.ones(len(data.X_cal))

    # 3. Pre-compute weighted stats per cohort (shared across bound families)
    cohort_ids = np.unique(data.cohorts_cal)
    cohort_stats = {}

    for cohort_id in cohort_ids:
        cohort_mask = data.cohorts_cal == cohort_id
        pos_mask = cohort_mask & (data.preds_cal == 1)

        y_pos = data.y_cal[pos_mask]
        w_pos = weights[pos_mask]

        if len(y_pos) < 2 or w_pos.sum() == 0:
            cohort_stats[cohort_id] = None
        else:
            stats = weighted_stats_01(y_pos, w_pos)
            cohort_stats[cohort_id] = {
                "mu_hat": stats.mu,
                "var_hat": stats.var,
                "n_eff": stats.n_eff,
                "y_pos": y_pos,
                "w_pos": w_pos,
            }

    # 4. Run each bound family
    results = []
    bound_families = ["eb", "hoeffding", "bootstrap"]

    for bound_name in bound_families:
        t0 = time.time()

        pvals = []
        test_info = []

        for cohort_id in cohort_ids:
            cs = cohort_stats[cohort_id]

            for tau in tau_grid:
                if cs is None:
                    pvals.append(1.0)
                    test_info.append((cohort_id, tau, np.nan, np.nan, np.nan))
                    continue

                mu = cs["mu_hat"]
                var = cs["var_hat"]
                n_eff = cs["n_eff"]

                if bound_name == "eb":
                    lb = eb_lower_bound(mu, var, n_eff, alpha)
                    pval = eb_p_value(mu, var, n_eff, tau)
                elif bound_name == "hoeffding":
                    lb = hoeffding_lower_bound(mu, n_eff, alpha)
                    pval = hoeffding_p_value(mu, n_eff, tau)
                elif bound_name == "bootstrap":
                    bs_seed = (seed + int(cohort_id) * 137) % (2**31)
                    lb = bootstrap_lower_bound(
                        cs["y_pos"], cs["w_pos"], alpha,
                        n_bootstrap=n_bootstrap, seed=bs_seed,
                    )
                    # Bootstrap p-value: fraction of bootstrap means < tau
                    rng_bs = np.random.RandomState(bs_seed + 1)
                    probs = cs["w_pos"] / cs["w_pos"].sum()
                    boot_means = np.array([
                        cs["y_pos"][rng_bs.choice(
                            len(cs["y_pos"]), size=len(cs["y_pos"]),
                            replace=True, p=probs
                        )].mean()
                        for _ in range(n_bootstrap)
                    ])
                    pval = (boot_means < tau).mean()

                pvals.append(pval)
                test_info.append((cohort_id, tau, mu, n_eff, lb))

        # Holm step-down (same correction for all families)
        n_certified = 0
        n_false_certify = 0
        n_tests = 0
        n_eff_list = []
        coverage_list = []

        if len(pvals) > 0:
            pvals_series = pd.Series(pvals)
            rejected = holm_reject(pvals_series, alpha)

            for i, (cohort_id, tau, mu, n_eff, lb) in enumerate(test_info):
                n_tests += 1
                if not np.isnan(n_eff):
                    n_eff_list.append(n_eff)

                if rejected.iloc[i]:
                    n_certified += 1
                    # Check false certification
                    true_ppv = data.true_ppv.get(cohort_id, {}).get(tau, np.nan)
                    if not np.isnan(true_ppv) and true_ppv < tau:
                        n_false_certify += 1
                    # Check coverage
                    if not np.isnan(true_ppv) and not np.isnan(lb):
                        coverage_list.append(float(true_ppv >= lb))

        runtime = time.time() - t0

        results.append({
            "bound_family": bound_name,
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
            "runtime_sec": runtime,
        })

    return results


def run_experiment_b(
    n_trials: int = 50,
    n_cal: int = 2000,
    n_cohorts: int = 5,
    shift_severity: float = 1.0,
    positive_rate: float = 0.5,
    alpha: float = 0.05,
    tau_grid: np.ndarray = None,
    seed: int = 42,
    output_dir: str = "results/experiment_b",
):
    """Run Experiment B: bound-family sweep."""
    if tau_grid is None:
        tau_grid = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

    os.makedirs(output_dir, exist_ok=True)

    print(f"Experiment B: Bound-Family Sweep")
    print(f"  n_trials: {n_trials}")
    print(f"  n_cal: {n_cal}, n_cohorts: {n_cohorts}")
    print(f"  shift_severity: {shift_severity}")
    print(f"  alpha: {alpha}")
    print(f"  Bound families: EB, Hoeffding, Bootstrap")
    print()

    all_results = []
    for trial in range(n_trials):
        if trial % max(1, n_trials // 10) == 0:
            print(f"  Trial {trial+1}/{n_trials}...")

        trial_seed = seed + trial
        trial_results = run_bound_sweep_trial(
            n_cal=n_cal, n_test=5000, n_cohorts=n_cohorts,
            shift_severity=shift_severity, positive_rate=positive_rate,
            alpha=alpha, tau_grid=tau_grid,
            trial_id=trial, seed=trial_seed,
        )
        all_results.extend(trial_results)

    # Save raw
    raw_df = pd.DataFrame(all_results)
    raw_path = os.path.join(output_dir, "experiment_b_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"\nSaved raw: {raw_path}")

    # Aggregate by bound family
    summary = raw_df.groupby("bound_family").agg(
        mean_cert_rate=("cert_rate", "mean"),
        std_cert_rate=("cert_rate", "std"),
        fwer_violations=("false_certify_fwer", "sum"),
        n_trials=("trial_id", "count"),
        mean_n_eff=("mean_n_eff", "mean"),
        mean_coverage=("coverage", "mean"),
        mean_runtime=("runtime_sec", "mean"),
        total_certified=("n_certified", "sum"),
        total_false=("n_false_certify", "sum"),
    ).reset_index()

    summary["observed_fwer"] = summary["fwer_violations"] / summary["n_trials"]

    summary_path = os.path.join(output_dir, "experiment_b_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    # Print results
    print(f"\n{'='*70}")
    print("EXPERIMENT B: BOUND-FAMILY PARETO FRONTIER")
    print(f"{'='*70}")
    print(f"\n{'Family':<12} {'FWER':<12} {'Cert%':<8} {'Coverage':<10} "
          f"{'Runtime':<10} {'Status':<8}")
    print("-" * 60)

    for _, row in summary.iterrows():
        fwer_str = f"{row['fwer_violations']:.0f}/{row['n_trials']:.0f} " \
                   f"({row['observed_fwer']:.1%})"
        cov = f"{row['mean_coverage']:.3f}" \
            if not np.isnan(row["mean_coverage"]) else "N/A"
        status = "PASS" if row["observed_fwer"] <= alpha else "FAIL"
        print(f"{row['bound_family']:<12} {fwer_str:<12} "
              f"{row['mean_cert_rate']:<8.1%} {cov:<10} "
              f"{row['mean_runtime']:<10.3f}s {status:<8}")

    print("-" * 60)

    # Pareto analysis
    print("\nPareto Frontier Analysis:")
    valid_families = summary[summary["observed_fwer"] <= alpha]
    if len(valid_families) > 0:
        best_power = valid_families.loc[
            valid_families["mean_cert_rate"].idxmax()
        ]
        print(f"  Highest power with valid FWER: {best_power['bound_family']} "
              f"(cert={best_power['mean_cert_rate']:.1%})")

    most_conservative = summary.loc[summary["mean_cert_rate"].idxmin()]
    most_powerful = summary.loc[summary["mean_cert_rate"].idxmax()]
    print(f"  Most conservative: {most_conservative['bound_family']} "
          f"(cert={most_conservative['mean_cert_rate']:.1%})")
    print(f"  Most powerful: {most_powerful['bound_family']} "
          f"(cert={most_powerful['mean_cert_rate']:.1%})")

    print(f"\n{'='*70}")

    # Text summary
    text = format_bound_summary(summary, n_trials, n_cal, n_cohorts, alpha)
    text_path = os.path.join(output_dir, "experiment_b_summary.txt")
    with open(text_path, "w") as f:
        f.write(text)
    print(f"Saved text summary: {text_path}")

    return raw_df, summary


def format_bound_summary(summary, n_trials, n_cal, n_cohorts, alpha):
    lines = []
    lines.append("=" * 70)
    lines.append("EXPERIMENT B: BOUND-FAMILY SWEEP")
    lines.append("=" * 70)
    lines.append(f"Design: {n_trials} trials, n_cal={n_cal}, "
                 f"n_cohorts={n_cohorts}, alpha={alpha}")
    lines.append("Synthetic data with known ground-truth PPV.")
    lines.append("")
    for _, row in summary.iterrows():
        status = "PASS" if row["observed_fwer"] <= alpha else "FAIL"
        lines.append(f"{row['bound_family']}: FWER={row['observed_fwer']:.1%}, "
                     f"cert={row['mean_cert_rate']:.1%}, "
                     f"runtime={row['mean_runtime']:.3f}s [{status}]")
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Experiment B: Bound-family sweep"
    )
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--n_cal", type=int, default=2000)
    parser.add_argument("--n_cohorts", type=int, default=5)
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/experiment_b")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 10 trials")
    args = parser.parse_args()

    if args.quick:
        args.n_trials = 10

    run_experiment_b(
        n_trials=args.n_trials,
        n_cal=args.n_cal,
        n_cohorts=args.n_cohorts,
        shift_severity=args.shift,
        alpha=args.alpha,
        seed=args.seed,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
