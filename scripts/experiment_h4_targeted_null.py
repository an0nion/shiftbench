"""
H4 Targeted Null Experiment
==============================
Construct boundary cases where true_ppv = tau - epsilon for small epsilon.
Tests that FWER stays <= alpha even at "knife edge" nulls.

This answers the reviewer critique: "You never got close to the boundary."
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

shift_bench_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(shift_bench_dir / "src"))
sys.path.insert(0, str(shift_bench_dir.parent / "ravel" / "src"))
sys.path.insert(0, str(shift_bench_dir / "scripts"))

from ravel.bounds.empirical_bernstein import eb_lower_bound
from ravel.bounds.p_value import eb_p_value
from ravel.bounds.weighted_stats import weighted_stats_01
from ravel.bounds.holm import holm_reject


def generate_targeted_null_data(
    n_cal: int,
    n_cohorts: int,
    tau: float,
    epsilon: float,
    n_eff_target: float,
    seed: int,
):
    """Generate data where true_ppv = tau - epsilon (just below threshold).

    Args:
        n_cal: calibration samples
        n_cohorts: number of cohorts
        tau: certification threshold
        epsilon: how far below tau the true PPV is
        n_eff_target: target effective sample size (controls weight variance)
        seed: random seed

    Returns:
        dict with y_cal, preds_cal, cohorts_cal, weights, true_ppv
    """
    rng = np.random.RandomState(seed)
    true_ppv = tau - epsilon

    # Generate binary labels: P(Y=1 | pred=1) = true_ppv
    # All samples are "predicted positive" for simplicity
    n_per_cohort = n_cal // n_cohorts

    y_cal = np.zeros(n_cal, dtype=int)
    preds_cal = np.ones(n_cal, dtype=int)  # All predicted positive
    cohorts_cal = np.zeros(n_cal, dtype=int)

    for c in range(n_cohorts):
        start = c * n_per_cohort
        end = start + n_per_cohort if c < n_cohorts - 1 else n_cal
        cohorts_cal[start:end] = c
        # Each sample has probability true_ppv of being correctly positive
        y_cal[start:end] = rng.binomial(1, true_ppv, size=end - start)

    # Generate weights with controlled n_eff
    # For log-normal: ESS/n = 1/(1 + CV^2) where CV = sqrt(exp(sigma^2) - 1)
    # So sigma = sqrt(ln(n/n_eff_target + 1))  approximately
    target_ess_frac = n_eff_target / n_cal
    if target_ess_frac >= 0.99:
        weights = np.ones(n_cal)
    else:
        # CV^2 = 1/ess_frac - 1
        cv_sq = 1.0 / target_ess_frac - 1.0
        sigma = np.sqrt(np.log(cv_sq + 1))
        log_w = rng.normal(0, sigma, size=n_cal)
        weights = np.exp(log_w)
        weights = weights / weights.mean()

    return {
        "y_cal": y_cal,
        "preds_cal": preds_cal,
        "cohorts_cal": cohorts_cal,
        "weights": weights,
        "true_ppv": true_ppv,
        "tau": tau,
        "epsilon": epsilon,
    }


def run_targeted_null(n_trials=500, alpha=0.05, seed=42):
    """Run targeted null experiments across epsilon and n_eff settings."""
    out_dir = shift_bench_dir / "results" / "h4_targeted_null"
    os.makedirs(out_dir, exist_ok=True)

    tau = 0.7  # Fixed threshold
    epsilons = [0.005, 0.01, 0.02, 0.05, 0.10]
    n_eff_targets = [50, 100, 200, 500]
    n_cal = 2000
    n_cohorts = 5

    all_results = []

    for eps in epsilons:
        for neff_t in n_eff_targets:
            print(f"\n  epsilon={eps}, n_eff_target={neff_t}:")

            n_false_trials = 0
            cert_counts = []

            for trial in range(n_trials):
                if trial % max(1, n_trials // 3) == 0:
                    print(f"    Trial {trial+1}/{n_trials}...")

                trial_seed = seed + trial * 137

                data = generate_targeted_null_data(
                    n_cal=n_cal, n_cohorts=n_cohorts,
                    tau=tau, epsilon=eps, n_eff_target=neff_t,
                    seed=trial_seed,
                )

                # Run EB + Holm certification
                cohort_ids = np.unique(data["cohorts_cal"])
                tau_grid = np.array([tau])  # Single tau for clean test

                pvals = []
                test_info = []

                for cid in cohort_ids:
                    cmask = data["cohorts_cal"] == cid
                    pmask = cmask & (data["preds_cal"] == 1)
                    y_pos = data["y_cal"][pmask]
                    w_pos = data["weights"][pmask]

                    if len(y_pos) < 2 or w_pos.sum() == 0:
                        pvals.append(1.0)
                        test_info.append((cid, tau, np.nan, np.nan))
                        continue

                    w_pos = w_pos / w_pos.mean() if w_pos.mean() > 0 else w_pos
                    stats = weighted_stats_01(y_pos, w_pos)

                    pval = eb_p_value(stats.mu, stats.var, stats.n_eff, tau)
                    lb = eb_lower_bound(stats.mu, stats.var, stats.n_eff, alpha)
                    pvals.append(pval)
                    test_info.append((cid, tau, stats.mu, stats.n_eff))

                n_certified = 0
                n_false = 0

                if pvals:
                    rejected = holm_reject(pd.Series(pvals), alpha)
                    for i, (cid, t, mu, neff) in enumerate(test_info):
                        if rejected.iloc[i]:
                            n_certified += 1
                            # This IS a false cert because true_ppv < tau
                            n_false += 1

                has_false = int(n_false > 0)
                n_false_trials += has_false
                cert_counts.append(n_certified)

                all_results.append({
                    "epsilon": eps,
                    "n_eff_target": neff_t,
                    "trial_id": trial,
                    "n_certified": n_certified,
                    "n_false": n_false,
                    "fwer": has_false,
                    "true_ppv": data["true_ppv"],
                    "tau": tau,
                })

            obs_fwer = n_false_trials / n_trials
            mean_certs = np.mean(cert_counts)
            print(f"      FWER={obs_fwer:.3f} ({n_false_trials}/{n_trials}), "
                  f"mean_certs={mean_certs:.2f}")

    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "targeted_null_raw.csv", index=False)

    # Summary
    summary = df.groupby(["epsilon", "n_eff_target"]).agg(
        observed_fwer=("fwer", "mean"),
        total_false_trials=("fwer", "sum"),
        n_trials=("trial_id", "count"),
        mean_certified=("n_certified", "mean"),
        total_certified=("n_certified", "sum"),
    ).reset_index()
    summary.to_csv(out_dir / "targeted_null_summary.csv", index=False)

    # Wilson CI
    from scipy.stats import binomtest

    print(f"\n{'='*70}")
    print("H4 TARGETED NULL: FWER at knife-edge (true_ppv = tau - epsilon)")
    print(f"tau={tau}, alpha={alpha}, n_cal={n_cal}, n_cohorts={n_cohorts}")
    print(f"{'='*70}")
    print(f"{'epsilon':>8} {'n_eff':>6} {'FWER':>8} {'k/n':>10} "
          f"{'Wilson CI':>18} {'p(>alpha)':>10} {'certs':>6}")
    print("-" * 68)

    for _, row in summary.iterrows():
        k = int(row["total_false_trials"])
        n = int(row["n_trials"])
        obs = k / n
        z = 1.96
        denom = 1 + z**2/n
        center = (obs + z**2/(2*n)) / denom
        spread = z * np.sqrt(obs*(1-obs)/n + z**2/(4*n**2)) / denom
        wlo = max(0, center - spread)
        whi = min(1, center + spread)

        binom_p = binomtest(k, n, alpha, alternative='greater').pvalue
        marker = " *" if binom_p < 0.05 else ""

        print(f"{row['epsilon']:>8.3f} {row['n_eff_target']:>6.0f} {obs:>7.3f} "
              f"{k:>4}/{n:<4} [{wlo:.3f},{whi:.3f}] {binom_p:>10.4f}{marker} "
              f"{row['mean_certified']:>5.2f}")

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    run_targeted_null()
