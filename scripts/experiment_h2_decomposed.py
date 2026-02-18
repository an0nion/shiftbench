"""
H2 Decomposed Sweeps: H2-A (Tail Heaviness) and H2-B (ESS)
==============================================================
Cleanly separates the two mechanisms that cause FWER violations:
  A) Heavy-tailed importance weights -> bound invalidation
  B) Low effective sample size -> conservative but potentially unstable

H2-A: Fix true_ppv, cohort size, tau. Sweep tail parameter (log-normal sigma).
       Compare: ungated vs clip vs ESS-gate vs full n_eff.

H2-B: Well-behaved weight shapes, but vary CV to reduce ESS.
       Show cert_rate declines gracefully while FWER stays controlled.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

shift_bench_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(shift_bench_dir / "src"))
sys.path.insert(0, str(shift_bench_dir.parent / "ravel" / "src"))
sys.path.insert(0, str(shift_bench_dir / "scripts"))

from synthetic_shift_generator import SyntheticShiftGenerator
from experiment_d_gating_ablation import (
    generate_synthetic_weights,
    apply_ess_gate,
    clip_weights,
    run_gating_trial,
)


def run_h2a_tail_sweep(n_trials=200, alpha=0.05, seed=42):
    """H2-A: Sweep log-normal sigma to vary tail heaviness."""
    print("=" * 70)
    print("H2-A: TAIL HEAVINESS SWEEP")
    print("=" * 70)

    tau_grid = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    # Log-normal sigma controls tail: sigma=0.1 (near-uniform) to sigma=3.0 (extreme)
    sigmas = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]

    from ravel.bounds.empirical_bernstein import eb_lower_bound
    from ravel.bounds.p_value import eb_p_value
    from ravel.bounds.weighted_stats import weighted_stats_01
    from ravel.bounds.holm import holm_reject

    all_results = []

    for sigma in sigmas:
        print(f"\n  sigma={sigma}:")

        for trial in range(n_trials):
            if trial % max(1, n_trials // 3) == 0:
                print(f"    Trial {trial+1}/{n_trials}...")

            trial_seed = seed + trial * 137
            rng = np.random.RandomState(trial_seed)

            n_cal = 2000
            n_cohorts = 5

            # Generate synthetic data
            gen = SyntheticShiftGenerator(
                n_cal=n_cal, n_test=5000, n_cohorts=n_cohorts,
                d_features=10, shift_severity=1.0,
                positive_rate=0.5, seed=trial_seed,
            )
            data = gen.generate(tau_grid=tau_grid)

            # Generate log-normal weights with controlled sigma
            log_w = rng.normal(0, sigma, size=n_cal)
            raw_weights = np.exp(log_w)
            raw_weights = raw_weights / raw_weights.mean()

            # Compute weight diagnostics
            w_n_eff = (raw_weights.sum() ** 2) / (raw_weights ** 2).sum()
            w_cv = raw_weights.std() / raw_weights.mean()
            w_ess_frac = w_n_eff / n_cal

            # Run each variant
            variants = {
                "naive_ungated": {"use_neff": False, "process": "none"},
                "naive_clipped": {"use_neff": False, "process": "clip"},
                "neff_ungated": {"use_neff": True, "process": "none"},
                "neff_ess_gated": {"use_neff": True, "process": "ess_gate"},
            }

            cohort_ids = np.unique(data.cohorts_cal)

            for vname, vconfig in variants.items():
                pvals = []
                test_info = []
                n_eff_list = []
                gated = 0

                for cid in cohort_ids:
                    cmask = data.cohorts_cal == cid
                    pmask = cmask & (data.preds_cal == 1)
                    y_pos = data.y_cal[pmask]
                    w_pos = raw_weights[pmask].copy()

                    if len(y_pos) < 2 or w_pos.sum() == 0:
                        for tau in tau_grid:
                            pvals.append(1.0)
                            test_info.append((cid, tau, np.nan, np.nan, np.nan))
                        continue

                    if vconfig["process"] == "clip":
                        w_pos = clip_weights(w_pos, 0.99)
                    elif vconfig["process"] == "ess_gate":
                        gate = apply_ess_gate(w_pos, 0.15)
                        if gate["gated"]:
                            gated += 1
                            for tau in tau_grid:
                                pvals.append(1.0)
                                test_info.append((cid, tau, np.nan, np.nan, np.nan))
                            continue
                        w_pos = gate["weights"]

                    stats = weighted_stats_01(y_pos, w_pos)
                    n_eff_list.append(stats.n_eff)

                    bound_n = stats.n_eff if vconfig["use_neff"] else float(len(y_pos))
                    for tau in tau_grid:
                        pval = eb_p_value(stats.mu, stats.var, bound_n, tau)
                        lb = eb_lower_bound(stats.mu, stats.var, bound_n, alpha)
                        pvals.append(pval)
                        test_info.append((cid, tau, stats.mu, bound_n, lb))

                # Holm
                n_certified = 0
                n_false = 0
                coverage_list = []

                if pvals:
                    rejected = holm_reject(pd.Series(pvals), alpha)
                    for i, (cid, tau, mu, neff, lb) in enumerate(test_info):
                        if rejected.iloc[i]:
                            n_certified += 1
                            true_ppv = data.true_ppv.get(cid, {}).get(tau, np.nan)
                            if not np.isnan(true_ppv) and true_ppv < tau:
                                n_false += 1
                            if not np.isnan(true_ppv) and not np.isnan(lb):
                                coverage_list.append(float(true_ppv >= lb))

                all_results.append({
                    "sweep": "h2a_tail",
                    "sigma": sigma,
                    "variant": vname,
                    "trial_id": trial,
                    "n_certified": n_certified,
                    "n_false": n_false,
                    "fwer": int(n_false > 0),
                    "cert_rate": n_certified / (len(cohort_ids) * len(tau_grid)),
                    "coverage": np.mean(coverage_list) if coverage_list else np.nan,
                    "gated_cohorts": gated,
                    "weight_sigma": sigma,
                    "weight_cv": w_cv,
                    "weight_ess_frac": w_ess_frac,
                    "mean_n_eff": np.mean(n_eff_list) if n_eff_list else np.nan,
                })

    return pd.DataFrame(all_results)


def run_h2b_ess_sweep(n_trials=200, alpha=0.05, seed=42):
    """H2-B: Well-behaved weights with varying ESS."""
    print("\n" + "=" * 70)
    print("H2-B: ESS SWEEP (well-behaved weights, varying CV)")
    print("=" * 70)

    tau_grid = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    # Use low sigma for well-behaved shape, but scale n_cal to vary ESS
    # Or: fix n_cal, vary sigma slightly (0.2-0.8) so weights are NOT pathological
    # but ESS drops from ~90% to ~20% of n
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    from ravel.bounds.empirical_bernstein import eb_lower_bound
    from ravel.bounds.p_value import eb_p_value
    from ravel.bounds.weighted_stats import weighted_stats_01
    from ravel.bounds.holm import holm_reject

    all_results = []

    for sigma in sigmas:
        print(f"\n  sigma={sigma}:")

        for trial in range(n_trials):
            if trial % max(1, n_trials // 3) == 0:
                print(f"    Trial {trial+1}/{n_trials}...")

            trial_seed = seed + trial * 137 + 50000
            rng = np.random.RandomState(trial_seed)

            n_cal = 2000
            n_cohorts = 5

            gen = SyntheticShiftGenerator(
                n_cal=n_cal, n_test=5000, n_cohorts=n_cohorts,
                d_features=10, shift_severity=1.0,
                positive_rate=0.5, seed=trial_seed,
            )
            data = gen.generate(tau_grid=tau_grid)

            # Log-normal weights (controlled, not pathological)
            log_w = rng.normal(0, sigma, size=n_cal)
            raw_weights = np.exp(log_w)
            raw_weights = raw_weights / raw_weights.mean()

            w_n_eff = (raw_weights.sum() ** 2) / (raw_weights ** 2).sum()
            w_ess_frac = w_n_eff / n_cal

            # Only run neff_ungated (the "correct" method) -- focus on power loss
            cohort_ids = np.unique(data.cohorts_cal)
            pvals = []
            test_info = []
            n_eff_list = []

            for cid in cohort_ids:
                cmask = data.cohorts_cal == cid
                pmask = cmask & (data.preds_cal == 1)
                y_pos = data.y_cal[pmask]
                w_pos = raw_weights[pmask].copy()

                if len(y_pos) < 2 or w_pos.sum() == 0:
                    for tau in tau_grid:
                        pvals.append(1.0)
                        test_info.append((cid, tau, np.nan, np.nan, np.nan))
                    continue

                stats = weighted_stats_01(y_pos, w_pos)
                n_eff_list.append(stats.n_eff)

                for tau in tau_grid:
                    pval = eb_p_value(stats.mu, stats.var, stats.n_eff, tau)
                    lb = eb_lower_bound(stats.mu, stats.var, stats.n_eff, alpha)
                    pvals.append(pval)
                    test_info.append((cid, tau, stats.mu, stats.n_eff, lb))

            n_certified = 0
            n_false = 0
            coverage_list = []

            if pvals:
                rejected = holm_reject(pd.Series(pvals), alpha)
                for i, (cid, tau, mu, neff, lb) in enumerate(test_info):
                    if rejected.iloc[i]:
                        n_certified += 1
                        true_ppv = data.true_ppv.get(cid, {}).get(tau, np.nan)
                        if not np.isnan(true_ppv) and true_ppv < tau:
                            n_false += 1
                        if not np.isnan(true_ppv) and not np.isnan(lb):
                            coverage_list.append(float(true_ppv >= lb))

            all_results.append({
                "sweep": "h2b_ess",
                "sigma": sigma,
                "variant": "neff_ungated",
                "trial_id": trial,
                "n_certified": n_certified,
                "n_false": n_false,
                "fwer": int(n_false > 0),
                "cert_rate": n_certified / (len(cohort_ids) * len(tau_grid)),
                "coverage": np.mean(coverage_list) if coverage_list else np.nan,
                "gated_cohorts": 0,
                "weight_sigma": sigma,
                "weight_cv": raw_weights.std() / raw_weights.mean(),
                "weight_ess_frac": w_ess_frac,
                "mean_n_eff": np.mean(n_eff_list) if n_eff_list else np.nan,
            })

    return pd.DataFrame(all_results)


def main():
    out_dir = shift_bench_dir / "results" / "h2_decomposed"
    os.makedirs(out_dir, exist_ok=True)

    h2a = run_h2a_tail_sweep(n_trials=200, alpha=0.05, seed=42)
    h2b = run_h2b_ess_sweep(n_trials=200, alpha=0.05, seed=42)

    combined = pd.concat([h2a, h2b], ignore_index=True)
    combined.to_csv(out_dir / "h2_decomposed_raw.csv", index=False)

    # H2-A summary
    print("\n" + "=" * 70)
    print("H2-A SUMMARY: FWER by tail heaviness (sigma)")
    print("=" * 70)
    h2a_summary = h2a.groupby(["sigma", "variant"]).agg(
        fwer_rate=("fwer", "mean"),
        mean_cert_rate=("cert_rate", "mean"),
        mean_coverage=("coverage", "mean"),
        mean_ess_frac=("weight_ess_frac", "mean"),
        n_trials=("trial_id", "count"),
    ).reset_index()
    h2a_summary.to_csv(out_dir / "h2a_tail_summary.csv", index=False)

    print(f"{'sigma':>6} {'Variant':<18} {'FWER':>7} {'Cert%':>7} {'Cov':>7} {'ESS%':>7}")
    print("-" * 56)
    for _, row in h2a_summary.sort_values(["sigma", "variant"]).iterrows():
        cov = f"{row['mean_coverage']:.3f}" if not np.isnan(row['mean_coverage']) else "N/A"
        print(f"{row['sigma']:>6.1f} {row['variant']:<18} {row['fwer_rate']:>6.1%} "
              f"{row['mean_cert_rate']:>6.1%} {cov:>7} {row['mean_ess_frac']:>6.1%}")

    # H2-B summary
    print("\n" + "=" * 70)
    print("H2-B SUMMARY: Power vs ESS (neff_ungated only)")
    print("=" * 70)
    h2b_summary = h2b.groupby("sigma").agg(
        fwer_rate=("fwer", "mean"),
        mean_cert_rate=("cert_rate", "mean"),
        mean_coverage=("coverage", "mean"),
        mean_ess_frac=("weight_ess_frac", "mean"),
        mean_neff=("mean_n_eff", "mean"),
        n_trials=("trial_id", "count"),
    ).reset_index()
    h2b_summary.to_csv(out_dir / "h2b_ess_summary.csv", index=False)

    print(f"{'sigma':>6} {'FWER':>7} {'Cert%':>7} {'Cov':>7} {'ESS%':>7} {'n_eff':>7}")
    print("-" * 44)
    for _, row in h2b_summary.iterrows():
        cov = f"{row['mean_coverage']:.3f}" if not np.isnan(row['mean_coverage']) else "N/A"
        print(f"{row['sigma']:>6.1f} {row['fwer_rate']:>6.1%} "
              f"{row['mean_cert_rate']:>6.1%} {cov:>7} "
              f"{row['mean_ess_frac']:>6.1%} {row['mean_neff']:>7.1f}")

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
