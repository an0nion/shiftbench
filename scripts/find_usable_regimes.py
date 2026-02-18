"""
Non-Degenerate Regime Search
=============================

Sweeps (n_cal, n_cohorts, shift_severity, positive_rate) to find at least
ONE regime with certification rate >= 20% while maintaining FWER <= alpha.

This uses the full synthetic pipeline (data generation + importance weights
+ EB bounds + Holm correction) to measure end-to-end certification rates
including all sources of conservatism.

Goal: Find practitioner guidance for when the method is useful.
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add project paths
shift_bench_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(shift_bench_dir / "src"))
sys.path.insert(0, str(shift_bench_dir.parent / "ravel" / "src"))

# Import synthetic generator from same scripts directory
scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(scripts_dir))
from synthetic_shift_generator import SyntheticShiftGenerator


def run_regime_trial(
    n_cal: int,
    n_test: int,
    n_cohorts: int,
    shift_severity: float,
    positive_rate: float,
    alpha: float,
    tau_grid: np.ndarray,
    trial_id: int,
    seed: int,
) -> dict:
    """
    Run a single trial of the full synthetic pipeline.

    Returns: dict with certification stats for this trial.
    """
    from ravel.bounds.empirical_bernstein import eb_lower_bound
    from ravel.bounds.weighted_stats import weighted_stats_01
    from ravel.bounds.p_value import eb_p_value
    from ravel.bounds.holm import holm_reject

    # Generate synthetic data
    gen = SyntheticShiftGenerator(
        n_cal=n_cal,
        n_test=n_test,
        n_cohorts=n_cohorts,
        d_features=10,
        shift_severity=shift_severity,
        positive_rate=positive_rate,
        seed=seed,
    )
    data = gen.generate(tau_grid=tau_grid)

    # Estimate importance weights using uLSIF
    try:
        from ravel.weights.ulsif import uLSIF
        ulsif = uLSIF()
        weights = ulsif.fit_predict(data.X_cal, data.X_test)
        # Clip negative weights
        weights = np.maximum(weights, 0.0)
        # Normalize
        if weights.sum() > 0:
            weights = weights / weights.mean()
        else:
            weights = np.ones(len(data.X_cal))
    except Exception:
        weights = np.ones(len(data.X_cal))

    # Evaluate each (cohort, tau) pair
    n_certified = 0
    n_abstain = 0
    n_false_certify = 0
    n_tests = 0
    n_eff_list = []

    cohort_ids = np.unique(data.cohorts_cal)
    pvals = []
    test_info = []

    # Compute p-values for all (cohort, tau) pairs
    for cohort_id in cohort_ids:
        cohort_mask = data.cohorts_cal == cohort_id
        pos_mask = cohort_mask & (data.preds_cal == 1)

        y_pos = data.y_cal[pos_mask]
        w_pos = weights[pos_mask]

        if len(y_pos) < 2 or w_pos.sum() == 0:
            for tau in tau_grid:
                pvals.append(1.0)
                test_info.append((cohort_id, tau, np.nan, np.nan, 0.0))
            continue

        # Weighted stats
        result = weighted_stats_01(y_pos, w_pos)
        mu_hat = result.mu
        var_hat = result.var
        n_eff = result.n_eff

        n_eff_list.append(n_eff)

        for tau in tau_grid:
            pval = eb_p_value(mu_hat, var_hat, n_eff, tau)
            pvals.append(pval)
            test_info.append((cohort_id, tau, mu_hat, n_eff, pval))

    # Holm step-down
    if len(pvals) > 0:
        pvals_series = pd.Series(pvals)
        rejected = holm_reject(pvals_series, alpha)

        for i, (cohort_id, tau, mu_hat, n_eff, pval) in enumerate(test_info):
            n_tests += 1
            if rejected[i]:
                n_certified += 1
                # Check false certification
                true_ppv = data.true_ppv.get(cohort_id, {}).get(tau, np.nan)
                if not np.isnan(true_ppv) and true_ppv < tau:
                    n_false_certify += 1
            else:
                n_abstain += 1

    cert_rate = n_certified / n_tests if n_tests > 0 else 0.0
    false_certify_fwer = int(n_false_certify > 0)
    mean_n_eff = np.mean(n_eff_list) if n_eff_list else np.nan

    return {
        "trial_id": trial_id,
        "n_cal": n_cal,
        "n_test": n_test,
        "n_cohorts": n_cohorts,
        "shift_severity": shift_severity,
        "positive_rate": positive_rate,
        "alpha": alpha,
        "n_tests": n_tests,
        "n_certified": n_certified,
        "n_abstain": n_abstain,
        "cert_rate": cert_rate,
        "n_false_certify": n_false_certify,
        "false_certify_fwer": false_certify_fwer,
        "mean_n_eff": mean_n_eff,
    }


def run_regime_search(
    n_cal_grid: list,
    n_cohorts_grid: list,
    shift_grid: list,
    positive_rate_grid: list,
    alpha: float = 0.05,
    n_trials: int = 20,
    n_test: int = 5000,
    tau_grid: np.ndarray = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Search over parameter grid to find usable regimes.
    """
    if tau_grid is None:
        tau_grid = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

    combos = [
        (nc, ncoh, ss, pr)
        for nc in n_cal_grid
        for ncoh in n_cohorts_grid
        for ss in shift_grid
        for pr in positive_rate_grid
    ]

    total_combos = len(combos)
    total_trials = total_combos * n_trials
    print(f"Regime search: {total_combos} combinations x {n_trials} trials = {total_trials} total")

    all_results = []
    done = 0

    for combo_idx, (n_cal, n_cohorts, shift_sev, pos_rate) in enumerate(combos):
        combo_results = []
        for trial_id in range(n_trials):
            trial_seed = seed + combo_idx * 1000 + trial_id
            result = run_regime_trial(
                n_cal=n_cal,
                n_test=n_test,
                n_cohorts=n_cohorts,
                shift_severity=shift_sev,
                positive_rate=pos_rate,
                alpha=alpha,
                tau_grid=tau_grid,
                trial_id=trial_id,
                seed=trial_seed,
            )
            combo_results.append(result)
            done += 1

        all_results.extend(combo_results)

        # Progress report per combo
        combo_df = pd.DataFrame(combo_results)
        mean_cert = combo_df["cert_rate"].mean()
        max_fc = combo_df["false_certify_fwer"].max()
        mean_neff = combo_df["mean_n_eff"].mean()
        print(f"  [{combo_idx+1}/{total_combos}] n_cal={n_cal}, n_coh={n_cohorts}, "
              f"shift={shift_sev:.1f}, pos={pos_rate:.1f} => "
              f"cert_rate={mean_cert:.3f}, FWER_max={max_fc}, n_eff={mean_neff:.1f}")

    return pd.DataFrame(all_results)


def aggregate_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trial results into per-regime summary."""
    grouped = df.groupby(["n_cal", "n_cohorts", "shift_severity", "positive_rate"]).agg(
        mean_cert_rate=("cert_rate", "mean"),
        std_cert_rate=("cert_rate", "std"),
        max_cert_rate=("cert_rate", "max"),
        min_cert_rate=("cert_rate", "min"),
        fwer_violations=("false_certify_fwer", "sum"),
        n_trials=("trial_id", "count"),
        mean_n_eff=("mean_n_eff", "mean"),
        total_certified=("n_certified", "sum"),
        total_tests=("n_tests", "sum"),
    ).reset_index()

    grouped["observed_fwer"] = grouped["fwer_violations"] / grouped["n_trials"]
    grouped = grouped.sort_values("mean_cert_rate", ascending=False)

    return grouped


def summarize_regimes(agg: pd.DataFrame, target_cert: float = 0.20) -> str:
    """Generate human-readable summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("NON-DEGENERATE REGIME SEARCH RESULTS")
    lines.append("=" * 70)

    lines.append(f"\nTotal regimes tested: {len(agg)}")
    lines.append(f"Target certification rate: >= {target_cert:.0%}")

    # FWER check
    any_fwer = agg[agg["fwer_violations"] > 0]
    if len(any_fwer) > 0:
        lines.append(f"\nFWER VIOLATIONS: {len(any_fwer)} regimes had >= 1 false certification")
        for _, row in any_fwer.iterrows():
            lines.append(f"  n_cal={row['n_cal']}, n_coh={row['n_cohorts']}, "
                          f"shift={row['shift_severity']:.1f}, pos={row['positive_rate']:.1f}: "
                          f"observed FWER={row['observed_fwer']:.3f}")
    else:
        lines.append("\nFWER VALIDATION: No violations (0 false certifications across all regimes)")

    # Usable regimes
    usable = agg[agg["mean_cert_rate"] >= target_cert]
    if len(usable) > 0:
        lines.append(f"\nUSABLE REGIMES (cert >= {target_cert:.0%}): Found {len(usable)}")
        lines.append(f"\n  {'n_cal':>6} {'n_coh':>5} {'shift':>5} {'pos':>5} "
                      f"{'cert%':>6} {'n_eff':>6} {'FWER':>5}")
        lines.append("  " + "-" * 50)
        for _, row in usable.head(20).iterrows():
            lines.append(f"  {row['n_cal']:>6.0f} {row['n_cohorts']:>5.0f} "
                          f"{row['shift_severity']:>5.1f} {row['positive_rate']:>5.1f} "
                          f"{row['mean_cert_rate']:>5.1%} {row['mean_n_eff']:>6.1f} "
                          f"{row['observed_fwer']:>5.3f}")

        best = usable.iloc[0]
        lines.append(f"\n  BEST REGIME: n_cal={best['n_cal']:.0f}, "
                      f"n_cohorts={best['n_cohorts']:.0f}, "
                      f"shift={best['shift_severity']:.1f}, "
                      f"pos={best['positive_rate']:.1f} "
                      f"=> cert_rate={best['mean_cert_rate']:.1%}")
    else:
        lines.append(f"\nWARNING: No regimes reached {target_cert:.0%} certification!")
        lines.append("Method may be too conservative for practical use.")
        # Show best anyway
        if len(agg) > 0:
            best = agg.iloc[0]
            lines.append(f"\nBest available: n_cal={best['n_cal']:.0f}, "
                          f"n_cohorts={best['n_cohorts']:.0f}, "
                          f"shift={best['shift_severity']:.1f} "
                          f"=> cert_rate={best['mean_cert_rate']:.1%}")

    # Practitioner guidance
    lines.append("\nPRACTITIONER GUIDANCE:")
    for n_cal in sorted(agg["n_cal"].unique()):
        sub = agg[agg["n_cal"] == n_cal]
        mean_cert = sub["mean_cert_rate"].mean()
        max_cert = sub["mean_cert_rate"].max()
        lines.append(f"  n_cal={n_cal:>5}: mean cert={mean_cert:.1%}, max cert={max_cert:.1%}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Find usable certification regimes")
    parser.add_argument("--output", default="results/regime_search",
                        help="Output directory")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Trials per parameter combination")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer combos for testing")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.quick:
        n_cal_grid = [500, 2000]
        n_cohorts_grid = [5, 10]
        shift_grid = [0.5, 1.0]
        positive_rate_grid = [0.5]
        n_trials = 10
    else:
        n_cal_grid = [500, 1000, 2000, 5000]
        n_cohorts_grid = [3, 5, 10, 20]
        shift_grid = [0.3, 0.5, 0.8, 1.0]
        positive_rate_grid = [0.3, 0.5, 0.7]
        n_trials = args.n_trials

    print(f"Parameter grid:")
    print(f"  n_cal: {n_cal_grid}")
    print(f"  n_cohorts: {n_cohorts_grid}")
    print(f"  shift_severity: {shift_grid}")
    print(f"  positive_rate: {positive_rate_grid}")
    print(f"  trials per combo: {n_trials}")
    print()

    # Run search
    df = run_regime_search(
        n_cal_grid=n_cal_grid,
        n_cohorts_grid=n_cohorts_grid,
        shift_grid=shift_grid,
        positive_rate_grid=positive_rate_grid,
        alpha=args.alpha,
        n_trials=n_trials,
        seed=args.seed,
    )

    # Save raw results
    raw_path = os.path.join(args.output, "regime_search_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"\nSaved raw results: {raw_path}")

    # Aggregate
    agg = aggregate_regimes(df)
    agg_path = os.path.join(args.output, "regime_search_summary.csv")
    agg.to_csv(agg_path, index=False)
    print(f"Saved aggregated results: {agg_path}")

    # Summary
    summary = summarize_regimes(agg)
    print("\n" + summary)

    summary_path = os.path.join(args.output, "regime_search_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"\nSaved summary: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
