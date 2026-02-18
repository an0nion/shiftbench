"""
Power Analysis for EB-Based Certification Protocol
===================================================

Measures P(CERTIFY | true_ppv, n_eff, tau) to prove non-degeneracy.

Two modes:
  1. DIRECT: Sample from Bernoulli(true_ppv) with controlled n_eff,
     compute EB lower bound, check if LB >= tau. This measures
     intrinsic power of the EB bound (no weight estimation noise).

  2. PIPELINE: Run full synthetic pipeline with importance weights,
     cohort subsetting, Holm correction. This measures end-to-end power
     including all sources of conservatism.

Output:
  - Power curves: P(CERTIFY) vs true_ppv for each (n_eff, tau)
  - Sample size requirements: n_eff needed for 50% power at each margin
  - False-certify validation: Confirms FWER <= alpha throughout
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from math import log, sqrt

# Add ravel to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ravel" / "src"))

from ravel.bounds.empirical_bernstein import eb_lower_bound


def compute_direct_power(
    true_ppv: float,
    n_eff: float,
    tau: float,
    alpha: float = 0.05,
    n_trials: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Compute P(CERTIFY | true_ppv, n_eff, tau) using direct EB bound.

    Samples n_eff observations from Bernoulli(true_ppv) with unit weights,
    computes EB lower bound at level alpha, checks if LB >= tau.

    This measures INTRINSIC power of the EB bound method,
    without weight estimation noise or Holm correction overhead.

    Returns dict with: power, false_certify_rate, mean_lb, mean_mu_hat
    """
    rng = np.random.RandomState(seed)

    n_certify = 0
    n_false_certify = 0
    lb_list = []
    mu_hat_list = []

    n_int = max(int(round(n_eff)), 2)

    for _ in range(n_trials):
        # Sample from Bernoulli(true_ppv)
        y = rng.binomial(1, true_ppv, size=n_int).astype(float)

        # Compute weighted stats (unit weights)
        mu_hat = y.mean()
        if n_int > 1:
            var_hat = y.var(ddof=1)
        else:
            var_hat = 0.0

        # EB lower bound
        lb = eb_lower_bound(mu_hat, var_hat, float(n_int), alpha)

        if not np.isnan(lb):
            lb_list.append(lb)
            mu_hat_list.append(mu_hat)

            if lb >= tau:
                n_certify += 1
                # Check false certification
                if true_ppv < tau:
                    n_false_certify += 1

    power = n_certify / n_trials if n_trials > 0 else 0.0
    false_certify_rate = n_false_certify / n_trials if n_trials > 0 else 0.0

    return {
        "true_ppv": true_ppv,
        "n_eff": n_eff,
        "tau": tau,
        "alpha": alpha,
        "power": power,
        "n_certify": n_certify,
        "n_trials": n_trials,
        "false_certify_rate": false_certify_rate,
        "mean_lb": np.mean(lb_list) if lb_list else np.nan,
        "mean_mu_hat": np.mean(mu_hat_list) if mu_hat_list else np.nan,
    }


def run_power_grid(
    true_ppv_grid: np.ndarray,
    n_eff_grid: np.ndarray,
    tau_grid: np.ndarray,
    alpha: float = 0.05,
    n_trials: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run power analysis over full (true_ppv, n_eff, tau) grid.

    Returns DataFrame with power for each combination.
    """
    results = []
    total = len(true_ppv_grid) * len(n_eff_grid) * len(tau_grid)
    done = 0

    for tau in tau_grid:
        for n_eff in n_eff_grid:
            for true_ppv in true_ppv_grid:
                result = compute_direct_power(
                    true_ppv=true_ppv,
                    n_eff=n_eff,
                    tau=tau,
                    alpha=alpha,
                    n_trials=n_trials,
                    seed=seed + done,
                )
                results.append(result)
                done += 1
                if done % 20 == 0:
                    print(f"  Progress: {done}/{total} ({100*done/total:.0f}%)")

    return pd.DataFrame(results)


def find_sample_size_requirements(
    df: pd.DataFrame,
    target_power: float = 0.50,
) -> pd.DataFrame:
    """
    For each (tau, margin), find minimum n_eff for target power.

    margin = true_ppv - tau
    """
    rows = []
    for tau in sorted(df["tau"].unique()):
        for true_ppv in sorted(df["true_ppv"].unique()):
            margin = true_ppv - tau
            if margin <= 0:
                continue

            subset = df[(df["tau"] == tau) & (df["true_ppv"] == true_ppv)]
            subset = subset.sort_values("n_eff")

            # Find smallest n_eff with power >= target
            above = subset[subset["power"] >= target_power]
            if len(above) > 0:
                min_n_eff = above["n_eff"].min()
            else:
                min_n_eff = np.nan

            rows.append({
                "tau": tau,
                "true_ppv": true_ppv,
                "margin": round(margin, 3),
                "min_n_eff_for_target_power": min_n_eff,
                "target_power": target_power,
            })

    return pd.DataFrame(rows)


def summarize_results(df: pd.DataFrame) -> str:
    """Generate human-readable summary of power analysis results."""
    lines = []
    lines.append("=" * 70)
    lines.append("POWER ANALYSIS SUMMARY")
    lines.append("=" * 70)

    # Overall
    lines.append(f"\nTotal experiments: {len(df)}")
    lines.append(f"Alpha: {df['alpha'].iloc[0]}")
    lines.append(f"Trials per experiment: {df['n_trials'].iloc[0]}")

    # False-certify validation
    fc = df[df["true_ppv"] < df["tau"]]
    if len(fc) > 0:
        max_fc = fc["false_certify_rate"].max()
        mean_fc = fc["false_certify_rate"].mean()
        lines.append(f"\nFALSE-CERTIFY VALIDATION (true_ppv < tau):")
        lines.append(f"  Max false-certify rate: {max_fc:.4f}")
        lines.append(f"  Mean false-certify rate: {mean_fc:.4f}")
        if max_fc <= df["alpha"].iloc[0]:
            lines.append("  STATUS: PASSED (max <= alpha)")
        else:
            lines.append("  STATUS: FAILED (max > alpha) !!!")
    else:
        lines.append("\nFALSE-CERTIFY: No experiments with true_ppv < tau")

    # Power by tau
    lines.append("\nPOWER BY TAU (averaged over n_eff, true_ppv > tau):")
    for tau in sorted(df["tau"].unique()):
        above = df[(df["tau"] == tau) & (df["true_ppv"] > tau)]
        if len(above) > 0:
            mean_p = above["power"].mean()
            max_p = above["power"].max()
            lines.append(f"  tau={tau:.1f}: mean power={mean_p:.3f}, max power={max_p:.3f}")

    # Power by n_eff (for tau=0.8, true_ppv=0.9 as reference)
    lines.append("\nPOWER BY N_EFF (tau=0.8, true_ppv=0.9):")
    ref = df[(df["tau"] == 0.8) & (np.abs(df["true_ppv"] - 0.9) < 0.01)]
    if len(ref) > 0:
        for _, row in ref.sort_values("n_eff").iterrows():
            lines.append(f"  n_eff={row['n_eff']:>6.0f}: power={row['power']:.3f}")

    # Non-degenerate regime check
    lines.append("\nNON-DEGENERATE REGIME CHECK (power >= 0.20):")
    nondegen = df[(df["power"] >= 0.20) & (df["true_ppv"] > df["tau"])]
    if len(nondegen) > 0:
        lines.append(f"  Found {len(nondegen)} experiments with power >= 20%")
        best = nondegen.loc[nondegen["power"].idxmax()]
        lines.append(f"  Best: tau={best['tau']:.1f}, true_ppv={best['true_ppv']:.2f}, "
                      f"n_eff={best['n_eff']:.0f}, power={best['power']:.3f}")
        # Min n_eff for 20% power
        for tau in sorted(nondegen["tau"].unique()):
            t_sub = nondegen[nondegen["tau"] == tau]
            min_n = t_sub["n_eff"].min()
            lines.append(f"  tau={tau:.1f}: min n_eff for 20% power = {min_n:.0f}")
    else:
        lines.append("  WARNING: No experiments reached 20% power!")
        lines.append("  Method may be too conservative for practical use.")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Power analysis for EB certification")
    parser.add_argument("--output", default="results/power_analysis",
                        help="Output directory")
    parser.add_argument("--n-trials", type=int, default=2000,
                        help="Trials per (true_ppv, n_eff, tau) combination")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="FWER significance level")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Define grids
    true_ppv_grid = np.round(np.linspace(0.50, 1.00, 11), 2)  # 11 points: 0.50 to 1.00
    n_eff_grid = np.array([5, 10, 15, 25, 50, 75, 100, 150, 200, 500])  # 10 points
    tau_grid = np.array([0.5, 0.6, 0.7, 0.8, 0.9])  # 5 thresholds

    total = len(true_ppv_grid) * len(n_eff_grid) * len(tau_grid)
    print(f"Running power analysis: {total} experiments")
    print(f"  true_ppv: {true_ppv_grid}")
    print(f"  n_eff: {n_eff_grid}")
    print(f"  tau: {tau_grid}")
    print(f"  trials per experiment: {args.n_trials}")
    print(f"  alpha: {args.alpha}")
    print()

    # Run power grid
    df = run_power_grid(
        true_ppv_grid=true_ppv_grid,
        n_eff_grid=n_eff_grid,
        tau_grid=tau_grid,
        alpha=args.alpha,
        n_trials=args.n_trials,
        seed=args.seed,
    )

    # Save raw results
    raw_path = os.path.join(args.output, "power_analysis_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"\nSaved raw results: {raw_path}")

    # Sample size requirements
    req_df = find_sample_size_requirements(df, target_power=0.50)
    req_path = os.path.join(args.output, "sample_size_requirements.csv")
    req_df.to_csv(req_path, index=False)
    print(f"Saved sample size requirements: {req_path}")

    # Summary
    summary = summarize_results(df)
    print("\n" + summary)

    summary_path = os.path.join(args.output, "power_analysis_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"\nSaved summary: {summary_path}")

    # Quick pivot table for reference
    for tau in tau_grid:
        pivot_path = os.path.join(args.output, f"power_pivot_tau_{tau:.1f}.csv")
        subset = df[df["tau"] == tau].pivot_table(
            index="true_ppv", columns="n_eff", values="power"
        )
        subset.to_csv(pivot_path)

    print(f"\nSaved pivot tables to {args.output}/power_pivot_tau_*.csv")
    print("Done.")


if __name__ == "__main__":
    main()
