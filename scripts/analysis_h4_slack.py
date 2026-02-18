"""
H4 Slack Distribution Analysis
================================
Computes (LB - tau), (true_ppv - tau), and (LB - true_ppv) distributions
for certified pairs, stratified by n_eff bins.

Produces: results/h4_slack/h4_slack_summary.csv
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

from synthetic_shift_generator import SyntheticShiftGenerator
from ravel.bounds.empirical_bernstein import eb_lower_bound
from ravel.bounds.p_value import eb_p_value
from ravel.bounds.weighted_stats import weighted_stats_01
from ravel.bounds.holm import holm_reject


def run_h4_slack_analysis(
    n_trials: int = 200,
    alpha: float = 0.05,
    seed: int = 42,
):
    """Run H4 with detailed per-decision tracking for slack analysis."""
    tau_grid = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

    configs = [
        # (n_cal, n_test, n_cohorts, shift_sev, pos_rate, label)
        (500, 5000, 5, 0.5, 0.3, "easy_low_shift"),
        (500, 5000, 5, 1.0, 0.5, "moderate_shift"),
        (500, 5000, 10, 1.0, 0.5, "moderate_many_cohorts"),
        (500, 5000, 5, 2.0, 0.5, "severe_shift"),
        (2000, 5000, 5, 1.0, 0.5, "large_cal"),
        (2000, 5000, 10, 0.5, 0.7, "large_cal_easy"),
    ]

    all_decisions = []

    for n_cal, n_test, n_cohorts, shift_sev, pos_rate, label in configs:
        print(f"\nConfig: {label} (n_cal={n_cal}, cohorts={n_cohorts}, shift={shift_sev})")

        for trial in range(n_trials):
            if trial % max(1, n_trials // 5) == 0:
                print(f"  Trial {trial+1}/{n_trials}...")

            trial_seed = seed + trial * 137

            gen = SyntheticShiftGenerator(
                n_cal=n_cal, n_test=n_test, n_cohorts=n_cohorts,
                d_features=10, shift_severity=shift_sev,
                positive_rate=pos_rate, seed=trial_seed,
            )
            data = gen.generate(tau_grid=tau_grid)

            # uLSIF-style: uniform weights (for clean slack measurement)
            from shiftbench.baselines.ulsif import create_ulsif_baseline
            method = create_ulsif_baseline(alpha=alpha)

            try:
                method.estimate_weights(data.X_cal, data.X_test)
            except Exception:
                continue

            cohort_ids = np.unique(data.cohorts_cal)
            pvals = []
            test_info = []

            for cid in cohort_ids:
                cmask = data.cohorts_cal == cid
                pos_mask = cmask & (data.preds_cal == 1)
                y_pos = data.y_cal[pos_mask]

                if hasattr(method, 'weights_') and method.weights_ is not None:
                    w_pos = method.weights_[pos_mask]
                else:
                    w_pos = np.ones(pos_mask.sum())

                if len(y_pos) < 2 or w_pos.sum() == 0:
                    for tau in tau_grid:
                        pvals.append(1.0)
                        test_info.append((cid, tau, np.nan, np.nan, np.nan))
                    continue

                w_pos = w_pos / w_pos.mean() if w_pos.mean() > 0 else w_pos
                stats = weighted_stats_01(y_pos, w_pos)

                for tau in tau_grid:
                    pval = eb_p_value(stats.mu, stats.var, stats.n_eff, tau)
                    lb = eb_lower_bound(stats.mu, stats.var, stats.n_eff, alpha)
                    pvals.append(pval)
                    test_info.append((cid, tau, stats.mu, stats.n_eff, lb))

            if not pvals:
                continue

            pvals_s = pd.Series(pvals)
            rejected = holm_reject(pvals_s, alpha)

            for i, (cid, tau, mu_hat, n_eff, lb) in enumerate(test_info):
                true_ppv = data.true_ppv.get(cid, {}).get(tau, np.nan)
                certified = bool(rejected.iloc[i]) if i < len(rejected) else False

                all_decisions.append({
                    "config": label,
                    "trial_id": trial,
                    "cohort_id": cid,
                    "tau": tau,
                    "certified": certified,
                    "mu_hat": mu_hat,
                    "n_eff": n_eff,
                    "lower_bound": lb,
                    "true_ppv": true_ppv,
                    "lb_minus_tau": lb - tau if not np.isnan(lb) else np.nan,
                    "true_ppv_minus_tau": true_ppv - tau if not np.isnan(true_ppv) else np.nan,
                    "slack": (lb - true_ppv) if (not np.isnan(lb) and not np.isnan(true_ppv)) else np.nan,
                })

    df = pd.DataFrame(all_decisions)

    # Output directory
    out_dir = shift_bench_dir / "results" / "h4_slack"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_dir / "h4_slack_raw.csv", index=False)
    print(f"\nSaved raw: {out_dir / 'h4_slack_raw.csv'} ({len(df)} rows)")

    # === Analysis: certified decisions only ===
    cert = df[df["certified"] == True].copy()
    print(f"\nCertified decisions: {len(cert)}")

    if len(cert) == 0:
        print("No certified decisions -- cannot compute slack.")
        return df

    # n_eff bins
    cert["neff_bin"] = pd.cut(
        cert["n_eff"],
        bins=[0, 5, 25, 100, 300, float("inf")],
        labels=["1-5", "5-25", "25-100", "100-300", "300+"],
    )

    # Slack summary by n_eff bin
    slack_summary = cert.groupby("neff_bin", observed=True).agg(
        count=("slack", "size"),
        mean_lb_minus_tau=("lb_minus_tau", "mean"),
        std_lb_minus_tau=("lb_minus_tau", "std"),
        mean_true_ppv_minus_tau=("true_ppv_minus_tau", "mean"),
        std_true_ppv_minus_tau=("true_ppv_minus_tau", "std"),
        mean_slack=("slack", "mean"),
        std_slack=("slack", "std"),
        median_slack=("slack", "median"),
        mean_n_eff=("n_eff", "mean"),
        mean_mu_hat=("mu_hat", "mean"),
        mean_lb=("lower_bound", "mean"),
        mean_true_ppv=("true_ppv", "mean"),
        false_cert_rate=("true_ppv_minus_tau", lambda x: (x < 0).mean()),
    ).reset_index()

    slack_summary.to_csv(out_dir / "h4_slack_by_neff.csv", index=False)
    print(f"\nSaved: {out_dir / 'h4_slack_by_neff.csv'}")

    # Slack summary by config
    config_summary = cert.groupby("config").agg(
        count=("slack", "size"),
        mean_lb_minus_tau=("lb_minus_tau", "mean"),
        mean_true_ppv_minus_tau=("true_ppv_minus_tau", "mean"),
        mean_slack=("slack", "mean"),
        median_slack=("slack", "median"),
        mean_n_eff=("n_eff", "mean"),
        false_cert_rate=("true_ppv_minus_tau", lambda x: (x < 0).mean()),
    ).reset_index()

    config_summary.to_csv(out_dir / "h4_slack_by_config.csv", index=False)
    print(f"Saved: {out_dir / 'h4_slack_by_config.csv'}")

    # Print summary
    print(f"\n{'='*70}")
    print("H4 SLACK DISTRIBUTIONS (certified decisions only)")
    print(f"{'='*70}")

    print(f"\n--- By n_eff bin ---")
    print(f"{'n_eff bin':<12} {'N':>6} {'LB-tau':>10} {'PPV-tau':>10} {'Slack':>10} {'FalseCert':>10}")
    print("-" * 60)
    for _, row in slack_summary.iterrows():
        print(f"{row['neff_bin']:<12} {row['count']:>6.0f} "
              f"{row['mean_lb_minus_tau']:>10.4f} "
              f"{row['mean_true_ppv_minus_tau']:>10.4f} "
              f"{row['mean_slack']:>10.4f} "
              f"{row['false_cert_rate']:>10.4f}")

    print(f"\n--- By config ---")
    for _, row in config_summary.iterrows():
        print(f"  {row['config']}: N={row['count']:.0f}, "
              f"mean_slack={row['mean_slack']:.4f}, "
              f"false_cert={row['false_cert_rate']:.4f}")

    print(f"\n{'='*70}")
    return df


if __name__ == "__main__":
    run_h4_slack_analysis(n_trials=200, alpha=0.05, seed=42)
