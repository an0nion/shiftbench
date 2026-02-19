"""
H4 Real-Data Validation + Bootstrap Comparison
=================================================
Two experiments:
  1. Semi-synthetic real-data FWER validation:
     Use real datasets (features, cohorts) with synthetic labels
     where true_ppv is known. Check FWER <= alpha.

  2. Bootstrap vs EB width comparison (P4.3):
     Compare EB lower bound width to bootstrap percentile width.
     EB should be wider >= 70% of the time (more conservative).
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

from experiment_a_real_data_calibration import load_real_dataset, DATASET_CONFIG
from ravel.bounds.empirical_bernstein import eb_lower_bound
from ravel.bounds.p_value import eb_p_value
from ravel.bounds.weighted_stats import weighted_stats_01
from ravel.bounds.holm import holm_reject
from shiftbench.baselines.ulsif import uLSIFBaseline


# =============================================================================
# PART 1: SEMI-SYNTHETIC REAL-DATA FWER VALIDATION
# =============================================================================

def inject_synthetic_labels(data, true_ppv_map, seed):
    """Replace real labels with synthetic labels where PPV is known.

    true_ppv_map: {cohort_id: true_ppv} - P(Y=1 | pred=1, cohort=c)
    Only modifies labels for predicted-positive samples.
    """
    rng = np.random.RandomState(seed)
    y_synthetic = data["y_source"].copy()
    preds = data["preds_source"]
    cohorts = data["cohorts_source"]

    for cid, ppv in true_ppv_map.items():
        mask = (cohorts == cid) & (preds == 1)
        n = mask.sum()
        if n > 0:
            y_synthetic[mask] = rng.binomial(1, ppv, size=n)

    return y_synthetic


def run_semi_synthetic_trial(data, weights, tau_grid, alpha, true_ppv_map, seed):
    """Run certification on real features/cohorts with synthetic labels."""
    y_cal = inject_synthetic_labels(data, true_ppv_map, seed)
    preds_cal = data["preds_source"]
    cohorts_cal = data["cohorts_source"]

    cohort_ids = np.unique(cohorts_cal)
    pvals = []
    test_info = []

    for cid in cohort_ids:
        cmask = cohorts_cal == cid
        pmask = cmask & (preds_cal == 1)
        y_pos = y_cal[pmask]
        w_pos = weights[pmask].copy()

        if len(y_pos) < 2 or w_pos.sum() == 0:
            for tau in tau_grid:
                pvals.append(1.0)
                test_info.append((cid, tau, np.nan, np.nan))
            continue

        w_pos = w_pos / w_pos.mean() if w_pos.mean() > 0 else w_pos
        stats = weighted_stats_01(y_pos, w_pos)

        for tau in tau_grid:
            pval = eb_p_value(stats.mu, stats.var, stats.n_eff, tau)
            pvals.append(pval)
            test_info.append((cid, tau, stats.mu, stats.n_eff))

    n_certified = 0
    n_false = 0

    if pvals:
        rejected = holm_reject(pd.Series(pvals), alpha)
        for i, (cid, tau, mu, neff) in enumerate(test_info):
            if rejected.iloc[i]:
                n_certified += 1
                true_ppv = true_ppv_map.get(cid, 1.0)  # default to safe
                if true_ppv < tau:
                    n_false += 1

    return {
        "n_certified": n_certified,
        "n_false": n_false,
        "fwer": int(n_false > 0),
    }


def run_real_data_fwer_validation(n_trials=200, alpha=0.05, seed=42):
    """Validate FWER on real datasets with injected null labels."""
    print("=" * 70)
    print("H4: SEMI-SYNTHETIC REAL-DATA FWER VALIDATION")
    print("=" * 70)

    data_dir = str(shift_bench_dir / "data" / "processed")
    model_dir = str(shift_bench_dir / "models")

    tau_grid = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    datasets = ["adult", "compas", "imdb", "yelp", "bace", "bbbp"]

    # Null scenario: true PPV = tau - 0.05 for all cohorts (below threshold)
    null_offsets = [0.02, 0.05, 0.10]

    all_results = []

    for ds_name in datasets:
        try:
            data = load_real_dataset(ds_name, data_dir, model_dir)
        except Exception as e:
            print(f"  [SKIP] {ds_name}: {e}")
            continue

        # Estimate weights once (real features, real shift)
        n_basis = min(100, len(data["X_source"]))
        try:
            ulsif = uLSIFBaseline(n_basis=n_basis, sigma=None, lambda_=0.1,
                                   random_state=seed)
            weights = ulsif.estimate_weights(data["X_source"], data["X_test"])
        except Exception:
            weights = np.ones(len(data["X_source"]))

        cohort_ids = np.unique(data["cohorts_source"])

        for offset in null_offsets:
            # All cohorts have true PPV = tau - offset (null for all tau)
            # Use highest tau to make it hardest
            base_ppv = max(tau_grid) - offset

            true_ppv_map = {cid: base_ppv for cid in cohort_ids}

            print(f"\n  {ds_name}, offset={offset} (true_ppv={base_ppv:.2f}):")

            n_false_trials = 0
            for trial in range(n_trials):
                if trial % max(1, n_trials // 3) == 0:
                    print(f"    Trial {trial+1}/{n_trials}...")

                trial_seed = seed + trial * 137
                result = run_semi_synthetic_trial(
                    data, weights, tau_grid, alpha, true_ppv_map, trial_seed
                )
                result["dataset"] = ds_name
                result["domain"] = data["config"]["domain"]
                result["null_offset"] = offset
                result["true_ppv"] = base_ppv
                result["trial_id"] = trial
                all_results.append(result)

                n_false_trials += result["fwer"]

            obs_fwer = n_false_trials / n_trials
            print(f"      FWER={obs_fwer:.3f} ({n_false_trials}/{n_trials})")

    return pd.DataFrame(all_results)


# =============================================================================
# PART 2: BOOTSTRAP vs EB WIDTH COMPARISON (P4.3)
# =============================================================================

def bootstrap_lower_bound_width(y, w, alpha, n_boot=2000, seed=42):
    """Compute bootstrap percentile lower bound and its width from point estimate."""
    rng = np.random.RandomState(seed)
    n = len(y)
    if n < 2:
        return np.nan, np.nan

    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        wb = w[idx]
        if wb.sum() > 0:
            boot_means[b] = np.average(y[idx], weights=wb)
        else:
            boot_means[b] = np.nan

    boot_means = boot_means[~np.isnan(boot_means)]
    if len(boot_means) < 100:
        return np.nan, np.nan

    lb = np.percentile(boot_means, alpha * 100)
    point = np.average(y, weights=w)
    width = point - lb  # How far below the point estimate
    return lb, width


def run_bootstrap_comparison(n_trials=100, alpha=0.05, seed=42):
    """Compare EB bound width vs bootstrap percentile width."""
    print("\n" + "=" * 70)
    print("H4: BOOTSTRAP vs EB WIDTH COMPARISON (P4.3)")
    print("=" * 70)

    data_dir = str(shift_bench_dir / "data" / "processed")
    model_dir = str(shift_bench_dir / "models")

    datasets = ["adult", "compas", "imdb", "yelp", "bace", "bbbp"]
    tau_grid = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

    all_comparisons = []

    for ds_name in datasets:
        try:
            data = load_real_dataset(ds_name, data_dir, model_dir)
        except Exception as e:
            print(f"  [SKIP] {ds_name}: {e}")
            continue

        # Estimate weights
        n_basis = min(100, len(data["X_source"]))
        try:
            ulsif = uLSIFBaseline(n_basis=n_basis, sigma=None, lambda_=0.1,
                                   random_state=seed)
            weights = ulsif.estimate_weights(data["X_source"], data["X_test"])
        except Exception:
            weights = np.ones(len(data["X_source"]))

        cohort_ids = np.unique(data["cohorts_source"])
        preds_cal = data["preds_source"]
        y_cal = data["y_source"]
        cohorts_cal = data["cohorts_source"]

        print(f"\n  {ds_name}:")

        for trial in range(n_trials):
            trial_seed = seed + trial * 137
            rng = np.random.RandomState(trial_seed)

            # Subsample for bootstrap variability
            n = len(y_cal)
            idx = rng.choice(n, size=n, replace=True)

            for cid in cohort_ids:
                cmask = cohorts_cal[idx] == cid
                pmask = cmask & (preds_cal[idx] == 1)
                y_pos = y_cal[idx][pmask]
                w_pos = weights[idx][pmask]

                if len(y_pos) < 5 or w_pos.sum() == 0:
                    continue

                w_pos = w_pos / w_pos.mean()
                stats = weighted_stats_01(y_pos, w_pos)

                # EB width
                eb_lb = eb_lower_bound(stats.mu, stats.var, stats.n_eff, alpha)
                eb_width = stats.mu - eb_lb if not np.isnan(eb_lb) else np.nan

                # Bootstrap width
                boot_lb, boot_width = bootstrap_lower_bound_width(
                    y_pos, w_pos, alpha, n_boot=1000, seed=trial_seed + hash(str(cid)) % 10000
                )

                if np.isnan(eb_width) or np.isnan(boot_width):
                    continue

                all_comparisons.append({
                    "dataset": ds_name,
                    "domain": data["config"]["domain"],
                    "cohort_id": cid,
                    "trial_id": trial,
                    "n_pos": len(y_pos),
                    "n_eff": stats.n_eff,
                    "mu_hat": stats.mu,
                    "eb_lb": eb_lb,
                    "eb_width": eb_width,
                    "boot_lb": boot_lb,
                    "boot_width": boot_width,
                    "eb_wider": int(eb_width > boot_width),
                    "width_ratio": eb_width / max(boot_width, 1e-10),
                })

        sub = [c for c in all_comparisons if c["dataset"] == ds_name]
        if sub:
            eb_wider_pct = np.mean([c["eb_wider"] for c in sub]) * 100
            mean_ratio = np.mean([c["width_ratio"] for c in sub])
            print(f"    EB wider {eb_wider_pct:.1f}% of time, "
                  f"mean width ratio = {mean_ratio:.2f}")

    return pd.DataFrame(all_comparisons)


# =============================================================================
# MAIN
# =============================================================================

def main():
    out_dir = shift_bench_dir / "results" / "h4_real_bootstrap"
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()

    # Part 1: Real-data FWER
    df_fwer = run_real_data_fwer_validation(n_trials=200, alpha=0.05, seed=42)
    df_fwer.to_csv(out_dir / "real_data_fwer_raw.csv", index=False)

    fwer_summary = df_fwer.groupby(["dataset", "domain", "null_offset"]).agg(
        observed_fwer=("fwer", "mean"),
        total_false=("fwer", "sum"),
        n_trials=("trial_id", "count"),
        mean_certs=("n_certified", "mean"),
    ).reset_index()
    fwer_summary.to_csv(out_dir / "real_data_fwer_summary.csv", index=False)

    # Wilson CI
    print(f"\n{'='*70}")
    print("H4 REAL-DATA FWER SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<10} {'Domain':<10} {'Offset':>7} {'FWER':>7} {'k/n':>8} "
          f"{'Wilson CI':>16}")
    print("-" * 62)

    for _, row in fwer_summary.iterrows():
        k = int(row["total_false"])
        n = int(row["n_trials"])
        obs = k / n if n > 0 else 0
        z = 1.96
        denom = 1 + z**2/n
        center = (obs + z**2/(2*n)) / denom
        spread = z * np.sqrt(obs*(1-obs)/n + z**2/(4*n**2)) / denom
        wlo = max(0, center - spread)
        whi = min(1, center + spread)

        print(f"{row['dataset']:<10} {row['domain']:<10} {row['null_offset']:>7.2f} "
              f"{obs:>6.3f} {k:>3}/{n:<3} [{wlo:.3f},{whi:.3f}]")

    # Part 2: Bootstrap comparison
    df_boot = run_bootstrap_comparison(n_trials=50, alpha=0.05, seed=42)
    df_boot.to_csv(out_dir / "bootstrap_comparison_raw.csv", index=False)

    boot_summary = df_boot.groupby(["dataset", "domain"]).agg(
        eb_wider_pct=("eb_wider", "mean"),
        mean_width_ratio=("width_ratio", "mean"),
        median_width_ratio=("width_ratio", "median"),
        mean_eb_width=("eb_width", "mean"),
        mean_boot_width=("boot_width", "mean"),
        n_comparisons=("trial_id", "count"),
    ).reset_index()
    boot_summary["eb_wider_pct"] = boot_summary["eb_wider_pct"] * 100
    boot_summary.to_csv(out_dir / "bootstrap_comparison_summary.csv", index=False)

    print(f"\n{'='*70}")
    print("H4 BOOTSTRAP vs EB COMPARISON (P4.3)")
    print(f"{'='*70}")
    print(f"{'Dataset':<10} {'Domain':<10} {'EB wider%':>10} {'Ratio':>7} "
          f"{'EB width':>9} {'Boot width':>11}")
    print("-" * 60)

    for _, row in boot_summary.iterrows():
        print(f"{row['dataset']:<10} {row['domain']:<10} "
              f"{row['eb_wider_pct']:>9.1f}% {row['mean_width_ratio']:>7.2f} "
              f"{row['mean_eb_width']:>9.4f} {row['mean_boot_width']:>11.4f}")

    total_wider = df_boot["eb_wider"].mean() * 100
    print(f"\n  Overall: EB wider in {total_wider:.1f}% of cases "
          f"(target >= 70% for P4.3)")
    print(f"  P4.3 {'CONFIRMED' if total_wider >= 70 else 'NOT CONFIRMED'}")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.0f}s")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
