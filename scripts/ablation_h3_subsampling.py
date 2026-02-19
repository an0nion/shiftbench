"""
H3 Subsampling Intervention: Does text become molecular-hard at small cohorts?
================================================================================
P3.4: Subsample text datasets to molecular-sized cohorts and re-run
certification. If cert_rate drops >= 60 pp, then domain difficulty is
driven by cohort properties, not domain identity.

Also tests intermediate sizes to trace the relationship.
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


def subsample_to_cohort_size(data, target_per_cohort, seed):
    """Subsample each cohort to exactly target_per_cohort samples.

    Drops cohorts that have fewer than target_per_cohort samples.
    Returns modified data dict (same keys as input).
    """
    rng = np.random.RandomState(seed)
    cohort_ids = np.unique(data["cohorts_source"])

    keep_indices = []
    kept_cohorts = []

    for cid in cohort_ids:
        mask = data["cohorts_source"] == cid
        idx = np.where(mask)[0]
        if len(idx) >= target_per_cohort:
            selected = rng.choice(idx, size=target_per_cohort, replace=False)
            keep_indices.extend(selected)
            kept_cohorts.append(cid)

    if len(keep_indices) == 0:
        return None, 0

    keep = np.array(keep_indices)

    subsampled = {
        "X_source": data["X_source"][keep],
        "y_source": data["y_source"][keep],
        "cohorts_source": data["cohorts_source"][keep],
        "preds_source": data["preds_source"][keep],
        "X_test": data["X_test"],
        "dataset_name": data["dataset_name"],
        "config": data["config"],
    }
    return subsampled, len(kept_cohorts)


def run_certification_trial(data, tau_grid, alpha, seed):
    """Run one certification trial with uLSIF weights."""
    X_cal = data["X_source"]
    y_cal = data["y_source"]
    cohorts_cal = data["cohorts_source"]
    preds_cal = data["preds_source"]
    X_test = data["X_test"]

    n_basis = min(100, len(X_cal))

    try:
        ulsif = uLSIFBaseline(n_basis=n_basis, sigma=None, lambda_=0.1,
                               random_state=seed)
        weights = ulsif.estimate_weights(X_cal, X_test)
    except Exception:
        weights = np.ones(len(X_cal))

    cohort_ids = np.unique(cohorts_cal)
    pvals = []
    test_info = []
    neff_list = []

    for cid in cohort_ids:
        cmask = cohorts_cal == cid
        pmask = cmask & (preds_cal == 1)
        y_pos = y_cal[pmask]
        w_pos = weights[pmask]

        if len(y_pos) < 2 or w_pos.sum() == 0:
            for tau in tau_grid:
                pvals.append(1.0)
                test_info.append((cid, tau, np.nan, np.nan))
            continue

        w_pos = w_pos / w_pos.mean() if w_pos.mean() > 0 else w_pos
        stats = weighted_stats_01(y_pos, w_pos)
        neff_list.append(stats.n_eff)

        for tau in tau_grid:
            pval = eb_p_value(stats.mu, stats.var, stats.n_eff, tau)
            pvals.append(pval)
            test_info.append((cid, tau, stats.mu, stats.n_eff))

    n_certified = 0
    if pvals:
        rejected = holm_reject(pd.Series(pvals), alpha)
        n_certified = rejected.sum()

    n_tests = len(cohort_ids) * len(tau_grid)
    cert_rate = n_certified / n_tests if n_tests > 0 else 0
    mean_neff = np.mean(neff_list) if neff_list else 0
    median_neff = np.median(neff_list) if neff_list else 0

    return {
        "n_certified": n_certified,
        "n_tests": n_tests,
        "cert_rate": cert_rate,
        "mean_neff": mean_neff,
        "median_neff": median_neff,
        "n_cohorts": len(cohort_ids),
        "n_samples": len(X_cal),
    }


def main():
    out_dir = shift_bench_dir / "results" / "h3_subsampling"
    os.makedirs(out_dir, exist_ok=True)

    data_dir = str(shift_bench_dir / "data" / "processed")
    model_dir = str(shift_bench_dir / "models")

    alpha = 0.05
    tau_grid = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    n_trials = 20
    seed = 42

    # Target cohort sizes: from molecular-like (3-5) to original
    cohort_sizes = [3, 5, 10, 20, 50, 100, 200, 500, None]  # None = full dataset

    # Datasets: text (high cert) + tabular (moderate) for comparison
    datasets = ["imdb", "yelp", "adult", "compas"]

    # Also get molecular baseline for reference
    mol_datasets = ["bace", "bbbp"]

    all_results = []

    # Molecular baselines (no subsampling)
    for ds_name in mol_datasets:
        try:
            data = load_real_dataset(ds_name, data_dir, model_dir)
        except Exception as e:
            print(f"  [SKIP] {ds_name}: {e}")
            continue

        print(f"\n{ds_name} (molecular baseline):")
        for trial in range(n_trials):
            trial_seed = seed + trial * 137
            result = run_certification_trial(data, tau_grid, alpha, trial_seed)
            result["dataset"] = ds_name
            result["domain"] = "molecular"
            result["target_cohort_size"] = "original"
            result["trial_id"] = trial
            all_results.append(result)

        sub = [r for r in all_results if r["dataset"] == ds_name]
        mean_cr = np.mean([r["cert_rate"] for r in sub])
        mean_ne = np.mean([r["median_neff"] for r in sub])
        print(f"  cert_rate={mean_cr*100:.2f}%, median_neff={mean_ne:.1f}")

    # Main intervention: subsample text/tabular to different sizes
    for ds_name in datasets:
        try:
            data = load_real_dataset(ds_name, data_dir, model_dir)
        except Exception as e:
            print(f"  [SKIP] {ds_name}: {e}")
            continue

        domain = data["config"]["domain"]
        orig_cohort_sizes_arr = []
        for cid in np.unique(data["cohorts_source"]):
            orig_cohort_sizes_arr.append((data["cohorts_source"] == cid).sum())
        median_orig_size = np.median(orig_cohort_sizes_arr)

        print(f"\n{'='*70}")
        print(f"{ds_name} ({domain}): {len(data['X_source'])} samples, "
              f"median cohort size={median_orig_size:.0f}")
        print(f"{'='*70}")

        for target_size in cohort_sizes:
            if target_size is None:
                label = "original"
                print(f"\n  original (no subsample):")
                for trial in range(n_trials):
                    trial_seed = seed + trial * 137
                    result = run_certification_trial(data, tau_grid, alpha, trial_seed)
                    result["dataset"] = ds_name
                    result["domain"] = domain
                    result["target_cohort_size"] = "original"
                    result["trial_id"] = trial
                    all_results.append(result)
            else:
                if target_size > median_orig_size:
                    continue  # Skip if target is larger than natural

                print(f"\n  target_per_cohort={target_size}:")
                for trial in range(n_trials):
                    trial_seed = seed + trial * 137
                    subsampled, n_kept = subsample_to_cohort_size(
                        data, target_size, seed=trial_seed
                    )
                    if subsampled is None:
                        continue

                    result = run_certification_trial(
                        subsampled, tau_grid, alpha, trial_seed
                    )
                    result["dataset"] = ds_name
                    result["domain"] = domain
                    result["target_cohort_size"] = target_size
                    result["n_kept_cohorts"] = n_kept
                    result["trial_id"] = trial
                    all_results.append(result)

            # Progress
            sub = [r for r in all_results
                   if r["dataset"] == ds_name
                   and str(r["target_cohort_size"]) == str(target_size if target_size else "original")]
            if sub:
                mean_cr = np.mean([r["cert_rate"] for r in sub])
                mean_ne = np.mean([r["median_neff"] for r in sub])
                n_coh = np.mean([r["n_cohorts"] for r in sub])
                print(f"    cert_rate={mean_cr*100:.2f}%, median_neff={mean_ne:.1f}, "
                      f"cohorts={n_coh:.0f}")

    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "subsampling_raw.csv", index=False)

    # Summary table
    print(f"\n{'='*70}")
    print("H3 SUBSAMPLING INTERVENTION RESULTS")
    print(f"{'='*70}")

    summary = df.groupby(["dataset", "domain", "target_cohort_size"]).agg(
        mean_cert_rate=("cert_rate", "mean"),
        std_cert_rate=("cert_rate", "std"),
        mean_neff=("median_neff", "mean"),
        mean_cohorts=("n_cohorts", "mean"),
        n_trials=("trial_id", "count"),
    ).reset_index()
    summary.to_csv(out_dir / "subsampling_summary.csv", index=False)

    print(f"\n{'Dataset':<10} {'Domain':<10} {'CohortSz':<10} {'Cert%':>8} "
          f"{'StdCert':>8} {'MedNeff':>8} {'Cohorts':>8}")
    print("-" * 70)
    for _, row in summary.iterrows():
        print(f"{row['dataset']:<10} {row['domain']:<10} "
              f"{str(row['target_cohort_size']):<10} "
              f"{row['mean_cert_rate']*100:>7.2f}% "
              f"{row['std_cert_rate']*100:>7.2f}% "
              f"{row['mean_neff']:>8.1f} {row['mean_cohorts']:>7.0f}")

    # Compute cert_rate drop for text datasets
    print(f"\n{'='*70}")
    print("CERT RATE DROP (text datasets, original vs molecular-sized)")
    print(f"{'='*70}")

    for ds_name in ["imdb", "yelp"]:
        orig = summary[(summary["dataset"] == ds_name) &
                        (summary["target_cohort_size"] == "original")]
        small = summary[(summary["dataset"] == ds_name) &
                         (summary["target_cohort_size"].isin([3, 5]))]

        if len(orig) > 0:
            orig_cr = orig["mean_cert_rate"].values[0] * 100
            print(f"\n  {ds_name}: original cert_rate = {orig_cr:.1f}%")
            for _, row in small.iterrows():
                small_cr = row["mean_cert_rate"] * 100
                drop = orig_cr - small_cr
                print(f"    cohort_size={row['target_cohort_size']}: "
                      f"cert_rate={small_cr:.1f}%, drop={drop:.1f} pp "
                      f"{'(>= 60 pp P3.4 CONFIRMED)' if drop >= 60 else ''}")

    # Reference: molecular baselines
    print(f"\n  Molecular baselines:")
    for ds_name in mol_datasets:
        sub = summary[summary["dataset"] == ds_name]
        if len(sub) > 0:
            print(f"    {ds_name}: cert_rate={sub['mean_cert_rate'].values[0]*100:.2f}%")

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
