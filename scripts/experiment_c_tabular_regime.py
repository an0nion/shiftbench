"""
H1 Tabular Regime Adjustment
==============================
Re-run H1 (KLIEP vs uLSIF agreement) on Adult and COMPAS with adjusted
regime to obtain active (non-trivial) certification pairs.

Changes from original experiment_c:
- Use n_cohort_bins=15 (not 5) to preserve PPV signal in cohorts
- Use tau_grid starting at 0.3 to catch lower-threshold certifications
- Use full source set (no subsampling cap)
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
from experiment_c_cross_domain_h1 import run_h1_trial, compute_cohens_kappa


def run_tabular_regime():
    n_trials = 30
    alpha = 0.05
    tau_grid = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    seed = 42
    out_dir = shift_bench_dir / "results" / "h1_tabular_regime"
    os.makedirs(out_dir, exist_ok=True)

    data_dir = str(shift_bench_dir / "data" / "processed")
    model_dir = str(shift_bench_dir / "models")

    datasets = ["adult", "compas"]
    cohort_configs = [10, 15, 20]  # Try multiple bin counts

    all_results = []

    for n_bins in cohort_configs:
        print(f"\n{'='*70}")
        print(f"Cohort bins: {n_bins}")
        print(f"{'='*70}")

        for ds in datasets:
            # Override cohort bins
            if ds in DATASET_CONFIG:
                DATASET_CONFIG[ds]["n_cohort_bins"] = n_bins

            try:
                data = load_real_dataset(ds, data_dir, model_dir)
            except Exception as e:
                print(f"  [SKIP] {ds}: {e}")
                continue

            n_source = len(data["X_source"])
            n_cohorts = len(np.unique(data["cohorts_source"]))
            print(f"\n  {ds}: {n_source} source, {n_cohorts} cohorts")

            for trial in range(n_trials):
                if trial % max(1, n_trials // 3) == 0:
                    print(f"    Trial {trial+1}/{n_trials}...")

                trial_seed = seed + trial * 137
                trial_results = run_h1_trial(
                    data=data,
                    n_cal=n_source,  # Use full source, no subsampling
                    tau_grid=tau_grid,
                    alpha=alpha,
                    trial_id=trial,
                    seed=trial_seed,
                )
                for r in trial_results:
                    r["n_cohort_bins"] = n_bins
                all_results.extend(trial_results)

    df = pd.DataFrame(all_results)
    df["disagree"] = ~df["agree"]
    df["active"] = df["either_certify"]
    df["min_neff"] = df[["ulsif_neff", "kliep_neff"]].min(axis=1)

    # Save raw
    df.to_csv(out_dir / "tabular_regime_raw.csv", index=False)

    # Summary by (dataset, n_bins)
    print(f"\n{'='*70}")
    print("TABULAR REGIME RESULTS")
    print(f"{'='*70}")
    print(f"{'Dataset':<10} {'Bins':>5} {'Pairs':>6} {'Active':>6} {'Disagree':>8} "
          f"{'Agree%':>7} {'ActDis%':>8} {'MedNeff':>8}")
    print("-" * 65)

    summaries = []
    for (ds, nb), group in df.groupby(["dataset", "n_cohort_bins"]):
        n_pairs = len(group)
        n_active = group["active"].sum()
        n_disagree = group["disagree"].sum()
        agree_pct = group["agree"].mean() * 100
        act_dis = (group.loc[group["active"], "disagree"].mean() * 100
                   if n_active > 0 else float("nan"))
        med_neff = group["min_neff"].median()

        summaries.append({
            "dataset": ds, "n_cohort_bins": nb,
            "n_pairs": n_pairs, "n_active": n_active,
            "n_disagree": n_disagree, "agree_pct": agree_pct,
            "active_disagree_pct": act_dis, "median_neff": med_neff,
        })

        act_str = f"{act_dis:.1f}%" if not np.isnan(act_dis) else "N/A"
        print(f"{ds:<10} {nb:>5} {n_pairs:>6} {n_active:>6} {n_disagree:>8} "
              f"{agree_pct:>6.1f}% {act_str:>8} {med_neff:>8.1f}")

    pd.DataFrame(summaries).to_csv(out_dir / "tabular_regime_summary.csv", index=False)

    # Best config: which n_bins gives most active pairs?
    for ds in datasets:
        sub = [s for s in summaries if s["dataset"] == ds]
        if sub:
            best = max(sub, key=lambda x: x["n_active"])
            print(f"\n  Best for {ds}: {best['n_cohort_bins']} bins, "
                  f"{best['n_active']} active pairs")

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    run_tabular_regime()
