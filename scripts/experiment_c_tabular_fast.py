"""
H1 Tabular Regime - Fast Version (uLSIF only, skip KLIEP on COMPAS)
=====================================================================
Same as experiment_c_tabular_regime but runs only uLSIF for weight
estimation (avoiding O(n^2) KLIEP on COMPAS).

For H1 analysis: we measure "does uLSIF produce active pairs in tabular?"
which is the prerequisite for meaningful H1 testing.
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


def run_ulsif_trial(data, n_cal, tau_grid, alpha, trial_id, seed):
    """Run a single trial with only uLSIF."""
    rng = np.random.RandomState(seed)

    n_source = len(data["X_source"])
    sample_idx = rng.choice(n_source, size=min(n_cal, n_source), replace=True)

    X_cal = data["X_source"][sample_idx]
    y_cal = data["y_source"][sample_idx]
    cohorts_cal = data["cohorts_source"][sample_idx]
    preds_cal = data["preds_source"][sample_idx]
    X_test = data["X_test"]

    from shiftbench.baselines.ulsif import uLSIFBaseline
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

    for cid in cohort_ids:
        cmask = cohorts_cal == cid
        pmask = cmask & (preds_cal == 1)
        y_pos = y_cal[pmask]
        w_pos = weights[pmask]

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

    decisions = {}
    if pvals:
        rejected = holm_reject(pd.Series(pvals), alpha)
        for i, info in enumerate(test_info):
            cid, tau = info[0], info[1]
            mu_hat = info[2] if len(info) > 2 else np.nan
            n_eff = info[3] if len(info) > 3 else np.nan
            lb = info[4] if len(info) > 4 else np.nan
            key = (cid, tau)
            decisions[key] = {
                "certified": bool(rejected.iloc[i]),
                "mu_hat": mu_hat,
                "n_eff": n_eff,
                "lower_bound": lb,
            }

    results = []
    for (cid, tau), dec in decisions.items():
        results.append({
            "dataset": data["dataset_name"],
            "domain": data["config"]["domain"],
            "trial_id": trial_id,
            "cohort_id": cid,
            "tau": tau,
            "certified": dec["certified"],
            "mu_hat": dec["mu_hat"],
            "n_eff": dec["n_eff"],
            "lower_bound": dec["lower_bound"],
        })

    return results


def main():
    n_trials = 30
    alpha = 0.05
    tau_grid = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    seed = 42
    out_dir = shift_bench_dir / "results" / "h1_tabular_regime"
    os.makedirs(out_dir, exist_ok=True)

    data_dir = str(shift_bench_dir / "data" / "processed")
    model_dir = str(shift_bench_dir / "models")

    datasets = ["adult", "compas"]
    cohort_configs = [10, 15, 20]

    all_results = []

    for n_bins in cohort_configs:
        print(f"\n{'='*70}")
        print(f"Cohort bins: {n_bins}")
        print(f"{'='*70}")

        for ds in datasets:
            if ds in DATASET_CONFIG:
                DATASET_CONFIG[ds]["n_cohort_bins"] = n_bins

            t0 = time.time()
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
                trial_results = run_ulsif_trial(
                    data=data,
                    n_cal=n_source,
                    tau_grid=tau_grid,
                    alpha=alpha,
                    trial_id=trial,
                    seed=trial_seed,
                )
                for r in trial_results:
                    r["n_cohort_bins"] = n_bins
                all_results.extend(trial_results)

            elapsed = time.time() - t0
            print(f"    {ds} done in {elapsed:.1f}s")

    df = pd.DataFrame(all_results)
    df["active"] = df["certified"]  # For uLSIF-only, active = certified

    df.to_csv(out_dir / "tabular_regime_ulsif_raw.csv", index=False)

    print(f"\n{'='*70}")
    print("TABULAR REGIME RESULTS (uLSIF only)")
    print(f"{'='*70}")
    print(f"{'Dataset':<10} {'Bins':>5} {'Pairs':>6} {'Certs':>6} {'Cert%':>7} "
          f"{'MedNeff':>8} {'MaxMu':>7}")
    print("-" * 55)

    summaries = []
    for (ds, nb), group in df.groupby(["dataset", "n_cohort_bins"]):
        n_pairs = len(group)
        n_cert = group["certified"].sum()
        cert_pct = group["certified"].mean() * 100
        med_neff = group["n_eff"].median()
        max_mu = group["mu_hat"].max()

        summaries.append({
            "dataset": ds, "n_cohort_bins": nb,
            "n_pairs": n_pairs, "n_certified": n_cert,
            "cert_pct": cert_pct, "median_neff": med_neff,
            "max_mu": max_mu,
        })

        print(f"{ds:<10} {nb:>5} {n_pairs:>6} {n_cert:>6} {cert_pct:>6.2f}% "
              f"{med_neff:>8.1f} {max_mu:>7.3f}")

    pd.DataFrame(summaries).to_csv(out_dir / "tabular_regime_summary.csv", index=False)

    # Find best config
    for ds in datasets:
        sub = [s for s in summaries if s["dataset"] == ds]
        if sub:
            best = max(sub, key=lambda x: x["n_certified"])
            print(f"\n  Best for {ds}: {best['n_cohort_bins']} bins, "
                  f"{best['n_certified']} certifications ({best['cert_pct']:.1f}%)")

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
