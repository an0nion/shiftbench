"""
Binarization Sensitivity: Median-split vs Clinically Motivated Thresholds
==========================================================================
Tests whether certification results are sensitive to binarization choice
for regression datasets (ESOL, FreeSolv, Lipophilicity).

Thresholds tested:
  1. Median-split (default): threshold = median(y_train)
  2. Clinical/domain-motivated:
     - ESOL: logS = -3.0 (poor vs moderate solubility, Delaney 2004)
     - FreeSolv: deltaG = -5.0 kcal/mol (favorable vs unfavorable binding)
     - Lipophilicity: logD = 3.0 (Lipinski rule-of-5 threshold)
  3. Quartile splits: Q1 (25th), Q3 (75th) for robustness check
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

shift_bench_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(shift_bench_dir / "src"))
sys.path.insert(0, str(shift_bench_dir.parent / "ravel" / "src"))

from shiftbench.data import load_dataset
from shiftbench.baselines.ulsif import uLSIFBaseline
from ravel.bounds.empirical_bernstein import eb_lower_bound
from ravel.bounds.p_value import eb_p_value
from ravel.bounds.weighted_stats import weighted_stats_01
from ravel.bounds.holm import holm_reject


# Clinically motivated thresholds
CLINICAL_THRESHOLDS = {
    "esol": {
        "name": "Poor solubility cutoff",
        "threshold": -3.0,
        "unit": "logS (mol/L)",
        "reference": "Delaney 2004",
    },
    "freesolv": {
        "name": "Favorable binding",
        "threshold": -5.0,
        "unit": "deltaG (kcal/mol)",
        "reference": "FreeSolv database convention",
    },
    "lipophilicity": {
        "name": "Lipinski Rule-of-5",
        "threshold": 3.0,
        "unit": "logD",
        "reference": "Lipinski et al. 1997",
    },
}


def run_certification(X_cal, y_binary, cohorts, preds, X_test, tau_grid, alpha, seed):
    """Run uLSIF certification on binarized data."""
    n_basis = min(100, len(X_cal))

    try:
        ulsif = uLSIFBaseline(n_basis=n_basis, sigma=None, lambda_=0.1,
                               random_state=seed)
        weights = ulsif.estimate_weights(X_cal, X_test)
    except Exception:
        weights = np.ones(len(X_cal))

    cohort_ids = np.unique(cohorts)
    pvals = []
    test_info = []
    neff_list = []

    for cid in cohort_ids:
        cmask = cohorts == cid
        pmask = cmask & (preds == 1)
        y_pos = y_binary[pmask]
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
    return {
        "n_certified": n_certified,
        "n_tests": n_tests,
        "cert_rate": n_certified / n_tests if n_tests > 0 else 0,
        "mean_neff": np.mean(neff_list) if neff_list else 0,
        "median_neff": np.median(neff_list) if neff_list else 0,
        "n_cohorts": len(cohort_ids),
    }


def main():
    out_dir = shift_bench_dir / "results" / "binarization_sensitivity"
    os.makedirs(out_dir, exist_ok=True)

    alpha = 0.05
    tau_grid = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    seed = 42

    datasets = ["esol", "freesolv", "lipophilicity"]
    all_results = []

    for ds_name in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*70}")

        try:
            X, y, cohorts, splits = load_dataset(ds_name)
        except Exception as e:
            print(f"  [SKIP] {ds_name}: {e}")
            continue

        train_mask = (splits["split"] == "train").values
        cal_mask = (splits["split"] == "cal").values
        test_mask = (splits["split"] == "test").values

        X_cal = X[cal_mask]
        y_cal_raw = y[cal_mask]
        cohorts_cal = cohorts[cal_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]

        print(f"  y_train range: [{y_train.min():.3f}, {y_train.max():.3f}]")
        print(f"  y_train median: {np.median(y_train):.3f}")
        print(f"  y_train mean: {np.mean(y_train):.3f}")

        # Define thresholds to test
        thresholds = {}

        # Median split
        median_thresh = np.median(y_train)
        thresholds["median"] = median_thresh

        # Quartile splits
        thresholds["Q1"] = np.percentile(y_train, 25)
        thresholds["Q3"] = np.percentile(y_train, 75)

        # Clinical threshold
        if ds_name in CLINICAL_THRESHOLDS:
            clin = CLINICAL_THRESHOLDS[ds_name]
            thresholds[f"clinical ({clin['name']})"] = clin["threshold"]

        for thresh_name, threshold in thresholds.items():
            y_binary = (y_cal_raw > threshold).astype(int)
            pos_rate = y_binary.mean()

            # Use binary labels as predictions (oracle-like for sensitivity test)
            preds = y_binary.copy()

            print(f"\n  {thresh_name}: threshold={threshold:.3f}, "
                  f"pos_rate={pos_rate:.3f}")

            result = run_certification(
                X_cal, y_binary, cohorts_cal, preds, X_test,
                tau_grid, alpha, seed
            )

            result["dataset"] = ds_name
            result["threshold_name"] = thresh_name
            result["threshold_value"] = threshold
            result["pos_rate"] = pos_rate
            all_results.append(result)

            print(f"    cert_rate={result['cert_rate']*100:.2f}% "
                  f"({result['n_certified']}/{result['n_tests']}), "
                  f"median_neff={result['median_neff']:.1f}")

    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "binarization_sensitivity.csv", index=False)

    # Summary
    print(f"\n{'='*70}")
    print("BINARIZATION SENSITIVITY SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<15} {'Threshold':<25} {'Value':>7} {'PosRate':>8} "
          f"{'Cert%':>7} {'Neff':>6}")
    print("-" * 72)

    for _, row in df.iterrows():
        print(f"{row['dataset']:<15} {row['threshold_name']:<25} "
              f"{row['threshold_value']:>7.3f} {row['pos_rate']:>7.3f} "
              f"{row['cert_rate']*100:>6.2f}% {row['median_neff']:>6.1f}")

    # Sensitivity assessment
    print(f"\n{'='*70}")
    print("SENSITIVITY ASSESSMENT")
    print(f"{'='*70}")

    for ds_name in datasets:
        sub = df[df["dataset"] == ds_name]
        if len(sub) < 2:
            continue
        median_cr = sub[sub["threshold_name"] == "median"]["cert_rate"].values
        if len(median_cr) == 0:
            continue
        median_cr = median_cr[0]

        print(f"\n  {ds_name} (median cert_rate = {median_cr*100:.2f}%):")
        for _, row in sub.iterrows():
            if row["threshold_name"] == "median":
                continue
            diff = (row["cert_rate"] - median_cr) * 100
            print(f"    {row['threshold_name']}: cert_rate={row['cert_rate']*100:.2f}% "
                  f"(diff={diff:+.2f} pp)")

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
