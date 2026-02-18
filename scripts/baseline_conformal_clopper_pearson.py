"""
Conformal / Clopper-Pearson Baseline
=======================================
Non-density-ratio comparator baselines:

1. Clopper-Pearson (unshifted): standard binomial confidence interval on PPV
   per cohort, ignoring any covariate shift. Conservative but shift-unaware.

2. Group-conditional Clopper-Pearson: same as above but applied per-cohort
   with Holm correction for FWER.

This answers: "Why should I care about shift-aware certification?"
Expected: Clopper-Pearson is valid when shift is small, but either over-certifies
(false positives) or is overly conservative when shift is present.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

shift_bench_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(shift_bench_dir / "src"))
sys.path.insert(0, str(shift_bench_dir.parent / "ravel" / "src"))

from shiftbench.data import load_dataset
from ravel.bounds.holm import holm_reject


def clopper_pearson_lower(k, n, alpha):
    """Lower bound of Clopper-Pearson confidence interval.

    P(Bin(n, p) >= k) <= alpha/2
    """
    if n == 0 or k == 0:
        return 0.0
    return beta_dist.ppf(alpha / 2, k, n - k + 1)


def evaluate_clopper_pearson(
    y_cal, preds_cal, cohorts_cal, tau_grid, alpha=0.05
):
    """Run Clopper-Pearson certification with Holm correction.

    Unlike IW methods, this ignores covariate shift entirely.
    """
    cohort_ids = np.unique(cohorts_cal)

    records = []
    pvals = []
    pval_info = []

    for cid in cohort_ids:
        cmask = cohorts_cal == cid
        pmask = cmask & (preds_cal == 1)
        y_pos = y_cal[pmask]
        n_pos = len(y_pos)
        k = y_pos.sum()  # number of true positives

        for tau in tau_grid:
            if n_pos == 0:
                pvals.append(1.0)
                pval_info.append((cid, tau, 0, 0, 0.0, 0.0))
                continue

            # Clopper-Pearson lower bound
            lb = clopper_pearson_lower(int(k), int(n_pos), alpha)
            mu_hat = k / n_pos if n_pos > 0 else 0.0

            # p-value for H0: PPV <= tau
            # P(Bin(n, tau) >= k) -- one-sided
            from scipy.stats import binom
            pval = 1.0 - binom.cdf(int(k) - 1, int(n_pos), tau) if k > 0 else 1.0
            pvals.append(pval)
            pval_info.append((cid, tau, int(n_pos), int(k), mu_hat, lb))

    # Holm step-down for FWER control
    decisions = []
    if pvals:
        rejected = holm_reject(pd.Series(pvals), alpha)
        for i, (cid, tau, n, k, mu, lb) in enumerate(pval_info):
            decisions.append({
                "cohort_id": cid,
                "tau": tau,
                "decision": "CERTIFY" if rejected.iloc[i] else "ABSTAIN",
                "mu_hat": mu,
                "lower_bound": lb,
                "n_eff": n,  # For CP, n_eff = n (no weighting)
                "n_positive_preds": n,
                "k_true_positive": k,
            })

    return decisions


def run_conformal_baseline():
    out_dir = shift_bench_dir / "results" / "conformal_baseline"
    os.makedirs(out_dir, exist_ok=True)

    # Same datasets as Tier A
    datasets = [
        ("bace", "molecular"), ("bbbp", "molecular"),
        ("clintox", "molecular"), ("esol", "molecular"),
        ("adult", "tabular"), ("compas", "tabular"),
        ("bank", "tabular"), ("german_credit", "tabular"),
        ("imdb", "text"), ("yelp", "text"),
    ]

    tau_grid = [0.5, 0.6, 0.7, 0.8, 0.9]
    alpha = 0.05
    all_results = []

    # Regression binarization sets
    REGRESSION_DATASETS = {"esol", "freesolv", "lipophilicity"}

    for ds_name, domain in datasets:
        print(f"\n  {ds_name} ({domain}):")

        try:
            X, y, cohorts, splits = load_dataset(ds_name)
        except Exception as e:
            print(f"    SKIP: {e}")
            continue

        # Handle regression binarization
        if ds_name in REGRESSION_DATASETS:
            train_mask = (splits["split"] == "train").values
            threshold = np.median(y[train_mask])
            y = (y > threshold).astype(int)
            print(f"    Binarized: threshold={threshold:.3f}")

        cal_mask = (splits["split"] == "cal").values
        y_cal = y[cal_mask].astype(int)
        cohorts_cal = cohorts[cal_mask]

        # Load real predictions
        import json
        mapping_path = shift_bench_dir / "models" / "prediction_mapping.json"
        preds_cal = y_cal.copy()  # default oracle
        if mapping_path.exists():
            with open(mapping_path) as f:
                mapping = json.load(f)
            if ds_name in mapping:
                pred_path = shift_bench_dir / mapping[ds_name]["cal_binary"]
                if pred_path.exists():
                    loaded = np.load(pred_path)
                    if len(loaded) == len(y_cal):
                        preds_cal = loaded.astype(int)
                        print(f"    Real predictions loaded")

        # Run Clopper-Pearson baseline
        decisions = evaluate_clopper_pearson(
            y_cal, preds_cal, cohorts_cal, tau_grid, alpha
        )

        for d in decisions:
            d["dataset"] = ds_name
            d["domain"] = domain
            d["method"] = "clopper_pearson"

        all_results.extend(decisions)

        n_cert = sum(1 for d in decisions if d["decision"] == "CERTIFY")
        print(f"    CP: {n_cert} CERTIFY / {len(decisions)} total "
              f"({n_cert/len(decisions)*100:.1f}%)")

    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "conformal_baseline_raw.csv", index=False)

    # Compare with uLSIF from Tier A
    ulsif_path = shift_bench_dir / "results" / "cross_domain_tier_a" / "cross_domain_raw_results.csv"
    if ulsif_path.exists():
        ulsif_raw = pd.read_csv(ulsif_path)
        ulsif_subset = ulsif_raw[ulsif_raw["method"] == "ulsif"]

        # Build comparison
        comparison = []
        for ds_name, domain in datasets:
            cp_ds = df[df["dataset"] == ds_name]
            ul_ds = ulsif_subset[ulsif_subset["dataset"] == ds_name]

            if len(cp_ds) > 0 and len(ul_ds) > 0:
                cp_cert = (cp_ds["decision"] == "CERTIFY").mean() * 100
                ul_cert = (ul_ds["decision"] == "CERTIFY").mean() * 100
                cp_neff = cp_ds["n_eff"].mean()
                ul_neff = ul_ds["n_eff"].mean()

                comparison.append({
                    "dataset": ds_name,
                    "domain": domain,
                    "cp_cert_pct": cp_cert,
                    "ulsif_cert_pct": ul_cert,
                    "cp_n_eff": cp_neff,
                    "ulsif_n_eff": ul_neff,
                    "cp_advantage": cp_cert - ul_cert,
                })

        comp_df = pd.DataFrame(comparison)
        comp_df.to_csv(out_dir / "cp_vs_ulsif_comparison.csv", index=False)

        print(f"\n{'='*70}")
        print("CLOPPER-PEARSON vs uLSIF COMPARISON")
        print(f"{'='*70}")
        print(f"{'Dataset':<16} {'Domain':<12} {'CP Cert%':>9} {'uLSIF Cert%':>12} {'CP-uLSIF':>9}")
        print("-" * 60)
        for _, row in comp_df.iterrows():
            diff = row["cp_advantage"]
            marker = "+" if diff > 0 else ""
            print(f"{row['dataset']:<16} {row['domain']:<12} {row['cp_cert_pct']:>8.2f}% "
                  f"{row['ulsif_cert_pct']:>11.2f}% {marker}{diff:>8.2f}%")

        # Summary by domain
        domain_comp = comp_df.groupby("domain").agg(
            mean_cp=("cp_cert_pct", "mean"),
            mean_ulsif=("ulsif_cert_pct", "mean"),
        ).reset_index()

        print(f"\nDomain averages:")
        for _, row in domain_comp.iterrows():
            print(f"  {row['domain']}: CP={row['mean_cp']:.1f}%, uLSIF={row['mean_ulsif']:.1f}%")

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    run_conformal_baseline()
