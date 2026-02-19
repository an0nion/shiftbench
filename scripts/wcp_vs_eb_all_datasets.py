"""
PI Priority 1: WCP vs EB Comparison on All Real Datasets
=========================================================
Validates whether the 6.5x WCP advantage on BACE generalizes
to text and tabular domains, or is specific to molecular/sparse-cohort regimes.

Design:
    For each of 6 Tier-A datasets (adult, compas, imdb, yelp, bace, bbbp):
    1. Load real features + real model predictions
    2. Estimate uLSIF weights (X_cal, X_test)
    3. Run WCP (weighted quantile) on (y_cal, preds_cal, cohorts_cal, weights)
    4. Run EB  (uLSIF + EB bound)   on the same inputs
    5. Compare cert rates and mean lower bounds per (cohort, tau) pair

NOTE: Neither WCP nor EB has Holm correction here -- raw per-cohort decisions.
This matches the original BACE comparison and the compare_conformal_vs_eb.py script.

Output:
    results/wcp_vs_eb_all_datasets/
        wcp_eb_raw.csv       -- per-(dataset, cohort, tau) decisions
        wcp_eb_summary.csv   -- per-dataset summary stats
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

shift_bench_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(shift_bench_dir / "src"))
sys.path.insert(0, str(shift_bench_dir.parent / "ravel" / "src"))
sys.path.insert(0, str(shift_bench_dir / "scripts"))

from experiment_a_real_data_calibration import load_real_dataset, DATASET_CONFIG
from shiftbench.baselines.weighted_conformal import WeightedConformalBaseline
from shiftbench.baselines.ulsif import uLSIFBaseline

DATA_DIR   = str(shift_bench_dir / "data" / "processed")
MODEL_DIR  = str(shift_bench_dir / "models")
OUT_DIR    = shift_bench_dir / "results" / "wcp_vs_eb_all_datasets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAU_GRID   = [0.5, 0.6, 0.7, 0.8, 0.9]
ALPHA      = 0.05
DATASETS   = list(DATASET_CONFIG.keys())   # adult, compas, imdb, yelp, bace, bbbp


def run_dataset(dataset_name: str) -> list:
    print(f"\n{'='*60}")
    print(f"  {dataset_name.upper()}")
    print(f"{'='*60}")

    data = load_real_dataset(dataset_name, DATA_DIR, MODEL_DIR)
    X_source   = data["X_source"]
    y_source   = data["y_source"]
    cohorts    = data["cohorts_source"]
    preds      = data["preds_source"]
    X_test     = data["X_test"]

    print(f"  Source: {len(X_source)} | Test: {len(X_test)} | Cohorts: {len(np.unique(cohorts))}")

    # -- Estimate weights (shared between both methods) --
    weight_est = uLSIFBaseline(n_basis=100, random_state=42)
    weights = weight_est.estimate_weights(X_source, X_test)
    n_eff_global = (weights.sum())**2 / (weights**2).sum()
    print(f"  uLSIF weights: mean={weights.mean():.3f}, n_eff={n_eff_global:.1f}")

    # -- WCP bounds (per-cohort, no Holm) --
    # estimate_bounds() takes pre-computed weights directly
    wcp = WeightedConformalBaseline(weight_method="ulsif", n_basis=100, random_state=42)
    decisions_wcp = wcp.estimate_bounds(y_source, preds, cohorts, weights, TAU_GRID, ALPHA)

    # -- EB bounds via uLSIF baseline --
    eb = uLSIFBaseline(n_basis=100, random_state=42)
    decisions_eb = eb.estimate_bounds(y_source, preds, cohorts, weights, TAU_GRID, ALPHA)

    # Build dict for fast lookup: (cohort, tau) -> decision_eb
    eb_lookup = {(d.cohort_id, d.tau): d for d in decisions_eb}

    rows = []
    for d_wcp in decisions_wcp:
        d_eb = eb_lookup.get((d_wcp.cohort_id, d_wcp.tau))
        if d_eb is None:
            continue
        rows.append({
            "dataset":    dataset_name,
            "domain":     DATASET_CONFIG[dataset_name]["domain"],
            "cohort":     str(d_wcp.cohort_id),
            "tau":        d_wcp.tau,
            "n_eff":      float(d_wcp.n_eff) if np.isfinite(d_wcp.n_eff) else 0.0,
            "mu_hat":     float(d_wcp.mu_hat) if (d_wcp.mu_hat is not None and np.isfinite(d_wcp.mu_hat)) else float("nan"),
            "wcp_lb":     float(d_wcp.lower_bound) if d_wcp.lower_bound is not None and np.isfinite(d_wcp.lower_bound) else float("nan"),
            "eb_lb":      float(d_eb.lower_bound) if d_eb.lower_bound is not None and np.isfinite(d_eb.lower_bound) else float("nan"),
            "wcp_cert":   int(d_wcp.decision == "CERTIFY"),
            "eb_cert":    int(d_eb.decision == "CERTIFY"),
        })

    df = pd.DataFrame(rows)
    valid = df[np.isfinite(df["wcp_lb"]) & np.isfinite(df["eb_lb"]) & (df["n_eff"] > 0)]
    n_total = len(valid)
    wcp_rate = valid["wcp_cert"].mean() * 100 if n_total > 0 else 0
    eb_rate  = valid["eb_cert"].mean() * 100  if n_total > 0 else 0
    ratio    = (valid["wcp_cert"].sum() / max(valid["eb_cert"].sum(), 1))
    wcp_lb_mean = valid["wcp_lb"].mean() if n_total > 0 else float("nan")
    eb_lb_mean  = valid["eb_lb"].mean()  if n_total > 0 else float("nan")
    print(f"  Valid pairs: {n_total} | WCP cert: {wcp_rate:.1f}% | EB cert: {eb_rate:.1f}% | Ratio: {ratio:.1f}x")
    print(f"  WCP mean LB: {wcp_lb_mean:.4f} | EB mean LB: {eb_lb_mean:.4f}")
    return rows


def main():
    all_rows = []
    failed = []

    for ds in DATASETS:
        try:
            rows = run_dataset(ds)
            all_rows.extend(rows)
        except Exception as e:
            import traceback
            print(f"\n[ERROR] {ds}: {e}")
            traceback.print_exc()
            failed.append(ds)

    if not all_rows:
        print("\nNo results collected.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "wcp_eb_raw.csv", index=False)
    print(f"\nRaw results: {len(df)} rows -> {OUT_DIR / 'wcp_eb_raw.csv'}")

    # -- Summary per dataset --
    summary_rows = []
    valid = df[np.isfinite(df["wcp_lb"]) & np.isfinite(df["eb_lb"]) & (df["n_eff"] > 0)].copy()
    valid["wcp_tighter"] = (valid["wcp_lb"] > valid["eb_lb"]).astype(int)
    valid["only_wcp"]    = ((valid["wcp_cert"] == 1) & (valid["eb_cert"] == 0)).astype(int)
    valid["only_eb"]     = ((valid["wcp_cert"] == 0) & (valid["eb_cert"] == 1)).astype(int)

    for (ds, domain), g in valid.groupby(["dataset", "domain"]):
        n_wcp = int(g["wcp_cert"].sum())
        n_eb  = int(g["eb_cert"].sum())
        summary_rows.append({
            "dataset":       ds,
            "domain":        domain,
            "n_pairs":       len(g),
            "wcp_cert_rate": float(g["wcp_cert"].mean() * 100),
            "eb_cert_rate":  float(g["eb_cert"].mean() * 100),
            "wcp_n_cert":    n_wcp,
            "eb_n_cert":     n_eb,
            "ratio_wcp_eb":  float(n_wcp / max(n_eb, 1)),
            "wcp_mean_lb":   float(g["wcp_lb"].mean()),
            "eb_mean_lb":    float(g["eb_lb"].mean()),
            "pct_wcp_tighter": float(g["wcp_tighter"].mean() * 100),
            "n_only_wcp":    int(g["only_wcp"].sum()),
            "n_only_eb":     int(g["only_eb"].sum()),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "wcp_eb_summary.csv", index=False)

    print("\n" + "="*70)
    print("SUMMARY: WCP vs EB Certification Rates")
    print("="*70)
    print(summary_df[["dataset", "domain", "n_pairs",
                       "wcp_cert_rate", "eb_cert_rate", "ratio_wcp_eb",
                       "pct_wcp_tighter"]].to_string(index=False))

    # -- n_eff bin breakdown --
    valid["n_eff_bin"] = pd.cut(
        valid["n_eff"],
        bins=[0, 10, 25, 100, 300, np.inf],
        labels=["<10", "10-25", "25-100", "100-300", "300+"]
    )
    print("\nBy n_eff bin (all datasets):")
    for bin_name, g in valid.groupby("n_eff_bin", observed=True):
        if len(g) == 0:
            continue
        print(f"  n_eff {bin_name:>7} ({len(g):4d} pairs): "
              f"WCP={g['wcp_cert'].mean()*100:5.1f}%  EB={g['eb_cert'].mean()*100:5.1f}%  "
              f"WCP_tighter={g['wcp_tighter'].mean()*100:.0f}%")

    if failed:
        print(f"\nFailed datasets: {failed}")

    print(f"\nFiles written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
