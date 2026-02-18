"""
H1 Disagreement-by-n_eff Analysis
====================================
Analyzes when KLIEP and uLSIF disagree, conditioned on n_eff bins
and estimated margin (mu_hat - tau).

Data sources (merged automatically):
  1. results/experiment_c_real/experiment_c_raw.csv
     - bace, bbbp (molecular), imdb, yelp (text), adult, compas (tabular)
     - 30 trials each; adult/compas have 0 active pairs at n_cohorts=5
  2. results/h1_tabular_regime/tabular_regime_raw.csv  [if available]
     - adult, compas re-run with n_cohorts=10/15/20 and tau down to 0.3
     - Provides active pairs for tabular domain

If the tabular regime data is missing, run:
  python scripts/experiment_c_tabular_regime.py

Produces: results/h1_disagreement/
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

shift_bench_dir = Path(__file__).resolve().parent.parent


def load_and_merge_h1_data() -> pd.DataFrame:
    """Load experiment_c_real + tabular regime data, merge into unified format."""
    dfs = []

    # ── Source 1: experiment_c_real ─────────────────────────────────────────
    main_path = shift_bench_dir / "results" / "experiment_c_real" / "experiment_c_raw.csv"
    if main_path.exists():
        df1 = pd.read_csv(main_path)
        print(f"Loaded experiment_c_real: {len(df1)} rows, "
              f"datasets={sorted(df1['dataset'].unique())}")
        dfs.append(df1)
    else:
        print(f"WARNING: {main_path} not found.")

    # ── Source 2: h1_tabular_regime ─────────────────────────────────────────
    tabular_path = shift_bench_dir / "results" / "h1_tabular_regime" / "tabular_regime_raw.csv"
    if tabular_path.exists():
        df2 = pd.read_csv(tabular_path)
        tabular_datasets = sorted(df2["dataset"].unique())
        print(f"Loaded tabular regime: {len(df2)} rows, datasets={tabular_datasets}")

        # If experiment_c_real also has these datasets, use tabular regime
        # (it has adjusted n_cohorts for nontrivial active pairs)
        if len(dfs) > 0:
            existing_datasets = set(dfs[0]["dataset"].unique())
            tabular_new = set(tabular_datasets) - existing_datasets
            if tabular_new:
                # New datasets not in experiment_c_real: add all
                dfs.append(df2[df2["dataset"].isin(tabular_new)])
            else:
                # Same datasets: replace with tabular regime (better active pairs)
                # Use the best n_cohort_bins config per dataset (most active pairs)
                best_rows = []
                for ds in tabular_datasets:
                    sub = df2[df2["dataset"] == ds]
                    if "n_cohort_bins" in sub.columns:
                        active_by_bins = sub.groupby("n_cohort_bins")["either_certify"].sum()
                        best_bins = active_by_bins.idxmax()
                        best_rows.append(sub[sub["n_cohort_bins"] == best_bins])
                    else:
                        best_rows.append(sub)
                if best_rows:
                    df2_best = pd.concat(best_rows, ignore_index=True)
                    # Remove these datasets from experiment_c_real, add tabular regime
                    dfs[0] = dfs[0][~dfs[0]["dataset"].isin(tabular_datasets)]
                    dfs.append(df2_best)
                    print(f"  Replaced {tabular_datasets} with tabular regime data "
                          f"(more active pairs)")
    else:
        print(f"NOTE: Tabular regime data not found at {tabular_path}")
        print(f"  Run: python scripts/experiment_c_tabular_regime.py")
        print(f"  Without it, adult and compas will show 0 active pairs.")

    if not dfs:
        raise FileNotFoundError("No H1 data found. Run experiment_c first.")

    df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows: {len(df)}, datasets: {sorted(df['dataset'].unique())}")
    return df


def analyze_h1_disagreement():
    """Load H1 data and compute disagreement by n_eff."""
    df = load_and_merge_h1_data()

    out_dir = shift_bench_dir / "results" / "h1_disagreement"
    os.makedirs(out_dir, exist_ok=True)

    # ── Core columns ─────────────────────────────────────────────────────────
    df["disagree"] = ~df["agree"]
    df["active"] = df["either_certify"]

    df["min_neff"] = df[["ulsif_neff", "kliep_neff"]].min(axis=1)
    df["mean_neff"] = df[["ulsif_neff", "kliep_neff"]].mean(axis=1)
    df["mean_mu"]  = df[["ulsif_mu",   "kliep_mu"]].mean(axis=1)
    df["margin"]   = df["mean_mu"] - df["tau"]

    df["neff_bin"] = pd.cut(
        df["min_neff"],
        bins=[0, 5, 25, 100, 300, float("inf")],
        labels=["1-5", "5-25", "25-100", "100-300", "300+"],
    )
    df["margin_bin"] = pd.cut(
        df["margin"],
        bins=[-float("inf"), -0.2, -0.1, 0.0, 0.1, 0.2, float("inf")],
        labels=["<-0.2", "-0.2:-0.1", "-0.1:0", "0:0.1", "0.1:0.2", ">0.2"],
    )

    # ── Analysis 1: Disagreement by (dataset, n_eff bin) ────────────────────
    print(f"\n{'='*70}")
    print("DISAGREEMENT BY n_eff BIN (per dataset)")
    print(f"{'='*70}")

    by_neff = df.groupby(["dataset", "domain", "neff_bin"], observed=True).agg(
        n_pairs   =("agree", "size"),
        n_disagree=("disagree", "sum"),
        n_active  =("active", "sum"),
        disagree_rate=("disagree", "mean"),
    ).reset_index()
    by_neff["disagree_pct"] = by_neff["disagree_rate"] * 100
    by_neff.to_csv(out_dir / "disagree_by_neff_dataset.csv", index=False)

    for dataset in by_neff["dataset"].unique():
        sub = by_neff[by_neff["dataset"] == dataset]
        domain = sub.iloc[0]["domain"]
        print(f"\n  {dataset} ({domain}):")
        print(f"  {'n_eff bin':<12} {'Pairs':>6} {'Active':>6} {'Disagree':>8} {'Rate':>8}")
        print(f"  {'-'*44}")
        for _, row in sub.iterrows():
            print(f"  {row['neff_bin']:<12} {row['n_pairs']:>6.0f} {row['n_active']:>6.0f} "
                  f"{row['n_disagree']:>8.0f} {row['disagree_pct']:>7.2f}%")

    # ── Analysis 2: Aggregated by n_eff bin ─────────────────────────────────
    by_neff_agg = df.groupby("neff_bin", observed=True).agg(
        n_pairs      =("agree", "size"),
        n_disagree   =("disagree", "sum"),
        n_active     =("active", "sum"),
        disagree_rate=("disagree", "mean"),
        active_disagree_rate=(
            "disagree",
            lambda x: x[df.loc[x.index, "active"]].mean()
                      if df.loc[x.index, "active"].any() else np.nan
        ),
    ).reset_index()
    by_neff_agg.to_csv(out_dir / "disagree_by_neff_agg.csv", index=False)

    print(f"\n{'='*70}")
    print("AGGREGATED DISAGREEMENT BY n_eff BIN")
    print(f"{'='*70}")
    print(f"{'n_eff bin':<12} {'Pairs':>6} {'Active':>6} {'Disagree':>8} {'Rate':>8}")
    print("-" * 44)
    for _, row in by_neff_agg.iterrows():
        print(f"{row['neff_bin']:<12} {row['n_pairs']:>6.0f} {row['n_active']:>6.0f} "
              f"{row['n_disagree']:>8.0f} {row['disagree_rate']*100:>7.2f}%")

    # ── Analysis 3: Disagreement by margin bin ───────────────────────────────
    by_margin = df.groupby(["dataset", "margin_bin"], observed=True).agg(
        n_pairs      =("agree", "size"),
        n_disagree   =("disagree", "sum"),
        n_active     =("active", "sum"),
        disagree_rate=("disagree", "mean"),
    ).reset_index()
    by_margin.to_csv(out_dir / "disagree_by_margin.csv", index=False)

    # ── Analysis 4: Top 20 disagreements ────────────────────────────────────
    disagreements = df[df["disagree"] == True].copy()
    print(f"\n{'='*70}")
    print(f"TOTAL DISAGREEMENTS: {len(disagreements)}")
    print(f"{'='*70}")

    if len(disagreements) > 0:
        top20 = disagreements.sort_values("min_neff", ascending=False).head(20)
        cols = ["dataset", "domain", "trial_id", "cohort_id", "tau",
                "ulsif_certified", "kliep_certified",
                "ulsif_neff", "kliep_neff", "min_neff",
                "ulsif_mu", "kliep_mu", "ulsif_lb", "kliep_lb", "margin"]
        top20_out = top20[[c for c in cols if c in top20.columns]].copy()
        top20_out.to_csv(out_dir / "top20_disagreements.csv", index=False)

        print(f"\nTop 20 Disagreements (by n_eff desc):")
        print(f"{'Dataset':<8} {'Trial':>5} {'Coh':>3} {'Tau':>4} "
              f"{'uLSIF':>6} {'KLIEP':>6} {'neff_u':>7} {'neff_k':>7} "
              f"{'mu_u':>6} {'mu_k':>6} {'lb_u':>6} {'lb_k':>6}")
        print("-" * 80)
        for _, row in top20_out.iterrows():
            u_dec = "CERT" if row.get("ulsif_certified", False) else "ABS"
            k_dec = "CERT" if row.get("kliep_certified", False) else "ABS"
            print(f"{row['dataset']:<8} {row.get('trial_id', 0):>5.0f} "
                  f"{row.get('cohort_id', 0):>3.0f} {row['tau']:>4.1f} "
                  f"{u_dec:>6} {k_dec:>6} "
                  f"{row.get('ulsif_neff', np.nan):>7.1f} "
                  f"{row.get('kliep_neff', np.nan):>7.1f} "
                  f"{row.get('ulsif_mu', np.nan):>6.3f} "
                  f"{row.get('kliep_mu', np.nan):>6.3f} "
                  f"{row.get('ulsif_lb', np.nan):>6.3f} "
                  f"{row.get('kliep_lb', np.nan):>6.3f}")

    # ── Analysis 5: Per-dataset summary ─────────────────────────────────────
    dataset_summary = df.groupby(["dataset", "domain"]).agg(
        total_pairs   =("agree", "size"),
        total_active  =("active", "sum"),
        total_disagree=("disagree", "sum"),
        overall_agree_pct=(
            "agree", lambda x: x.mean() * 100
        ),
        active_disagree_pct=(
            "disagree",
            lambda x: x[df.loc[x.index, "active"]].mean() * 100
                      if df.loc[x.index, "active"].any() else np.nan
        ),
        mean_neff   =("min_neff", "mean"),
        median_neff =("min_neff", "median"),
    ).reset_index()
    dataset_summary.to_csv(out_dir / "h1_dataset_summary.csv", index=False)

    print(f"\n{'='*70}")
    print("DATASET-LEVEL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<10} {'Domain':<10} {'Pairs':>6} {'Active':>6} {'Disagree':>8} "
          f"{'Agree%':>7} {'ActDis%':>8} {'MedNeff':>8}")
    print("-" * 72)
    for _, row in dataset_summary.iterrows():
        act_dis = (f"{row['active_disagree_pct']:.1f}%"
                   if not np.isnan(row["active_disagree_pct"]) else "N/A")
        print(f"{row['dataset']:<10} {row['domain']:<10} {row['total_pairs']:>6.0f} "
              f"{row['total_active']:>6.0f} {row['total_disagree']:>8.0f} "
              f"{row['overall_agree_pct']:>6.1f}% {act_dis:>8} "
              f"{row['median_neff']:>8.1f}")

    # ── Summary note ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    tabular_active = dataset_summary[
        dataset_summary["domain"] == "tabular"
    ]["total_active"].sum()
    if tabular_active == 0:
        print("NOTICE: No tabular active pairs found.")
        print("  Run: python scripts/experiment_c_tabular_regime.py")
        print("  Then re-run this script.")
    else:
        print(f"Tabular active pairs: {tabular_active:.0f}")

    print(f"\nSaved all results to {out_dir}/")
    return df


if __name__ == "__main__":
    analyze_h1_disagreement()

    # Run cross_domain supplement to provide tabular active pairs at natural n_eff
    try:
        from analysis_h1_supplement_crossdomain import load_crossdomain_h1, print_and_save, combine_with_experiment_c
        print(f"\n{'='*70}")
        print("RUNNING CROSS-DOMAIN SUPPLEMENT (tabular active pairs)")
        print(f"{'='*70}")
        wide = load_crossdomain_h1()
        cd_by_neff = print_and_save(wide)
        combine_with_experiment_c(cd_by_neff)
    except Exception as e:
        print(f"\nNote: Cross-domain supplement skipped: {e}")
