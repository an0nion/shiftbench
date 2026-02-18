"""
H1 Disagreement-by-n_eff Analysis
====================================
Analyzes when KLIEP and uLSIF disagree, conditioned on n_eff bins
and estimated margin (mu_hat - tau).

Uses existing experiment_c raw data.

Produces: results/h1_disagreement/
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

shift_bench_dir = Path(__file__).resolve().parent.parent


def analyze_h1_disagreement():
    """Load experiment_c raw data and compute disagreement by n_eff."""
    raw_path = shift_bench_dir / "results" / "experiment_c_real" / "experiment_c_raw.csv"

    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found. Run experiment_c first.")
        return

    df = pd.read_csv(raw_path)
    print(f"Loaded {len(df)} rows from {raw_path}")
    print(f"Columns: {list(df.columns)}")

    out_dir = shift_bench_dir / "results" / "h1_disagreement"
    os.makedirs(out_dir, exist_ok=True)

    # Compute disagreement
    df["disagree"] = ~df["agree"]
    df["active"] = df["either_certify"]

    # Use min(ulsif_neff, kliep_neff) as the "limiting" n_eff
    df["min_neff"] = df[["ulsif_neff", "kliep_neff"]].min(axis=1)
    df["mean_neff"] = df[["ulsif_neff", "kliep_neff"]].mean(axis=1)

    # Margin: how far mu_hat is above tau (average across methods)
    df["mean_mu"] = df[["ulsif_mu", "kliep_mu"]].mean(axis=1)
    df["margin"] = df["mean_mu"] - df["tau"]

    # n_eff bins
    df["neff_bin"] = pd.cut(
        df["min_neff"],
        bins=[0, 5, 25, 100, 300, float("inf")],
        labels=["1-5", "5-25", "25-100", "100-300", "300+"],
    )

    # Margin bins
    df["margin_bin"] = pd.cut(
        df["margin"],
        bins=[-float("inf"), -0.2, -0.1, 0.0, 0.1, 0.2, float("inf")],
        labels=["<-0.2", "-0.2 to -0.1", "-0.1 to 0", "0 to 0.1", "0.1 to 0.2", ">0.2"],
    )

    # ===== Analysis 1: Disagreement by n_eff bin per dataset =====
    print(f"\n{'='*70}")
    print("DISAGREEMENT BY n_eff BIN (per dataset)")
    print(f"{'='*70}")

    by_neff = df.groupby(["dataset", "domain", "neff_bin"], observed=True).agg(
        n_pairs=("agree", "size"),
        n_disagree=("disagree", "sum"),
        n_active=("active", "sum"),
        disagree_rate=("disagree", "mean"),
    ).reset_index()

    by_neff["disagree_pct"] = by_neff["disagree_rate"] * 100
    by_neff.to_csv(out_dir / "disagree_by_neff_dataset.csv", index=False)

    for dataset in by_neff["dataset"].unique():
        sub = by_neff[by_neff["dataset"] == dataset]
        print(f"\n  {dataset} ({sub.iloc[0]['domain']}):")
        print(f"  {'n_eff bin':<12} {'Pairs':>6} {'Active':>6} {'Disagree':>8} {'Rate':>8}")
        print(f"  {'-'*44}")
        for _, row in sub.iterrows():
            print(f"  {row['neff_bin']:<12} {row['n_pairs']:>6.0f} {row['n_active']:>6.0f} "
                  f"{row['n_disagree']:>8.0f} {row['disagree_pct']:>7.2f}%")

    # ===== Analysis 2: Disagreement by n_eff bin (aggregated) =====
    by_neff_agg = df.groupby("neff_bin", observed=True).agg(
        n_pairs=("agree", "size"),
        n_disagree=("disagree", "sum"),
        n_active=("active", "sum"),
        disagree_rate=("disagree", "mean"),
        active_disagree_rate=("disagree", lambda x: x[df.loc[x.index, "active"]].mean()
                              if df.loc[x.index, "active"].any() else np.nan),
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

    # ===== Analysis 3: Disagreement by margin bin =====
    by_margin = df.groupby(["dataset", "margin_bin"], observed=True).agg(
        n_pairs=("agree", "size"),
        n_disagree=("disagree", "sum"),
        n_active=("active", "sum"),
        disagree_rate=("disagree", "mean"),
    ).reset_index()

    by_margin.to_csv(out_dir / "disagree_by_margin.csv", index=False)

    # ===== Analysis 4: Top 20 disagreements =====
    disagreements = df[df["disagree"] == True].copy()
    print(f"\n{'='*70}")
    print(f"TOTAL DISAGREEMENTS: {len(disagreements)}")
    print(f"{'='*70}")

    if len(disagreements) > 0:
        # Sort by n_eff (most informative disagrees first)
        disagreements_sorted = disagreements.sort_values("min_neff", ascending=False)
        top20 = disagreements_sorted.head(20)

        top20_out = top20[[
            "dataset", "domain", "trial_id", "cohort_id", "tau",
            "ulsif_certified", "kliep_certified",
            "ulsif_neff", "kliep_neff", "min_neff",
            "ulsif_mu", "kliep_mu",
            "ulsif_lb", "kliep_lb",
            "margin",
        ]].copy()

        top20_out.to_csv(out_dir / "top20_disagreements.csv", index=False)

        print(f"\nTop 20 Disagreements (sorted by n_eff descending):")
        print(f"{'Dataset':<8} {'Trial':>5} {'Coh':>3} {'Tau':>4} "
              f"{'uLSIF':>6} {'KLIEP':>6} {'neff_u':>7} {'neff_k':>7} "
              f"{'mu_u':>6} {'mu_k':>6} {'lb_u':>6} {'lb_k':>6}")
        print("-" * 80)

        for _, row in top20_out.iterrows():
            u_dec = "CERT" if row["ulsif_certified"] else "ABS"
            k_dec = "CERT" if row["kliep_certified"] else "ABS"
            print(f"{row['dataset']:<8} {row['trial_id']:>5.0f} {row['cohort_id']:>3.0f} "
                  f"{row['tau']:>4.1f} {u_dec:>6} {k_dec:>6} "
                  f"{row['ulsif_neff']:>7.1f} {row['kliep_neff']:>7.1f} "
                  f"{row['ulsif_mu']:>6.3f} {row['kliep_mu']:>6.3f} "
                  f"{row['ulsif_lb']:>6.3f} {row['kliep_lb']:>6.3f}")

    # ===== Analysis 5: Per-dataset summary =====
    dataset_summary = df.groupby(["dataset", "domain"]).agg(
        total_pairs=("agree", "size"),
        total_active=("active", "sum"),
        total_disagree=("disagree", "sum"),
        overall_agree_pct=("agree", lambda x: x.mean() * 100),
        active_disagree_pct=("disagree", lambda x: x[df.loc[x.index, "active"]].mean() * 100
                             if df.loc[x.index, "active"].any() else np.nan),
        mean_neff=("min_neff", "mean"),
        median_neff=("min_neff", "median"),
    ).reset_index()

    dataset_summary.to_csv(out_dir / "h1_dataset_summary.csv", index=False)

    print(f"\n{'='*70}")
    print("DATASET-LEVEL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<10} {'Domain':<10} {'Pairs':>6} {'Active':>6} {'Disagree':>8} "
          f"{'Agree%':>7} {'ActDis%':>8} {'MedNeff':>8}")
    print("-" * 72)
    for _, row in dataset_summary.iterrows():
        act_dis = f"{row['active_disagree_pct']:.1f}%" if not np.isnan(row['active_disagree_pct']) else "N/A"
        print(f"{row['dataset']:<10} {row['domain']:<10} {row['total_pairs']:>6.0f} "
              f"{row['total_active']:>6.0f} {row['total_disagree']:>8.0f} "
              f"{row['overall_agree_pct']:>6.1f}% {act_dis:>8} "
              f"{row['median_neff']:>8.1f}")

    print(f"\n{'='*70}")
    print(f"Saved all results to {out_dir}/")
    return df


if __name__ == "__main__":
    analyze_h1_disagreement()
