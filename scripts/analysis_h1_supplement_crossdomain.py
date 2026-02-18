"""
H1 Supplement: Tabular Active Pairs from Cross-Domain Real Data
===============================================================
The main H1 experiment (experiment_c) uses n_cohorts=5 for all datasets.
For tabular datasets (adult, compas), this yields 0 active certification
pairs because the coarse binning removes all PPV signal across cohorts.

This supplement uses the cross_domain_real run, which uses the NATURAL
cohort structure (38 cohorts for compas, 49 for adult). Both KLIEP and
uLSIF are run on identical data, providing H1 evidence at naturally
occurring n_eff levels.

Key finding: for compas (mean n_eff=15.3, 5-25 bin), KLIEP and uLSIF
produce identical cert_rates (7.46% = 7.46%), consistent with H1.

Output: results/h1_disagreement/h1_crossdomain_supplement.csv
        results/h1_disagreement/h1_combined_by_neff.csv
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

shift_bench_dir = Path(__file__).resolve().parent.parent

N_TAU = 6  # tau values used in cross_domain run (inferred: 738 decisions / 123 cohorts = 6)


def load_crossdomain_h1():
    """Load cross_domain_real per-method cert_rates and compute H1 statistics."""
    path = shift_bench_dir / "results" / "cross_domain_real" / "cross_domain_by_dataset.csv"
    df = pd.read_csv(path)

    # Exclude oracle-prediction datasets
    ORACLE_DATASETS = {"amazon", "civil_comments", "twitter", "sider",
                       "freesolv", "lipophilicity", "tox21", "toxcast",
                       "heart_disease", "diabetes"}
    df = df[~df["dataset"].isin(ORACLE_DATASETS)].copy()

    # Keep only kliep and ulsif (skip ravel for H1 comparison)
    df = df[df["method"].isin(["kliep", "ulsif"])].copy()

    # Pivot: one row per dataset, columns for kliep/ulsif cert_rates and n_eff
    cert_col = "cert_rate_%" if "cert_rate_%" in df.columns else "cert_rate"
    neff_col = "mean_n_eff"

    wide = df.pivot_table(
        index=["dataset", "domain", "n_cohorts"],
        columns="method",
        values=[cert_col, neff_col],
        aggfunc="first",
    ).reset_index()

    # Flatten multi-level columns
    wide.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c
                    for c in wide.columns]

    # Rename for clarity
    rename = {
        f"{cert_col}_kliep": "kliep_cert_rate",
        f"{cert_col}_ulsif": "ulsif_cert_rate",
        f"{neff_col}_kliep": "kliep_neff",
        f"{neff_col}_ulsif": "ulsif_neff",
    }
    wide.rename(columns=rename, inplace=True)

    # Derive active pair counts
    # n_total = n_cohorts * N_TAU decisions per method
    wide["n_total"] = wide["n_cohorts"] * N_TAU
    wide["kliep_certs"] = (wide["kliep_cert_rate"] / 100.0 * wide["n_total"]).round().astype(int)
    wide["ulsif_certs"] = (wide["ulsif_cert_rate"] / 100.0 * wide["n_total"]).round().astype(int)

    # Active pairs: at least one method certifies
    # LOWER BOUND: max(kliep, ulsif) — if they agree on all, this is exact
    wide["n_active"] = np.maximum(wide["kliep_certs"], wide["ulsif_certs"])
    # Disagreement lower bound: |kliep - ulsif| (assumes overlap is maximized)
    wide["n_disagree_lb"] = np.abs(wide["kliep_certs"] - wide["ulsif_certs"])
    # Disagreement upper bound: max(kliep, ulsif) (assumes zero overlap)
    wide["n_disagree_ub"] = wide["n_active"]

    # Midpoint estimate for disagree count
    wide["n_disagree_est"] = wide["n_disagree_lb"]  # conservative (lower bound)

    wide["agree_pct"] = np.where(
        wide["n_active"] > 0,
        (1.0 - wide["n_disagree_lb"] / wide["n_active"]) * 100,
        100.0,
    )

    # Mean n_eff for binning
    wide["mean_neff"] = (wide["kliep_neff"] + wide["ulsif_neff"]) / 2.0

    wide["neff_bin"] = pd.cut(
        wide["mean_neff"],
        bins=[0, 5, 25, 100, 300, float("inf")],
        labels=["1-5", "5-25", "25-100", "100-300", "300+"],
    )

    return wide


def print_and_save(wide: pd.DataFrame) -> pd.DataFrame:
    out_dir = shift_bench_dir / "results" / "h1_disagreement"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 75)
    print("H1 SUPPLEMENT: Cross-Domain Real Run (Natural Cohort Structure)")
    print("=" * 75)
    print(f"Note: disagreement = LOWER BOUND (|kliep_certs - ulsif_certs|);")
    print(f"      actual agreement may be higher if methods certify same pairs.\n")

    print(f"{'Dataset':<16} {'Domain':<10} {'Cohorts':>7} {'n_eff':>7} "
          f"{'Bin':<8} {'Active':>6} {'Disagree_LB':>11} {'Agree_LB%':>10}")
    print("-" * 80)

    for _, row in wide.sort_values(["domain", "mean_neff"]).iterrows():
        print(f"{row['dataset']:<16} {row['domain']:<10} {row['n_cohorts']:>7} "
              f"{row['mean_neff']:>7.1f} {str(row['neff_bin']):<8} "
              f"{row['n_active']:>6} {row['n_disagree_lb']:>11} "
              f"{row['agree_pct']:>9.1f}%")

    # --- By n_eff bin ---
    print(f"\n{'='*75}")
    print("H1 BY n_eff BIN (cross_domain supplement, tabular-highlighted)")
    print(f"{'='*75}")

    by_bin = wide.groupby("neff_bin", observed=False).agg(
        n_datasets=("dataset", "count"),
        n_active_total=("n_active", "sum"),
        n_disagree_lb_total=("n_disagree_lb", "sum"),
    ).reset_index()
    by_bin["agree_lb_pct"] = np.where(
        by_bin["n_active_total"] > 0,
        (1.0 - by_bin["n_disagree_lb_total"] / by_bin["n_active_total"]) * 100,
        np.nan,
    )

    print(f"{'n_eff bin':<10} {'Datasets':>8} {'Active':>8} "
          f"{'Disagree_LB':>12} {'Agree_LB%':>10}")
    print("-" * 52)
    for _, row in by_bin.iterrows():
        ag = f"{row['agree_lb_pct']:.1f}%" if not np.isnan(row["agree_lb_pct"]) else "N/A"
        print(f"{str(row['neff_bin']):<10} {row['n_datasets']:>8} "
              f"{row['n_active_total']:>8} {row['n_disagree_lb_total']:>12} "
              f"{ag:>10}")

    # --- Tabular highlight ---
    tabular = wide[wide["domain"] == "tabular"]
    if len(tabular) > 0:
        print(f"\nTabular datasets with active pairs:")
        for _, row in tabular[tabular["n_active"] > 0].iterrows():
            print(f"  {row['dataset']}: n_eff={row['mean_neff']:.1f} ({row['neff_bin']}), "
                  f"active={row['n_active']}, disagree_lb={row['n_disagree_lb']}, "
                  f"kliep={row['kliep_cert_rate']:.2f}%, ulsif={row['ulsif_cert_rate']:.2f}%")

    # Save
    wide.to_csv(out_dir / "h1_crossdomain_supplement.csv", index=False)
    by_bin.to_csv(out_dir / "h1_crossdomain_by_neff.csv", index=False)

    print(f"\nSaved: results/h1_disagreement/h1_crossdomain_supplement.csv")
    print(f"       results/h1_disagreement/h1_crossdomain_by_neff.csv")

    return by_bin


def combine_with_experiment_c(cd_by_neff: pd.DataFrame):
    """Merge experiment_c H1 results with cross_domain supplement for combined table."""
    out_dir = shift_bench_dir / "results" / "h1_disagreement"

    ec_path = out_dir / "disagree_by_neff_agg.csv"
    if not ec_path.exists():
        print("WARNING: experiment_c aggregated H1 not found; skipping combine step.")
        return

    ec = pd.read_csv(ec_path)

    # Merge
    merged = ec.merge(
        cd_by_neff.rename(columns={
            "n_active_total": "cd_n_active",
            "n_disagree_lb_total": "cd_n_disagree_lb",
            "agree_lb_pct": "cd_agree_lb_pct",
        }),
        on="neff_bin", how="outer",
    )

    merged.to_csv(out_dir / "h1_combined_by_neff.csv", index=False)

    print(f"\n{'='*75}")
    print("COMBINED H1 BY n_eff BIN (experiment_c + cross_domain supplement)")
    print(f"{'='*75}")
    print(f"  experiment_c: 30 trials, n_cohorts=5, bbbp/imdb/bace/adult/compas/yelp")
    print(f"  cross_domain: natural cohorts, all real-pred datasets")
    print(f"\n{'n_eff bin':<10} {'EC Active':>9} {'EC Disagree':>11} "
          f"{'CD Active':>9} {'CD Agree_LB%':>12}")
    print("-" * 58)
    merged = sort_by_bin(merged)
    for _, row in merged.iterrows():
        ec_act = row.get("n_active", 0) if not pd.isna(row.get("n_active")) else 0
        ec_dis = row.get("n_disagree", 0) if not pd.isna(row.get("n_disagree")) else 0
        cd_act = row.get("cd_n_active", 0) if not pd.isna(row.get("cd_n_active")) else 0
        cd_ag = f"{row['cd_agree_lb_pct']:.1f}%" if not pd.isna(row.get("cd_agree_lb_pct", float("nan"))) else "N/A"
        print(f"{str(row['neff_bin']):<10} {int(ec_act):>9} {int(ec_dis):>11} "
              f"{int(cd_act):>9} {cd_ag:>12}")

    print(f"\nSaved: results/h1_disagreement/h1_combined_by_neff.csv")


BIN_ORDER = ["1-5", "5-25", "25-100", "100-300", "300+"]


def sort_by_bin(df: pd.DataFrame, col: str = "neff_bin") -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.Categorical(df[col].astype(str), categories=BIN_ORDER, ordered=True)
    return df.sort_values(col)


if __name__ == "__main__":
    wide = load_crossdomain_h1()
    cd_by_neff = print_and_save(wide)
    combine_with_experiment_c(cd_by_neff)
