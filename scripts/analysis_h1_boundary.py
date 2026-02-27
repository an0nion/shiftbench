"""
H1 Boundary Analysis: Where do KLIEP/uLSIF disagreements live?

PI order: For BBBP disagreements (and all datasets), compute and tabulate:
  - disagreements vs |LB - tau|  (distance to certification boundary)
  - disagreements vs n_eff bins
  - disagreements vs weight CV (proxy for k-hat gate; available as n_eff ratio)

If disagreements concentrate near |LB - tau| ≈ 0, the story is:
  "EB bounds differ only at the margin; fundamental guarantee is the same."
If they spread broadly, the narrative must change.

Data source: results/experiment_c_real/experiment_c_raw.csv
             (columns: dataset, tau, cohort_id, ulsif_certified, kliep_certified,
              ulsif_lb, kliep_lb, ulsif_neff, kliep_neff, ulsif_mu, kliep_mu, ...)

Produces: results/h1_boundary/
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

shift_bench = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(shift_bench / "src"))

OUT_DIR = shift_bench / "results" / "h1_boundary"


def classify_decision(row):
    """Classify each active (either_certify) pair."""
    u = row["ulsif_certified"]
    k = row["kliep_certified"]
    if u and k:
        return "both_certify"
    elif u and not k:
        return "ulsif_only"
    elif k and not u:
        return "kliep_only"
    else:
        return "neither"


def boundary_distance(row):
    """
    Minimum distance to the certification boundary across the two methods.

    For a certifying method: LB - tau (positive, certify iff > 0).
    For a non-certifying method: LB - tau (negative).

    We report the *certifying* method's distance to the boundary, as that's
    where the difference matters.  If only one certifies, it's that LB - tau.
    If both certify, it's the minimum of the two (tightest margin).
    """
    tau = row["tau"]
    u_dist = row["ulsif_lb"] - tau if not np.isnan(row.get("ulsif_lb", float("nan"))) else float("nan")
    k_dist = row["kliep_lb"] - tau if not np.isnan(row.get("kliep_lb", float("nan"))) else float("nan")

    u = row["ulsif_certified"]
    k = row["kliep_certified"]

    if u and k:
        # Both certify — take the tighter (smaller) margin
        vals = [v for v in [u_dist, k_dist] if not np.isnan(v)]
        return min(vals) if vals else float("nan")
    elif u and not k:
        return u_dist  # uLSIF just certifies; kliep doesn't
    elif k and not u:
        return k_dist  # kliep certifies; ulsif doesn't
    else:
        return float("nan")


def neff_ratio(row):
    """Ratio of max(neff) to min(neff): measures disagreement in weight estimates."""
    u = row.get("ulsif_neff", float("nan"))
    k = row.get("kliep_neff", float("nan"))
    if np.isnan(u) or np.isnan(k) or min(u, k) <= 0:
        return float("nan")
    return max(u, k) / min(u, k)


NEFF_BINS = [0, 5, 25, 100, 300, float("inf")]
NEFF_LABELS = ["<5", "5-25", "25-100", "100-300", "300+"]

DIST_BINS  = [-float("inf"), -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, float("inf")]
DIST_LABELS = ["<-0.20", "-0.20 to -0.10", "-0.10 to -0.05",
               "-0.05 to 0", "0 to 0.05", "0.05 to 0.10",
               "0.10 to 0.20", ">0.20"]


def main():
    raw_path = shift_bench / "results" / "experiment_c_real" / "experiment_c_raw.csv"
    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found.")
        sys.exit(1)

    df = pd.read_csv(raw_path)
    print(f"Loaded experiment_c_raw: {len(df)} rows, datasets={sorted(df['dataset'].unique())}")

    # Cross-domain supplement is aggregate-level (no per-pair rows), skip for boundary analysis.
    supp_path = shift_bench / "results" / "h1_disagreement" / "h1_crossdomain_supplement.csv"
    if supp_path.exists():
        print(f"Note: crossdomain supplement is aggregate-level — skipped for per-pair analysis.")

    # ── Derived columns ───────────────────────────────────────────────────────
    df["decision_type"] = df.apply(classify_decision, axis=1)
    df["disagree"] = (df["ulsif_certified"] != df["kliep_certified"])
    df["either_certify"] = df["ulsif_certified"] | df["kliep_certified"]

    # Mean n_eff across the two methods
    df["mean_neff"] = df[["ulsif_neff", "kliep_neff"]].mean(axis=1)
    df["neff_ratio"] = df.apply(neff_ratio, axis=1)

    # Boundary distance (for active pairs)
    df["boundary_dist"] = df.apply(boundary_distance, axis=1)

    # n_eff bin (use mean n_eff)
    df["neff_bin"] = pd.cut(
        df["mean_neff"],
        bins=NEFF_BINS,
        labels=NEFF_LABELS,
        right=False,
    )

    # Boundary distance bin
    df["dist_bin"] = pd.cut(
        df["boundary_dist"],
        bins=DIST_BINS,
        labels=DIST_LABELS,
        right=True,
    )

    # ── Output directory ──────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Overall summary ────────────────────────────────────────────────────
    n_active = df["either_certify"].sum()
    n_disagree = df["disagree"].sum()
    print(f"\nOverall: {n_active} active pairs, {n_disagree} disagreements "
          f"({100*n_disagree/max(n_active,1):.1f}% of active)")

    # ── 2. Per-dataset summary ────────────────────────────────────────────────
    ds_summary = (
        df.groupby("dataset")
        .apply(lambda g: pd.Series({
            "n_pairs":      len(g),
            "n_active":     g["either_certify"].sum(),
            "n_disagree":   g["disagree"].sum(),
            "disagree_pct": 100 * g["disagree"].sum() / max(g["either_certify"].sum(), 1),
            "mean_neff":    g["mean_neff"].mean(),
        }))
        .reset_index()
    )
    ds_path = OUT_DIR / "h1_boundary_by_dataset.csv"
    ds_summary.to_csv(ds_path, index=False)
    print(f"\nPer-dataset summary:")
    print(ds_summary.to_string(index=False))
    print(f"Saved to {ds_path}")

    # ── 3. Disagreements by n_eff bin (global, across all datasets) ──────────
    active = df[df["either_certify"]].copy()
    neff_tbl = (
        active.groupby("neff_bin", observed=False)
        .agg(
            n_active   = ("either_certify", "sum"),
            n_disagree = ("disagree", "sum"),
        )
        .reset_index()
    )
    neff_tbl["disagree_pct"] = (
        100 * neff_tbl["n_disagree"] / neff_tbl["n_active"].replace(0, float("nan"))
    ).round(1)
    neff_path = OUT_DIR / "h1_boundary_by_neff.csv"
    neff_tbl.to_csv(neff_path, index=False)
    print(f"\nDisagreements by n_eff bin:")
    print(neff_tbl.to_string(index=False))
    print(f"Saved to {neff_path}")

    # ── 4. Disagreements by boundary distance (|LB - tau|) ───────────────────
    dist_tbl = (
        active.groupby("dist_bin", observed=False)
        .agg(
            n_active   = ("either_certify", "sum"),
            n_disagree = ("disagree", "sum"),
        )
        .reset_index()
    )
    dist_tbl["disagree_pct"] = (
        100 * dist_tbl["n_disagree"] / dist_tbl["n_active"].replace(0, float("nan"))
    ).round(1)
    dist_path = OUT_DIR / "h1_boundary_by_dist.csv"
    dist_tbl.to_csv(dist_path, index=False)
    print(f"\nDisagreements by boundary distance (LB - tau):")
    print(dist_tbl.to_string(index=False))
    print(f"Saved to {dist_path}")

    # ── 5. BBBP-specific breakdown ────────────────────────────────────────────
    bbbp = active[active["dataset"] == "bbbp"].copy()
    if len(bbbp) == 0:
        print("\nWARNING: No BBBP active pairs found in data.")
    else:
        bbbp_disagree = bbbp[bbbp["disagree"]]
        print(f"\nBBBP: {len(bbbp)} active pairs, {len(bbbp_disagree)} disagreements "
              f"({100*len(bbbp_disagree)/max(len(bbbp),1):.1f}%)")

        bbbp_dist = (
            bbbp.groupby("dist_bin", observed=False)
            .agg(n_active=("either_certify", "sum"),
                 n_disagree=("disagree", "sum"))
            .reset_index()
        )
        bbbp_dist["disagree_pct"] = (
            100 * bbbp_dist["n_disagree"] / bbbp_dist["n_active"].replace(0, float("nan"))
        ).round(1)

        bbbp_neff = (
            bbbp.groupby("neff_bin", observed=False)
            .agg(n_active=("either_certify", "sum"),
                 n_disagree=("disagree", "sum"))
            .reset_index()
        )
        bbbp_neff["disagree_pct"] = (
            100 * bbbp_neff["n_disagree"] / bbbp_neff["n_active"].replace(0, float("nan"))
        ).round(1)

        bbbp_path = OUT_DIR / "h1_bbbp_boundary.csv"
        bbbp_dist.to_csv(bbbp_path, index=False)
        print("BBBP by boundary distance:")
        print(bbbp_dist.to_string(index=False))
        print(f"Saved to {bbbp_path}")

        bbbp_neff_path = OUT_DIR / "h1_bbbp_by_neff.csv"
        bbbp_neff.to_csv(bbbp_neff_path, index=False)
        print("BBBP by n_eff bin:")
        print(bbbp_neff.to_string(index=False))
        print(f"Saved to {bbbp_neff_path}")

        if len(bbbp_disagree) > 0:
            print(f"\nBBBP disagree cases: mean boundary_dist={bbbp_disagree['boundary_dist'].mean():.4f}, "
                  f"mean neff_ratio={bbbp_disagree['neff_ratio'].mean():.3f}")
            print(f"Distribution of dist_bin among disagreements:")
            print(bbbp_disagree["dist_bin"].value_counts().to_string())

    # ── 6. neff_ratio (weight estimate spread) for disagreements ─────────────
    ratio_disagree = active[active["disagree"]]["neff_ratio"].dropna()
    ratio_agree    = active[~active["disagree"]]["neff_ratio"].dropna()
    if len(ratio_disagree) > 0 and len(ratio_agree) > 0:
        print(f"\nNeff ratio (max/min across methods):")
        print(f"  Disagreements: mean={ratio_disagree.mean():.3f}, "
              f"median={ratio_disagree.median():.3f}")
        print(f"  Agreements:    mean={ratio_agree.mean():.3f}, "
              f"median={ratio_agree.median():.3f}")

    # ── Save full active-pair table ───────────────────────────────────────────
    keep_cols = [c for c in [
        "dataset", "domain", "trial_id", "cohort_id", "tau",
        "ulsif_certified", "kliep_certified", "disagree", "decision_type",
        "ulsif_lb", "kliep_lb", "ulsif_neff", "kliep_neff",
        "mean_neff", "neff_ratio", "boundary_dist", "neff_bin", "dist_bin",
    ] if c in active.columns]
    full_path = OUT_DIR / "h1_boundary_full.csv"
    active[keep_cols].to_csv(full_path, index=False)
    print(f"\nFull active-pair table saved to {full_path}")


if __name__ == "__main__":
    main()
