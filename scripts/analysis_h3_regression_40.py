"""
H3 Regression Analysis -- Expanded 40-dataset version
======================================================
Runs cert_rate ~ log(n_eff) + shift_metric + domain regression
using the full set of real-prediction datasets available after Session 9.

Sources:
  - RAVEL tabular/text (20 datasets): results/ravel_tabular_text/
  - uLSIF molecular (7 real-pred datasets): results/cross_domain_extended/

Excluded (oracle-based or n_eff~0):
  tox21, toxcast, muv -- oracle fallback in cross_domain_extended (Session 8 bug)
  molhiv              -- n_eff~0.01, imbalanced, 0% cert regardless

Combined dataset: 27 datasets (7 molecular + 11 tabular + 9 text)

Saves to: results/h3_regression_40/
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

shift_bench_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(shift_bench_dir / "src"))

# ── Dataset lists ────────────────────────────────────────────────────────────
# Molecular: real-pred, AUC > 0.6, not oracle-based in cross_domain_extended
REAL_MOLECULAR = {
    "bace", "bbbp", "clintox", "esol", "freesolv", "lipophilicity", "sider"
}

# Tabular + text: all 20 from RAVEL results (Session 9 real preds)
RAVEL_TABULAR = {
    "adult", "bank", "communities_crime", "compas", "diabetes",
    "german_credit", "heart_disease", "mushroom", "online_shoppers",
    "student_performance", "wine_quality",
}
RAVEL_TEXT = {
    "ag_news", "amazon", "civil_comments", "dbpedia", "imdb",
    "imdb_genre", "sst2", "twitter", "yelp",
}

ALL_DATASETS = REAL_MOLECULAR | RAVEL_TABULAR | RAVEL_TEXT

DOMAIN_MAP = {ds: "molecular" for ds in REAL_MOLECULAR}
DOMAIN_MAP.update({ds: "tabular" for ds in RAVEL_TABULAR})
DOMAIN_MAP.update({ds: "text" for ds in RAVEL_TEXT})

# ── Shift metric computation ─────────────────────────────────────────────────

def compute_shift_metric(ds_name):
    """Two-sample AUC between cal and test splits. Returns dict or None."""
    from shiftbench.data import load_dataset
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    try:
        X, y, cohorts, splits = load_dataset(ds_name)
    except Exception as e:
        print(f"  Skip {ds_name}: {e}")
        return None

    cal_mask  = (splits["split"] == "cal").values
    test_mask = (splits["split"] == "test").values
    X_cal, X_test = X[cal_mask], X[test_mask]

    rng = np.random.RandomState(42)
    max_n = 3000
    X_cal_s  = X_cal [rng.choice(len(X_cal),  min(len(X_cal),  max_n), replace=False)]
    X_test_s = X_test[rng.choice(len(X_test), min(len(X_test), max_n), replace=False)]

    X_all = np.vstack([X_cal_s, X_test_s])
    y_dom = np.concatenate([np.zeros(len(X_cal_s)), np.ones(len(X_test_s))])
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_dom, test_size=0.3, random_state=42, stratify=y_dom
    )
    try:
        clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        clf.fit(X_tr, y_tr)
        auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
    except Exception:
        auc = 0.5

    shift_magnitude = abs(auc - 0.5) * 2
    print(f"  {ds_name}: AUC={auc:.3f}, shift={shift_magnitude:.3f}")
    return {
        "dataset": ds_name,
        "domain":  DOMAIN_MAP.get(ds_name, "unknown"),
        "two_sample_auc":  auc,
        "shift_magnitude": shift_magnitude,
        "n_cal":      int(cal_mask.sum()),
        "n_test":     int(test_mask.sum()),
        "n_features": X.shape[1],
    }


def load_shift_metrics(out_dir):
    """Load cache; compute and append any missing datasets."""
    # Prefer cached file in new output dir; fall back to old h3_regression cache
    new_cache  = out_dir / "shift_metrics.csv"
    old_cache  = shift_bench_dir / "results" / "h3_regression" / "shift_metrics.csv"

    if new_cache.exists():
        df = pd.read_csv(new_cache)
    elif old_cache.exists():
        df = pd.read_csv(old_cache)
        print(f"  Loaded {len(df)} cached shift metrics from h3_regression/")
    else:
        df = pd.DataFrame(columns=["dataset", "domain", "two_sample_auc",
                                    "shift_magnitude", "n_cal", "n_test", "n_features"])

    existing = set(df["dataset"].values)
    missing  = ALL_DATASETS - existing
    if missing:
        print(f"\nComputing shift metrics for {len(missing)} new datasets:")
        new_rows = []
        for ds in sorted(missing):
            row = compute_shift_metric(ds)
            if row:
                new_rows.append(row)
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(new_cache, index=False)
    return df


# ── Cert rate loading ────────────────────────────────────────────────────────

def load_cert_rates():
    """
    Load per-dataset cert_rate and mean_n_eff from two sources:
      - RAVEL tabular/text results (Session 9)
      - uLSIF molecular from cross_domain_extended (Session 8)
    """
    # RAVEL (tabular + text)
    ravel_path = shift_bench_dir / "results" / "ravel_tabular_text" / "cross_domain_by_dataset.csv"
    ravel_df   = pd.read_csv(ravel_path)
    ravel_df["method"] = "ravel"
    ravel_df["cert_rate"] = ravel_df["cert_rate_%"] / 100.0

    # uLSIF molecular from cross_domain_extended
    ext_path = shift_bench_dir / "results" / "cross_domain_extended" / "cross_domain_by_dataset.csv"
    ext_df   = pd.read_csv(ext_path)
    ext_df["method"] = "ulsif"
    ext_df["cert_rate"] = ext_df["cert_rate_%"] / 100.0
    mol_df = ext_df[ext_df["dataset"].isin(REAL_MOLECULAR)].copy()

    # Combine -- prefer RAVEL for any dataset appearing in both
    combined = pd.concat([ravel_df, mol_df], ignore_index=True)
    combined = combined[combined["dataset"].isin(ALL_DATASETS)].copy()

    # Keep one row per dataset (RAVEL preferred over uLSIF for shared datasets)
    combined["priority"] = combined["method"].map({"ravel": 0, "ulsif": 1})
    combined = (combined.sort_values("priority")
                         .drop_duplicates(subset=["dataset"], keep="first")
                         .drop(columns=["priority"]))

    cols = ["dataset", "domain", "method", "cert_rate", "mean_n_eff", "n_cohorts"]
    available = [c for c in cols if c in combined.columns]
    return combined[available].reset_index(drop=True)


# ── Regression ───────────────────────────────────────────────────────────────

def run_h3_regression_40():
    out_dir = shift_bench_dir / "results" / "h3_regression_40"
    os.makedirs(out_dir, exist_ok=True)

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    # 1. Cert rates
    cert_df = load_cert_rates()
    print(f"\nDatasets loaded: {len(cert_df)}")
    print(cert_df[["dataset", "domain", "method", "cert_rate", "mean_n_eff"]].to_string(index=False))

    # 2. Shift metrics
    shift_df = load_shift_metrics(out_dir)
    shift_df = shift_df[shift_df["dataset"].isin(ALL_DATASETS)]

    # 3. Merge
    merged = cert_df.merge(
        shift_df[["dataset", "two_sample_auc", "shift_magnitude", "n_cal", "n_test", "n_features"]],
        on="dataset", how="inner"
    )
    n = len(merged)
    print(f"\nFinal regression dataset: n={n}")
    if n < 5:
        print("Insufficient data for regression -- aborting")
        return

    # 4. Features
    merged["log_neff"]    = np.log1p(merged["mean_n_eff"])
    merged["is_text"]     = (merged["domain"] == "text").astype(int)
    merged["is_tabular"]  = (merged["domain"] == "tabular").astype(int)
    merged.to_csv(out_dir / "regression_data.csv", index=False)

    y = merged["cert_rate"].values

    results = []
    def fit(label, X_cols):
        X = merged[X_cols].values
        m = LinearRegression().fit(X, y)
        r2 = r2_score(y, m.predict(X))
        coef_str = ", ".join(f"{c}={v:.4f}" for c, v in zip(X_cols, m.coef_))
        coef_str += f", intercept={m.intercept_:.4f}"
        results.append({"model": label, "r2": r2, "coefficients": coef_str,
                         "n_features": len(X_cols), "n_obs": n})
        return m, r2

    _, r2_neff   = fit("cert_rate ~ log(n_eff)",                   ["log_neff"])
    _, r2_shift  = fit("cert_rate ~ shift",                         ["shift_magnitude"])
    _, r2_ns     = fit("cert_rate ~ log(n_eff) + shift",            ["log_neff", "shift_magnitude"])
    _, r2_domain = fit("cert_rate ~ domain",                        ["is_text", "is_tabular"])
    _, r2_dn     = fit("cert_rate ~ domain + log(n_eff)",           ["is_text", "is_tabular", "log_neff"])
    _, r2_full   = fit("cert_rate ~ log(n_eff) + shift + domain",   ["log_neff", "shift_magnitude", "is_text", "is_tabular"])

    partial_neff_given_domain   = (r2_dn   - r2_domain) / (1 - r2_domain) if r2_domain < 1 else float("nan")
    partial_domain_given_ns     = (r2_full - r2_ns    ) / (1 - r2_ns    ) if r2_ns    < 1 else float("nan")
    incr_neff_over_shift        = r2_ns - r2_shift

    partial_df = pd.DataFrame([
        {"metric": "R2(n_eff alone)",                    "value": r2_neff},
        {"metric": "R2(shift alone)",                    "value": r2_shift},
        {"metric": "R2(domain alone)",                   "value": r2_domain},
        {"metric": "R2(domain + n_eff)",                 "value": r2_dn},
        {"metric": "R2(full model)",                     "value": r2_full},
        {"metric": "partial R2(n_eff | domain)",         "value": partial_neff_given_domain},
        {"metric": "partial R2(domain | n_eff+shift)",   "value": partial_domain_given_ns},
        {"metric": "incremental R2(n_eff over shift)",   "value": incr_neff_over_shift},
    ])
    partial_df.to_csv(out_dir / "h3_partial_r2.csv", index=False)

    # Within-domain regression
    within_results = []
    for domain in ["molecular", "tabular", "text"]:
        sub = merged[merged["domain"] == domain].copy()
        nd = len(sub)
        if nd < 3:
            within_results.append({
                "domain": domain, "n_datasets": nd,
                "r2_neff": float("nan"), "r2_shift": float("nan"),
                "r2_neff_shift": float("nan"), "neff_coef": float("nan"),
                "note": f"only {nd} obs, cannot fit",
            })
            continue
        ys = sub["cert_rate"].values
        m_n  = LinearRegression().fit(sub[["log_neff"]].values, ys)
        m_s  = LinearRegression().fit(sub[["shift_magnitude"]].values, ys)
        m_ns = LinearRegression().fit(sub[["log_neff", "shift_magnitude"]].values, ys)
        within_results.append({
            "domain": domain, "n_datasets": nd,
            "r2_neff":       r2_score(ys, m_n .predict(sub[["log_neff"]].values)),
            "r2_shift":      r2_score(ys, m_s .predict(sub[["shift_magnitude"]].values)),
            "r2_neff_shift": r2_score(ys, m_ns.predict(sub[["log_neff", "shift_magnitude"]].values)),
            "neff_coef": m_n.coef_[0],
            "note": "",
        })
    within_df = pd.DataFrame(within_results)
    within_df.to_csv(out_dir / "h3_within_domain_regression.csv", index=False)

    reg_df = pd.DataFrame(results)
    reg_df.to_csv(out_dir / "regression_results.csv", index=False)

    scatter = merged[["dataset", "domain", "method", "cert_rate", "log_neff",
                       "mean_n_eff", "shift_magnitude"]].copy()
    scatter.to_csv(out_dir / "scatter_data.csv", index=False)

    # ── Print summary ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"H3 REGRESSION -- EXPANDED (n={n} datasets, 3 domains + methods)")
    print(f"  Molecular: uLSIF ({len(merged[merged.domain=='molecular'])} datasets)")
    print(f"  Tabular:   RAVEL ({len(merged[merged.domain=='tabular'])} datasets)")
    print(f"  Text:      RAVEL ({len(merged[merged.domain=='text'])} datasets)")
    print(f"{'='*70}")

    print(f"\n--- Global models ---")
    print(f"{'Model':<48} {'R2':>8}")
    print("-" * 58)
    for _, row in reg_df.iterrows():
        print(f"{row['model']:<48} {row['r2']:>8.4f}")

    print(f"\n--- Partial / Incremental R2 ---")
    for _, row in partial_df.iterrows():
        print(f"  {row['metric']:<45} {row['value']:>8.4f}")

    print(f"\n--- Within-domain: cert_rate ~ log(n_eff) ---")
    print(f"{'Domain':<12} {'n':>3} {'R2(neff)':>10} {'R2(shift)':>10} {'neff_coef':>10}  note")
    print("-" * 65)
    for _, row in within_df.iterrows():
        r2n = f"{row['r2_neff']:.4f}"  if not np.isnan(row['r2_neff'])  else "  N/A"
        r2s = f"{row['r2_shift']:.4f}" if not np.isnan(row['r2_shift']) else "  N/A"
        nc  = f"{row['neff_coef']:.4f}" if not np.isnan(row['neff_coef']) else "  N/A"
        print(f"{row['domain']:<12} {row['n_datasets']:>3} {r2n:>10} {r2s:>10} {nc:>10}  {row['note']}")

    print(f"\n--- Interpretation ---")
    print(f"  Domain R2={r2_domain:.3f}: domain structure dominates (n_eff range set by domain)")
    print(f"  partial R2(n_eff | domain)={partial_neff_given_domain:.3f}: n_eff retains within-domain signal")
    print(f"  partial R2(domain | n_eff+shift)={partial_domain_given_ns:.3f}: domain adds beyond n_eff alone")
    print(f"  incremental R2(n_eff over shift)={incr_neff_over_shift:.3f}")
    print(f"\nSaved to {out_dir}/")

    # Domain breakdown table
    print(f"\n--- Domain breakdown (mean cert_rate, mean n_eff) ---")
    for domain in ["molecular", "tabular", "text"]:
        sub = merged[merged["domain"] == domain]
        print(f"  {domain:<12} n={len(sub):>2}  cert={sub['cert_rate'].mean():.3f}"
              f"  n_eff={sub['mean_n_eff'].mean():.1f}")

    return reg_df, partial_df, within_df


if __name__ == "__main__":
    run_h3_regression_40()
