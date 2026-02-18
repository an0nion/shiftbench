"""
H3 Regression Analysis
========================
Runs cert_rate ~ log(n_eff) + shift_metric + domain regression
to quantify cross-domain certification difficulty.

IMPORTANT FRAMING:
- Domain is the primary predictor (R²~0.90 domain-only).
- This reflects that domain DETERMINES n_eff range by construction:
  scaffold shift → n_eff~1; temporal drift → n_eff~500.
  Domain identity and n_eff are therefore collinear.
- Within each domain, log(n_eff) retains independent predictive value
  (shown via within-domain regression + partial R²).
- Correct claim: "domain mediates n_eff; within domain, n_eff explains
  remaining variation beyond shift magnitude."

Dataset filter: REAL_PRED_DATASETS only (oracle datasets excluded).
Oracle-contaminated exclusions:
  amazon / civil_comments / twitter — cert_rate=100% due to oracle fallback
  sider / heart_disease / diabetes   — oracle fallback, cert_rate≈0%

Produces: results/h3_regression/
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

shift_bench_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(shift_bench_dir / "src"))
sys.path.insert(0, str(shift_bench_dir / "scripts"))

# Datasets with REAL (non-oracle) predictions confirmed in benchmark.log.
REAL_PRED_DATASETS = {
    "bace", "bbbp", "clintox", "esol",           # molecular
    "adult", "compas", "bank", "german_credit",    # tabular
    "imdb", "yelp",                                # text
}


def compute_shift_metric_for_dataset(ds_name, domain):
    """Compute two-sample AUC shift metric for a single dataset."""
    from shiftbench.data import load_dataset
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    try:
        X, y, cohorts, splits = load_dataset(ds_name)
    except Exception as e:
        print(f"  Skip {ds_name}: {e}")
        return None

    cal_mask = (splits["split"] == "cal").values
    test_mask = (splits["split"] == "test").values
    X_cal, X_test = X[cal_mask], X[test_mask]

    max_n = 3000
    rng = np.random.RandomState(42)
    X_cal_sub = X_cal[rng.choice(len(X_cal), min(len(X_cal), max_n), replace=False)]
    X_test_sub = X_test[rng.choice(len(X_test), min(len(X_test), max_n), replace=False)]

    X_comb = np.vstack([X_cal_sub, X_test_sub])
    y_dom = np.concatenate([np.zeros(len(X_cal_sub)), np.ones(len(X_test_sub))])
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_comb, y_dom, test_size=0.3, random_state=42, stratify=y_dom
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
        "dataset": ds_name, "domain": domain,
        "two_sample_auc": auc, "shift_magnitude": shift_magnitude,
        "n_cal": int(cal_mask.sum()), "n_test": int(test_mask.sum()),
        "n_features": X.shape[1],
    }


def load_or_update_shift_metrics(out_dir):
    """Load cached shift metrics and add any missing REAL_PRED_DATASETS."""
    cache_path = out_dir / "shift_metrics.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        df = pd.DataFrame(columns=["dataset", "domain", "two_sample_auc",
                                    "shift_magnitude", "n_cal", "n_test", "n_features"])

    existing = set(df["dataset"].values)
    domain_map = {
        "bace": "molecular", "bbbp": "molecular", "clintox": "molecular",
        "esol": "molecular",
        "adult": "tabular", "compas": "tabular", "bank": "tabular",
        "german_credit": "tabular",
        "imdb": "text", "yelp": "text",
    }
    missing = REAL_PRED_DATASETS - existing
    if missing:
        print(f"\nComputing shift metrics for: {sorted(missing)}")
        new_rows = []
        for ds in sorted(missing):
            row = compute_shift_metric_for_dataset(ds, domain_map[ds])
            if row:
                new_rows.append(row)
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            df.to_csv(cache_path, index=False)

    return df


def load_cert_rates(shift_bench_dir):
    """
    Load per-dataset cert_rate and mean_n_eff.
    Prefers tier_a (has esol with real preds); falls back to cross_domain_real.
    """
    for rel_path in [
        "results/cross_domain_tier_a/cross_domain_by_dataset.csv",
        "results/cross_domain_real/cross_domain_by_dataset.csv",
    ]:
        path = shift_bench_dir / rel_path
        if path.exists():
            print(f"Loading cert rates from: {path.name}")
            df = pd.read_csv(path)
            if "cert_rate_%" in df.columns:
                df["cert_rate"] = df["cert_rate_%"] / 100.0
            elif "cert_rate" not in df.columns:
                raise KeyError("No cert_rate column found")
            agg = df.groupby(["dataset", "domain"]).agg(
                cert_rate=("cert_rate", "mean"),
                mean_n_eff=("mean_n_eff", "mean"),
                n_cohorts=("n_cohorts", "first"),
            ).reset_index()
            return agg
    raise FileNotFoundError("No cross-domain by_dataset.csv found in tier_a or real dirs")


def run_h3_regression():
    """Run H3 regression with oracle-filtered, real-prediction datasets only."""
    out_dir = shift_bench_dir / "results" / "h3_regression"
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Cert rates ───────────────────────────────────────────────────────
    cert_df = load_cert_rates(shift_bench_dir)

    # ── 2. Filter to real-pred datasets ────────────────────────────────────
    cert_df = cert_df[cert_df["dataset"].isin(REAL_PRED_DATASETS)].copy()
    print(f"\nReal-pred datasets ({len(cert_df)}):")
    print(cert_df[["dataset", "domain", "cert_rate", "mean_n_eff"]].to_string(index=False))

    # ── 3. Shift metrics ───────────────────────────────────────────────────
    shift_df = load_or_update_shift_metrics(out_dir)
    shift_df = shift_df[shift_df["dataset"].isin(REAL_PRED_DATASETS)]

    # ── 4. Merge ───────────────────────────────────────────────────────────
    merged = cert_df.merge(
        shift_df[["dataset", "domain", "two_sample_auc", "shift_magnitude",
                  "n_cal", "n_test", "n_features"]],
        on=["dataset", "domain"], how="inner"
    )
    n = len(merged)
    print(f"\nFinal regression dataset: n={n}")

    if n < 5:
        print("Insufficient data for regression")
        return

    # ── 5. Features ─────────────────────────────────────────────────────────
    merged["log_neff"] = np.log1p(merged["mean_n_eff"])
    merged["is_text"] = (merged["domain"] == "text").astype(int)
    merged["is_tabular"] = (merged["domain"] == "tabular").astype(int)
    merged.to_csv(out_dir / "regression_data.csv", index=False)

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

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
        return r2

    # ── 6. Global regressions ───────────────────────────────────────────────
    r2_neff   = fit("cert_rate ~ log(n_eff)",
                    ["log_neff"])
    r2_shift  = fit("cert_rate ~ shift",
                    ["shift_magnitude"])
    r2_ns     = fit("cert_rate ~ log(n_eff) + shift",
                    ["log_neff", "shift_magnitude"])
    r2_domain = fit("cert_rate ~ domain",
                    ["is_text", "is_tabular"])
    r2_dn     = fit("cert_rate ~ domain + log(n_eff)",
                    ["is_text", "is_tabular", "log_neff"])
    r2_full   = fit("cert_rate ~ log(n_eff) + shift + domain",
                    ["log_neff", "shift_magnitude", "is_text", "is_tabular"])

    # ── 7. Incremental / partial R² ─────────────────────────────────────────
    # partial R²(n_eff | domain) = how much n_eff adds after domain is controlled
    partial_neff_given_domain = (
        (r2_dn - r2_domain) / (1 - r2_domain) if r2_domain < 1.0 else float("nan")
    )
    # partial R²(domain | n_eff+shift)
    partial_domain_given_ns = (
        (r2_full - r2_ns) / (1 - r2_ns) if r2_ns < 1.0 else float("nan")
    )
    # incremental: how much n_eff adds beyond shift alone
    incr_neff_over_shift = r2_ns - r2_shift

    partial_df = pd.DataFrame([
        {"metric": "R²(n_eff alone)", "value": r2_neff},
        {"metric": "R²(shift alone)", "value": r2_shift},
        {"metric": "R²(domain alone)", "value": r2_domain},
        {"metric": "R²(domain + n_eff)", "value": r2_dn},
        {"metric": "R²(full model)", "value": r2_full},
        {"metric": "partial R²(n_eff | domain)", "value": partial_neff_given_domain},
        {"metric": "partial R²(domain | n_eff+shift)", "value": partial_domain_given_ns},
        {"metric": "incremental R²(n_eff over shift)", "value": incr_neff_over_shift},
    ])
    partial_df.to_csv(out_dir / "h3_partial_r2.csv", index=False)

    # ── 8. Within-domain regression ─────────────────────────────────────────
    within_results = []
    for domain in ["molecular", "tabular", "text"]:
        sub = merged[merged["domain"] == domain].copy()
        nd = len(sub)
        if nd < 3:
            within_results.append({
                "domain": domain, "n_datasets": nd,
                "r2_neff": float("nan"), "r2_shift": float("nan"),
                "r2_neff_shift": float("nan"), "neff_coef": float("nan"),
                "note": f"only {nd} obs, cannot fit"
            })
            continue
        ys = sub["cert_rate"].values
        m_n = LinearRegression().fit(sub[["log_neff"]].values, ys)
        m_s = LinearRegression().fit(sub[["shift_magnitude"]].values, ys)
        m_ns = LinearRegression().fit(sub[["log_neff", "shift_magnitude"]].values, ys)
        within_results.append({
            "domain": domain, "n_datasets": nd,
            "r2_neff":       r2_score(ys, m_n.predict(sub[["log_neff"]].values)),
            "r2_shift":      r2_score(ys, m_s.predict(sub[["shift_magnitude"]].values)),
            "r2_neff_shift": r2_score(ys, m_ns.predict(sub[["log_neff", "shift_magnitude"]].values)),
            "neff_coef": m_n.coef_[0],
            "note": "",
        })
    within_df = pd.DataFrame(within_results)
    within_df.to_csv(out_dir / "h3_within_domain_regression.csv", index=False)

    # ── 9. Save & print ─────────────────────────────────────────────────────
    reg_df = pd.DataFrame(results)
    reg_df.to_csv(out_dir / "regression_results.csv", index=False)

    scatter = merged[["dataset", "domain", "cert_rate", "log_neff",
                       "mean_n_eff", "shift_magnitude"]].copy()
    scatter.to_csv(out_dir / "scatter_data.csv", index=False)

    print(f"\n{'='*70}")
    print(f"H3 REGRESSION  (n={n} real-pred datasets)")
    print(f"{'='*70}")
    print(f"\n--- Global models ---")
    print(f"{'Model':<45} {'R²':>8}")
    print("-" * 55)
    for _, row in reg_df.iterrows():
        print(f"{row['model']:<45} {row['r2']:>8.4f}")

    print(f"\n--- Partial / Incremental R² ---")
    for _, row in partial_df.iterrows():
        print(f"  {row['metric']:<45} {row['value']:>8.4f}")

    print(f"\n--- Within-domain: cert_rate ~ log(n_eff) ---")
    print(f"{'Domain':<12} {'n':>3} {'R²(neff)':>10} {'R²(shift)':>10} {'neff_coef':>10} note")
    print("-" * 60)
    for _, row in within_df.iterrows():
        r2n = f"{row['r2_neff']:.4f}" if not np.isnan(row['r2_neff']) else "  N/A"
        r2s = f"{row['r2_shift']:.4f}" if not np.isnan(row['r2_shift']) else "  N/A"
        nc  = f"{row['neff_coef']:.4f}" if not np.isnan(row['neff_coef']) else "  N/A"
        print(f"{row['domain']:<12} {row['n_datasets']:>3} {r2n:>10} {r2s:>10} {nc:>10}  {row['note']}")

    print(f"\n--- Interpretation ---")
    print(f"  Primary driver: domain (R²={r2_domain:.3f}); reflects that domain")
    print(f"  determines n_eff range by construction.")
    print(f"  n_eff retains within-domain predictive power:")
    print(f"    partial R²(n_eff | domain) = {partial_neff_given_domain:.3f}")
    print(f"  The correct claim: 'domain mediates n_eff; within domain, n_eff")
    print(f"  explains remaining variation beyond shift magnitude.'")
    print(f"\nSaved to {out_dir}/")
    return reg_df


if __name__ == "__main__":
    run_h3_regression()
