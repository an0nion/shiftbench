"""
H3 Regression Analysis
========================
Runs cert_rate ~ log(n_eff) + shift_metric + domain regression
to show n_eff explains most variance in cross-domain certification difficulty.

Uses existing cross_domain_real results + computes shift metrics per dataset.

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


def compute_shift_metrics_per_dataset():
    """Compute two-sample AUC shift metric for each dataset."""
    from shiftbench.data import load_dataset
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier

    datasets_to_check = [
        ("bace", "molecular"), ("bbbp", "molecular"),
        ("clintox", "molecular"), ("sider", "molecular"),
        ("adult", "tabular"), ("compas", "tabular"),
        ("bank", "tabular"), ("german_credit", "tabular"),
        ("heart_disease", "tabular"), ("diabetes", "tabular"),
        ("imdb", "text"), ("yelp", "text"),
        ("amazon", "text"), ("civil_comments", "text"),
        ("twitter", "text"),
    ]

    results = []
    for ds_name, domain in datasets_to_check:
        try:
            X, y, cohorts, splits = load_dataset(ds_name)
        except Exception as e:
            print(f"  Skip {ds_name}: {e}")
            continue

        cal_mask = (splits["split"] == "cal").values
        test_mask = (splits["split"] == "test").values

        X_cal = X[cal_mask]
        X_test = X[test_mask]

        # Two-sample AUC: train classifier to distinguish cal from test
        n_cal = len(X_cal)
        n_test = len(X_test)

        # Subsample for speed if needed
        max_n = 3000
        if n_cal > max_n:
            idx = np.random.RandomState(42).choice(n_cal, max_n, replace=False)
            X_cal_sub = X_cal[idx]
        else:
            X_cal_sub = X_cal

        if n_test > max_n:
            idx = np.random.RandomState(43).choice(n_test, max_n, replace=False)
            X_test_sub = X_test[idx]
        else:
            X_test_sub = X_test

        # Combine and label
        X_combined = np.vstack([X_cal_sub, X_test_sub])
        y_domain = np.concatenate([
            np.zeros(len(X_cal_sub)),
            np.ones(len(X_test_sub))
        ])

        # Train/val split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_combined, y_domain, test_size=0.3, random_state=42, stratify=y_domain
        )

        try:
            clf = GradientBoostingClassifier(
                n_estimators=50, max_depth=3, random_state=42
            )
            clf.fit(X_tr, y_tr)
            probs = clf.predict_proba(X_val)[:, 1]

            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_val, probs)
        except Exception:
            auc = 0.5

        # Shift magnitude = |AUC - 0.5| * 2 (rescaled to [0, 1])
        shift_magnitude = abs(auc - 0.5) * 2

        results.append({
            "dataset": ds_name,
            "domain": domain,
            "two_sample_auc": auc,
            "shift_magnitude": shift_magnitude,
            "n_cal": n_cal,
            "n_test": n_test,
            "n_features": X.shape[1],
        })
        print(f"  {ds_name}: AUC={auc:.3f}, shift={shift_magnitude:.3f}")

    return pd.DataFrame(results)


def run_h3_regression():
    """Run regression analysis for H3."""
    out_dir = shift_bench_dir / "results" / "h3_regression"
    os.makedirs(out_dir, exist_ok=True)

    # Load cross-domain results
    raw_path = shift_bench_dir / "results" / "cross_domain_real" / "cross_domain_raw_results.csv"
    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found")
        return

    raw = pd.read_csv(raw_path)
    print(f"Loaded {len(raw)} raw decisions")

    # Compute per-dataset metrics
    dataset_stats = raw.groupby(["dataset", "domain", "method"]).agg(
        cert_rate=("decision", lambda x: (x == "CERTIFY").mean()),
        mean_n_eff=("n_eff", "mean"),
        median_n_eff=("n_eff", "median"),
        mean_mu_hat=("mu_hat", "mean"),
        mean_lb=("lower_bound", "mean"),
        n_decisions=("decision", "size"),
    ).reset_index()

    # Average across methods for the regression
    dataset_avg = dataset_stats.groupby(["dataset", "domain"]).agg(
        cert_rate=("cert_rate", "mean"),
        mean_n_eff=("mean_n_eff", "mean"),
        median_n_eff=("median_n_eff", "mean"),
        n_decisions=("n_decisions", "sum"),
    ).reset_index()

    print(f"\nDataset-level metrics ({len(dataset_avg)} datasets):")
    print(dataset_avg.to_string(index=False))

    # Compute shift metrics
    print("\nComputing shift metrics...")
    shift_df = compute_shift_metrics_per_dataset()
    shift_df.to_csv(out_dir / "shift_metrics.csv", index=False)

    # Merge
    merged = dataset_avg.merge(shift_df, on=["dataset", "domain"], how="inner")
    print(f"\nMerged: {len(merged)} datasets")

    if len(merged) < 3:
        print("Not enough datasets for regression")
        return

    # Add log(n_eff)
    merged["log_neff"] = np.log1p(merged["mean_n_eff"])

    # Domain dummies
    merged["is_text"] = (merged["domain"] == "text").astype(int)
    merged["is_tabular"] = (merged["domain"] == "tabular").astype(int)

    merged.to_csv(out_dir / "regression_data.csv", index=False)

    # === Regression models ===
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    results = []

    # Model 1: cert_rate ~ log(n_eff) only
    X1 = merged[["log_neff"]].values
    y = merged["cert_rate"].values
    m1 = LinearRegression().fit(X1, y)
    r2_1 = r2_score(y, m1.predict(X1))
    results.append({
        "model": "cert_rate ~ log(n_eff)",
        "r2": r2_1,
        "coefficients": f"log_neff={m1.coef_[0]:.4f}, intercept={m1.intercept_:.4f}",
        "n_features": 1,
    })

    # Model 2: cert_rate ~ log(n_eff) + shift_magnitude
    X2 = merged[["log_neff", "shift_magnitude"]].values
    m2 = LinearRegression().fit(X2, y)
    r2_2 = r2_score(y, m2.predict(X2))
    results.append({
        "model": "cert_rate ~ log(n_eff) + shift",
        "r2": r2_2,
        "coefficients": f"log_neff={m2.coef_[0]:.4f}, shift={m2.coef_[1]:.4f}, intercept={m2.intercept_:.4f}",
        "n_features": 2,
    })

    # Model 3: cert_rate ~ log(n_eff) + shift + domain
    X3 = merged[["log_neff", "shift_magnitude", "is_text", "is_tabular"]].values
    m3 = LinearRegression().fit(X3, y)
    r2_3 = r2_score(y, m3.predict(X3))
    results.append({
        "model": "cert_rate ~ log(n_eff) + shift + domain",
        "r2": r2_3,
        "coefficients": (f"log_neff={m3.coef_[0]:.4f}, shift={m3.coef_[1]:.4f}, "
                        f"is_text={m3.coef_[2]:.4f}, is_tabular={m3.coef_[3]:.4f}, "
                        f"intercept={m3.intercept_:.4f}"),
        "n_features": 4,
    })

    # Model 4: cert_rate ~ domain only (ablation)
    X4 = merged[["is_text", "is_tabular"]].values
    m4 = LinearRegression().fit(X4, y)
    r2_4 = r2_score(y, m4.predict(X4))
    results.append({
        "model": "cert_rate ~ domain (ablation)",
        "r2": r2_4,
        "coefficients": f"is_text={m4.coef_[0]:.4f}, is_tabular={m4.coef_[1]:.4f}, intercept={m4.intercept_:.4f}",
        "n_features": 2,
    })

    # Model 5: cert_rate ~ shift_magnitude only (ablation)
    X5 = merged[["shift_magnitude"]].values
    m5 = LinearRegression().fit(X5, y)
    r2_5 = r2_score(y, m5.predict(X5))
    results.append({
        "model": "cert_rate ~ shift (ablation)",
        "r2": r2_5,
        "coefficients": f"shift={m5.coef_[0]:.4f}, intercept={m5.intercept_:.4f}",
        "n_features": 1,
    })

    reg_df = pd.DataFrame(results)
    reg_df.to_csv(out_dir / "regression_results.csv", index=False)

    # Print results
    print(f"\n{'='*70}")
    print("H3 REGRESSION ANALYSIS: cert_rate explanatory models")
    print(f"{'='*70}")
    print(f"{'Model':<45} {'R2':>8} {'#feat':>6}")
    print("-" * 62)
    for _, row in reg_df.iterrows():
        print(f"{row['model']:<45} {row['r2']:>8.4f} {row['n_features']:>6}")

    print(f"\nKey finding: log(n_eff) alone explains R2={r2_1:.4f}")
    print(f"Adding shift: R2={r2_2:.4f} (delta={r2_2-r2_1:.4f})")
    print(f"Adding domain: R2={r2_3:.4f} (delta={r2_3-r2_2:.4f})")
    print(f"Domain only (no n_eff): R2={r2_4:.4f}")
    print(f"\n=> n_eff ablation: removing n_eff from full model loses {r2_3-r2_4:.4f} R2")

    # Scatter data for plotting
    scatter = merged[["dataset", "domain", "cert_rate", "log_neff",
                       "mean_n_eff", "shift_magnitude"]].copy()
    scatter.to_csv(out_dir / "scatter_data.csv", index=False)
    print(f"\nSaved scatter data: {out_dir / 'scatter_data.csv'}")

    print(f"\n{'='*70}")
    return reg_df


if __name__ == "__main__":
    run_h3_regression()
