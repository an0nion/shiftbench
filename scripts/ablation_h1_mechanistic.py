"""
H1 Mechanistic Ablation: WHY do KLIEP and uLSIF agree?
========================================================
Tests all pre-registered predictions P1.1-P1.5:

  P1.1 - Alpha sweep: agreement decreases with stricter alpha?
  P1.2 - Tau grid density: finer grid reveals more disagreements?
  P1.3 - Hoeffding bound: even looser bound => still agree?
  P1.4 - Bootstrap percentile CI: tighter bound => breaks agreement?
  P1.5 - Bandwidth sensitivity: vary kernel bandwidth manually

Additional:
  - Weight correlation: how similar are KLIEP/uLSIF weights numerically?
  - Lower-bound gap analysis: does LB gap predict disagreements?
  - n_eff contribution: is agreement driven by n_eff similarity?
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

shift_bench_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(shift_bench_dir / "src"))
sys.path.insert(0, str(shift_bench_dir.parent / "ravel" / "src"))
sys.path.insert(0, str(shift_bench_dir / "scripts"))

from experiment_a_real_data_calibration import load_real_dataset, DATASET_CONFIG
from ravel.bounds.empirical_bernstein import eb_lower_bound
from ravel.bounds.p_value import eb_p_value
from ravel.bounds.weighted_stats import weighted_stats_01
from ravel.bounds.holm import holm_reject
from shiftbench.baselines.ulsif import uLSIFBaseline
from shiftbench.baselines.kliep import KLIEPBaseline
from shiftbench.baselines.kliep_fast import KLIEPFastBaseline


# =============================================================================
# ALTERNATIVE BOUND FAMILIES
# =============================================================================

def hoeffding_lower_bound(mu_hat, n_eff, delta):
    """Hoeffding bound for [0,1]-bounded mean. Variance-free, looser than EB."""
    if n_eff <= 1 or np.isnan(mu_hat):
        return np.nan
    t = np.sqrt(np.log(1.0 / max(delta, 1e-300)) / (2 * n_eff))
    return max(0.0, mu_hat - t)


def hoeffding_p_value(mu_hat, n_eff, tau):
    """Invert Hoeffding bound to get p-value for H0: PPV < tau."""
    if np.isnan(mu_hat) or n_eff <= 1 or mu_hat < tau:
        return 1.0
    # LB(delta) = mu_hat - sqrt(ln(1/delta)/(2*n_eff)) >= tau
    # => ln(1/delta) <= 2*n_eff*(mu_hat - tau)^2
    # => delta >= exp(-2*n_eff*(mu_hat - tau)^2)
    gap = mu_hat - tau
    if gap <= 0:
        return 1.0
    return np.exp(-2 * n_eff * gap * gap)


def bootstrap_lower_bound(y, w, tau, alpha, n_boot=500, seed=42):
    """Bootstrap percentile lower bound for weighted mean.

    Tighter than EB (data-adaptive), but no formal guarantee.
    Returns (lower_bound, p_value_approx).
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    if n < 2:
        return np.nan, 1.0

    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yb = y[idx]
        wb = w[idx]
        if wb.sum() > 0:
            boot_means[b] = np.average(yb, weights=wb)
        else:
            boot_means[b] = np.nan

    boot_means = boot_means[~np.isnan(boot_means)]
    if len(boot_means) < 100:
        return np.nan, 1.0

    lb = np.percentile(boot_means, alpha * 100)
    # p-value approx: fraction of bootstraps below tau
    p_val = np.mean(boot_means < tau)
    return lb, max(p_val, 1.0 / (len(boot_means) + 1))


# =============================================================================
# CORE COMPARISON ENGINE
# =============================================================================

def run_comparison_trial(data, n_cal, tau_grid, alpha, trial_id, seed,
                         bound_family="eb", ulsif_sigma=None, kliep_sigma=None):
    """Run a single trial comparing KLIEP vs uLSIF under specified config.

    Args:
        bound_family: "eb" | "hoeffding" | "bootstrap"
        ulsif_sigma: override bandwidth for uLSIF (None = median heuristic)
        kliep_sigma: override bandwidth for KLIEP (None = median heuristic)
    """
    rng = np.random.RandomState(seed)

    n_source = len(data["X_source"])
    sample_idx = rng.choice(n_source, size=min(n_cal, n_source), replace=True)

    X_cal = data["X_source"][sample_idx]
    y_cal = data["y_source"][sample_idx]
    cohorts_cal = data["cohorts_source"][sample_idx]
    preds_cal = data["preds_source"][sample_idx]
    X_test = data["X_test"]

    n_basis = min(100, len(X_cal))

    # Estimate weights with both methods
    try:
        ulsif = uLSIFBaseline(n_basis=n_basis, sigma=ulsif_sigma, lambda_=0.1,
                               random_state=seed)
        w_ulsif = ulsif.estimate_weights(X_cal, X_test)
    except Exception:
        w_ulsif = np.ones(len(X_cal))

    try:
        kliep = KLIEPFastBaseline(n_basis=n_basis, sigma=kliep_sigma, max_iter=5000,
                                   tol=1e-6, n_subsample_cal=500, n_subsample_target=500,
                                   random_state=seed)
        w_kliep = kliep.estimate_weights(X_cal, X_test)
    except Exception:
        w_kliep = np.ones(len(X_cal))

    # Weight diagnostics
    w_corr = np.corrcoef(w_ulsif, w_kliep)[0, 1] if len(w_ulsif) > 1 else np.nan
    w_mae = np.mean(np.abs(w_ulsif - w_kliep))
    w_max_ratio = np.max(np.abs(w_ulsif - w_kliep) / (np.maximum(w_ulsif, w_kliep) + 1e-10))

    # Per-method computations
    cohort_ids = np.unique(cohorts_cal)

    ulsif_pvals = []
    kliep_pvals = []
    test_info = []  # (cid, tau, ulsif_stats, kliep_stats)

    for cid in cohort_ids:
        cmask = cohorts_cal == cid
        pmask = cmask & (preds_cal == 1)
        y_pos = y_cal[pmask]
        wu_pos = w_ulsif[pmask].copy()
        wk_pos = w_kliep[pmask].copy()

        if len(y_pos) < 2:
            for tau in tau_grid:
                ulsif_pvals.append(1.0)
                kliep_pvals.append(1.0)
                test_info.append({
                    "cid": cid, "tau": tau, "n_pos": len(y_pos),
                    "u_mu": np.nan, "k_mu": np.nan,
                    "u_neff": np.nan, "k_neff": np.nan,
                    "u_lb": np.nan, "k_lb": np.nan,
                })
            continue

        # Normalize weights
        if wu_pos.mean() > 0:
            wu_pos = wu_pos / wu_pos.mean()
        if wk_pos.mean() > 0:
            wk_pos = wk_pos / wk_pos.mean()

        u_stats = weighted_stats_01(y_pos, wu_pos)
        k_stats = weighted_stats_01(y_pos, wk_pos)

        for tau in tau_grid:
            if bound_family == "eb":
                u_pval = eb_p_value(u_stats.mu, u_stats.var, u_stats.n_eff, tau)
                k_pval = eb_p_value(k_stats.mu, k_stats.var, k_stats.n_eff, tau)
                u_lb = eb_lower_bound(u_stats.mu, u_stats.var, u_stats.n_eff, alpha)
                k_lb = eb_lower_bound(k_stats.mu, k_stats.var, k_stats.n_eff, alpha)
            elif bound_family == "hoeffding":
                u_pval = hoeffding_p_value(u_stats.mu, u_stats.n_eff, tau)
                k_pval = hoeffding_p_value(k_stats.mu, k_stats.n_eff, tau)
                u_lb = hoeffding_lower_bound(u_stats.mu, u_stats.n_eff, alpha)
                k_lb = hoeffding_lower_bound(k_stats.mu, k_stats.n_eff, alpha)
            elif bound_family == "bootstrap":
                u_lb, u_pval = bootstrap_lower_bound(y_pos, wu_pos, tau, alpha,
                                                      seed=seed + hash(str(cid)) % 10000)
                k_lb, k_pval = bootstrap_lower_bound(y_pos, wk_pos, tau, alpha,
                                                      seed=seed + hash(str(cid)) % 10000 + 1)
            else:
                raise ValueError(f"Unknown bound family: {bound_family}")

            ulsif_pvals.append(u_pval)
            kliep_pvals.append(k_pval)
            test_info.append({
                "cid": cid, "tau": tau, "n_pos": len(y_pos),
                "u_mu": u_stats.mu, "k_mu": k_stats.mu,
                "u_var": u_stats.var, "k_var": k_stats.var,
                "u_neff": u_stats.n_eff, "k_neff": k_stats.n_eff,
                "u_lb": u_lb, "k_lb": k_lb,
            })

    # Holm correction (independently for each method)
    u_rejected = holm_reject(pd.Series(ulsif_pvals), alpha)
    k_rejected = holm_reject(pd.Series(kliep_pvals), alpha)

    results = []
    for i, info in enumerate(test_info):
        u_cert = bool(u_rejected.iloc[i])
        k_cert = bool(k_rejected.iloc[i])
        agree = u_cert == k_cert

        results.append({
            "dataset": data["dataset_name"],
            "domain": data["config"]["domain"],
            "trial_id": trial_id,
            "cohort_id": info["cid"],
            "tau": info["tau"],
            "bound_family": bound_family,
            "alpha": alpha,
            "n_tau": len(tau_grid),
            "ulsif_certified": u_cert,
            "kliep_certified": k_cert,
            "agree": agree,
            "both_certify": u_cert and k_cert,
            "disagree_type": ("none" if agree else
                              "ulsif_only" if u_cert else "kliep_only"),
            # Weight diagnostics
            "weight_corr": w_corr,
            "weight_mae": w_mae,
            # Bound diagnostics
            "ulsif_mu": info.get("u_mu", np.nan),
            "kliep_mu": info.get("k_mu", np.nan),
            "mu_gap": abs(info.get("u_mu", 0) - info.get("k_mu", 0)),
            "ulsif_neff": info.get("u_neff", np.nan),
            "kliep_neff": info.get("k_neff", np.nan),
            "neff_ratio": (info.get("u_neff", 1) / max(info.get("k_neff", 1), 0.01)
                           if not np.isnan(info.get("u_neff", np.nan)) else np.nan),
            "ulsif_lb": info.get("u_lb", np.nan),
            "kliep_lb": info.get("k_lb", np.nan),
            "lb_gap": abs(info.get("u_lb", 0) - info.get("k_lb", 0)),
            "n_pos": info.get("n_pos", 0),
        })

    return results


# =============================================================================
# ABLATION P1.1: ALPHA SWEEP
# =============================================================================

def run_p11_alpha_sweep(datasets_data, n_trials=30, seed=42):
    """P1.1: Does agreement decrease with stricter alpha (tighter bounds)?"""
    print("\n" + "=" * 70)
    print("P1.1: ALPHA SWEEP")
    print("=" * 70)

    alphas = [0.001, 0.005, 0.01, 0.05, 0.10, 0.20]
    tau_grid = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

    all_results = []
    for alpha in alphas:
        print(f"\n  alpha={alpha}:")
        for ds_name, data in datasets_data.items():
            for trial in range(n_trials):
                trial_seed = seed + trial * 137
                results = run_comparison_trial(
                    data, n_cal=len(data["X_source"]), tau_grid=tau_grid,
                    alpha=alpha, trial_id=trial, seed=trial_seed,
                    bound_family="eb",
                )
                all_results.extend(results)

            n_active = sum(1 for r in all_results
                          if r["alpha"] == alpha and r["dataset"] == ds_name
                          and (r["ulsif_certified"] or r["kliep_certified"]))
            n_disagree = sum(1 for r in all_results
                            if r["alpha"] == alpha and r["dataset"] == ds_name
                            and not r["agree"]
                            and (r["ulsif_certified"] or r["kliep_certified"]))
            total = sum(1 for r in all_results
                       if r["alpha"] == alpha and r["dataset"] == ds_name)
            agree_pct = (1 - n_disagree / max(n_active, 1)) * 100
            print(f"    {ds_name}: {total} pairs, {n_active} active, "
                  f"{n_disagree} disagree, agreement={agree_pct:.1f}%")

    return pd.DataFrame(all_results)


# =============================================================================
# ABLATION P1.2: TAU GRID DENSITY
# =============================================================================

def run_p12_tau_density(datasets_data, n_trials=30, seed=42):
    """P1.2: Does finer tau grid reveal more disagreements?"""
    print("\n" + "=" * 70)
    print("P1.2: TAU GRID DENSITY")
    print("=" * 70)

    alpha = 0.05
    grids = {
        "coarse_5": np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
        "medium_10": np.linspace(0.45, 0.95, 10),
        "fine_20": np.linspace(0.40, 0.95, 20),
        "ultra_50": np.linspace(0.35, 0.95, 50),
    }

    all_results = []
    for grid_name, tau_grid in grids.items():
        print(f"\n  Grid: {grid_name} ({len(tau_grid)} points):")
        for ds_name, data in datasets_data.items():
            for trial in range(n_trials):
                trial_seed = seed + trial * 137
                results = run_comparison_trial(
                    data, n_cal=len(data["X_source"]), tau_grid=tau_grid,
                    alpha=alpha, trial_id=trial, seed=trial_seed,
                    bound_family="eb",
                )
                for r in results:
                    r["grid_name"] = grid_name
                all_results.extend(results)

            sub = [r for r in all_results
                   if r.get("grid_name") == grid_name and r["dataset"] == ds_name]
            n_active = sum(1 for r in sub if r["ulsif_certified"] or r["kliep_certified"])
            n_disagree = sum(1 for r in sub if not r["agree"]
                            and (r["ulsif_certified"] or r["kliep_certified"]))
            print(f"    {ds_name}: {n_active} active, {n_disagree} disagree")

    return pd.DataFrame(all_results)


# =============================================================================
# ABLATION P1.3/P1.4: BOUND FAMILY COMPARISON
# =============================================================================

def run_p13_p14_bound_families(datasets_data, n_trials=30, seed=42):
    """P1.3 (Hoeffding) and P1.4 (Bootstrap): Different bounds change agreement?"""
    print("\n" + "=" * 70)
    print("P1.3/P1.4: BOUND FAMILY COMPARISON")
    print("=" * 70)

    alpha = 0.05
    tau_grid = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    families = ["eb", "hoeffding", "bootstrap"]

    all_results = []
    for family in families:
        print(f"\n  Bound: {family}:")
        for ds_name, data in datasets_data.items():
            for trial in range(n_trials):
                trial_seed = seed + trial * 137
                results = run_comparison_trial(
                    data, n_cal=len(data["X_source"]), tau_grid=tau_grid,
                    alpha=alpha, trial_id=trial, seed=trial_seed,
                    bound_family=family,
                )
                all_results.extend(results)

            sub = [r for r in all_results
                   if r["bound_family"] == family and r["dataset"] == ds_name]
            n_active = sum(1 for r in sub if r["ulsif_certified"] or r["kliep_certified"])
            n_disagree = sum(1 for r in sub if not r["agree"]
                            and (r["ulsif_certified"] or r["kliep_certified"]))
            agree_pct = (1 - n_disagree / max(n_active, 1)) * 100
            print(f"    {ds_name}: {n_active} active, {n_disagree} disagree, "
                  f"agreement={agree_pct:.1f}%")

    return pd.DataFrame(all_results)


# =============================================================================
# ABLATION P1.5: BANDWIDTH SENSITIVITY
# =============================================================================

def run_p15_bandwidth(datasets_data, n_trials=20, seed=42):
    """P1.5: Vary kernel bandwidth manually (instead of median heuristic)."""
    print("\n" + "=" * 70)
    print("P1.5: BANDWIDTH SENSITIVITY")
    print("=" * 70)

    alpha = 0.05
    tau_grid = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    # Multipliers relative to median heuristic
    sigma_multipliers = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]

    all_results = []
    for ds_name, data in datasets_data.items():
        # First compute median heuristic sigma
        from sklearn.metrics import pairwise_distances
        n_sample = min(500, len(data["X_source"]))
        rng_tmp = np.random.RandomState(seed)
        idx = rng_tmp.choice(len(data["X_source"]), size=n_sample, replace=False)
        dists = pairwise_distances(data["X_source"][idx]).ravel()
        dists = dists[dists > 0]
        median_sigma = np.median(dists) if len(dists) > 0 else 1.0
        print(f"\n  {ds_name}: median_sigma={median_sigma:.4f}")

        for mult in sigma_multipliers:
            sigma = median_sigma * mult
            print(f"    sigma={sigma:.4f} (x{mult}):")

            for trial in range(n_trials):
                trial_seed = seed + trial * 137

                # Same sigma for both methods
                results = run_comparison_trial(
                    data, n_cal=len(data["X_source"]), tau_grid=tau_grid,
                    alpha=alpha, trial_id=trial, seed=trial_seed,
                    bound_family="eb",
                    ulsif_sigma=sigma, kliep_sigma=sigma,
                )
                for r in results:
                    r["sigma_mult"] = mult
                    r["sigma"] = sigma
                    r["median_sigma"] = median_sigma
                all_results.extend(results)

            sub = [r for r in all_results
                   if r.get("sigma_mult") == mult and r["dataset"] == ds_name]
            n_active = sum(1 for r in sub if r["ulsif_certified"] or r["kliep_certified"])
            n_disagree = sum(1 for r in sub if not r["agree"]
                            and (r["ulsif_certified"] or r["kliep_certified"]))
            w_corr = np.nanmean([r["weight_corr"] for r in sub])
            print(f"      active={n_active}, disagree={n_disagree}, "
                  f"w_corr={w_corr:.3f}")

    return pd.DataFrame(all_results)


# =============================================================================
# ANALYSIS: WHAT DRIVES DISAGREEMENT?
# =============================================================================

def analyze_disagreement_drivers(df):
    """Analyze which variables predict KLIEP-uLSIF disagreement."""
    print("\n" + "=" * 70)
    print("DISAGREEMENT DRIVER ANALYSIS")
    print("=" * 70)

    # Only look at active pairs (at least one method certifies)
    active = df[(df["ulsif_certified"] | df["kliep_certified"])].copy()

    if len(active) == 0:
        print("  No active pairs found.")
        return pd.DataFrame()

    active["disagree"] = (~active["agree"]).astype(int)

    print(f"\n  Total active pairs: {len(active)}")
    print(f"  Total disagreements: {active['disagree'].sum()} "
          f"({active['disagree'].mean()*100:.2f}%)")

    # Feature importance via logistic regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    features = ["weight_corr", "mu_gap", "neff_ratio", "lb_gap", "n_pos",
                "ulsif_neff", "kliep_neff"]

    valid = active.dropna(subset=features + ["disagree"])
    if len(valid) < 20 or valid["disagree"].sum() < 5:
        print("  Too few disagreements for driver analysis.")
        # Still compute correlations
        print("\n  Correlations with disagreement:")
        for feat in features:
            if feat in valid.columns:
                corr = valid[feat].corr(valid["disagree"])
                print(f"    {feat}: r={corr:.4f}")
        return pd.DataFrame()

    X_feat = valid[features].values
    y_feat = valid["disagree"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)

    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X_scaled, y_feat)

    importance = pd.DataFrame({
        "feature": features,
        "coefficient": lr.coef_[0],
        "abs_coeff": np.abs(lr.coef_[0]),
    }).sort_values("abs_coeff", ascending=False)

    print("\n  Logistic regression feature importance (standardized):")
    for _, row in importance.iterrows():
        print(f"    {row['feature']:<15} coef={row['coefficient']:>8.4f} "
              f"|coef|={row['abs_coeff']:>8.4f}")

    # Group analysis
    print("\n  Disagreement rate by weight correlation bin:")
    active["w_corr_bin"] = pd.cut(active["weight_corr"],
                                   bins=[-1, 0.5, 0.8, 0.9, 0.95, 0.99, 1.01],
                                   labels=["<0.5", "0.5-0.8", "0.8-0.9",
                                           "0.9-0.95", "0.95-0.99", "0.99+"])
    for bin_name, group in active.groupby("w_corr_bin", observed=True):
        if len(group) > 0:
            dr = group["disagree"].mean() * 100
            print(f"    {bin_name}: {dr:.2f}% disagree ({len(group)} pairs)")

    print("\n  Disagreement rate by LB gap:")
    active["lb_gap_bin"] = pd.cut(active["lb_gap"],
                                   bins=[0, 0.01, 0.05, 0.10, 0.20, 1.0],
                                   labels=["<0.01", "0.01-0.05", "0.05-0.10",
                                           "0.10-0.20", "0.20+"])
    for bin_name, group in active.groupby("lb_gap_bin", observed=True):
        if len(group) > 0:
            dr = group["disagree"].mean() * 100
            print(f"    {bin_name}: {dr:.2f}% disagree ({len(group)} pairs)")

    return importance


# =============================================================================
# MAIN
# =============================================================================

def main():
    out_dir = shift_bench_dir / "results" / "h1_ablation"
    os.makedirs(out_dir, exist_ok=True)

    data_dir = str(shift_bench_dir / "data" / "processed")
    model_dir = str(shift_bench_dir / "models")

    # Focused subset: IMDB (text, 100% agreement), BBBP (molecular, 88% agreement - the outlier)
    # Skip COMPAS (KLIEP too slow ~57 min even with fast mode on 4937 samples)
    # Use these two to understand WHY BBBP disagrees and why others don't
    datasets = ["bbbp", "imdb", "bace", "adult"]
    datasets_data = {}

    for ds in datasets:
        try:
            data = load_real_dataset(ds, data_dir, model_dir)
            datasets_data[ds] = data
            print(f"  Loaded {ds}: {len(data['X_source'])} source, "
                  f"{len(np.unique(data['cohorts_source']))} cohorts")
        except Exception as e:
            print(f"  [SKIP] {ds}: {e}")

    if not datasets_data:
        print("No datasets loaded. Exiting.")
        return

    t0 = time.time()

    # =========================================================================
    # P1.1: Alpha sweep
    # =========================================================================
    df_alpha = run_p11_alpha_sweep(datasets_data, n_trials=8, seed=42)
    df_alpha.to_csv(out_dir / "p11_alpha_sweep_raw.csv", index=False)

    # Summarize
    alpha_summary = []
    for (ds, alpha), group in df_alpha.groupby(["dataset", "alpha"]):
        active = group[group["ulsif_certified"] | group["kliep_certified"]]
        alpha_summary.append({
            "dataset": ds, "alpha": alpha,
            "total_pairs": len(group),
            "active_pairs": len(active),
            "n_disagree": (~active["agree"]).sum() if len(active) > 0 else 0,
            "agree_pct": active["agree"].mean() * 100 if len(active) > 0 else 100,
            "mean_w_corr": group["weight_corr"].mean(),
            "mean_lb_gap": group["lb_gap"].mean(),
        })
    pd.DataFrame(alpha_summary).to_csv(out_dir / "p11_alpha_summary.csv", index=False)

    # =========================================================================
    # P1.2: Tau grid density
    # =========================================================================
    df_tau = run_p12_tau_density(datasets_data, n_trials=8, seed=42)
    df_tau.to_csv(out_dir / "p12_tau_density_raw.csv", index=False)

    tau_summary = []
    for (ds, grid), group in df_tau.groupby(["dataset", "grid_name"]):
        active = group[group["ulsif_certified"] | group["kliep_certified"]]
        tau_summary.append({
            "dataset": ds, "grid_name": grid,
            "n_tau": group["n_tau"].iloc[0],
            "active_pairs": len(active),
            "n_disagree": (~active["agree"]).sum() if len(active) > 0 else 0,
            "agree_pct": active["agree"].mean() * 100 if len(active) > 0 else 100,
        })
    pd.DataFrame(tau_summary).to_csv(out_dir / "p12_tau_summary.csv", index=False)

    # =========================================================================
    # P1.3/P1.4: Bound families
    # =========================================================================
    df_bounds = run_p13_p14_bound_families(datasets_data, n_trials=8, seed=42)
    df_bounds.to_csv(out_dir / "p13_p14_bounds_raw.csv", index=False)

    bound_summary = []
    for (ds, bf), group in df_bounds.groupby(["dataset", "bound_family"]):
        active = group[group["ulsif_certified"] | group["kliep_certified"]]
        bound_summary.append({
            "dataset": ds, "bound_family": bf,
            "active_pairs": len(active),
            "n_disagree": (~active["agree"]).sum() if len(active) > 0 else 0,
            "agree_pct": active["agree"].mean() * 100 if len(active) > 0 else 100,
            "mean_w_corr": group["weight_corr"].mean(),
        })
    pd.DataFrame(bound_summary).to_csv(out_dir / "p13_p14_bound_summary.csv", index=False)

    # =========================================================================
    # P1.5: Bandwidth sensitivity
    # =========================================================================
    df_bw = run_p15_bandwidth(datasets_data, n_trials=5, seed=42)
    df_bw.to_csv(out_dir / "p15_bandwidth_raw.csv", index=False)

    bw_summary = []
    for (ds, mult), group in df_bw.groupby(["dataset", "sigma_mult"]):
        active = group[group["ulsif_certified"] | group["kliep_certified"]]
        bw_summary.append({
            "dataset": ds, "sigma_mult": mult,
            "sigma": group["sigma"].iloc[0],
            "active_pairs": len(active),
            "n_disagree": (~active["agree"]).sum() if len(active) > 0 else 0,
            "agree_pct": active["agree"].mean() * 100 if len(active) > 0 else 100,
            "mean_w_corr": group["weight_corr"].mean(),
        })
    pd.DataFrame(bw_summary).to_csv(out_dir / "p15_bandwidth_summary.csv", index=False)

    # =========================================================================
    # DRIVER ANALYSIS (on combined data)
    # =========================================================================
    df_all = pd.concat([df_alpha, df_bounds], ignore_index=True)
    importance = analyze_disagreement_drivers(df_all)
    if len(importance) > 0:
        importance.to_csv(out_dir / "disagreement_drivers.csv", index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"H1 ABLATION COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"Results saved to {out_dir}/")

    # =========================================================================
    # FINAL SUMMARY TABLE
    # =========================================================================
    print(f"\n{'='*70}")
    print("H1 ABLATION SUMMARY")
    print(f"{'='*70}")

    print("\nP1.1 Alpha Sweep (agreement % on active pairs):")
    print(f"  {'Dataset':<10}", end="")
    for alpha in [0.001, 0.01, 0.05, 0.10, 0.20]:
        print(f"  a={alpha:<6}", end="")
    print()
    for ds in datasets_data:
        print(f"  {ds:<10}", end="")
        for alpha in [0.001, 0.01, 0.05, 0.10, 0.20]:
            sub = [s for s in alpha_summary
                   if s["dataset"] == ds and s["alpha"] == alpha]
            if sub:
                print(f"  {sub[0]['agree_pct']:>6.1f}%", end="")
            else:
                print(f"  {'N/A':>7}", end="")
        print()

    print("\nP1.3/P1.4 Bound Families (agreement % on active pairs):")
    print(f"  {'Dataset':<10}  {'EB':>7}  {'Hoeffding':>10}  {'Bootstrap':>10}")
    for ds in datasets_data:
        vals = {}
        for bf in ["eb", "hoeffding", "bootstrap"]:
            sub = [s for s in bound_summary
                   if s["dataset"] == ds and s["bound_family"] == bf]
            vals[bf] = sub[0]["agree_pct"] if sub else float('nan')
        print(f"  {ds:<10}  {vals.get('eb', 0):>6.1f}%  {vals.get('hoeffding', 0):>9.1f}%  "
              f"{vals.get('bootstrap', 0):>9.1f}%")


if __name__ == "__main__":
    main()
