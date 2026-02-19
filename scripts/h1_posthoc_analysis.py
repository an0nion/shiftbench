"""
H1 Post-hoc Ablation: Re-analyze existing experiment_c_real data.
Tests P1.1 (alpha sweep), P1.2 (tau density), P1.5 (bandwidth proxy),
and driver analysis -- WITHOUT re-running expensive KLIEP.

Uses stored ulsif_mu, kliep_mu, ulsif_neff, kliep_neff, ulsif_lb, kliep_lb.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

shift_bench_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(shift_bench_dir / "src"))
sys.path.insert(0, str(shift_bench_dir.parent / "ravel" / "src"))

from ravel.bounds.holm import holm_reject
from ravel.bounds.p_value import eb_p_value
from ravel.bounds.empirical_bernstein import eb_lower_bound

out_dir = shift_bench_dir / "results" / "h1_ablation"
out_dir.mkdir(exist_ok=True)

# Load existing raw results
df = pd.read_csv(shift_bench_dir / "results" / "experiment_c_real" / "experiment_c_raw.csv")
print(f"Loaded {len(df)} rows. Datasets: {sorted(df['dataset'].unique())}, "
      f"Trials: {df['trial_id'].nunique()} per dataset")

# Derive variance approximation from mu (upper bound on variance for Bernoulli)
df["ulsif_var"] = (df["ulsif_mu"] * (1 - df["ulsif_mu"])).clip(0.001, 0.25)
df["kliep_var"] = (df["kliep_mu"] * (1 - df["kliep_mu"])).clip(0.001, 0.25)
df["lb_gap"] = (df["ulsif_lb"] - df["kliep_lb"]).abs()
df["neff_ratio"] = df["ulsif_neff"] / df["kliep_neff"].clip(0.01)
df["mu_gap"] = (df["ulsif_mu"] - df["kliep_mu"]).abs()


# =============================================================================
# P1.1: Alpha sweep
# =============================================================================
print("\n" + "=" * 60)
print("P1.1: ALPHA SWEEP (post-hoc)")
print("=" * 60)

alphas = [0.001, 0.005, 0.01, 0.05, 0.10, 0.20]
alpha_rows = []
for alpha in alphas:
    for ds, ds_df in df.groupby("dataset"):
        for trial, t_df in ds_df.groupby("trial_id"):
            u_pvals = [eb_p_value(r["ulsif_mu"], r["ulsif_var"], r["ulsif_neff"], r["tau"])
                       for _, r in t_df.iterrows()]
            k_pvals = [eb_p_value(r["kliep_mu"], r["kliep_var"], r["kliep_neff"], r["tau"])
                       for _, r in t_df.iterrows()]
            u_rej = holm_reject(pd.Series(u_pvals), alpha)
            k_rej = holm_reject(pd.Series(k_pvals), alpha)
            for i, (_, r) in enumerate(t_df.iterrows()):
                alpha_rows.append({
                    "dataset": ds, "alpha": alpha,
                    "ulsif_cert": bool(u_rej.iloc[i]),
                    "kliep_cert": bool(k_rej.iloc[i]),
                    "agree": bool(u_rej.iloc[i]) == bool(k_rej.iloc[i]),
                    "lb_gap": float(r["lb_gap"]),
                    "neff_ratio": float(r["neff_ratio"]),
                    "mu_gap": float(r["mu_gap"]),
                })

alpha_df = pd.DataFrame(alpha_rows)
alpha_summary = []
for (ds, alpha), g in alpha_df.groupby(["dataset", "alpha"]):
    active = g[g["ulsif_cert"] | g["kliep_cert"]]
    n_dis = int((active["agree"] == False).sum()) if len(active) > 0 else 0
    alpha_summary.append({
        "dataset": ds, "alpha": alpha,
        "active_pairs": len(active),
        "n_disagree": n_dis,
        "agree_pct": float(active["agree"].mean() * 100) if len(active) > 0 else 100.0,
        "mean_lb_gap": float(g["lb_gap"].mean()),
        "mean_neff_ratio": float(g["neff_ratio"].mean()),
        "mean_mu_gap": float(g["mu_gap"].mean()),
    })

alpha_sum_df = pd.DataFrame(alpha_summary)
alpha_sum_df.to_csv(out_dir / "p11_alpha_summary.csv", index=False)
print(alpha_sum_df[["dataset", "alpha", "active_pairs", "n_disagree", "agree_pct"]].to_string(index=False))
print(f"\nOverall: mean LB gap={df['lb_gap'].mean():.4f}, "
      f"neff_ratio={df['neff_ratio'].mean():.4f}, mu_gap={df['mu_gap'].mean():.4f}")


# =============================================================================
# P1.2: Tau grid density (using tau subsets)
# =============================================================================
print("\n" + "=" * 60)
print("P1.2: TAU GRID DENSITY (post-hoc)")
print("=" * 60)

tau_configs = {
    "coarse_5":     [0.5, 0.6, 0.7, 0.8, 0.9],  # original
    "medium_3":     [0.5, 0.7, 0.9],
    "low_only":     [0.5, 0.6],
    "high_only":    [0.8, 0.9],
}
alpha = 0.05
tau_rows = []
for cfg, taus in tau_configs.items():
    sub = df[df["tau"].isin(taus)]
    for ds, ds_df in sub.groupby("dataset"):
        for trial, t_df in ds_df.groupby("trial_id"):
            u_pvals = [eb_p_value(r["ulsif_mu"], r["ulsif_var"], r["ulsif_neff"], r["tau"])
                       for _, r in t_df.iterrows()]
            k_pvals = [eb_p_value(r["kliep_mu"], r["kliep_var"], r["kliep_neff"], r["tau"])
                       for _, r in t_df.iterrows()]
            u_rej = holm_reject(pd.Series(u_pvals), alpha)
            k_rej = holm_reject(pd.Series(k_pvals), alpha)
            for i, (_, r) in enumerate(t_df.iterrows()):
                tau_rows.append({
                    "cfg": cfg, "n_tau": len(taus), "dataset": ds,
                    "ulsif_cert": bool(u_rej.iloc[i]),
                    "kliep_cert": bool(k_rej.iloc[i]),
                    "agree": bool(u_rej.iloc[i]) == bool(k_rej.iloc[i]),
                })

tau_df = pd.DataFrame(tau_rows)
tau_summary = []
for (cfg, n_tau, ds), g in tau_df.groupby(["cfg", "n_tau", "dataset"]):
    active = g[g["ulsif_cert"] | g["kliep_cert"]]
    n_dis = int((active["agree"] == False).sum()) if len(active) > 0 else 0
    tau_summary.append({
        "cfg": cfg, "n_tau": n_tau, "dataset": ds,
        "active": len(active),
        "n_disagree": n_dis,
        "agree_pct": float(active["agree"].mean() * 100) if len(active) > 0 else 100.0,
    })

tau_sum_df = pd.DataFrame(tau_summary)
tau_sum_df.to_csv(out_dir / "p12_tau_summary.csv", index=False)

print("BBBP (the outlier) agreement by tau subset:")
bbbp_t = tau_sum_df[tau_sum_df["dataset"] == "bbbp"]
print(bbbp_t[["cfg", "n_tau", "active", "n_disagree", "agree_pct"]].to_string(index=False))

# Tau-level breakdown for BBBP
bbbp_active = df[(df["dataset"] == "bbbp") &
                  (df["ulsif_certified"] | df["kliep_certified"])].copy()
bbbp_active["disagree"] = bbbp_active["ulsif_certified"] != bbbp_active["kliep_certified"]
print("\nBBBP disagreement by tau level:")
for tau, g in bbbp_active.groupby("tau"):
    rate = g["disagree"].mean() * 100
    print(f"  tau={tau:.1f}: {rate:.1f}% disagree ({g['disagree'].sum()}/{len(g)})")


# =============================================================================
# DRIVER ANALYSIS: What predicts disagreements?
# =============================================================================
print("\n" + "=" * 60)
print("DRIVER ANALYSIS: Predictors of KLIEP-uLSIF Disagreement")
print("=" * 60)

active_all = df[df["ulsif_certified"] | df["kliep_certified"]].copy()
active_all["disagree"] = active_all["ulsif_certified"] != active_all["kliep_certified"]
print(f"All datasets: {len(active_all)} active pairs, "
      f"{active_all['disagree'].sum()} disagree ({active_all['disagree'].mean()*100:.2f}%)")

print("\nCorrelations with disagreement (all active pairs):")
for feat in ["lb_gap", "neff_ratio", "mu_gap", "ulsif_neff", "kliep_neff"]:
    corr = active_all[feat].corr(active_all["disagree"].astype(float))
    print(f"  {feat:<15}: r={corr:+.4f}")

# BBBP specifically
bbbp_active = df[(df["dataset"] == "bbbp") &
                  (df["ulsif_certified"] | df["kliep_certified"])].copy()
bbbp_active["disagree"] = bbbp_active["ulsif_certified"] != bbbp_active["kliep_certified"]
print(f"\nBBBP only: {len(bbbp_active)} active pairs, "
      f"{bbbp_active['disagree'].sum()} disagree")
print("Correlations with disagreement (BBBP):")
for feat in ["lb_gap", "neff_ratio", "mu_gap", "ulsif_neff", "kliep_neff"]:
    corr = bbbp_active[feat].corr(bbbp_active["disagree"].astype(float))
    print(f"  {feat:<15}: r={corr:+.4f}")

# LB gap bins
print("\nBBBP disagreement rate by LB gap:")
bbbp_active["lb_gap_bin"] = pd.cut(
    bbbp_active["lb_gap"],
    bins=[0, 0.01, 0.05, 0.10, 0.20, 1.0],
    labels=["<0.01", "0.01-0.05", "0.05-0.10", "0.10-0.20", "0.20+"])
for bin_, g in bbbp_active.groupby("lb_gap_bin", observed=True):
    rate = g["disagree"].mean() * 100 if len(g) > 0 else 0
    print(f"  {bin_}: {rate:.1f}% ({g['disagree'].sum()}/{len(g)})")

# n_eff range for BBBP active pairs
print(f"\nBBBP active n_eff range: uLSIF "
      f"[{bbbp_active['ulsif_neff'].min():.1f}, {bbbp_active['ulsif_neff'].max():.1f}] "
      f"median={bbbp_active['ulsif_neff'].median():.1f}")
print(f"BBBP disagree n_eff range: uLSIF "
      f"[{bbbp_active.loc[bbbp_active['disagree'], 'ulsif_neff'].min():.1f}, "
      f"{bbbp_active.loc[bbbp_active['disagree'], 'ulsif_neff'].max():.1f}] "
      f"median={bbbp_active.loc[bbbp_active['disagree'], 'ulsif_neff'].median():.1f}")

# Save driver summary
driver_rows = [{"feature": f,
                "corr_all": float(active_all[f].corr(active_all["disagree"].astype(float))),
                "corr_bbbp": float(bbbp_active[f].corr(bbbp_active["disagree"].astype(float)))}
               for f in ["lb_gap", "neff_ratio", "mu_gap", "ulsif_neff", "kliep_neff"]]
pd.DataFrame(driver_rows).to_csv(out_dir / "disagreement_drivers.csv", index=False)

# Final summary
print("\n" + "=" * 60)
print("H1 POST-HOC ABLATION COMPLETE")
print("=" * 60)
print("Files written:")
for f in sorted(out_dir.iterdir()):
    print(f"  {f.name}")
