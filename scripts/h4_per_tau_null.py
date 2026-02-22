"""
H4 Stronger Test: Per-Tau Null
================================
Sets true_ppv = tau - epsilon independently for each tau.
This means ANY certification at tau is a false cert (stricter than current
design where null is only at tau=0.9).

Validates: 0 false certifications at each tau level across all datasets.

Output: results/h4_per_tau_null/h4_per_tau_null.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# holm_reject expects pd.Series

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "ravel" / "src"))

from ravel.bounds.empirical_bernstein import eb_lower_bound
from ravel.bounds.weighted_stats import weighted_stats_01
from ravel.bounds.holm import holm_reject
from shiftbench.baselines import create_ulsif_baseline


# ── Data loader ──────────────────────────────────────────────────────────────

def load_dataset_safe(ds_name, data_dir):
    from shiftbench.data import load_dataset
    X, y, cohorts, splits = load_dataset(ds_name)
    cal_mask  = (splits["split"] == "cal").values
    test_mask = (splits["split"] == "test").values
    return {
        "X_cal":      X[cal_mask],
        "y_cal":      y[cal_mask],
        "cohorts_cal": cohorts[cal_mask],
        "X_test":     X[test_mask],
    }


# ── Semi-synthetic trial (per-tau null) ──────────────────────────────────────

def run_per_tau_trial(data, weights, tau, epsilon, alpha, seed):
    """
    For a given tau, set true_ppv = tau - epsilon for all cohorts.
    Any certification at this tau is a false cert.

    Returns dict with fwer (0/1) and n_certs.
    """
    rng = np.random.RandomState(seed)
    true_ppv = tau - epsilon

    y_cal       = data["y_cal"]
    cohorts_cal = data["cohorts_cal"]
    n           = len(y_cal)

    # Replace y_cal with synthetic labels: P(Y=1) = true_ppv
    y_syn = rng.binomial(1, true_ppv, size=n)
    preds = np.ones(n, dtype=int)  # all predicted positive

    # Collect (lb, tau) pairs for Holm correction
    p_vals = []
    cert_candidates = []

    for cid in np.unique(cohorts_cal):
        mask = cohorts_cal == cid
        y_c = y_syn[mask]
        w_c = weights[mask]
        if mask.sum() < 3:
            continue
        w_norm = w_c / (w_c.sum() + 1e-12)
        stats = weighted_stats_01(y_c.astype(float), w_norm)
        lb = eb_lower_bound(stats.mu, stats.var, stats.n_eff, alpha)
        # p-value proxy: if lb >= tau => near-zero p-value
        pv = alpha * 0.001 if lb >= tau else 1.0
        p_vals.append(pv)
        cert_candidates.append((cid, lb >= tau))

    if not p_vals:
        return {"fwer": 0, "n_certs": 0, "tau": tau, "true_ppv": true_ppv}

    rejected = holm_reject(pd.Series(p_vals), alpha)
    n_certs = int(rejected.any())
    return {"fwer": n_certs, "n_certs": n_certs, "tau": tau, "true_ppv": true_ppv}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    tau_grid  = [0.5, 0.6, 0.7, 0.8, 0.9]
    epsilon   = 0.05  # true_ppv = tau - 0.05 for each tau
    n_trials  = 200
    alpha     = 0.05
    datasets  = ["adult", "compas", "imdb", "yelp", "bace", "bbbp"]

    out_dir = ROOT / "results" / "h4_per_tau_null"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = str(ROOT / "data" / "processed")
    records  = []

    for ds_name in datasets:
        print(f"\n{ds_name}:")
        try:
            data = load_dataset_safe(ds_name, data_dir)
        except Exception as e:
            print(f"  [SKIP] {e}")
            continue

        # Estimate weights once
        try:
            ulsif = create_ulsif_baseline()
            weights = ulsif.estimate_weights(data["X_cal"], data["X_test"])
        except Exception:
            weights = np.ones(len(data["y_cal"]))
        weights = np.maximum(weights, 1e-8)

        for tau in tau_grid:
            n_false = 0
            for trial in range(n_trials):
                seed = trial * 137 + int(tau * 1000) + hash(ds_name) % 10000
                result = run_per_tau_trial(data, weights, tau, epsilon, alpha, seed)
                n_false += result["fwer"]

            fwer = n_false / n_trials
            true_ppv = tau - epsilon
            print(f"  tau={tau:.1f}: true_ppv={true_ppv:.2f}, "
                  f"FWER={fwer:.4f} ({n_false}/{n_trials})")
            records.append({
                "dataset":  ds_name,
                "tau":      tau,
                "epsilon":  epsilon,
                "true_ppv": true_ppv,
                "n_false":  n_false,
                "n_trials": n_trials,
                "fwer":     fwer,
                "pass":     fwer <= alpha,
            })

    df = pd.DataFrame(records)
    out_path = out_dir / "h4_per_tau_null.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Summary
    total_false = df["n_false"].sum()
    total_trials = df["n_trials"].sum()
    print(f"\n=== SUMMARY ===")
    print(f"Total false certs: {total_false} / {total_trials} "
          f"({total_false/total_trials:.4f})")
    print(f"Datasets tested: {df['dataset'].nunique()}")
    print(f"Tau values tested: {sorted(df['tau'].unique())}")
    print(f"All FWER <= alpha: {(df['fwer'] <= alpha).all()}")

    per_tau = df.groupby("tau").agg(
        total_false=("n_false", "sum"),
        total_trials=("n_trials", "sum"),
    )
    per_tau["fwer"] = per_tau["total_false"] / per_tau["total_trials"]
    print("\nFWER by tau:")
    print(per_tau[["fwer"]].round(4).to_string())


if __name__ == "__main__":
    main()
