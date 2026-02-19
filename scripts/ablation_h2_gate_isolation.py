"""
H2 Gate Isolation: Which diagnostic matters most?
===================================================
Isolates contributions of k-hat, ESS, and clip-mass by removing
each individually and measuring FWER change.

Variants tested:
  1. full_gating     - k-hat + ESS + clip (RAVEL default)
  2. no_khat         - ESS + clip only
  3. no_ess          - k-hat + clip only
  4. no_clip         - k-hat + ESS only
  5. ess_only        - ESS gate alone
  6. khat_only       - k-hat gate alone
  7. clip_only       - clip alone
  8. ungated         - no diagnostics at all

Tests across weight distributions (log-normal sigma 0.5-3.0).
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

from synthetic_shift_generator import SyntheticShiftGenerator
from ravel.bounds.empirical_bernstein import eb_lower_bound
from ravel.bounds.p_value import eb_p_value
from ravel.bounds.weighted_stats import weighted_stats_01
from ravel.bounds.holm import holm_reject


def psis_khat(weights):
    """Pareto shape diagnostic (simplified Vehtari et al. 2017).
    k > 0.7 indicates unreliable importance weights.
    """
    n = len(weights)
    if n < 20:
        return 0.0
    sorted_w = np.sort(weights)[::-1]
    m = max(int(np.ceil(min(n / 5, 3 * np.sqrt(n)))), 3)
    tail = sorted_w[:m]
    if tail[-1] <= 0:
        return 0.0
    log_tail = np.log(tail) - np.log(tail[-1])
    return np.mean(log_tail)


def clip_weights_quantile(weights, quantile=0.99):
    """Clip weights at given quantile."""
    threshold = np.quantile(weights, quantile)
    clipped = np.minimum(weights, threshold)
    if clipped.mean() > 0:
        clipped = clipped / clipped.mean()
    return clipped


def ess_fraction(weights):
    """ESS / n ratio."""
    n = len(weights)
    if n == 0:
        return 0.0
    n_eff = (weights.sum() ** 2) / (weights ** 2).sum()
    return n_eff / n


def apply_gates(weights, use_khat=True, use_ess=True, use_clip=True,
                khat_threshold=0.7, ess_threshold=0.15, clip_quantile=0.99):
    """Apply specified combination of gates to weights.

    Returns: (processed_weights, gated: bool, diagnostics: dict)
    """
    w = weights.copy()
    diag = {"khat": None, "ess_frac": None, "clipped": False, "gated": False}

    # k-hat check
    if use_khat:
        k = psis_khat(w)
        diag["khat"] = k
        if k > khat_threshold:
            diag["gated"] = True
            return w, True, diag

    # ESS check
    if use_ess:
        ef = ess_fraction(w)
        diag["ess_frac"] = ef
        if ef < ess_threshold:
            diag["gated"] = True
            return w, True, diag

    # Clipping
    if use_clip:
        w = clip_weights_quantile(w, clip_quantile)
        diag["clipped"] = True

    return w, False, diag


GATE_CONFIGS = {
    "full_gating":    {"use_khat": True,  "use_ess": True,  "use_clip": True},
    "no_khat":        {"use_khat": False, "use_ess": True,  "use_clip": True},
    "no_ess":         {"use_khat": True,  "use_ess": False, "use_clip": True},
    "no_clip":        {"use_khat": True,  "use_ess": True,  "use_clip": False},
    "ess_only":       {"use_khat": False, "use_ess": True,  "use_clip": False},
    "khat_only":      {"use_khat": False, "use_ess": False, "use_clip": False},  # khat gates whole cohort
    "clip_only":      {"use_khat": False, "use_ess": False, "use_clip": True},
    "ungated":        {"use_khat": False, "use_ess": False, "use_clip": False},
}
# Fix: khat_only should use khat but not others
GATE_CONFIGS["khat_only"] = {"use_khat": True, "use_ess": False, "use_clip": False}


def run_gate_trial(data, raw_weights, tau_grid, alpha, trial_id, gate_config_name):
    """Run certification with specified gate configuration."""
    config = GATE_CONFIGS[gate_config_name]

    cohort_ids = np.unique(data.cohorts_cal)
    pvals = []
    test_info = []
    n_gated = 0

    for cid in cohort_ids:
        cmask = data.cohorts_cal == cid
        pmask = cmask & (data.preds_cal == 1)
        y_pos = data.y_cal[pmask]
        w_pos = raw_weights[pmask].copy()

        if len(y_pos) < 2 or w_pos.sum() == 0:
            for tau in tau_grid:
                pvals.append(1.0)
                test_info.append((cid, tau, np.nan, np.nan, np.nan))
            continue

        # Apply gate configuration
        w_processed, gated, diag = apply_gates(w_pos, **config)

        if gated:
            n_gated += 1
            for tau in tau_grid:
                pvals.append(1.0)
                test_info.append((cid, tau, np.nan, np.nan, np.nan))
            continue

        # Normalize
        if w_processed.mean() > 0:
            w_processed = w_processed / w_processed.mean()

        stats = weighted_stats_01(y_pos, w_processed)

        for tau in tau_grid:
            pval = eb_p_value(stats.mu, stats.var, stats.n_eff, tau)
            lb = eb_lower_bound(stats.mu, stats.var, stats.n_eff, alpha)
            pvals.append(pval)
            test_info.append((cid, tau, stats.mu, stats.n_eff, lb))

    # Holm
    n_certified = 0
    n_false = 0

    if pvals:
        rejected = holm_reject(pd.Series(pvals), alpha)
        for i, info in enumerate(test_info):
            cid, tau = info[0], info[1]
            if rejected.iloc[i]:
                n_certified += 1
                true_ppv = data.true_ppv.get(cid, {}).get(tau, np.nan)
                if not np.isnan(true_ppv) and true_ppv < tau:
                    n_false += 1

    return {
        "gate_config": gate_config_name,
        "trial_id": trial_id,
        "n_certified": n_certified,
        "n_false": n_false,
        "fwer": int(n_false > 0),
        "cert_rate": n_certified / max(len(cohort_ids) * len(tau_grid), 1),
        "n_gated": n_gated,
        "n_cohorts": len(cohort_ids),
    }


def main():
    out_dir = shift_bench_dir / "results" / "h2_gate_isolation"
    os.makedirs(out_dir, exist_ok=True)

    n_trials = 300
    alpha = 0.05
    tau_grid = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    seed = 42

    # Test across weight severity levels
    sigmas = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]

    all_results = []

    for sigma in sigmas:
        print(f"\n{'='*70}")
        print(f"Weight sigma={sigma}")
        print(f"{'='*70}")

        for trial in range(n_trials):
            if trial % max(1, n_trials // 3) == 0:
                print(f"  Trial {trial+1}/{n_trials}...")

            trial_seed = seed + trial * 137
            rng = np.random.RandomState(trial_seed)

            n_cal = 2000
            n_cohorts = 5

            gen = SyntheticShiftGenerator(
                n_cal=n_cal, n_test=5000, n_cohorts=n_cohorts,
                d_features=10, shift_severity=1.0,
                positive_rate=0.5, seed=trial_seed,
            )
            data = gen.generate(tau_grid=tau_grid)

            # Generate log-normal weights
            log_w = rng.normal(0, sigma, size=n_cal)
            raw_weights = np.exp(log_w)
            raw_weights = raw_weights / raw_weights.mean()

            w_neff = (raw_weights.sum() ** 2) / (raw_weights ** 2).sum()
            w_khat = psis_khat(raw_weights)
            w_ess_frac = w_neff / n_cal

            for config_name in GATE_CONFIGS:
                result = run_gate_trial(
                    data, raw_weights, tau_grid, alpha, trial, config_name
                )
                result["sigma"] = sigma
                result["weight_neff"] = w_neff
                result["weight_khat"] = w_khat
                result["weight_ess_frac"] = w_ess_frac
                all_results.append(result)

    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "gate_isolation_raw.csv", index=False)

    # Summary
    print(f"\n{'='*70}")
    print("H2 GATE ISOLATION SUMMARY")
    print(f"{'='*70}")

    summary = df.groupby(["sigma", "gate_config"]).agg(
        fwer_rate=("fwer", "mean"),
        mean_cert_rate=("cert_rate", "mean"),
        mean_gated=("n_gated", "mean"),
        n_trials=("trial_id", "count"),
    ).reset_index()
    summary.to_csv(out_dir / "gate_isolation_summary.csv", index=False)

    print(f"\n{'sigma':>5} {'Config':<18} {'FWER':>7} {'Cert%':>7} {'Gated':>6}")
    print("-" * 50)
    for _, row in summary.sort_values(["sigma", "gate_config"]).iterrows():
        print(f"{row['sigma']:>5.1f} {row['gate_config']:<18} "
              f"{row['fwer_rate']:>6.1%} {row['mean_cert_rate']:>6.1%} "
              f"{row['mean_gated']:>5.1f}")

    # Compute marginal contribution of each gate
    print(f"\n{'='*70}")
    print("MARGINAL GATE CONTRIBUTIONS (FWER reduction vs ungated)")
    print(f"{'='*70}")

    for sigma in sigmas:
        sub = summary[summary["sigma"] == sigma]
        ungated_fwer = sub[sub["gate_config"] == "ungated"]["fwer_rate"].values
        if len(ungated_fwer) == 0:
            continue
        ungated_fwer = ungated_fwer[0]

        print(f"\n  sigma={sigma} (ungated FWER={ungated_fwer:.1%}):")
        for config in ["khat_only", "ess_only", "clip_only",
                        "no_khat", "no_ess", "no_clip", "full_gating"]:
            row = sub[sub["gate_config"] == config]
            if len(row) > 0:
                fwer = row["fwer_rate"].values[0]
                reduction = ungated_fwer - fwer
                print(f"    {config:<18} FWER={fwer:.1%}  "
                      f"reduction={reduction:+.1%}")

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
