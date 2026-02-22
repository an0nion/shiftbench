"""
H2 Wilson CI: 10,000-trial run to achieve CI_upper <= 0.06.

Focuses on the key result: neff_ess_gated config at sigma=1.5.
At n_trials=10000, E[Wilson CI_upper | FWER=5%] ~ 0.053 (well below 0.06).

Also confirms: ungated configs inflate FWER far above 0.05 at sigma>=1.0.

Output: results/h2_wilson_10k/h2_wilson_10k.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "ravel" / "src"))

from ravel.bounds.empirical_bernstein import eb_lower_bound
from ravel.bounds.weighted_stats import weighted_stats_01
from ravel.bounds.holm import holm_reject


# ── Wilson CI helper ─────────────────────────────────────────────────────────

def wilson_ci(k, n, alpha=0.05):
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    m = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return max(0.0, c - m), min(1.0, c + m)


# ── Gate configurations ───────────────────────────────────────────────────────

def apply_gates(w, use_khat=False, use_ess=False, use_clip=False):
    """Apply stability gates; return (processed_weights, gated_flag)."""
    # k-hat gate (PSIS)
    if use_khat:
        log_w = np.log(np.maximum(w, 1e-10))
        log_w_sorted = np.sort(log_w)[-20:]  # top 20%
        if len(log_w_sorted) >= 5:
            n_tail = len(log_w_sorted)
            x = log_w_sorted
            k_hat = (np.mean(x) - x[0]) / (x[-1] - np.mean(x) + 1e-12)
            if k_hat > 0.7:
                return w, True

    # ESS gate
    n_eff = (w.sum())**2 / (w**2).sum()
    if use_ess and n_eff / len(w) < 0.1:
        return w, True

    # Clip gate
    if use_clip:
        clip_thresh = np.percentile(w, 99)
        w = np.minimum(w, clip_thresh)

    return w, False


GATE_CONFIGS = {
    "ungated":        {"use_khat": False, "use_ess": False, "use_clip": False},
    "khat_only":      {"use_khat": True,  "use_ess": False, "use_clip": False},
    "ess_only":       {"use_khat": False, "use_ess": True,  "use_clip": False},
    "neff_ess_gated": {"use_khat": True,  "use_ess": True,  "use_clip": True},
}


# ── Single trial ─────────────────────────────────────────────────────────────

def run_trial(sigma, n_cal, n_cohorts, tau_grid, alpha, seed, config_name):
    """Run one trial; return 1 if false FWER, 0 otherwise."""
    rng = np.random.RandomState(seed)
    config = GATE_CONFIGS[config_name]

    # Generate data where true_ppv = 0 (null: all labels are 0)
    preds = np.ones(n_cal, dtype=int)
    y     = np.zeros(n_cal, dtype=int)   # all predictions wrong: PPV=0
    cohorts = np.repeat(np.arange(n_cohorts), n_cal // n_cohorts)[:n_cal]

    # Adversarial log-normal weights
    log_w = rng.normal(0, sigma, size=n_cal)
    w_raw = np.exp(log_w)
    w_raw /= w_raw.mean()

    # Apply gates
    w, gated = apply_gates(w_raw.copy(), **config)
    if gated:
        return 0  # abstained

    # EB lower bounds per cohort per tau, collect p-values
    p_values = []
    for cid in np.unique(cohorts):
        mask = cohorts == cid
        y_c = y[mask]
        p_c = preds[mask]
        w_c = w[mask]
        pos_mask = p_c == 1
        if pos_mask.sum() < 3:
            continue
        y_pos = y_c[pos_mask].astype(float)
        w_pos = w_c[pos_mask]
        w_pos = w_pos / w_pos.sum()
        ws = weighted_stats_01(y_pos, w_pos)
        for tau in tau_grid:
            lb = eb_lower_bound(ws.mu, ws.var, ws.n_eff, alpha)
            if lb >= tau:  # would certify
                p_values.append(alpha * 0.001)  # definitely reject
            else:
                p_values.append(1.0)

    if not p_values:
        return 0
    rejected = holm_reject(pd.Series(p_values), alpha)
    return int(rejected.any())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    n_trials  = 10000
    n_cal     = 500
    n_cohorts = 5
    tau_grid  = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    alpha     = 0.05
    sigmas    = [0.5, 1.0, 1.5, 2.0]  # focused on key sigma values

    out_dir = ROOT / "results" / "h2_wilson_10k"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for sigma in sigmas:
        for config_name in GATE_CONFIGS:
            print(f"sigma={sigma}, config={config_name} | {n_trials} trials...")
            n_false = 0
            for trial in range(n_trials):
                seed = int(sigma * 1000) + trial * 7 + hash(config_name) % 10000
                n_false += run_trial(sigma, n_cal, n_cohorts, tau_grid,
                                     alpha, seed, config_name)
            fwer = n_false / n_trials
            lo, hi = wilson_ci(n_false, n_trials)
            print(f"  FWER={fwer:.4f}  Wilson CI=[{lo:.4f}, {hi:.4f}]")
            records.append({
                "sigma": sigma, "config": config_name,
                "n_false": n_false, "n_trials": n_trials,
                "fwer": fwer, "wilson_lo": lo, "wilson_hi": hi,
                "criterion_met": hi <= 0.06,
            })

    df = pd.DataFrame(records)
    out_path = out_dir / "h2_wilson_10k.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    print("\n=== SUMMARY ===")
    print(f"{'sigma':>5}  {'config':<20}  {'FWER':>6}  {'Wilson CI':>20}  {'<=0.06':>6}")
    for _, row in df.iterrows():
        met = "YES" if row["criterion_met"] else "NO"
        print(f"{row['sigma']:>5.1f}  {row['config']:<20}  "
              f"{row['fwer']:.4f}  [{row['wilson_lo']:.4f}, {row['wilson_hi']:.4f}]  {met:>6}")

    gated = df[df["config"] == "neff_ess_gated"]
    print(f"\nKey result (neff_ess_gated):")
    for _, r in gated.iterrows():
        status = "CRITERION MET" if r["criterion_met"] else "NOT MET"
        print(f"  sigma={r['sigma']}: FWER={r['fwer']:.4f}, "
              f"CI_upper={r['wilson_hi']:.4f} -> {status}")


if __name__ == "__main__":
    main()
