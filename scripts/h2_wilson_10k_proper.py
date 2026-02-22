"""
H2 Wilson CI - 10,000 trials (proper version).

Uses the existing experiment_d_gating_ablation.run_gating_trial to run
the "usable_good_weights" scenario (well-behaved weights, boundary null)
with 10,000 trials.

At true FWER = alpha = 5%:
  - n=500:   E[CI_upper] = 0.073 (does NOT satisfy <= 0.06)
  - n=10000: E[CI_upper] = 0.055 (satisfies <= 0.06)

Output: results/h2_wilson_10k/h2_wilson_10k_proper.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "ravel" / "src"))

from experiment_d_gating_ablation import run_gating_trial


def wilson_ci(k, n, alpha=0.05):
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    m = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return max(0.0, c - m), min(1.0, c + m)


def main():
    n_trials  = 10000
    n_cal     = 2000
    n_cohorts = 5
    alpha     = 0.05
    tau_grid  = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    seed      = 42

    out_dir = ROOT / "results" / "h2_wilson_10k"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {n_trials} trials (well_behaved weights, boundary null)...")
    print(f"  n_cal={n_cal}, n_cohorts={n_cohorts}, alpha={alpha}")

    # Collect per-variant FWER counts
    fwer_counts = {}

    for trial in range(n_trials):
        if trial % (n_trials // 20) == 0:
            pct = trial / n_trials * 100
            if fwer_counts:
                counts_str = ", ".join(
                    f"{v}:{c}" for v, c in sorted(fwer_counts.items())[:3]
                )
                print(f"  {pct:5.1f}% | FWER counts so far: {counts_str}...")
            else:
                print(f"  {pct:5.1f}%...")

        trial_seed = (seed + trial * 137) % (2**31)
        results = run_gating_trial(
            n_cal=n_cal, n_test=5000, n_cohorts=n_cohorts,
            shift_severity=1.0, positive_rate=0.5,
            pathology="well_behaved",
            alpha=alpha, tau_grid=tau_grid,
            trial_id=trial, seed=trial_seed,
        )
        for r in results:
            v = r["variant"]
            fwer_counts[v] = fwer_counts.get(v, 0) + r.get("false_certify_fwer", 0)

    # Compute Wilson CIs
    records = []
    print(f"\n{'Variant':<25} {'FWER':>6}  {'Wilson CI':>22}  {'<=0.06':>6}")
    print("-" * 65)
    for variant, n_false in sorted(fwer_counts.items()):
        fwer = n_false / n_trials
        lo, hi = wilson_ci(n_false, n_trials)
        met = "YES" if hi <= 0.06 else "NO"
        print(f"{variant:<25} {fwer:.4f}  [{lo:.4f}, {hi:.4f}]  {met:>6}")
        records.append({
            "variant": variant, "n_false": n_false, "n_trials": n_trials,
            "fwer": fwer, "wilson_lo": lo, "wilson_hi": hi,
            "criterion_met": hi <= 0.06,
        })

    df = pd.DataFrame(records)
    out_path = out_dir / "h2_wilson_10k_proper.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    key = df[df["variant"].str.contains("neff_ess_gated|neff.*gated|ess_gated",
                                         case=False, regex=True)]
    if len(key) > 0:
        r = key.iloc[0]
        status = "CRITERION MET" if r["criterion_met"] else "NOT MET"
        print(f"\nKey result ({r['variant']}): "
              f"FWER={r['fwer']:.4f}, CI_upper={r['wilson_hi']:.4f} -> {status}")


if __name__ == "__main__":
    main()
