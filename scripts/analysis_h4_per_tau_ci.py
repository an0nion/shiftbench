"""
H4 Per-tau null test: add Wilson CIs to existing FWER estimates.

Reads:  results/h4_per_tau_null/h4_per_tau_null.csv
        (dataset x tau, n_false, n_trials, fwer)

Adds Wilson 95% CI on FWER for each (dataset, tau) cell and
produces an aggregated per-tau summary across datasets.

PI acceptance criterion: FWER ≤ alpha=0.05 with Wilson CI_upper < alpha
for each tau independently.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

shift_bench = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(shift_bench / "src"))

# ── Wilson CI for a proportion ────────────────────────────────────────────────
def wilson_ci(x: int, n: int, z: float = 1.96):
    """Wilson score interval for x successes out of n trials.

    Returns (lower, upper).  Handles x=0 and x=n correctly.
    """
    if n == 0:
        return float("nan"), float("nan")
    p_hat = x / n
    centre = (p_hat + z**2 / (2 * n)) / (1 + z**2 / n)
    spread  = z * np.sqrt((p_hat * (1 - p_hat) / n) + z**2 / (4 * n**2))
    spread /= (1 + z**2 / n)
    lower = max(0.0, centre - spread)
    upper = min(1.0, centre + spread)
    return lower, upper


def main():
    in_path = shift_bench / "results" / "h4_per_tau_null" / "h4_per_tau_null.csv"
    if not in_path.exists():
        print(f"ERROR: {in_path} not found. Run analysis_h4_slack.py first.")
        sys.exit(1)

    df = pd.read_csv(in_path)
    print(f"Loaded {len(df)} rows from {in_path}")

    alpha = 0.05
    z95 = 1.96

    rows = []
    for _, row in df.iterrows():
        n_false = int(row["n_false"])
        n_trials = int(row["n_trials"])
        fwer = n_false / n_trials if n_trials > 0 else float("nan")
        ci_lo, ci_hi = wilson_ci(n_false, n_trials, z95)
        rows.append({
            "dataset":      row["dataset"],
            "tau":          row["tau"],
            "true_ppv":     row.get("true_ppv", row.get("epsilon", float("nan"))),
            "epsilon":      row.get("epsilon", float("nan")),
            "n_false":      n_false,
            "n_trials":     n_trials,
            "fwer":         round(fwer, 6),
            "ci_lower":     round(ci_lo, 6),
            "ci_upper":     round(ci_hi, 6),
            "ci_upper_lt_alpha": ci_hi < alpha,
            "pass":         fwer <= alpha,
        })

    df_out = pd.DataFrame(rows)

    # ── Per-tau aggregated summary (pooled across datasets) ──────────────────
    tau_rows = []
    for tau, grp in df_out.groupby("tau"):
        n_false_total = grp["n_false"].sum()
        n_trials_total = grp["n_trials"].sum()
        fwer_pooled = n_false_total / n_trials_total
        ci_lo, ci_hi = wilson_ci(int(n_false_total), int(n_trials_total), z95)
        tau_rows.append({
            "tau":              tau,
            "n_datasets":       len(grp),
            "n_false_total":    int(n_false_total),
            "n_trials_total":   int(n_trials_total),
            "fwer_pooled":      round(fwer_pooled, 6),
            "ci_lower":         round(ci_lo, 6),
            "ci_upper":         round(ci_hi, 6),
            "ci_upper_lt_alpha": ci_hi < alpha,
            "all_pass":         all(grp["pass"]),
        })

    df_tau = pd.DataFrame(tau_rows)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = shift_bench / "results" / "h4_per_tau_null"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "h4_per_tau_null_with_ci.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved per-cell CI table to {out_path}")

    tau_path = out_dir / "h4_per_tau_summary.csv"
    df_tau.to_csv(tau_path, index=False)
    print(f"Saved per-tau summary to {tau_path}\n")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("Per-tau pooled FWER with Wilson 95% CI:")
    print("-" * 75)
    print(f"{'tau':>5}  {'n_trials':>9}  {'n_false':>8}  {'FWER%':>7}  "
          f"{'CI_lo%':>7}  {'CI_hi%':>7}  {'CI_hi<5%':>9}  pass")
    print("-" * 75)
    all_pass = True
    for _, r in df_tau.iterrows():
        flag = "PASS" if r["ci_upper_lt_alpha"] else "FAIL"
        if not r["ci_upper_lt_alpha"]:
            all_pass = False
        print(
            f"{r['tau']:>5.1f}  {r['n_trials_total']:>9}  {r['n_false_total']:>8}  "
            f"{r['fwer_pooled']*100:>6.2f}%  "
            f"{r['ci_lower']*100:>6.2f}%  {r['ci_upper']*100:>6.2f}%  "
            f"{'YES' if r['ci_upper_lt_alpha'] else 'NO':>9}  {flag}"
        )

    print("-" * 75)
    print(f"\nOverall: {'ALL PASS — FWER controlled at each tau.' if all_pass else 'SOME FAIL — check above.'}")
    print(f"Note: With 0 events in 200 trials, Wilson CI_upper = "
          f"{wilson_ci(0, 200)[1]*100:.2f}% << alpha=5%.")
    print("This is substantially more conservative than required.")


if __name__ == "__main__":
    main()
