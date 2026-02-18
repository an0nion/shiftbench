"""
Experiment C: Cross-Domain H1 Validation (KLIEP vs uLSIF Agreement)
=====================================================================

Tests Hypothesis 1: KLIEP and uLSIF produce identical certify/abstain
decisions under EB-style certification because EB conservativeness absorbs
weight variance differences.

Design:
    For each of 6 real datasets (2 per domain):
    1. Load real data with trained model predictions
    2. Rebin cohorts to n_cohorts=5 (usable regime)
    3. For each trial (N=30):
        a. Subsample n_cal from source
        b. Run BOTH uLSIF and KLIEP on same data
        c. Compare certify/abstain decisions
    4. Compute: agreement_on_active, Cohen's kappa, total certifications

Targets:
    - >= 100 certifications per dataset (from both methods combined)
    - Report agreement on "active" pairs (where at least one method certifies)
    - Cohen's kappa for inter-method reliability

Usage:
    python scripts/experiment_c_cross_domain_h1.py --quick
    python scripts/experiment_c_cross_domain_h1.py --n_trials 30
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project paths
shift_bench_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(shift_bench_dir / "src"))
sys.path.insert(0, str(shift_bench_dir.parent / "ravel" / "src"))

from experiment_a_real_data_calibration import (
    load_real_dataset,
    DATASET_CONFIG,
)


def run_h1_trial(
    data: dict,
    n_cal: int,
    tau_grid: np.ndarray,
    alpha: float,
    trial_id: int,
    seed: int,
) -> list:
    """Run a single H1 trial: compare KLIEP vs uLSIF on same data.

    Returns:
        List of dicts, one per (cohort, tau) pair, with decisions from both methods.
    """
    from ravel.bounds.empirical_bernstein import eb_lower_bound
    from ravel.bounds.p_value import eb_p_value
    from ravel.bounds.weighted_stats import weighted_stats_01
    from ravel.bounds.holm import holm_reject

    rng = np.random.RandomState(seed)

    # 1. Subsample calibration set (same for both methods)
    n_source = len(data["X_source"])
    sample_idx = rng.choice(n_source, size=min(n_cal, n_source), replace=True)

    X_cal = data["X_source"][sample_idx]
    y_cal = data["y_source"][sample_idx]
    cohorts_cal = data["cohorts_source"][sample_idx]
    preds_cal = data["preds_source"][sample_idx]
    X_test = data["X_test"]

    # 2. Estimate weights with BOTH methods
    from shiftbench.baselines.ulsif import uLSIFBaseline
    from shiftbench.baselines.kliep import KLIEPBaseline

    methods = {}
    n_basis = min(100, len(X_cal))

    # uLSIF
    try:
        ulsif = uLSIFBaseline(n_basis=n_basis, sigma=None, lambda_=0.1,
                              random_state=seed)
        methods["ulsif"] = ulsif.estimate_weights(X_cal, X_test)
    except Exception:
        methods["ulsif"] = np.ones(len(X_cal))

    # KLIEP
    try:
        kliep = KLIEPBaseline(n_basis=n_basis, sigma=None, max_iter=5000,
                              random_state=seed)
        methods["kliep"] = kliep.estimate_weights(X_cal, X_test)
    except Exception:
        methods["kliep"] = np.ones(len(X_cal))

    # 3. For each method, compute p-values and Holm decisions
    method_decisions = {}

    for method_name, weights in methods.items():
        cohort_ids = np.unique(cohorts_cal)
        pvals = []
        test_info = []

        for cohort_id in cohort_ids:
            cohort_mask = cohorts_cal == cohort_id
            pos_mask = cohort_mask & (preds_cal == 1)

            y_pos = y_cal[pos_mask]
            w_pos = weights[pos_mask]

            if len(y_pos) < 2 or w_pos.sum() == 0:
                for tau in tau_grid:
                    pvals.append(1.0)
                    test_info.append((cohort_id, tau, np.nan, np.nan, np.nan))
                continue

            result = weighted_stats_01(y_pos, w_pos)

            for tau in tau_grid:
                pval = eb_p_value(result.mu, result.var, result.n_eff, tau)
                lb = eb_lower_bound(result.mu, result.var, result.n_eff, alpha)
                pvals.append(pval)
                test_info.append((cohort_id, tau, result.mu, result.n_eff, lb))

        # Holm step-down
        decisions = {}
        if len(pvals) > 0:
            pvals_series = pd.Series(pvals)
            rejected = holm_reject(pvals_series, alpha)

            for i, (cohort_id, tau, mu_hat, n_eff, lb) in enumerate(test_info):
                key = (cohort_id, tau)
                decisions[key] = {
                    "certified": bool(rejected.iloc[i]),
                    "mu_hat": mu_hat,
                    "n_eff": n_eff,
                    "lower_bound": lb,
                }

        method_decisions[method_name] = decisions

    # 4. Compare decisions
    results = []
    all_keys = set()
    for dec in method_decisions.values():
        all_keys.update(dec.keys())

    for key in sorted(all_keys):
        cohort_id, tau = key

        ulsif_dec = method_decisions.get("ulsif", {}).get(key, {})
        kliep_dec = method_decisions.get("kliep", {}).get(key, {})

        ulsif_cert = ulsif_dec.get("certified", False)
        kliep_cert = kliep_dec.get("certified", False)

        results.append({
            "dataset": data["dataset_name"],
            "domain": data["config"]["domain"],
            "trial_id": trial_id,
            "cohort_id": cohort_id,
            "tau": tau,
            "ulsif_certified": ulsif_cert,
            "kliep_certified": kliep_cert,
            "agree": ulsif_cert == kliep_cert,
            "both_certify": ulsif_cert and kliep_cert,
            "either_certify": ulsif_cert or kliep_cert,
            "ulsif_mu": ulsif_dec.get("mu_hat", np.nan),
            "kliep_mu": kliep_dec.get("mu_hat", np.nan),
            "ulsif_neff": ulsif_dec.get("n_eff", np.nan),
            "kliep_neff": kliep_dec.get("n_eff", np.nan),
            "ulsif_lb": ulsif_dec.get("lower_bound", np.nan),
            "kliep_lb": kliep_dec.get("lower_bound", np.nan),
        })

    return results


def compute_cohens_kappa(y1: np.ndarray, y2: np.ndarray) -> float:
    """Compute Cohen's kappa for inter-rater agreement."""
    n = len(y1)
    if n == 0:
        return np.nan

    # Observed agreement
    p_o = (y1 == y2).mean()

    # Expected agreement by chance
    p_yes_1 = y1.mean()
    p_yes_2 = y2.mean()
    p_e = p_yes_1 * p_yes_2 + (1 - p_yes_1) * (1 - p_yes_2)

    if p_e == 1.0:
        return 1.0  # Perfect agreement when both always agree

    kappa = (p_o - p_e) / (1 - p_e)
    return kappa


def run_experiment_c(
    datasets: list,
    data_dir: str,
    model_dir: str,
    n_trials: int = 30,
    n_cal: int = 2000,
    n_cohort_bins: int = 5,
    alpha: float = 0.05,
    tau_grid: np.ndarray = None,
    seed: int = 42,
    output_dir: str = "results/experiment_c",
):
    """Run Experiment C across all specified datasets.

    Forces n_cohorts=5 rebinning on all datasets for usable regime.
    """
    if tau_grid is None:
        tau_grid = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

    os.makedirs(output_dir, exist_ok=True)

    # Override cohort binning for usable regime
    for ds in datasets:
        if ds in DATASET_CONFIG:
            DATASET_CONFIG[ds]["n_cohort_bins"] = n_cohort_bins

    all_results = []
    dataset_summaries = []

    for ds_idx, dataset_name in enumerate(datasets):
        print(f"\n{'='*70}")
        print(f"Dataset {ds_idx+1}/{len(datasets)}: {dataset_name}")
        print(f"{'='*70}")

        try:
            data = load_real_dataset(dataset_name, data_dir, model_dir)
        except Exception as e:
            print(f"  [SKIP] Failed to load {dataset_name}: {e}")
            continue

        print(f"  Source: {len(data['X_source'])} samples, "
              f"{len(np.unique(data['cohorts_source']))} cohorts")
        print(f"  Test: {len(data['X_test'])} samples, "
              f"{len(np.unique(data['cohorts_test']))} cohorts")

        t0 = time.time()
        for trial in range(n_trials):
            if trial % max(1, n_trials // 5) == 0:
                print(f"  Trial {trial+1}/{n_trials}...")

            trial_seed = seed + ds_idx * 10000 + trial
            trial_results = run_h1_trial(
                data=data,
                n_cal=min(n_cal, len(data["X_source"])),
                tau_grid=tau_grid,
                alpha=alpha,
                trial_id=trial,
                seed=trial_seed,
            )
            all_results.extend(trial_results)

        elapsed = time.time() - t0

        # Per-dataset analysis
        ds_df = pd.DataFrame([r for r in all_results
                              if r["dataset"] == dataset_name])

        total_pairs = len(ds_df)
        total_agree = ds_df["agree"].sum()
        agreement_rate = ds_df["agree"].mean()

        # Agreement on "active" pairs (at least one certifies)
        active = ds_df[ds_df["either_certify"]]
        n_active = len(active)
        if n_active > 0:
            agree_on_active = active["agree"].mean()
            both_certify = active["both_certify"].sum()
        else:
            agree_on_active = np.nan
            both_certify = 0

        total_ulsif_cert = ds_df["ulsif_certified"].sum()
        total_kliep_cert = ds_df["kliep_certified"].sum()

        # Cohen's kappa
        kappa = compute_cohens_kappa(
            ds_df["ulsif_certified"].values.astype(int),
            ds_df["kliep_certified"].values.astype(int),
        )

        # Weight correlation (where both have valid estimates)
        valid_mu = ds_df.dropna(subset=["ulsif_mu", "kliep_mu"])
        if len(valid_mu) > 1:
            mu_corr = np.corrcoef(valid_mu["ulsif_mu"],
                                  valid_mu["kliep_mu"])[0, 1]
        else:
            mu_corr = np.nan

        # LB difference
        valid_lb = ds_df.dropna(subset=["ulsif_lb", "kliep_lb"])
        if len(valid_lb) > 0:
            lb_diff = (valid_lb["kliep_lb"] - valid_lb["ulsif_lb"]).abs()
            mean_lb_diff = lb_diff.mean()
            max_lb_diff = lb_diff.max()
        else:
            mean_lb_diff = np.nan
            max_lb_diff = np.nan

        summary = {
            "dataset": dataset_name,
            "domain": data["config"]["domain"],
            "n_trials": n_trials,
            "total_pairs": total_pairs,
            "agreement_rate": agreement_rate,
            "n_active": n_active,
            "agree_on_active": agree_on_active,
            "cohens_kappa": kappa,
            "ulsif_certifications": total_ulsif_cert,
            "kliep_certifications": total_kliep_cert,
            "both_certify": both_certify,
            "mu_correlation": mu_corr,
            "mean_lb_diff": mean_lb_diff,
            "max_lb_diff": max_lb_diff,
            "runtime_sec": elapsed,
        }
        dataset_summaries.append(summary)

        # Print
        print(f"\n  --- {dataset_name} H1 Results ---")
        print(f"  Overall agreement: {agreement_rate:.1%} "
              f"({total_agree}/{total_pairs})")
        print(f"  Active pairs: {n_active} "
              f"(uLSIF cert={total_ulsif_cert}, KLIEP cert={total_kliep_cert})")
        if n_active > 0:
            print(f"  Agreement on active: {agree_on_active:.1%}")
            print(f"  Both certify: {both_certify}")
        print(f"  Cohen's kappa: {kappa:.3f}")
        print(f"  mu_hat correlation: {mu_corr:.3f}" if not np.isnan(mu_corr)
              else "  mu_hat correlation: N/A")
        print(f"  Mean |LB diff|: {mean_lb_diff:.4f}" if not np.isnan(mean_lb_diff)
              else "  Mean |LB diff|: N/A")
        print(f"  Time: {elapsed:.1f}s")

    # Save raw results
    raw_df = pd.DataFrame(all_results)
    raw_path = os.path.join(output_dir, "experiment_c_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"\nSaved raw results: {raw_path}")

    # Save summary
    summary_df = pd.DataFrame(dataset_summaries)
    summary_path = os.path.join(output_dir, "experiment_c_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    # Print overall
    print_h1_summary(summary_df)

    # Save text summary
    text = format_h1_summary(summary_df, n_trials, n_cal, n_cohort_bins, alpha)
    text_path = os.path.join(output_dir, "experiment_c_summary.txt")
    with open(text_path, "w") as f:
        f.write(text)
    print(f"Saved text summary: {text_path}")

    return raw_df, summary_df


def print_h1_summary(summary_df: pd.DataFrame):
    """Print formatted H1 summary."""
    print(f"\n{'='*70}")
    print("EXPERIMENT C: CROSS-DOMAIN H1 (KLIEP vs uLSIF AGREEMENT)")
    print(f"{'='*70}")

    print(f"\n{'Dataset':<12} {'Domain':<10} {'Agree%':<8} {'Active':<8} "
          f"{'AgreeAct':<10} {'Kappa':<8} {'uLSIF#':<8} {'KLIEP#':<8}")
    print("-" * 72)

    for _, row in summary_df.iterrows():
        agree_act = f"{row['agree_on_active']:.1%}" \
            if not np.isnan(row["agree_on_active"]) else "N/A"
        print(f"{row['dataset']:<12} {row['domain']:<10} "
              f"{row['agreement_rate']:<8.1%} {row['n_active']:<8.0f} "
              f"{agree_act:<10} {row['cohens_kappa']:<8.3f} "
              f"{row['ulsif_certifications']:<8.0f} "
              f"{row['kliep_certifications']:<8.0f}")

    print("-" * 72)

    # Overall
    total_active = summary_df["n_active"].sum()
    total_ulsif = summary_df["ulsif_certifications"].sum()
    total_kliep = summary_df["kliep_certifications"].sum()
    overall_agreement = summary_df["agreement_rate"].mean()
    mean_kappa = summary_df["cohens_kappa"].mean()

    print(f"\nOverall agreement: {overall_agreement:.1%}")
    print(f"Mean Cohen's kappa: {mean_kappa:.3f}")
    print(f"Total certifications: uLSIF={total_ulsif:.0f}, "
          f"KLIEP={total_kliep:.0f}")
    print(f"Total active pairs: {total_active:.0f}")

    # Interpretation
    if overall_agreement >= 0.99:
        print("\n[STRONG] Near-perfect agreement supports H1: "
              "EB conservativeness absorbs weight differences.")
    elif overall_agreement >= 0.95:
        print("\n[MODERATE] High agreement supports H1 with minor discrepancies.")
    elif overall_agreement >= 0.90:
        print("\n[WEAK] Agreement present but with meaningful discrepancies.")
    else:
        print("\n[DISAGREE] Methods produce different decisions. "
              "H1 not supported.")

    print(f"\n{'='*70}")


def format_h1_summary(
    summary_df: pd.DataFrame,
    n_trials: int,
    n_cal: int,
    n_cohort_bins: int,
    alpha: float,
) -> str:
    """Format results as text."""
    lines = []
    lines.append("=" * 70)
    lines.append("EXPERIMENT C: CROSS-DOMAIN H1 (KLIEP vs uLSIF AGREEMENT)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Design: {n_trials} trials per dataset, n_cal={n_cal}, "
                 f"n_cohorts={n_cohort_bins}, alpha={alpha}")
    lines.append("Both methods run on identical subsampled data per trial.")
    lines.append("Agreement = same certify/abstain decision on each "
                 "(cohort, tau) pair.")
    lines.append("")

    for _, row in summary_df.iterrows():
        lines.append(f"{row['dataset']} ({row['domain']}): "
                     f"agree={row['agreement_rate']:.1%}, "
                     f"kappa={row['cohens_kappa']:.3f}, "
                     f"uLSIF={row['ulsif_certifications']:.0f} certs, "
                     f"KLIEP={row['kliep_certifications']:.0f} certs")

    lines.append("")
    overall = summary_df["agreement_rate"].mean()
    lines.append(f"Overall agreement: {overall:.1%}")
    lines.append(f"Mean kappa: {summary_df['cohens_kappa'].mean():.3f}")
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Experiment C: Cross-domain H1 validation"
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["bace", "bbbp", "adult", "compas", "imdb", "yelp"],
        help="Datasets to evaluate"
    )
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--n_cal", type=int, default=2000)
    parser.add_argument("--n_cohorts", type=int, default=5,
                        help="Rebin all datasets to this many cohorts")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/experiment_c")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 5 trials, 3 datasets")
    args = parser.parse_args()

    if args.quick:
        args.n_trials = 5
        args.datasets = ["bbbp", "imdb", "yelp"]

    print("Experiment C: Cross-Domain H1 Validation")
    print(f"  Datasets: {args.datasets}")
    print(f"  Trials: {args.n_trials}")
    print(f"  n_cal: {args.n_cal}, n_cohorts: {args.n_cohorts}")
    print(f"  alpha: {args.alpha}")
    print()

    run_experiment_c(
        datasets=args.datasets,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        n_trials=args.n_trials,
        n_cal=args.n_cal,
        n_cohort_bins=args.n_cohorts,
        alpha=args.alpha,
        seed=args.seed,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
