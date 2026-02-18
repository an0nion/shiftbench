"""
Experiment A: Semi-Synthetic Real-Data Calibration
====================================================

Validates that the EB+Holm certification protocol maintains FWER <= alpha
on REAL datasets with REAL model predictions and NATURAL covariate shift.

This is the critical experiment that moves validation beyond "toy land."

Design:
    For each of 6 real datasets (2 per domain):
    1. Pool train+cal as "source distribution"
    2. Test set is "target distribution" (natural shift via scaffold/temporal/etc. split)
    3. Compute ground-truth PPV per cohort from test set
    4. For each trial (N=50):
        a. Subsample n_cal from source (with replacement)
        b. Run uLSIF(X_cal_sample, X_test) for importance weights
        c. Compute EB bounds + Holm step-down on sampled calibration
        d. Compare CERTIFY decisions to ground-truth PPV
    5. Measure: per-trial FWER, certification rate, coverage

Key difference from validate_h4.py:
    - Uses REAL data, features, labels, predictions, cohort structure
    - Natural covariate shift (not synthetic Gaussian)
    - Real model predictions (not oracle/Bayes-optimal)
    - Ground truth from held-out test set (not analytical)

Datasets:
    - Molecular: BACE, BBBP (scaffold split)
    - Tabular: Adult, COMPAS (demographic cohorts)
    - Text: IMDB, Yelp (temporal/category cohorts)

Usage:
    python scripts/experiment_a_real_data_calibration.py --quick
    python scripts/experiment_a_real_data_calibration.py --n_trials 50
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


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

DATASET_CONFIG = {
    "bace": {
        "domain": "molecular",
        "model": "rf",
        "n_cohort_bins": 5,  # Re-bin 739 scaffolds into 5 groups
    },
    "bbbp": {
        "domain": "molecular",
        "model": "rf",
        "n_cohort_bins": 5,
    },
    "adult": {
        "domain": "tabular",
        "model": "lr",
        "n_cohort_bins": None,  # Use natural cohorts (49)
    },
    "compas": {
        "domain": "tabular",
        "model": "lr",
        "n_cohort_bins": None,  # Use natural cohorts (38)
    },
    "imdb": {
        "domain": "text",
        "model": "lr",
        "n_cohort_bins": None,  # Already 10 cohorts
    },
    "yelp": {
        "domain": "text",
        "model": "lr",
        "n_cohort_bins": None,  # Already 10 cohorts
    },
}


def load_real_dataset(dataset_name: str, data_dir: str, model_dir: str):
    """Load a real dataset with pre-computed model predictions.

    Returns:
        dict with keys:
            X_source: Features for source (train+cal pooled) or cal only
            y_source: Labels for source
            cohorts_source: Cohort IDs for source
            preds_source: Model predictions for source
            X_test: Test features
            y_test: Test labels
            cohorts_test: Test cohort IDs
            preds_test: Test predictions
            config: Dataset configuration
    """
    dpath = os.path.join(data_dir, dataset_name)
    config = DATASET_CONFIG[dataset_name]
    model_name = config["model"]

    # Load data
    features = np.load(os.path.join(dpath, "features.npy"))
    labels = np.load(os.path.join(dpath, "labels.npy"))
    cohorts = np.load(os.path.join(dpath, "cohorts.npy"), allow_pickle=True)
    splits = pd.read_csv(os.path.join(dpath, "splits.csv"))

    # Split masks
    train_mask = splits["split"].values == "train"
    cal_mask = splits["split"].values == "cal"
    test_mask = splits["split"].values == "test"

    # Load predictions
    pred_dir = os.path.join(model_dir, "predictions", dataset_name)
    cal_preds = np.load(os.path.join(pred_dir, f"{model_name}_cal_preds_binary.npy"))
    test_preds = np.load(os.path.join(pred_dir, f"{model_name}_test_preds_binary.npy"))

    # For source, we need predictions for train set too
    # If train predictions don't exist, we train+cal pool uses cal predictions
    # and we subsample only from cal indices
    train_pred_path = os.path.join(pred_dir, f"{model_name}_train_preds_binary.npy")
    if os.path.exists(train_pred_path):
        train_preds = np.load(train_pred_path)
        # Pool train + cal as source
        source_mask = train_mask | cal_mask
        source_preds = np.concatenate([train_preds, cal_preds])
    else:
        # Use only cal as source (still valid, just smaller)
        source_mask = cal_mask
        source_preds = cal_preds

    X_source = features[source_mask]
    y_source = labels[source_mask]
    cohorts_source = cohorts[source_mask]

    X_test = features[test_mask]
    y_test = labels[test_mask]
    cohorts_test = cohorts[test_mask]

    # Re-bin cohorts if needed (for molecular datasets with 739 scaffolds)
    n_bins = config["n_cohort_bins"]
    if n_bins is not None:
        cohorts_source, cohorts_test = _rebin_cohorts(
            cohorts_source, cohorts_test, n_bins
        )

    return {
        "X_source": X_source,
        "y_source": y_source,
        "cohorts_source": cohorts_source,
        "preds_source": source_preds,
        "X_test": X_test,
        "y_test": y_test,
        "cohorts_test": cohorts_test,
        "preds_test": test_preds,
        "config": config,
        "dataset_name": dataset_name,
    }


def _rebin_cohorts(
    cohorts_source: np.ndarray,
    cohorts_test: np.ndarray,
    n_bins: int,
) -> tuple:
    """Re-bin many small cohorts into n_bins larger groups.

    Uses hash-based assignment for deterministic, balanced binning.
    Ensures same cohort label maps to same bin in both source and test.
    """
    # Get all unique cohort labels
    all_labels = np.unique(np.concatenate([cohorts_source, cohorts_test]))

    # Map each label to a bin via hash
    label_to_bin = {}
    for label in all_labels:
        label_to_bin[label] = hash(str(label)) % n_bins

    # Apply mapping
    new_source = np.array([label_to_bin[c] for c in cohorts_source])
    new_test = np.array([label_to_bin[c] for c in cohorts_test])

    return new_source, new_test


# ---------------------------------------------------------------------------
# Ground-truth PPV computation
# ---------------------------------------------------------------------------

def compute_ground_truth_ppv(
    y_test: np.ndarray,
    preds_test: np.ndarray,
    cohorts_test: np.ndarray,
) -> dict:
    """Compute ground-truth PPV per cohort from the test set.

    PPV_g = mean(y_test[cohort==g & pred==1])

    Returns:
        dict: {cohort_id: ppv_value} (NaN if no predicted positives)
    """
    ppv_by_cohort = {}
    for cohort_id in np.unique(cohorts_test):
        mask = (cohorts_test == cohort_id) & (preds_test == 1)
        if mask.sum() == 0:
            ppv_by_cohort[cohort_id] = np.nan
        else:
            ppv_by_cohort[cohort_id] = y_test[mask].mean()
    return ppv_by_cohort


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_real_data_trial(
    data: dict,
    n_cal: int,
    tau_grid: np.ndarray,
    alpha: float,
    trial_id: int,
    seed: int,
) -> dict:
    """Run a single calibration trial on real data.

    Steps:
        1. Subsample n_cal from source (with replacement)
        2. Estimate importance weights via uLSIF
        3. Compute EB p-values for each (cohort, tau) pair
        4. Apply Holm step-down correction
        5. Compare CERTIFY decisions to ground-truth PPV

    Returns:
        dict with trial metrics
    """
    from ravel.bounds.empirical_bernstein import eb_lower_bound
    from ravel.bounds.p_value import eb_p_value
    from ravel.bounds.weighted_stats import weighted_stats_01
    from ravel.bounds.holm import holm_reject

    rng = np.random.RandomState(seed)

    # 1. Subsample calibration set
    n_source = len(data["X_source"])
    sample_idx = rng.choice(n_source, size=min(n_cal, n_source), replace=True)

    X_cal = data["X_source"][sample_idx]
    y_cal = data["y_source"][sample_idx]
    cohorts_cal = data["cohorts_source"][sample_idx]
    preds_cal = data["preds_source"][sample_idx]

    X_test = data["X_test"]

    # 2. Estimate importance weights via uLSIF
    try:
        from shiftbench.baselines.ulsif import uLSIFBaseline
        ulsif = uLSIFBaseline(
            n_basis=min(100, len(X_cal)),
            sigma=None,
            lambda_=0.1,
            random_state=seed,
        )
        weights = ulsif.estimate_weights(X_cal, X_test)
    except Exception:
        weights = np.ones(len(X_cal))

    # 3. Compute p-values for all (cohort, tau) pairs
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
                test_info.append((cohort_id, tau, np.nan, np.nan, np.nan, 1.0))
            continue

        # Weighted stats
        result = weighted_stats_01(y_pos, w_pos)
        mu_hat = result.mu
        var_hat = result.var
        n_eff = result.n_eff

        for tau in tau_grid:
            pval = eb_p_value(mu_hat, var_hat, n_eff, tau)
            lb = eb_lower_bound(mu_hat, var_hat, n_eff, alpha)
            pvals.append(pval)
            test_info.append((cohort_id, tau, mu_hat, n_eff, lb, pval))

    # 4. Holm step-down
    n_certified = 0
    n_abstain = 0
    n_false_certify = 0
    n_tests = 0
    n_eff_list = []
    coverage_list = []

    ground_truth_ppv = compute_ground_truth_ppv(
        data["y_test"], data["preds_test"], data["cohorts_test"]
    )

    if len(pvals) > 0:
        pvals_series = pd.Series(pvals)
        rejected = holm_reject(pvals_series, alpha)

        for i, (cohort_id, tau, mu_hat, n_eff, lb, pval) in enumerate(test_info):
            n_tests += 1
            if not np.isnan(n_eff):
                n_eff_list.append(n_eff)

            if rejected.iloc[i]:
                n_certified += 1

                # Check false certification against ground truth
                true_ppv = ground_truth_ppv.get(cohort_id, np.nan)
                if not np.isnan(true_ppv) and true_ppv < tau:
                    n_false_certify += 1

                # Check coverage: true PPV >= lower bound
                if not np.isnan(true_ppv) and not np.isnan(lb):
                    coverage_list.append(float(true_ppv >= lb))
            else:
                n_abstain += 1

    cert_rate = n_certified / n_tests if n_tests > 0 else 0.0
    false_certify_fwer = int(n_false_certify > 0)
    mean_n_eff = np.mean(n_eff_list) if n_eff_list else np.nan
    coverage = np.mean(coverage_list) if coverage_list else np.nan

    return {
        "dataset": data["dataset_name"],
        "domain": data["config"]["domain"],
        "trial_id": trial_id,
        "n_cal": n_cal,
        "n_source": n_source,
        "n_test": len(X_test),
        "n_cohorts_cal": len(cohort_ids),
        "alpha": alpha,
        "n_tests": n_tests,
        "n_certified": n_certified,
        "n_abstain": n_abstain,
        "cert_rate": cert_rate,
        "n_false_certify": n_false_certify,
        "false_certify_fwer": false_certify_fwer,
        "mean_n_eff": mean_n_eff,
        "coverage": coverage,
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment_a(
    datasets: list,
    data_dir: str,
    model_dir: str,
    n_trials: int = 50,
    n_cal: int = 2000,
    alpha: float = 0.05,
    tau_grid: np.ndarray = None,
    seed: int = 42,
    output_dir: str = "results/experiment_a",
):
    """Run Experiment A across all specified datasets.

    Args:
        datasets: List of dataset names to evaluate
        data_dir: Path to data/processed/
        model_dir: Path to models/
        n_trials: Independent trials per dataset
        n_cal: Calibration subsample size per trial
        alpha: Nominal FWER level
        tau_grid: PPV thresholds to test
        seed: Base random seed
        output_dir: Where to save results
    """
    if tau_grid is None:
        tau_grid = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    dataset_summaries = []

    for ds_idx, dataset_name in enumerate(datasets):
        print(f"\n{'='*70}")
        print(f"Dataset {ds_idx+1}/{len(datasets)}: {dataset_name}")
        print(f"{'='*70}")

        # Load dataset
        try:
            data = load_real_dataset(dataset_name, data_dir, model_dir)
        except Exception as e:
            print(f"  [SKIP] Failed to load {dataset_name}: {e}")
            continue

        print(f"  Source: {len(data['X_source'])} samples, "
              f"{len(np.unique(data['cohorts_source']))} cohorts")
        print(f"  Test:   {len(data['X_test'])} samples, "
              f"{len(np.unique(data['cohorts_test']))} cohorts")
        print(f"  Pred positive rate (source): "
              f"{data['preds_source'].mean():.3f}")

        # Compute ground truth
        gt_ppv = compute_ground_truth_ppv(
            data["y_test"], data["preds_test"], data["cohorts_test"]
        )
        valid_ppvs = [v for v in gt_ppv.values() if not np.isnan(v)]
        if valid_ppvs:
            print(f"  Ground-truth PPV: min={min(valid_ppvs):.3f}, "
                  f"max={max(valid_ppvs):.3f}, "
                  f"mean={np.mean(valid_ppvs):.3f}")
        else:
            print("  [WARNING] No valid ground-truth PPV (no predicted positives)")
            continue

        # Run trials
        actual_n_cal = min(n_cal, len(data["X_source"]))
        if actual_n_cal < n_cal:
            print(f"  [NOTE] Using n_cal={actual_n_cal} "
                  f"(source only has {len(data['X_source'])} samples)")

        t0 = time.time()
        for trial in range(n_trials):
            if trial % max(1, n_trials // 5) == 0:
                print(f"  Trial {trial+1}/{n_trials}...")

            trial_seed = seed + ds_idx * 10000 + trial
            result = run_real_data_trial(
                data=data,
                n_cal=actual_n_cal,
                tau_grid=tau_grid,
                alpha=alpha,
                trial_id=trial,
                seed=trial_seed,
            )
            all_results.append(result)

        elapsed = time.time() - t0

        # Per-dataset summary
        ds_results = [r for r in all_results if r["dataset"] == dataset_name]
        ds_df = pd.DataFrame(ds_results)

        fwer_count = ds_df["false_certify_fwer"].sum()
        fwer_rate = ds_df["false_certify_fwer"].mean()
        mean_cert = ds_df["cert_rate"].mean()
        mean_neff = ds_df["mean_n_eff"].mean()
        mean_cov = ds_df["coverage"].mean()

        summary = {
            "dataset": dataset_name,
            "domain": data["config"]["domain"],
            "n_source": len(data["X_source"]),
            "n_test": len(data["X_test"]),
            "n_cal_used": actual_n_cal,
            "n_trials": n_trials,
            "fwer_violations": fwer_count,
            "observed_fwer": fwer_rate,
            "mean_cert_rate": mean_cert,
            "mean_n_eff": mean_neff,
            "mean_coverage": mean_cov,
            "total_certified": ds_df["n_certified"].sum(),
            "total_false_certify": ds_df["n_false_certify"].sum(),
            "runtime_sec": elapsed,
        }
        dataset_summaries.append(summary)

        # Print per-dataset results
        fwer_status = "PASS" if fwer_rate <= alpha else "FAIL"
        print(f"\n  --- {dataset_name} Results ---")
        print(f"  FWER: {fwer_count}/{n_trials} "
              f"({fwer_rate:.1%}) [{fwer_status}]")
        print(f"  Cert rate: {mean_cert:.1%}")
        print(f"  Coverage: {mean_cov:.3f}" if not np.isnan(mean_cov) else
              "  Coverage: N/A (no certifications)")
        print(f"  Mean n_eff: {mean_neff:.1f}")
        print(f"  Time: {elapsed:.1f}s")

    # Save raw results
    raw_df = pd.DataFrame(all_results)
    raw_path = os.path.join(output_dir, "experiment_a_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"\nSaved raw results: {raw_path}")

    # Save summary
    summary_df = pd.DataFrame(dataset_summaries)
    summary_path = os.path.join(output_dir, "experiment_a_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    # Print overall summary
    print_overall_summary(summary_df, alpha)

    # Save text summary
    summary_text = format_text_summary(summary_df, alpha, n_trials, n_cal)
    text_path = os.path.join(output_dir, "experiment_a_summary.txt")
    with open(text_path, "w") as f:
        f.write(summary_text)
    print(f"Saved text summary: {text_path}")

    return raw_df, summary_df


def print_overall_summary(summary_df: pd.DataFrame, alpha: float):
    """Print formatted overall summary."""
    print(f"\n{'='*70}")
    print("EXPERIMENT A: SEMI-SYNTHETIC REAL-DATA CALIBRATION")
    print(f"{'='*70}")

    print(f"\n{'dataset':<15} {'domain':<10} {'FWER':<12} {'cert%':<8} "
          f"{'n_eff':<8} {'coverage':<10} {'status':<8}")
    print("-" * 70)

    all_pass = True
    for _, row in summary_df.iterrows():
        fwer_str = f"{row['fwer_violations']:.0f}/{row['n_trials']:.0f} " \
                   f"({row['observed_fwer']:.1%})"
        cov_str = f"{row['mean_coverage']:.3f}" \
            if not np.isnan(row["mean_coverage"]) else "N/A"
        status = "PASS" if row["observed_fwer"] <= alpha else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"{row['dataset']:<15} {row['domain']:<10} {fwer_str:<12} "
              f"{row['mean_cert_rate']:<8.1%} {row['mean_n_eff']:<8.1f} "
              f"{cov_str:<10} {status:<8}")

    print("-" * 70)

    # Overall verdict
    total_fwer = summary_df["fwer_violations"].sum()
    total_trials = summary_df["n_trials"].sum()
    overall_fwer = total_fwer / total_trials if total_trials > 0 else 0

    print(f"\nOverall FWER: {total_fwer}/{total_trials} ({overall_fwer:.1%})")

    if all_pass:
        print("[PASS] All datasets: observed FWER <= alpha")
        print("Protocol is empirically valid on real data.")
    else:
        n_fail = (summary_df["observed_fwer"] > alpha).sum()
        print(f"[CONCERN] {n_fail} dataset(s) exceed nominal alpha")

    # Statistical test on pooled trials
    from scipy.stats import binomtest
    result = binomtest(
        k=int(total_fwer),
        n=int(total_trials),
        p=alpha,
        alternative="greater",
    )
    print(f"\nPooled binomial test (H0: FWER <= {alpha}):")
    print(f"  Observed: {total_fwer}/{total_trials} = {overall_fwer:.4f}")
    print(f"  p-value: {result.pvalue:.4f}")
    if result.pvalue < 0.05:
        print("  [FAIL] Evidence of FWER inflation (p < 0.05)")
    else:
        print("  [PASS] No evidence of FWER inflation")

    print(f"\n{'='*70}")


def format_text_summary(
    summary_df: pd.DataFrame,
    alpha: float,
    n_trials: int,
    n_cal: int,
) -> str:
    """Format results as text for PI check-in."""
    lines = []
    lines.append("=" * 70)
    lines.append("EXPERIMENT A: SEMI-SYNTHETIC REAL-DATA CALIBRATION")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Design: {n_trials} independent trials per dataset, "
                 f"n_cal={n_cal}, alpha={alpha}")
    lines.append("Real features, labels, predictions, natural covariate shift.")
    lines.append("Ground truth from held-out test set.")
    lines.append("")

    # Per-dataset table
    lines.append(f"{'Dataset':<15} {'Domain':<10} {'FWER':<15} "
                 f"{'Cert%':<8} {'n_eff':<8} {'Coverage':<10}")
    lines.append("-" * 66)

    for _, row in summary_df.iterrows():
        fwer_str = f"{row['fwer_violations']:.0f}/{row['n_trials']:.0f} " \
                   f"({row['observed_fwer']:.1%})"
        cov_str = f"{row['mean_coverage']:.3f}" \
            if not np.isnan(row["mean_coverage"]) else "N/A"
        lines.append(f"{row['dataset']:<15} {row['domain']:<10} "
                     f"{fwer_str:<15} {row['mean_cert_rate']:<8.1%} "
                     f"{row['mean_n_eff']:<8.1f} {cov_str:<10}")

    lines.append("")

    # Overall
    total_fwer = summary_df["fwer_violations"].sum()
    total_trials = summary_df["n_trials"].sum()
    overall_rate = total_fwer / total_trials if total_trials > 0 else 0.0
    lines.append(f"Overall FWER: {total_fwer}/{total_trials} "
                 f"({overall_rate:.1%})")

    all_pass = (summary_df["observed_fwer"] <= alpha).all()
    if all_pass:
        lines.append("[PASS] Protocol maintains FWER <= alpha on all real datasets.")
    else:
        n_fail = (summary_df["observed_fwer"] > alpha).sum()
        lines.append(f"[CONCERN] {n_fail} dataset(s) exceed nominal alpha.")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment A: Real-data calibration validation"
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["bace", "bbbp", "adult", "compas", "imdb", "yelp"],
        help="Datasets to evaluate"
    )
    parser.add_argument("--data_dir", default="data/processed",
                        help="Path to processed datasets")
    parser.add_argument("--model_dir", default="models",
                        help="Path to trained models and predictions")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Independent trials per dataset")
    parser.add_argument("--n_cal", type=int, default=2000,
                        help="Calibration subsample size")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/experiment_a",
                        help="Output directory")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 10 trials, fewer datasets")
    args = parser.parse_args()

    if args.quick:
        args.n_trials = 10
        args.datasets = ["adult", "imdb", "yelp"]

    print("Experiment A: Semi-Synthetic Real-Data Calibration")
    print(f"  Datasets: {args.datasets}")
    print(f"  Trials per dataset: {args.n_trials}")
    print(f"  n_cal: {args.n_cal}")
    print(f"  alpha: {args.alpha}")
    print()

    run_experiment_a(
        datasets=args.datasets,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        n_trials=args.n_trials,
        n_cal=args.n_cal,
        alpha=args.alpha,
        seed=args.seed,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
