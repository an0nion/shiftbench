#!/usr/bin/env python3
"""
Cross-Domain Benchmark Runner for ShiftBench

Evaluates all methods (uLSIF, KLIEP, RAVEL) across all domains (molecular, text, tabular)
to identify domain-specific insights and compare method robustness.

Usage:
    python scripts/run_cross_domain_benchmark.py --output results/cross_domain/
    python scripts/run_cross_domain_benchmark.py --methods ulsif,kliep --domains molecular,text
"""

import argparse
import concurrent.futures
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from shiftbench.data import get_registry
from shiftbench.baselines import (
    create_ulsif_baseline,
    create_kliep_baseline,
    create_kmm_baseline,
    create_rulsif_baseline,
    create_weighted_conformal_baseline,
    create_split_conformal_baseline,
    create_cvplus_baseline,
    create_group_dro_baseline,
    create_bbse_baseline,
)

# Try importing RAVEL if available
try:
    from shiftbench.baselines import create_ravel_baseline
    RAVEL_AVAILABLE = True
except ImportError:
    RAVEL_AVAILABLE = False
    logging.warning("RAVEL not available - will skip RAVEL baseline")


# Dataset categorization by domain
DOMAIN_DATASETS = {
    "molecular": [
        "bace", "bbbp", "clintox", "esol", "freesolv", "lipophilicity",
        "sider", "tox21", "toxcast"
    ],
    "text": [
        "imdb", "yelp", "amazon", "civil_comments", "twitter"
    ],
    "tabular": [
        "adult", "compas", "bank", "german_credit", "heart_disease", "diabetes"
    ]
}


def setup_logging(output_dir: Path):
    """Configure logging to both file and console."""
    log_file = output_dir / "benchmark.log"
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def get_available_methods() -> Dict[str, callable]:
    """Get dictionary of available method creators."""
    methods = {
        "ulsif": create_ulsif_baseline,
        "kliep": create_kliep_baseline,
        "kmm": create_kmm_baseline,
        "rulsif": create_rulsif_baseline,
        "weighted_conformal": create_weighted_conformal_baseline,
        "split_conformal": create_split_conformal_baseline,
        "cvplus": create_cvplus_baseline,
        "group_dro": create_group_dro_baseline,
        "bbse": create_bbse_baseline,
    }

    if RAVEL_AVAILABLE:
        methods["ravel"] = create_ravel_baseline

    return methods


REGRESSION_DATASETS = {"esol", "freesolv", "lipophilicity"}
NAN_LABEL_DATASETS = {"tox21", "toxcast", "muv"}

# Runtime control: KMM's QP solver is O(n²) — prohibitively slow for large n_cal.
# Cap n_cal at 1000 for KMM; any dataset exceeding this is marked NOT_RUN.
KMM_CAP_N_CAL = 1000
# Group DRO is O(n_iter × n_cohorts × n_cal) — skip large calibration sets.
# molhiv (~11k pairs) took 109s; muv (78k samples) would exceed 10 min.
GROUP_DRO_CAP_N_CAL = 5000
# Wall-clock budget per (method, dataset) evaluation.  Exceeded → TIMEOUT row.
METHOD_TIMEOUT_SECONDS = 600


def binarize_regression_labels(y, splits, dataset_name):
    """Median-split binarization using training set median only.

    Returns (y_binary, threshold).
    """
    train_mask = (splits["split"] == "train").values
    threshold = np.median(y[train_mask])
    y_bin = (y > threshold).astype(int)
    logging.info(f"  {dataset_name}: binarized via median-split "
                 f"(threshold={threshold:.3f}, pos_rate={y_bin.mean():.3f})")
    return y_bin, threshold


def drop_nan_labels(X, y, cohorts, splits, dataset_name):
    """Drop samples with NaN labels and return filtered arrays."""
    valid = ~np.isnan(y)
    n_dropped = (~valid).sum()
    if n_dropped > 0:
        logging.info(f"  {dataset_name}: dropped {n_dropped}/{len(y)} NaN labels")
        X = X[valid]
        y = y[valid]
        cohorts = cohorts[valid]
        splits = splits[valid].reset_index(drop=True)
    return X, y, cohorts, splits


def load_dataset_safe(dataset_name: str):
    """Load dataset with error handling, binarization, and NaN handling."""
    try:
        from shiftbench.data import load_dataset
        X, y, cohorts, splits = load_dataset(dataset_name)

        # Handle NaN labels (tox21, toxcast, muv): replace NaN with 0,
        # matching train_new_datasets.py which uses nan_to_num(nan=0).
        # Dropping NaN causes a sample-count mismatch with saved predictions.
        if dataset_name in NAN_LABEL_DATASETS:
            nan_count = int(np.isnan(y).sum())
            if nan_count > 0:
                logging.info(
                    f"  {dataset_name}: replacing {nan_count} NaN labels with 0 "
                    f"(nan_frac={nan_count/len(y):.3f})"
                )
            y = np.nan_to_num(y, nan=0.0).astype(int)

        # Handle regression datasets (median-split binarization)
        if dataset_name in REGRESSION_DATASETS:
            y, _ = binarize_regression_labels(y, splits, dataset_name)

        return X, y, cohorts, splits
    except Exception as e:
        logging.error(f"Failed to load dataset {dataset_name}: {e}")
        return None


def _load_predictions_for_dataset(dataset_name: str, y_cal: np.ndarray) -> np.ndarray:
    """Load real model predictions for a dataset, fall back to oracle."""
    import json
    project_root = Path(__file__).resolve().parent.parent
    mapping_path = project_root / "models" / "prediction_mapping.json"

    if mapping_path.exists():
        with open(mapping_path) as f:
            mapping = json.load(f)

        if dataset_name in mapping:
            info = mapping[dataset_name]
            pred_path = project_root / info["cal_binary"]
            if pred_path.exists():
                preds = np.load(pred_path)
                if len(preds) == len(y_cal):
                    logging.info(f"  Using real predictions for {dataset_name}")
                    return preds.astype(int)
                else:
                    logging.warning(
                        f"  Prediction length mismatch for {dataset_name}: "
                        f"{len(preds)} vs {len(y_cal)}, falling back to oracle"
                    )

    logging.info(f"  Using oracle predictions for {dataset_name}")
    return (y_cal > 0.5).astype(int)


def _make_status_row(
    dataset_name: str,
    method_name: str,
    status: str,
    elapsed_sec: float = 0.0,
    n_cal: int = 0,
) -> pd.DataFrame:
    """Return a single-row sentinel DataFrame for NOT_RUN or TIMEOUT evaluations."""
    return pd.DataFrame([{
        "dataset": dataset_name,
        "method": method_name,
        "cohort_id": "__status__",
        "tau": float("nan"),
        "decision": "NOT_RUN",
        "mu_hat": float("nan"),
        "lower_bound": float("nan"),
        "p_value": float("nan"),
        "n_eff": float("nan"),
        "elapsed_sec": elapsed_sec,
        "diagnostics": f"status={status} n_cal={n_cal}",
        "status": status,
    }])


def evaluate_method_on_dataset(
    method_name: str,
    method_creator: callable,
    dataset_name: str,
    tau_grid: List[float],
    alpha: float = 0.05
) -> Optional[pd.DataFrame]:
    """
    Evaluate a single method on a single dataset.

    Returns DataFrame with columns:
        dataset, method, cohort_id, tau, decision, mu_hat, lower_bound,
        p_value, n_eff, elapsed_sec, diagnostics, status

    status values:
        "OK"       — normal evaluation
        "NOT_RUN"  — skipped (e.g. KMM n_cal > KMM_CAP_N_CAL)
        "TIMEOUT"  — exceeded METHOD_TIMEOUT_SECONDS wall-clock budget
    """
    logging.info(f"Evaluating {method_name} on {dataset_name}...")

    # Load dataset
    result = load_dataset_safe(dataset_name)
    if result is None:
        return None

    X, y, cohorts, splits = result

    # Get calibration and test splits
    cal_mask = (splits["split"] == "cal").values
    test_mask = (splits["split"] == "test").values

    X_cal = X[cal_mask]
    y_cal = y[cal_mask]
    cohorts_cal = cohorts[cal_mask]
    X_test = X[test_mask]
    n_cal = int(X_cal.shape[0])

    # KMM cap: QP solver is O(n²); skip large calibration sets
    if method_name == "kmm" and n_cal > KMM_CAP_N_CAL:
        logging.warning(
            f"  KMM skipped: n_cal={n_cal} > KMM_CAP_N_CAL={KMM_CAP_N_CAL}. "
            f"Marking as NOT_RUN."
        )
        return _make_status_row(dataset_name, method_name, "NOT_RUN", 0.0, n_cal)

    # Group DRO cap: O(n_iter × n_cohorts × n_cal) — prohibitively slow for large datasets
    if method_name == "group_dro" and n_cal > GROUP_DRO_CAP_N_CAL:
        logging.warning(
            f"  group_dro skipped: n_cal={n_cal} > GROUP_DRO_CAP_N_CAL={GROUP_DRO_CAP_N_CAL}. "
            f"Marking as NOT_RUN."
        )
        return _make_status_row(dataset_name, method_name, "NOT_RUN", 0.0, n_cal)

    # Create method instance
    try:
        method = method_creator()
    except Exception as e:
        logging.error(f"Failed to create method {method_name}: {e}")
        return None

    # Load real model predictions if available, fall back to oracle
    predictions_cal = _load_predictions_for_dataset(dataset_name, y_cal)

    # Core evaluation (weight estimation + bounds) with wall-clock timeout.
    # NOTE: do NOT use 'with ThreadPoolExecutor(...) as executor:' here — the
    # context manager calls shutdown(wait=True) on __exit__, which blocks until
    # the submitted future completes even after a TimeoutError is caught.  This
    # caused the process to deadlock on large datasets (e.g. muv).  Instead we
    # create the executor manually and always call shutdown(wait=False).
    def _run_core():
        t0 = time.time()
        weights = method.estimate_weights(X_cal, X_test)
        t_weights = time.time() - t0

        t0 = time.time()
        decisions = method.estimate_bounds(
            y_cal, predictions_cal, cohorts_cal, weights, tau_grid, alpha
        )
        t_bounds = time.time() - t0
        return decisions, t_weights + t_bounds

    t_wall_start = time.time()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_run_core)
        try:
            decisions, elapsed_sec = future.result(timeout=METHOD_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            elapsed_wall = time.time() - t_wall_start
            logging.warning(
                f"  TIMEOUT: {method_name} on {dataset_name} exceeded "
                f"{METHOD_TIMEOUT_SECONDS}s (elapsed={elapsed_wall:.1f}s)"
            )
            return _make_status_row(dataset_name, method_name, "TIMEOUT", elapsed_wall, n_cal)
        except Exception as e:
            logging.error(f"Evaluation failed for {method_name} on {dataset_name}: {e}")
            return None
    finally:
        # shutdown(wait=False): release the executor without blocking on stuck threads.
        executor.shutdown(wait=False)

    # Convert decisions to DataFrame
    records = []
    for d in decisions:
        records.append({
            "dataset": dataset_name,
            "method": method_name,
            "cohort_id": d.cohort_id,
            "tau": d.tau,
            "decision": d.decision if isinstance(d.decision, str) else d.decision.value,
            "mu_hat": d.mu_hat,
            "lower_bound": d.lower_bound,
            "p_value": d.p_value,
            "n_eff": d.n_eff,
            "elapsed_sec": elapsed_sec,
            "diagnostics": str(d.diagnostics) if d.diagnostics else "",
            "status": "OK",
        })

    df = pd.DataFrame(records)

    # Log summary
    n_certify = (df["decision"] == "CERTIFY").sum()
    n_abstain = (df["decision"] == "ABSTAIN").sum()
    n_no_guarantee = (df["decision"] == "NO-GUARANTEE").sum()

    logging.info(
        f"  Results: {n_certify} CERTIFY, {n_abstain} ABSTAIN, "
        f"{n_no_guarantee} NO-GUARANTEE (total: {len(df)}, elapsed={elapsed_sec:.1f}s)"
    )

    return df


def run_cross_domain_benchmark(
    methods: List[str],
    domains: List[str],
    output_dir: Path,
    tau_grid: List[float] = None,
    alpha: float = 0.05
):
    """
    Run comprehensive cross-domain benchmark.

    Args:
        methods: List of method names to evaluate
        domains: List of domain names to include
        output_dir: Directory to save results
        tau_grid: List of PPV thresholds to test
        alpha: Family-wise error rate
    """
    if tau_grid is None:
        tau_grid = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir)

    # Get available methods
    available_methods = get_available_methods()

    # Validate methods
    for method in methods:
        if method not in available_methods:
            logging.error(f"Unknown method: {method}")
            logging.info(f"Available methods: {list(available_methods.keys())}")
            sys.exit(1)

    # Get datasets for requested domains
    datasets = []
    for domain in domains:
        if domain not in DOMAIN_DATASETS:
            logging.error(f"Unknown domain: {domain}")
            logging.info(f"Available domains: {list(DOMAIN_DATASETS.keys())}")
            sys.exit(1)
        datasets.extend(DOMAIN_DATASETS[domain])

    # Verify datasets exist in registry
    registry = get_registry()
    available_datasets = registry.list_datasets()
    datasets = [d for d in datasets if d in available_datasets]

    logging.info(f"=" * 80)
    logging.info(f"Cross-Domain Benchmark Configuration")
    logging.info(f"=" * 80)
    logging.info(f"Methods: {methods}")
    logging.info(f"Domains: {domains}")
    logging.info(f"Datasets: {len(datasets)} ({', '.join(datasets)})")
    logging.info(f"Tau grid: {tau_grid}")
    logging.info(f"Alpha: {alpha}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"=" * 80)

    # Run evaluations
    all_results = []
    n_total = len(methods) * len(datasets)

    with tqdm(total=n_total, desc="Overall Progress") as pbar:
        for method_name in methods:
            method_creator = available_methods[method_name]

            for dataset_name in datasets:
                # Get domain
                domain = None
                for d, dsets in DOMAIN_DATASETS.items():
                    if dataset_name in dsets:
                        domain = d
                        break

                # Evaluate
                df = evaluate_method_on_dataset(
                    method_name, method_creator, dataset_name, tau_grid, alpha
                )

                if df is not None:
                    df["domain"] = domain
                    all_results.append(df)

                pbar.update(1)

    # Combine all results
    if not all_results:
        logging.error("No results collected!")
        return

    df_all = pd.concat(all_results, ignore_index=True)

    # Save raw results
    raw_file = output_dir / "cross_domain_raw_results.csv"
    df_all.to_csv(raw_file, index=False)
    logging.info(f"Saved raw results to {raw_file}")

    # Generate summary tables
    generate_summary_tables(df_all, output_dir)

    # Run statistical analysis
    run_statistical_analysis(df_all, output_dir)

    logging.info(f"=" * 80)
    logging.info(f"Cross-Domain Benchmark Complete!")
    logging.info(f"Results saved to: {output_dir}")
    logging.info(f"=" * 80)


def generate_summary_tables(df: pd.DataFrame, output_dir: Path):
    """Generate summary CSV files."""
    logging.info("Generating summary tables...")

    # Separate OK rows (real decisions) from sentinel rows (NOT_RUN / TIMEOUT)
    if "status" in df.columns:
        df_ok = df[df["status"] == "OK"].copy()
        df_sentinel = df[df["status"] != "OK"].copy()
    else:
        df_ok = df.copy()
        df_sentinel = pd.DataFrame()

    # 1. Cross-domain summary: Aggregated by domain (OK rows only)
    summary_by_domain = df_ok.groupby(["domain", "method"]).agg({
        "decision": lambda x: (x == "CERTIFY").sum() / len(x) * 100,  # cert rate %
        "mu_hat": "mean",
        "lower_bound": "mean",
        "n_eff": "mean",
        "elapsed_sec": "mean",
        "dataset": "nunique",  # number of datasets
        "cohort_id": "nunique"  # number of cohorts
    }).round(2)

    summary_by_domain.columns = [
        "cert_rate_%", "mean_mu_hat", "mean_lower_bound",
        "mean_n_eff", "mean_runtime_sec", "n_datasets", "n_cohorts"
    ]

    summary_file = output_dir / "cross_domain_summary.csv"
    summary_by_domain.to_csv(summary_file)
    logging.info(f"  Saved domain summary to {summary_file}")

    # 2. Per-dataset breakdown (OK rows only)
    summary_by_dataset = df_ok.groupby(["dataset", "domain", "method"]).agg({
        "decision": lambda x: (x == "CERTIFY").sum() / len(x) * 100,
        "mu_hat": "mean",
        "lower_bound": "mean",
        "n_eff": "mean",
        "elapsed_sec": "first",  # same value repeated across cohort/tau rows
        "cohort_id": "nunique"
    }).round(2)

    summary_by_dataset.columns = [
        "cert_rate_%", "mean_mu_hat", "mean_lower_bound",
        "mean_n_eff", "runtime_sec", "n_cohorts"
    ]

    dataset_file = output_dir / "cross_domain_by_dataset.csv"
    summary_by_dataset.to_csv(dataset_file)
    logging.info(f"  Saved dataset breakdown to {dataset_file}")

    # 3. Per-method breakdown (OK rows only)
    summary_by_method = df_ok.groupby(["method", "domain"]).agg({
        "decision": lambda x: (x == "CERTIFY").sum() / len(x) * 100,
        "mu_hat": "mean",
        "lower_bound": "mean",
        "n_eff": "mean",
        "elapsed_sec": "mean",
        "dataset": "nunique"
    }).round(2)

    summary_by_method.columns = [
        "cert_rate_%", "mean_mu_hat", "mean_lower_bound",
        "mean_n_eff", "mean_runtime_sec", "n_datasets"
    ]

    method_file = output_dir / "cross_domain_by_method.csv"
    summary_by_method.to_csv(method_file)
    logging.info(f"  Saved method breakdown to {method_file}")

    # 4. Decision distribution table (OK rows only)
    decision_dist = pd.crosstab(
        [df_ok["domain"], df_ok["method"]],
        df_ok["decision"],
        margins=True
    )

    decision_file = output_dir / "cross_domain_decision_distribution.csv"
    decision_dist.to_csv(decision_file)
    logging.info(f"  Saved decision distribution to {decision_file}")

    # 5. Runtime table — all (method, dataset) combinations including NOT_RUN / TIMEOUT
    runtime_rows = []
    # OK evaluations: one row per (dataset, method), runtime = first elapsed_sec value
    if not df_ok.empty:
        for (dataset, method), grp in df_ok.groupby(["dataset", "method"]):
            domain = grp["domain"].iloc[0] if "domain" in grp.columns else ""
            runtime_rows.append({
                "dataset": dataset,
                "domain": domain,
                "method": method,
                "status": "OK",
                "runtime_sec": round(grp["elapsed_sec"].iloc[0], 2),
                "n_cohort_tau_pairs": len(grp),
            })
    # Sentinel evaluations (NOT_RUN, TIMEOUT)
    if not df_sentinel.empty:
        for _, row in df_sentinel.iterrows():
            runtime_rows.append({
                "dataset": row["dataset"],
                "domain": row.get("domain", ""),
                "method": row["method"],
                "status": row["status"],
                "runtime_sec": round(row["elapsed_sec"], 2),
                "n_cohort_tau_pairs": 0,
            })

    if runtime_rows:
        runtime_table = (
            pd.DataFrame(runtime_rows)
            .sort_values(["method", "domain", "dataset"])
            .reset_index(drop=True)
        )
        runtime_file = output_dir / "runtime_table.csv"
        runtime_table.to_csv(runtime_file, index=False)
        logging.info(f"  Saved runtime table to {runtime_file}")

        # Log a compact summary of NOT_RUN / TIMEOUT entries
        skipped = runtime_table[runtime_table["status"] != "OK"]
        if not skipped.empty:
            logging.info(
                f"  Skipped evaluations: {len(skipped)} "
                f"({skipped['status'].value_counts().to_dict()})"
            )


def run_statistical_analysis(df: pd.DataFrame, output_dir: Path):
    """Run statistical analyses on cross-domain results."""
    logging.info("Running statistical analysis...")

    try:
        from scipy import stats
        from scipy.stats import f_oneway, kruskal
    except ImportError:
        logging.warning("scipy not available - skipping statistical tests")
        return

    # Use only completed evaluations for statistical analysis
    if "status" in df.columns:
        df = df[df["status"] == "OK"].copy()

    if df.empty:
        logging.warning("No completed evaluations for statistical analysis.")
        return

    # Prepare binary outcome (1 = CERTIFY, 0 = otherwise)
    df["certified"] = (df["decision"] == "CERTIFY").astype(int)

    analysis_results = []

    # 1. Do certification rates differ by domain? (Kruskal-Wallis test)
    logging.info("  Testing: Do certification rates differ by domain?")

    domain_groups = [group["certified"].values for name, group in df.groupby("domain")]

    if len(domain_groups) >= 2:
        h_stat, p_value = kruskal(*domain_groups)
        analysis_results.append({
            "test": "Certification rate by domain (Kruskal-Wallis)",
            "statistic": h_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "interpretation": "Certification rates differ significantly across domains"
                            if p_value < 0.05
                            else "No significant difference in certification rates across domains"
        })

    # 2. Does method ranking change by domain? (chi-square test)
    logging.info("  Testing: Does method ranking change by domain?")

    contingency_table = pd.crosstab(df["domain"], df["method"], values=df["certified"], aggfunc="sum")

    if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        analysis_results.append({
            "test": "Method-domain interaction (Chi-square)",
            "statistic": chi2,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "interpretation": "Method performance varies significantly by domain"
                            if p_value < 0.05
                            else "Method performance is consistent across domains"
        })

    # 3. Which domain is hardest? (mean certification rate)
    logging.info("  Computing: Domain difficulty ranking")

    domain_difficulty = df.groupby("domain")["certified"].agg(["mean", "std", "count"]).round(4)
    domain_difficulty = domain_difficulty.sort_values("mean")
    domain_difficulty.columns = ["cert_rate", "std", "n_decisions"]

    analysis_results.append({
        "test": "Domain difficulty (by certification rate)",
        "statistic": None,
        "p_value": None,
        "significant": None,
        "interpretation": f"Hardest domain: {domain_difficulty.index[0]} (cert rate: {domain_difficulty.iloc[0]['cert_rate']:.2%})"
    })

    # 4. Mean effective sample size by domain
    logging.info("  Computing: Effective sample size by domain")

    ess_by_domain = df.groupby("domain")["n_eff"].agg(["mean", "median", "std"]).round(2)

    # 5. Runtime comparison
    logging.info("  Computing: Runtime by domain")

    runtime_by_domain = df.groupby(["domain", "method"])["elapsed_sec"].mean().round(3)

    # Save analysis results
    df_analysis = pd.DataFrame(analysis_results)
    analysis_file = output_dir / "cross_domain_statistical_analysis.csv"
    df_analysis.to_csv(analysis_file, index=False)
    logging.info(f"  Saved statistical analysis to {analysis_file}")

    # Save domain difficulty
    difficulty_file = output_dir / "cross_domain_difficulty.csv"
    domain_difficulty.to_csv(difficulty_file)
    logging.info(f"  Saved domain difficulty to {difficulty_file}")

    # Save ESS summary
    ess_file = output_dir / "cross_domain_ess_summary.csv"
    ess_by_domain.to_csv(ess_file)
    logging.info(f"  Saved ESS summary to {ess_file}")

    # Save runtime summary
    runtime_file = output_dir / "cross_domain_runtime.csv"
    runtime_by_domain.to_csv(runtime_file)
    logging.info(f"  Saved runtime summary to {runtime_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run cross-domain benchmark for ShiftBench",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--methods",
        type=str,
        default="ulsif,kliep,ravel",
        help="Comma-separated list of methods to evaluate"
    )

    parser.add_argument(
        "--domains",
        type=str,
        default="molecular,text,tabular",
        help="Comma-separated list of domains to include"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/cross_domain/",
        help="Output directory for results"
    )

    parser.add_argument(
        "--tau",
        type=str,
        default="0.5,0.6,0.7,0.8,0.85,0.9",
        help="Comma-separated list of tau thresholds"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Family-wise error rate"
    )

    args = parser.parse_args()

    # Parse arguments
    methods = [m.strip() for m in args.methods.split(",")]
    domains = [d.strip() for d in args.domains.split(",")]
    tau_grid = [float(t.strip()) for t in args.tau.split(",")]

    # Run benchmark
    run_cross_domain_benchmark(
        methods=methods,
        domains=domains,
        output_dir=Path(args.output),
        tau_grid=tau_grid,
        alpha=args.alpha
    )


if __name__ == "__main__":
    main()
