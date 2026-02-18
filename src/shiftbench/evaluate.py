"""ShiftBench Evaluation Harness.

This module provides a unified interface for evaluating baseline methods
on ShiftBench datasets. It handles:
- Dataset loading and splitting
- Weight estimation
- Oracle or real model prediction loading
- Bound estimation for all cohorts and tau values
- Result aggregation and CSV export

Usage:
    # Evaluate with oracle predictions (default)
    python -m shiftbench.evaluate --method ulsif --dataset bace --output results/

    # Evaluate with real model predictions
    python -m shiftbench.evaluate --method ulsif --dataset bace --no-oracle --output results/

    # Batch evaluation (all methods on all datasets)
    python -m shiftbench.evaluate --method all --dataset all --output results/
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from shiftbench.data import DatasetRegistry, load_dataset

# Root of the shift-bench project
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Registry of available methods
AVAILABLE_METHODS = {
    "ulsif": {
        "module": "shiftbench.baselines.ulsif",
        "factory": "create_ulsif_baseline",
        "default_params": {
            "n_basis": 100,
            "sigma": None,
            "lambda_": 0.1,
            "random_state": 42,
        },
    },
    "kliep": {
        "module": "shiftbench.baselines.kliep",
        "factory": "create_kliep_baseline",
        "default_params": {
            "n_basis": 100,
            "sigma": None,
            "max_iter": 10000,
            "random_state": 42,
        },
    },
    "kmm": {
        "module": "shiftbench.baselines.kmm",
        "factory": "create_kmm_baseline",
        "default_params": {
            "sigma": None,
            "lambda_": 0.1,
            "B": 1000.0,
            "random_state": 42,
            "solver": "auto",
        },
    },
    "rulsif": {
        "module": "shiftbench.baselines.rulsif",
        "factory": "create_rulsif_baseline",
        "default_params": {
            "n_basis": 100,
            "sigma": None,
            "lambda_": 0.1,
            "random_state": 42,
        },
    },
    "ravel": {
        "module": "shiftbench.baselines.ravel",
        "factory": "create_ravel_baseline",
        "default_params": {
            "n_folds": 5,
            "random_state": 42,
            "logit_temp": 1.75,
            "psis_k_cap": 0.70,
            "ess_min_frac": 0.30,
            "clip_mass_cap": 0.10,
        },
    },
    "weighted_conformal": {
        "module": "shiftbench.baselines.weighted_conformal",
        "factory": "create_weighted_conformal_baseline",
        "default_params": {
            "weight_method": "ulsif",
            "n_basis": 100,
            "sigma": None,
            "lambda_": 0.1,
            "max_iter": 10000,
            "tol": 1e-6,
            "random_state": 42,
        },
    },
}


def load_method(method_name: str, **kwargs):
    """Load a baseline method by name.

    Args:
        method_name: Name of method (e.g., "ulsif", "ravel")
        **kwargs: Override default hyperparameters

    Returns:
        Baseline method instance

    Raises:
        ValueError: If method not found
        ImportError: If method dependencies not available
    """
    if method_name not in AVAILABLE_METHODS:
        raise ValueError(
            f"Method '{method_name}' not found. "
            f"Available methods: {list(AVAILABLE_METHODS.keys())}"
        )

    method_info = AVAILABLE_METHODS[method_name]
    module_name = method_info["module"]
    factory_name = method_info["factory"]

    # Import module
    try:
        import importlib
        module = importlib.import_module(module_name)
        factory = getattr(module, factory_name)
    except ImportError as e:
        raise ImportError(
            f"Failed to import {method_name} from {module_name}. "
            f"Make sure all dependencies are installed. Error: {e}"
        )

    # Merge default params with overrides
    params = method_info["default_params"].copy()
    params.update(kwargs)

    # Create method instance
    logger.info(f"Creating {method_name} with params: {params}")
    return factory(**params)


def _load_real_predictions(dataset_name: str, expected_n: int) -> np.ndarray:
    """Load real model predictions for a dataset from models/predictions/.

    Uses the prediction_mapping.json to find the right model and file.

    Args:
        dataset_name: Name of dataset (e.g., "bace", "adult")
        expected_n: Expected number of calibration samples (for validation)

    Returns:
        Binary prediction array (n_cal,) of 0/1 ints

    Raises:
        FileNotFoundError: If predictions not available for this dataset
        ValueError: If prediction array size doesn't match calibration set
    """
    mapping_path = _PROJECT_ROOT / "models" / "prediction_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Prediction mapping not found at {mapping_path}. "
            "Run scripts/train_core_models.py first."
        )

    with open(mapping_path) as f:
        mapping = json.load(f)

    if dataset_name not in mapping:
        raise FileNotFoundError(
            f"No real predictions available for '{dataset_name}'. "
            f"Available: {list(mapping.keys())}"
        )

    info = mapping[dataset_name]
    pred_path = _PROJECT_ROOT / info["cal_binary"]

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    predictions = np.load(pred_path)

    if len(predictions) != expected_n:
        raise ValueError(
            f"Prediction array length ({len(predictions)}) doesn't match "
            f"calibration set size ({expected_n}) for {dataset_name}"
        )

    return predictions.astype(int)


def evaluate_single_run(
    dataset_name: str,
    method_name: str,
    tau_grid: Optional[List[float]] = None,
    alpha: float = 0.05,
    use_oracle: bool = True,
    method_params: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Evaluate a single (dataset, method) pair.

    Args:
        dataset_name: Name of dataset (e.g., "bace", "test_dataset")
        method_name: Name of method (e.g., "ulsif", "ravel")
        tau_grid: List of PPV thresholds. If None, use dataset default.
        alpha: Significance level (default: 0.05 for 95% confidence)
        use_oracle: If True, use true labels as predictions (for testing)
        method_params: Optional dict of method hyperparameters

    Returns:
        results_df: DataFrame with columns [dataset, method, cohort_id, tau,
            decision, mu_hat, lower_bound, p_value, n_eff, elapsed_sec]
        metadata: Dict with run metadata (n_samples, n_cohorts, etc.)

    Raises:
        FileNotFoundError: If dataset files not found
        ValueError: If method not available
        RuntimeError: If evaluation fails
    """
    logger.info(f"[START] Evaluating {method_name} on {dataset_name}")
    start_time = time.time()

    # Step 1: Load dataset
    logger.info("  [1/5] Loading dataset...")
    try:
        X, y, cohorts, splits = load_dataset(dataset_name)
        logger.info(f"    Loaded {len(X)} samples with {X.shape[1]} features")
        logger.info(f"    Cohorts: {len(np.unique(cohorts))} unique")
        logger.info(f"    Positive rate: {y.mean():.2%}")
    except FileNotFoundError as e:
        logger.error(f"    Dataset not found: {e}")
        raise

    # Step 2: Split into calibration and test
    logger.info("  [2/5] Splitting into calibration and test sets...")
    cal_mask = (splits["split"] == "cal").values
    test_mask = (splits["split"] == "test").values

    X_cal, y_cal, cohorts_cal = X[cal_mask], y[cal_mask], cohorts[cal_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # --- EVALUATION CONTRACT ---
    # y_test is held out and MUST NOT be used for:
    #   - Weight estimation (only X_cal, X_test allowed)
    #   - Bound computation (only y_cal, weights allowed)
    #   - Hyperparameter tuning
    # y_test is ONLY used for post-hoc validation (ground-truth PPV comparison).
    # Violation of this contract invalidates all certification results.
    # ----------------------------

    logger.info(f"    Calibration: {len(X_cal)} samples ({y_cal.mean():.2%} positive)")
    logger.info(f"    Test: {len(X_test)} samples ({y_test.mean():.2%} positive)")

    # Step 3: Load method and estimate weights
    logger.info(f"  [3/5] Estimating weights with {method_name}...")
    try:
        method = load_method(method_name, **(method_params or {}))
        weight_start = time.time()
        weights = method.estimate_weights(X_cal, X_test)
        weight_elapsed = time.time() - weight_start

        logger.info(f"    Estimated weights in {weight_elapsed:.1f}s")
        logger.info(f"    Mean: {weights.mean():.3f}, Std: {weights.std():.3f}")
        logger.info(f"    Range: [{weights.min():.3f}, {weights.max():.3f}]")

        # Validate weights
        assert np.all(weights > 0), "All weights must be positive"
        assert np.all(np.isfinite(weights)), "All weights must be finite"

    except Exception as e:
        logger.error(f"    Weight estimation failed: {e}")
        raise RuntimeError(f"Weight estimation failed for {method_name}: {e}")

    # Step 4: Generate or load predictions
    logger.info("  [4/5] Loading predictions...")
    if use_oracle:
        predictions_cal = y_cal.astype(int)
        logger.info(f"    Using oracle predictions (true labels)")
        logger.info(f"    Predicted positives: {predictions_cal.sum()} / {len(predictions_cal)}")
    else:
        predictions_cal = _load_real_predictions(dataset_name, len(y_cal))
        logger.info(f"    Using real model predictions")
        logger.info(f"    Predicted positives: {predictions_cal.sum()} / {len(predictions_cal)}")
        accuracy = (predictions_cal == y_cal.astype(int)).mean()
        logger.info(f"    Prediction accuracy: {accuracy:.3f}")

    # Step 5: Get tau grid
    if tau_grid is None:
        # Use dataset default from registry
        registry = DatasetRegistry()
        info = registry.get_dataset_info(dataset_name)
        tau_grid = info.get("tau_grid", [0.5, 0.6, 0.7, 0.8, 0.85, 0.9])
    logger.info(f"    Using tau grid: {tau_grid}")

    # Step 6: Estimate bounds
    logger.info(f"  [5/5] Estimating bounds for {len(np.unique(cohorts_cal))} cohorts x {len(tau_grid)} tau values...")
    try:
        bound_start = time.time()
        decisions = method.estimate_bounds(
            y_cal,
            predictions_cal,
            cohorts_cal,
            weights,
            tau_grid,
            alpha=alpha,
        )
        bound_elapsed = time.time() - bound_start

        logger.info(f"    Generated {len(decisions)} decisions in {bound_elapsed:.1f}s")

        # Summary stats
        n_certify = sum(1 for d in decisions if d.decision == "CERTIFY")
        n_abstain = sum(1 for d in decisions if d.decision == "ABSTAIN")
        n_no_guarantee = sum(1 for d in decisions if d.decision == "NO-GUARANTEE")

        logger.info(f"    CERTIFY: {n_certify} ({n_certify/len(decisions):.1%})")
        logger.info(f"    ABSTAIN: {n_abstain} ({n_abstain/len(decisions):.1%})")
        logger.info(f"    NO-GUARANTEE: {n_no_guarantee} ({n_no_guarantee/len(decisions):.1%})")

    except Exception as e:
        logger.error(f"    Bound estimation failed: {e}")
        raise RuntimeError(f"Bound estimation failed for {method_name}: {e}")

    # Step 7: Convert to DataFrame
    total_elapsed = time.time() - start_time

    results_df = pd.DataFrame([
        {
            "dataset": dataset_name,
            "method": method_name,
            "cohort_id": d.cohort_id,
            "tau": d.tau,
            "decision": d.decision,
            "mu_hat": d.mu_hat,
            "var_hat": d.var_hat,
            "n_eff": d.n_eff,
            "lower_bound": d.lower_bound,
            "p_value": d.p_value,
            "elapsed_sec": total_elapsed,  # Total runtime for this (dataset, method) pair
        }
        for d in decisions
    ])

    # Metadata
    metadata = {
        "dataset": dataset_name,
        "method": method_name,
        "n_samples_total": len(X),
        "n_calibration": len(X_cal),
        "n_test": len(X_test),
        "n_features": X.shape[1],
        "n_cohorts": len(np.unique(cohorts_cal)),
        "n_decisions": len(decisions),
        "n_certify": n_certify,
        "n_abstain": n_abstain,
        "n_no_guarantee": n_no_guarantee,
        "weight_elapsed_sec": weight_elapsed,
        "bound_elapsed_sec": bound_elapsed,
        "total_elapsed_sec": total_elapsed,
        "alpha": alpha,
        "use_oracle": use_oracle,
    }

    # Add method diagnostics
    diagnostics = method.get_diagnostics()
    if diagnostics:
        for key, value in diagnostics.items():
            metadata[f"diag_{key}"] = value

    logger.info(f"[SUCCESS] Completed {method_name} on {dataset_name} in {total_elapsed:.1f}s")

    return results_df, metadata


def evaluate_batch(
    dataset_names: List[str],
    method_names: List[str],
    output_dir: Path,
    tau_grid: Optional[List[float]] = None,
    alpha: float = 0.05,
    use_oracle: bool = True,
    continue_on_error: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate multiple methods on multiple datasets.

    Args:
        dataset_names: List of dataset names
        method_names: List of method names
        output_dir: Directory to save results
        tau_grid: Optional custom tau grid
        alpha: Significance level
        use_oracle: Use oracle predictions
        continue_on_error: If True, continue to next run on failure

    Returns:
        all_results: DataFrame with all decisions
        all_metadata: DataFrame with run metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_metadata = []

    # Total number of runs
    n_total = len(dataset_names) * len(method_names)

    # Progress bar
    with tqdm(total=n_total, desc="Evaluating", unit="run") as pbar:
        for dataset_name in dataset_names:
            for method_name in method_names:
                pbar.set_description(f"Evaluating {method_name} on {dataset_name}")

                try:
                    results_df, metadata = evaluate_single_run(
                        dataset_name=dataset_name,
                        method_name=method_name,
                        tau_grid=tau_grid,
                        alpha=alpha,
                        use_oracle=use_oracle,
                    )

                    all_results.append(results_df)
                    all_metadata.append(metadata)

                    # Save intermediate results
                    output_file = output_dir / f"{method_name}_{dataset_name}_results.csv"
                    results_df.to_csv(output_file, index=False)
                    logger.info(f"  Saved results to {output_file}")

                except Exception as e:
                    logger.error(f"[ERROR] Failed {method_name} on {dataset_name}: {e}")
                    if not continue_on_error:
                        raise
                    else:
                        # Record failure in metadata
                        all_metadata.append({
                            "dataset": dataset_name,
                            "method": method_name,
                            "status": "FAILED",
                            "error": str(e),
                        })

                pbar.update(1)

    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_metadata = pd.DataFrame(all_metadata)

        # Save combined results
        combined_results.to_csv(output_dir / "all_results.csv", index=False)
        combined_metadata.to_csv(output_dir / "all_metadata.csv", index=False)

        logger.info(f"\n[SUCCESS] Batch evaluation complete!")
        logger.info(f"  Total runs: {n_total}")
        logger.info(f"  Successful: {len(all_results)}")
        logger.info(f"  Failed: {n_total - len(all_results)}")
        logger.info(f"  Results saved to: {output_dir}")

        return combined_results, combined_metadata
    else:
        logger.error("No successful runs!")
        return pd.DataFrame(), pd.DataFrame()


def aggregate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results by (method, dataset, tau).

    Args:
        results_df: Raw results from evaluate_batch

    Returns:
        aggregated_df: Summary statistics
    """
    if results_df.empty:
        return pd.DataFrame()

    # Group by method, dataset, tau
    grouped = results_df.groupby(["method", "dataset", "tau"])

    agg_df = grouped.agg(
        n_cohorts=("cohort_id", "count"),
        n_certify=("decision", lambda x: (x == "CERTIFY").sum()),
        n_abstain=("decision", lambda x: (x == "ABSTAIN").sum()),
        n_no_guarantee=("decision", lambda x: (x == "NO-GUARANTEE").sum()),
        mean_mu_hat=("mu_hat", lambda x: x[~np.isnan(x)].mean()),
        mean_lower_bound=("lower_bound", lambda x: x[~np.isnan(x)].mean()),
        mean_n_eff=("n_eff", lambda x: x[x > 0].mean()),
        elapsed_sec=("elapsed_sec", "first"),
    ).reset_index()

    # Add certification rate
    agg_df["cert_rate"] = agg_df["n_certify"] / agg_df["n_cohorts"]

    return agg_df


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="ShiftBench Evaluation Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate uLSIF on BACE
  python -m shiftbench.evaluate --method ulsif --dataset bace --output results/

  # Batch evaluation (all methods on all datasets)
  python -m shiftbench.evaluate --method all --dataset all --output results/

  # Custom tau grid
  python -m shiftbench.evaluate --method ulsif --dataset bace --tau 0.5,0.7,0.9

  # Evaluate on test dataset (for debugging)
  python -m shiftbench.evaluate --method ulsif --dataset test_dataset
        """,
    )

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help=f"Method name or 'all'. Available: {list(AVAILABLE_METHODS.keys())}",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name or 'all'. Use 'list' to see available datasets.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--tau",
        type=str,
        default=None,
        help="Comma-separated tau values (e.g., '0.5,0.7,0.9'). If None, use dataset default.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05 for 95%% confidence)",
    )
    parser.add_argument(
        "--no-oracle",
        action="store_true",
        help="Don't use oracle predictions (requires model)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first error (default: continue on error)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse tau grid
    tau_grid = None
    if args.tau:
        tau_grid = [float(t.strip()) for t in args.tau.split(",")]
        logger.info(f"Using custom tau grid: {tau_grid}")

    # Get dataset registry
    registry = DatasetRegistry()

    # Handle special cases
    if args.dataset == "list":
        print("\nAvailable datasets:")
        for name in registry.list_datasets():
            info = registry.get_dataset_info(name)
            print(f"  - {name:20s} ({info['domain']:10s}, {info['n_samples']:6d} samples)")
        return

    # Expand "all"
    if args.method == "all":
        method_names = list(AVAILABLE_METHODS.keys())
        logger.info(f"Evaluating all methods: {method_names}")
    else:
        method_names = [args.method]

    if args.dataset == "all":
        dataset_names = registry.list_datasets()
        logger.info(f"Evaluating all datasets: {dataset_names}")
    else:
        dataset_names = [args.dataset]

    # Run batch evaluation
    output_dir = Path(args.output)
    logger.info(f"\nStarting batch evaluation:")
    logger.info(f"  Methods: {method_names}")
    logger.info(f"  Datasets: {dataset_names}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Total runs: {len(method_names) * len(dataset_names)}")
    logger.info("")

    results_df, metadata_df = evaluate_batch(
        dataset_names=dataset_names,
        method_names=method_names,
        output_dir=output_dir,
        tau_grid=tau_grid,
        alpha=args.alpha,
        use_oracle=not args.no_oracle,
        continue_on_error=not args.fail_fast,
    )

    # Generate aggregated summary
    if not results_df.empty:
        agg_df = aggregate_results(results_df)
        agg_df.to_csv(output_dir / "aggregated_summary.csv", index=False)
        logger.info(f"\nAggregated summary saved to: {output_dir / 'aggregated_summary.csv'}")

        # Print summary table
        print("\n" + "=" * 80)
        print("SUMMARY: Certification Rates by (Method, Dataset, Tau)")
        print("=" * 80)
        print(agg_df[["method", "dataset", "tau", "n_cohorts", "n_certify", "cert_rate"]].to_string(index=False))
        print("=" * 80)


if __name__ == "__main__":
    main()
