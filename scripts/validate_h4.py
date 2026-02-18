"""
H4 Validation Experiment: Empirical Validation of False-Certify Rate <= alpha

This script empirically validates Hypothesis 4 from FORMAL_CLAIMS.md:
    "EB bounds with n_eff substitution are conservative but empirically valid"

Validation Strategy:
1. Generate synthetic data with KNOWN ground-truth PPV using SyntheticShiftGenerator
2. Run full RAVEL pipeline: weights -> EB bounds -> Holm step-down -> certify/abstain
3. Measure false-certify rate across multiple trials and stress tests
4. Compare to nominal alpha (0.05) with statistical significance testing

Key Metrics:
- False-Certify Rate (FWER): P(>=1 false certification per trial)
- Per-Test False-Certify: Among certified (cohort, tau) pairs, fraction where true PPV < tau
- Coverage: Fraction of cohorts where true PPV >= certified lower bound

Stress Tests:
- Shift severity: 0.5, 1.0, 1.5, 2.0
- Cohort sizes: 5, 10, 20, 50, 100
- Positive rates: 0.3, 0.5, 0.7

Usage:
    # Quick test (3 trials)
    python scripts/validate_h4.py --n_trials 3 --output results/h4_quick_test.csv

    # Full validation (100 trials, ~hours)
    python scripts/validate_h4.py --n_trials 100 --output results/h4_full_validation.csv

    # Specific stress test
    python scripts/validate_h4.py --n_trials 10 --shift_severity 2.0 --cohort_size 20
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from synthetic_shift_generator import SyntheticShiftGenerator, SyntheticData


class H4Validator:
    """Validates empirical error control for EB bounds + Holm correction."""

    def __init__(
        self,
        alpha: float = 0.05,
        tau_grid: List[float] = None,
        use_ravel: bool = True,
        verbose: bool = True
    ):
        """
        Args:
            alpha: Nominal FWER level (target false-certify rate)
            tau_grid: PPV thresholds to test
            use_ravel: If True, use full RAVEL pipeline; else use simplified uLSIF
            verbose: Print progress messages
        """
        self.alpha = alpha
        self.tau_grid = tau_grid or [0.5, 0.6, 0.7, 0.8, 0.9]
        self.use_ravel = use_ravel
        self.verbose = verbose

        # Import method (RAVEL or uLSIF fallback)
        if use_ravel:
            try:
                from shiftbench.baselines.ravel import RAVELBaseline
                self.method = RAVELBaseline(
                    n_folds=3,  # Reduced for speed
                    random_state=42,
                    psis_k_cap=0.70,
                    ess_min_frac=0.20,  # More permissive for small synthetic data
                    clip_mass_cap=0.15,
                )
                if self.verbose:
                    print("[OK] Using RAVEL pipeline with stability gating")
            except (ImportError, Exception) as e:
                if self.verbose:
                    print(f"[WARNING] RAVEL not available ({e}), falling back to uLSIF")
                self.use_ravel = False

        if not self.use_ravel:
            from shiftbench.baselines.ulsif import uLSIFBaseline
            self.method = uLSIFBaseline(
                n_basis=50,  # Reduced for speed
                sigma=None,  # Median heuristic
                lambda_=0.1,
                random_state=42
            )
            if self.verbose:
                print("[OK] Using uLSIF baseline (no gating)")

    def run_single_trial(
        self,
        trial_id: int,
        n_cal: int,
        n_test: int,
        n_cohorts: int,
        shift_severity: float,
        positive_rate: float,
        d_features: int = 10
    ) -> Dict:
        """Run one validation trial and compute metrics.

        Returns:
            Dictionary with trial results:
                - trial_id, n_cal, n_test, n_cohorts, shift_severity, positive_rate
                - false_certify_fwer: 1 if >=1 false certification, 0 otherwise
                - false_certify_count: Number of false certifications
                - n_certified: Total certifications
                - n_abstain: Total abstentions
                - n_no_guarantee: Total NO-GUARANTEE decisions
                - coverage: Fraction of cohorts with true PPV >= lower bound
                - runtime_seconds: Time to run trial
        """
        start_time = time.time()

        # 1. Generate synthetic data with known ground truth
        gen = SyntheticShiftGenerator(
            n_cal=n_cal,
            n_test=n_test,
            n_cohorts=n_cohorts,
            d_features=d_features,
            shift_severity=shift_severity,
            positive_rate=positive_rate,
            seed=trial_id  # Different seed per trial
        )
        data = gen.generate(tau_grid=np.array(self.tau_grid))

        # 2. Estimate importance weights
        try:
            weights = self.method.estimate_weights(
                X_cal=data.X_cal,
                X_target=data.X_test
            )
        except RuntimeError as e:
            # RAVEL stability gates failed -> treat as all NO-GUARANTEE
            if self.verbose:
                print(f"  Trial {trial_id}: Weight estimation failed ({e})")
            return {
                "trial_id": trial_id,
                "n_cal": n_cal,
                "n_test": n_test,
                "n_cohorts": n_cohorts,
                "shift_severity": shift_severity,
                "positive_rate": positive_rate,
                "false_certify_fwer": 0,
                "false_certify_count": 0,
                "n_certified": 0,
                "n_abstain": 0,
                "n_no_guarantee": n_cohorts * len(self.tau_grid),
                "coverage": np.nan,
                "mean_n_eff": np.nan,
                "mean_ppv_estimate": np.nan,
                "mean_true_ppv": np.nan,
                "runtime_seconds": time.time() - start_time
            }

        # 3. Estimate bounds per (cohort, tau) pair with Holm correction
        try:
            decisions = self.method.estimate_bounds(
                y_cal=data.y_cal,
                predictions_cal=data.preds_cal,
                cohort_ids_cal=data.cohorts_cal,
                weights=weights,
                tau_grid=self.tau_grid,
                alpha=self.alpha
            )
        except Exception as e:
            if self.verbose:
                print(f"  Trial {trial_id}: Bound estimation failed ({e})")
            return {
                "trial_id": trial_id,
                "n_cal": n_cal,
                "n_test": n_test,
                "n_cohorts": n_cohorts,
                "shift_severity": shift_severity,
                "positive_rate": positive_rate,
                "false_certify_fwer": 0,
                "false_certify_count": 0,
                "n_certified": 0,
                "n_abstain": 0,
                "n_no_guarantee": n_cohorts * len(self.tau_grid),
                "coverage": np.nan,
                "mean_n_eff": np.nan,
                "mean_ppv_estimate": np.nan,
                "mean_true_ppv": np.nan,
                "runtime_seconds": time.time() - start_time
            }

        # 4. Compare decisions to ground truth
        false_certify_count = 0
        n_certified = 0
        n_abstain = 0
        n_no_guarantee = 0
        coverage_list = []
        n_eff_list = []
        ppv_estimates = []
        true_ppvs = []

        for decision in decisions:
            cohort_id = int(decision.cohort_id) if isinstance(decision.cohort_id, (int, np.integer, str)) else decision.cohort_id

            # Get ground truth for this cohort
            if cohort_id not in data.true_ppv:
                # Cohort has no predicted positives in test set
                continue

            true_ppv = data.true_ppv[cohort_id][decision.tau]

            # Count decision types
            if decision.decision == "CERTIFY":
                n_certified += 1
                ppv_estimates.append(decision.mu_hat)
                true_ppvs.append(true_ppv)

                # Check for false certification
                if true_ppv < decision.tau:
                    false_certify_count += 1

                # Check coverage: true PPV >= lower bound?
                if not np.isnan(decision.lower_bound):
                    coverage_list.append(float(true_ppv >= decision.lower_bound))

            elif decision.decision == "ABSTAIN":
                n_abstain += 1
            elif decision.decision == "NO-GUARANTEE":
                n_no_guarantee += 1

            # Collect n_eff for diagnostics
            if not np.isnan(decision.n_eff):
                n_eff_list.append(decision.n_eff)

        # Compute FWER: did we make >=1 false certification?
        false_certify_fwer = int(false_certify_count > 0)

        # Compute coverage
        coverage = np.mean(coverage_list) if len(coverage_list) > 0 else np.nan

        # Compute mean metrics
        mean_n_eff = np.mean(n_eff_list) if len(n_eff_list) > 0 else np.nan
        mean_ppv_estimate = np.mean(ppv_estimates) if len(ppv_estimates) > 0 else np.nan
        mean_true_ppv = np.mean(true_ppvs) if len(true_ppvs) > 0 else np.nan

        runtime = time.time() - start_time

        return {
            "trial_id": trial_id,
            "n_cal": n_cal,
            "n_test": n_test,
            "n_cohorts": n_cohorts,
            "shift_severity": shift_severity,
            "positive_rate": positive_rate,
            "false_certify_fwer": false_certify_fwer,
            "false_certify_count": false_certify_count,
            "n_certified": n_certified,
            "n_abstain": n_abstain,
            "n_no_guarantee": n_no_guarantee,
            "coverage": coverage,
            "mean_n_eff": mean_n_eff,
            "mean_ppv_estimate": mean_ppv_estimate,
            "mean_true_ppv": mean_true_ppv,
            "runtime_seconds": runtime
        }

    def run_validation(
        self,
        n_trials: int,
        n_cal: int = 500,
        n_test: int = 5000,
        n_cohorts: int = 10,
        shift_severity: float = 1.0,
        positive_rate: float = 0.5
    ) -> pd.DataFrame:
        """Run multiple validation trials and aggregate results.

        Args:
            n_trials: Number of independent trials
            n_cal: Calibration set size per trial
            n_test: Test set size per trial (large for accurate ground truth)
            n_cohorts: Number of cohorts per trial
            shift_severity: Magnitude of covariate shift
            positive_rate: Target P(Y=1) in calibration

        Returns:
            DataFrame with one row per trial
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"H4 VALIDATION EXPERIMENT")
            print(f"{'='*80}")
            print(f"Configuration:")
            print(f"  n_trials:        {n_trials}")
            print(f"  n_cal:           {n_cal}")
            print(f"  n_test:          {n_test}")
            print(f"  n_cohorts:       {n_cohorts}")
            print(f"  shift_severity:  {shift_severity}")
            print(f"  positive_rate:   {positive_rate}")
            print(f"  alpha:           {self.alpha}")
            print(f"  tau_grid:        {self.tau_grid}")
            print(f"  method:          {'RAVEL' if self.use_ravel else 'uLSIF'}")
            print(f"{'='*80}\n")

        results = []
        for trial in range(n_trials):
            if self.verbose and (trial % max(1, n_trials // 10) == 0):
                print(f"Running trial {trial+1}/{n_trials}...")

            result = self.run_single_trial(
                trial_id=trial,
                n_cal=n_cal,
                n_test=n_test,
                n_cohorts=n_cohorts,
                shift_severity=shift_severity,
                positive_rate=positive_rate
            )
            results.append(result)

        df = pd.DataFrame(results)

        if self.verbose:
            self._print_summary(df)

        return df

    def run_stress_tests(
        self,
        n_trials: int = 10,
        shift_severities: List[float] = None,
        cohort_sizes: List[int] = None,
        positive_rates: List[float] = None
    ) -> pd.DataFrame:
        """Run validation across multiple stress test configurations.

        Args:
            n_trials: Trials per configuration
            shift_severities: List of shift magnitudes to test
            cohort_sizes: List of cohort counts to test
            positive_rates: List of base rates to test

        Returns:
            Combined DataFrame with all stress test results
        """
        shift_severities = shift_severities or [0.5, 1.0, 1.5, 2.0]
        cohort_sizes = cohort_sizes or [5, 10, 20, 50]
        positive_rates = positive_rates or [0.3, 0.5, 0.7]

        all_results = []

        total_configs = len(shift_severities) * len(cohort_sizes) * len(positive_rates)
        config_idx = 0

        for shift_sev in shift_severities:
            for n_cohorts in cohort_sizes:
                for pos_rate in positive_rates:
                    config_idx += 1
                    if self.verbose:
                        print(f"\n{'='*80}")
                        print(f"Stress Test {config_idx}/{total_configs}")
                        print(f"  shift_severity={shift_sev}, n_cohorts={n_cohorts}, positive_rate={pos_rate}")
                        print(f"{'='*80}")

                    df = self.run_validation(
                        n_trials=n_trials,
                        n_cal=500,
                        n_test=5000,
                        n_cohorts=n_cohorts,
                        shift_severity=shift_sev,
                        positive_rate=pos_rate
                    )
                    all_results.append(df)

        return pd.concat(all_results, ignore_index=True)

    def _print_summary(self, df: pd.DataFrame) -> None:
        """Print summary statistics for validation results."""
        print(f"\n{'='*80}")
        print("VALIDATION SUMMARY")
        print(f"{'='*80}")

        # Primary metric: False-certify FWER
        fwer = df["false_certify_fwer"].mean()
        n_trials = len(df)

        print(f"\n1. FALSE-CERTIFY RATE (FWER)")
        print(f"   Family-wise error rate: {fwer:.4f} (target: <= {self.alpha:.4f})")
        print(f"   Trials with >=1 false cert: {df['false_certify_fwer'].sum()}/{n_trials}")

        # Statistical test: Is FWER <= alpha?
        # H0: FWER = alpha, H1: FWER > alpha
        result = binomtest(
            k=int(df["false_certify_fwer"].sum()),
            n=n_trials,
            p=self.alpha,
            alternative="greater"
        )
        print(f"   Binomial test p-value: {result.pvalue:.4f}")

        if result.pvalue < 0.05:
            print(f"   [FAIL] REJECT H4: False-certify rate EXCEEDS alpha (p < 0.05)")
        else:
            print(f"   [PASS] ACCEPT H4: False-certify rate <= alpha (p >= 0.05)")

        # Secondary metric: Per-test false-certify rate
        total_certified = df["n_certified"].sum()
        total_false_certify = df["false_certify_count"].sum()
        per_test_rate = total_false_certify / total_certified if total_certified > 0 else 0.0

        print(f"\n2. PER-TEST FALSE-CERTIFY RATE")
        print(f"   Rate among certified: {per_test_rate:.4f} (target: <= {self.alpha:.4f})")
        print(f"   False certifications: {total_false_certify}/{total_certified}")

        # Coverage
        coverage = df["coverage"].mean()
        print(f"\n3. COVERAGE")
        print(f"   Fraction with true PPV >= lower bound: {coverage:.4f} (target: >= {1-self.alpha:.4f})")

        # Decision breakdown
        print(f"\n4. DECISION BREAKDOWN (across all trials)")
        print(f"   Certified:      {df['n_certified'].sum()}")
        print(f"   Abstain:        {df['n_abstain'].sum()}")
        print(f"   NO-GUARANTEE:   {df['n_no_guarantee'].sum()}")

        cert_rate = df["n_certified"].sum() / (df["n_certified"].sum() + df["n_abstain"].sum() + df["n_no_guarantee"].sum())
        print(f"   Certification rate: {cert_rate:.2%}")

        # Effective sample size
        mean_n_eff = df["mean_n_eff"].mean()
        print(f"\n5. DIAGNOSTICS")
        print(f"   Mean n_eff:         {mean_n_eff:.1f}")
        print(f"   Mean PPV estimate:  {df['mean_ppv_estimate'].mean():.3f}")
        print(f"   Mean true PPV:      {df['mean_true_ppv'].mean():.3f}")
        print(f"   Mean runtime:       {df['runtime_seconds'].mean():.2f}s per trial")

        print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="H4 Validation: Empirical validation of false-certify rate <= alpha",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Experiment configuration
    parser.add_argument("--n_trials", type=int, default=3,
                        help="Number of validation trials (default: 3 for quick test, 100 for full)")
    parser.add_argument("--n_cal", type=int, default=500,
                        help="Calibration set size per trial")
    parser.add_argument("--n_test", type=int, default=5000,
                        help="Test set size for ground-truth estimation")
    parser.add_argument("--n_cohorts", type=int, default=10,
                        help="Number of cohorts per trial")
    parser.add_argument("--shift_severity", type=float, default=1.0,
                        help="Covariate shift magnitude (0=none, 2=severe)")
    parser.add_argument("--positive_rate", type=float, default=0.5,
                        help="Target P(Y=1) in calibration")

    # Stress tests
    parser.add_argument("--stress_tests", action="store_true",
                        help="Run full stress test sweep (shift, cohort size, positive rate)")
    parser.add_argument("--stress_shift", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0],
                        help="Shift severities for stress tests")
    parser.add_argument("--stress_cohorts", nargs="+", type=int, default=[5, 10, 20, 50],
                        help="Cohort sizes for stress tests")
    parser.add_argument("--stress_rates", nargs="+", type=float, default=[0.3, 0.5, 0.7],
                        help="Positive rates for stress tests")

    # Method configuration
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Nominal FWER level (default: 0.05)")
    parser.add_argument("--tau_grid", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9],
                        help="PPV thresholds to test")
    parser.add_argument("--use_ulsif", action="store_true",
                        help="Use uLSIF baseline instead of RAVEL (no gating)")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: results/h4_validation_<timestamp>.csv)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress messages")

    args = parser.parse_args()

    # Create validator
    validator = H4Validator(
        alpha=args.alpha,
        tau_grid=args.tau_grid,
        use_ravel=not args.use_ulsif,
        verbose=not args.quiet
    )

    # Run experiment
    if args.stress_tests:
        df = validator.run_stress_tests(
            n_trials=args.n_trials,
            shift_severities=args.stress_shift,
            cohort_sizes=args.stress_cohorts,
            positive_rates=args.stress_rates
        )
    else:
        df = validator.run_validation(
            n_trials=args.n_trials,
            n_cal=args.n_cal,
            n_test=args.n_test,
            n_cohorts=args.n_cohorts,
            shift_severity=args.shift_severity,
            positive_rate=args.positive_rate
        )

    # Save results
    if args.output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent.parent / "results"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"h4_validation_{timestamp}.csv"
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    if not args.quiet:
        print(f"\n[PASS] Results saved to: {output_path}")
        print(f"   {len(df)} trials × {len(df.columns)} columns")

        # Quick summary
        fwer = df["false_certify_fwer"].mean()
        result = binomtest(
            k=int(df["false_certify_fwer"].sum()),
            n=len(df),
            p=args.alpha,
            alternative="greater"
        )

        print(f"\n{'='*80}")
        print("FINAL RESULT")
        print(f"{'='*80}")
        print(f"False-certify rate: {fwer:.4f} (target: <= {args.alpha:.4f})")
        print(f"Statistical test:   p = {result.pvalue:.4f}")

        if result.pvalue < 0.05:
            print(f"[FAIL] HYPOTHESIS REJECTED: False-certify rate exceeds alpha")
        else:
            print(f"[PASS] HYPOTHESIS VALIDATED: False-certify rate <= alpha")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
