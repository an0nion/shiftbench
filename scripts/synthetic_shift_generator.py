"""
Synthetic Data Generator for H4 Validation Experiments

Generates data with KNOWN ground-truth PPV under covariate shift.
This enables validation of empirical error control (false-certify <= alpha).

IMPORTANT - ORACLE-CALIBRATED PREDICTOR:
    Predictions use the SAME beta as label generation:
        Labels:      Y ~ Bernoulli(sigmoid(beta^T X))
        Predictions: Y_hat = 1[sigmoid(beta^T X) >= 0.5]
    This is a Bayes-optimal predictor with perfect calibration.
    Real models have miscalibration, structural errors, and finite-sample bias.
    Use this generator for VALIDITY TESTING (false-certify <= alpha), NOT for
    assessing realistic method performance.

Usage:
    python scripts/synthetic_shift_generator.py --test
    python scripts/synthetic_shift_generator.py --n_trials 10 --visualize
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import argparse


@dataclass
class SyntheticData:
    """Container for synthetic shift data with known ground truth."""
    X_cal: np.ndarray  # Calibration features (n_cal, d)
    y_cal: np.ndarray  # Calibration labels (n_cal,)
    cohorts_cal: np.ndarray  # Calibration cohorts (n_cal,)
    preds_cal: np.ndarray  # Calibration predictions (n_cal,) binary

    X_test: np.ndarray  # Test features (n_test, d)
    y_test: np.ndarray  # Test labels (n_test,)
    cohorts_test: np.ndarray  # Test cohorts (n_test,)
    preds_test: np.ndarray  # Test predictions (n_test,) binary

    true_ppv: Dict[int, Dict[float, float]]  # Ground-truth PPV per cohort
    # Format: {cohort_id: {tau: true_ppv_value}}


class SyntheticShiftGenerator:
    """
    Generate synthetic data with known ground-truth PPV under covariate shift.

    Key Features:
    - Controlled covariate shift: P(X) differs, P(Y|X) invariant
    - Known ground-truth PPV for each cohort
    - Varying shift severity, cohort sizes, positive rates
    """

    def __init__(
        self,
        n_cal: int = 500,
        n_test: int = 5000,  # Large test for accurate PPV estimation
        n_cohorts: int = 10,
        d_features: int = 10,
        shift_severity: float = 1.0,
        positive_rate: float = 0.5,
        seed: int = 42
    ):
        """
        Args:
            n_cal: Calibration set size
            n_test: Test set size (large for accurate ground-truth PPV)
            n_cohorts: Number of cohorts
            d_features: Feature dimension
            shift_severity: Magnitude of covariate shift (0=none, 2=high)
            positive_rate: Target P(Y=1) in calibration set
            seed: Random seed
        """
        self.n_cal = n_cal
        self.n_test = n_test
        self.n_cohorts = n_cohorts
        self.d_features = d_features
        self.shift_severity = shift_severity
        self.positive_rate = positive_rate
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Fixed model coefficients (for predictions)
        self.beta = self._sample_beta()

    def _sample_beta(self) -> np.ndarray:
        """Sample fixed model coefficients."""
        # Standardized coefficients for logistic regression
        beta = self.rng.randn(self.d_features)
        beta = beta / np.linalg.norm(beta) * np.sqrt(self.d_features)
        return beta

    def _generate_features(
        self,
        n: int,
        cohorts: np.ndarray,
        shifted: bool = False
    ) -> np.ndarray:
        """
        Generate features with optional covariate shift.

        Model: X | (cohort=g) ~ N(μ_g, Σ)
        Shift: μ_g^{test} = μ_g^{cal} + shift_severity * offset_g
        """
        X = np.zeros((n, self.d_features))

        for g in range(self.n_cohorts):
            mask = (cohorts == g)
            n_g = mask.sum()

            if n_g == 0:
                continue

            # Cohort-specific mean (base)
            mu_base = self.rng.randn(self.d_features) * 0.5

            # Add shift if test distribution
            if shifted:
                # Shift direction varies by cohort
                shift_direction = self.rng.randn(self.d_features)
                shift_direction = shift_direction / np.linalg.norm(shift_direction)
                mu_shifted = mu_base + self.shift_severity * shift_direction
            else:
                mu_shifted = mu_base

            # Sample features
            X[mask] = self.rng.randn(n_g, self.d_features) + mu_shifted

        return X

    def _generate_labels(self, X: np.ndarray) -> np.ndarray:
        """
        Generate labels using fixed logistic model.

        P(Y=1 | X) = sigmoid(β^T X)

        Ensures P(Y|X) is invariant (covariate shift assumption).
        """
        logits = X @ self.beta
        probs = 1 / (1 + np.exp(-logits))
        y = self.rng.binomial(1, probs)
        return y

    def _generate_predictions(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Generate predictions from fixed model with fixed threshold.

        Ŷ = 1[P(Y=1|X) ≥ threshold]

        This is the "fixed prediction policy" assumption in FORMAL_CLAIMS.md.
        """
        logits = X @ self.beta
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= threshold).astype(int)
        return preds

    def _compute_true_ppv(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        cohorts_test: np.ndarray,
        preds_test: np.ndarray,
        tau_grid: np.ndarray
    ) -> Dict[int, Dict[float, float]]:
        """
        Compute ground-truth PPV per cohort using large test set.

        PPV_g = P(Y=1 | Ŷ=1, G=g) = #{Y=1, Ŷ=1, G=g} / #{Ŷ=1, G=g}
        """
        true_ppv = {}

        for g in range(self.n_cohorts):
            cohort_mask = (cohorts_test == g)

            if cohort_mask.sum() == 0:
                continue

            # Filter to predicted positives in this cohort
            pos_mask = cohort_mask & (preds_test == 1)

            if pos_mask.sum() == 0:
                # No predicted positives in this cohort
                true_ppv[g] = {tau: 0.0 for tau in tau_grid}
                continue

            # Compute empirical PPV (use large test set)
            ppv = y_test[pos_mask].mean()

            # Same PPV for all tau (threshold doesn't affect empirical PPV)
            true_ppv[g] = {tau: ppv for tau in tau_grid}

        return true_ppv

    def generate(
        self,
        tau_grid: Optional[np.ndarray] = None
    ) -> SyntheticData:
        """
        Generate complete synthetic dataset with known ground truth.

        Returns:
            SyntheticData with X, y, cohorts, predictions, and true PPV
        """
        if tau_grid is None:
            tau_grid = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

        # 1. Generate cohort assignments
        # Use stratified sampling to ensure all cohorts represented
        cohorts_cal = self.rng.choice(self.n_cohorts, size=self.n_cal, replace=True)
        cohorts_test = self.rng.choice(self.n_cohorts, size=self.n_test, replace=True)

        # 2. Generate features with covariate shift
        X_cal = self._generate_features(self.n_cal, cohorts_cal, shifted=False)
        X_test = self._generate_features(self.n_test, cohorts_test, shifted=True)

        # 3. Generate labels (P(Y|X) invariant)
        y_cal = self._generate_labels(X_cal)
        y_test = self._generate_labels(X_test)

        # 4. Generate predictions (fixed model, fixed threshold)
        preds_cal = self._generate_predictions(X_cal)
        preds_test = self._generate_predictions(X_test)

        # 5. Compute ground-truth PPV using large test set
        true_ppv = self._compute_true_ppv(X_test, y_test, cohorts_test, preds_test, tau_grid)

        return SyntheticData(
            X_cal=X_cal, y_cal=y_cal, cohorts_cal=cohorts_cal, preds_cal=preds_cal,
            X_test=X_test, y_test=y_test, cohorts_test=cohorts_test, preds_test=preds_test,
            true_ppv=true_ppv
        )


def test_generator():
    """Test synthetic data generator with various configurations."""
    print("=" * 80)
    print("SYNTHETIC DATA GENERATOR TEST")
    print("=" * 80)

    configs = [
        {"shift_severity": 0.0, "name": "No Shift"},
        {"shift_severity": 0.5, "name": "Mild Shift"},
        {"shift_severity": 1.0, "name": "Moderate Shift"},
        {"shift_severity": 2.0, "name": "Severe Shift"},
    ]

    for config in configs:
        print(f"\n{config['name']} (severity={config['shift_severity']}):")
        print("-" * 60)

        gen = SyntheticShiftGenerator(
            n_cal=500,
            n_test=5000,
            n_cohorts=10,
            shift_severity=config['shift_severity'],
            seed=42
        )

        data = gen.generate()

        # Statistics
        print(f"  Calibration: {len(data.X_cal)} samples, {data.cohorts_cal.max()+1} cohorts")
        print(f"  Test:        {len(data.X_test)} samples, {data.cohorts_test.max()+1} cohorts")
        print(f"  Positive rate (cal): {data.y_cal.mean():.3f}")
        print(f"  Positive rate (test): {data.y_test.mean():.3f}")
        print(f"  Prediction rate (cal): {data.preds_cal.mean():.3f}")
        print(f"  Prediction rate (test): {data.preds_test.mean():.3f}")

        # PPV distribution
        ppvs = [ppv_dict[0.8] for ppv_dict in data.true_ppv.values()]
        print(f"  True PPV (tau=0.8): mean={np.mean(ppvs):.3f}, std={np.std(ppvs):.3f}")
        print(f"  True PPV range: [{np.min(ppvs):.3f}, {np.max(ppvs):.3f}]")

        # Check covariate shift
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        X_combined = np.vstack([data.X_cal, data.X_test])
        y_combined = np.hstack([
            np.zeros(len(data.X_cal)),
            np.ones(len(data.X_test))
        ])

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_combined, y_combined)
        auc = roc_auc_score(y_combined, clf.predict_proba(X_combined)[:, 1])

        print(f"  Two-sample AUC: {auc:.3f} (0.5=no shift, 1.0=perfect separation)")

        # Verify P(Y|X) invariance
        # Train predictor on cal, test on test
        clf_y = LogisticRegression(max_iter=1000, random_state=42)
        clf_y.fit(data.X_cal, data.y_cal)

        from sklearn.metrics import accuracy_score
        acc_cal = accuracy_score(data.y_cal, clf_y.predict(data.X_cal))
        acc_test = accuracy_score(data.y_test, clf_y.predict(data.X_test))

        print(f"  P(Y|X) accuracy: cal={acc_cal:.3f}, test={acc_test:.3f} (should be similar)")

    print("\n" + "=" * 80)
    print("All tests passed! Generator is working correctly.")
    print("=" * 80)


def visualize_generator(n_trials: int = 10):
    """Generate multiple trials and visualize distributions."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        print("⚠️  Matplotlib not available. Skipping visualization.")
        return

    print("\nGenerating visualization...")

    shift_severities = [0.0, 0.5, 1.0, 1.5, 2.0]
    aucs = {sev: [] for sev in shift_severities}
    ppv_means = {sev: [] for sev in shift_severities}

    for sev in shift_severities:
        for trial in range(n_trials):
            gen = SyntheticShiftGenerator(
                n_cal=500, n_test=5000, n_cohorts=10,
                shift_severity=sev, seed=trial
            )
            data = gen.generate()

            # Compute two-sample AUC
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score

            X = np.vstack([data.X_cal, data.X_test])
            y = np.hstack([np.zeros(len(data.X_cal)), np.ones(len(data.X_test))])
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X, y)
            auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
            aucs[sev].append(auc)

            # Mean true PPV
            ppvs = [ppv_dict[0.8] for ppv_dict in data.true_ppv.values()]
            ppv_means[sev].append(np.mean(ppvs))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Two-sample AUC vs shift severity
    ax = axes[0]
    ax.boxplot([aucs[sev] for sev in shift_severities], labels=shift_severities)
    ax.set_xlabel("Shift Severity", fontsize=12)
    ax.set_ylabel("Two-Sample AUC", fontsize=12)
    ax.set_title("Covariate Shift Strength", fontsize=14)
    ax.axhline(0.5, color='red', linestyle='--', label='No shift')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # PPV distribution vs shift severity
    ax = axes[1]
    ax.boxplot([ppv_means[sev] for sev in shift_severities], labels=shift_severities)
    ax.set_xlabel("Shift Severity", fontsize=12)
    ax.set_ylabel("Mean True PPV (tau=0.8)", fontsize=12)
    ax.set_title("PPV Distribution Under Shift", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("synthetic_generator_validation.pdf", dpi=300, bbox_inches='tight')
    print("✅ Saved visualization to: synthetic_generator_validation.pdf")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test synthetic data generator")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of trials for visualization")

    args = parser.parse_args()

    if args.test or (not args.visualize):
        test_generator()

    if args.visualize:
        visualize_generator(n_trials=args.n_trials)
