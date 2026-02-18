"""KMM: Kernel Mean Matching.

Direct density ratio estimation via Maximum Mean Discrepancy (MMD) minimization.
Solves a Quadratic Programming (QP) problem with box and sum constraints.

Key advantages:
- Minimizes MMD between weighted calibration and target distributions
- Direct optimization of distribution matching (not a proxy like KL or L2)
- Bounded weights (0 <= w_i <= B) prevent extreme values
- Well-studied theoretical properties (bounded generalization error)

Key disadvantages:
- Requires QP solver (slower than uLSIF's closed-form solution)
- Sensitive to kernel bandwidth selection
- May be unstable if calibration and target distributions differ greatly

References:
    Huang et al. 2007. "Correcting Sample Selection Bias by Unlabeled Data"
    Neural Information Processing Systems (NIPS) 2007.

    Gretton et al. 2009. "Covariate Shift by Kernel Mean Matching"
    In "Dataset Shift in Machine Learning", MIT Press.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import warnings

import numpy as np
from scipy.spatial.distance import cdist

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

from shiftbench.baselines.base import (
    BaselineMethod,
    CohortDecision,
    MethodMetadata,
)


class KMMBaseline(BaselineMethod):
    """KMM density ratio estimator + Empirical-Bernstein bounds.

    Estimates importance weights w(x) = p_target(x) / p_cal(x) using:
    1. Gaussian kernel basis functions centered on all calibration samples
    2. QP optimization to minimize MMD: min ||K_cal @ w - K_target @ 1||² + λ||w||²
    3. Subject to: 0 <= w_i <= B, sum(w) = n_cal
    4. Self-normalization to ensure mean(w) = 1

    Does NOT have stability gating, so never returns NO-GUARANTEE.
    """

    def __init__(
        self,
        sigma: Optional[float] = None,
        lambda_: float = 0.1,
        B: float = 1000.0,
        random_state: int = 42,
        solver: str = "auto",  # "cvxpy", "scipy", or "auto"
        **kwargs,
    ):
        """Initialize KMM with hyperparameters.

        Args:
            sigma: Kernel bandwidth (std dev). If None, use median heuristic.
            lambda_: Ridge regularization parameter (higher = smoother weights)
            B: Box constraint upper bound (weights bounded in [0, B])
            random_state: Random seed for reproducibility
            solver: QP solver to use ("cvxpy", "scipy", or "auto")
            **kwargs: Additional hyperparameters
        """
        super().__init__(
            sigma=sigma,
            lambda_=lambda_,
            B=B,
            random_state=random_state,
            solver=solver,
            **kwargs,
        )
        self._fitted_params = None  # Store weights and kernel info for diagnostics

    def estimate_weights(
        self,
        X_cal: np.ndarray,
        X_target: np.ndarray,
        domain_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Estimate importance weights using KMM.

        Args:
            X_cal: Calibration features (n_cal, n_features)
            X_target: Target features (n_target, n_features)
            domain_labels: Not used (KMM is direct method)

        Returns:
            weights: Importance weights for calibration samples (n_cal,)
        """
        n_cal = len(X_cal)
        n_target = len(X_target)

        # Step 1: Determine kernel bandwidth (sigma)
        rng = np.random.RandomState(self.hyperparameters["random_state"])
        sigma = self.hyperparameters["sigma"]
        if sigma is None:
            # Median heuristic: sigma = median of pairwise distances
            sample_size = min(1000, n_cal)
            sample_indices = rng.choice(n_cal, size=sample_size, replace=False)
            dists = cdist(X_cal[sample_indices], X_cal[sample_indices], metric="euclidean")
            sigma = np.median(dists[dists > 0])
            if sigma == 0:
                sigma = 1.0  # Fallback

        # Step 2: Compute kernel matrices
        # K_cal[i, j] = k(x_cal_i, x_cal_j) - kernel matrix on calibration set
        # K_cross[i, j] = k(x_cal_i, x_target_j) - cross-kernel between cal and target
        K_cal = self._gaussian_kernel(X_cal, X_cal, sigma)
        K_cross = self._gaussian_kernel(X_cal, X_target, sigma)

        # Step 3: Formulate KMM QP problem
        # Minimize: (1/2) * w^T K_cal w - kappa^T w + (lambda/2) * ||w||^2
        # Subject to: 0 <= w_i <= B, sum(w) = n_cal
        #
        # where kappa = (n_cal/n_target) * sum_j K_cross[:, j]
        #       = (n_cal/n_target) * K_cross @ 1

        kappa = (n_cal / n_target) * K_cross.sum(axis=1)
        lambda_ = self.hyperparameters["lambda_"]
        B = self.hyperparameters["B"]

        # Solve QP
        solver = self.hyperparameters["solver"]
        if solver == "auto":
            # Prefer cvxpy if available (more robust), else scipy
            solver = "cvxpy" if HAS_CVXPY else "scipy"

        if solver == "cvxpy":
            if not HAS_CVXPY:
                raise ImportError(
                    "cvxpy is not installed. Install with: pip install cvxpy"
                )
            weights_unnormalized, success, solve_time = self._solve_qp_cvxpy(
                K_cal, kappa, lambda_, B, n_cal
            )
        elif solver == "scipy":
            weights_unnormalized, success, solve_time = self._solve_qp_scipy(
                K_cal, kappa, lambda_, B, n_cal
            )
        else:
            raise ValueError(f"Unknown solver: {solver}. Use 'cvxpy', 'scipy', or 'auto'.")

        if not success:
            warnings.warn(
                "KMM optimization did not converge. Using best solution found. "
                "Consider increasing lambda_ or reducing B."
            )

        # Step 4: Ensure positivity and self-normalize
        weights_unnormalized = np.maximum(weights_unnormalized, 1e-8)
        weights = weights_unnormalized / weights_unnormalized.mean()

        # Store for diagnostics
        self._fitted_params = {
            "weights_unnormalized": weights_unnormalized,
            "sigma": sigma,
            "lambda": lambda_,
            "B": B,
            "n_cal": n_cal,
            "n_target": n_target,
            "optimization_success": success,
            "solve_time": solve_time,
        }

        return weights

    def _solve_qp_cvxpy(
        self,
        K_cal: np.ndarray,
        kappa: np.ndarray,
        lambda_: float,
        B: float,
        n_cal: int,
    ) -> tuple[np.ndarray, bool, float]:
        """Solve KMM QP using cvxpy.

        Minimize: (1/2) * w^T (K_cal + lambda*I) w - kappa^T w
        Subject to: 0 <= w_i <= B, sum(w) = n_cal

        Returns:
            weights: Solution (n_cal,)
            success: True if optimization converged
            solve_time: Time taken to solve (seconds)
        """
        import time

        start = time.time()

        # Decision variable
        w = cp.Variable(n_cal)

        # Objective: (1/2) * w^T (K_cal + lambda*I) w - kappa^T w
        P = K_cal + lambda_ * np.eye(n_cal)
        objective = cp.Minimize(0.5 * cp.quad_form(w, P) - kappa @ w)

        # Constraints
        constraints = [
            w >= 0,
            w <= B,
            cp.sum(w) == n_cal,
        ]

        # Solve
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.ECOS)
            success = problem.status in ["optimal", "optimal_inaccurate"]
            weights = w.value
            if weights is None:
                # Solver failed, return uniform weights
                weights = np.ones(n_cal)
                success = False
        except Exception as e:
            warnings.warn(f"cvxpy solver failed: {e}. Returning uniform weights.")
            weights = np.ones(n_cal)
            success = False

        solve_time = time.time() - start
        return weights, success, solve_time

    def _solve_qp_scipy(
        self,
        K_cal: np.ndarray,
        kappa: np.ndarray,
        lambda_: float,
        B: float,
        n_cal: int,
    ) -> tuple[np.ndarray, bool, float]:
        """Solve KMM QP using scipy.optimize.minimize with SLSQP.

        Minimize: (1/2) * w^T (K_cal + lambda*I) w - kappa^T w
        Subject to: 0 <= w_i <= B, sum(w) = n_cal

        Returns:
            weights: Solution (n_cal,)
            success: True if optimization converged
            solve_time: Time taken to solve (seconds)
        """
        from scipy.optimize import minimize
        import time

        start = time.time()

        # Objective function
        P = K_cal + lambda_ * np.eye(n_cal)

        def objective(w):
            return 0.5 * w @ P @ w - kappa @ w

        def objective_grad(w):
            return P @ w - kappa

        # Initial guess: uniform weights satisfying sum constraint
        w_init = np.ones(n_cal)

        # Bounds: 0 <= w_i <= B
        bounds = [(0, B) for _ in range(n_cal)]

        # Constraints: sum(w) = n_cal
        constraints = {
            'type': 'eq',
            'fun': lambda w: np.sum(w) - n_cal,
            'jac': lambda w: np.ones(n_cal),
        }

        # Optimize
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            result = minimize(
                objective,
                w_init,
                method='SLSQP',
                jac=objective_grad,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9},
            )

        success = result.success
        weights = result.x

        solve_time = time.time() - start
        return weights, success, solve_time

    def estimate_bounds(
        self,
        y_cal: np.ndarray,
        predictions_cal: np.ndarray,
        cohort_ids_cal: np.ndarray,
        weights: np.ndarray,
        tau_grid: List[float],
        alpha: float = 0.05,
    ) -> List[CohortDecision]:
        """Estimate PPV lower bounds using Empirical-Bernstein.

        NOTE: KMM does NOT have stability gating, so never returns NO-GUARANTEE.
        This may produce unreliable bounds when shift is severe.
        """
        self.validate_inputs(y_cal, predictions_cal, cohort_ids_cal, weights)

        # Import EB bound utilities (assume RAVEL is available)
        from ravel.bounds.empirical_bernstein import eb_lower_bound
        from ravel.bounds.p_value import eb_p_value
        from ravel.bounds.weighted_stats import weighted_stats_01

        all_decisions = []
        unique_cohorts = np.unique(cohort_ids_cal)

        for tau in tau_grid:
            for cohort_id in unique_cohorts:
                # Filter to cohort predicted positives
                cohort_mask = (cohort_ids_cal == cohort_id)
                pos_mask = cohort_mask & (predictions_cal == 1)

                y_cohort = y_cal[pos_mask]
                w_cohort = weights[pos_mask]

                if len(y_cohort) < 5:
                    # Insufficient data
                    decision = CohortDecision(
                        cohort_id=cohort_id,
                        tau=tau,
                        decision="ABSTAIN",
                        mu_hat=np.nan,
                        var_hat=np.nan,
                        n_eff=0,
                        lower_bound=np.nan,
                        p_value=1.0,
                        diagnostics={"reason": "insufficient_positives"},
                    )
                else:
                    # Compute weighted stats
                    stats = weighted_stats_01(y_cohort, w_cohort)

                    # Compute EB lower bound
                    lb = eb_lower_bound(stats.mu, stats.var, stats.n_eff, alpha)

                    # Compute p-value testing H0: PPV < tau
                    pval = eb_p_value(stats.mu, stats.var, stats.n_eff, tau)

                    # Decision: CERTIFY if lower bound >= tau
                    dec = "CERTIFY" if lb >= tau else "ABSTAIN"

                    decision = CohortDecision(
                        cohort_id=cohort_id,
                        tau=tau,
                        decision=dec,
                        mu_hat=stats.mu,
                        var_hat=stats.var,
                        n_eff=stats.n_eff,
                        lower_bound=lb,
                        p_value=pval,
                        diagnostics=self.get_diagnostics(),
                    )

                all_decisions.append(decision)

        return all_decisions

    def get_metadata(self) -> MethodMetadata:
        """Return KMM metadata."""
        return MethodMetadata(
            name="kmm",
            version="1.0.0",
            description=(
                "Kernel Mean Matching. "
                "Direct density ratio estimation via MMD minimization and QP optimization."
            ),
            paper_title="Correcting Sample Selection Bias by Unlabeled Data",
            paper_url="https://papers.nips.cc/paper/2007/hash/8d3bba7425e7c98c50f52ca1b52d3735-Abstract.html",
            code_url="https://github.com/anthropics/shift-bench",
            hyperparameters=self.hyperparameters,
            supports_abstention=False,  # No stability gating
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return KMM-specific diagnostics."""
        if self._fitted_params is None:
            return {}

        w = self._fitted_params["weights_unnormalized"]
        return {
            "method": "kmm",
            "sigma": self._fitted_params["sigma"],
            "lambda": self._fitted_params["lambda"],
            "B": self._fitted_params["B"],
            "n_cal": self._fitted_params["n_cal"],
            "n_target": self._fitted_params["n_target"],
            "weights_min": float(w.min()),
            "weights_max": float(w.max()),
            "weights_mean": float(w.mean()),
            "weights_std": float(w.std()),
            "weights_clipped_fraction": float((w >= self._fitted_params["B"] - 1e-6).mean()),
            "optimization_success": self._fitted_params["optimization_success"],
            "solve_time": self._fitted_params["solve_time"],
        }

    @staticmethod
    def _gaussian_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
        """Compute Gaussian kernel matrix K[i,j] = exp(-||x_i - y_j||^2 / (2*sigma^2)).

        Args:
            X: Data points (n, d)
            Y: Data points (m, d)
            sigma: Kernel bandwidth

        Returns:
            K: Kernel matrix (n, m)
        """
        dists_sq = cdist(X, Y, metric="sqeuclidean")
        return np.exp(-dists_sq / (2 * sigma ** 2))


def create_kmm_baseline(**kwargs) -> KMMBaseline:
    """Create a KMM baseline with default or custom hyperparameters.

    Examples:
        >>> # Default KMM (cvxpy if available, else scipy)
        >>> kmm = create_kmm_baseline()

        >>> # Force scipy solver
        >>> kmm_scipy = create_kmm_baseline(solver="scipy")

        >>> # Custom hyperparameters
        >>> kmm_custom = create_kmm_baseline(sigma=1.0, lambda_=0.01, B=500)

        >>> # Tighter box constraints (more conservative weights)
        >>> kmm_tight = create_kmm_baseline(B=100)
    """
    return KMMBaseline(**kwargs)
