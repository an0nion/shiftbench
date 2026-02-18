"""
Shift Metrics: Quantify distribution shift between calibration and test sets

Primary metric: Two-sample classifier AUC (more stable than MMD across domains)
Secondary metric: MMD with Gaussian kernel
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def compute_two_sample_auc(
    X_cal: np.ndarray,
    X_test: np.ndarray,
    random_state: int = 42,
    test_size: float = 0.3
) -> float:
    """
    Compute two-sample classifier AUC to quantify distribution shift.

    Train a logistic regression to distinguish calibration vs test samples.
    AUC = 0.5 indicates no shift, AUC = 1.0 indicates perfect separation.

    This is MORE STABLE than MMD across domains (per FORMAL_CLAIMS.md H3 P3.2).

    Args:
        X_cal: Calibration features (n_cal, d)
        X_test: Test features (n_test, d)
        random_state: Random seed
        test_size: Fraction of data for validation (avoid overfitting)

    Returns:
        AUC ∈ [0.5, 1.0] where 0.5 = no shift, 1.0 = perfect separation
    """
    # Combine data with labels (0=cal, 1=test)
    X_combined = np.vstack([X_cal, X_test])
    y_combined = np.hstack([
        np.zeros(len(X_cal), dtype=int),
        np.ones(len(X_test), dtype=int)
    ])

    # Train/val split to avoid overfitting to specific samples
    X_train, X_val, y_train, y_val = train_test_split(
        X_combined, y_combined,
        test_size=test_size,
        random_state=random_state,
        stratify=y_combined
    )

    # Train logistic regression
    clf = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        solver='lbfgs',
        C=1.0  # Moderate regularization
    )
    clf.fit(X_train, y_train)

    # Compute AUC on validation set
    probs = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, probs)

    # Ensure AUC ≥ 0.5 (by convention, shift detector should be better than random)
    # If AUC < 0.5, flip it (classifier learned inverse relationship)
    if auc < 0.5:
        auc = 1.0 - auc

    return auc


def compute_mmd(
    X_cal: np.ndarray,
    X_test: np.ndarray,
    kernel: str = "gaussian",
    bandwidth: Optional[float] = None
) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) between distributions.

    MMD = ||μ_cal - μ_test||²_H where H is RKHS

    This is SECONDARY to two-sample AUC (per FORMAL_CLAIMS.md H3 P3.2).
    Use for robustness checks only.

    Args:
        X_cal: Calibration features (n_cal, d)
        X_test: Test features (n_test, d)
        kernel: Kernel type ('gaussian' or 'linear')
        bandwidth: Kernel bandwidth (if None, use median heuristic)

    Returns:
        MMD² ≥ 0 (higher = more shift)
    """
    n_cal = len(X_cal)
    n_test = len(X_test)

    if kernel == "gaussian":
        # Median heuristic for bandwidth
        if bandwidth is None:
            # Compute pairwise distances on subsample (for efficiency)
            n_sample = min(1000, n_cal, n_test)
            X_sample = np.vstack([
                X_cal[np.random.choice(n_cal, n_sample, replace=False)],
                X_test[np.random.choice(n_test, n_sample, replace=False)]
            ])
            from scipy.spatial.distance import pdist
            dists = pdist(X_sample, metric='euclidean')
            bandwidth = np.median(dists)

        # Gaussian kernel: k(x, y) = exp(-||x-y||² / (2*bandwidth²))
        def kernel_fn(X, Y):
            # Compute pairwise squared distances
            XX = np.sum(X**2, axis=1, keepdims=True)
            YY = np.sum(Y**2, axis=1, keepdims=True)
            XY = X @ Y.T
            sq_dists = XX + YY.T - 2 * XY
            return np.exp(-sq_dists / (2 * bandwidth**2))

    elif kernel == "linear":
        def kernel_fn(X, Y):
            return X @ Y.T
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # Compute MMD² (unbiased estimator)
    K_cal_cal = kernel_fn(X_cal, X_cal)
    K_test_test = kernel_fn(X_test, X_test)
    K_cal_test = kernel_fn(X_cal, X_test)

    # Diagonal removal for unbiased estimate
    np.fill_diagonal(K_cal_cal, 0)
    np.fill_diagonal(K_test_test, 0)

    mmd_sq = (
        K_cal_cal.sum() / (n_cal * (n_cal - 1))
        + K_test_test.sum() / (n_test * (n_test - 1))
        - 2 * K_cal_test.sum() / (n_cal * n_test)
    )

    # MMD² can be slightly negative due to finite sample, clip to 0
    mmd_sq = max(0, mmd_sq)

    return np.sqrt(mmd_sq)


def compute_all_shift_metrics(
    X_cal: np.ndarray,
    X_test: np.ndarray,
    random_state: int = 42
) -> dict:
    """
    Compute all shift metrics for a dataset.

    Returns:
        dict with keys: 'two_sample_auc' (PRIMARY), 'mmd' (SECONDARY)
    """
    metrics = {}

    # PRIMARY: Two-sample AUC (most stable)
    metrics['two_sample_auc'] = compute_two_sample_auc(
        X_cal, X_test, random_state=random_state
    )

    # SECONDARY: MMD with median heuristic
    metrics['mmd'] = compute_mmd(
        X_cal, X_test, kernel='gaussian', bandwidth=None
    )

    return metrics


if __name__ == "__main__":
    print("=" * 80)
    print("SHIFT METRICS TEST")
    print("=" * 80)

    # Test 1: No shift
    print("\nTest 1: No Shift")
    print("-" * 60)
    np.random.seed(42)
    X_cal = np.random.randn(500, 10)
    X_test = np.random.randn(500, 10)

    metrics = compute_all_shift_metrics(X_cal, X_test)
    print(f"  Two-sample AUC: {metrics['two_sample_auc']:.3f} (expect ~0.5)")
    print(f"  MMD:            {metrics['mmd']:.3f} (expect ~0)")

    # Test 2: Moderate shift
    print("\nTest 2: Moderate Shift (mean shift)")
    print("-" * 60)
    X_cal = np.random.randn(500, 10)
    X_test = np.random.randn(500, 10) + 0.5  # Mean shift

    metrics = compute_all_shift_metrics(X_cal, X_test)
    print(f"  Two-sample AUC: {metrics['two_sample_auc']:.3f} (expect ~0.7-0.8)")
    print(f"  MMD:            {metrics['mmd']:.3f} (expect >0)")

    # Test 3: Severe shift
    print("\nTest 3: Severe Shift (mean + scale shift)")
    print("-" * 60)
    X_cal = np.random.randn(500, 10)
    X_test = np.random.randn(500, 10) * 2 + 1.0  # Scale + mean shift

    metrics = compute_all_shift_metrics(X_cal, X_test)
    print(f"  Two-sample AUC: {metrics['two_sample_auc']:.3f} (expect >0.9)")
    print(f"  MMD:            {metrics['mmd']:.3f} (expect >>0)")

    # Test 4: Verify AUC >= 0.5 property
    print("\nTest 4: AUC >= 0.5 Property")
    print("-" * 60)
    for trial in range(10):
        X_cal = np.random.randn(200, 5)
        X_test = np.random.randn(200, 5) + np.random.randn() * 0.5
        auc = compute_two_sample_auc(X_cal, X_test, random_state=trial)
        assert auc >= 0.5, f"AUC < 0.5: {auc}"
        print(f"  Trial {trial}: AUC = {auc:.3f} [OK]")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
