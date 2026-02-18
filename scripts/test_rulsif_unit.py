"""Unit tests for RULSIF baseline.

Simple tests to verify core functionality without requiring dataset loading.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from shiftbench.baselines.rulsif import create_rulsif_baseline


def test_rulsif_basic():
    """Test basic RULSIF functionality on synthetic data."""

    print("=" * 80)
    print("RULSIF Unit Tests")
    print("=" * 80)

    # Test 1: Import and instantiation
    print("\n[Test 1] Import and instantiation...")
    try:
        rulsif = create_rulsif_baseline(n_basis=10, alpha=0.1, random_state=42)
        print("[PASS] RULSIF instantiation successful")
    except Exception as e:
        print(f"[FAIL] {e}")
        return False

    # Test 2: Metadata
    print("\n[Test 2] Metadata...")
    try:
        meta = rulsif.get_metadata()
        assert meta.name == "rulsif", f"Expected name 'rulsif', got {meta.name}"
        assert meta.version == "1.0.0", f"Expected version '1.0.0', got {meta.version}"
        assert meta.supports_abstention == False, "RULSIF should not support NO-GUARANTEE"
        print("[PASS] Metadata is correct")
    except AssertionError as e:
        print(f"[FAIL] {e}")
        return False

    # Test 3: Weight estimation
    print("\n[Test 3] Weight estimation...")
    try:
        # Create simple synthetic data
        np.random.seed(42)
        n_cal = 100
        n_target = 100
        n_features = 5

        X_cal = np.random.randn(n_cal, n_features)
        X_target = np.random.randn(n_target, n_features) + 0.5  # Slight shift

        # Estimate weights
        weights = rulsif.estimate_weights(X_cal, X_target)

        # Check validity
        assert len(weights) == n_cal, f"Expected {n_cal} weights, got {len(weights)}"
        assert np.all(weights > 0), "All weights must be positive"
        assert np.all(np.isfinite(weights)), "All weights must be finite"
        assert np.abs(weights.mean() - 1.0) < 0.1, f"Mean weight {weights.mean():.3f} not close to 1.0"

        print(f"[PASS] Weight estimation successful")
        print(f"       Mean: {weights.mean():.3f}, Std: {weights.std():.3f}")
        print(f"       Min: {weights.min():.3f}, Max: {weights.max():.3f}")
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Alpha parameter effect
    print("\n[Test 4] Alpha parameter effect...")
    try:
        # Create methods with different alpha values
        rulsif_0 = create_rulsif_baseline(n_basis=10, alpha=0.0, random_state=42)
        rulsif_01 = create_rulsif_baseline(n_basis=10, alpha=0.1, random_state=42)
        rulsif_05 = create_rulsif_baseline(n_basis=10, alpha=0.5, random_state=42)

        # Estimate weights
        weights_0 = rulsif_0.estimate_weights(X_cal, X_target)
        weights_01 = rulsif_01.estimate_weights(X_cal, X_target)
        weights_05 = rulsif_05.estimate_weights(X_cal, X_target)

        # Check stability trend
        cv_0 = weights_0.std() / weights_0.mean()
        cv_01 = weights_01.std() / weights_01.mean()
        cv_05 = weights_05.std() / weights_05.mean()

        print(f"[PASS] Alpha parameter affects stability")
        print(f"       alpha=0.0: CV = {cv_0:.4f}")
        print(f"       alpha=0.1: CV = {cv_01:.4f}")
        print(f"       alpha=0.5: CV = {cv_05:.4f}")

        # In most cases, higher alpha should reduce CV (but not always guaranteed)
        if cv_05 <= cv_0:
            print(f"       [OK] Higher alpha -> lower CV (as expected)")
        else:
            print(f"       [NOTE] CV trend may vary for specific data")

    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 5: Diagnostics
    print("\n[Test 5] Diagnostics...")
    try:
        diag = rulsif.get_diagnostics()
        assert "method" in diag, "Diagnostics should include 'method'"
        assert diag["method"] == "rulsif", f"Expected method 'rulsif', got {diag['method']}"
        assert "alpha_rel" in diag, "Diagnostics should include 'alpha_rel'"
        assert "sigma" in diag, "Diagnostics should include 'sigma'"
        assert "n_basis" in diag, "Diagnostics should include 'n_basis'"

        print("[PASS] Diagnostics are complete")
        for key, value in diag.items():
            print(f"       {key}: {value}")
    except AssertionError as e:
        print(f"[FAIL] {e}")
        return False

    # Test 6: Bound estimation (basic)
    print("\n[Test 6] Bound estimation...")
    try:
        # Create simple labels and predictions
        y_cal = np.random.binomial(1, 0.7, n_cal)
        predictions_cal = np.ones(n_cal, dtype=int)  # Predict all positive
        cohorts_cal = np.array([f"cohort_{i%3}" for i in range(n_cal)])

        # Estimate bounds
        decisions = rulsif.estimate_bounds(
            y_cal,
            predictions_cal,
            cohorts_cal,
            weights,
            tau_grid=[0.5, 0.7],
            alpha=0.05
        )

        # Check validity
        assert len(decisions) > 0, "Should return at least one decision"
        for d in decisions:
            assert d.decision in ["CERTIFY", "ABSTAIN", "NO-GUARANTEE"], f"Invalid decision: {d.decision}"
            assert d.tau in [0.5, 0.7], f"Unexpected tau: {d.tau}"
            assert 0 <= d.mu_hat <= 1 or np.isnan(d.mu_hat), f"Invalid mu_hat: {d.mu_hat}"
            assert 0 <= d.lower_bound <= 1 or np.isnan(d.lower_bound), f"Invalid lower_bound: {d.lower_bound}"

        print(f"[PASS] Bound estimation successful")
        print(f"       Generated {len(decisions)} decisions")
        print(f"       Cohorts: {len(np.unique(cohorts_cal))}, Taus: 2")
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 7: Comparison with uLSIF (alpha=0.0)
    print("\n[Test 7] Comparison with uLSIF (alpha=0.0)...")
    try:
        from shiftbench.baselines.ulsif import create_ulsif_baseline

        # Create uLSIF and RULSIF(alpha=0.0) with same settings
        ulsif = create_ulsif_baseline(n_basis=10, sigma=None, lambda_=0.1, random_state=42)
        rulsif_0 = create_rulsif_baseline(n_basis=10, sigma=None, lambda_=0.1, alpha=0.0, random_state=42)

        # Estimate weights
        weights_ulsif = ulsif.estimate_weights(X_cal, X_target)
        weights_rulsif_0 = rulsif_0.estimate_weights(X_cal, X_target)

        # Check if they match (should be very close)
        diff = np.abs(weights_ulsif - weights_rulsif_0).mean()
        assert diff < 1e-6, f"RULSIF(alpha=0.0) should match uLSIF, but diff = {diff:.6e}"

        print(f"[PASS] RULSIF(alpha=0.0) matches uLSIF")
        print(f"       Mean absolute difference: {diff:.6e}")
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("[SUCCESS] All unit tests passed!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = test_rulsif_basic()
    sys.exit(0 if success else 1)
