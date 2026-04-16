"""
Regression tests for ``compute_ewm_covar(is_corr=True)`` with mixed
zero/positive variances.

Before the fix, any diagonal element equal to zero would cause
``np.reciprocal(np.sqrt(0)) = inf``, which then poisoned the whole
correlation matrix with ``inf * 0 = NaN`` during the outer product.
In downstream HCGL clustering this would silently corrupt the
Ward-linkage distances. The fix is a per-element guard: zero-variance
assets get a zero correlation row/column and a diagonal of 1.

These tests would have caught the production warning flood seen in
``solve_for_risk_budgets_from_given_weights`` when shorts were excluded
from the optimization universe.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from factorlasso.ewm_utils import compute_ewm_covar


@pytest.fixture
def rng():
    return np.random.default_rng(20260416)


@pytest.fixture
def panel_one_zero_variance(rng):
    """5-column panel where column 2 has identically zero variance."""
    x = rng.standard_normal((200, 5))
    x[:, 2] = 0.0  # flat — zero variance for all time
    return x


@pytest.fixture
def panel_one_allnan(rng):
    """5-column panel where column 3 is all NaN."""
    x = rng.standard_normal((200, 5))
    x[:, 3] = np.nan
    return x


@pytest.fixture
def panel_all_positive(rng):
    """5-column panel with all columns having positive variance."""
    return rng.standard_normal((200, 5))


# ═══════════════════════════════════════════════════════════════════════
# Warning hygiene
# ═══════════════════════════════════════════════════════════════════════

class TestNoRuntimeWarnings:
    def test_zero_variance_column_no_warnings(self, panel_one_zero_variance):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            compute_ewm_covar(panel_one_zero_variance, span=20, is_corr=True)

    def test_all_nan_column_no_warnings(self, panel_one_allnan):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            compute_ewm_covar(panel_one_allnan, span=20, is_corr=True)

    def test_covar_not_corr_still_quiet(self, panel_one_zero_variance):
        """Plain covariance (is_corr=False) should also be warning-free."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            compute_ewm_covar(panel_one_zero_variance, span=20, is_corr=False)


# ═══════════════════════════════════════════════════════════════════════
# Correctness of the fix
# ═══════════════════════════════════════════════════════════════════════

class TestCorrelationMatrixValidity:
    def test_diagonal_is_one_for_zero_variance_column(
        self, panel_one_zero_variance,
    ):
        c = compute_ewm_covar(panel_one_zero_variance, span=20, is_corr=True)
        # All diagonal elements should be 1 (or close to it for the
        # positive-variance columns; exactly 1 for the zero-variance one)
        np.testing.assert_allclose(np.diag(c), 1.0, atol=1e-10)

    def test_zero_variance_row_col_is_zero_offdiagonal(
        self, panel_one_zero_variance,
    ):
        c = compute_ewm_covar(panel_one_zero_variance, span=20, is_corr=True)
        # Column 2 is the zero-variance one
        off_diag_row = np.delete(c[2, :], 2)
        off_diag_col = np.delete(c[:, 2], 2)
        np.testing.assert_allclose(off_diag_row, 0.0, atol=1e-10)
        np.testing.assert_allclose(off_diag_col, 0.0, atol=1e-10)

    def test_no_inf_or_nan_in_output(self, panel_one_zero_variance):
        c = compute_ewm_covar(panel_one_zero_variance, span=20, is_corr=True)
        assert np.all(np.isfinite(c)), (
            "Correlation matrix must not contain inf or NaN"
        )

    def test_is_symmetric(self, panel_one_zero_variance):
        c = compute_ewm_covar(panel_one_zero_variance, span=20, is_corr=True)
        np.testing.assert_allclose(c, c.T, atol=1e-12)

    def test_offdiagonal_bounded(self, panel_one_zero_variance):
        """Correlations must lie in [-1, 1]."""
        c = compute_ewm_covar(panel_one_zero_variance, span=20, is_corr=True)
        assert np.all(c <= 1.0 + 1e-10)
        assert np.all(c >= -1.0 - 1e-10)


# ═══════════════════════════════════════════════════════════════════════
# Regression guard: all-positive-variance case is unchanged
# ═══════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    def test_all_positive_unchanged(self, panel_all_positive):
        """
        When all diagonal elements are strictly positive, the fix must
        produce exactly the same result as the original code path.
        This is the condition that ensures qis parity for normal inputs.
        """
        c = compute_ewm_covar(panel_all_positive, span=20, is_corr=True)
        # Manual reference: standard correlation normalisation
        raw = compute_ewm_covar(panel_all_positive, span=20, is_corr=False)
        d = np.sqrt(np.diag(raw))
        expected = raw / np.outer(d, d)
        np.testing.assert_allclose(c, expected, atol=1e-12)


# ═══════════════════════════════════════════════════════════════════════
# The HCGL scenario that triggered the production bug
# ═══════════════════════════════════════════════════════════════════════

class TestHCGLClusteringScenario:
    def test_clustering_on_mixed_variance_panel(self, rng):
        """
        Reproduce the HCGL use case: correlation matrix of responses,
        some with zero variance, fed to Ward linkage via
        compute_clusters_from_corr_matrix.
        """
        from factorlasso import compute_clusters_from_corr_matrix

        T, N = 150, 6
        y = rng.standard_normal((T, N))
        y[:, 2] = 0.0  # zero-variance asset (e.g., recently-delisted)
        y[:, 4] = np.nan  # missing-history asset

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            corr = compute_ewm_covar(y, span=20, is_corr=True)

        corr_df = pd.DataFrame(
            corr, columns=[f"y{i}" for i in range(N)],
            index=[f"y{i}" for i in range(N)],
        )

        # Clustering should succeed (no inf/nan to poison Ward distances)
        clusters, linkage, cutoff = compute_clusters_from_corr_matrix(corr_df)
        assert len(clusters) == N
        assert np.all(np.isfinite(linkage))
