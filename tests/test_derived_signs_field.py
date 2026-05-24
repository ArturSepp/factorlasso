"""
Regression tests for the v0.3.10 ``derived_signs`` field on
``CurrentFactorCovarData``.

These pin three properties:

1. ``derived_signs`` defaults to ``None`` (backward-compatible).
2. ``filter_on_tickers`` subsets the sign matrix in lock-step with the
   loadings.
3. ``save`` / ``load`` round-trip preserves the sign matrix when it was
   present, and load returns ``None`` for files that pre-date the field.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from factorlasso import CurrentFactorCovarData


@pytest.fixture
def factor_covar_with_signs():
    """Minimal CurrentFactorCovarData carrying a derived_signs matrix."""
    N, M = 4, 3
    x_covar = pd.DataFrame(
        np.eye(M), index=[f"f{j}" for j in range(M)],
        columns=[f"f{j}" for j in range(M)],
    )
    y_betas = pd.DataFrame(
        np.array([[0.5, 0.3, 0.0],
                  [0.4, 0.0, 0.2],
                  [0.6, 0.5, 0.1],
                  [0.2, 0.4, 0.0]]),
        index=[f"a{k}" for k in range(N)],
        columns=[f"f{j}" for j in range(M)],
    )
    y_var = pd.DataFrame(
        {"residual_var": np.ones(N) * 0.01},
        index=y_betas.index,
    )
    signs = pd.DataFrame(
        np.array([[1.0,  1.0, 0.0],
                  [1.0,  0.0, 1.0],
                  [1.0,  1.0, 1.0],
                  [1.0,  1.0, 0.0]]),
        index=y_betas.index,
        columns=y_betas.columns,
    )
    return CurrentFactorCovarData(
        x_covar=x_covar, y_betas=y_betas, y_variances=y_var,
        derived_signs=signs,
    )


def test_derived_signs_default_is_none():
    """No derived_signs argument → field is None (backward compatible)."""
    M, N = 2, 3
    x_covar = pd.DataFrame(np.eye(M), index=[f"f{j}" for j in range(M)],
                           columns=[f"f{j}" for j in range(M)])
    y_betas = pd.DataFrame(np.zeros((N, M)),
                           index=[f"a{k}" for k in range(N)],
                           columns=x_covar.index)
    y_var = pd.DataFrame({"residual_var": np.ones(N) * 0.01},
                         index=y_betas.index)
    data = CurrentFactorCovarData(
        x_covar=x_covar, y_betas=y_betas, y_variances=y_var,
    )
    assert data.derived_signs is None


def test_filter_on_tickers_subsets_derived_signs_list(factor_covar_with_signs):
    """List-form filter_on_tickers subsets derived_signs to the kept rows."""
    keep = ["a1", "a3"]
    sub = factor_covar_with_signs.filter_on_tickers(keep)
    assert sub.derived_signs is not None
    assert list(sub.derived_signs.index) == keep
    # Row content is the original a1 and a3 rows
    np.testing.assert_array_equal(
        sub.derived_signs.loc["a1"].to_numpy(), [1.0, 0.0, 1.0],
    )
    np.testing.assert_array_equal(
        sub.derived_signs.loc["a3"].to_numpy(), [1.0, 1.0, 0.0],
    )


def test_filter_on_tickers_subsets_derived_signs_dict(factor_covar_with_signs):
    """Dict-form filter_on_tickers (rename) subsets and renames the sign
    matrix to match the loadings rename."""
    rename = {"a0": "first", "a2": "third"}
    sub = factor_covar_with_signs.filter_on_tickers(rename)
    assert sub.derived_signs is not None
    assert list(sub.derived_signs.index) == ["first", "third"]
    # Verify the renamed rows carry the original a0 and a2 sign rows
    np.testing.assert_array_equal(
        sub.derived_signs.loc["first"].to_numpy(), [1.0, 1.0, 0.0],
    )
    np.testing.assert_array_equal(
        sub.derived_signs.loc["third"].to_numpy(), [1.0, 1.0, 1.0],
    )


def test_filter_with_no_derived_signs_remains_none():
    """If derived_signs was None on the source, filter_on_tickers
    preserves None on the filtered result."""
    M, N = 2, 3
    x_covar = pd.DataFrame(np.eye(M), index=[f"f{j}" for j in range(M)],
                           columns=[f"f{j}" for j in range(M)])
    y_betas = pd.DataFrame(np.zeros((N, M)),
                           index=[f"a{k}" for k in range(N)],
                           columns=x_covar.index)
    y_var = pd.DataFrame({"residual_var": np.ones(N) * 0.01},
                         index=y_betas.index)
    data = CurrentFactorCovarData(
        x_covar=x_covar, y_betas=y_betas, y_variances=y_var,
    )
    sub = data.filter_on_tickers(["a0", "a1"])
    assert sub.derived_signs is None


def test_save_load_roundtrip_preserves_derived_signs(factor_covar_with_signs):
    """save() writes a derived_signs sheet, load() reconstructs it."""
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        path = f.name
    try:
        factor_covar_with_signs.save(path)
        loaded = CurrentFactorCovarData.load(path)
        assert loaded.derived_signs is not None
        pd.testing.assert_frame_equal(
            loaded.derived_signs,
            factor_covar_with_signs.derived_signs,
            check_dtype=False,
        )
    finally:
        os.unlink(path)


def test_load_backward_compat_no_derived_signs_sheet():
    """Files saved before v0.3.10 lack a derived_signs sheet; load()
    returns None for the field rather than raising."""
    M, N = 2, 3
    x_covar = pd.DataFrame(np.eye(M), index=[f"f{j}" for j in range(M)],
                           columns=[f"f{j}" for j in range(M)])
    y_betas = pd.DataFrame(np.zeros((N, M)),
                           index=[f"a{k}" for k in range(N)],
                           columns=x_covar.index)
    y_var = pd.DataFrame({"residual_var": np.ones(N) * 0.01},
                         index=y_betas.index)
    data = CurrentFactorCovarData(
        x_covar=x_covar, y_betas=y_betas, y_variances=y_var,
    )
    # Save (no derived_signs sheet written) and reload
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        path = f.name
    try:
        data.save(path)
        loaded = CurrentFactorCovarData.load(path)
        assert loaded.derived_signs is None
    finally:
        os.unlink(path)
