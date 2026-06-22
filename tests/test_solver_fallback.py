"""Opt-in solver fallback (`solver_fallbacks`) behaviour."""
import numpy as np
import pandas as pd
import pytest
import cvxpy as cvx

from factorlasso.lasso_estimator import LassoModel, LassoModelType


def _panel(T=200, K=4, N=3, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, K))
    B = rng.standard_normal((N, K))
    Y = X @ B.T + 0.3 * rng.standard_normal((T, N))
    Xdf = pd.DataFrame(X, columns=[f"f{j}" for j in range(K)])
    Ydf = pd.DataFrame(Y, columns=[f"a{i}" for i in range(N)])
    return Xdf, Ydf


def _fit_betas(**kw):
    X, Y = _panel()
    m = LassoModel(
        model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
        reg_lambda=1e-4, span=None, warmup_period=None, **kw,
    )
    m.fit(x=X, y=Y, verbose=False)
    return m.coef_.to_numpy(dtype=float)


def test_default_no_fallback_is_unchanged():
    """`solver_fallbacks=None` (default) reproduces the plain single solve."""
    a = _fit_betas()                        # field defaults to None
    b = _fit_betas(solver_fallbacks=None)   # explicit None
    assert np.max(np.abs(a - b)) == 0.0


def test_fallback_engages_when_primary_invalid():
    """A bogus primary solver name falls through to a working fallback and
    returns the same coefficients as solving directly with that fallback."""
    base = _fit_betas(solver="CLARABEL")
    fb = _fit_betas(solver="NO_SUCH_SOLVER", solver_fallbacks=["CLARABEL"])
    assert np.max(np.abs(base - fb)) < 1e-8


def test_all_solvers_failing_raises_solvererror():
    """Primary and every fallback invalid -> SolverError surfaces."""
    with pytest.raises(cvx.error.SolverError):
        _fit_betas(solver="NO_SUCH_SOLVER", solver_fallbacks=["ALSO_BOGUS"])
