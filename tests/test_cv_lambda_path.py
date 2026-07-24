"""
Tests for ``LassoModel.fit_reg_lambda_path`` and the opt-in
``LassoModelCV(use_lambda_path=True)`` wiring.

Coverage:
- ``fit_reg_lambda_path`` returns one model per grid point, each equivalent
  (to solver tolerance) to a fresh ``fit`` at that ``reg_lambda``. Group
  family uses the DPP path; LASSO / cooperative / UniLasso fall back to a
  full fit per point and must match exactly.
- Order preservation and the empty-grid guard.
- ``LassoModelCV(use_lambda_path=True)`` reproduces the per-lambda loop's
  ``cv_scores_`` (to tolerance) and selects the same ``best_lambda_``, for
  the group family.
- The flag is a no-op for non-group estimators (the per-lambda loop runs),
  so ``True`` and ``False`` give identical ``cv_scores_``.
- Default (``use_lambda_path=False``) leaves the production CV path
  untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorlasso import LassoModel, LassoModelCV, LassoModelType

COEF_ATOL = 1e-5      # path vs fresh-fit loadings (solver tolerance)
SCORE_ATOL = 1e-6     # cv_scores_ path vs loop


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

def _data(T=180, M=5, N=9, seed=0):
    r = np.random.default_rng(seed)
    idx = pd.date_range("2014-01-01", periods=T, freq="W-WED")
    x = pd.DataFrame(r.standard_normal((T, M)) * 0.02,
                     index=idx, columns=[f"f{i}" for i in range(M)])
    B = np.zeros((N, M))
    for k in range(N):
        B[k, k % M] = 0.8
        B[k, (k + 1) % M] = 0.3
    y = pd.DataFrame(x.to_numpy() @ B.T + 0.01 * r.standard_normal((T, N)),
                     index=idx, columns=[f"a{i}" for i in range(N)])
    return x, y


X9, Y9 = _data(N=9, seed=0)
X6, Y6 = _data(N=6, seed=2)
Y6 = Y6[[f"a{i}" for i in range(6)]]
G6 = pd.Series([1, 1, 1, 2, 2, 2], index=Y6.columns)
GRID = [float(v) for v in np.logspace(-4, -1, 6)]


def _group_models():
    return {
        "GROUP": (LassoModel(model_type=LassoModelType.GROUP_LASSO,
                             group_data=G6, span=52), X6, Y6),
        "GROUP_sgl_autosign": (
            LassoModel(model_type=LassoModelType.GROUP_LASSO, group_data=G6,
                       span=52, l1_weight=0.3, auto_sign_constraints=True),
            X6, Y6),
        "HCGL": (LassoModel(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                            span=52, cutoff_fraction=0.5), X9, Y9),
        "HCGL_adaptive": (
            LassoModel(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                       span=52, cutoff_fraction=0.5, auto_sign_constraints=True,
                       auto_sign_adaptive_weights=True), X9, Y9),
        "FCGL_adaptive": (
            LassoModel(model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
                       span=52, cutoff_fraction=0.5, auto_sign_constraints=True,
                       auto_sign_adaptive_weights=True), X9, Y9),
    }


def _fallback_models():
    return {
        "LASSO": (LassoModel(model_type=LassoModelType.LASSO, span=52), X9, Y9),
        "UNILASSO": (LassoModel(model_type=LassoModelType.UNILASSO, span=52), X9, Y9),
        "COOP": (LassoModel(model_type=LassoModelType.COOPERATIVE_GROUP_LASSO,
                            group_data=G6, span=52), X6, Y6),
    }


# ═══════════════════════════════════════════════════════════════════════
# fit_reg_lambda_path parity
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", list(_group_models()))
def test_path_fit_matches_fresh_fits_group_family(name):
    base, x, y = _group_models()[name]
    fitted = base.fit_reg_lambda_path(x, y, reg_lambdas=GRID)
    assert len(fitted) == len(GRID)
    for lam, fm in zip(GRID, fitted):
        params = base.get_params()
        params["reg_lambda"] = lam
        ref = LassoModel(**params).fit(x=x, y=y)
        assert np.allclose(fm.coef_.to_numpy(), ref.coef_.to_numpy(),
                           atol=COEF_ATOL, rtol=0.0), f"{name} coef @ {lam:g}"
        assert np.allclose(fm.alpha_const_.to_numpy(),
                           ref.alpha_const_.to_numpy(),
                           atol=COEF_ATOL, rtol=0.0, equal_nan=True)


@pytest.mark.parametrize("name", list(_fallback_models()))
def test_path_fit_matches_fresh_fits_fallback_exact(name):
    base, x, y = _fallback_models()[name]
    fitted = base.fit_reg_lambda_path(x, y, reg_lambdas=GRID)
    assert len(fitted) == len(GRID)
    for lam, fm in zip(GRID, fitted):
        params = base.get_params()
        params["reg_lambda"] = lam
        ref = LassoModel(**params).fit(x=x, y=y)
        # fallback path is a full fit per point -> exact
        np.testing.assert_array_equal(fm.coef_.to_numpy(), ref.coef_.to_numpy())


def test_path_fit_order_preserved():
    base, x, y = _group_models()["HCGL"]
    grid = [1e-2, 1e-4, 5e-2, 1e-3]
    fitted = base.fit_reg_lambda_path(x, y, reg_lambdas=grid)
    for lam, fm in zip(grid, fitted):
        assert fm.reg_lambda == lam
        params = base.get_params()
        params["reg_lambda"] = lam
        ref = LassoModel(**params).fit(x=x, y=y)
        assert np.allclose(fm.coef_.to_numpy(), ref.coef_.to_numpy(),
                           atol=COEF_ATOL, rtol=0.0)


def test_path_fit_empty_grid_raises():
    base, x, y = _group_models()["HCGL"]
    with pytest.raises(ValueError, match="non-empty"):
        base.fit_reg_lambda_path(x, y, reg_lambdas=[])


# ═══════════════════════════════════════════════════════════════════════
# LassoModelCV(use_lambda_path=True) parity
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", list(_group_models()))
def test_cv_path_matches_loop_group_family(name):
    base, x, y = _group_models()[name]
    common = dict(base_model=base, lambdas=GRID, n_splits=4, refit=False)
    loop = LassoModelCV(use_lambda_path=False, **common).fit(x, y)
    path = LassoModelCV(use_lambda_path=True, **common).fit(x, y)
    assert np.allclose(loop.cv_scores_.to_numpy(), path.cv_scores_.to_numpy(),
                       atol=SCORE_ATOL, rtol=0.0, equal_nan=True), name
    assert loop.best_lambda_ == path.best_lambda_


@pytest.mark.parametrize("name", list(_fallback_models()))
def test_cv_path_flag_is_noop_for_non_group(name):
    base, x, y = _fallback_models()[name]
    common = dict(base_model=base, lambdas=GRID, n_splits=4, refit=False)
    loop = LassoModelCV(use_lambda_path=False, **common).fit(x, y)
    path = LassoModelCV(use_lambda_path=True, **common).fit(x, y)
    # non-group falls through to the per-lambda loop in both cases -> identical
    np.testing.assert_array_equal(
        np.nan_to_num(loop.cv_scores_.to_numpy(), nan=-999.0),
        np.nan_to_num(path.cv_scores_.to_numpy(), nan=-999.0),
    )
    assert loop.best_lambda_ == path.best_lambda_


def test_cv_default_is_lambda_path_off():
    """The production default must not engage the path."""
    cv = LassoModelCV()
    assert cv.use_lambda_path is False
