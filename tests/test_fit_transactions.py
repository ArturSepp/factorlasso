"""Fitted state is replaced only by a completed fit, and inputs are aligned only when unlabelled.

A fit that raises keeps the previous fitted attributes; ``fit_reg_lambda_path`` never changes
the template's fitted attributes; a solve that fails without raising still stores its NaN
result. Two labelled inputs with different index labels are refused instead of being paired
by position.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import factorlasso.linear_model._dispatch as dispatch
import factorlasso.linear_model._estimator as estimator_module
from factorlasso import LassoModel, LassoModelType
from factorlasso.linear_model._solvers.common import _nan_result
from factorlasso.linear_model._state import fitted_attribute_names


def _panel(seed=7, t=90):
    """Six responses in two blocks driven by three factors, on a date index."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2010-01-31", periods=t, freq="ME")
    x = pd.DataFrame(rng.standard_normal((t, 3)), index=index, columns=["f0", "f1", "f2"])
    beta = np.array([[1, .3, 0], [1, .2, 0], [1, .1, 0], [0, 1, .3], [0, 1, .2], [0, 1, .1]])
    y = pd.DataFrame(x.to_numpy() @ beta.T + 0.1 * rng.standard_normal((t, 6)),
                     index=index, columns=[f"y{j}" for j in range(6)])
    return x, y


def _snapshot(model):
    """Identity of every fitted attribute, and which ones are set on the instance."""
    namespace = vars(model)
    return {name: (name in namespace, id(getattr(model, name)))
            for name in fitted_attribute_names(model)}


def _signed_hcgl(**overrides):
    """An HCGL model that populates sign, prior and bound diagnostics."""
    params = dict(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO, reg_lambda=1e-3,
                  n_clusters=2, span=36, auto_sign_constraints=True,
                  auto_sign_adaptive_weights=True, apply_ols_prior=True,
                  expert_prior_bound_n_std=2.0)
    params.update(overrides)
    return LassoModel(**params)


def test_validation_failure_keeps_previous_fit():
    """A fit-time validation error leaves every fitted attribute as it was."""
    x, y = _panel()
    model = _signed_hcgl().fit(x, y)
    before = _snapshot(model)
    model.set_params(auto_sign_excluded_factors=["missing"])
    with pytest.raises(ValueError, match="not present in x"):
        model.fit(x, y)
    assert _snapshot(model) == before
    assert model.auto_sign_excluded_factors == ["missing"]   # parameters are not rolled back


def test_escaping_solver_exception_keeps_previous_fit(monkeypatch):
    """An exception raised by the solve, after preparation has run, restores the previous fit."""
    x, y = _panel()
    model = _signed_hcgl().fit(x, y)
    before, coef = _snapshot(model), model.coef_.copy()

    def failing(*args, **kwargs):
        raise RuntimeError("solver crashed")

    monkeypatch.setattr(dispatch, "solve_group_lasso_cvx_problem", failing)
    with pytest.raises(RuntimeError, match="solver crashed"):
        model.fit(x.iloc[5:], y.iloc[5:])
    assert _snapshot(model) == before
    pd.testing.assert_frame_equal(model.coef_, coef)


def test_finalisation_failure_keeps_previous_fit(monkeypatch):
    """An exception while assembling the fitted state restores the previous fit."""
    x, y = _panel()
    model = _signed_hcgl().fit(x, y)
    before = _snapshot(model)

    def failing(*args, **kwargs):
        raise FloatingPointError("finalisation failed")

    monkeypatch.setattr(estimator_module, "fitted_state", failing)
    with pytest.raises(FloatingPointError):
        model.fit(x, y)
    assert _snapshot(model) == before


def test_failed_first_fit_leaves_model_unfitted():
    """An unfitted model whose fit raises after preparation began stays exactly unfitted."""
    x, y = _panel()
    model = LassoModel(apply_ols_prior=True, factors_beta_prior=pd.DataFrame(
        np.inf, index=y.columns, columns=x.columns))
    before = _snapshot(model)
    with pytest.raises(ValueError, match="finite or NaN"):
        model.fit(x, y)                       # raises after the OLS prior was stored
    assert _snapshot(model) == before
    assert model.coef_ is None and model.ols_betas_ is None


def test_nan_result_without_exception_is_still_installed(monkeypatch):
    """A solve that fails without raising warns and stores NaN coefficients, as before."""
    x, y = _panel()
    model = LassoModel(model_type=LassoModelType.LASSO).fit(x, y)

    def nan_solve(*args, x, y, **kwargs):
        warnings.warn("solver failed")
        return _nan_result(y.shape[1], x.shape[1])

    monkeypatch.setattr(dispatch, "solve_lasso_cvx_problem", nan_solve)
    with pytest.warns(UserWarning, match="solver failed"):
        model.fit(x, y)
    assert model.coef_.isna().all().all()


@pytest.mark.parametrize("prefit", [False, True], ids=["unfitted", "fitted"])
@pytest.mark.parametrize("model_type", [
    LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO, LassoModelType.LASSO,
], ids=["group-path", "per-lambda"])
def test_path_leaves_template_fitted_state_unchanged(model_type, prefit):
    """Group paths, single-response fallbacks and per-lambda paths keep the template's state."""
    x, y = _panel()
    template = _signed_hcgl(model_type=model_type)
    if prefit:
        template.fit(x.iloc[:60], y.iloc[:60])
    before = _snapshot(template)
    path = template.fit_reg_lambda_path(x, y, reg_lambdas=[1e-4, 1e-3])
    single = template.fit_reg_lambda_path(x, y[["y0"]], reg_lambdas=[1e-4])
    assert _snapshot(template) == before
    assert len(path) == 2 and path[0].detected_signs_ is not None
    assert single[0].coef_.shape == (1, 3)


def test_failed_path_leaves_template_unchanged(monkeypatch):
    """A path whose solve raises restores the template's fitted state."""
    x, y = _panel()
    template = _signed_hcgl().fit(x, y)
    before = _snapshot(template)

    def failing(*args, **kwargs):
        raise RuntimeError("path solver crashed")

    monkeypatch.setattr(dispatch, "solve_group_lasso_path", failing)
    with pytest.raises(RuntimeError):
        template.fit_reg_lambda_path(x, y, reg_lambdas=[1e-4, 1e-3])
    assert _snapshot(template) == before


def test_repeated_fits_describe_the_latest_data():
    """A second fit on other data replaces every fitted attribute consistently."""
    x, y = _panel()
    model = _signed_hcgl().fit(x, y)
    model.fit(x.iloc[10:], y.iloc[10:])
    fresh = _signed_hcgl().fit(x.iloc[10:], y.iloc[10:])
    pd.testing.assert_frame_equal(model.coef_, fresh.coef_)
    pd.testing.assert_frame_equal(model.detected_signs_, fresh.detected_signs_)
    pd.testing.assert_frame_equal(model.prior_lower_bounds_, fresh.prior_lower_bounds_)
    assert model.x_.index.equals(x.index[10:])


def test_differently_labelled_inputs_are_refused():
    """Two labelled inputs of equal length but different labels are not paired by position."""
    x, y = _panel()
    shifted = y.set_axis(y.index + pd.offsets.MonthEnd(1), axis=0)
    model = LassoModel()
    with pytest.raises(ValueError, match="different index labels"):
        model.fit(x, shifted)
    assert model.coef_ is None


def test_unlabelled_inputs_take_the_other_inputs_labels():
    """ndarrays and default-indexed frames adopt the labelled input's index, on either side."""
    x, y = _panel()
    with_unlabelled_y = LassoModel().fit(x, y.to_numpy())
    assert with_unlabelled_y.y_.index.equals(x.index)
    with_unlabelled_x = LassoModel().fit(x.reset_index(drop=True), y)
    assert with_unlabelled_x.x_.index.equals(y.index)
    pd.testing.assert_frame_equal(with_unlabelled_x.coef_, LassoModel().fit(x, y).coef_)
    both_arrays = LassoModel().fit(x.to_numpy(), y.to_numpy())
    np.testing.assert_allclose(both_arrays.coef_.to_numpy(),
                               LassoModel().fit(x, y).coef_.to_numpy())
