"""Causal residual-alpha nowcasting contracts for :class:`LassoModel`."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

import factorlasso.lasso_estimator as estimator_module
from factorlasso import (
    LassoModel,
    LassoModelType,
    LassoNowcastResult,
    get_x_y_np,
)


def _panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return a small dated factor/response panel and two future rows."""
    rng = np.random.default_rng(20260831)
    index = pd.date_range("2021-01-31", periods=36, freq="ME")
    x = pd.DataFrame(
        rng.normal(scale=0.025, size=(len(index), 2)),
        index=index,
        columns=["Equity", "Rates"],
    )
    beta = np.array([[0.8, -0.2], [0.3, 0.6]])
    y = pd.DataFrame(
        np.array([0.006, -0.004])
        + x.to_numpy() @ beta.T
        + rng.normal(scale=0.004, size=(len(index), 2)),
        index=index,
        columns=["Manager", "PE Index"],
    )
    y.iloc[:2, 0] = np.nan
    y.iloc[1, 1] = np.nan
    y.iloc[11, 0] = np.nan
    y.iloc[18, 1] = np.nan
    target_index = pd.date_range(index[-1] + pd.offsets.MonthEnd(), periods=2, freq="ME")
    target = pd.DataFrame(
        [[0.012, -0.003], [-0.008, 0.004]],
        index=target_index,
        columns=x.columns,
    )
    return x, y, target


def _fit(
    x: pd.DataFrame,
    y: pd.DataFrame,
    *,
    span: float | None = 8.0,
    demean: bool = True,
) -> LassoModel:
    """Fit the test panel without a warmup eligibility side effect."""
    return LassoModel(
        model_type=LassoModelType.LASSO,
        reg_lambda=1e-5,
        span=span,
        demean=demean,
        warmup_period=None,
    ).fit(x=x, y=y)


def _manual_terminal_alpha(residuals: pd.DataFrame, span: float) -> pd.Series:
    """Compute terminal residual EWMA without FactorLasso helpers."""
    decay = (span - 1.0) / (span + 1.0)
    output = {}
    for column in residuals:
        values = residuals[column].to_numpy(dtype=float)
        first = int(np.flatnonzero(np.isfinite(values))[0])
        state = values[first]
        for value in values[first + 1:]:
            if np.isfinite(value):
                state = decay * state + (1.0 - decay) * value
        output[column] = state
    return pd.Series(output, name="stat_alpha")


def _manual_ewm_for_solver(values: np.ndarray, span: float) -> np.ndarray:
    """Reproduce the solver's adjust-false EWMA state independently."""
    decay = (span - 1.0) / (span + 1.0)
    initial = np.where(np.isfinite(values[0]), values[0], 0.0)
    last = np.where(np.isfinite(values[0]), initial, np.nan)
    output = np.full_like(values, np.nan, dtype=float)
    output[0] = last
    for row in range(1, len(values)):
        observation = values[row]
        start = np.logical_and(~np.isfinite(last), np.isfinite(observation))
        last = np.where(start, initial, last)
        current = decay * last + (1.0 - decay) * observation
        current = np.where(np.isfinite(current), current, last)
        output[row] = current
        last = current
    return output


def _manual_solver_diagnostics(
    x: pd.DataFrame,
    y: pd.DataFrame,
    beta: pd.DataFrame,
    span: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return independent SS, R2, Kish size, and used-count references."""
    x_values = x.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)
    valid = np.isfinite(y_values).astype(float)
    if span is None:
        x_demeaned = x_values - np.nanmean(x_values, axis=0)
        y_demeaned = y_values - np.nanmean(y_values, axis=0)
        loss_weights = valid
    else:
        x_demeaned = x_values - _manual_ewm_for_solver(x_values, span)
        y_demeaned = y_values - _manual_ewm_for_solver(y_values, span)
        x_demeaned = x_demeaned[1:]
        y_demeaned = y_demeaned[1:]
        valid = valid[1:]
        decay = (span - 1.0) / (span + 1.0)
        time_weights = decay ** np.arange(len(valid) - 1, -1, -1)
        loss_weights = time_weights[:, None] * valid

    x_safe = np.nan_to_num(x_demeaned, nan=0.0)
    y_safe = np.nan_to_num(y_demeaned, nan=0.0)
    weight_sums = loss_weights.sum(axis=0)
    normalized = np.divide(
        loss_weights,
        weight_sums,
        out=np.zeros_like(loss_weights),
        where=weight_sums > 0.0,
    )
    residuals = y_safe - x_safe @ beta.to_numpy(dtype=float).T
    y_mean = np.sum(normalized * y_safe, axis=0)
    ss_total = np.sum(normalized * np.square(y_safe - y_mean), axis=0)
    ss_res = np.sum(normalized * np.square(residuals), axis=0)
    r2 = 1.0 - ss_res / ss_total
    effective_n = np.square(weight_sums) / np.square(loss_weights).sum(axis=0)
    return ss_total, ss_res, r2, effective_n, valid.sum(axis=0)


def _assert_same_nowcast(left: LassoNowcastResult, right: LassoNowcastResult) -> None:
    """Assert equality of every nowcast result field."""
    pd.testing.assert_frame_equal(left.prediction, right.prediction)
    pd.testing.assert_frame_equal(left.factor_component, right.factor_component)
    pd.testing.assert_frame_equal(left.target_factors, right.target_factors)
    pd.testing.assert_series_equal(left.stat_alpha, right.stat_alpha)
    pd.testing.assert_frame_equal(left.betas, right.betas)
    pd.testing.assert_frame_equal(left.residuals, right.residuals)
    pd.testing.assert_frame_equal(left.diagnostics, right.diagnostics)


@pytest.mark.parametrize("span", [None, 8.0])
def test_fit_uses_demeaned_solver_inputs(monkeypatch, span):
    """The solver receives the public de-meaned preparation for both spans."""
    x, y, _ = _panel()
    expected = get_x_y_np(x=x, y=y, span=span, demean=True)
    captured = {}
    original = estimator_module.solve_lasso_cvx_problem

    def capture_solver(*args, **kwargs):
        """Capture solver arrays and delegate to the real implementation."""
        captured.update(
            x=kwargs["x"].copy(),
            y=kwargs["y"].copy(),
            valid_mask=kwargs["valid_mask"].copy(),
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(estimator_module, "solve_lasso_cvx_problem", capture_solver)
    model = _fit(x, y, span=span)
    np.testing.assert_array_equal(captured["x"], expected[0])
    np.testing.assert_array_equal(captured["y"], expected[1])
    np.testing.assert_array_equal(captured["valid_mask"], expected[2])
    assert model.fit_demeaned_ is True
    assert model.effective_span_ == span


def test_nowcast_matches_independent_alpha_prediction_and_diagnostics():
    """Prediction and fitted diagnostics match independent calculations."""
    x, y, target = _panel()
    model = _fit(x, y, span=8.0)
    result = model.nowcast(target)

    expected_alpha = _manual_terminal_alpha(result.residuals, span=8.0)
    expected_factor = target @ model.coef_.T
    expected_prediction = expected_factor.add(expected_alpha, axis="columns")
    pd.testing.assert_series_equal(result.stat_alpha, expected_alpha)
    pd.testing.assert_frame_equal(result.factor_component, expected_factor)
    pd.testing.assert_frame_equal(result.prediction, expected_prediction)
    assert not np.allclose(
        result.prediction,
        expected_prediction.add(model.alpha_const_, axis="columns"),
    )
    assert not np.allclose(
        result.prediction,
        expected_prediction.add(model.intercept_, axis="columns"),
    )

    ss_total, ss_res, r2, effective_n, n_used = _manual_solver_diagnostics(
        x, y, model.coef_, span=8.0
    )
    diagnostics = result.diagnostics
    expected_columns = [
        "fit_start_date",
        "fit_end_date",
        "n_response_obs_to_cutoff",
        "n_obs_used",
        "effective_n_obs",
        "beta_span",
        "alpha_span",
        "fit_demeaned",
        "stat_alpha",
        "alpha_const",
        "factorlasso_fit_ss_total_ewma_demeaned",
        "factorlasso_fit_ss_res_ewma_demeaned",
        "factorlasso_fit_r2_ewma_demeaned",
        "n_nonzero_betas",
    ]
    assert diagnostics.columns.tolist() == expected_columns
    assert diagnostics.index.equals(y.columns)
    np.testing.assert_allclose(
        diagnostics["factorlasso_fit_ss_total_ewma_demeaned"], ss_total
    )
    np.testing.assert_allclose(
        diagnostics["factorlasso_fit_ss_res_ewma_demeaned"], ss_res
    )
    np.testing.assert_allclose(
        diagnostics["factorlasso_fit_r2_ewma_demeaned"], r2
    )
    np.testing.assert_array_equal(
        diagnostics["factorlasso_fit_ss_total_ewma_demeaned"],
        model.estimation_result_.ss_total,
    )
    np.testing.assert_array_equal(
        diagnostics["factorlasso_fit_ss_res_ewma_demeaned"],
        model.estimation_result_.ss_res,
    )
    np.testing.assert_array_equal(
        diagnostics["factorlasso_fit_r2_ewma_demeaned"],
        model.estimation_result_.r2,
    )
    np.testing.assert_allclose(diagnostics["effective_n_obs"], effective_n)
    np.testing.assert_array_equal(diagnostics["n_obs_used"], n_used.astype(int))
    np.testing.assert_array_equal(
        diagnostics["n_response_obs_to_cutoff"], y.notna().sum(axis=0)
    )
    np.testing.assert_array_equal(diagnostics["beta_span"], 8.0)
    np.testing.assert_array_equal(diagnostics["alpha_span"], 8.0)
    np.testing.assert_array_equal(diagnostics["fit_demeaned"], True)
    np.testing.assert_array_equal(diagnostics["alpha_const"], model.alpha_const_)
    np.testing.assert_array_equal(
        diagnostics["stat_alpha"], expected_alpha.to_numpy()
    )
    expected_start_dates = pd.Series(
        [result.residuals[column].first_valid_index() for column in y.columns],
        index=y.columns,
        name="fit_start_date",
    )
    pd.testing.assert_series_equal(
        diagnostics["fit_start_date"], expected_start_dates
    )
    pd.testing.assert_series_equal(
        diagnostics["fit_end_date"],
        pd.Series(
            result.residuals.index[-1],
            index=y.columns,
            name="fit_end_date",
        ),
    )
    np.testing.assert_array_equal(
        diagnostics["n_nonzero_betas"],
        np.count_nonzero(~np.isclose(model.coef_.to_numpy(), 0.0), axis=1),
    )
    pd.testing.assert_frame_equal(result.betas, model.coef_)
    pd.testing.assert_frame_equal(result.residuals, y - x @ model.coef_.T)


def test_uniform_alpha_is_residual_mean_and_effective_size_is_count():
    """A uniform beta fit uses a uniform residual mean and Kish count."""
    x, y, target = _panel()
    model = _fit(x, y, span=None)
    result = model.nowcast(target)
    expected = result.residuals.mean(axis=0, skipna=True).rename("stat_alpha")
    pd.testing.assert_series_equal(result.stat_alpha, expected)
    np.testing.assert_array_equal(
        result.diagnostics["effective_n_obs"], result.diagnostics["n_obs_used"]
    )
    assert result.diagnostics["beta_span"].isna().all()
    assert result.diagnostics["alpha_span"].isna().all()


def test_explicit_alpha_span_uses_independent_recursion():
    """A valid explicit alpha span overrides only the residual recursion."""
    x, y, target = _panel()
    model = _fit(x, y, span=8.0)
    result = model.nowcast(target, alpha_span=4.0)
    expected = _manual_terminal_alpha(result.residuals, span=4.0)
    pd.testing.assert_series_equal(result.stat_alpha, expected)
    np.testing.assert_array_equal(result.diagnostics["beta_span"], 8.0)
    np.testing.assert_array_equal(result.diagnostics["alpha_span"], 4.0)


def test_nowcast_uses_fit_time_demeaning_provenance():
    """Post-fit mutation of the hyperparameter cannot change admission."""
    x, y, target = _panel()
    demeaned = _fit(x, y, demean=True)
    demeaned.demean = False
    assert demeaned.nowcast(target).diagnostics["fit_demeaned"].all()

    through_origin = _fit(x, y, demean=False)
    through_origin.demean = True
    with pytest.raises(ValueError, match="fitted with demean=True"):
        through_origin.nowcast(target)
    with pytest.raises(RuntimeError, match="not fitted"):
        LassoModel().nowcast(target)


def test_training_and_target_mutations_do_not_change_results():
    """Caller-owned training and target mutations cannot alter snapshots."""
    x, y, target = _panel()
    model = _fit(x, y)
    first = model.nowcast(target)
    x.iloc[:, :] = 100.0
    y.iloc[:, :] = -100.0
    second = model.nowcast(target)
    _assert_same_nowcast(first, second)

    saved_target = second.target_factors.copy(deep=True)
    saved_prediction = second.prediction.copy(deep=True)
    target.iloc[:, :] = 50.0
    pd.testing.assert_frame_equal(second.target_factors, saved_target)
    pd.testing.assert_frame_equal(second.prediction, saved_prediction)
    assert not np.shares_memory(second.target_factors.to_numpy(), target.to_numpy())
    assert not np.shares_memory(second.residuals.to_numpy(), y.to_numpy())


def test_target_change_only_changes_factor_component():
    """Future factors cannot change alpha, beta, history, or fit diagnostics."""
    x, y, target = _panel()
    model = _fit(x, y)
    first = model.nowcast(target)
    changed_target = target.copy(deep=True)
    changed_target.iloc[0, 0] += 0.01
    second = model.nowcast(changed_target)

    pd.testing.assert_series_equal(first.stat_alpha, second.stat_alpha)
    pd.testing.assert_frame_equal(first.betas, second.betas)
    pd.testing.assert_frame_equal(first.residuals, second.residuals)
    pd.testing.assert_frame_equal(first.diagnostics, second.diagnostics)
    delta = changed_target - target
    expected_delta = delta @ first.betas.T
    pd.testing.assert_frame_equal(
        second.factor_component - first.factor_component, expected_delta
    )
    pd.testing.assert_frame_equal(second.prediction - first.prediction, expected_delta)


@pytest.mark.parametrize(
    ("case", "error_type", "message"),
    [
        ("not_frame", TypeError, "pandas DataFrame"),
        ("empty", ValueError, "at least one"),
        ("missing_column", ValueError, "exactly match"),
        ("reordered", ValueError, "same order"),
        ("range_index", TypeError, "DatetimeIndex"),
        ("duplicate_index", ValueError, "sorted and unique"),
        ("unsorted_index", ValueError, "sorted and unique"),
        ("timezone", ValueError, "same timezone"),
        ("at_cutoff", ValueError, "strictly after"),
        ("nonfinite", ValueError, "all be finite"),
        ("bad_alpha_span", ValueError, "alpha_span"),
    ],
)
def test_target_validation(case, error_type, message):
    """Target frames fail closed on every schema and time violation."""
    x, y, target = _panel()
    model = _fit(x, y)
    candidate = target.copy(deep=True)
    kwargs = {}
    if case == "not_frame":
        candidate = candidate.to_numpy()
    elif case == "empty":
        candidate = candidate.iloc[0:0]
    elif case == "missing_column":
        candidate = candidate.drop(columns=["Rates"])
    elif case == "reordered":
        candidate = candidate[["Rates", "Equity"]]
    elif case == "range_index":
        candidate.index = pd.RangeIndex(len(candidate))
    elif case == "duplicate_index":
        candidate.index = pd.DatetimeIndex([candidate.index[0], candidate.index[0]])
    elif case == "unsorted_index":
        candidate = candidate.iloc[::-1]
    elif case == "timezone":
        candidate.index = candidate.index.tz_localize("UTC")
    elif case == "at_cutoff":
        candidate.index = pd.DatetimeIndex([x.index[-1], target.index[0]])
    elif case == "nonfinite":
        candidate.iloc[0, 0] = np.nan
    elif case == "bad_alpha_span":
        kwargs["alpha_span"] = 0.0
    with pytest.raises(error_type, match=message):
        model.nowcast(candidate, **kwargs)


def test_fit_data_validation_for_nowcast():
    """Incomplete fitted factors and final responses are never nowcastable."""
    x, y, target = _panel()
    incomplete_x = x.copy(deep=True)
    incomplete_x.iloc[5, 0] = np.nan
    with pytest.raises(ValueError, match="complete fitted factor rows"):
        _fit(incomplete_x, y).nowcast(target)

    incomplete_y = y.copy(deep=True)
    incomplete_y.iloc[-1, 0] = np.nan
    with pytest.raises(ValueError, match="fully observed final response row"):
        _fit(x, incomplete_y).nowcast(target)


def test_appended_target_with_missing_response_is_rejected():
    """A future factor row cannot be smuggled into fit with a blank response."""
    x, y, target = _panel()
    x_with_target = pd.concat([x, target.iloc[[0]]])
    missing_response = pd.DataFrame(np.nan, index=target.index[:1], columns=y.columns)
    y_with_target = pd.concat([y, missing_response])
    next_target = target.iloc[[1]]
    with pytest.raises(ValueError, match="fully observed final response row"):
        _fit(x_with_target, y_with_target).nowcast(next_target)


@pytest.mark.parametrize("case", ["range", "duplicate", "unsorted"])
def test_fit_index_validation(case):
    """Fit snapshots need a sorted, unique DatetimeIndex for a dated cutoff."""
    x, y, target = _panel()
    if case == "range":
        x.index = pd.RangeIndex(len(x))
        y.index = x.index
        message = "fitted data must have a DatetimeIndex"
        error_type = TypeError
    elif case == "duplicate":
        index = x.index.to_list()
        index[1] = index[0]
        x.index = pd.DatetimeIndex(index)
        y.index = x.index
        message = "sorted and unique"
        error_type = ValueError
    else:
        order = list(range(len(x)))
        order[5], order[6] = order[6], order[5]
        x = x.iloc[order]
        y = y.iloc[order]
        message = "sorted and unique"
        error_type = ValueError
    with pytest.raises(error_type, match=message):
        _fit(x, y).nowcast(target)


@pytest.mark.parametrize("case", ["beta", "infinite_residual", "terminal_residual"])
def test_corrupt_fitted_state_is_rejected(case):
    """Non-finite fitted coefficients or residual state fail closed."""
    x, y, target = _panel()
    model = _fit(x, y)
    if case == "beta":
        model.coef_.iloc[0, 0] = np.nan
        message = "finite fitted betas"
    elif case == "infinite_residual":
        model.nowcast_residuals_.iloc[0, 0] = np.inf
        message = "cannot contain infinite"
    else:
        model.nowcast_residuals_.iloc[-1, 0] = np.nan
        message = "finite terminal residuals"
    with pytest.raises(ValueError, match=message):
        model.nowcast(target)


def test_negative_solver_r2_is_copied_without_clipping():
    """Nowcast diagnostics preserve a negative fitted solver R-squared."""
    x, y, target = _panel()
    model = _fit(x, y)
    model.estimation_result_.r2[0] = -0.25
    result = model.nowcast(target)
    assert result.diagnostics.loc["Manager", "factorlasso_fit_r2_ewma_demeaned"] == -0.25


def test_result_is_frozen_and_fitted_state_is_not_a_parameter():
    """The result is frozen and nowcast provenance is fitted-only state."""
    x, y, target = _panel()
    model = _fit(x, y)
    result = model.nowcast(target)
    assert isinstance(result, LassoNowcastResult)
    with pytest.raises(FrozenInstanceError):
        result.prediction = result.prediction.copy()

    fitted_names = {
        "fit_demeaned_",
        "nowcast_residuals_",
        "nowcast_factors_complete_",
        "nowcast_final_response_complete_",
    }
    assert fitted_names.isdisjoint(inspect.signature(LassoModel).parameters)
    assert fitted_names.isdisjoint(model.get_params())
    copied = model.copy()
    for name in fitted_names:
        assert getattr(copied, name) is None
    with pytest.raises(RuntimeError, match="not fitted"):
        copied.nowcast(target)


def test_sklearn_clone_excludes_nowcast_fitted_state():
    """Scikit-learn cloning keeps the estimator specification unfitted."""
    pytest.importorskip("sklearn", exc_type=ImportError)
    from sklearn.base import clone

    x, y, target = _panel()
    model = _fit(x, y)
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()
    assert cloned.nowcast_residuals_ is None
    with pytest.raises(RuntimeError, match="not fitted"):
        cloned.nowcast(target)


def test_lambda_path_retains_one_independent_residual_snapshot_per_model():
    """Path clones share the contract while retaining only one T x N snapshot."""
    x, y, target = _panel()
    groups = pd.Series(["Liquid", "Private"], index=y.columns)
    models = LassoModel(
        model_type=LassoModelType.GROUP_LASSO,
        group_data=groups,
        span=8.0,
        warmup_period=None,
    ).fit_reg_lambda_path(x=x, y=y, reg_lambdas=[1e-5, 1e-4])

    retained_bytes = []
    for model in models:
        assert model.fit_demeaned_ is True
        expected_residuals = y - x @ model.coef_.T
        pd.testing.assert_frame_equal(model.nowcast_residuals_, expected_residuals)
        result = model.nowcast(target)
        pd.testing.assert_frame_equal(
            result.prediction,
            result.factor_component.add(result.stat_alpha, axis="columns"),
        )
        full_panel_buffers = {
            name: value
            for name, value in vars(model).items()
            if isinstance(value, (pd.DataFrame, np.ndarray))
            and value.shape in {x.shape, y.shape}
        }
        assert set(full_panel_buffers) == {"x_", "y_", "nowcast_residuals_"}
        assert full_panel_buffers["x_"] is x
        assert full_panel_buffers["y_"] is y
        assert full_panel_buffers["nowcast_residuals_"] is model.nowcast_residuals_
        assert not np.shares_memory(
            model.nowcast_residuals_.to_numpy(), x.to_numpy()
        )
        assert not np.shares_memory(
            model.nowcast_residuals_.to_numpy(), y.to_numpy()
        )
        retained_bytes.append(
            int(model.nowcast_residuals_.memory_usage(index=True, deep=True).sum())
        )
    assert all(size >= y.size * np.dtype(float).itemsize for size in retained_bytes)
    assert not np.shares_memory(
        models[0].nowcast_residuals_.to_numpy(),
        models[1].nowcast_residuals_.to_numpy(),
    )
