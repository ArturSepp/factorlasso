"""Independent numerical contracts for the optional ridge estimator mode."""

import numpy as np
import pandas as pd
import pytest

from factorlasso import (
    CurrentFactorCovarData, LassoModel, LassoModelCV, LassoModelType, VarianceColumns,
)


def _panel(*, missing=False):
    """Correlated factors and signed responses with reproducible nonzero means."""
    rng = np.random.default_rng(291026)
    x = rng.normal(size=(84, 3)) + [0.4, -0.3, 0.8]
    x[:, 1] += 0.9 * x[:, 0]
    beta = np.array([[1.3, -0.7, 0.2], [-0.5, 0.4, 0.8]])
    y = x @ beta.T + [0.6, -0.9] + rng.normal(scale=0.15, size=(84, 2))
    if missing:
        y[:23, 1] = np.nan
        y[38:43, 0] = np.nan
        x[29] = np.nan
    dates = pd.date_range("2010-01-31", periods=len(x), freq="ME")
    return (pd.DataFrame(x, index=dates, columns=["f1", "f2", "f3"]),
            pd.DataFrame(y, index=dates, columns=["asset_a", "asset_b"]))


def _model(**options):
    """Disable warmup and demeaning explicitly when comparing solver objectives."""
    kwargs = dict(model_type=LassoModelType.RIDGE, demean=False, warmup_period=None)
    kwargs.update(options)
    return LassoModel(**kwargs)


def _augmented_reference(x, y, penalty, span=None, normalization="sample"):
    """Solve each response using weighted augmented least squares, without normal equations."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t, k = x.shape
    valid = np.isfinite(y) & np.isfinite(x).any(axis=1)[:, None]
    decay = 1.0 if span is None else 1.0 - 2.0 / (span + 1.0)
    row_weights = decay ** np.arange(t - 1, -1, -1)
    coefficients = np.zeros((y.shape[1], k))
    for col in range(y.shape[1]):
        mask = valid[:, col]
        if not mask.any() or np.isposinf(penalty):
            continue
        weights = row_weights[mask]
        denominator = t if normalization == "sample" else weights.sum()
        design = np.nan_to_num(x[mask]) * np.sqrt(weights / denominator)[:, None]
        response = y[mask, col] * np.sqrt(weights / denominator)
        augmented_x = np.vstack([design, np.sqrt(penalty) * np.eye(k)])
        augmented_y = np.concatenate([response, np.zeros(k)])
        coefficients[col] = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
    return coefficients


def test_ridge_is_additive_and_defaults_are_unchanged():
    """The opt-in mode preserves historical enum values and estimator defaults."""
    assert LassoModelType.RIDGE.value == 8
    assert LassoModelType.LASSO.value == 1
    assert LassoModelType.COOPERATIVE_CLUSTER_GROUP_LASSO.value == 7
    model = LassoModel()
    assert model.model_type == LassoModelType.LASSO
    assert model.reg_lambda == 1e-5
    assert model.loss_normalization == "sample"
    assert model.auto_sign_constraints is False
    assert np.isfinite(model.reg_lambda)


@pytest.mark.parametrize("penalty", [0.0, 0.25, 5.0])
@pytest.mark.parametrize("span", [None, 17.0])
@pytest.mark.parametrize("normalization", ["sample", "weight_sum"])
def test_ragged_ridge_matches_augmented_least_squares(penalty, span, normalization):
    """The full estimator preserves each response's mask and the declared EW loss mass."""
    x, y = _panel(missing=True)
    model = _model(reg_lambda=penalty, span=span, loss_normalization=normalization).fit(x, y)
    expected = _augmented_reference(x, y, penalty, span, normalization)
    np.testing.assert_allclose(model.coef_, expected, rtol=2e-7, atol=2e-8)
    assert model.coef_.index.equals(y.columns)
    assert model.coef_.columns.equals(x.columns)
    assert model.clusters_ is None
    assert model.derived_signs_ is None
    assert model.sign_penalty_weights_ is None
    valid = y.notna().to_numpy() & x.notna().any(axis=1).to_numpy()[:, None]
    decay = 1.0 if span is None else 1 - 2 / (span + 1)
    mass = (valid * decay ** np.arange(len(x) - 1, -1, -1)[:, None]).sum(axis=0)
    np.testing.assert_allclose(model.loss_weight_mass_, mass, rtol=1e-13)
    expected_denominator = np.full(y.shape[1], len(x)) if normalization == "sample" else mass
    np.testing.assert_allclose(model.loss_denominator_, expected_denominator, rtol=1e-13)


@pytest.mark.parametrize("span", [None, 17.0])
def test_sklearn_reference_requires_correct_penalty_scaling(span):
    """An independent library agrees on identical transformed arrays and unnormalised loss."""
    from sklearn.linear_model import Ridge

    x, y = _panel()
    penalty = 0.35
    model = _model(reg_lambda=penalty, span=span).fit(x, y)
    decay = 1.0 if span is None else 1 - 2 / (span + 1)
    weights = decay ** np.arange(len(x) - 1, -1, -1)
    expected = Ridge(alpha=len(x) * penalty, fit_intercept=False, solver="svd").fit(
        x, y, sample_weight=weights)
    np.testing.assert_allclose(model.coef_, expected.coef_, rtol=2e-7, atol=2e-8)
    incorrectly_scaled = Ridge(alpha=penalty, fit_intercept=False, solver="svd").fit(
        x, y, sample_weight=weights)
    assert np.max(np.abs(model.coef_.to_numpy() - incorrectly_scaled.coef_)) > 0.05


@pytest.mark.parametrize("span", [None, 17.0])
def test_demeaning_and_original_unit_predictions(span):
    """EW recursive demeaning is distinguished from an ordinary weighted intercept fit."""
    x, y = _panel()
    penalty = 0.2
    model = _model(reg_lambda=penalty, span=span, demean=True).fit(x, y)
    if span is None:
        transformed_x, transformed_y = x - x.mean(), y - y.mean()
    else:
        transformed_x = (x - x.ewm(span=span, adjust=False).mean()).iloc[1:]
        transformed_y = (y - y.ewm(span=span, adjust=False).mean()).iloc[1:]
    expected = _augmented_reference(transformed_x, transformed_y, penalty, span)
    np.testing.assert_allclose(model.coef_, expected, rtol=2e-7, atol=2e-8)
    decay = 1.0 if span is None else 1 - 2 / (span + 1)
    weights = decay ** np.arange(len(x) - 1, -1, -1)
    intercept = np.average(y, weights=weights, axis=0) - expected @ np.average(
        x, weights=weights, axis=0)
    np.testing.assert_allclose(model.alpha_const_, intercept, rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(model.predict(x), x.to_numpy() @ expected.T + intercept,
                               rtol=2e-7, atol=2e-8)
    assert model.n_loss_rows_ == len(transformed_x)


@pytest.mark.parametrize("penalty", [0.0, 0.1, np.inf])
def test_rank_deficient_design_and_zero_penalty_minimum_norm(penalty):
    """Exactly redundant factors still admit a deterministic minimum-norm zero limit."""
    x, y = _panel()
    x["f3"] = 2 * x["f1"]
    model = _model(reg_lambda=penalty).fit(x, y)
    expected = _augmented_reference(x, y, penalty)
    np.testing.assert_allclose(model.coef_, expected, rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(model.predict(x), x.to_numpy() @ expected.T,
                               rtol=2e-7, atol=2e-8)
    if np.isinf(penalty):
        np.testing.assert_array_equal(model.coef_, np.zeros((2, 3)))


@pytest.mark.parametrize("normalization", ["sample", "weight_sum"])
def test_empty_response_has_zero_loading_and_loss_mass(normalization):
    """No-response data do not enter the loss or create NaN coefficients for observed assets."""
    x, y = _panel()
    y["unobserved"] = np.nan
    model = _model(reg_lambda=0.3, loss_normalization=normalization).fit(x, y)
    np.testing.assert_allclose(model.coef_, _augmented_reference(
        x, y, 0.3, normalization=normalization), atol=2e-8, rtol=2e-7)
    np.testing.assert_allclose(model.coef_.loc["unobserved"], 0, atol=1e-12)
    assert model.loss_weight_mass_.loc["unobserved"] == 0
    assert np.isnan(model.alpha_const_.loc["unobserved"])


def test_single_response_ridge_does_not_reduce_to_lasso():
    """One response keeps the squared-L2 penalty and ndarray interface."""
    x, y = _panel()
    model = _model(reg_lambda=0.4).fit(x.to_numpy(), y.iloc[:, 0].to_numpy())
    expected = _augmented_reference(x, y.iloc[:, :1], 0.4)
    np.testing.assert_allclose(model.coef_, expected, atol=2e-8, rtol=2e-7)
    assert model.coef_.shape == (1, 3)


def test_large_penalty_and_infinite_limit_keep_intercept():
    """Strong shrinkage targets zero slopes, while the unpenalised mean survives demeaning."""
    x, y = _panel()
    strong = _model(reg_lambda=1e6, demean=True).fit(x, y)
    limit = _model(reg_lambda=np.inf, demean=True).fit(x, y)
    np.testing.assert_allclose(strong.coef_, 0, atol=3e-6)
    np.testing.assert_array_equal(limit.coef_, np.zeros((2, 3)))
    np.testing.assert_allclose(limit.alpha_const_, y.mean(), atol=1e-14)
    np.testing.assert_allclose(limit.predict(x), np.tile(y.mean(), (len(x), 1)), atol=1e-14)


def test_weight_sum_prefix_invariance():
    """Prepending missing response history does not alter valid EW mass or shrinkage."""
    x, y = _panel(missing=True)
    extra = pd.date_range(end=x.index[0] - pd.offsets.MonthEnd(1), periods=25, freq="ME")
    xp = pd.concat([pd.DataFrame(0.3, index=extra, columns=x.columns), x])
    yp = y.reindex(xp.index)
    first = _model(reg_lambda=0.4, span=17, loss_normalization="weight_sum").fit(x, y)
    second = _model(reg_lambda=0.4, span=17, loss_normalization="weight_sum").fit(xp, yp)
    np.testing.assert_allclose(first.coef_, second.coef_, rtol=2e-7, atol=2e-8)


@pytest.mark.parametrize("penalty", [-0.1, np.nan, -np.inf, True])
def test_invalid_penalties_rejected_at_construction_and_refit(penalty):
    """A changed estimator parameter cannot bypass ridge-specific validation."""
    with pytest.raises(ValueError):
        _model(reg_lambda=penalty)
    x, y = _panel()
    model = _model(reg_lambda=0.3).fit(x, y)
    with pytest.raises(ValueError):
        model.set_params(reg_lambda=penalty).fit(x, y)


@pytest.mark.parametrize("options", [
    {"nonneg": True},
    {"auto_sign_constraints": True},
    {"auto_sign_adaptive_weights": True},
    {"apply_ols_prior": True},
    {"factors_beta_loading_signs": pd.DataFrame([[1]])},
    {"factors_beta_prior": pd.DataFrame([[1]])},
    {"group_data": pd.Series([1, 1], index=["asset_a", "asset_b"])},
    {"group_penalty": "yuan_lin"},
    {"l1_weight": 0.5},
    {"n_clusters": 2},
    {"excluded_factors": {"asset_a": ["f1"]}},
    {"factor_for_prior": {"asset_a": "f1"}},
])
def test_unsupported_restrictions_and_penalties_rejected(options):
    """Pure ridge must not silently accept another method's active assumptions."""
    with pytest.raises(ValueError):
        _model(**options)
    x, y = _panel()
    model = _model()
    with pytest.raises(ValueError):
        model.set_params(**options).fit(x, y)


def test_ridge_rejects_external_groups_at_fit():
    """A supplied response partition does not silently change a nongrouped baseline."""
    x, y = _panel()
    with pytest.raises(ValueError, match="external_clusters"):
        _model().fit(x, y, external_clusters=pd.Series(1, index=y.columns))


def test_path_and_clone_preserve_configuration_without_mutating_template():
    """Repeated full-fit path entries own their state and preserve the ridge mode."""
    from sklearn.base import clone

    x, y = _panel()
    template = _model(span=17, loss_normalization="weight_sum")
    penalties = [0.0, 0.3, np.inf]
    fitted = template.fit_reg_lambda_path(x, y, penalties)
    assert template.coef_ is None
    assert clone(template).model_type == LassoModelType.RIDGE
    for penalty, model in zip(penalties, fitted):
        expected = _augmented_reference(x, y, penalty, 17, "weight_sum")
        np.testing.assert_allclose(model.coef_, expected, rtol=2e-7, atol=2e-8)
    fitted[0].coef_.iloc[0, 0] += 5
    np.testing.assert_allclose(fitted[1].coef_, _augmented_reference(
        x, y, 0.3, 17, "weight_sum"), rtol=2e-7, atol=2e-8)


def test_chronological_cv_scores_match_independent_prefix_fits():
    """Selection uses expanding training prefixes, with the public mean-R2 criterion unchanged."""
    x, y = _panel()
    penalties = [0.0, 0.1, 3.0]
    cv = LassoModelCV(base_model=_model(), lambdas=penalties, n_splits=3,
                      use_lambda_path=True).fit(x, y)
    expected = np.zeros((len(penalties), 3))
    block = len(x) // 4
    for fold in range(3):
        stop = (fold + 1) * block
        held_x = x.iloc[stop:stop + block].to_numpy()
        held_y = y.iloc[stop:stop + block].to_numpy()
        for row, penalty in enumerate(penalties):
            beta = _augmented_reference(x.iloc[:stop], y.iloc[:stop], penalty)
            error = np.square(held_y - held_x @ beta.T).sum(axis=0)
            total = np.square(held_y - held_y.mean(axis=0)).sum(axis=0)
            expected[row, fold] = np.mean(1 - error / total)
    np.testing.assert_allclose(cv.cv_scores_, expected, rtol=2e-7, atol=2e-8)
    assert cv.best_lambda_ == penalties[np.argmax(expected.mean(axis=1))]
    assert cv.best_model_.model_type == LassoModelType.RIDGE


def test_covariance_snapshot_accepts_ridge_results_in_original_units():
    """Existing downstream assembly consumes ridge coefficients and residual diagnostics."""
    x, y = _panel()
    penalty = 0.3
    model = _model(reg_lambda=penalty).fit(x, y)
    beta = _augmented_reference(x, y, penalty)
    residual_variance = np.square(y.to_numpy() - x.to_numpy() @ beta.T).mean(axis=0)
    snapshot = CurrentFactorCovarData(
        x_covar=x.cov(), y_betas=model.coef_,
        y_variances=pd.DataFrame(
            {VarianceColumns.RESIDUAL_VARS: model.estimation_result_.ss_res}, index=y.columns),
    )
    expected = beta @ x.cov().to_numpy() @ beta.T + np.diag(residual_variance)
    np.testing.assert_allclose(snapshot.get_y_covar(), expected, rtol=2e-7, atol=2e-8)


@pytest.mark.parametrize("mode", ["RIDGE", "LASSO"])
@pytest.mark.parametrize("normalization", ["sample", "weight_sum"])
def test_span_one_uses_only_last_observation(mode, normalization):
    """Zero EW decay has an exact scalar solution, including a missing final response."""
    x = np.linspace(1.0, 3.0, 12)[:, None]
    y = np.column_stack([0.7 * x[:, 0] + 0.1, -1.2 * x[:, 0]])
    y[-1, 1] = np.nan
    penalty = 0.25
    denominator = len(x) if normalization == "sample" else 1.0
    final_x, final_y = x[-1, 0], y[-1, 0]
    if mode == "RIDGE":
        expected = final_x * final_y / (final_x**2 + penalty * denominator)
    else:
        cross = final_x * final_y
        expected = np.sign(cross) * max(abs(cross) - penalty * denominator / 2, 0) / final_x**2
    model = LassoModel(
        model_type=LassoModelType[mode], reg_lambda=penalty, span=1,
        demean=False, warmup_period=None, loss_normalization=normalization,
    ).fit(x, y)
    np.testing.assert_allclose(model.coef_.to_numpy()[:, 0], [expected, 0.0], atol=2e-7)
    np.testing.assert_array_equal(model.loss_weight_mass_, [1.0, 0.0])
    expected_denominators = [len(x), len(x)] if normalization == "sample" else [1.0, 0.0]
    np.testing.assert_array_equal(model.loss_denominator_, expected_denominators)
