"""Regression contracts for optional per-response loss normalization."""

import numpy as np
import pandas as pd
import pytest

from factorlasso import LassoModel, LassoModelType, LassoModelCV
from factorlasso import lasso_estimator as le


def panel():
    """Return unequal-history synthetic responses on a fixed date grid."""
    rng = np.random.default_rng(914)
    x = rng.normal(0, 0.035, (120, 3))
    y = x @ np.array([[0.8, 0.2, 0], [0.4, -0.3, 0.1]]).T
    y += rng.normal(0, 0.003, y.shape)
    y[:45, 1] = np.nan
    y[76:81, 0] = np.nan
    # Work in percent units so public solver absolute tolerances do not dominate
    # near-zero coefficients. Penalties below use the matching squared units.
    return 100 * x, 100 * y


def solve(kind, x, y, **kwargs):
    """Exercise each penalty geometry through its supported solver."""
    if kind == "lasso":
        return le.solve_lasso_cvx_problem(x, y, **kwargs)
    groups = np.ones((y.shape[1], 1))
    if kind == "cooperative":
        return le.solve_cooperative_group_lasso_cvx_problem(x, y, groups, **kwargs)
    return le.solve_group_lasso_cvx_problem(x, y, groups, block_mode=kind, l1_weight=0.2, **kwargs)


@pytest.mark.parametrize("kind", ["lasso", "row", "cluster_factor", "cooperative"])
@pytest.mark.parametrize("span", [None, 60])
def test_missing_prefix_does_not_change_normalized_fit(kind, span):
    """Extra factor history without any response is not extra shrinkage."""
    x, y = panel()
    options = dict(reg_lambda=2.0, span=span, loss_normalization="weight_sum")
    a = solve(kind, x, y, **options)
    rng = np.random.default_rng(62)
    xp = np.vstack([rng.normal(0, 4, (180, 3)), x])
    yp = np.vstack([np.full((180, 2), np.nan), y])
    b = solve(kind, xp, yp, **options)
    np.testing.assert_allclose(a.estimated_beta, b.estimated_beta, atol=2e-5, rtol=0)


@pytest.mark.parametrize("kind", ["lasso", "row", "cluster_factor", "cooperative"])
@pytest.mark.parametrize("span", [None, 60])
def test_common_mass_lambda_conversion_preserves_fit(kind, span):
    """Multiplying the entire objective preserves its optimum."""
    x, y = panel()
    y = np.nan_to_num(y)
    t = len(x)
    decay = 1 if span is None else 1 - 2 / (span + 1)
    mass = np.sum(decay ** np.arange(t - 1, -1, -1))
    old = solve(kind, x, y, reg_lambda=0.2, span=span)
    new = solve(kind, x, y, reg_lambda=0.2 * t / mass, span=span, loss_normalization="weight_sum")
    np.testing.assert_allclose(old.estimated_beta, new.estimated_beta, atol=3e-5, rtol=0)
    np.testing.assert_allclose(old.ss_res, new.ss_res, atol=0.002, rtol=0)


def test_normalized_lasso_matches_scalar_soft_threshold_reference():
    """Unequal masks use separate masses, verified without package weights."""
    x = np.linspace(-6, 5, 100)[:, None]
    y = np.column_stack([0.8 * x[:, 0], -0.5 * x[:, 0]])
    y[:75, 1] = np.nan
    y[45:49, 0] = np.nan
    lam = 1.0
    result = le.solve_lasso_cvx_problem(
        x, y, span=60, reg_lambda=lam, loss_normalization="weight_sum"
    )
    w = (1 - 2 / 61) ** np.arange(99, -1, -1)
    expected = []
    for col in range(2):
        valid = np.isfinite(y[:, col])
        a = np.sum(w[valid] * x[valid, 0] ** 2) / np.sum(w[valid])
        b = np.sum(w[valid] * x[valid, 0] * y[valid, col]) / np.sum(w[valid])
        expected.append(np.sign(b) * max(abs(b) - lam / 2, 0) / a)
    np.testing.assert_allclose(result.estimated_beta[:, 0], expected, atol=2e-5, rtol=0)


def test_loss_weight_rescaling_and_empty_response():
    """Zero mass remains zero and arbitrary weight units cancel."""
    residual = np.arange(30, dtype=float).reshape(10, 3) / 10
    weights = np.tile(np.linspace(0.2, 1, 10)[:, None], (1, 3))
    weights[:, 2] = 0
    a = le._weighted_squared_loss(residual, weights, 10, "weight_sum").value
    b = le._weighted_squared_loss(residual, 7 * weights, 10, "weight_sum").value
    reference = sum(np.average(residual[:, k] ** 2, weights=weights[:, k] ** 2) for k in range(2))
    assert a == pytest.approx(reference)
    assert b == pytest.approx(reference)


def model_panel():
    """Attach aligned dates and explicit stable groups."""
    x, y = panel()
    dates = pd.date_range("2010-01-31", periods=len(x), freq="ME")
    return pd.DataFrame(x, index=dates, columns=["f1", "f2", "f3"]), pd.DataFrame(
        y, index=dates, columns=["a", "b"]
    )


def make_model(**kwargs):
    """Use fixed groups to isolate loss scaling from cluster inference."""
    return LassoModel(
        model_type=LassoModelType.GROUP_LASSO,
        group_data=pd.Series(1, index=["a", "b"]),
        span=60,
        demean=False,
        loss_normalization="weight_sum",
        **kwargs,
    )


def test_single_path_clone_and_diagnostics():
    """The common path and standalone fits share objective and fitted metadata."""
    x, y = model_panel()
    model = make_model()
    assert model.get_params()["loss_normalization"] == "weight_sum"
    fitted = model.fit_reg_lambda_path(x, y, [1.0, 2.0])
    for value, path in zip([1.0, 2.0], fitted):
        single = make_model(reg_lambda=value).fit(x, y)
        np.testing.assert_allclose(path.coef_, single.coef_, atol=2e-5, rtol=0)
        np.testing.assert_allclose(path.loss_weight_mass_, single.loss_weight_mass_)
        np.testing.assert_allclose(path.loss_denominator_, single.loss_weight_mass_)
        assert path.n_loss_rows_ == 120
    from sklearn.base import clone

    assert clone(model).loss_normalization == "weight_sum"


def test_invalid_modes_and_unilasso_rejected():
    """Unsupported conventions fail rather than silently using another loss."""
    with pytest.raises(ValueError, match="loss_normalization"):
        LassoModel(loss_normalization="invalid")
    with pytest.raises(ValueError, match="UNILASSO"):
        LassoModel(model_type=LassoModelType.UNILASSO, loss_normalization="weight_sum")
    x, y = panel()
    with pytest.raises(ValueError, match="loss_normalization"):
        le.solve_lasso_cvx_problem(x, y, loss_normalization="bad")


def test_cv_single_and_path_keep_normalization():
    """Temporal folds and the final refit retain the requested convention."""
    x, y = model_panel()
    y = y.fillna(0.0)
    options = dict(lambdas=[0.2, 1.0], n_splits=3, base_model=make_model())
    single = LassoModelCV(**options).fit(x, y)
    path = LassoModelCV(**options, use_lambda_path=True).fit(x, y)
    assert (
        single.best_model_.loss_normalization == path.best_model_.loss_normalization == "weight_sum"
    )
    assert single.best_lambda_ == path.best_lambda_
    np.testing.assert_allclose(single.cv_scores_, path.cv_scores_, atol=2e-5, rtol=0)


def test_normalized_model_prefix_invariance_with_fixed_groups():
    """The estimator, beyond its direct solver, preserves the objective."""
    x, y = model_panel()
    prefix = pd.date_range(end=x.index[0] - pd.offsets.MonthEnd(1), periods=180, freq="ME")
    xp = pd.concat([pd.DataFrame(0.1, index=prefix, columns=x.columns), x])
    yp = y.reindex(xp.index)
    a = make_model(reg_lambda=1.0).fit(x, y)
    b = make_model(reg_lambda=1.0).fit(xp, yp)
    np.testing.assert_allclose(a.coef_, b.coef_, atol=2e-5, rtol=0)
    np.testing.assert_allclose(a.loss_weight_mass_, b.loss_weight_mass_, atol=1e-12)


def test_public_default_retains_sample_loss():
    """This opt-in revision does not silently switch archived public fits."""
    assert LassoModel().loss_normalization == "sample"
    assert LassoModel().solver == "CLARABEL"
    x, y = panel()
    a = solve("cluster_factor", x, y, reg_lambda=0.1, span=60)
    b = solve("cluster_factor", x, y, reg_lambda=0.1, span=60, loss_normalization="sample")
    np.testing.assert_array_equal(a.estimated_beta, b.estimated_beta)
