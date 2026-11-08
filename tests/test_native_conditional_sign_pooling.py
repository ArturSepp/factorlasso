"""Native conditional gates match the isolated research adapter, not new penalties."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import factorlasso as fl
from factorlasso.linear_model import _preparation as preparation
from factorlasso.priors._signs import _sign_observation_weights


def panel():
    """Correlated factors and asynchronous responses expose marginal sign confounding."""
    rng = np.random.default_rng(2026110307)
    dates = pd.date_range("2000-01-31", periods=150, freq="ME")
    values = rng.normal(size=(150, 3))
    values[:, 1] = 0.9 * values[:, 0] + 0.35 * values[:, 1]
    x = pd.DataFrame(values, index=dates, columns=["market", "credit", "Fx"])
    beta = np.array([[0.9, -0.6, 0.1], [0.8, -0.5, 0], [0.7, -0.4, 0.1], [1.0, -0.7, 0]])
    y = pd.DataFrame(
        values @ beta.T + 0.06 * rng.normal(size=(150, 4)),
        index=dates,
        columns=["a", "b", "c", "d"],
    )
    for j in range(4):
        y.iloc[np.arange(len(y)) % 2 == j % 2, j] = np.nan
    groups = pd.Series(["g1", "g1", "g2", "g2"], index=y.columns)
    return x, y, groups


def settings(mode, groups, **overrides):
    """Use stable conic fits, the production gate and deliberately active adaptive weights."""
    options = dict(
        model_type=mode,
        solver="CLARABEL",
        reg_lambda=1e-3,
        span=36,
        warmup_period=0,
        auto_sign_constraints=True,
        auto_sign_use_fit_span=True,
        auto_sign_variance="date",
        auto_sign_threshold_t=1.0,
        auto_sign_adaptive_weights=True,
        auto_sign_adaptive_floor=0.1,
    )
    if mode == fl.LassoModelType.GROUP_LASSO:
        options["group_data"] = groups
    options.update(overrides)
    return options


def test_generic_default_stays_ols_sign_guided():
    """ROSAA's policy must not become a FactorLasso-wide default change.

    The default rule is the OLS-sign-guided Lasso (pooled single-factor OLS signs), spelled
    ``'marginal'`` before 1.0.9.
    """
    model = fl.LassoModel()
    assert model.auto_sign_method == "ols_sign_guided"
    assert model.auto_sign_on_unidentified == "raise"


@pytest.mark.parametrize(
    "name,value", [("auto_sign_method", "unknown"), ("auto_sign_on_unidentified", "marginal")]
)
def test_new_options_validate_at_construction_and_after_set_params(name, value):
    """Unsupported evidence/failure policies cannot slip through estimator mutation."""
    with pytest.raises(ValueError, match=name):
        fl.LassoModel(**{name: value})
    x, y, _ = panel()
    model = fl.LassoModel().set_params(**{name: value})
    with pytest.raises(ValueError, match=name):
        model.fit(x, y)


@pytest.mark.parametrize(
    "mode",
    [
        fl.LassoModelType.LASSO,
        fl.LassoModelType.GROUP_LASSO,
        fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
        fl.LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
    ],
)
def test_native_conditional_matches_research_and_keeps_marginal_adaptive_weights(mode):
    """An independently wired public-gate adapter gives the same signs, solve and penalties."""
    x, y, groups = panel()
    external = (
        {"external_clusters": groups}
        if mode
        in (
            fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
            fl.LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
        )
        else {}
    )
    signs = pd.DataFrame(np.nan, index=y.columns, columns=x.columns)
    signs.loc["a", "market"] = 1.0
    options = settings(
        mode,
        groups,
        apply_ols_prior=True,
        factors_beta_loading_signs=signs,
        auto_sign_excluded_factors=("Fx",),
    )
    marginal = fl.LassoModel(**options).fit(x, y, **external)
    # ROSAA passes the pre-1.0.9 spelling; the fitted attributes report the canonical names.
    native = fl.LassoModel(**options, auto_sign_method="conditional").fit(x, y, **external)
    original_signs = preparation.automatic_signs
    original_adaptive = preparation.adaptive_penalty_weights
    context = {}

    def reference_gate(x, x_np, y_np, mask, clusters, span, threshold, variance):
        """Reproduce the frozen empirical adapter without calling the native new helper."""
        context["legacy"] = original_signs(x, x_np, y_np, mask, clusters, span, threshold, variance)
        offset = len(x) - len(x_np)
        xx = pd.DataFrame(
            np.where(x.iloc[offset:].notna(), x_np, np.nan),
            index=x.index[offset:],
            columns=x.columns,
        )
        yy = pd.DataFrame(np.where(mask > 0, y_np, np.nan), index=xx.index, columns=y.columns)
        labels = (
            pd.Series(np.arange(len(y.columns)), index=y.columns)
            if clusters is None
            else clusters.reindex(y.columns)
        )
        weights = pd.Series(_sign_observation_weights(len(xx), span), index=xx.index)
        gate = fl.compute_ragged_pooled_sign_gate(
            xx, yy, labels, weights=weights, threshold=threshold, action="exclude"
        )
        support = gate.diagnostics.loc[labels.to_numpy()]
        return (
            gate.restrictions.to_numpy(),
            gate.slopes.to_numpy(),
            dict(
                t_stats=gate.t_stats.to_numpy(),
                effective_n=np.repeat(
                    support.effective_n.to_numpy()[:, None], len(x.columns), axis=1
                ),
                n_obs=np.repeat(
                    support.response_observations.to_numpy()[:, None], len(x.columns), axis=1
                ),
            ),
        )

    def reference_adaptive(model, signs, slopes, clusters, is_lasso):
        """Freeze both marginal zero gates and marginal slope magnitudes, as in the sweep."""
        old = context["legacy"]
        return original_adaptive(model, old[0], old[1], clusters, is_lasso)

    with (
        patch.object(preparation, "automatic_signs", reference_gate),
        patch.object(preparation, "adaptive_penalty_weights", reference_adaptive),
    ):
        reference = fl.LassoModel(**options).fit(x, y, **external)
    for name in (
        "detected_signs_",
        "sign_slopes_",
        "sign_t_stats_",
        "derived_signs_",
        "sign_penalty_weights_",
        "ols_beta_prior_",
        "effective_beta_prior_",
    ):
        pd.testing.assert_frame_equal(getattr(native, name), getattr(reference, name))
    np.testing.assert_allclose(native.coef_, reference.coef_, atol=1e-10, rtol=1e-9)
    pd.testing.assert_frame_equal(native.sign_penalty_weights_, marginal.sign_penalty_weights_)
    np.testing.assert_equal(native.sign_block_weights_, marginal.sign_block_weights_)
    pd.testing.assert_frame_equal(native.adaptive_detected_signs_, marginal.detected_signs_)
    pd.testing.assert_frame_equal(native.adaptive_sign_slopes_, marginal.sign_slopes_)
    assert native.auto_sign_method_ == "sign_guided"
    assert native.adaptive_sign_method_ == "ols_sign_guided"
    assert marginal.auto_sign_method_ == "ols_sign_guided"
    assert native.sign_group_diagnostics_.status.eq("ok").all()
    assert not native.detected_signs_.equals(marginal.detected_signs_)


def test_unidentified_group_raises_or_explicitly_releases_without_marginal_fallback():
    """A rank failure is not evidence for zero restrictions or old marginal signs."""
    x, y, groups = panel()
    x["credit"] = x["market"]
    options = settings(fl.LassoModelType.GROUP_LASSO, groups, auto_sign_method="conditional")
    with pytest.raises(ValueError, match="unidentified"):
        fl.LassoModel(**options).fit(x, y)
    with pytest.warns(UserWarning, match="no marginal fallback"):
        model = fl.LassoModel(**options, auto_sign_on_unidentified="release").fit(x, y)
    assert model.detected_signs_.isna().all().all()
    assert model.derived_signs_.isna().all().all()
    assert model.sign_group_diagnostics_.status.ne("ok").all()
    assert np.isfinite(model.coef_.to_numpy()).all()


@pytest.mark.parametrize("threshold", [None, 0.0, np.inf])
def test_existing_threshold_endpoints_keep_their_meaning(threshold):
    """None disables thresholding and positive infinity excludes identified directions."""
    x, y, groups = panel()
    options = settings(
        fl.LassoModelType.GROUP_LASSO,
        groups,
        auto_sign_method="conditional",
        auto_sign_threshold_t=threshold,
    )
    model = fl.LassoModel(**options).fit(x, y)
    if np.isinf(threshold) if threshold is not None else False:
        assert model.detected_signs_.eq(0).all().all()
    else:
        np.testing.assert_array_equal(model.detected_signs_, np.sign(model.sign_slopes_))


def test_path_and_refits_own_and_reset_conditional_diagnostics():
    """Path caching preserves the new state and refits cannot leak conditional evidence."""
    x, y, groups = panel()
    model = fl.LassoModel(
        **settings(fl.LassoModelType.GROUP_LASSO, groups, auto_sign_method="conditional")
    )
    path = model.fit_reg_lambda_path(x, y, reg_lambdas=[1e-3, 2e-3])
    assert model.sign_group_diagnostics_ is None
    for fitted in path:
        fresh = model.copy({"reg_lambda": fitted.reg_lambda}).fit(x, y)
        pd.testing.assert_frame_equal(fitted.sign_group_diagnostics_, fresh.sign_group_diagnostics_)
        pd.testing.assert_frame_equal(fitted.adaptive_sign_slopes_, fresh.adaptive_sign_slopes_)
        np.testing.assert_allclose(fitted.coef_, fresh.coef_, atol=2e-6, rtol=2e-5)
    model.fit(x, y)
    model.set_params(auto_sign_method="ols_sign_guided").fit(x, y)
    assert model.sign_group_diagnostics_ is None
    assert model.auto_sign_method_ == "ols_sign_guided"
    model.set_params(auto_sign_constraints=False).fit(x, y)
    assert model.auto_sign_method_ is None
    assert model.adaptive_sign_method_ is None
    assert model.adaptive_sign_slopes_ is None
