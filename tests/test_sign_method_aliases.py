"""The 1.0.9 sign-method names select the pre-1.0.9 rules exactly, and the old spellings still work.

``auto_sign_method='sign_guided'`` is the sign-guided Lasso (pooled joint-regression signs),
spelled ``'conditional'`` before 1.0.9; ``'ols_sign_guided'`` is the OLS-sign-guided Lasso (pooled
single-factor OLS signs), spelled ``'marginal'`` before 1.0.9 and still the default rule.
``compute_sign_guided_gate`` is ``compute_pooled_sign_gate`` under the method's name.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import factorlasso as fl
from factorlasso import inference
from factorlasso.linear_model._settings import (
    LEGACY_SIGN_METHODS, SIGN_METHODS, resolve_sign_method,
)


def panel():
    """Correlated factors and asynchronous responses, so the two rules give different signs."""
    rng = np.random.default_rng(2026110401)
    dates = pd.date_range("2000-01-31", periods=150, freq="ME")
    values = rng.normal(size=(150, 3))
    values[:, 1] = 0.9 * values[:, 0] + 0.35 * values[:, 1]
    x = pd.DataFrame(values, index=dates, columns=["market", "credit", "fx"])
    beta = np.array([[0.9, -0.6, 0.1], [0.8, -0.5, 0], [0.7, -0.4, 0.1], [1.0, -0.7, 0]])
    y = pd.DataFrame(values @ beta.T + 0.06 * rng.normal(size=(150, 4)),
                     index=dates, columns=["a", "b", "c", "d"])
    for j in range(4):
        y.iloc[np.arange(len(y)) % 2 == j % 2, j] = np.nan
    groups = pd.Series(["g1", "g1", "g2", "g2"], index=y.columns)
    return x, y, groups


def settings(groups, **overrides):
    options = dict(model_type=fl.LassoModelType.GROUP_LASSO, group_data=groups,
                   solver="CLARABEL", reg_lambda=1e-3, span=36, warmup_period=0,
                   auto_sign_constraints=True, auto_sign_use_fit_span=True,
                   auto_sign_variance="date", auto_sign_threshold_t=1.0,
                   auto_sign_adaptive_weights=True, auto_sign_adaptive_floor=0.1)
    options.update(overrides)
    return options


FITTED_FRAMES = ("derived_signs_", "detected_signs_", "sign_slopes_", "sign_penalty_weights_",
                 "adaptive_detected_signs_", "adaptive_sign_slopes_")


def test_resolve_sign_method_maps_legacy_spellings_and_rejects_the_rest():
    assert SIGN_METHODS == ("sign_guided", "ols_sign_guided")
    assert LEGACY_SIGN_METHODS == {"conditional": "sign_guided", "marginal": "ols_sign_guided"}
    for name in SIGN_METHODS:
        assert resolve_sign_method(name) == name
    for legacy, canonical in LEGACY_SIGN_METHODS.items():
        assert resolve_sign_method(legacy) == canonical
    for bad in ("unknown", "Sign_Guided", "", None, 1):
        with pytest.raises(ValueError, match="auto_sign_method"):
            resolve_sign_method(bad)


@pytest.mark.parametrize("name", ["sign_guided", "ols_sign_guided", "conditional", "marginal"])
def test_every_spelling_validates_at_construction_and_after_set_params(name):
    model = fl.LassoModel(auto_sign_method=name)
    assert model.auto_sign_method == name  # constructor parameters are stored unmodified
    mutated = fl.LassoModel().set_params(auto_sign_method=name)
    assert mutated.get_params()["auto_sign_method"] == name


def test_default_is_the_ols_sign_guided_rule_under_its_new_name():
    model = fl.LassoModel()
    assert model.auto_sign_method == "ols_sign_guided"
    assert model.get_params()["auto_sign_method"] == "ols_sign_guided"


@pytest.mark.parametrize(
    "legacy,canonical", [("conditional", "sign_guided"), ("marginal", "ols_sign_guided")]
)
def test_legacy_and_canonical_spellings_fit_identically_without_warnings(legacy, canonical):
    """Same signs, same penalties, same coefficients; the fitted attributes report the new name."""
    x, y, groups = panel()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        old = fl.LassoModel(**settings(groups, auto_sign_method=legacy)).fit(x, y)
        new = fl.LassoModel(**settings(groups, auto_sign_method=canonical)).fit(x, y)
    for attribute in FITTED_FRAMES:
        pd.testing.assert_frame_equal(getattr(old, attribute), getattr(new, attribute))
    np.testing.assert_array_equal(old.coef_.to_numpy(), new.coef_.to_numpy())
    assert old.auto_sign_method == legacy and new.auto_sign_method == canonical
    assert old.auto_sign_method_ == new.auto_sign_method_ == canonical
    assert old.adaptive_sign_method_ == new.adaptive_sign_method_ == "ols_sign_guided"
    if canonical == "sign_guided":
        pd.testing.assert_frame_equal(old.sign_group_diagnostics_, new.sign_group_diagnostics_)
    else:
        assert old.sign_group_diagnostics_ is None and new.sign_group_diagnostics_ is None


def test_the_two_rules_differ_on_the_panel():
    """The alias test above is not vacuous: the two rules give different automatic signs here."""
    x, y, groups = panel()
    guided = fl.LassoModel(**settings(groups, auto_sign_method="sign_guided")).fit(x, y)
    ols = fl.LassoModel(**settings(groups, auto_sign_method="ols_sign_guided")).fit(x, y)
    assert not guided.detected_signs_.equals(ols.detected_signs_)
    pd.testing.assert_frame_equal(guided.adaptive_detected_signs_, ols.detected_signs_)


def test_compute_sign_guided_gate_is_the_pooled_gate_under_the_methods_name():
    assert fl.compute_sign_guided_gate is fl.compute_pooled_sign_gate
    assert inference.compute_sign_guided_gate is fl.compute_pooled_sign_gate
    assert "compute_sign_guided_gate" in fl.__all__
    assert "compute_sign_guided_gate" in inference.__all__
    x, y, groups = panel()
    complete = y.ffill().bfill()
    weights = pd.Series(np.linspace(0.5, 1.5, len(x)), index=x.index)
    guided = fl.compute_sign_guided_gate(x, complete, groups, weights=weights, threshold=2.0)
    pooled = fl.compute_pooled_sign_gate(x, complete, groups, weights=weights, threshold=2.0)
    assert isinstance(guided, fl.PooledSignGateResult)
    pd.testing.assert_frame_equal(guided.restrictions, pooled.restrictions)
    pd.testing.assert_frame_equal(guided.t_stats, pooled.t_stats)
    assert guided.evidence == pooled.evidence == "conditional"
