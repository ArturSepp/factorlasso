"""Factor-specific auto-sign exclusions preserve all other estimation controls."""

from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from factorlasso import LassoModel, LassoModelType


def panel():
    """Correlated factors have positive marginal but negative partial slopes."""
    rng = np.random.default_rng(846)
    z = rng.normal(size=(320, 3))
    x = pd.DataFrame(
        {
            "Equity": z[:, 0],
            "Oil": 0.9 * z[:, 0] + 0.35 * z[:, 1],
            "Gold": 0.7 * z[:, 0] + 0.45 * z[:, 2],
        }
    )
    y = pd.DataFrame(
        {
            k: 0.9 * x.Equity - 0.35 * x.Oil - 0.2 * x.Gold + rng.normal(0, 0.002, len(x))
            for k in ["Stock", "Bond"]
        }
    )
    return x, y


@pytest.mark.parametrize(
    "kind",
    [
        LassoModelType.LASSO,
        LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
        LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
    ],
)
def test_exclusions_allow_negative_partial_slopes_and_preserve_weights(kind, monkeypatch):
    """Only selected solver signs change; pooling and penalty weights remain identical."""
    x, y = panel()
    captured = []
    original = LassoModel._prepare_fit

    def record(self, *args, **kwargs):
        """Capture the actual solver inputs, rather than reimplementing their formulas."""
        result = original(self, *args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(LassoModel, "_prepare_fit", record)
    base = LassoModel(
        model_type=kind,
        n_clusters=1,
        reg_lambda=1e-8,
        auto_sign_constraints=True,
        auto_sign_adaptive_weights=True,
        auto_sign_adaptive_floor=0.5,
        auto_sign_threshold_t=1.0,
        solver="CLARABEL",
    )
    base.fit(x, y, verbose=False)
    free = replace(base, auto_sign_excluded_factors=("Oil", "Gold")).fit(x, y, verbose=False)
    assert (base.derived_signs_[["Oil", "Gold"]] == 1).all().all()
    assert free.derived_signs_[["Oil", "Gold"]].isna().all().all()
    pd.testing.assert_series_equal(base.derived_signs_.Equity, free.derived_signs_.Equity)
    assert (free.coef_.Oil < -0.3).all() and (free.coef_.Gold < -0.15).all()
    np.testing.assert_allclose(
        free.coef_[["Equity", "Oil", "Gold"]],
        np.tile([0.9, -0.35, -0.2], (2, 1)),
        atol=0.005,
        rtol=0,
    )
    for attr in ["penalty_weights_np", "row_weights_np", "col_weights_np"]:
        np.testing.assert_equal(getattr(captured[0], attr), getattr(captured[1], attr))
    assert clone(free).auto_sign_excluded_factors == ("Oil", "Gold")


def test_explicit_signs_still_win_and_empty_exclusion_is_exact():
    """Excluding automatic signs does not bypass explicit asset restrictions."""
    x, y = panel()
    signs = pd.DataFrame(np.nan, index=y.columns, columns=x.columns)
    signs.loc["Stock", "Oil"] = 0.0
    base = LassoModel(
        auto_sign_constraints=True,
        factors_beta_loading_signs=signs,
        reg_lambda=1e-8,
        solver="CLARABEL",
    ).fit(x, y, verbose=False)
    empty = replace(base, auto_sign_excluded_factors=()).fit(x, y, verbose=False)
    np.testing.assert_array_equal(base.coef_, empty.coef_)
    free = replace(base, auto_sign_excluded_factors=("Oil", "Gold")).fit(x, y, verbose=False)
    assert free.derived_signs_.loc["Stock", "Oil"] == 0.0
    assert np.isnan(free.derived_signs_.loc["Bond", "Oil"])
    assert abs(free.coef_.loc["Stock", "Oil"]) < 1e-6


@pytest.mark.parametrize("bad", ["Oil", ("Oil", "Oil"), ("missing",), (3,)])
def test_invalid_exclusions_fail_closed(bad):
    """Reject ambiguous, duplicate and unavailable factor names at fit time."""
    x, y = panel()
    with pytest.raises(ValueError, match="auto_sign_excluded_factors"):
        LassoModel(auto_sign_constraints=True, auto_sign_excluded_factors=bad).fit(x, y)


def test_exclusion_bypasses_zero_gate_and_is_inert_without_auto_signs():
    """Weak-evidence gating is removed only on excluded factor columns."""
    x, y = panel()
    gated = LassoModel(
        auto_sign_constraints=True, auto_sign_threshold_t=1e6, reg_lambda=1e-8, solver="CLARABEL"
    ).fit(x, y, verbose=False)
    free = replace(gated, auto_sign_excluded_factors=("Oil", "Gold")).fit(x, y, verbose=False)
    assert (gated.derived_signs_ == 0).all().all()
    assert free.derived_signs_[["Oil", "Gold"]].isna().all().all()
    assert (free.derived_signs_.Equity == 0).all()
    plain = LassoModel(reg_lambda=1e-8, solver="CLARABEL").fit(x, y, verbose=False)
    excluded = replace(plain, auto_sign_excluded_factors=("Oil",)).fit(x, y, verbose=False)
    np.testing.assert_array_equal(plain.coef_, excluded.coef_)
