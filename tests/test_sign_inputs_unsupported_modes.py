"""Sign inputs reach only the solvers that enforce them.

``UNILASSO`` and the two cooperative modes take no sign constraint. An explicit sign matrix or
``nonneg=True`` is rejected at construction and at fit instead of being dropped silently, and
``derived_signs_`` stays ``None`` after a fit whose solver did not receive the derived signs.
"""

import numpy as np
import pandas as pd
import pytest

from factorlasso import LassoModel, LassoModelType

UNSUPPORTED = [
    LassoModelType.UNILASSO,
    LassoModelType.COOPERATIVE_GROUP_LASSO,
    LassoModelType.COOPERATIVE_CLUSTER_GROUP_LASSO,
]


def _panel():
    rng = np.random.default_rng(3)
    x = pd.DataFrame(rng.standard_normal((60, 3)), columns=["f0", "f1", "f2"])
    beta = np.array([[1.0, 0.0, 0.0], [0.8, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.6, 0.7]])
    y = pd.DataFrame(x.to_numpy() @ beta.T + 0.1 * rng.standard_normal((60, 4)),
                     columns=[f"a{i}" for i in range(4)])
    return x, y


def _kwargs(model_type, y):
    kwargs = {"model_type": model_type, "reg_lambda": 1e-3}
    if model_type == LassoModelType.COOPERATIVE_GROUP_LASSO:
        kwargs["group_data"] = pd.Series(["g0", "g0", "g1", "g1"], index=y.columns)
    return kwargs


def _signs(x, y):
    signs = pd.DataFrame(np.nan, index=y.columns, columns=x.columns)
    signs.loc["a0", "f1"] = 0.0
    signs.loc["a2", "f0"] = 1.0
    return signs


@pytest.mark.parametrize("model_type", UNSUPPORTED)
def test_explicit_sign_matrix_is_rejected_at_construction(model_type):
    x, y = _panel()
    with pytest.raises(ValueError, match="factors_beta_loading_signs"):
        LassoModel(factors_beta_loading_signs=_signs(x, y), **_kwargs(model_type, y))


@pytest.mark.parametrize("model_type", UNSUPPORTED)
def test_nonneg_is_rejected_at_construction(model_type):
    _, y = _panel()
    with pytest.raises(ValueError, match="nonneg"):
        LassoModel(nonneg=True, **_kwargs(model_type, y))


@pytest.mark.parametrize("model_type", UNSUPPORTED)
def test_sign_matrix_set_after_construction_is_rejected_at_fit(model_type):
    x, y = _panel()
    model = LassoModel(**_kwargs(model_type, y))
    model.set_params(factors_beta_loading_signs=_signs(x, y))
    with pytest.raises(ValueError, match="factors_beta_loading_signs"):
        model.fit(x=x, y=y)


@pytest.mark.parametrize("model_type", UNSUPPORTED)
def test_derived_signs_stay_none_when_the_solver_takes_none(model_type):
    x, y = _panel()
    model = LassoModel(auto_sign_constraints=True, **_kwargs(model_type, y)).fit(x=x, y=y)
    assert model.coef_.shape == (4, 3)
    assert model.derived_signs_ is None


def test_supported_modes_still_enforce_and_report_the_matrix():
    x, y = _panel()
    model = LassoModel(reg_lambda=1e-3, factors_beta_loading_signs=_signs(x, y)).fit(x=x, y=y)
    pd.testing.assert_frame_equal(model.derived_signs_, _signs(x, y))
    assert abs(model.coef_.loc["a0", "f1"]) < 1e-7
    assert model.coef_.loc["a2", "f0"] > -1e-7
