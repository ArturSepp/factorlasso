"""
Tests for scikit-learn interoperability and the summary/plot helpers.

These pin the behaviour the paper claims in Section 3: LassoModel composes
with scikit-learn's Pipeline, GridSearchCV, and cross_val_score, and the
fitted object exposes a human-readable summary() and a plot_signs() helper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorlasso import LassoModel, LassoModelType


@pytest.fixture
def panel():
    rng = np.random.default_rng(0)
    T, N, M = 120, 24, 6
    X = pd.DataFrame(rng.standard_normal((T, M)), columns=[f"F{j}" for j in range(M)])
    beta = rng.standard_normal((N, M)) * 0.4
    Y = pd.DataFrame(
        X.values @ beta.T + 0.3 * rng.standard_normal((T, N)),
        columns=[f"A{k}" for k in range(N)],
    )
    return X, Y


def test_gridsearchcv(panel):
    pytest.importorskip("sklearn", exc_type=ImportError)
    from sklearn.model_selection import GridSearchCV

    X, Y = panel
    gs = GridSearchCV(
        LassoModel(model_type=LassoModelType.GROUP_LASSO_CLUSTERS),
        param_grid={"reg_lambda": [1e-4, 1e-3, 1e-2]},
        cv=3,
    )
    gs.fit(X, Y)
    assert gs.best_params_["reg_lambda"] in (1e-4, 1e-3, 1e-2)


def test_pipeline_with_ndarray(panel):
    pytest.importorskip("sklearn", exc_type=ImportError)
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, Y = panel
    pipe = Pipeline(
        [("scale", StandardScaler()), ("model", LassoModel(reg_lambda=1e-3))]
    )
    pipe.fit(X.values, Y.values)
    pred = pipe.predict(X.values)
    assert pred.shape == (len(X), Y.shape[1])


def test_cross_val_score(panel):
    pytest.importorskip("sklearn", exc_type=ImportError)
    from sklearn.model_selection import cross_val_score

    X, Y = panel
    scores = cross_val_score(LassoModel(reg_lambda=1e-3), X, Y, cv=3)
    assert len(scores) == 3
    assert np.isfinite(scores).all()


def test_summary(panel):
    X, Y = panel
    model = LassoModel(
        model_type=LassoModelType.GROUP_LASSO_CLUSTERS,
        reg_lambda=1e-3,
        auto_sign_constraints=True,
        auto_sign_adaptive_weights=True,
    ).fit(x=X, y=Y)
    s = model.summary()
    assert "LassoModel summary" in s
    assert "GROUP_LASSO_CLUSTERS" in s
    assert "clusters (HCGL)" in s


def test_summary_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        LassoModel().summary()


def test_plot_signs(panel):
    matplotlib = pytest.importorskip("matplotlib", exc_type=ImportError)
    matplotlib.use("Agg")
    X, Y = panel
    model = LassoModel(
        model_type=LassoModelType.GROUP_LASSO_CLUSTERS,
        reg_lambda=1e-3,
        auto_sign_constraints=True,
    ).fit(x=X, y=Y)
    ax = model.plot_signs()
    assert ax is not None


def test_plot_signs_without_signs_raises(panel):
    X, Y = panel
    model = LassoModel(reg_lambda=1e-3).fit(x=X, y=Y)
    with pytest.raises(RuntimeError, match="auto_sign_constraints"):
        model.plot_signs()
