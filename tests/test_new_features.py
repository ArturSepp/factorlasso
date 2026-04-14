"""
Tests for new package features in 0.1.9:

- LassoModelCV (time-series CV for reg_lambda)
- LassoModel.get_params / set_params (sklearn compat)
- LassoModel.fit input validation (ValueError / TypeError)
- expanding_window_splits helper
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorlasso import LassoModel, LassoModelCV, LassoModelType
from factorlasso.cv import expanding_window_splits

# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def panel():
    rng = np.random.default_rng(0)
    T, M, N = 120, 3, 4
    idx = pd.date_range("2018-01-31", periods=T, freq="ME")
    X = pd.DataFrame(
        rng.standard_normal((T, M)),
        index=idx, columns=[f"f{i}" for i in range(M)],
    )
    beta = rng.standard_normal((N, M))
    Y = pd.DataFrame(
        X.values @ beta.T + 0.1 * rng.standard_normal((T, N)),
        index=idx, columns=[f"y{i}" for i in range(N)],
    )
    return X, Y


# ═══════════════════════════════════════════════════════════════════════
# expanding_window_splits
# ═══════════════════════════════════════════════════════════════════════

class TestExpandingWindowSplits:
    def test_basic_shape(self):
        splits = list(expanding_window_splits(n_samples=100, n_splits=4))
        assert len(splits) == 4
        for tr, te in splits:
            # Train and test are disjoint and contiguous
            assert tr.max() < te.min()
            # Test windows are equal-sized
            assert len(te) == 100 // 5

    def test_train_window_expands(self):
        splits = list(expanding_window_splits(n_samples=100, n_splits=4))
        train_sizes = [len(tr) for tr, _ in splits]
        assert train_sizes == sorted(train_sizes)
        assert len(set(train_sizes)) == len(train_sizes)  # strictly increasing

    def test_indices_are_within_bounds(self):
        for tr, te in expanding_window_splits(n_samples=50, n_splits=3):
            assert tr.min() >= 0
            assert te.max() < 50

    def test_invalid_n_splits(self):
        with pytest.raises(ValueError, match="n_splits must be >= 1"):
            list(expanding_window_splits(n_samples=100, n_splits=0))

    def test_too_few_samples(self):
        with pytest.raises(ValueError, match="too small"):
            list(expanding_window_splits(n_samples=2, n_splits=5))


# ═══════════════════════════════════════════════════════════════════════
# LassoModelCV
# ═══════════════════════════════════════════════════════════════════════

class TestLassoModelCV:
    def test_fit_picks_a_lambda(self, panel):
        X, Y = panel
        cv = LassoModelCV(
            lambdas=[1e-5, 1e-4, 1e-3, 1e-2], n_splits=3,
        ).fit(x=X, y=Y)
        assert cv.best_lambda_ in [1e-5, 1e-4, 1e-3, 1e-2]
        assert cv.best_score_ is not None
        assert cv.cv_scores_.shape == (4, 3)
        assert cv.best_model_ is not None
        assert cv.best_model_.coef_ is not None

    def test_default_lambda_grid(self, panel):
        X, Y = panel
        cv = LassoModelCV(n_splits=3).fit(x=X, y=Y)
        assert cv.cv_scores_.shape[0] == 20  # default grid size

    def test_no_refit(self, panel):
        X, Y = panel
        cv = LassoModelCV(
            lambdas=[1e-4, 1e-3], n_splits=3, refit=False,
        ).fit(x=X, y=Y)
        assert cv.best_model_ is None
        with pytest.raises(RuntimeError, match="refit=True"):
            cv.predict(X)
        with pytest.raises(RuntimeError, match="refit=True"):
            cv.score(X, Y)

    def test_predict_and_score_after_refit(self, panel):
        X, Y = panel
        cv = LassoModelCV(lambdas=[1e-4, 1e-3], n_splits=3).fit(x=X, y=Y)
        y_hat = cv.predict(X)
        assert y_hat.shape == Y.shape
        assert isinstance(cv.score(X, Y), float)

    def test_inherits_base_model_hyperparameters(self, panel):
        X, Y = panel
        base = LassoModel(model_type=LassoModelType.LASSO, span=24, demean=False)
        cv = LassoModelCV(
            lambdas=[1e-4, 1e-3], n_splits=3, base_model=base,
        ).fit(x=X, y=Y)
        # Refitted model should have inherited span and demean
        assert cv.best_model_.span == 24
        assert cv.best_model_.demean is False
        # And of course the chosen reg_lambda
        assert cv.best_model_.reg_lambda == cv.best_lambda_

    def test_series_inputs_are_coerced(self, panel):
        X, Y = panel
        cv = LassoModelCV(lambdas=[1e-3], n_splits=3).fit(
            x=X.iloc[:, 0], y=Y.iloc[:, 0],
        )
        assert cv.best_lambda_ == 1e-3

    def test_index_mismatch_raises(self, panel):
        X, Y = panel
        with pytest.raises(ValueError, match="must share the same index"):
            LassoModelCV(lambdas=[1e-3], n_splits=3).fit(x=X.iloc[:-5], y=Y)

    def test_empty_lambda_grid_raises(self, panel):
        X, Y = panel
        with pytest.raises(ValueError, match="lambdas must be non-empty"):
            LassoModelCV(lambdas=[], n_splits=3).fit(x=X, y=Y)

    def test_all_folds_failing_raises(self, panel, monkeypatch):
        X, Y = panel
        # Force every LassoModel.score() call to return NaN so every fold fails
        monkeypatch.setattr(
            LassoModel, "score", lambda self, x, y: float("nan"),
        )
        # Also make fit raise to take the except path occasionally
        cv = LassoModelCV(lambdas=[1e-4, 1e-3], n_splits=3)
        with pytest.raises(RuntimeError, match="All CV folds failed"):
            cv.fit(x=X, y=Y)

    def test_fold_exception_is_swallowed(self, panel, monkeypatch):
        """A fit/score exception in one fold leaves a NaN but does not abort CV."""
        X, Y = panel
        original_fit = LassoModel.fit
        call_count = {"n": 0}

        def flaky_fit(self, x, y, verbose=False, span=None):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("simulated fold failure")
            return original_fit(self, x=x, y=y, verbose=verbose, span=span)

        monkeypatch.setattr(LassoModel, "fit", flaky_fit)
        cv = LassoModelCV(lambdas=[1e-3, 1e-4], n_splits=3, refit=False).fit(x=X, y=Y)
        # At least one NaN, but a best lambda was still chosen
        assert cv.cv_scores_.isna().any().any()
        assert cv.best_lambda_ is not None


# ═══════════════════════════════════════════════════════════════════════
# get_params / set_params
# ═══════════════════════════════════════════════════════════════════════

class TestGetSetParams:
    def test_get_params_returns_constructor_args_only(self):
        m = LassoModel(reg_lambda=1e-3, span=24, demean=False)
        p = m.get_params()
        assert p["reg_lambda"] == 1e-3
        assert p["span"] == 24
        assert p["demean"] is False
        assert "model_type" in p
        # No fitted attributes
        assert not any(k.endswith("_") for k in p)

    def test_get_params_deep_argument_is_accepted(self):
        # sklearn GridSearchCV passes deep=True; just verify it doesn't error
        m = LassoModel()
        assert m.get_params(deep=True) == m.get_params(deep=False)

    def test_set_params_updates_hyperparameters(self):
        m = LassoModel(reg_lambda=1e-5)
        out = m.set_params(reg_lambda=1e-2, span=36)
        assert out is m  # chainable
        assert m.reg_lambda == 1e-2
        assert m.span == 36

    def test_set_params_rejects_unknown_keys(self):
        m = LassoModel()
        with pytest.raises(ValueError, match="Invalid parameter"):
            m.set_params(reg_lambda=1e-3, nonexistent_param=42)

    def test_set_params_rejects_fitted_attributes(self):
        m = LassoModel()
        with pytest.raises(ValueError, match="Invalid parameter"):
            m.set_params(coef_=pd.DataFrame())

    def test_round_trip_via_constructor(self):
        m1 = LassoModel(reg_lambda=1e-3, span=24, nonneg=True)
        m2 = LassoModel(**m1.get_params())
        assert m2.get_params() == m1.get_params()


# ═══════════════════════════════════════════════════════════════════════
# fit() input validation
# ═══════════════════════════════════════════════════════════════════════

class TestFitValidation:
    def test_index_mismatch_raises_valueerror(self, panel):
        X, Y = panel
        with pytest.raises(ValueError, match="must share the same index"):
            LassoModel().fit(x=X.iloc[:-5], y=Y)

    def test_empty_input_raises(self):
        empty_idx = pd.date_range("2024-01-31", periods=0, freq="ME")
        X = pd.DataFrame(index=empty_idx, columns=["f0"])
        Y = pd.DataFrame(index=empty_idx, columns=["y0"])
        with pytest.raises(ValueError, match="Empty input"):
            LassoModel().fit(x=X, y=Y)

    def test_non_dataframe_x_raises_typeerror(self, panel):
        _, Y = panel
        with pytest.raises(TypeError, match="x must be"):
            LassoModel().fit(x=np.zeros((len(Y), 2)), y=Y)

    def test_non_dataframe_y_raises_typeerror(self, panel):
        X, _ = panel
        with pytest.raises(TypeError, match="y must be"):
            LassoModel().fit(x=X, y=np.zeros((len(X), 2)))


# ═══════════════════════════════════════════════════════════════════════
# Package metadata
# ═══════════════════════════════════════════════════════════════════════

class TestVersion:
    def test_version_is_a_string(self):
        import factorlasso
        assert isinstance(factorlasso.__version__, str)
        assert factorlasso.__version__ != "0.0.0+unknown"
