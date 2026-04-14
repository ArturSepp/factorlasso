"""
Targeted coverage tests for branches missed by the main suite.

These cover defensive raises, backward-compat property setters, the
per-frequency dict path of estimate_alpha, RollingFactorCovarData
accessors, and solver edge cases (t<5, infeasible, missing model_type).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorlasso import (
    CurrentFactorCovarData,
    LassoModel,
    RollingFactorCovarData,
    VarianceColumns,
    compute_ewm_covar,
    get_x_y_np,
    set_group_loadings,
    solve_group_lasso_cvx_problem,
    solve_lasso_cvx_problem,
)
from factorlasso.lasso_estimator import (
    _build_sign_constraints,
    _derive_valid_mask_from_y,
)

# ═══════════════════════════════════════════════════════════════════════
# Shared builders
# ═══════════════════════════════════════════════════════════════════════

def _build_snapshot(seed: int = 0, with_residuals: bool = True,
                    date: str = "2024-01-31") -> CurrentFactorCovarData:
    """Build a small CurrentFactorCovarData by fitting a real model."""
    rng = np.random.default_rng(seed)
    T, M, N = 60, 3, 4
    dates = pd.date_range("2019-01-31", periods=T, freq="MS")
    X = pd.DataFrame(rng.standard_normal((T, M)),
                     index=dates, columns=[f"f{i}" for i in range(M)])
    beta_true = rng.standard_normal((N, M))
    Y = pd.DataFrame(
        X.values @ beta_true.T + 0.1 * rng.standard_normal((T, N)),
        index=dates, columns=[f"y{i}" for i in range(N)],
    )
    model = LassoModel(reg_lambda=1e-4, span=24).fit(x=X, y=Y)
    result = model.estimation_result_

    x_np, _, _ = get_x_y_np(x=X, y=Y, span=24)
    x_covar = pd.DataFrame(
        compute_ewm_covar(a=x_np, span=24),
        index=X.columns, columns=X.columns,
    )
    y_variances = pd.DataFrame({
        VarianceColumns.EWMA_VARIANCE.value: result.ss_total,
        VarianceColumns.RESIDUAL_VARS.value: result.ss_res,
        VarianceColumns.INSAMPLE_ALPHA.value: result.alpha,
        VarianceColumns.R2.value: result.r2,
    }, index=Y.columns)
    residuals = (Y - X @ model.coef_.T) if with_residuals else None
    return CurrentFactorCovarData(
        x_covar=x_covar,
        y_betas=model.coef_,
        y_variances=y_variances,
        residuals=residuals,
        estimation_date=pd.Timestamp(date),
    )


@pytest.fixture
def snapshot_with_residuals():
    return _build_snapshot(seed=0, with_residuals=True)


@pytest.fixture
def snapshot_without_residuals():
    return _build_snapshot(seed=1, with_residuals=False)


@pytest.fixture
def rolling():
    r = RollingFactorCovarData()
    r.add(pd.Timestamp("2024-01-31"), _build_snapshot(seed=10, date="2024-01-31"))
    r.add(pd.Timestamp("2024-02-29"), _build_snapshot(seed=11, date="2024-02-29"))
    return r


# ═══════════════════════════════════════════════════════════════════════
# factor_covar.estimate_alpha — defensive + per-frequency dict path
# (lines 191, 199–233)
# ═══════════════════════════════════════════════════════════════════════

class TestEstimateAlpha:
    def test_raises_when_residuals_missing(self, snapshot_without_residuals):
        with pytest.raises(ValueError, match="Residuals required"):
            snapshot_without_residuals.estimate_alpha(alpha_span=60)

    def test_dict_span_default_freq(self, snapshot_with_residuals):
        # asset_frequencies=None → all assets fall through default_freq='ME'
        out = snapshot_with_residuals.estimate_alpha(
            alpha_span={"ME": 24}, asset_frequencies=None,
        )
        assert isinstance(out, pd.Series)
        assert list(out.index) == list(snapshot_with_residuals.y_betas.index)
        assert out.notna().all()

    def test_dict_span_str_broadcast(self, snapshot_with_residuals):
        out = snapshot_with_residuals.estimate_alpha(
            alpha_span={"ME": 24}, asset_frequencies="ME",
        )
        assert out.notna().all()

    def test_dict_span_series_lookup(self, snapshot_with_residuals):
        cols = snapshot_with_residuals.residuals.columns
        # Mix ME and QE; only first asset gets QE, rest fall back to default_freq=ME
        freqs = pd.Series({cols[0]: "QE", cols[1]: "ME"})
        out = snapshot_with_residuals.estimate_alpha(
            alpha_span={"ME": 24, "QE": 8}, asset_frequencies=freqs,
        )
        assert out.notna().all()
        assert len(out) == len(cols)

    def test_dict_span_invalid_type_raises(self, snapshot_with_residuals):
        with pytest.raises(TypeError, match="asset_frequencies must be"):
            snapshot_with_residuals.estimate_alpha(
                alpha_span={"ME": 24}, asset_frequencies=42,
            )

    def test_dict_span_missing_freq_raises(self, snapshot_with_residuals):
        # default_freq='ME' but alpha_span only has 'QE'
        with pytest.raises(KeyError, match="alpha_span missing entry"):
            snapshot_with_residuals.estimate_alpha(
                alpha_span={"QE": 8}, asset_frequencies=None,
            )


# ═══════════════════════════════════════════════════════════════════════
# get_snapshot fallback when residuals is None  (line 259)
# ═══════════════════════════════════════════════════════════════════════

class TestSnapshotWithoutResiduals:
    def test_get_snapshot_uses_insample_alpha(self, snapshot_without_residuals):
        snap = snapshot_without_residuals.get_snapshot()
        assert isinstance(snap, pd.DataFrame)
        assert VarianceColumns.R2.value in snap.columns


# ═══════════════════════════════════════════════════════════════════════
# RollingFactorCovarData accessors
# (lines 335, 348, 360, 387, 415–421, 423–424, 443)
# ═══════════════════════════════════════════════════════════════════════

class TestRollingAccessors:
    def test_dates_property(self, rolling):
        d = rolling.dates
        assert isinstance(d, pd.DatetimeIndex)
        assert len(d) == 2
        assert list(d) == sorted(d)

    def test_iter(self, rolling):
        out = list(iter(rolling))
        assert len(out) == 2
        assert out == sorted(out)

    def test_get_x_covars(self, rolling):
        xs = rolling.get_x_covars()
        assert len(xs) == 2
        for cov in xs.values():
            assert isinstance(cov, pd.DataFrame)
            assert cov.shape[0] == cov.shape[1]

    def test_get_ewma_vars(self, rolling):
        df = rolling.get_ewma_vars()
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 2

    def test_get_alphas_with_residuals(self, rolling):
        df = rolling.get_alphas(alpha_span=24)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 2

    def test_get_alphas_without_residuals(self):
        r = RollingFactorCovarData()
        r.add(pd.Timestamp("2024-01-31"),
              _build_snapshot(seed=20, with_residuals=False, date="2024-01-31"))
        r.add(pd.Timestamp("2024-02-29"),
              _build_snapshot(seed=21, with_residuals=False, date="2024-02-29"))
        df = r.get_alphas(alpha_span=24)
        assert df.shape[0] == 2

    def test_get_alphas_empty(self):
        assert RollingFactorCovarData().get_alphas().empty

    def test_get_factor_var(self, rolling):
        s = rolling.get_factor_var("f0")
        assert isinstance(s, pd.Series)
        assert s.name == "f0"
        assert len(s) == 2

    def test_rolling_get_snapshot(self, rolling):
        snaps = rolling.get_snapshot(alpha_span=24)
        assert len(snaps) == 2
        for v in snaps.values():
            assert isinstance(v, pd.DataFrame)


# ═══════════════════════════════════════════════════════════════════════
# Backward-compatible property setters
# (lines 608, 617, 622, 626, 631, 635, 644, 653)
# ═══════════════════════════════════════════════════════════════════════

class TestPropertyAliasSetters:
    def test_all_setters_round_trip(self):
        m = LassoModel()
        df = pd.DataFrame([[1.0]], index=["y0"], columns=["f0"])
        s = pd.Series([1], index=["y0"])
        link = np.array([[0.0, 1.0, 0.5, 2.0]])

        m.estimated_betas = df
        m.clusters = s
        m.linkage = link
        m.cutoff = 0.42
        m.x = df
        m.y = df

        assert m.coef_ is df
        assert m.clusters_ is s
        assert m.linkage_ is link
        assert m.cutoff_ == 0.42
        assert m.x_ is df
        assert m.y_ is df
        # Read back via the alias getters too
        assert m.estimated_betas is df
        assert m.clusters is s
        assert m.linkage is link
        assert m.cutoff == 0.42
        assert m.x is df
        assert m.y is df


# ═══════════════════════════════════════════════════════════════════════
# Solver edge paths
# (lines 315, 339–340, 398, 400–401, 405–406, 432–433, 755)
# ═══════════════════════════════════════════════════════════════════════

class TestSolverEdgePaths:
    def test_lasso_too_few_obs_returns_nan(self):
        x = np.random.randn(4, 2)
        y = np.random.randn(4, 3)
        with pytest.warns(UserWarning, match="insufficient observations for lasso"):
            r = solve_lasso_cvx_problem(x=x, y=y)
        assert np.all(np.isnan(r.estimated_beta))
        assert np.all(np.isnan(r.r2))

    def test_lasso_derives_valid_mask_when_none(self):
        # valid_mask=None branch + NaN handling in y
        rng = np.random.default_rng(0)
        x = rng.standard_normal((30, 2))
        y = rng.standard_normal((30, 2))
        y[0, 0] = np.nan
        r = solve_lasso_cvx_problem(x=x, y=y, valid_mask=None, reg_lambda=1e-3)
        assert r.estimated_beta.shape == (2, 2)

    def test_group_lasso_too_few_obs_returns_nan(self):
        x = np.random.randn(4, 2)
        y = np.random.randn(4, 3)
        gl = np.eye(3)
        with pytest.warns(UserWarning, match="insufficient observations for group lasso"):
            r = solve_group_lasso_cvx_problem(x=x, y=y, group_loadings=gl)
        assert np.all(np.isnan(r.estimated_beta))

    def test_group_lasso_derives_valid_mask_when_none(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((30, 2))
        y = rng.standard_normal((30, 3))
        y[0, 0] = np.nan
        gl = np.eye(3)
        r = solve_group_lasso_cvx_problem(
            x=x, y=y, group_loadings=gl, valid_mask=None, reg_lambda=1e-3,
        )
        assert r.estimated_beta.shape == (3, 2)

    def test_group_lasso_with_sign_constraints(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((40, 2))
        y = rng.standard_normal((40, 3))
        gl = np.array([[1, 0], [1, 0], [0, 1]], dtype=float)
        # Mix of signs: 0=zero, 1=nonneg, -1=nonpos
        signs = np.array([[1.0, -1.0], [0.0, 1.0], [-1.0, 0.0]])
        r = solve_group_lasso_cvx_problem(
            x=x, y=y, group_loadings=gl, reg_lambda=1e-3,
            factors_beta_loading_signs=signs,
        )
        b = r.estimated_beta
        # Respect zero entries
        assert abs(b[1, 0]) < 1e-6
        assert abs(b[2, 1]) < 1e-6
        # Nonneg / nonpos entries
        assert b[0, 0] >= -1e-9
        assert b[0, 1] <= 1e-9
        assert b[1, 1] >= -1e-9
        assert b[2, 0] <= 1e-9

    def test_lasso_solver_failure_returns_nan(self, monkeypatch):
        """Force solve_lasso_cvx_problem into the beta.value is None branch."""
        import cvxpy as cvx

        original_solve = cvx.Problem.solve

        def broken_solve(self, *a, **k):
            try:
                original_solve(self, *a, **k)
            except Exception:
                pass
            # Wipe variable values to simulate a failed solve
            for v in self.variables():
                v.value = None
            return None

        monkeypatch.setattr(cvx.Problem, "solve", broken_solve)
        rng = np.random.default_rng(3)
        x = rng.standard_normal((20, 2))
        y = rng.standard_normal((20, 2))
        with pytest.warns(UserWarning, match="lasso problem not solved"):
            r = solve_lasso_cvx_problem(x=x, y=y, reg_lambda=1e-3)
        assert np.all(np.isnan(r.estimated_beta))

    def test_group_lasso_solver_failure_returns_nan(self, monkeypatch):
        import cvxpy as cvx

        original_solve = cvx.Problem.solve

        def broken_solve(self, *a, **k):
            try:
                original_solve(self, *a, **k)
            except Exception:
                pass
            for v in self.variables():
                v.value = None
            return None

        monkeypatch.setattr(cvx.Problem, "solve", broken_solve)
        rng = np.random.default_rng(4)
        x = rng.standard_normal((20, 2))
        y = rng.standard_normal((20, 3))
        gl = np.eye(3)
        with pytest.warns(UserWarning, match="group lasso problem not solved"):
            r = solve_group_lasso_cvx_problem(
                x=x, y=y, group_loadings=gl, reg_lambda=1e-3,
            )
        assert np.all(np.isnan(r.estimated_beta))

    def test_lassomodel_unsupported_model_type_raises(self):
        rng = np.random.default_rng(5)
        X = pd.DataFrame(rng.standard_normal((30, 2)), columns=["f0", "f1"])
        Y = pd.DataFrame(rng.standard_normal((30, 3)),
                         columns=["y0", "y1", "y2"])
        m = LassoModel(reg_lambda=1e-4)
        m.model_type = "INVALID"  # bypass enum, hit the else-branch
        with pytest.raises(NotImplementedError, match="Unsupported model_type"):
            m.fit(x=X, y=Y)


# ═══════════════════════════════════════════════════════════════════════
# Misc helper coverage
# (lasso_estimator: 158–159, 177, 225, 227 ; ewm_utils: 148, 158, 184)
# ═══════════════════════════════════════════════════════════════════════

class TestHelpers:
    def test_derive_valid_mask_zero_fills_nans(self):
        y = np.array([[1.0, np.nan], [2.0, 3.0]])
        y_filled, mask = _derive_valid_mask_from_y(y)
        assert y_filled[0, 1] == 0.0
        np.testing.assert_array_equal(mask, np.array([[1.0, 0.0], [1.0, 1.0]]))

    def test_build_sign_constraints_nonpos_branch(self):
        import cvxpy as cvx
        beta = cvx.Variable((2, 2))
        signs = np.array([[-1.0, -1.0], [-1.0, -1.0]])
        cons = _build_sign_constraints(beta, signs)
        # Only the nonpos branch should produce a constraint
        assert len(cons) == 1

    def test_get_x_y_np_accepts_series(self):
        idx = pd.date_range("2024-01-31", periods=20, freq="MS")
        x = pd.Series(np.arange(20, dtype=float), index=idx, name="f0")
        y = pd.Series(np.arange(20, dtype=float) * 2.0, index=idx, name="y0")
        x_np, y_np, mask = get_x_y_np(x=x, y=y, span=10)
        assert x_np.ndim == 2 and y_np.ndim == 2
        assert x_np.shape[1] == 1 and y_np.shape[1] == 1

    def test_compute_ewm_covar_1d_path(self):
        # 1-D inputs hit the else-branch (line 148 in ewm_utils)
        a = np.array([1.0, 2.0, 3.0])
        cov = compute_ewm_covar(a=a, span=10)
        assert cov.shape == (3, 3)

    def test_compute_ewm_covar_zero_variance_corr_falls_back_to_identity(self):
        # All-zero data → diag is zero → fall back to identity (line 158)
        a = np.zeros((20, 3))
        cor = compute_ewm_covar(a=a, span=10, is_corr=True)
        np.testing.assert_array_equal(cor, np.eye(3))

    def test_set_group_loadings_rejects_non_series(self):
        with pytest.raises(ValueError, match="Expected pd.Series"):
            set_group_loadings(group_data={"a": 1, "b": 1})
