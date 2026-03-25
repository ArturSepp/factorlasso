"""
Test suite for factorlasso.

Targets near-100% coverage as required by JMLR MLOSS.
"""

import numpy as np
import pandas as pd
import pytest

from factorlasso import (
    LassoModel,
    LassoModelType,
    LassoEstimationResult,
    solve_lasso_cvx_problem,
    solve_group_lasso_cvx_problem,
    get_x_y_np,
    compute_clusters_from_corr_matrix,
    CurrentFactorCovarData,
    RollingFactorCovarData,
    VarianceColumns,
)


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def factor_data():
    """Synthetic factor model: Y_t = β X_t + noise (code: Y = X @ β' + noise)."""
    np.random.seed(42)
    T, M, N = 200, 3, 5
    X = pd.DataFrame(np.random.randn(T, M), columns=['f0', 'f1', 'f2'])
    beta_true = np.array([
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.0],
        [0.3, 0.0, 0.0],
        [0.0, 0.8, 0.2],
        [1.0, 0.5, 0.0],
    ])
    Y = pd.DataFrame(
        X.values @ beta_true.T + 0.1 * np.random.randn(T, N),
        columns=[f'y{i}' for i in range(N)],
    )
    return X, Y, beta_true


@pytest.fixture
def group_data():
    return pd.Series(['A', 'A', 'B', 'B', 'A'],
                     index=[f'y{i}' for i in range(5)])


@pytest.fixture
def factor_covar_data(factor_data):
    """Fit a model and build CurrentFactorCovarData."""
    X, Y, _ = factor_data
    model = LassoModel(reg_lambda=1e-4, span=52)
    model.fit(x=X, y=Y)
    betas = model.estimated_betas
    result = model.estimation_result_

    from factorlasso.ewm_utils import compute_ewm_covar
    x_np, _, _ = get_x_y_np(x=X, y=Y, span=52)
    x_cov_np = compute_ewm_covar(a=x_np, span=52)
    x_covar = pd.DataFrame(x_cov_np, index=X.columns, columns=X.columns)

    y_variances = pd.DataFrame({
        VarianceColumns.EWMA_VARIANCE: result.ss_total,
        VarianceColumns.RESIDUAL_VARS: result.ss_res,
        VarianceColumns.INSAMPLE_ALPHA: result.alpha,
        VarianceColumns.R2: result.r2,
    }, index=Y.columns)

    residuals = Y - X @ betas.T

    return CurrentFactorCovarData(
        x_covar=x_covar,
        y_betas=betas,
        y_variances=y_variances,
        residuals=residuals,
        estimation_date=pd.Timestamp('2024-01-01'),
    )


# ═══════════════════════════════════════════════════════════════════════
# LassoModel — basic
# ═══════════════════════════════════════════════════════════════════════

class TestLassoBasic:
    def test_fit_returns_self(self, factor_data):
        X, Y, _ = factor_data
        model = LassoModel(reg_lambda=1e-4)
        assert model.fit(x=X, y=Y) is model

    def test_shape(self, factor_data):
        X, Y, _ = factor_data
        model = LassoModel(reg_lambda=1e-4)
        model.fit(x=X, y=Y)
        assert model.coef_.shape == (5, 3)
        assert list(model.coef_.index) == list(Y.columns)
        assert list(model.coef_.columns) == list(X.columns)

    def test_intercept(self, factor_data):
        X, Y, _ = factor_data
        model = LassoModel(reg_lambda=1e-4)
        model.fit(x=X, y=Y)
        assert model.intercept_ is not None
        assert model.intercept_.shape == (5,)
        assert model.intercept_.name == 'intercept'

    def test_recovers_betas(self, factor_data):
        X, Y, beta_true = factor_data
        model = LassoModel(reg_lambda=1e-5, demean=False)
        model.fit(x=X, y=Y)
        np.testing.assert_allclose(model.coef_.values, beta_true, atol=0.15)

    def test_r2_reasonable(self, factor_data):
        X, Y, _ = factor_data
        model = LassoModel(reg_lambda=1e-5)
        model.fit(x=X, y=Y)
        assert np.all(model.estimation_result_.r2 > 0.5)

    def test_result_fields(self, factor_data):
        X, Y, _ = factor_data
        model = LassoModel(reg_lambda=1e-4)
        model.fit(x=X, y=Y)
        r = model.estimation_result_
        assert isinstance(r, LassoEstimationResult)
        for attr in ('estimated_beta', 'alpha', 'ss_total', 'ss_res', 'r2'):
            assert getattr(r, attr) is not None

    def test_copy(self, factor_data):
        X, Y, _ = factor_data
        m1 = LassoModel(reg_lambda=1e-4)
        m2 = m1.copy(kwargs={'reg_lambda': 1e-3})
        assert m2.reg_lambda == 1e-3
        assert m1.reg_lambda == 1e-4

    def test_backward_compat_estimated_betas(self, factor_data):
        """estimated_betas property aliases coef_."""
        X, Y, _ = factor_data
        model = LassoModel(reg_lambda=1e-4)
        model.fit(x=X, y=Y)
        assert model.estimated_betas is model.coef_

    def test_backward_compat_clusters(self, factor_data):
        """clusters property aliases clusters_."""
        X, Y, _ = factor_data
        model = LassoModel(
            model_type=LassoModelType.GROUP_LASSO_CLUSTERS,
            reg_lambda=1e-5, span=52,
        )
        model.fit(x=X, y=Y)
        assert model.clusters is model.clusters_

    def test_backward_compat_x_y(self, factor_data):
        """x/y properties alias x_/y_."""
        X, Y, _ = factor_data
        model = LassoModel(reg_lambda=1e-4)
        model.fit(x=X, y=Y)
        assert model.x is model.x_
        assert model.y is model.y_


# ═══════════════════════════════════════════════════════════════════════
# predict and score (sklearn-compatible)
# ═══════════════════════════════════════════════════════════════════════

class TestPredictScore:
    def test_predict_shape(self, factor_data):
        X, Y, _ = factor_data
        model = LassoModel(reg_lambda=1e-4)
        model.fit(x=X, y=Y)
        y_hat = model.predict(X)
        assert y_hat.shape == Y.shape

    def test_predict_before_fit_raises(self, factor_data):
        X, Y, _ = factor_data
        model = LassoModel(reg_lambda=1e-4)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(X)

    def test_predict_reasonable(self, factor_data):
        X, Y, _ = factor_data
        model = LassoModel(reg_lambda=1e-5, demean=False)
        model.fit(x=X, y=Y)
        y_hat = model.predict(X)
        # Residuals should be small (low noise in test data)
        residuals = Y - y_hat
        assert residuals.abs().mean().mean() < 0.5

    def test_score_positive(self, factor_data):
        X, Y, _ = factor_data
        model = LassoModel(reg_lambda=1e-5, demean=False)
        model.fit(x=X, y=Y)
        r2 = model.score(X, Y)
        assert r2 > 0.5

    def test_score_between_0_and_1(self, factor_data):
        X, Y, _ = factor_data
        model = LassoModel(reg_lambda=1e-5, demean=False)
        model.fit(x=X, y=Y)
        r2 = model.score(X, Y)
        assert 0.0 < r2 <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# Sign constraints
# ═══════════════════════════════════════════════════════════════════════

class TestSignConstraints:
    def test_nonneg(self, factor_data):
        X, Y, _ = factor_data
        model = LassoModel(reg_lambda=1e-4, nonneg=True)
        model.fit(x=X, y=Y)
        assert np.all(model.estimated_betas.values >= -1e-8)

    def test_sign_matrix(self, factor_data):
        X, Y, _ = factor_data
        signs = pd.DataFrame(
            [[1, np.nan, np.nan],
             [1, np.nan, 0],
             [1, np.nan, np.nan],
             [1, np.nan, np.nan],
             [1, np.nan, np.nan]],
            index=Y.columns, columns=X.columns,
        )
        model = LassoModel(reg_lambda=1e-4, factors_beta_loading_signs=signs)
        model.fit(x=X, y=Y)
        b = model.estimated_betas.values
        assert np.all(b[:, 0] >= -1e-8)     # f0 non-negative
        assert abs(b[1, 2]) < 1e-8           # y1.f2 = 0


# ═══════════════════════════════════════════════════════════════════════
# Prior-centered regularisation
# ═══════════════════════════════════════════════════════════════════════

class TestPrior:
    def test_shrinkage_toward_prior(self, factor_data):
        X, Y, beta_true = factor_data
        prior = pd.DataFrame(beta_true, index=Y.columns, columns=X.columns)
        model = LassoModel(reg_lambda=1e-2, factors_beta_prior=prior)
        model.fit(x=X, y=Y)
        np.testing.assert_allclose(model.estimated_betas.values, beta_true, atol=0.3)


# ═══════════════════════════════════════════════════════════════════════
# EWMA weighting
# ═══════════════════════════════════════════════════════════════════════

class TestEWMA:
    def test_span(self, factor_data):
        X, Y, _ = factor_data
        model = LassoModel(reg_lambda=1e-4, span=52)
        model.fit(x=X, y=Y)
        assert model.estimated_betas.shape == (5, 3)

    def test_no_demean(self, factor_data):
        X, Y, _ = factor_data
        model = LassoModel(reg_lambda=1e-4, demean=False)
        model.fit(x=X, y=Y)
        assert model.estimated_betas.shape == (5, 3)


# ═══════════════════════════════════════════════════════════════════════
# Group LASSO
# ═══════════════════════════════════════════════════════════════════════

class TestGroupLasso:
    def test_runs(self, factor_data, group_data):
        X, Y, _ = factor_data
        model = LassoModel(
            model_type=LassoModelType.GROUP_LASSO,
            group_data=group_data, reg_lambda=1e-5,
        )
        model.fit(x=X, y=Y)
        assert model.estimated_betas.shape == (5, 3)

    def test_requires_group_data(self):
        with pytest.raises(ValueError, match="group_data"):
            LassoModel(model_type=LassoModelType.GROUP_LASSO)


# ═══════════════════════════════════════════════════════════════════════
# HCGL
# ═══════════════════════════════════════════════════════════════════════

class TestHCGL:
    def test_runs(self, factor_data):
        X, Y, _ = factor_data
        model = LassoModel(
            model_type=LassoModelType.GROUP_LASSO_CLUSTERS,
            reg_lambda=1e-5, span=52,
        )
        model.fit(x=X, y=Y)
        assert model.estimated_betas.shape == (5, 3)
        assert model.clusters is not None
        assert len(model.clusters) == 5



# ═══════════════════════════════════════════════════════════════════════
# NaN handling
# ═══════════════════════════════════════════════════════════════════════

class TestNaN:
    def test_nan_in_y(self, factor_data):
        X, Y, _ = factor_data
        Y2 = Y.copy()
        Y2.iloc[:30, 2] = np.nan
        model = LassoModel(reg_lambda=1e-4)
        model.fit(x=X, y=Y2)
        assert np.all(np.isfinite(model.estimated_betas.values))

    def test_warmup_zeros_short_history(self, factor_data):
        X, Y, _ = factor_data
        Y2 = Y.copy()
        Y2.iloc[:195, 4] = np.nan
        model = LassoModel(reg_lambda=1e-4, warmup_period=12)
        model.fit(x=X, y=Y2)
        assert np.allclose(model.estimated_betas.iloc[4].values, 0.0)

    def test_x_all_nan_rows(self, factor_data):
        X, Y, _ = factor_data
        X2 = X.copy()
        X2.iloc[0:3, :] = np.nan
        model = LassoModel(reg_lambda=1e-4)
        model.fit(x=X2, y=Y)
        assert model.estimated_betas.shape == (5, 3)


# ═══════════════════════════════════════════════════════════════════════
# Low-level solver functions
# ═══════════════════════════════════════════════════════════════════════

class TestSolvers:
    def test_lasso_solver_direct(self, factor_data):
        X, Y, _ = factor_data
        x_np, y_np, mask = get_x_y_np(x=X, y=Y)
        r = solve_lasso_cvx_problem(x=x_np, y=y_np, valid_mask=mask, reg_lambda=1e-4)
        assert r.estimated_beta.shape == (5, 3)

    def test_insufficient_obs(self):
        x = np.random.randn(3, 2)
        y = np.random.randn(3, 4)
        with pytest.warns(UserWarning, match="insufficient"):
            r = solve_lasso_cvx_problem(x=x, y=y, valid_mask=np.ones_like(y))
        assert np.all(np.isnan(r.estimated_beta))

    def test_group_lasso_solver_direct(self, factor_data, group_data):
        X, Y, _ = factor_data
        from factorlasso.ewm_utils import set_group_loadings
        x_np, y_np, mask = get_x_y_np(x=X, y=Y)
        gl = set_group_loadings(group_data=group_data).to_numpy()
        r = solve_group_lasso_cvx_problem(
            x=x_np, y=y_np, group_loadings=gl, valid_mask=mask, reg_lambda=1e-5,
        )
        assert r.estimated_beta.shape == (5, 3)


# ═══════════════════════════════════════════════════════════════════════
# Clustering
# ═══════════════════════════════════════════════════════════════════════

class TestClustering:
    def test_basic(self):
        np.random.seed(0)
        c = np.eye(4) + 0.01 * np.random.randn(4, 4)
        c = (c + c.T) / 2
        np.fill_diagonal(c, 1.0)
        corr = pd.DataFrame(c, columns=list('abcd'), index=list('abcd'))
        clusters, linkage, cutoff = compute_clusters_from_corr_matrix(corr)
        assert len(clusters) == 4
        assert cutoff > 0


# ═══════════════════════════════════════════════════════════════════════
# get_x_y_np
# ═══════════════════════════════════════════════════════════════════════

class TestGetXYNP:
    def test_basic(self, factor_data):
        X, Y, _ = factor_data
        x_np, y_np, mask = get_x_y_np(x=X, y=Y)
        assert x_np.shape[0] == y_np.shape[0] == mask.shape[0]
        assert np.all(mask == 1.0)

    def test_ewma_shortens(self, factor_data):
        X, Y, _ = factor_data
        x_np, y_np, _ = get_x_y_np(x=X, y=Y, span=52)
        assert x_np.shape[0] == X.shape[0] - 1

    def test_mismatched_index_raises(self, factor_data):
        X, Y, _ = factor_data
        with pytest.raises(AssertionError):
            get_x_y_np(x=X.iloc[:-5], y=Y)


# ═══════════════════════════════════════════════════════════════════════
# CurrentFactorCovarData
# ═══════════════════════════════════════════════════════════════════════

class TestCurrentFactorCovarData:
    def test_y_covar_shape(self, factor_covar_data):
        cov = factor_covar_data.get_y_covar()
        N = factor_covar_data.y_betas.shape[0]
        assert cov.shape == (N, N)

    def test_y_covar_symmetric(self, factor_covar_data):
        cov = factor_covar_data.get_y_covar()
        np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-12)

    def test_y_covar_psd(self, factor_covar_data):
        cov = factor_covar_data.get_y_covar()
        eigvals = np.linalg.eigvalsh(cov.values)
        assert np.all(eigvals >= -1e-10)

    def test_residual_weight_zero(self, factor_covar_data):
        cov0 = factor_covar_data.get_y_covar(residual_var_weight=0.0)
        cov1 = factor_covar_data.get_y_covar(residual_var_weight=1.0)
        # cov1 should have larger diagonal
        assert np.all(np.diag(cov1.values) >= np.diag(cov0.values) - 1e-12)

    def test_subset(self, factor_covar_data):
        sub = factor_covar_data.filter_on_tickers(['y0', 'y2'])
        assert sub.y_betas.shape[0] == 2
        assert sub.get_y_covar().shape == (2, 2)

    def test_model_vols(self, factor_covar_data):
        vols = factor_covar_data.get_model_vols()
        assert VarianceColumns.TOTAL_VOL.value in vols.columns
        assert np.all(vols.values >= 0)

    def test_estimate_alpha(self, factor_covar_data):
        alpha = factor_covar_data.estimate_alpha(alpha_span=50)
        assert len(alpha) == 5

    def test_snapshot(self, factor_covar_data):
        snap = factor_covar_data.get_snapshot()
        assert snap.shape[0] == 5

    def test_property_shorthand(self, factor_covar_data):
        cov = factor_covar_data.y_covar
        assert cov.shape[0] == cov.shape[1]

    def test_save_load(self, factor_covar_data, tmp_path):
        p = str(tmp_path / "test.xlsx")
        factor_covar_data.save(p)
        loaded = CurrentFactorCovarData.load(p)
        np.testing.assert_allclose(
            loaded.y_betas.values, factor_covar_data.y_betas.values, atol=1e-10,
        )

    def test_rename_filter(self, factor_covar_data):
        rename = {'y0': 'asset_A', 'y1': 'asset_B'}
        sub = factor_covar_data.filter_on_tickers(rename)
        assert 'asset_A' in sub.y_betas.index


# ═══════════════════════════════════════════════════════════════════════
# RollingFactorCovarData
# ═══════════════════════════════════════════════════════════════════════

class TestRollingFactorCovarData:
    def test_add_and_access(self, factor_covar_data):
        rolling = RollingFactorCovarData()
        d1 = pd.Timestamp('2024-01-01')
        d2 = pd.Timestamp('2024-04-01')
        rolling.add(d1, factor_covar_data)
        rolling.add(d2, factor_covar_data)
        assert len(rolling) == 2
        assert rolling.n_observations == 2
        assert rolling[d1] is factor_covar_data

    def test_panel_accessors(self, factor_covar_data):
        rolling = RollingFactorCovarData()
        for i, m in enumerate([1, 4, 7]):
            rolling.add(pd.Timestamp(f'2024-{m:02d}-01'), factor_covar_data)

        r2 = rolling.get_r2()
        assert r2.shape == (3, 5)

        betas = rolling.get_y_betas()
        assert len(betas) == 3

        covars = rolling.get_y_covars()
        assert len(covars) == 3

        beta_f0 = rolling.get_beta('f0')
        assert beta_f0.shape == (3, 5)

    def test_systematic_vars(self, factor_covar_data):
        rolling = RollingFactorCovarData()
        for m in [1, 4, 7]:
            rolling.add(pd.Timestamp(f'2024-{m:02d}-01'), factor_covar_data)
        sys_vars = rolling.get_systematic_vars()
        assert sys_vars.shape == (3, 5)
        assert np.all(sys_vars.values >= -1e-12)
        # systematic var should match diag(beta @ Sigma_x @ beta.T) from single date
        single = factor_covar_data
        betas_np = single.y_betas.values
        expected = np.diag(betas_np @ single.x_covar.values @ betas_np.T)
        np.testing.assert_allclose(sys_vars.iloc[0].values, expected, atol=1e-10)

    def test_total_vols(self, factor_covar_data):
        rolling = RollingFactorCovarData()
        for m in [1, 4, 7]:
            rolling.add(pd.Timestamp(f'2024-{m:02d}-01'), factor_covar_data)
        total_vols = rolling.get_total_vols()
        resid_vols = rolling.get_residual_vols()
        sys_vars = rolling.get_systematic_vars()
        resid_vars = rolling.get_residual_vars()
        # total_vol = sqrt(systematic_var + residual_var)
        expected = np.sqrt(sys_vars.values + resid_vars.values)
        np.testing.assert_allclose(total_vols.values, expected, atol=1e-10)
        # total_vol >= residual_vol
        assert np.all(total_vols.values >= resid_vols.values - 1e-10)

    def test_get_latest(self, factor_covar_data):
        rolling = RollingFactorCovarData()
        rolling.add(pd.Timestamp('2024-01-01'), factor_covar_data)
        rolling.add(pd.Timestamp('2024-07-01'), factor_covar_data)
        assert rolling.get_latest().estimation_date == factor_covar_data.estimation_date

    def test_filter_on_tickers(self, factor_covar_data):
        rolling = RollingFactorCovarData()
        rolling.add(pd.Timestamp('2024-01-01'), factor_covar_data)
        sub = rolling.filter_on_tickers(['y0', 'y1'])
        assert sub[pd.Timestamp('2024-01-01')].y_betas.shape[0] == 2


# ═══════════════════════════════════════════════════════════════════════
# ewm_utils
# ═══════════════════════════════════════════════════════════════════════

class TestEWMUtils:
    def test_expanding_power(self):
        from factorlasso.ewm_utils import compute_expanding_power
        w = compute_expanding_power(5, 0.9, reverse_columns=True)
        assert len(w) == 5
        assert w[-1] == pytest.approx(1.0)
        assert w[0] < w[-1]

    def test_ewm_pandas(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        from factorlasso.ewm_utils import compute_ewm
        result = compute_ewm(s, span=3)
        assert len(result) == 5
        assert result.iloc[-1] > result.iloc[0]

    def test_ewm_numpy(self):
        from factorlasso.ewm_utils import compute_ewm
        a = np.array([1.0, 2.0, 3.0, 4.0])
        result = compute_ewm(a, span=3)
        assert result.shape == (4,)

    def test_ewm_covar_shape(self):
        from factorlasso.ewm_utils import compute_ewm_covar
        a = np.random.randn(100, 3)
        cov = compute_ewm_covar(a, span=20)
        assert cov.shape == (3, 3)

    def test_ewm_covar_correlation(self):
        from factorlasso.ewm_utils import compute_ewm_covar
        a = np.random.randn(100, 3)
        corr = compute_ewm_covar(a, span=20, is_corr=True)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)

    def test_group_loadings(self):
        from factorlasso.ewm_utils import set_group_loadings
        g = pd.Series(['A', 'B', 'A', 'C'], index=['x', 'y', 'z', 'w'])
        gl = set_group_loadings(g)
        assert gl.shape == (4, 3)
        assert gl.loc['x', 'A'] == 1.0
        assert gl.loc['y', 'A'] == 0.0