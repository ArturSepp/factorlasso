"""Independent projection geometry, full-estimator statistics and calibration gates."""
import numpy as np
import pandas as pd
import pytest
from scipy.linalg import null_space

from factorlasso import (FactorTestInversion, LassoModel, LassoModelType,
                         calibrate_inversion_tests)


def panel(seed=9):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(0, .04, (40, 2)), columns=['equity', 'credit'])
    y = pd.DataFrame(x.to_numpy() @ np.array([[.6, .3], [.2, .7]])
                     + rng.normal(0, .01, (40, 2)), columns=['a', 'b'])
    y.loc[:7, 'b'] = np.nan
    model = LassoModel(model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
        reg_lambda=1e-5, span=36, warmup_period=8, loss_normalization='weight_sum',
        apply_ols_prior=True, auto_sign_constraints=True, auto_sign_adaptive_weights=True,
        solver='CLARABEL')
    contrast = pd.DataFrame([[1.2, 0.], [-.4, 0.]], index=y.columns, columns=x.columns)
    return model, x, y, contrast


def test_working_shape_preserves_common_noise_and_profiles_reml_against_gls_reference():
    rng = np.random.default_rng(812)
    x = pd.DataFrame(rng.normal(0, .04, (80, 2)), columns=['equity', 'credit'])
    e = rng.normal(size=(80, 1))+.06*rng.normal(size=(80, 3))
    for i in range(1, len(e)):
        e[i] += .45*e[i-1]
    y = pd.DataFrame(x.to_numpy() @ rng.normal(size=(2, 3))+.01*e, columns=list('abc'))
    y.loc[:19, 'b'] = np.nan
    y.loc[:9, 'c'] = np.nan
    model = panel()[0]
    engine = FactorTestInversion(model, x, y, draws=19, seed=1)
    meta = engine.noise_metadata
    corr = np.asarray(meta['correlation'])
    assert corr[np.tril_indices(3, -1)].min() > .95
    assert meta['correlation_shrinkage'] < .01
    assert abs(meta['rho']-.45) < .25
    # Independent GLS criterion in unwhitened coordinates, using dense inverses
    # and normal equations rather than the implementation's triangular solve/QR.
    d, response, dates, names = [], [], [], []
    for t in range(len(y)):
        for a in range(3):
            if pd.notna(y.iloc[t, a]):
                row = np.zeros(9)
                row[3*a:3*a+3] = [1., *x.iloc[t]]
                d.append(row)
                response.append(y.iloc[t, a])
                dates.append(t)
                names.append(a)
    d, response, dates, names = map(np.asarray, (d, response, dates, names))
    scales = np.asarray(meta['asset_scales'])[names]
    cross = corr[names[:, None], names[None, :]]*np.outer(scales, scales)

    def reference(rho):
        cov = cross*rho**np.abs(dates[:, None]-dates[None, :])
        precision = np.linalg.inv(cov)
        normal = d.T @ precision @ d
        beta = np.linalg.solve(normal, d.T @ precision @ response)
        residual = response-d @ beta
        df = len(d)-d.shape[1]
        return (np.linalg.slogdet(cov)[1]+np.linalg.slogdet(normal)[1]
                +df*np.log((residual @ precision @ residual)/df))

    optimum = reference(meta['rho'])
    np.testing.assert_allclose(optimum, meta['reml_objective'], rtol=1e-11, atol=1e-9)
    for rho in [0., .85, max(0., meta['rho']-.01), min(.85, meta['rho']+.01)]:
        assert optimum <= reference(rho)+1e-8


def test_estimated_shape_rejects_unidentified_cross_asset_dependence():
    model, x, y, _ = panel()
    y.iloc[:20, 1] = np.nan
    y.iloc[20:, 0] = np.nan
    with pytest.raises(ValueError, match='common-window residual'):
        FactorTestInversion(model, x, y, draws=19)


def test_conditional_geometry_against_independent_null_space():
    model, x, y, contrast = panel()
    n = int(y.notna().sum().sum())
    cov = .4**np.abs(np.arange(n)[:, None]-np.arange(n)[None, :])
    engine = FactorTestInversion(model, x, y, covariance=cov, draws=20000, seed=82)
    c, scale, direction, nuisance, directions, point = engine._geometry(contrast)
    # Independent null parameterisation and QR-like projector; no engine SVD used.
    design = []
    for t, asset in engine.observed_index:
        row = []
        for name in y.columns:
            row.extend([1., *x.loc[t]]) if asset == name else row.extend([0., 0., 0.])
        design.append(row)
    d = np.linalg.solve(np.linalg.cholesky(cov), np.asarray(design))
    cn = np.array([0., 1.2, 0., 0., -.4, 0.])
    np.testing.assert_array_equal(c, cn)
    dn = d @ null_space(cn[None, :])
    pn = dn @ np.linalg.pinv(dn)
    reference_cov = (np.eye(n)-pn)/(n-np.linalg.matrix_rank(dn))
    np.testing.assert_allclose(nuisance, pn @ engine.response, atol=2e-12)
    np.testing.assert_allclose(direction @ direction, 1., atol=2e-12)
    np.testing.assert_allclose(directions @ dn, 0., atol=2e-12)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1., atol=2e-12)
    np.testing.assert_allclose(directions.T @ directions/len(directions), reference_cov,
                               atol=.0006)
    expected_scale = np.sqrt(cn @ np.linalg.solve(d.T @ d, cn))
    np.testing.assert_allclose(scale, expected_scale, rtol=1e-12)
    assert point == pytest.approx((engine.estimate*contrast).to_numpy().sum())
    assert model.coef_ is None


def test_every_null_draw_matches_an_independent_complete_fit(monkeypatch):
    model, x, y, contrast = panel()
    engine = FactorTestInversion(model, x, y, covariance=np.eye(y.notna().sum().sum()),
                                 draws=19, seed=17)
    c, scale, direction, nuisance, directions, point = engine._geometry(contrast)
    candidate = .2
    z = nuisance+direction*candidate/scale+directions[0]*np.sqrt(
        engine.rss+((c @ engine.gamma-candidate)/scale)**2)
    response = y.copy()
    arr = response.to_numpy(copy=True)
    arr[engine.rows, engine.assets] = engine.chol @ z
    response = pd.DataFrame(arr, index=y.index, columns=y.columns)
    expected = model.copy().fit(x, response).coef_
    original_fit, captured = LassoModel.fit, []

    def recording(self, xx, yy, **kwargs):
        pd.testing.assert_frame_equal(yy.isna(), y.isna())
        result = original_fit(self, xx, yy, **kwargs)
        captured.append(result.coef_.copy())
        return result

    monkeypatch.setattr(LassoModel, 'fit', recording)
    test = engine.test(contrast, candidate)
    pd.testing.assert_frame_equal(captured[0], expected)
    vals = np.array([(value*contrast).to_numpy().sum() for value in captured])
    assert len(captured) == 19
    assert test['pvalue'] == (1+(abs(vals-candidate) >= abs(point-candidate)-1e-10).sum())/20


def test_early_acceptance_is_a_bound_on_the_fixed_sample_test():
    model, x, y, contrast = panel()
    engine = FactorTestInversion(model, x, y, covariance=np.eye(y.notna().sum().sum()),
                                 draws=39, seed=8)
    candidate = float((engine.estimate*contrast).to_numpy().sum())+.01
    short = engine.test(contrast, candidate, accept_above=.05)
    full = engine.test(contrast, candidate)
    assert short['pvalue'] <= full['pvalue'] <= short['pvalue_upper']
    assert short['draws_completed'] < full['draws_completed']
    assert (short['pvalue'] > .05) == (full['pvalue'] > .05)


def test_calibration_cannot_reuse_seeds_or_hide_a_failed_holdout():
    def make(start, values):
        return pd.DataFrame(dict(case='one', seed=np.arange(start, start+len(values)),
                                 pvalue=values, draws=199))
    train = make(0, np.linspace(.005, 1., 1000))
    valid = make(10000, np.linspace(.005, 1., 1000))
    result = calibrate_inversion_tests(train, valid)
    assert result.passed and result.rejection_probability < .05
    failing = make(20000, np.full(1000, .005))
    bad = calibrate_inversion_tests(train, failing)
    assert not bad.passed
    assert bad.rejection_probability == result.rejection_probability
    with pytest.raises(ValueError, match='disjoint'):
        calibrate_inversion_tests(train, train)


def test_estimated_covariance_requires_validation_and_fixed_shape_does_not():
    model, x, y, contrast = panel()
    estimated = FactorTestInversion(model, x, y, draws=39)
    with pytest.raises(ValueError, match='passing independent calibration'):
        estimated.intervals({'equity': contrast})
    with pytest.raises(ValueError, match='exact label order'):
        estimated.test(contrast.iloc[::-1], .3)


def test_inversion_does_not_truncate_an_open_tail(monkeypatch):
    model, x, y, contrast = panel()
    engine = FactorTestInversion(model, x, y, covariance=np.eye(y.notna().sum().sum()), draws=19)
    point = float((engine.estimate*contrast).to_numpy().sum())
    monkeypatch.setattr(engine, 'test', lambda c, b, **kw:
                        dict(estimate=point, candidate=b, pvalue=1., draws=0))
    result = engine.intervals({'equity': contrast}, grid_size=5, max_expansions=2)
    assert result.intervals.loc['equity', 'lower'] == -np.inf
    assert result.intervals.loc['equity', 'upper'] == np.inf
    assert result.intervals.loc['equity', 'estimate'] == point


def test_monte_carlo_failure_is_not_discarded(monkeypatch):
    model, x, y, contrast = panel()
    engine = FactorTestInversion(model, x, y, covariance=np.eye(y.notna().sum().sum()), draws=19)
    def fail(*args, **kwargs):
        raise ValueError('solver failure')
    monkeypatch.setattr(LassoModel, 'fit', fail)
    with pytest.raises(RuntimeError, match='no draws dropped'):
        engine.test(contrast, 100.)


def test_inversion_outer_brackets_enclose_independent_acceptance_set(monkeypatch):
    model, x, y, contrast = panel()
    engine = FactorTestInversion(model, x, y,
        covariance=np.eye(y.notna().sum().sum()), draws=39)
    point = float((engine.estimate*contrast).to_numpy().sum())
    # Two disjoint accepted islands, including the statistic's production point.
    bounds = [(point-.25, point+.2), (point+.35, point+.6)]

    def reference_test(c, b, **kw):
        accepted = any(lo <= b <= hi for lo, hi in bounds)
        return dict(estimate=point, candidate=b, pvalue=.5 if accepted else .025, draws=39)

    monkeypatch.setattr(engine, 'test', reference_test)
    result = engine.intervals({'equity': contrast}, grid_size=81, refinements=10)
    row = result.intervals.loc['equity']
    assert row.acceptance_regions == 2
    assert row.lower <= bounds[0][0] <= row.lower+row.lower_resolution+1e-12
    assert row.upper-row.upper_resolution-1e-12 <= bounds[1][1] <= row.upper
    assert row.estimate == point


def test_binomial_coverage_gate_matches_independent_exact_tail():
    from scipy.stats import binom
    train = pd.DataFrame(dict(case='one', seed=np.arange(1000), draws=199, pvalue=.5))
    valid = pd.DataFrame(dict(case='one', seed=np.arange(2000, 3000), draws=199,
                              pvalue=np.r_[np.full(980, .5), np.full(20, .005)]))
    result = calibrate_inversion_tests(train, valid)
    lo = result.coverage.loc['one', 'lower_bound']
    # Tail inversion is computed independently of the beta-quantile implementation.
    assert binom.sf(979, 1000, lo) == pytest.approx(.05, rel=1e-10)
    assert result.passed


def test_fixed_design_cache_matches_full_refits_and_invalidates_all_geometry_inputs():
    from factorlasso.priors._bounds import (
        _competitor_geometry, _fixed_design_competitor_geometry)
    model, x, y, _ = panel()
    model.expert_prior_bound_n_std = .5
    cache = {}
    rng = np.random.default_rng(171)
    for i in range(8):
        response = y+rng.normal(0., .03, y.shape)
        uncached = model.copy().fit(x, response)
        cached = model.copy()
        cached._inference_fixed_design_cache = cache
        cached.fit(x, response)
        pd.testing.assert_frame_equal(cached.coef_, uncached.coef_, check_exact=True)
        pd.testing.assert_frame_equal(cached.prior_bound_contributions_,
                                       uncached.prior_bound_contributions_, check_exact=True)
    assert cache
    original = _fixed_design_competitor_geometry(x, y.a, ['equity'], 36, 8, cache)
    original[0].iloc[:] = 999.  # A caller mutation must not poison the cache.
    for xx, yy, factors, span, minimum in [
        (x, y.a, ['equity'], 36, 8),
        (x, y.b, ['equity'], 36, 8),
        (x, y.a, ['credit'], 36, 8),
        (x*1.1, y.a, ['equity'], 36, 8),
        (x, y.a, ['equity'], 12, 8),
        (x, y.a, ['equity'], 36, 39),
    ]:
        actual = _fixed_design_competitor_geometry(xx, yy, factors, span, minimum, cache)
        reference = _competitor_geometry(xx, yy, factors, span, minimum)
        for a, b in zip(actual, reference):
            pd.testing.assert_frame_equal(a, b, check_exact=True)
