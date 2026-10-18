"""Independent score geometry and full-chain delta-method checks."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from factorlasso import (
    aggregate_influence_calendar, compute_hac_covariance, compute_wls_hac_statistics,
    delta_method_hac_intervals,
)


def reference_covariance(scores, lags):
    """Dense time kernel reference, independent of the package's lag loop."""
    t = np.arange(len(scores))
    kernel = np.maximum(0., 1-np.abs(t[:, None]-t[None, :])/(lags+1))
    return scores.T @ kernel @ scores


@pytest.mark.parametrize('intercept', [True, False])
def test_original_unit_influence_matches_independent_normal_equations(intercept):
    rng = np.random.default_rng(431)
    x = rng.normal(size=(90, 3)) * [1., 10., .01] + [1., 0., .02]
    y = x @ [.3, -.01, 2.] + rng.normal(size=90)
    weights = (11/13)**np.arange(89, -1, -1)
    y[[12, 13, 47]] = np.nan
    weights[27] = 0.
    fit = compute_wls_hac_statistics(x, y, weights, hac_lags=2,
        fit_intercept=intercept, return_covariance=True, return_influence=True)
    valid = np.isfinite(y) & (weights > 0)
    d = np.column_stack([np.ones(valid.sum()), x[valid]]) if intercept else x[valid]
    w = weights[valid]
    h = np.linalg.solve(d.T @ (w[:, None]*d), d.T*w)
    beta = h @ y[valid]
    scores = np.zeros((len(x), d.shape[1]))
    scores[valid] = h.T*(y[valid]-d @ beta)[:, None]*np.sqrt(len(d)/(len(d)-d.shape[1]))
    np.testing.assert_allclose(fit.coefficients, beta, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(fit.influence, scores, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(fit.covariance, reference_covariance(scores, 2), rtol=1e-9)
    np.testing.assert_allclose(compute_hac_covariance(fit.influence, 2), fit.covariance)
    assert fit.condition_number == pytest.approx(np.linalg.cond(np.sqrt(w)[:, None]*d))
    assert not fit.influence[~valid].any()
    scaled = compute_wls_hac_statistics(x, y, 200*weights, fit_intercept=intercept,
                                        return_influence=True)
    np.testing.assert_allclose(scaled.influence, fit.influence, atol=1e-11)


def test_opt_in_and_unidentified_influences():
    x = np.arange(12.)[:, None]
    y = x[:, 0] + np.sin(x[:, 0])
    old = compute_wls_hac_statistics(x, y, return_covariance=True)
    new = compute_wls_hac_statistics(x, y, return_covariance=True, return_influence=True)
    assert old.influence is None
    np.testing.assert_array_equal(new.coefficients, old.coefficients)
    np.testing.assert_array_equal(new.covariance, old.covariance)
    singular = compute_wls_hac_statistics(np.column_stack([x, x]), y, return_influence=True)
    assert singular.status == 'rank_deficient'
    assert singular.influence.shape == (12, 3)
    assert np.isnan(singular.influence).all()


def test_calendar_aggregation_preserves_gaps_and_mixed_histories():
    dates = pd.to_datetime(['2020-01-31', '2020-03-31', '2020-09-30'])
    monthly = pd.DataFrame({'M': [1., 2., 4.]}, index=dates)
    quarterly = pd.DataFrame({'Q': [-3., 5.]},
                             index=pd.period_range('2020Q1', periods=2, freq='Q'))
    m = aggregate_influence_calendar(monthly, end='2020Q4')
    q = aggregate_influence_calendar(quarterly, end='2020Q4')
    np.testing.assert_array_equal(m.M, [3., 0., 4., 0.])
    np.testing.assert_array_equal(q.Q, [-3., 5., 0., 0.])
    scores = pd.concat([m, q], axis=1).to_numpy()
    np.testing.assert_allclose(compute_hac_covariance(scores, 1), reference_covariance(scores, 1))
    with pytest.raises(ValueError, match='discard'):
        aggregate_influence_calendar(monthly, start='2020Q2')
    with pytest.raises(ValueError, match='finite'):
        aggregate_influence_calendar(monthly * np.nan)
    with pytest.raises(ValueError, match='unique'):
        aggregate_influence_calendar(pd.concat([monthly, monthly]))


def test_cross_asset_cancellation_detects_diagonal_covariance_defect():
    u = np.sin(np.arange(40.))
    scores = np.column_stack([u, -u])
    result = delta_method_hac_intervals([.4], scores, [[.5, .5]], hac_lags=1)
    assert result['intervals'][.95]['standard_error'][0] == 0.
    assert result['intervals'][.95]['lower'][0] == .4
    covariance = compute_hac_covariance(scores, 1)
    np.testing.assert_allclose(covariance, reference_covariance(scores, 1))
    assert covariance[0, 1] < 0.
    assert np.array([.5, .5]) @ np.diag(np.diag(covariance)) @ [.5, .5] > 0.


def test_product_chain_contains_both_stages_and_both_cadences():
    rng = np.random.default_rng(123)
    scores = rng.normal(size=(50, 4))*.001
    scores[:, 2] += scores[:, 0]*2.
    a, b = np.array([.2, .15]), np.array([-.008, -.01])
    # Column order: two sleeve coefficients, then the two conversion slopes.
    gradient = np.r_[b, a][None, :]
    value = a @ b
    result = delta_method_hac_intervals([value], scores, gradient, hac_lags=1,
                                      confidence_levels=(.8, .95))
    covariance = reference_covariance(scores, 1)
    expected = float((gradient @ covariance @ gradient.T)[0, 0])
    np.testing.assert_allclose(result['covariance'], [[expected]], rtol=1e-13)
    np.testing.assert_allclose(result['influence'][:, 0], scores[:, :2] @ b+scores[:, 2:] @ a)
    for level in (.8, .95):
        radius = norm.ppf((1+level)/2)*np.sqrt(expected)
        interval = result['intervals'][level]
        np.testing.assert_allclose([interval['lower'][0], interval['upper'][0]],
                                  [value-radius, value+radius])
    fixed_bridge = b @ covariance[:2, :2] @ b
    assert not np.isclose(expected, fixed_bridge, rtol=1e-3, atol=1e-15)
    # The Jacobian itself agrees with an independent central finite difference.
    coefficients = np.r_[a, b]
    def product(theta):
        return theta[:2] @ theta[2:]
    finite_diff = [(product(coefficients+v)-product(coefficients-v))/2e-6
                   for v in 1e-6*np.eye(4)]
    np.testing.assert_allclose(gradient[0], finite_diff, rtol=1e-8)


@pytest.mark.parametrize('lags', [-1, True, 1.2])
def test_invalid_hac_bandwidth(lags):
    with pytest.raises(ValueError, match='nonnegative integer'):
        compute_hac_covariance(np.ones((5, 2)), lags)


def test_invalid_target_inputs_and_multiple_targets():
    scores = np.arange(20.).reshape(10, 2)*.001
    gradient = np.array([[1., -1.], [2., 3.]])
    result = delta_method_hac_intervals([.1, .2], scores, gradient)
    assert set(result['intervals']) == {.95}
    np.testing.assert_allclose(result['covariance'], gradient @ (scores.T @ scores) @ gradient.T)
    for levels in [(), (.8, .8), (1.,)]:
        with pytest.raises(ValueError):
            delta_method_hac_intervals([.1], scores, [[1., 1.]], confidence_levels=levels)
    with pytest.raises(ValueError, match='jacobian'):
        delta_method_hac_intervals([.1], scores, [[1.]])
    with pytest.raises(ValueError, match='finite'):
        delta_method_hac_intervals([.1], scores*np.nan, [[1., 1.]])
