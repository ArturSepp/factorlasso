"""Independent checks of exact recursive EWMA and joint calendar HAC inference."""
import numpy as np
import pandas as pd
import pytest
from factorlasso import (
    compute_ewm, compute_prior_hac_geometry, estimate_alpha_uncertainty,
    gaussian_quadratic_summary, sample_gaussian_estimates,
)


def panel():
    """Mixed cadence, unequal histories and internal gaps on a common monthly grid."""
    rng = np.random.default_rng(20261004)
    frame = pd.DataFrame(rng.normal(size=(90, 3)), columns=['a', 'b', 'c'])
    frame.loc[:12, 'a'] = np.nan
    frame.loc[22:25, 'a'] = np.nan
    frame.loc[frame.index % 3 != 2, 'c'] = np.nan
    return frame


def test_exact_recursion_and_independent_calendar_sandwich():
    """Reference recursion weights by impulses, then compute every HAC pair directly."""
    data = panel()
    spans = pd.Series([30., 60., 20.], index=data.columns)
    result = estimate_alpha_uncertainty(data, spans, calendar=np.arange(90), bandwidth=6.)
    scores = np.zeros(data.shape)
    for j, name in enumerate(data):
        valid = data[name].notna().to_numpy()
        n = valid.sum()
        impulses = np.full((90, n), np.nan)
        impulses[valid] = np.eye(n)
        q = compute_ewm(impulses, span=spans[name])[-1]
        alpha = compute_ewm(data[name], span=spans[name]).iloc[-1]
        np.testing.assert_allclose(result.estimates[name], alpha, atol=1e-14)
        np.testing.assert_allclose(result.weights.loc[valid, name], q, atol=1e-14)
        center = alpha / q.sum()
        scores[valid, j] = q * (data.loc[valid, name] - center) * np.sqrt(n/(n-1))
    expected = np.zeros((3, 3))
    for t in range(90):
        for s in range(90):
            expected += max(0., 1-abs(t-s)/6.) * np.outer(scores[t], scores[s])
    np.testing.assert_allclose(result.covariance, expected, atol=2e-14)
    assert np.linalg.eigvalsh(result.covariance).min() >= -1e-14


def test_matches_existing_geometry_and_scales_as_estimation_variance():
    """The new diagonal must equal FactorLasso's existing arbitrary-weight intercept HAC."""
    data = panel()[['a', 'b']].iloc[30:].reset_index(drop=True)
    result = estimate_alpha_uncertainty(data, 30, calendar=np.arange(60), bandwidth=4.)
    for name in data:
        q = result.weights[name].to_numpy()
        geometry = compute_prior_hac_geometry(np.ones((60, 1)), q, 3, coefficient=0)
        mean, se = geometry.statistics(data[name])
        assert result.estimates[name] == pytest.approx(mean[0] * q.sum())
        assert result.diagnostics.loc[name, 'standard_error'] == pytest.approx(se[0]*q.sum())
    scale = pd.Series([12., 4.], index=data.columns)
    other = estimate_alpha_uncertainty(data, 30, calendar=np.arange(60), bandwidth=4.,
                                       scale=scale)
    np.testing.assert_allclose(other.estimates, result.estimates*scale)
    np.testing.assert_allclose(other.covariance, result.covariance*np.outer(scale, scale))


def test_initialization_and_calendar_gaps_are_distinct():
    """Leading zero initialisation retains weight mass; a time gap changes HAC only."""
    data = panel()[['a']]
    result = estimate_alpha_uncertainty(data, 60, calendar=np.arange(90), bandwidth=6.)
    n = data.a.notna().sum()
    assert result.diagnostics.loc['a', 'weight_mass'] == pytest.approx(1-(1-2/61)**n)
    stretched = estimate_alpha_uncertainty(data, 60, calendar=2*np.arange(90), bandwidth=6.)
    np.testing.assert_array_equal(result.estimates, stretched.estimates)
    assert not np.allclose(result.covariance, stretched.covariance)


@pytest.mark.parametrize('change', ['order', 'span', 'calendar', 'infinite', 'short', 'overlap'])
def test_invalid_inputs_do_not_create_confidence(change):
    """Reject ill-specified estimators or histories instead of inferring independence."""
    data = panel()
    spans = pd.Series(30., index=data.columns)
    calendar = np.arange(90, dtype=float)
    if change == 'order':
        spans = spans.iloc[::-1]
    elif change == 'span':
        spans.iloc[0] = 0
    elif change == 'calendar':
        calendar[10] = calendar[9]
    elif change == 'infinite':
        data.iloc[40, 0] = np.inf
    elif change == 'short':
        data.iloc[:-2, 0] = np.nan
    else:
        data.iloc[:45, 0] = np.nan
        data.iloc[45:, 1] = np.nan
    with pytest.raises(ValueError):
        estimate_alpha_uncertainty(data, spans, calendar=calendar, bandwidth=6.)


def test_quadratic_noise_and_bounds_allow_the_null():
    """Noise correction is signed; confidence-region bounds do not square away zero."""
    result = gaussian_quadratic_summary(np.zeros(3), np.eye(3), np.diag([1., 2., 0.]))
    assert result['observed'] == 0
    assert result['noise'] == 3
    assert result['noise_adjusted'] == -3
    assert result['lower'] == 0
    assert result['upper'] > 0
    exact = gaussian_quadratic_summary(np.array([1., 2.]), np.zeros((2, 2)), np.eye(2))
    assert exact['lower'] == exact['upper'] == exact['observed'] == 5.


def test_joint_scenarios_and_known_covariance_region_coverage():
    """Verify correlated draws, quadratic bias, and conservative known-V coverage."""
    mean = np.array([.3, -.2])
    covariance = np.array([[.5, .3], [.3, .4]])
    metric = np.diag([1., 2.])
    draws = sample_gaussian_estimates(mean, covariance, draws=30000, seed=41)
    np.testing.assert_allclose(np.cov(draws.T), covariance, atol=.012)
    np.testing.assert_allclose(draws[:5], sample_gaussian_estimates(mean, covariance,
                                                                  draws=5, seed=41))
    values = np.einsum('ij,jk,ik->i', draws, metric, draws)
    expected = mean @ metric @ mean + np.trace(metric @ covariance)
    assert abs(values.mean()-expected) < .04
    intervals = [gaussian_quadratic_summary(x, covariance, metric) for x in draws[:500]]
    truth = mean @ metric @ mean
    assert np.mean([x['lower'] <= truth <= x['upper'] for x in intervals]) >= .94


def test_bad_covariance_rejected_and_singular_covariance_supported():
    """No material PSD repairs; genuinely singular uncertainty remains singular."""
    with pytest.raises(ValueError):
        sample_gaussian_estimates(np.zeros(2), np.diag([1., -.01]), draws=3, seed=1)
    draws = sample_gaussian_estimates(np.zeros(2), np.ones((2, 2)), draws=10, seed=1)
    np.testing.assert_allclose(draws[:, 0], draws[:, 1], atol=1e-14)
