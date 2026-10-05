"""Independent exact-weight, pivot and quadratic-distribution references."""
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import chi2

from factorlasso.inference._mean import compute_weighted_mean_hac_geometry
from factorlasso.inference._quadratic import gaussian_quadratic_quantile
from factorlasso.covariance._alpha_uncertainty import (
    calibrate_alpha_uncertainty, estimate_alpha_uncertainty,
)
from factorlasso import compute_ewm, gaussian_critical_value, gaussian_quadratic_summary


def test_exact_initialized_geometry_and_known_shape_calibration():
    """Production impulses and direct score pairs independently verify the adapter."""
    rng = np.random.default_rng(91)
    data = pd.DataFrame(rng.normal(size=(39, 1)), columns=['asset'])
    data.iloc[0] = np.nan
    fit = estimate_alpha_uncertainty(data, 60, calendar=np.arange(39), scale=12.)
    q = fit.weights.asset.iloc[1:].to_numpy()
    geometry = compute_weighted_mean_hac_geometry(q, calendar=np.arange(1, 39), scale=12.)
    estimate, se = geometry.statistics(data.asset.iloc[1:])
    assert estimate[0] == pytest.approx(12*compute_ewm(data.asset, span=60).iloc[-1])
    assert se[0] == pytest.approx(fit.diagnostics.loc['asset', 'standard_error'])
    assert geometry.target_contrast[0] == pytest.approx(12*q.sum())
    critical = gaussian_critical_value(geometry, np.eye(38))
    assert critical == pytest.approx(2.455426777258591, rel=1e-7)
    calibrated = calibrate_alpha_uncertainty(
        fit, data, method='known_shape', covariance_shapes={'asset': np.eye(38)})
    assert calibrated.loc['asset', 'critical_value'] == pytest.approx(critical)
    assert calibrated.loc['asset', 'lower'] == pytest.approx(estimate[0]-critical*se[0])


def test_calendar_gaps_and_ar1_support_are_explicit():
    """No regular-grid AR guarantee is assigned after compressing internal gaps."""
    data = pd.DataFrame({'asset': np.arange(20.)})
    data.loc[8, 'asset'] = np.nan
    fit = estimate_alpha_uncertainty(data, 20, calendar=np.arange(20))
    result = calibrate_alpha_uncertainty(fit, data, method='ar1', cells=21)
    assert result.loc['asset', 'status'] == 'unsupported_irregular_calendar'
    assert np.isnan(result.loc['asset', 'lower'])
    weights = np.array([.1, .2, .3, .4])
    geometry = compute_weighted_mean_hac_geometry(weights, calendar=[0, 1, 4, 5])
    y = np.array([1., -1., 2., .5])
    scores = weights*(y-weights@y)*np.sqrt(4/3)
    expected = sum(scores[t]*scores[s]*max(0, 1-abs([0, 1, 4, 5][t]
                   - [0, 1, 4, 5][s])/6) for t in range(4) for s in range(4))
    assert geometry.statistics(y)[1][0]**2 == pytest.approx(expected)


@pytest.mark.parametrize('eigenvalues', [[3.], [2., 2., 2.], [0., 0., 0.]])
def test_quadratic_quantiles_reproduce_chi_square(eigenvalues):
    """Single and equal positive eigenvalues have an independent closed form."""
    positive = np.array(eigenvalues)[np.array(eigenvalues) > 0]
    expected = positive[0]*chi2.ppf(.95, len(positive)) if len(positive) else 0.
    assert gaussian_quadratic_quantile(eigenvalues) == pytest.approx(expected, rel=1e-8)


def test_unequal_quadratic_quantile_against_polar_integral():
    """Two-dimensional Gaussian radius/direction gives a separate CDF calculation."""
    values = np.array([.05, 3.])
    quantile = gaussian_quadratic_quantile(values)
    cdf = quad(lambda theta: 1-np.exp(-quantile/(2*(
        values[0]*np.cos(theta)**2+values[1]*np.sin(theta)**2))), 0, np.pi/2)[0]*2/np.pi
    assert cdf == pytest.approx(.95, abs=2e-8)
    assert gaussian_quadratic_quantile(7*values) == pytest.approx(7*quantile, rel=1e-8)
    result = gaussian_quadratic_summary(np.zeros(2), np.eye(2), np.diag(values),
                                        method='weighted_chi2')
    old = gaussian_quadratic_summary(np.zeros(2), np.eye(2), np.diag(values))
    assert result['upper'] == pytest.approx(quantile)
    assert result['upper'] < old['upper']
    assert result['lower'] == 0


def test_calibration_rejects_stale_residuals():
    """A result cannot be paired with a different residual panel or asset ordering."""
    data = pd.DataFrame({'a': np.arange(12.), 'b': np.arange(12.)**2})
    fit = estimate_alpha_uncertainty(data, 12, calendar=np.arange(12))
    with pytest.raises(ValueError):
        calibrate_alpha_uncertainty(fit, data+1, method='normal')
    with pytest.raises(ValueError):
        calibrate_alpha_uncertainty(fit, data[['b', 'a']], method='normal')


def test_regular_ar1_adapter_matches_generic_calibration():
    """A regular quarterly grid retains native AR1 interpretation and all audit fields."""
    from factorlasso import compute_ar1_interval
    data = pd.DataFrame({'asset': np.random.default_rng(15).normal(size=12)})
    fit = estimate_alpha_uncertainty(data, 20, calendar=3*np.arange(12))
    geometry = compute_weighted_mean_hac_geometry(fit.weights.asset,
                                                  calendar=3*np.arange(12))
    expected = compute_ar1_interval(geometry, data.asset, cells=21)
    actual = calibrate_alpha_uncertainty(fit, data, method='ar1', cells=21)
    assert actual.loc['asset', 'critical_value'] == pytest.approx(expected.critical_value[0])
    assert actual.loc['asset', 'lower'] == pytest.approx(expected.lower[0])
    assert actual.loc['asset', 'status'] == 'bounded_ar1_gaussian_model'
    assert 'retained_cells' in actual.loc['asset', 'calibration_audit']


@pytest.mark.parametrize('signal', [0., .5])
def test_weighted_quadratic_bounds_cover_known_gaussian_target(signal):
    """Independent Gaussian draws check null and nonzero target coverage."""
    from factorlasso import quadratic_confidence_summary
    covariance = np.array([[1., .6, .1], [.6, 2., .2], [.1, .2, .5]])
    metric = np.diag([2., .05, 0.])
    mean = np.array([signal, -signal, 2*signal])
    result = quadratic_confidence_summary(mean, covariance, metric)
    errors = np.random.default_rng(518).normal(size=(20000, 3)) @ np.linalg.cholesky(covariance).T
    observed = np.einsum('ij,ij->i', (mean+errors) @ metric, mean+errors)
    lower = np.maximum(0., np.sqrt(observed)-result['error_norm_radius'])**2
    upper = (np.sqrt(observed)+result['error_norm_radius'])**2
    truth = mean @ metric @ mean
    coverage = np.mean((lower <= truth) & (truth <= upper))
    assert coverage >= .94
    if signal == 0:
        assert coverage <= .96
    spectral = gaussian_quadratic_summary(mean, covariance, metric)
    assert result['upper'] <= spectral['upper']
