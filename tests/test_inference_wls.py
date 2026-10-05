"""Independent WLS covariance, target and numerical-conditioning checks."""
import numpy as np
import pandas as pd
import pytest

from factorlasso.inference._geometry import LinearHacGeometry, compute_wls_hac_geometry
from factorlasso.inference._wls import compute_wls_hac_statistics
from factorlasso.priors import compute_expert_prior_statistics


@pytest.mark.parametrize('intercept', [False, True])
@pytest.mark.parametrize('gaps', [False, True])
def test_full_wls_covariance_against_direct_sandwich(intercept, gaps):
    """A normal-equation solve and explicit score products verify the SVD backend."""
    rng = np.random.default_rng(402)
    x = rng.normal(size=(35, 2)) + [2., -1.]
    y = 1.5 + x @ [.7, -.4] + rng.normal(size=35)
    w = .93**np.arange(34, -1, -1)
    if gaps:
        y[[0, 1, 7]] = np.nan
        x[15, 0] = np.inf
        w[20] = 0
    result = compute_wls_hac_statistics(x, y, w, 3, fit_intercept=intercept,
                                        return_covariance=True)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y) & (w > 0)
    d = np.column_stack((np.ones(35), x)) if intercept else x.copy()
    d = np.where(valid[:, None], d, 0.)
    yy = np.where(valid, y, 0.)
    w = np.where(valid, w, 0.)
    bread = np.linalg.inv(d.T @ (w[:, None]*d))
    beta = bread @ d.T @ (w*yy)
    scores = w[:, None]*d*(yy-d @ beta)[:, None]
    meat = np.zeros_like(bread)
    for t in range(35):
        for s in range(35):
            meat += max(0., 1-abs(t-s)/4)*np.outer(scores[t], scores[s])
    expected = bread @ meat @ bread*valid.sum()/(valid.sum()-d.shape[1])
    np.testing.assert_allclose(result.coefficients, beta, atol=2e-13)
    np.testing.assert_allclose(result.covariance, expected, rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(result.standard_error**2, np.diag(expected), rtol=2e-12)
    scaled = compute_wls_hac_statistics(x, y, w*1e-14, 3, fit_intercept=intercept)
    np.testing.assert_allclose(scaled.coefficients, beta, atol=2e-13)
    assert scaled.covariance is None
    assert result.observations == valid.sum()


def test_weighted_mean_and_contrast_geometry():
    """Intercept-only means and arbitrary coefficient contrasts retain their targets."""
    rng = np.random.default_rng(403)
    y = rng.normal(size=24)
    weights = np.linspace(.2, 1., 24)
    mean = compute_wls_hac_statistics(np.empty((24, 0)), y, weights, 2,
                                      return_covariance=True)
    geometry = compute_wls_hac_geometry(np.ones((24, 1)), weights, 2, coefficient=0)
    estimates, errors = geometry.statistics(y)
    np.testing.assert_allclose(mean.coefficients, np.average(y, weights=weights))
    np.testing.assert_allclose(mean.standard_error, errors)
    np.testing.assert_allclose(mean.coefficients, estimates)
    x = rng.normal(size=(24, 2))
    full = compute_wls_hac_statistics(x, y, weights, 2, return_covariance=True)
    c = np.array([.5, 1., -2.])
    contrast = compute_wls_hac_geometry(np.column_stack((np.ones(24), x)), weights, 2,
                                       contrast=c)
    estimate, error = contrast.statistics(y)
    np.testing.assert_allclose(estimate, c @ full.coefficients, atol=1e-13)
    np.testing.assert_allclose(error**2, c @ full.covariance @ c, rtol=1e-12)
    np.testing.assert_allclose(contrast.linear @ contrast.design, c, atol=1e-13)


def test_large_offsets_factor_units_and_prior_wrapper():
    """Centering preserves slopes even when intercepts and factor units are extreme."""
    rng = np.random.default_rng(404)
    x = rng.normal(size=(60, 2)) + [1e10, -2e9]
    y = 1e11 + (x-x[-1]) @ [.7, -.4] + rng.normal(size=60)
    w = (1-2/25)**np.arange(59, -1, -1)
    stable = compute_wls_hac_statistics(x-x[-1], y-y[-1], w, 3)
    actual = compute_wls_hac_statistics(x, y, w, 3)
    np.testing.assert_allclose(actual.coefficients[1:], stable.coefficients[1:], atol=1e-13)
    np.testing.assert_allclose(actual.standard_error[1:], stable.standard_error[1:], atol=1e-13)
    prior = compute_expert_prior_statistics(pd.DataFrame(x), pd.Series(y), 24, 3)
    np.testing.assert_allclose(prior.reference_beta, actual.coefficients[1:])
    np.testing.assert_allclose(prior.se_hac, actual.standard_error[1:])
    units = np.array([1e-5, 1e5])
    scaled = compute_wls_hac_statistics((x-x[-1])*units, y-y[-1], w, 3)
    np.testing.assert_allclose(scaled.coefficients[1:]*units, stable.coefficients[1:])


def test_unidentified_and_invalid_inputs():
    """Unidentified fits report status; malformed settings raise."""
    result = compute_wls_hac_statistics(np.ones((10, 1)), np.arange(10))
    assert result.status == 'rank_deficient'
    assert np.isnan(result.coefficients).all()
    short = compute_wls_hac_statistics(np.ones((2, 1)), [1., 2.])
    assert short.status == 'insufficient_observations'
    empty = compute_wls_hac_statistics(np.ones((10, 1)), np.arange(10), np.zeros(10))
    assert empty.observations == empty.effective_n == 0
    for kwargs in ({'hac_lags': True}, {'weights': [-1]*10}, {'min_periods': -.1},
                   {'fit_intercept': 1}, {'return_covariance': 1}):
        with pytest.raises(ValueError):
            compute_wls_hac_statistics(np.ones((10, 1)), np.arange(10), **kwargs)


@pytest.mark.parametrize('defect', ['target', 'mean', 'residual', 'asymmetric', 'negative'])
def test_generic_geometry_rejects_invalid_contract(defect):
    """A finite matrix is insufficient: target, mean removal and PSD must agree."""
    d = np.column_stack((np.ones(12), np.linspace(-1, 1, 12)))
    geometry = compute_wls_hac_geometry(d, coefficient=1)
    values = {name: value.copy() for name, value in vars(geometry).items()}
    if defect == 'target':
        values['target_contrast'][1] = 2
    elif defect == 'mean':
        values['quadratic'] += np.ones((12, 12))
    elif defect == 'residual':
        values['residual_map'] *= 0
    elif defect == 'asymmetric':
        values['quadratic'][0, 1] += 1
    else:
        values['quadratic'] *= -1
    with pytest.raises(ValueError):
        LinearHacGeometry(**values)
    assert not geometry.linear.flags.writeable
    d[:] = 0
    assert np.any(geometry.design)


def test_production_statistics_do_not_allocate_dense_geometry(monkeypatch):
    """A long prior regression must not enter the optional dense calibration builder."""
    import factorlasso.inference._geometry as geometry_module

    def forbidden(*args, **kwargs):
        """Detect an accidental production dependency on dense geometry."""
        raise AssertionError('dense calibration geometry entered production')

    monkeypatch.setattr(geometry_module, '_wls_geometry_arrays', forbidden)
    result = compute_expert_prior_statistics(pd.DataFrame(np.arange(5000.)),
                                             pd.Series(np.sin(np.arange(5000.))), 256, 12)
    assert result.status.iloc[0] == 'ok'
