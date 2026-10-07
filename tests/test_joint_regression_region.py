"""Independent references for joint fixed-design Gaussian inference."""
import numpy as np
import pytest
from scipy.stats import chi2, norm

from factorlasso.inference._joint_regression import joint_wls_gaussian_region


def test_equal_weight_mean_against_chi_square_reference():
    """One Gaussian mean reduces to the explicitly allocated normal/chi-square bound."""
    y = np.array([-.4, .2, 1.1, -.7, .8, .1, .3, -.2])[:, None]
    result = joint_wls_gaussian_region(np.empty((8, 0)), y, np.ones_like(y),
                                        covariance_shape=np.eye(8))
    sample_variance = y[:, 0].var(ddof=1)
    expected = 7*sample_variance/chi2.ppf([.9875, .0125], 7)
    np.testing.assert_allclose(result['variance_estimate'], sample_variance, rtol=1e-12)
    np.testing.assert_allclose([result['variance_lower'][0], result['variance_upper'][0]],
                               expected, rtol=2e-7)
    radius = norm.isf(.0125)*np.sqrt(expected[1]/8)
    np.testing.assert_allclose(result['coefficient_lower'], [[y.mean()-radius]], rtol=2e-7)
    np.testing.assert_allclose(result['coefficient_upper'], [[y.mean()+radius]], rtol=2e-7)


def test_ragged_weighted_estimate_and_variance_against_direct_maps():
    """Normal equations and explicit residual traces verify weighting and calendar gaps."""
    rng = np.random.default_rng(891)
    x = rng.normal(size=(24, 2))
    y = rng.normal(size=(24, 2))
    y[[1, 7, 12], 1] = np.nan
    w = np.broadcast_to(.9**np.arange(23, -1, -1)[:, None], y.shape).copy()
    w[~np.isfinite(y)] = 0
    r = .4**abs(np.arange(24)[:, None]-np.arange(24))
    scales = np.array([[12*w[:, 0].sum(), 1, 1], [12*w[:, 1].sum(), 1, 1]])
    result = joint_wls_gaussian_region(x, y, w, covariance_shape=r, coefficient_scale=scales)
    for j in range(2):
        mask = np.isfinite(y[:, j])
        d = np.column_stack([np.ones(mask.sum()), x[mask]])
        ww = np.diag(w[mask, j])
        h = np.linalg.solve(d.T @ ww @ d, d.T @ ww)
        coef = h @ y[mask, j]
        residual_map = np.eye(mask.sum())-d @ h
        q = residual_map.T @ ww @ residual_map/np.trace(ww)
        variance = (y[mask, j] @ q @ y[mask, j])/np.trace(q @ r[np.ix_(mask, mask)])
        np.testing.assert_allclose(result['coefficient_estimate'][j], coef*scales[j], atol=1e-12)
        np.testing.assert_allclose(result['variance_estimate'][j], variance, rtol=1e-12)
    unscaled = joint_wls_gaussian_region(x, y, w*1e-10, covariance_shape=r)
    np.testing.assert_allclose(result['coefficient_lower'],
                               unscaled['coefficient_lower']*scales, rtol=1e-9)


def test_batched_joint_coverage_and_scale_target():
    """Known Gaussian samples check all parameters jointly, including correlated assets."""
    rng = np.random.default_rng(892)
    t, draws = 32, 4000
    x = np.linspace(-1, 1, t)[:, None]
    theta = np.array([[.2, .5], [-.3, .8]])
    spatial = np.array([[1., .7], [.7, 2.]])
    errors = rng.multivariate_normal([0., 0.], spatial, size=(draws, t))
    y = np.column_stack([np.ones(t), x]) @ theta.T+errors
    y[:, :8, 1] = np.nan
    w = np.broadcast_to(.94**np.arange(t-1, -1, -1)[:, None], (t, 2)).copy()
    w[:8, 1] = 0.
    result = joint_wls_gaussian_region(x, y, w, covariance_shape=np.eye(t))
    covered = ((result['coefficient_lower'] <= theta) &
               (theta <= result['coefficient_upper'])).all(axis=(1, 2))
    covered &= ((result['variance_lower'] <= np.diag(spatial)) &
                (np.diag(spatial) <= result['variance_upper'])).all(axis=1)
    assert covered.mean() >= .95
    np.testing.assert_allclose(result['variance_estimate'].mean(axis=0), np.diag(spatial),
                               rtol=.025)
    single = joint_wls_gaussian_region(x, y[0], w, covariance_shape=np.eye(t))
    np.testing.assert_allclose(single['coefficient_estimate'], result['coefficient_estimate'][0])


@pytest.mark.parametrize('defect', ['shape', 'infinite', 'mask', 'weights', 'x', 'rank',
                                   'scale', 'covariance', 'diagonal', 'short', 'confidence',
                                   'degenerate'])
def test_invalid_contracts_rejected(defect):
    """Malformed or unidentified problems must not become apparently precise regions."""
    x = np.arange(12.)[:, None]
    y = np.sin(x)
    w = np.ones_like(y)
    kwargs = dict(covariance_shape=np.eye(12))
    if defect == 'shape':
        y = y.ravel()
    elif defect == 'infinite':
        y[0, 0] = np.inf
    elif defect == 'mask':
        y = np.stack([y, y])
        y[0, 0, 0] = np.nan
    elif defect == 'weights':
        w[0] = -1
    elif defect == 'x':
        x[0] = np.nan
    elif defect == 'rank':
        x[:] = 1.
    elif defect == 'scale':
        kwargs['coefficient_scale'] = [[0., 1.]]
    elif defect == 'covariance':
        kwargs['covariance_shape'][0, 1] = .2
    elif defect == 'diagonal':
        kwargs['covariance_shape'] *= 2
    elif defect == 'short':
        w[:10] = 0
    elif defect == 'confidence':
        kwargs['confidence'] = True
    elif defect == 'degenerate':
        y[:] = 0.
    with pytest.raises(ValueError):
        joint_wls_gaussian_region(x, y, w, **kwargs)
