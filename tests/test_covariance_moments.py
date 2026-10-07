"""Independent matrix-root and explicit score-pair references for covariance outputs."""
import numpy as np
import pytest
from scipy.linalg import sqrtm

from factorlasso.inference._covariance_moments import calibrate_covariance_moments
from factorlasso.inference._resampling import bootstrap_weighted_means


def test_moment_identity_batch_and_independent_matrix_root():
    """A finite ensemble has exactly the declared mean, without Monte Carlo tolerance."""
    m = np.array([[2., .7, -.1], [.7, 1., .2], [-.1, .2, 3.]])
    v = np.array([[1., -.2, .4], [-.2, 2., .1], [.4, .1, 1.]])
    vectors = np.sqrt(3)*np.linalg.cholesky(m).T
    estimates = np.einsum('bi,bj->bij', vectors, vectors)
    result = calibrate_covariance_moments(estimates, expected_covariance=m, target_covariance=v)
    independent = sqrtm(v) @ np.linalg.inv(sqrtm(m))
    np.testing.assert_allclose(result['transform'], independent, atol=3e-15)
    np.testing.assert_allclose(result['covariance'].mean(axis=0), v, atol=5e-15)
    assert np.linalg.eigvalsh(result['covariance']).min() > -1e-14
    assert result['relative_moment_error'] < 1e-13
    permutation = [2, 0, 1]
    permuted = calibrate_covariance_moments(estimates[:, permutation][:, :, permutation],
        expected_covariance=m[permutation][:, permutation],
        target_covariance=v[permutation][:, permutation])
    np.testing.assert_allclose(permuted['covariance'],
                               result['covariance'][:, permutation][:, :, permutation], atol=2e-14)


def test_diagonal_rescaling_identity_and_singular_target():
    """Known marginal moment corrections preserve the estimated correlations."""
    s = np.array([[2., .8], [.8, 1.]])
    result = calibrate_covariance_moments(s, expected_covariance=np.diag([2., 4.]),
                                         target_covariance=np.diag([8., 1.]))
    np.testing.assert_allclose(result['covariance'], [[8., .8], [.8, .25]])
    np.testing.assert_allclose(calibrate_covariance_moments(s, expected_covariance=s,
        target_covariance=s)['covariance'], s, atol=1e-14)
    zero = calibrate_covariance_moments(s, expected_covariance=s,
        target_covariance=np.zeros((2, 2)))
    np.testing.assert_array_equal(zero['covariance'], np.zeros((2, 2)))
    assert zero['relative_moment_error'] == 0


@pytest.mark.parametrize('s,m,v', [
    (np.eye(2), np.ones((2, 2)), np.eye(2)),
    (np.eye(2), np.diag([1., 1e-18]), np.eye(2)),
    (np.eye(2), np.eye(2), np.diag([1., -.1])),
    ([[1., .1], [.2, 1.]], np.eye(2), np.eye(2)),
    (np.diag([1., np.nan]), np.eye(2), np.eye(2)),
    (np.empty((0, 2, 2)), np.eye(2), np.eye(2)),
    (np.eye(2), np.eye(3), np.eye(2)),
    (np.ones(2), np.eye(2), np.eye(2)),
])
def test_invalid_covariance_moments(s, m, v):
    """Reject invalid or numerically unresolved matrices rather than silently repairing."""
    with pytest.raises(ValueError):
        calibrate_covariance_moments(s, expected_covariance=m, target_covariance=v)


def test_bootstrap_joint_covariance_matches_explicit_pairs_and_default_draws():
    """Unequal masses, native quarterly support, contrasts and scales share the same draws."""
    x = np.random.default_rng(721).normal(size=(18, 3))
    x[:2, 1] = np.nan
    x[np.arange(18) % 3 != 2, 2] = np.nan
    q = np.where(np.isfinite(x), .9**np.arange(17, -1, -1)[:, None], 0.)*.05
    c, scale = np.array([[1., -.5, .3], [.2, .7, -.1]]), np.array([12., 4., 2.])
    kwargs = dict(calendar=np.arange(18), scale=scale, contrasts=c,
                  draws=100, seed=93, batch_size=13)
    plain = bootstrap_weighted_means(x, q, **kwargs)
    full = bootstrap_weighted_means(x, q, return_covariances=True, return_residuals=True, **kwargs)
    assert 'covariance_draws' not in plain
    for key in ('errors', 'standard_errors', 'covariance', 'estimate'):
        np.testing.assert_array_equal(plain[key], full[key])
    np.testing.assert_allclose(full['covariance_draws'].diagonal(axis1=1, axis2=2),
                               full['standard_errors']**2, atol=1e-14)
    kernel = np.maximum(1-abs(np.arange(18)[:, None]-np.arange(18))/6, 0)
    counts = np.isfinite(x).sum(axis=0)
    for b in range(3):
        sample = full['residual_draws'][b]
        mean = np.nansum(q*sample, axis=0)/q.sum(axis=0)
        scores = q*np.nan_to_num(sample-mean)*scale*np.sqrt(counts/(counts-1))
        targets = scores @ c.T
        reference = [[sum(targets[t, i]*targets[s, j]*kernel[t, s]
                          for t in range(18) for s in range(18))
                      for j in range(2)] for i in range(2)]
        np.testing.assert_allclose(full['covariance_draws'][b], reference, atol=1e-14)
    assert np.linalg.eigvalsh(full['covariance_draws']).min() >= -1e-13
    with pytest.raises(ValueError, match='boolean'):
        bootstrap_weighted_means(x, q, return_covariances=1, **kwargs)
