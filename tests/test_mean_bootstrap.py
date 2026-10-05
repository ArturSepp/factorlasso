"""Dependence-preserving mean resampling checked against scalar reconstruction."""
import numpy as np
import pytest

from factorlasso.inference._resampling import bootstrap_weighted_means
from factorlasso.inference._intervals import linear_confidence_intervals
from factorlasso.utils._hac import bartlett_kernel


def test_bootstrap_reconstructs_residuals_and_reestimates_every_variance():
    """Independent Cholesky multipliers and explicit scalar pairs verify the draws."""
    x = np.random.default_rng(11).normal(size=(12, 2))
    x[:2, 1] = np.nan
    q = np.where(np.isfinite(x), .9**np.arange(11, -1, -1)[:, None], 0.)/10
    times = np.arange(12)
    contrasts = np.array([[1., -.5], [.2, 1.]])
    result = bootstrap_weighted_means(x, q, calendar=times, bandwidth=3.,
                                      contrasts=contrasts, draws=120, seed=87, batch_size=13)
    kernel = bartlett_kernel(times, 3.)
    multipliers = np.random.default_rng(87).normal(size=(120, 12)) @ np.linalg.cholesky(kernel).T
    mean = np.nansum(q*x, axis=0)/q.sum(axis=0)
    counts = np.isfinite(x).sum(axis=0)
    for b in range(4):
        sample = mean + (x-mean)*multipliers[b, :, None]
        alpha = np.nansum(q*sample, axis=0)
        center = alpha/q.sum(axis=0)
        influence = q*np.nan_to_num(sample-center)*np.sqrt(counts/(counts-1))
        target_scores = influence @ contrasts.T
        variance = np.array([sum(target_scores[t, j]*target_scores[s, j]*kernel[t, s]
                                 for t in range(12) for s in range(12)) for j in range(2)])
        np.testing.assert_allclose(result['errors'][b], contrasts @ (alpha-q.sum(axis=0)*mean))
        np.testing.assert_allclose(result['standard_errors'][b]**2, variance, atol=1e-14)
    other = bootstrap_weighted_means(x, q, calendar=times, bandwidth=3., contrasts=contrasts,
                                     draws=120, seed=87, batch_size=60)
    np.testing.assert_allclose(result['errors'], other['errors'], atol=1e-14)
    np.testing.assert_allclose(result['standard_errors'], other['standard_errors'], atol=1e-14)


def test_studentized_calibration_and_family_scope():
    """The supplied replicate SEs, not a Gaussian multiplier, determine the interval."""
    errors = np.column_stack([np.linspace(-3, 3, 200), np.linspace(5, -5, 200)])
    errors_se = np.ones_like(errors)*.5
    result = linear_confidence_intervals([1., 2.], np.eye(2), method='bootstrap_t',
                                         errors=errors, replicate_standard_errors=errors_se)
    critical = np.quantile(np.abs(errors/errors_se), .95, axis=0)
    np.testing.assert_allclose(result['lower'], [1., 2.]-critical)
    assert result['scope'] == 'pointwise'
    joint = linear_confidence_intervals([1., 2.], np.eye(2), method='bootstrap_t',
                                        errors=errors, replicate_standard_errors=errors_se,
                                        simultaneous=True)
    expected = np.quantile(np.max(np.abs(errors/errors_se), axis=1), .95)
    assert np.all(joint['critical_value'] == expected)
    assert joint['status'] == 'bootstrap_approximation_unvalidated'
    with pytest.raises(ValueError):
        linear_confidence_intervals([1., 2.], np.eye(2), method='bootstrap_t', errors=errors)


def test_unknown_or_invalid_bootstrap_inputs_rejected():
    """A missing score is not an observed zero and covariance cannot be non-PSD."""
    x = np.ones((4, 2))
    x[0, 0] = np.nan
    with pytest.raises(ValueError):
        bootstrap_weighted_means(x, np.ones_like(x), calendar=np.arange(4), draws=100)
    with pytest.raises(ValueError):
        linear_confidence_intervals([1., 2.], np.diag([1., -.1]))
