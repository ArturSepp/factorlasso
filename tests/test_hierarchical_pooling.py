"""Independent Gaussian conditioning, quadrature, units and axis checks for pooling."""
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.linalg import cho_factor
from scipy.stats import halfnorm, multivariate_normal

from factorlasso import pool_gaussian_means
from factorlasso.inference._pooling import _conditional_simulation, _marginal_factor


def fixture_data():
    """Three nonexchangeable observations with two explicitly named pooling groups."""
    ids = pd.Index(['a', 'b', 'c'])
    mean = pd.Series([-.2, .0, .3], index=ids)
    covariance = pd.DataFrame([[.04, .015, -.005], [.015, .09, .02], [-.005, .02, .06]],
                              index=ids, columns=ids)
    groups = pd.Series(['one', 'one', 'two'], index=ids)
    return mean, covariance, groups


def test_conditional_joint_draws_against_block_precision_reference():
    """Prior-times-likelihood precision independently checks correlated conditional draws."""
    y, v, _ = fixture_data()
    z = np.array([[1., 0.], [1., 0.], [0., 1.]])
    tau, scales, mu = np.array([.2, .4]), np.array([.3, .5]), np.array([.05, -.05])
    codes = np.array([0, 0, 1])
    d, h = np.diag(tau[codes]**2), np.diag(scales**2)
    prior = np.block([[d+z @ h @ z.T, z @ h], [h @ z.T, h]])
    precision = np.linalg.inv(prior)
    precision[:3, :3] += np.linalg.inv(v)
    reference_cov = np.linalg.inv(precision)
    reference_mean = reference_cov @ (
        np.linalg.solve(prior, np.r_[z @ mu, mu]) + np.r_[np.linalg.solve(v, y), [0., 0.]])
    factor = cho_factor(v+d+z @ h @ z.T, lower=True)
    a, m = _conditional_simulation(y.to_numpy(), np.linalg.cholesky(v), z, codes, mu,
        scales, tau, factor, 120000, np.random.default_rng(810))
    joint = np.column_stack([a, m])
    np.testing.assert_allclose(joint.mean(axis=0), reference_mean, atol=.002)
    np.testing.assert_allclose(np.cov(joint, rowvar=False), reference_cov, atol=.001)


@pytest.mark.parametrize('adapt,order', [(False, 64), (True, 128)])
def test_one_group_quadrature_against_adaptive_integral(adapt, order):
    """Analytical compound-symmetry likelihood plus adaptive integration supplies a reference."""
    y, _, groups = fixture_data()
    v = pd.DataFrame(.04*np.eye(3), index=y.index, columns=y.index)
    groups[:] = 'one'
    sigma_mu, sigma_tau, location = .4, .5, .05

    def integrand(tau, moment):
        """Use independent eigenvalues in the mean and centred observation directions."""
        x = y.to_numpy()-location
        a = .04+tau*tau
        b = a+3*sigma_mu*sigma_mu
        loglik = -.5*(3*np.log(2*np.pi)+2*np.log(a)+np.log(b)
                      + np.sum((x-x.mean())**2)/a+3*x.mean()**2/b)
        return tau**moment*np.exp(loglik)*halfnorm.pdf(tau, scale=sigma_tau)

    mass = quad(integrand, 0, np.inf, args=(0,), epsabs=1e-11)[0]
    reference = quad(integrand, 0, np.inf, args=(1,), epsabs=1e-11)[0]/mass
    result = pool_gaussian_means(y, v, groups, mean_prior_scale=sigma_mu,
        dispersion_prior_scale=sigma_tau, mean_prior_location=location,
        quadrature_order=order, draws=3000, seed=41, adapt_quadrature=adapt)
    np.testing.assert_allclose(np.exp(result.diagnostics['log_evidence']), mass, rtol=2e-6)
    np.testing.assert_allclose(result.groups.dispersion_mean.iloc[0], reference, rtol=1e-5)


def test_marginal_evidence_uses_all_correlations():
    """Direct multivariate Gaussian densities retain cross-group covariance entries."""
    y, v, _ = fixture_data()
    z = np.array([[1., 0.], [1., 0.], [0., 1.]])
    base = v.to_numpy()+.2**2*(z @ z.T)
    variances = np.array([.1, .1, .3])**2
    _, observed = _marginal_factor(base, variances, y.to_numpy())
    expected = multivariate_normal.logpdf(y, cov=base+np.diag(variances))
    np.testing.assert_allclose(observed, expected, rtol=1e-12)
    _, wrong = _marginal_factor(np.diag(np.diag(v))+.2**2*(z @ z.T), variances,
                                y.to_numpy())
    assert abs(observed-wrong) > .01


def test_scale_equivariance_reproducibility_and_inputs_unchanged():
    """Changing all units preserves the posterior quadrature distribution exactly."""
    y, v, groups = fixture_data()
    originals = [x.copy() for x in [y, v, groups]]
    first = pool_gaussian_means(y, v, groups, mean_prior_scale=.3,
        dispersion_prior_scale=.4, quadrature_order=12, draws=1000, seed=5)
    again = pool_gaussian_means(y, v, groups, mean_prior_scale=.3,
        dispersion_prior_scale=.4, quadrature_order=12, draws=1000, seed=5)
    scaled = pool_gaussian_means(100*y, 10000*v, groups, mean_prior_scale=30.,
        dispersion_prior_scale=40., quadrature_order=12, draws=1000, seed=5)
    pd.testing.assert_frame_equal(first.draws, again.draws)
    np.testing.assert_allclose(first.quadrature.posterior_weight,
                               scaled.quadrature.posterior_weight, atol=1e-14)
    np.testing.assert_allclose(first.groups.dispersion_mean,
                               scaled.groups.dispersion_mean/100, rtol=1e-12)
    pd.testing.assert_series_equal(y, originals[0])
    pd.testing.assert_frame_equal(v, originals[1])
    pd.testing.assert_series_equal(groups, originals[2])


def test_exact_observations_have_zero_latent_uncertainty():
    """PSD zero measurement covariance fixes every latent mean despite uncertain hyperparameters."""
    y, v, groups = fixture_data()
    result = pool_gaussian_means(y, v*0, groups, mean_prior_scale=.3,
        dispersion_prior_scale=.4, quadrature_order=16, draws=2000)
    np.testing.assert_allclose(result.draws, np.tile(y, (2000, 1)), atol=1e-12)
    assert (result.groups.mean_sd > .02).all()
    assert (result.group_scale_draws.std() > .02).all()


@pytest.mark.parametrize('failure', ['mean_nan', 'duplicate', 'cov_axes', 'group_axes',
                                   'missing_group', 'negative_cov', 'zero_prior',
                                   'prior_axes', 'node_guard', 'bool_order'])
def test_invalid_inputs_fail_closed(failure):
    """Do not align, repair, drop or silently ignore invalid likelihood/prior metadata."""
    y, v, groups = fixture_data()
    kwargs = dict(mean_prior_scale=.3, dispersion_prior_scale=.4,
                  quadrature_order=4, draws=20)
    if failure == 'mean_nan':
        y.iloc[0] = np.nan
    elif failure == 'duplicate':
        y.index = ['a', 'a', 'c']
    elif failure == 'cov_axes':
        v = v.iloc[::-1]
    elif failure == 'group_axes':
        groups = groups.iloc[::-1]
    elif failure == 'missing_group':
        groups.iloc[0] = None
    elif failure == 'negative_cov':
        v.iloc[0, 0] = -1.
    elif failure == 'zero_prior':
        kwargs['dispersion_prior_scale'] = 0.
    elif failure == 'prior_axes':
        kwargs['mean_prior_scale'] = pd.Series([.3, .3], index=['two', 'one'])
    elif failure == 'node_guard':
        kwargs['max_nodes'] = 2
    elif failure == 'bool_order':
        kwargs['quadrature_order'] = True
    with pytest.raises(ValueError):
        pool_gaussian_means(y, v, groups, **kwargs)
