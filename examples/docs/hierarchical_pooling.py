"""Correlated partial pooling, checked against independent one-dimensional integration."""
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import halfnorm, multivariate_normal

from factorlasso import pool_gaussian_means


def main():
    """Integrate a synthetic hierarchy and verify its dispersion mean independently."""
    ids = pd.Index(['A', 'B', 'C', 'D'])
    means = pd.Series([.02, -.01, .04, .00], index=ids)
    covariance = pd.DataFrame(.0002*np.eye(4)+.0001*np.ones((4, 4)),
                              index=ids, columns=ids)
    groups = pd.Series('group', index=ids)
    posterior = pool_gaussian_means(
        means, covariance, groups, mean_prior_scale=.05,
        dispersion_prior_scale=.05, quadrature_order=128, draws=4096, seed=37)

    def density(tau, moment):
        """Direct multivariate normal density with centre integrated out."""
        total = covariance.to_numpy()+tau**2*np.eye(4)+.05**2*np.ones((4, 4))
        return (tau**moment*multivariate_normal.pdf(means, cov=total)
                *halfnorm.pdf(tau, scale=.05))

    mass = quad(density, 0, np.inf, args=(0,), epsabs=1e-8)[0]
    tau_mean = quad(density, 0, np.inf, args=(1,), epsabs=1e-8)[0]/mass
    np.testing.assert_allclose(posterior.groups.dispersion_mean.iloc[0], tau_mean, rtol=1e-4)
    np.testing.assert_allclose(np.exp(posterior.diagnostics['log_evidence']), mass, rtol=1e-4)
    exact = pool_gaussian_means(
        means, covariance*0, groups, mean_prior_scale=.05,
        dispersion_prior_scale=.05, quadrature_order=24, draws=100, seed=37)
    np.testing.assert_allclose(exact.draws, np.tile(means, (100, 1)), atol=1e-12)
    assert (posterior.estimates.posterior_sd > 0).all()
    print(posterior.groups)


if __name__ == '__main__':
    main()
