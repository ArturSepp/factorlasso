"""Offline example of optional prior inference and limiting-risk diagnostics."""
import numpy as np
import pandas as pd

from factorlasso import (
    compute_ar1_prior_interval, compute_expert_prior_statistics,
    compute_prior_hac_geometry, gaussian_dominance_information,
    gaussian_prior_critical_value, two_factor_limit_risk, two_factor_minimax_radius,
)


def run_example():
    """Check prior SE parity, interval construction and the conditional radius formula."""
    rng = np.random.default_rng(7292026)
    factors = rng.normal(size=(24, 2))
    design = np.column_stack((np.ones(24), factors))
    phi = .317
    covariance = phi**np.abs(np.arange(24)[:, None]-np.arange(24)[None, :])
    response = design @ [1., .8, -.3]+np.linalg.cholesky(covariance) @ rng.normal(size=24)
    span = 12
    weights = (1-2/(span+1))**np.arange(23, -1, -1)
    geometry = compute_prior_hac_geometry(design, weights, hac_lags=1, coefficient=1)
    estimate, se = geometry.statistics(response)
    known_k = gaussian_prior_critical_value(geometry, covariance, alpha=.05)
    full = compute_ar1_prior_interval(geometry, response, phi_max=.7, cells=41)
    adaptive = compute_ar1_prior_interval(
        geometry, response, phi_max=.7, cells=41, adaptive=True, delta=.01)

    # Independent production helper uses a different SVD/influence implementation.
    reference = compute_expert_prior_statistics(
        pd.DataFrame(factors, columns=['first', 'second']), pd.Series(response),
        span=span, hac_lags=1)
    np.testing.assert_allclose(estimate[0], reference.loc['first', 'reference_beta'])
    np.testing.assert_allclose(se[0], reference.loc['first', 'se_hac'])
    assert full.critical_value[0] >= known_k
    assert full.delta == 0 and adaptive.delta == .01
    assert full.retained_cells.all() and adaptive.retained_cells.any()
    np.testing.assert_allclose(full.upper-full.lower, 2*full.critical_value*se)

    radius = two_factor_minimax_radius(secondary_bound=0, bias_bound=8, penalty=4)
    np.testing.assert_allclose(radius, 4/7)
    # Direct endpoint algebra, independent of the package's piecewise risk routine.
    left, right = radius-8, radius+8
    endpoint_risk = max(1+left**2, 1+right**2-4*right+16)
    np.testing.assert_allclose(two_factor_limit_risk(radius, 0, 8, 4), endpoint_risk)
    assert two_factor_limit_risk(radius, 0, 8, 4) < two_factor_limit_risk(0, 0, 8, 4)
    information = gaussian_dominance_information(factors, covariance)
    precision = np.linalg.inv(covariance)
    one = np.ones(24)
    projected = precision-np.outer(precision @ one, one @ precision)/(one @ precision @ one)
    np.testing.assert_allclose(information.information, factors.T @ projected @ factors)
    assert information.risk_rate(0) == information.sole_variance
    return radius, full, adaptive


if __name__ == '__main__':
    run_example()
    print('Prior SE parity, Gaussian/AR intervals and limiting-risk checks passed.')
