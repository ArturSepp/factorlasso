"""Offline weighted regression and mean inference with independent covariance checks."""
import numpy as np
from scipy.stats import chi2, norm, t as student_t

from factorlasso import (
    compute_ar1_interval, compute_wls_hac_geometry, compute_wls_hac_statistics,
    gaussian_critical_value, weighted_mean_hac_expectation,
    compute_weighted_mean_hac_geometry, gaussian_quadratic_quantile,
    quadratic_confidence_summary, linear_confidence_intervals, bootstrap_weighted_means,
)


def compare_standard_errors():
    """Separate observation dispersion, mean uncertainty and interval calibration."""
    observations = .5 + np.array([-2., -2., -1., -1., 1., 1., 2., 2.])
    n = len(observations)
    no_factors = np.empty((n, 0))
    classic_se = observations.std(ddof=1)/np.sqrt(n)
    hc = compute_wls_hac_statistics(no_factors, observations, hac_lags=0,
                                    return_covariance=True)
    hac = compute_wls_hac_statistics(no_factors, observations, hac_lags=2,
                                     return_covariance=True)
    hc_interval = linear_confidence_intervals(hc.coefficients, hc.covariance)
    hac_interval = linear_confidence_intervals(hac.coefficients, hac.covariance)

    residual = observations-observations.mean()
    # Independent pair sum; neither the package kernel nor its SVD enters this reference.
    pair_sum = sum(max(0., 1-abs(i-j)/3)*residual[i]*residual[j]
                   for i in range(n) for j in range(n))
    reference_variance = pair_sum/(n*(n-1))
    np.testing.assert_allclose(hac.covariance, [[reference_variance]], rtol=1e-13)
    np.testing.assert_allclose(hc.standard_error, [classic_se], rtol=1e-13)
    np.testing.assert_allclose(hc.coefficients, hac.coefficients, atol=1e-14)
    np.testing.assert_allclose(hac.coefficients, [.5], atol=1e-14)
    np.testing.assert_allclose(classic_se**2, 5/14, rtol=1e-13)
    np.testing.assert_allclose(reference_variance, 31/42, rtol=1e-13)
    for result, error in ((hc_interval, classic_se),
                          (hac_interval, np.sqrt(reference_variance))):
        expected = observations.mean()+norm.ppf(.975)*error*np.array([-1., 1.])
        np.testing.assert_allclose([result['lower'][0], result['upper'][0]], expected)
        assert result['status'] == 'approximate_conditional'
    classic_t_interval = observations.mean()+student_t.ppf(.975, n-1)*classic_se*np.array([-1., 1.])
    np.testing.assert_allclose(student_t.cdf((classic_t_interval-observations.mean())/classic_se,
                                            n-1), [.025, .975], atol=1e-12)

    # Reordering preserves the sample SD but changes the time-lag products.
    permuted = observations[[0, 6, 1, 7, 2, 4, 3, 5]]
    reordered = compute_wls_hac_statistics(no_factors, permuted, hac_lags=2)
    np.testing.assert_allclose(permuted.std(ddof=1), observations.std(ddof=1))
    assert reordered.standard_error[0] < classic_se < hac.standard_error[0]

    # Independent AR(1) covariance-matrix calculation and finite-lag variance identity.
    size, phi = 1000, .5
    covariance = phi**np.abs(np.arange(size)[:, None]-np.arange(size)[None, :])
    mean_weights = np.full(size, 1/size)
    finite_inflation = 1+2*sum((1-lag/size)*phi**lag for lag in range(1, size))
    np.testing.assert_allclose(size*(mean_weights @ covariance @ mean_weights),
                               finite_inflation, rtol=1e-13)
    limit_inflation = (1+phi)/(1-phi)
    np.testing.assert_allclose(limit_inflation, 3.)
    np.testing.assert_allclose(np.sqrt(finite_inflation), np.sqrt(3.), rtol=1e-3)
    np.testing.assert_allclose(np.sqrt(limit_inflation), 1.73, atol=.005)

    # Weight concentration explains iid variance; cross-date covariance is additional.
    ewma_weights = .9**np.arange(23, -1, -1)
    normalized = ewma_weights/ewma_weights.sum()
    effective_n = ewma_weights.sum()**2/(ewma_weights @ ewma_weights)
    np.testing.assert_allclose(normalized @ normalized, 1/effective_n)
    temporal = phi**np.abs(np.arange(24)[:, None]-np.arange(24)[None, :])
    assert normalized @ temporal @ normalized > 1/effective_n
    return dict(classic_se=classic_se, hac_se=hac.standard_error[0],
                hc_interval=hc_interval, hac_interval=hac_interval,
                classic_t_interval=classic_t_interval)


def run_example():
    """Verify coefficients, a full sandwich, a contrast and weighted mean calibration."""
    rng = np.random.default_rng(2026100402)
    x = rng.normal(size=(24, 2))
    design = np.column_stack((np.ones(24), x))
    weights = .9**np.arange(23, -1, -1)
    response = design @ [1., .8, -.3] + rng.normal(size=24)
    statistics = compute_wls_hac_statistics(
        x, response, weights, hac_lags=2, return_covariance=True)
    bread = np.linalg.inv(design.T @ (weights[:, None]*design))
    reference = bread @ design.T @ (weights*response)
    scores = weights[:, None]*design*(response-design @ reference)[:, None]
    meat = np.zeros((3, 3))
    for t in range(24):
        for s in range(24):
            meat += max(0., 1-abs(t-s)/3)*np.outer(scores[t], scores[s])
    covariance = 24/21*bread @ meat @ bread
    np.testing.assert_allclose(statistics.coefficients, reference, atol=1e-13)
    np.testing.assert_allclose(statistics.covariance, covariance, rtol=1e-12)

    geometry = compute_wls_hac_geometry(design, weights, hac_lags=2, coefficient=1)
    estimate, standard_error = geometry.statistics(response)
    critical = gaussian_critical_value(geometry, np.eye(24), alpha=.05)
    interval = (estimate-critical*standard_error, estimate+critical*standard_error)
    np.testing.assert_allclose(estimate, reference[1])
    np.testing.assert_allclose(standard_error**2, covariance[1, 1])
    assert interval[0][0] <= estimate[0] <= interval[1][0]

    contrast = np.array([0., 1., -1.])
    contrast_geometry = compute_wls_hac_geometry(design, weights, 2, contrast=contrast)
    contrast_estimate, contrast_se = contrast_geometry.statistics(response)
    np.testing.assert_allclose(contrast_estimate, contrast @ reference)
    np.testing.assert_allclose(contrast_se**2, contrast @ covariance @ contrast)

    mean = compute_wls_hac_statistics(np.empty((24, 0)), response, weights, hac_lags=2)
    mean_geometry = compute_wls_hac_geometry(np.ones((24, 1)), weights, 2, coefficient=0)
    np.testing.assert_allclose(mean.coefficients, np.average(response, weights=weights))
    np.testing.assert_allclose(mean_geometry.statistics(response)[1], mean.standard_error)
    ar_interval = compute_ar1_interval(mean_geometry, response, phi_max=.7, cells=41)
    assert ar_interval.retained_cells.all()
    assert ar_interval.delta == 0
    np.testing.assert_allclose(ar_interval.upper-ar_interval.lower,
                               2*ar_interval.critical_value*mean.standard_error)
    q = weights/weights.sum()*.8
    initialized = compute_weighted_mean_hac_geometry(q, calendar=np.arange(24), bandwidth=3.)
    np.testing.assert_allclose(initialized.statistics(response)[0], q @ response)
    np.testing.assert_allclose(initialized.target_contrast, [.8])
    np.testing.assert_allclose(gaussian_quadratic_quantile([2., 2.]), 2*chi2.ppf(.95, 2))
    summary = quadratic_confidence_summary([0., 0.], np.eye(2), 2*np.eye(2))
    assert summary['noise_adjusted'] == -4
    np.testing.assert_allclose(summary['upper'], 2*chi2.ppf(.95, 2))
    bootstrap = bootstrap_weighted_means(response[:, None], q[:, None],
                                         calendar=np.arange(24), bandwidth=3., draws=199, seed=4)
    np.testing.assert_allclose(bootstrap['estimate'], q @ response)
    np.testing.assert_allclose(bootstrap['covariance'][0, 0],
                               initialized.statistics(response)[1][0]**2)
    bootstrap_interval = linear_confidence_intervals(
        bootstrap['estimate'], bootstrap['covariance'], method='bootstrap_t',
        errors=bootstrap['errors'], replicate_standard_errors=bootstrap['standard_errors'])
    assert bootstrap_interval['status'] == 'bootstrap_approximation_unvalidated'
    panels = bootstrap_weighted_means(response[:, None], q[:, None],
        calendar=np.arange(24), bandwidth=3., draws=199, seed=4, return_residuals=True)
    np.testing.assert_array_equal(panels['errors'], bootstrap['errors'])
    panel_means = np.sum(panels['residual_draws'] * q[None, :, None], axis=1)
    np.testing.assert_allclose(panel_means-panels['estimate'], panels['errors'], atol=1e-14)
    expectation = weighted_mean_hac_expectation(q[:, None],
        observed=np.ones((24, 1), dtype=bool), calendar=np.arange(24),
        covariance_shape=np.eye(24), bandwidth=3.)
    np.testing.assert_allclose(expectation['true_covariance'], [[q @ q]])
    np.testing.assert_allclose(expectation['expected_hac_covariance'],
                               [[np.trace(initialized.quadratic)]])
    return statistics, interval, ar_interval


if __name__ == '__main__':
    compare_standard_errors()
    run_example()
    print('Classical/HC/HAC, weighted regression, mean and calibration checks passed.')
