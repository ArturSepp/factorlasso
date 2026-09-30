"""Independent numerical checks for opt-in prior calibration and radius diagnostics."""
import numpy as np
import pandas as pd
import pytest
from scipy.linalg import null_space
from scipy.stats import t

from factorlasso.prior_bounds import compute_expert_prior_statistics
from factorlasso.prior_inference import (
    _angular_log_density, _ar_covariance, _gaussian_critical, _pivot_tail,
    compute_ar1_prior_interval, compute_prior_hac_geometry, gaussian_prior_critical_value,
)
from factorlasso.prior_risk import (
    gaussian_dominance_information, two_factor_limit_risk, two_factor_minimax_radius,
)


def panel(n=40):
    """Construct fixed factors and responses independently of all calibration grids."""
    rng = np.random.default_rng(2909202641)
    x = rng.normal(size=(n, 2))
    d = np.column_stack((np.ones(n), x))
    y = d @ [2., .8, -.3]+rng.normal(size=n)
    return d, y


@pytest.mark.parametrize('span,lags', [(None, 0), (12, 3), (36, 1)])
def test_hac_geometry_matches_existing_statistics(span, lags):
    """Quadratic HAC equals the existing SVD/influence estimator including both weights."""
    d, y = panel()
    weights = None if span is None else (1-2/(span+1))**np.arange(len(y)-1, -1, -1)
    geo = compute_prior_hac_geometry(d, weights, lags)
    estimate, se = geo.statistics(y)
    reference = compute_expert_prior_statistics(
        pd.DataFrame(d[:, 1:], columns=['a', 'b']), pd.Series(y), span=span, hac_lags=lags)
    np.testing.assert_allclose(estimate[0], reference.loc['a', 'reference_beta'], atol=2e-13)
    np.testing.assert_allclose(se[0], reference.loc['a', 'se_hac'], rtol=2e-12)
    np.testing.assert_allclose(se[0]**2, y @ geo.quadratic @ y, rtol=2e-12)
    if weights is not None:
        scaled = compute_prior_hac_geometry(d, weights*1e-8, lags)
        np.testing.assert_allclose(scaled.quadratic, geo.quadratic, atol=2e-14)


def test_gaussian_calibration_student_and_simulation():
    """The pivot reproduces Student's independent denominator and weighted-HAC simulation."""
    degrees = 11
    h = np.r_[1., np.zeros(degrees)]
    q = np.diag(np.r_[0., np.ones(degrees)/degrees])
    critical = _gaussian_critical(h, q, np.eye(degrees+1), .05)
    np.testing.assert_allclose(critical, t.ppf(.975, degrees), rtol=1e-9)
    d, _ = panel(24)
    geometry = compute_prior_hac_geometry(d, .9**np.arange(23, -1, -1), 2)
    covariance = _ar_covariance(24, .317)
    k = gaussian_prior_critical_value(geometry, covariance)
    rng = np.random.default_rng(2909202642)
    noise = rng.normal(size=(60000, 24)) @ np.linalg.cholesky(covariance).T
    estimate, se = geometry.statistics(noise)
    assert abs(np.mean(np.abs(estimate) > k*se)-.05) < .004
    np.testing.assert_allclose(gaussian_prior_critical_value(geometry, covariance*7), k)


def test_angular_density_dense_nullspace_reference():
    """Fast AR direction densities agree with an independent dense contrast calculation."""
    d, y = panel(25)
    responses = np.vstack((y, y*3+d @ [7., -2., .1]))
    phis = np.array([-.7, -.143, 0., .317, .7])
    actual = _angular_log_density(d, responses, phis)
    u = null_space(d.T)
    projected = responses @ u
    projected /= np.linalg.norm(projected, axis=1)[:, None]
    expected = []
    for phi in phis:
        covariance = u.T @ _ar_covariance(len(y), phi) @ u
        quadratic = np.einsum('ij,ji->i', projected, np.linalg.solve(covariance, projected.T))
        expected.append(-.5*np.linalg.slogdet(covariance)[1]
                        -.5*u.shape[1]*np.log(quadratic))
    np.testing.assert_allclose(actual, np.column_stack(expected), atol=3e-12)
    np.testing.assert_allclose(actual[0], actual[1], atol=3e-12)


@pytest.mark.parametrize('adaptive', [False, True])
def test_ar_interval_continuum_certificate_and_coverage(adaptive):
    """Off-grid shape tails respect cell allocation; fresh batch coverage respects the band."""
    d, _ = panel(16)
    geometry = compute_prior_hac_geometry(d, .93**np.arange(15, -1, -1), 1)
    phi = .317
    covariance = _ar_covariance(len(d), phi)
    rng = np.random.default_rng(2909202643)
    noise = rng.normal(size=(3000, len(d))) @ np.linalg.cholesky(covariance).T
    responses = noise+d @ [1., .8, -.3]
    result = compute_ar1_prior_interval(geometry, responses, cells=41, adaptive=adaptive)
    assert np.mean((result.lower <= .8) & (result.upper >= .8)) >= .94
    cell = np.searchsorted(result.phi_edges, phi)-1
    chol = np.linalg.cholesky(covariance)
    tail = _pivot_tail(result.cell_critical_values[cell], chol.T @ geometry.linear,
                       chol.T @ geometry.quadratic @ chol)
    assert tail <= .05-result.delta+1e-8
    half = np.arctanh(.7)/41
    expected = (.05-(.01 if adaptive else 0))*np.exp(-2*len(d)*half)
    assert result.calibration_alpha == pytest.approx(expected)
    if adaptive:
        assert np.mean(result.retained_cells[:, cell]) >= .985
    else:
        assert result.retained_cells.all() and result.delta == 0
    for value in (result.phi_edges[cell], phi, result.phi_edges[cell+1]):
        center = _ar_covariance(len(d), result.phi_centres[cell])
        target = _ar_covariance(len(d), value)
        assert np.linalg.eigvalsh(target-np.exp(-2*half)*center)[0] >= -1e-12
        assert np.linalg.eigvalsh(np.exp(2*half)*center-target)[0] >= -1e-12


def test_radius_formula_against_dense_independent_search():
    """The closed form minimizes a separately written piecewise risk over dense radii."""
    for d in (0., .2, 1., 4.):
        for tau in (0., .3, 2., 8.):
            for lam in (0., 1., 4.):
                optimum = two_factor_minimax_radius(d, tau, lam)
                radii = np.linspace(0, max(1., tau+lam), 10001)
                grid_risk = np.zeros_like(radii)
                for eta in (0., d):
                    for bias in (-tau, tau):
                        a = d+radii-bias
                        first = 1+(eta-a)**2+eta**2
                        second = 1+2*eta**2-2*a*eta+a*a-a*lam+lam*lam
                        grid_risk = np.maximum(grid_risk, np.where(a <= lam, first, second))
                exact = two_factor_limit_risk(optimum, d, tau, lam)
                assert exact <= grid_risk.min()+1e-10
    assert two_factor_minimax_radius(0, 8, 4) == pytest.approx(4/7)
    m = .8/np.sqrt(2*np.pi)
    assert two_factor_limit_risk(m+.05, 0, 0, .1, cross_moment=m) < 1


def test_information_matches_profiled_precision_and_scaling():
    """Whitened information matches an independent inverse-covariance projection."""
    d, _ = panel(35)
    x = d[:, 1:].copy()
    x[:, 1] = .999*x[:, 0]+.02*x[:, 1]
    covariance = _ar_covariance(len(d), .4)*.7
    result = gaussian_dominance_information(x, covariance)
    precision = np.linalg.inv(covariance)
    one = np.ones(len(d))
    projected = precision-np.outer(precision @ one, one @ precision)/(one @ precision @ one)
    expected = x.T @ projected @ x
    np.testing.assert_allclose(result.information, expected, rtol=2e-13)
    np.testing.assert_allclose(result.secondary_variance, np.linalg.inv(expected)[1, 1], rtol=1e-9)
    assert result.risk_rate(0) == result.sole_variance
    scaled = gaussian_dominance_information(x, covariance*9)
    np.testing.assert_allclose(scaled.secondary_variance, result.secondary_variance*9)


@pytest.mark.parametrize('kwargs', [dict(hac_lags=True), dict(hac_lags=-1),
                                    dict(coefficient=3), dict(coefficient=True),
                                    dict(weights=[1, 2]), dict(weights=np.zeros(8))])
def test_invalid_geometry_options(kwargs):
    """Invalid calendar, index and residual-rank options fail explicitly."""
    d, _ = panel(8)
    with pytest.raises(ValueError):
        compute_prior_hac_geometry(d, **kwargs)


@pytest.mark.parametrize('kwargs', [dict(cells=True), dict(cells=2), dict(phi_max=np.nan),
                                    dict(phi_max=1), dict(alpha=0), dict(adaptive='yes'),
                                    dict(adaptive=True, delta=.06), dict(cells=3)])
def test_invalid_ar_options(kwargs):
    """Reject invalid probabilities, shapes and a numerically uncertifiable coarse grid."""
    d, y = panel(80)
    geometry = compute_prior_hac_geometry(d)
    with pytest.raises(ValueError):
        compute_ar1_prior_interval(geometry, y, **kwargs)


def test_invalid_inputs_fail_instead_of_silently_changing_target():
    """Missing rows, singular means and indefinite shapes cannot silently inherit coverage."""
    d, y = panel(10)
    geometry = compute_prior_hac_geometry(d)
    y[2] = np.nan
    with pytest.raises(ValueError, match='gaps'):
        geometry.statistics(y)
    with pytest.raises(ValueError):
        compute_prior_hac_geometry(np.ones((10, 2)))
    with pytest.raises(ValueError):
        gaussian_prior_critical_value(geometry, -np.eye(10))
    with pytest.raises(ValueError):
        gaussian_dominance_information(np.ones((10, 2)), np.eye(10))
    for bad in (-1., np.nan, np.inf):
        with pytest.raises(ValueError):
            two_factor_minimax_radius(0, bad, 1)
    with pytest.raises(ValueError):
        two_factor_limit_risk(0, 0, 0, 1, cross_moment=4.)
