"""Independent references for direction-uniform scalar Gaussian capacity bounds."""
import numpy as np
import pytest
from scipy.stats import ncx2, norm

from factorlasso import quadratic_scalar_confidence_summary, positive_part_confidence_summary
from factorlasso.inference._scalar_capacity import _endpoint


def test_isotropic_quadratic_coverage_against_exact_noncentral_chi_square():
    """Deterministic exact distribution quantiles check coverage at/away from the null."""
    probabilities = (np.arange(1000)+.5)/1000
    for signal in [0., .05, 2., 50.]:
        observed = ncx2.ppf(probabilities, 6, signal)
        means = np.column_stack([np.sqrt(observed), np.zeros((len(observed), 5))])
        result = quadratic_scalar_confidence_summary(means, np.eye(6), np.eye(6))
        covered = (result['lower'] <= signal) & (signal <= result['upper'])
        assert covered.mean() >= .95
        np.testing.assert_allclose(result['noise_adjusted'], observed-6, rtol=1e-12)


def test_positive_part_boundary_against_exact_scalar_normal_distribution():
    """Include the atom at zero and negative means, not a fixed positive set."""
    errors = norm.ppf((np.arange(1000)+.5)/1000)
    for true_mean in [-3., -.1, 0., .1, 3.]:
        means = (true_mean+errors)[:, None]
        result = positive_part_confidence_summary(means, [[1.]], [2.])
        target = 2*max(0., true_mean)**2
        assert np.mean((result['lower'] <= target) & (target <= result['upper'])) >= .95
        np.testing.assert_array_equal(result['observed'], 2*np.maximum(means[:, 0], 0)**2)
        assert np.isnan(result['noise']) and np.isnan(result['noise_adjusted']).all()


def test_direction_envelope_against_exact_gaussian_mgf():
    """Verify the nuisance-direction inequality independently by matrix integration formula."""
    covariance = np.array([[2., -.6, .1], [-.6, 1., .2], [.1, .2, .3]])
    eigenvalues = np.linalg.eigvalsh(covariance)
    largest = eigenvalues[-1]
    for theta in [np.array([2., -1., .5]), np.array([0., .2, 3.])]:
        signal = theta @ theta
        for t in [-2., -.2, .01, .4/largest]:
            matrix = np.eye(3)-2*t*covariance
            direct = -.5*np.linalg.slogdet(matrix)[1]+t*(theta @ np.linalg.solve(matrix, theta))
            envelope = -.5*np.log1p(-2*t*eigenvalues).sum()+t*signal/(1-2*t*largest)
            assert direct <= envelope+1e-12


def test_endpoints_against_dense_parameter_grid():
    """Independently invert the original t-based MGF inequalities, not the u substitution."""
    eigenvalues = np.array([.1, .3, 1.2])
    tail = np.log(.025)
    for observed in [.01, 2., 30.]:
        for upper in [False, True]:
            t = (-np.geomspace(1e-6, 1e6, 40000) if upper
                 else np.linspace(1e-6, (1-1e-6)/(2*eigenvalues[-1]), 40000))
            c = -.5*np.log1p(-2*t[:, None]*eigenvalues).sum(axis=1)
            candidates = (tail+t*observed-c)*(1-2*t*eigenvalues[-1])/t
            reference = max(0., candidates.min() if upper else candidates.max())
            actual = _endpoint(observed, eigenvalues, tail, upper)
            np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=1e-7)


def test_correlation_is_retained_and_coordinate_scaling_is_invariant():
    """Scaling alpha, error covariance and inverse metric together leaves capacity unchanged."""
    a = np.array([.3, -.2, .5])
    v = np.array([[.1, .08, -.02], [.08, .2, .03], [-.02, .03, .08]])
    weights = np.array([2., .5, 1.])
    scale = np.array([100., .1, 3.])
    for positive in [False, True]:
        function = (positive_part_confidence_summary if positive
                    else quadratic_scalar_confidence_summary)
        metric = weights if positive else np.diag(weights)
        changed = weights/scale**2 if positive else np.diag(weights/scale**2)
        first = function(a, v, metric)
        second = function(a*scale, v*np.outer(scale, scale), changed)
        for key in ['observed', 'lower', 'upper', 'largest_error_eigenvalue']:
            np.testing.assert_allclose(first[key], second[key], rtol=1e-9, atol=1e-12)
        diagonal = function(a, np.diag(np.diag(v)), metric)
        assert not np.isclose(first['upper'], diagonal['upper'])


def test_zero_metric_covariance_and_singular_covariance():
    """Degenerate coordinates do not add noise or force a covariance repair."""
    a = np.array([1., -2.])
    q = quadratic_scalar_confidence_summary(a, np.eye(2), np.zeros((2, 2)))
    assert q['observed'] == q['lower'] == q['upper'] == 0
    for function, metric in [(quadratic_scalar_confidence_summary, np.eye(2)),
                             (positive_part_confidence_summary, np.ones(2))]:
        result = function(a, np.zeros((2, 2)), metric)
        assert result['observed'] == result['lower'] == result['upper']
        singular = function(a, np.ones((2, 2)), metric)
        assert singular['uncertainty_rank'] == 1
        assert singular['lower'] <= singular['upper']


@pytest.mark.parametrize('bad', [[], [np.nan], [[1., np.inf]], np.ones((2, 2, 2))])
def test_invalid_means_fail(bad):
    """Malformed and nonfinite inputs cannot silently become a capacity interval."""
    with pytest.raises(ValueError, match='mean'):
        positive_part_confidence_summary(bad, [[1.]], [1.])


def test_invalid_covariance_metric_weights_confidence_fail():
    """The public contracts reject indefinite matrices and mismatched metric axes."""
    with pytest.raises(ValueError):
        quadratic_scalar_confidence_summary([1., 2.], [[1., 2.], [2., 1.]], np.eye(2))
    with pytest.raises(ValueError):
        quadratic_scalar_confidence_summary([1., 2.], np.eye(2), -np.eye(2))
    with pytest.raises(ValueError):
        positive_part_confidence_summary([1., 2.], np.eye(2), [1.])
    with pytest.raises(ValueError):
        positive_part_confidence_summary([1.], [[1.]], [-1.])
    with pytest.raises(ValueError):
        positive_part_confidence_summary([1.], [[1.]], [1.], confidence=1.)
