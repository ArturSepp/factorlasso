"""Known distribution, continuum calibration and prior compatibility checks."""
import pickle

import numpy as np
import pytest
from scipy.stats import t

from factorlasso.inference._ar1 import compute_ar1_interval
from factorlasso.inference._gaussian import gaussian_critical_value
from factorlasso.inference._geometry import LinearHacGeometry, compute_wls_hac_geometry
from factorlasso.priors import (
    Ar1PriorInterval, PriorHacGeometry, compute_ar1_prior_interval,
    compute_prior_hac_geometry, gaussian_prior_critical_value,
)


def test_generic_gaussian_reproduces_student_distribution():
    """An independent chi-squared denominator yields the known Student quantile."""
    degrees = 11
    h = np.r_[1., np.zeros(degrees)]
    residual = np.diag(np.r_[0., np.ones(degrees)])
    geometry = LinearHacGeometry(h[:, None], h, residual/degrees, residual, np.ones(1))
    critical = gaussian_critical_value(geometry, np.eye(degrees+1))
    np.testing.assert_allclose(critical, t.ppf(.975, degrees), rtol=1e-9)
    np.testing.assert_allclose(gaussian_critical_value(geometry, 7*np.eye(degrees+1)), critical)


@pytest.mark.parametrize('adaptive', [False, True])
def test_generic_intervals_preserve_every_prior_audit_field(adaptive):
    """Adapters preserve old result types, shapes, masks and allocated probabilities."""
    rng = np.random.default_rng(405)
    d = np.column_stack((np.ones(16), rng.normal(size=16)))
    y = d @ [1., .8] + rng.normal(size=(2, 16))
    weights = .93**np.arange(15, -1, -1)
    old = compute_prior_hac_geometry(d, weights, 1)
    new = compute_wls_hac_geometry(d, weights, 1, coefficient=1)
    assert type(old) is PriorHacGeometry
    assert type(pickle.loads(pickle.dumps(old))) is PriorHacGeometry
    assert PriorHacGeometry.__module__ == 'factorlasso.priors._inference'
    assert Ar1PriorInterval.__module__ == 'factorlasso.priors._inference'
    np.testing.assert_allclose(old.statistics(y), new.statistics(y))
    covariance = .317**np.abs(np.arange(16)[:, None]-np.arange(16))
    np.testing.assert_allclose(gaussian_prior_critical_value(old, covariance),
                               gaussian_critical_value(new, covariance))
    before = compute_ar1_prior_interval(old, y, cells=41, adaptive=adaptive)
    after = compute_ar1_interval(new, y, cells=41, adaptive=adaptive)
    assert type(before) is Ar1PriorInterval
    assert type(pickle.loads(pickle.dumps(before))) is Ar1PriorInterval
    for field in vars(before):
        np.testing.assert_array_equal(getattr(before, field), getattr(after, field))
    expected_alpha = (.05-(.01 if adaptive else 0))*np.exp(-32*np.arctanh(.7)/41)
    assert after.calibration_alpha == pytest.approx(expected_alpha)
