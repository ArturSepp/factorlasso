"""Exact finite-mask expectations against independent dense score operators."""
import numpy as np
import pytest

from factorlasso.inference._mean import weighted_mean_hac_expectation


def fixture():
    """Unequal histories, quarterly support and unnormalized recursion masses."""
    calendar = np.arange(18.)
    observed = np.ones((18, 3), dtype=bool)
    observed[:4, 1] = False
    observed[:, 2] = calendar % 3 == 2
    weights = observed * np.exp(-.08*(17-calendar))[:, None] * np.array([.04, .03, .09])
    covariance = .6**np.abs(calendar[:, None]-calendar)
    return weights, observed, calendar, covariance


def test_joint_expectation_matches_independent_operators():
    """Cross terms must include both weighted centering and native counts."""
    q, mask, t, sigma = fixture()
    scale = np.array([12., 4., 2.])
    result = weighted_mean_hac_expectation(q, observed=mask, calendar=t,
                                           covariance_shape=sigma, scale=scale)
    kernel = np.maximum(0, 1-abs(t[:, None]-t)/6)
    operators = [np.diag(q[:, j])*scale[j]*np.sqrt(mask[:, j].sum()/(mask[:, j].sum()-1))
                 @ (np.eye(len(t))-np.outer(np.ones(len(t)), q[:, j]/q[:, j].sum()))
                 for j in range(3)]
    truth = (q*scale).T @ sigma @ (q*scale)
    expected = [[np.sum(kernel*(operators[i] @ sigma @ operators[j].T))
                 for j in range(3)] for i in range(3)]
    np.testing.assert_allclose(result['true_covariance'], truth, atol=1e-14)
    np.testing.assert_allclose(result['expected_hac_covariance'], expected, atol=1e-14)
    assert np.linalg.eigvalsh(result['expected_hac_covariance']).min() > -1e-13
    # Exact expectation does not depend on Gaussianity, only the supplied moments.
    root = np.linalg.cholesky(sigma)*np.sqrt(len(t))
    errors = np.vstack([root.T, -root.T])
    direct = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            direct[i, j] = np.mean(np.einsum('bt,ts,bs->b', errors @ operators[i].T,
                                            kernel, errors @ operators[j].T))
    np.testing.assert_allclose(result['expected_hac_covariance'], direct, atol=1e-14)


@pytest.mark.parametrize('defect', ['mask', 'negative', 'shape', 'scale', 'asymmetric'])
def test_invalid_inputs(defect):
    """Reject inconsistent support and invalid declared covariance shapes."""
    q, mask, t, sigma = fixture()
    scale = 1.
    if defect == 'mask':
        q[0, 1] = .1
    elif defect == 'negative':
        q[0, 0] = -.1
    elif defect == 'shape':
        sigma = sigma[:-1, :-1]
    elif defect == 'scale':
        scale = [1, 0, 1]
    else:
        sigma[0, 1] += .1
    with pytest.raises(ValueError):
        weighted_mean_hac_expectation(q, observed=mask, calendar=t,
                                      covariance_shape=sigma, scale=scale)
