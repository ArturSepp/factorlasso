"""Independent calendar and influence checks for shared inference primitives."""
import numpy as np
import pytest

from factorlasso.inference._sandwich import lag_covariance
from factorlasso.utils._hac import bartlett_kernel, score_covariance


@pytest.mark.parametrize('lags', [0, 1, 5, 25])
def test_lag_aggregation_matches_explicit_calendar_pairs(lags):
    """Both the full and diagonal paths equal a separately summed pair formula."""
    scores = np.random.default_rng(401).normal(size=(11, 3))
    scores[[0, 4, 7]] = 0
    expected = np.zeros((3, 3))
    for t in range(11):
        for s in range(11):
            expected += max(0, 1-abs(t-s)/(lags+1))*np.outer(scores[t], scores[s])
    np.testing.assert_allclose(lag_covariance(scores, lags), expected, atol=2e-14)
    np.testing.assert_allclose(lag_covariance(scores, lags, diagonal=True),
                               np.diag(expected), atol=2e-14)
    np.testing.assert_allclose(score_covariance(scores, bartlett_kernel(np.arange(11), lags+1)),
                               expected, atol=2e-14)


def test_calendar_gaps_are_not_compressed():
    """Zero scores retain original lag distances and explicit irregular spacing."""
    scores = np.array([[1.], [0.], [2.]])
    assert lag_covariance(scores, 1)[0, 0] == 5.
    assert lag_covariance(scores[[0, 2]], 1)[0, 0] == 7.
    kernel = bartlett_kernel([0, 2, 7], 3)
    np.testing.assert_allclose(kernel, [[1, 1/3, 0], [1/3, 1, 0], [0, 0, 1]])
