"""Independent orthogonal-design check of the post-inspection L1 endpoint."""

import unittest
from unittest.mock import patch

import numpy as np

from papers.sign_pooling_2026.replication import learned_pooling_endpoint as E


class EndpointTests(unittest.TestCase):
    def test_endpoint_matches_independent_soft_threshold_and_projection(self):
        rng = np.random.default_rng(491)
        raw = rng.normal(size=(50, 4))
        x = np.linalg.qr(raw - raw.mean(axis=0))[0] * np.sqrt(50)
        beta = np.array([[1., -.6, .3, 0], [-.8, .2, 0, .5], [.1, -.7, .9, -.4]])
        y = x @ beta.T + rng.normal(size=(50, 3)) * .1
        prep = E.B.prepare(x, y, .5)
        covariance = prep.y.T @ prep.x / len(x)
        np.testing.assert_allclose(prep.x.T @ prep.x / len(x), np.eye(4), atol=1e-14)
        lam_max = 2 * np.abs(covariance).max()
        for fraction in [.1, .37]:
            ordinary = np.sign(covariance) * np.maximum(
                np.abs(covariance) - fraction * lam_max / 2, 0)
            for method in E.METHODS:
                signs = np.full(beta.shape, np.nan)
                expected = ordinary.copy()
                if method in E.B.SIGNED_METHODS:
                    signs[0, 0] = -1  # Deliberately conflicts with unconstrained sign.
                    signs[1, 0] = 0
                    signs[2, 1] = 1
                    expected[0, 0] = min(expected[0, 0], 0)
                    expected[1, 0] = 0
                    expected[2, 1] = max(expected[2, 1], 0)
                with patch.object(E.B, "method_inputs", return_value=(
                        prep.labels, signs, np.ones_like(signs))):
                    estimates, _, lambdas, _ = E.B.fit_path(
                        prep, method, 1.5, E.endpoint_protocol("CLARABEL"),
                        fractions=[fraction])
                np.testing.assert_allclose(lambdas[0], fraction * lam_max, atol=1e-14)
                np.testing.assert_allclose(estimates[0], expected, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
