"""Independent numerical checks for the revision's estimator comparison."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from papers.sign_pooling_2026.replication import estimator_benchmark as B
from papers.sign_pooling_2026.replication.sign_pooling_robustness import _fitted_sign_metrics


class BenchmarkTests(unittest.TestCase):
    def test_common_zero_tolerance_rejects_solver_dust(self):
        """Fails under the former exact-zero comparator scoring mutation."""
        fitted = np.array([[1e-9, -1e-9, 1., -.3, .4]])
        truth = np.array([[1., -1., -1., -1., 0.]])
        recovery, flip, abstain, _ = _fitted_sign_metrics(fitted, truth, np.sign(truth))
        self.assertEqual((recovery, flip, abstain), (.25, .25, .5))
        metrics = B.coefficient_metrics(fitted, truth)
        self.assertEqual(metrics["support_precision"], 2/3)
        self.assertEqual(metrics["support_fdp"], 1/3)
        self.assertEqual(metrics["support_recall"], .5)
        self.assertEqual(metrics["active_sign_recovery"], recovery)
        self.assertEqual(metrics["active_sign_flip"], flip)
        self.assertEqual(metrics["active_abstention"], abstain)
        exact = B.coefficient_metrics(fitted, truth, 0.)
        self.assertNotEqual(exact["active_sign_recovery"], recovery)

    def test_unilasso_loo_features_against_explicit_refits(self):
        """The through-origin and no-intercept shortcuts do not pass this check."""
        rng = np.random.default_rng(44)
        x = rng.normal(size=(15, 3)) + [2., -3., .8]
        y = 4 + x@np.array([.8, -.4, .2]) + rng.normal(size=15)
        features, slopes, intercepts = B.unilasso_features(x, y)
        for j in range(x.shape[1]):
            design = np.column_stack([np.ones(len(x)), x[:, j]])
            full = np.linalg.lstsq(design, y, rcond=None)[0]
            np.testing.assert_allclose([intercepts[j], slopes[j]], full, atol=1e-12)
            for i in range(len(x)):
                keep = np.arange(len(x)) != i
                coefficients = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
                self.assertAlmostEqual(features[i, j], design[i]@coefficients, places=11)

    def test_hcgl_path_matches_orthogonal_closed_form(self):
        """Independent sparse-group proximal solution checks normalization and geometry."""
        rng = np.random.default_rng(11)
        x = rng.normal(size=(30, 3))
        x -= x.mean(axis=0)
        x = np.linalg.qr(x)[0]*np.sqrt(len(x))
        ols = np.array([[.6, -.2, .03], [.1, -.5, .4]])
        y = x@ols.T
        prep = B.Prepared(x, y, np.zeros(3), np.ones(3), np.zeros(2), np.ones(2), np.zeros(2))
        protocol = replace(B.Protocol(), solver="CLARABEL", lambda_fractions=(.1,))
        fitted, _, lambdas, _ = B.fit_path(prep, "hcgl_unconstrained", np.nan, protocol)
        lam, alpha, weight = lambdas[0], protocol.alpha, np.sqrt(2)
        soft = np.sign(ols)*np.maximum(np.abs(ols)-lam*alpha/2, 0)
        shrink = np.maximum(1-lam*(1-alpha)*weight/(2*np.linalg.norm(soft, axis=1)), 0)
        expected = soft*shrink[:, None]
        np.testing.assert_allclose(fitted[0], expected, atol=2e-5)

    def test_unilasso_zero_endpoint_is_nnls_with_intercept(self):
        """Independent bounded least squares checks the exact no-penalty endpoint."""
        from scipy.optimize import lsq_linear
        rng = np.random.default_rng(148)
        x = rng.normal(size=(40, 4))
        y = (2 + x@np.array([.7, -.4, .1, 0]) + rng.normal(size=40))[:, None]
        prep = B.prepare(x, y)
        beta, intercept, lambdas, _ = B.fit_path(
            prep, "unilasso", np.nan, B.Protocol(), fractions=[0.])
        eta, slopes, constants = B.unilasso_features(prep.x, prep.y[:, 0])
        theta = lsq_linear(eta-eta.mean(axis=0), prep.y[:, 0], bounds=(0, np.inf),
                           tol=1e-12).x
        expected_intercept = -eta.mean(axis=0)@theta + constants@theta
        np.testing.assert_allclose(beta[0, 0], slopes*theta, atol=1e-7)
        np.testing.assert_allclose(intercept[0, 0], expected_intercept, atol=1e-7)
        self.assertEqual(lambdas[0], 0.)

    def test_failed_gate_zero_versus_actual_abstention(self):
        """Mutating failed-gate NaNs back to zero makes the released fit fail."""
        rng = np.random.default_rng(30)
        x = rng.normal(size=(30, 3))
        y = x@np.array([[1., -.8, .5], [.8, -.5, .3]]).T + rng.normal(scale=.1, size=(30, 2))
        prep = B.prepare(x, y)
        protocol = replace(B.Protocol(), solver="CLARABEL", lambda_fractions=(.01,))
        gate = (np.zeros((2, 3)), np.zeros((2, 3)))
        with patch.object(B, "derive_gate", return_value=gate):
            zero = B.fit_path(prep, "hcgl_pooled_zero", .75, protocol)[0][0]
            free = B.fit_path(prep, "hcgl_pooled_free", .75, protocol)[0][0]
        self.assertLess(np.max(np.abs(zero)), 1e-7)
        self.assertGreater(np.linalg.norm(free), .5)

    def test_cv_preprocessing_uses_training_rows(self):
        """Recording inputs detects replacing fold preparation with full-sample preparation."""
        rng = np.random.default_rng(77)
        x, y = rng.normal(size=(24, 3)), rng.normal(size=(24, 2))
        protocol = replace(B.Protocol(), lambda_fractions=(.3, .03), folds=3)
        calls = []
        original = B.prepare

        def record(xx, yy, cutoff):
            calls.append((xx.copy(), yy.copy()))
            return original(xx, yy, cutoff)

        with patch.object(B, "prepare", side_effect=record):
            B.tune_and_fit(x, y, "lasso", protocol, seed=9)
        self.assertEqual(len(calls), 4)
        for observed, (train, _) in zip(calls[:-1], B.cv_splits(len(x), 3, 9)):
            np.testing.assert_array_equal(observed[0], x[train])
            np.testing.assert_array_equal(observed[1], y[train])
        np.testing.assert_array_equal(calls[-1][0], x)

    def test_infinite_gate_rejects_noiseless_signs_and_preserves_slopes(self):
        """An all-failed endpoint must override infinite exact-signal statistics."""
        x = np.array([-1., 1., -1., 1.])[:, None]
        y = np.column_stack((x[:, 0], 2*x[:, 0], -x[:, 0]))
        labels = np.array([0, 0, 1])
        expected_slopes = np.array([[1.5], [1.5], [-1.]])
        finite_signs, finite_slopes = B.derive_gate(x, y, labels, 1e6)
        np.testing.assert_array_equal(finite_signs, np.sign(expected_slopes))
        np.testing.assert_array_equal(finite_slopes, expected_slopes)
        signs, slopes = B.derive_gate(x, y, labels, np.inf)
        np.testing.assert_array_equal(signs, np.zeros_like(expected_slopes))
        np.testing.assert_array_equal(slopes, expected_slopes)
        prep = B.Prepared(x, y, np.zeros(1), np.ones(1), np.zeros(3), np.ones(3), labels)
        _, excluded, _ = B.method_inputs(prep, "hcgl_pooled_zero", np.inf, B.Protocol())
        _, released, _ = B.method_inputs(prep, "hcgl_pooled_free", np.inf, B.Protocol())
        np.testing.assert_array_equal(excluded, np.zeros_like(expected_slopes))
        self.assertTrue(np.isnan(released).all())

    def test_date_gate_against_independent_score_formula(self):
        rng = np.random.default_rng(52)
        x = rng.normal(size=(40, 2))
        x -= x.mean(axis=0)
        y = rng.normal(size=(40, 3))
        y -= y.mean(axis=0)
        tau = .8
        signs, slopes = B.derive_gate(x, y, np.zeros(3), tau)
        expected = []
        for j in range(2):
            denominator = 3*np.sum(x[:, j]**2)
            slope = np.sum(x[:, j,None]*y)/denominator
            scores = x[:, j]*np.sum(y-slope*x[:, j,None], axis=1)
            variance = np.sum(scores**2)/denominator**2 * len(x)/(len(x)-1)
            expected.append(np.sign(slope) if abs(slope)/np.sqrt(variance) >= tau else 0.)
            np.testing.assert_allclose(slopes[:, j], slope)
        np.testing.assert_array_equal(signs, np.tile(expected, (3, 1)))

    def test_ward_chord_and_cut_endpoints(self):
        rng = np.random.default_rng(3)
        y = rng.normal(size=(50, 6))
        np.testing.assert_array_equal(B.ward_labels(y, 0), np.arange(6))
        np.testing.assert_array_equal(B.ward_labels(y, 1), np.zeros(6))
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        chord = np.sqrt(np.maximum(2*(1-np.corrcoef(y, rowvar=False)), 0))
        np.fill_diagonal(chord, 0)
        tree = linkage(squareform(chord, checks=False), method="ward")
        expected = fcluster(tree, .5*tree[-1, 2], criterion="distance")
        actual = B.ward_labels(y, .5)
        np.testing.assert_array_equal(actual[:,None] == actual, expected[:,None] == expected)


if __name__ == "__main__":
    unittest.main()
