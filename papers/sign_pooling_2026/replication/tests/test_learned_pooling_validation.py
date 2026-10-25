"""Independent checks of the learned-group design and nested selection logic."""

import unittest
from unittest.mock import patch

import numpy as np

from papers.sign_pooling_2026.replication import learned_pooling_validation as L


class LearnedPoolingTests(unittest.TestCase):
    def test_design_changes_only_its_declared_coefficient_component(self):
        coherent = L.simulate(L.Design("coherent"), 231)
        mixed = L.simulate(L.Design("mixed", mixed=True), 231)
        partial = L.simulate(L.Design("partial", partial=True), 231)
        beta, labels = coherent[-2:]
        np.testing.assert_array_equal(coherent[0], mixed[0])
        np.testing.assert_array_equal(coherent[0], partial[0])
        np.testing.assert_array_equal(np.abs(beta), np.abs(mixed[-2]))
        self.assertEqual(np.sum(beta != mixed[-2]), 6)
        self.assertEqual(np.sum((beta != 0) != (partial[-2] != 0)), 12)
        self.assertFalse(np.array_equal(coherent[0], coherent[2][:80]))
        metrics = L.group_metrics(labels, labels, mixed[-2])
        self.assertAlmostEqual(metrics["within_active_sign_conflict"], 2 / 9)
        self.assertEqual(metrics["adjusted_rand"], 1.0)
        self.assertEqual(L.group_metrics(labels, labels, beta)["within_active_sign_conflict"], 0)

    def test_group_metrics_against_explicit_pairs(self):
        labels = np.array([0, 0, 0, 1])
        truth_labels = np.array([0, 0, 1, 1])
        beta = np.array([[1, 0], [-1, 1], [1, 0], [0, 1]])
        result = L.group_metrics(labels, truth_labels, beta)
        # Three inferred within pairs, one correct; two true within pairs.
        self.assertAlmostEqual(result["grouping_precision"], 1 / 3)
        self.assertEqual(result["grouping_recall"], 0.5)
        self.assertAlmostEqual(result["within_active_sign_conflict"], 2 / 3)
        self.assertAlmostEqual(result["within_support_disagreement"], 1 / 3)

    def test_cv_rebuilds_every_candidate_from_training_rows(self):
        rng = np.random.default_rng(44)
        x, y = rng.normal(size=(24, 3)), rng.normal(size=(24, 4))
        calls = []
        original = L.B.prepare

        def record(xx, yy, cutoff):
            calls.append((xx.copy(), yy.copy(), cutoff))
            return original(xx, yy, cutoff)

        def zero_path(prepared, method, tau, protocol):
            count = len(protocol.lambda_fractions)
            return (np.zeros((count, 4, 3)), np.zeros((count, 4)), np.ones(count), np.zeros((4, 3)))

        with (
            patch.object(L.B, "prepare", side_effect=record),
            patch.object(L.B, "fit_path", side_effect=zero_path),
        ):
            candidates, fractions, errors = L.tune_surfaces(x, y, "hcgl_pooled_zero", 7)
        self.assertEqual(len(candidates), 27)
        self.assertEqual(len(calls), 9)
        expected = 0.0
        for fold, (train, valid) in enumerate(L.B.cv_splits(24, 3, 7)):
            for j, cut in enumerate(L.CUTS):
                np.testing.assert_array_equal(calls[3 * fold + j][0], x[train])
                np.testing.assert_array_equal(calls[3 * fold + j][1], y[train])
                self.assertEqual(calls[3 * fold + j][2], cut)
            standardized = (y[valid] - y[train].mean(axis=0)) / y[train].std(axis=0)
            expected += np.sum(standardized**2) / (24 * 4)
        np.testing.assert_allclose(errors, expected)
        self.assertEqual(len(fractions), 7)

    def test_joint_and_one_factor_selection_are_distinct(self):
        candidates = L.candidates_for("hcgl_pooled_zero")
        errors = np.full((len(candidates), 2), 100.0)
        reference = candidates.index(L.Candidate())
        joint = candidates.index(L.Candidate(0.7, 0.75, 3.0))
        errors[reference] = [5.0, 4.0]
        errors[joint] = [1.0, 2.0]
        variants = L.select_variants(candidates, (1.0, 0.1), errors)
        self.assertEqual(variants["reference"], (L.Candidate(), 0.1, 4.0))
        self.assertEqual(variants["joint_cv"], (L.Candidate(0.7, 0.75, 3.0), 1.0, 1.0))
        self.assertEqual(len(variants), 8)
        self.assertEqual(variants["cut_0.7"][0], L.Candidate(0.7, 0.5, 1.5))

    def test_gate_metrics_denominators_and_oracle_interpolation(self):
        beta = np.array([[1.0, -1.0, 0.0, 0.0]])
        stats = L.gate_metrics(np.array([[1.0, 1.0, 1.0, 0.0]]), beta)
        self.assertEqual(stats["active_correct"], 0.5)
        self.assertEqual(stats["active_wrong"], 0.5)
        self.assertEqual(stats["null_retention"], 0.5)
        self.assertAlmostEqual(stats["signed_fdp"], 1 / 3)
        curves = []
        for gate, multiplier in (("pooled", 1.0), ("singleton", 0.5)):
            for tau, null, correct in ((0.0, 1.0, 0.9), (1.0, 0.2, 0.7), (2.0, 0.0, 0.3)):
                curves.append(
                    dict(
                        gate=gate,
                        tau=tau,
                        null_retention=null,
                        active_correct=correct * multiplier,
                        active_wrong=0.0,
                        signed_fdp=null,
                    )
                )
        result = L.equal_null_diagnostics(curves, targets=(0.1, 1.1))
        self.assertEqual(len(result), 1)  # Never extrapolate.
        self.assertAlmostEqual(result[0]["active_correct_pooled"], 0.5)
        self.assertAlmostEqual(result[0]["active_correct_difference"], 0.25)


if __name__ == "__main__":
    unittest.main()
