"""State-denominator, invariance and leakage tests for the eQTL follow-up."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from papers.sign_pooling_2026.replication import eqtl_stability as stability
from papers.sign_pooling_2026.replication import eqtl_sensitivity as sensitivity
from papers.sign_pooling_2026.replication import eqtl_validation as validation
from papers.sign_pooling_2026.replication.tests.test_eqtl_validation import panel, baseline_fitter


class StabilityTests(unittest.TestCase):
    def test_absent_zero_and_signed_denominators_are_distinct(self):
        left = np.array([2, 2, 0, 1, 1, -1, 0, 1])
        right = np.array([2, 1, 0, 0, 1, 1, 2, 2])
        result = stability.compare_states(left, right)
        self.assertEqual(result["absent_both"], 1)
        self.assertEqual(result["absent_one"], 3)
        self.assertEqual(result["zero_both"], 1)
        self.assertEqual(result["zero_one_present"], 1)
        self.assertEqual(result["same_direction"], 1)
        self.assertEqual(result["opposite_direction"], 1)
        self.assertEqual(result["selected_union"], 5)
        self.assertEqual(result["selection_jaccard"], 2 / 5)
        self.assertEqual(result["selection_jaccard_common_markers"], 2 / 3)
        self.assertEqual(result["direction_agreement_selected_both"], 0.5)
        self.assertEqual(result["signed_overlap_over_selected_union"], 1 / 5)

    def test_alignment_and_numerical_tolerance(self):
        states = stability.coefficient_states(
            np.array([[1e-8, -2.0], [3.0, 0.0]]),
            ["m1", "m0"],
            ["g1", "g0"],
            ["m0", "m1", "m2"],
            ["g0", "g1"],
            1e-6,
        )
        np.testing.assert_array_equal(states, [[0, 1, 2], [-1, 0, 2]])

    def test_partition_labels_are_arbitrary_and_cell_counts_partition_all_folds(self):
        manifests, coefficients = [], {}
        for fold, labels in enumerate(([1, 1, 2], [8, 8, 4], [1, 2, 2])):
            markers = ["m0", "m1"] if fold != 1 else ["m0"]
            prep = dict(
                selected_markers=markers,
                ld_markers=["m0", "m1", "m2"],
                cluster_by_gene=dict(zip(["a", "b", "c"], labels)),
            )
            manifests.append(dict(repeat=0, fold=fold, outer_preprocessing=prep))
            coefficients[f"r0_f{fold}_lasso"] = dict(
                beta=np.ones((3, len(markers))),
                genes=np.array(["a", "b", "c"]),
                markers=np.array(markers),
            )
        result = stability.stability_tables(manifests, coefficients, raw_markers=["m0", "m1", "m2"])
        self.assertAlmostEqual(result["partition_pairs"].cluster_ari.iloc[0], 1)
        self.assertAlmostEqual(result["partition_pairs"].cluster_ari.iloc[1], -0.5)
        frequencies = result["coefficient_frequencies"]
        np.testing.assert_array_equal(
            frequencies[["n_absent", "n_zero", "n_positive", "n_negative"]].sum(axis=1), 3
        )
        self.assertEqual(result["partition_summary"]["never_selected_markers"], 1)
        self.assertTrue((frequencies.loc[frequencies.marker == "m1", "n_absent"] == 1).all())

    def test_remaining_ld_uses_only_fold_training_rows(self):
        x = pd.DataFrame({"a": [0, 1, 2, 1000], "b": [0, 1, 2, -1000]})
        manifest = [
            dict(
                repeat=0,
                fold=0,
                outer_preprocessing=dict(train_ids=[0, 1, 2], selected_markers=["a", "b"]),
            )
        ]
        result = stability.remaining_ld(manifest, x)
        self.assertAlmostEqual(result.maximum.iloc[0], 1)
        self.assertEqual(result.fraction_above_92.iloc[0], 1)

    def test_independent_prediction_aggregation_detects_wrong_denominator(self):
        predictions = pd.DataFrame(
            dict(
                repeat=[0, 0, 0],
                method=["lasso"] * 3,
                sample=[0, 1, 2],
                gene=["a"] * 3,
                actual=[1.0, 2.0, 5.0],
                predicted=[0.0, 2.0, 3.0],
                training_mean=[0.0, 0.0, 3.0],
            )
        )
        repeats = pd.DataFrame(
            dict(
                repeat=[0],
                method=["lasso"],
                r2_gene_median=[4 / 9],
                r2_gene_mean=[4 / 9],
                r2_pooled=[4 / 9],
            )
        )
        self.assertLess(
            stability.verify_predictions(predictions, repeats)["independent_r2_maximum_difference"],
            1e-14,
        )
        repeats["r2_pooled"] = 0.9
        with self.assertRaises(AssertionError):
            stability.verify_predictions(predictions, repeats)


class SensitivityTests(unittest.TestCase):
    def test_prior_protocol_must_match_before_reusing_any_fits(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            protocol = validation.Protocol()
            (path / "receipt.json").write_text(json.dumps({"protocol": asdict(protocol)}))
            self.assertEqual(len(sensitivity.make_declaration(path)["fit_tasks"]), 11)
            wrong = replace(protocol, gate_threshold=1)
            (path / "receipt.json").write_text(json.dumps({"protocol": asdict(wrong)}))
            with self.assertRaisesRegex(ValueError, "Prior protocol"):
                sensitivity.make_declaration(path)

    def test_paired_alignment_rejects_missing_inner_fold(self):
        reference = [
            dict(
                repeat=0,
                fold=0,
                train_ids=[0, 1],
                test_ids=[2],
                inner=[
                    dict(train_ids=[0], validation_ids=[1]),
                    dict(train_ids=[1], validation_ids=[0]),
                ],
            )
        ]
        candidate = json.loads(json.dumps(reference))
        sensitivity.verify_fold_alignment(reference, candidate)
        candidate[0]["inner"].pop()
        with self.assertRaisesRegex(AssertionError, "Inner fold counts"):
            sensitivity.verify_fold_alignment(reference, candidate)

    def _audit_screening(self, protocol, inject_leak=False):
        x, y, positions = panel(n=24)
        calls = []
        original_screen = validation.screen_markers
        original_preprocessing = validation.fit_preprocessing

        def record_screen(a, b, *args):
            calls.append(a.index.tolist())
            return original_screen(a, b, *args)

        def preprocessing(a, b, p, config):
            # Deliberately recreate the original global-screening defect in the mutation check.
            if inject_leak:
                a, b = x, y
            return original_preprocessing(a, b, p, config)

        with patch.object(validation, "screen_markers", side_effect=record_screen):
            with patch.object(validation, "fit_preprocessing", side_effect=preprocessing):
                validation.run_nested_cv(
                    x, y, positions, protocol, fitter=baseline_fitter, progress=False
                )
        expected = []
        for fold, (train, _) in enumerate(
            validation.splits(len(x), protocol.outer_folds, protocol.seed)
        ):
            for inner, _ in validation.splits(
                len(train), protocol.inner_folds, protocol.seed + 97 * (fold + 1)
            ):
                expected.append(x.index[train[inner]].tolist())
            expected.append(x.index[train].tolist())
        self.assertEqual(calls, expected)

    def test_each_neighbor_repeats_training_only_screening(self):
        base = validation.Protocol(
            repeats=1,
            outer_folds=2,
            inner_folds=2,
            methods=("hcgl_singleton_exclude", "fcgl_unconstrained"),
            lambda_fractions=(0.3,),
            random_partitions=2,
            screen_min_genes=1,
        )
        for name, parameter, value in sensitivity.NEIGHBORS:
            with self.subTest(configuration=name):
                self._audit_screening(replace(base, **{parameter: value}))

    def test_leakage_auditor_rejects_deliberate_global_preprocessing(self):
        protocol = validation.Protocol(
            repeats=1,
            outer_folds=2,
            inner_folds=2,
            methods=("lasso",),
            lambda_fractions=(0.3,),
            random_partitions=2,
            screen_min_genes=1,
        )
        with self.assertRaises(AssertionError):
            self._audit_screening(protocol, inject_leak=True)

    def test_new_methods_have_correct_restrictions_and_zero_thresholds(self):
        x, y, positions = panel(n=36)
        protocol = validation.Protocol(solver="CLARABEL", screen_min_genes=1)
        state = validation.fit_preprocessing(x.iloc[:24], y.iloc[:24], positions, protocol)
        self.assertIsNone(validation.constraint_signs(state, "fcgl_unconstrained"))
        np.testing.assert_array_equal(
            validation.constraint_signs(state, "hcgl_singleton_exclude"), state.singleton_signs
        )
        for method in sensitivity.ADDITIONS:
            result = validation.fit_path(state, x.iloc[24:], method, (1.01,), protocol)[0]
            np.testing.assert_allclose(result["beta"], 0, atol=2e-4)

    def test_reused_estimators_are_numerically_invariant(self):
        x, y, positions = panel(n=36)
        base = validation.Protocol(solver="CLARABEL", screen_min_genes=1)
        original_state = validation.fit_preprocessing(x.iloc[:24], y.iloc[:24], positions, base)
        for parameter, value in (("cluster_cut", 0.9), ("gate_threshold", 3.0), ("l1_weight", 0.8)):
            protocol = replace(base, **{parameter: value})
            state = validation.fit_preprocessing(x.iloc[:24], y.iloc[:24], positions, protocol)
            unchanged = set(sensitivity.PRIMARY) - set(sensitivity.affected_methods(parameter))
            for method in unchanged:
                before = validation.fit_path(original_state, x.iloc[24:], method, (0.3,), base)[0]
                after = validation.fit_path(state, x.iloc[24:], method, (0.3,), protocol)[0]
                np.testing.assert_allclose(before["predictions"], after["predictions"], atol=1e-7)
                self.assertAlmostEqual(before["lambda_max"], after["lambda_max"])


if __name__ == "__main__":
    unittest.main()
