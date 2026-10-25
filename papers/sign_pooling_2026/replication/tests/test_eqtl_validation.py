"""Numerical and leakage regression checks for the new eQTL validation protocol."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from papers.sign_pooling_2026.replication import eqtl_validation as validation
from papers.sign_pooling_2026.replication import eqtl_pipeline as archived


def panel(seed=31, n=36):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"m{i}" for i in range(5)])
    y = pd.DataFrame(
        {
            "a": 0.8 * x.m0 + rng.normal(size=n) * 0.2,
            "b": 0.6 * x.m0 + rng.normal(size=n) * 0.2,
            "c": -0.7 * x.m1 + rng.normal(size=n) * 0.2,
        }
    )
    positions = pd.DataFrame({"chr": [1, 1, 2, 2, 3], "pos": np.arange(5)}, index=x.columns)
    return x, y, positions


def baseline_fitter(state, x_valid, method, fractions, protocol):
    """Fast test double: the leakage test exercises every real preprocessing stage."""
    return [
        dict(
            fraction=f,
            reg_lambda=f,
            lambda_max=1.0,
            seconds=0.0,
            predictions=np.broadcast_to(
                state.y_mean.to_numpy(), (len(x_valid), state.y.shape[1])
            ).copy(),
            beta=np.zeros((state.y.shape[1], state.x.shape[1])),
        )
        for f in fractions
    ]


class NestedValidationTests(unittest.TestCase):
    def test_source_export_has_explicit_non_git_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(validation, 'REPOSITORY_DIR', Path(directory)):
                self.assertEqual(validation._git('rev-parse', 'HEAD'),
                                 'unavailable (source export; see source hashes)')

    def test_legacy_prediction_path_warns_and_names_constrained_comparator(self):
        x, y, _ = panel(n=12)

        class ZeroModel:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def fit(self, x, y):
                self.estimated_betas = np.zeros((y.shape[1], x.shape[1]))
                return self

        with patch.object(archived, "LassoModel", ZeroModel):
            with self.assertWarnsRegex(RuntimeWarning, "globally outcome-screened"):
                result = archived.prediction_parity(x, y)
        self.assertEqual(set(result.method), {"HCGL", "FCGL", "gated_singleton_lasso"})

    def test_every_screen_uses_exact_training_rows(self):
        """Detect global outcome screening and reuse of outer screening inside inner CV."""
        x, y, positions = panel()
        protocol = validation.Protocol(
            repeats=2,
            outer_folds=3,
            inner_folds=3,
            methods=("lasso", "hcgl_exclude"),
            lambda_fractions=(1.0, 0.1),
            random_partitions=3,
        )
        calls = []
        real_screen = validation.screen_markers

        def audit_screen(x_train, y_train, *args):
            self.assertTrue(x_train.index.equals(y_train.index))
            calls.append(x_train.index.tolist())
            return real_screen(x_train, y_train, *args)

        with patch.object(validation, "screen_markers", side_effect=audit_screen):
            result = validation.run_nested_cv(
                x, y, positions, protocol, fitter=baseline_fitter, progress=False
            )
        expected = []
        for outer in result["manifest"]:
            test_ids = set(outer["test_ids"])
            self.assertFalse(test_ids & set(outer["train_ids"]))
            for inner in outer["inner"]:
                expected.append(inner["train_ids"])
                self.assertFalse(test_ids & set(inner["train_ids"]))
                self.assertFalse(set(inner["train_ids"]) & set(inner["validation_ids"]))
                self.assertEqual(
                    set(inner["train_ids"]) | set(inner["validation_ids"]), set(outer["train_ids"])
                )
            expected.append(outer["outer_preprocessing"]["train_ids"])
        self.assertEqual(calls, expected)
        counts = result["predictions"].groupby(["repeat", "method", "sample", "gene"]).size()
        self.assertTrue((counts == 1).all())
        self.assertEqual(len(result["predictions"]), len(x) * y.shape[1] * 2 * 2)
        np.testing.assert_allclose(result["repeats"].r2_pooled, 0, atol=1e-15)

    def test_training_scaling_and_selection_ignore_heldout_perturbation(self):
        x, y, positions = panel()
        protocol = validation.Protocol(screen_min_genes=1)
        train = np.arange(24)
        reference = validation.fit_preprocessing(x.iloc[train], y.iloc[train], positions, protocol)
        changed_x, changed_y = x.copy(), y.copy()
        changed_x.iloc[24:] *= 1000
        changed_y.iloc[24:] = -9000
        changed = validation.fit_preprocessing(
            changed_x.iloc[train], changed_y.iloc[train], positions, protocol
        )
        self.assertEqual(reference.manifest(), changed.manifest())
        np.testing.assert_allclose(reference.x.mean(), 0, atol=1e-14)
        np.testing.assert_allclose(reference.x.std(ddof=0), 1, atol=1e-14)
        np.testing.assert_array_equal(reference.pooled_signs, changed.pooled_signs)

    def test_pooled_r2_uses_fold_training_mean_and_sums_errors_before_dividing(self):
        actual = np.array([[1.0, 3.0], [2.0, 4.0], [5.0, 7.0]])
        predicted = np.array([[0.0, 3.0], [2.0, 2.0], [3.0, 6.0]])
        baseline = np.array([[0.0, 1.0], [0.0, 1.0], [3.0, 5.0]])
        result = validation.prediction_metrics(actual, predicted, baseline)
        expected = np.array([1 - 5 / 9, 1 - 5 / 17])
        np.testing.assert_allclose(result["r2_by_gene"], expected)
        self.assertAlmostEqual(result["r2_pooled"], 1 - 10 / 26)
        self.assertAlmostEqual(result["rmse"], np.sqrt(10 / 6))

    def test_exclusion_and_unrestricted_gate_are_distinct_fitted_estimators(self):
        x, y, positions = panel(n=42)
        protocol = validation.Protocol(
            solver="CLARABEL", screen_min_genes=1, gate_threshold=1e6, lambda_fractions=(0.03,)
        )
        state = validation.fit_preprocessing(x.iloc[:30], y.iloc[:30], positions, protocol)
        self.assertTrue((state.pooled_signs.to_numpy() == 0).all())
        outputs = {
            method: validation.fit_path(state, x.iloc[30:], method, (0.03,), protocol)[0]
            for method in ("hcgl_exclude", "hcgl_unrestricted", "hcgl_unconstrained")
        }
        np.testing.assert_allclose(outputs["hcgl_exclude"]["beta"], 0, atol=1e-10)
        self.assertGreater(np.linalg.norm(outputs["hcgl_unrestricted"]["beta"]), 0.2)
        np.testing.assert_allclose(
            outputs["hcgl_unrestricted"]["beta"], outputs["hcgl_unconstrained"]["beta"], atol=1e-7
        )

    def test_lambda_max_matches_zero_solution_for_every_penalty_geometry(self):
        x, y, positions = panel(n=42)
        protocol = validation.Protocol(solver="CLARABEL", screen_min_genes=1)
        state = validation.fit_preprocessing(x.iloc[:30], y.iloc[:30], positions, protocol)
        for method in validation.METHODS:
            with self.subTest(method=method):
                # Slightly above the KKT boundary avoids asking a numerical conic
                # solver to identify a zero coefficient exactly at a kink.
                result = validation.fit_path(state, x.iloc[30:], method, (1.01,), protocol)[0]
                np.testing.assert_allclose(result["beta"], 0, atol=2e-4)

    def test_size_matched_random_partitions_do_not_make_coherence_validation_circular(self):
        signs = np.array([[1, -1], [1, -1], [-1, 1], [-1, 1]])
        coherent, count = validation._coherence(signs, np.array([1, 1, 2, 2]))
        mixed, _ = validation._coherence(signs, np.array([1, 2, 1, 2]))
        self.assertEqual(count, 4)
        self.assertEqual(coherent, 1)
        self.assertEqual(mixed, 0.5)

    def test_invalid_protocol_and_empty_screen(self):
        with self.assertRaises(ValueError):
            validation.Protocol(lambda_fractions=(0.0,))
        x, y, positions = panel()
        protocol = validation.Protocol(screen_min_genes=20, random_partitions=3)
        state = validation.fit_preprocessing(x.iloc[:24], y.iloc[:24], positions, protocol)
        self.assertEqual(state.x.shape[1], 0)
        result = validation.fit_path(state, x.iloc[24:], "lasso", (0.1,), protocol)[0]
        np.testing.assert_allclose(
            result["predictions"], np.broadcast_to(y.iloc[:24].mean().to_numpy(), (12, 3))
        )


if __name__ == "__main__":
    unittest.main()
