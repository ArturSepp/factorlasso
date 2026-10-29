"""Public-ridge routing, immutable source upgrades and complete replay comparisons."""

import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from factorlasso import LassoModel, LassoModelType
from papers.sign_pooling_2026.replication import french_estimators as F
from papers.sign_pooling_2026.replication import french_full_run as S
from papers.sign_pooling_2026.replication import french_rolling_validation as R
from papers.sign_pooling_2026.replication import verify_french_ridge_migration as V
from papers.sign_pooling_2026.replication.tests.test_french_rolling_validation import (
    candidate, fixture, small_protocol,
)


class PublicRidgeRoutingTests(unittest.TestCase):
    def test_every_endpoint_uses_public_ridge_with_frozen_transformations(self):
        state = F.prepare(*fixture())
        spec = F.specification(state, candidate("ridge"))
        protocol = small_protocol()
        with patch.object(F, "LassoModel", wraps=LassoModel) as constructor:
            results = F.fit_path(state, spec, [np.inf, 1., .1, 0.], protocol)
        self.assertEqual(constructor.call_count, 4)
        for call in constructor.call_args_list:
            self.assertEqual(call.kwargs["model_type"], LassoModelType.RIDGE)
            self.assertFalse(call.kwargs["demean"])
            self.assertIsNone(call.kwargs["warmup_period"])
            self.assertIsNone(call.kwargs["span"])
            self.assertEqual(call.kwargs["loss_normalization"], "sample")
        for result in results:
            self.assertEqual(result["kind"], "package_ridge")
            np.testing.assert_array_equal(result["alpha"], np.zeros(state.y.shape[1]))
            lam = result["lambda"]
            if np.isinf(lam):
                np.testing.assert_array_equal(result["beta"], np.zeros_like(spec.signs))
            else:
                design = np.vstack([state.x, np.sqrt(len(state.x)*lam)*np.eye(state.x.shape[1])])
                response = np.vstack([state.y, np.zeros((state.x.shape[1], state.y.shape[1]))])
                np.testing.assert_allclose(result["beta"], np.linalg.lstsq(
                    design, response, rcond=None)[0].T, atol=1e-10, rtol=1e-9)

    def test_cost_counts_do_not_label_package_fits_as_direct_analytic(self):
        state = F.prepare(*fixture())
        evaluator = R.Evaluator(state, small_protocol())
        evaluator.evaluate([candidate("ridge")])
        counts = R.cost_counts([evaluator])
        self.assertEqual(counts["package_ridge_fits"], 1)
        self.assertEqual(counts["direct_or_analytic_fits"], 0)


class SourceUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.panel = Path(self.temp.name)
        self.old = {"src/factorlasso/old.py": "old-hash"}
        R.atomic_json(self.panel / "environment.json", {"source_sha256": self.old})
        self.binding = {"source_sha256": dict(self.old)}
        self.patch = patch.object(R, "execution_identity", side_effect=lambda *a: copy.deepcopy(
            self.binding))
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_unchanged_sources_preserve_old_binding_shape(self):
        binding = S.execution_binding(self.panel, small_protocol())
        self.assertNotIn("core_source_upgrade", binding)

    def test_changed_added_and_removed_sources_require_explicit_reason(self):
        variants = ({"src/factorlasso/old.py": "changed"}, {},
                    {**self.old, "src/factorlasso/ridge.py": "new"})
        for sources in variants:
            with self.subTest(sources=sources):
                self.binding["source_sha256"] = sources
                with self.assertRaisesRegex(ValueError, "Core sources differ"):
                    S.execution_binding(self.panel, small_protocol())

    def test_upgrade_records_both_hash_sets_and_reason_and_rejects_blank(self):
        self.binding["source_sha256"] = {**self.old, "src/factorlasso/ridge.py": "new"}
        for reason in ("", "  ", 1):
            with self.assertRaisesRegex(ValueError, "nonempty reason"):
                S.execution_binding(self.panel, small_protocol(), source_upgrade_reason=reason)
        a = S.execution_binding(self.panel, small_protocol(),
                                source_upgrade_reason="ridge migration")
        record = a["core_source_upgrade"]
        self.assertEqual(record["reason"], "ridge migration")
        self.assertEqual(record["stage1_core_sha256"], self.old)
        self.assertEqual(record["current_core_sha256"], self.binding["source_sha256"])
        self.assertEqual(record["changed_sources"]["src/factorlasso/ridge.py"],
                         {"stage1_sha256": None, "current_sha256": "new"})
        self.binding["source_sha256"]["src/factorlasso/ridge.py"] = "changed-again"
        b = S.execution_binding(self.panel, small_protocol(),
                                source_upgrade_reason="ridge migration")
        self.assertNotEqual(R.digest(a), R.digest(b))

    def test_source_upgrade_cannot_reuse_an_old_run_manifest(self):
        self.binding["source_sha256"] = {"src/factorlasso/old.py": "new"}
        output = self.panel / "old-run"
        R.atomic_json(output / "run_manifest.json", {"identity": "old", "binding": {}})
        with patch.object(S, "job_inventory", return_value=[]):
            with self.assertRaisesRegex(ValueError, "Stale or mixed"):
                S.initialize(self.panel, output, small_protocol(), fixture()[0],
                             source_upgrade_reason="ridge migration")


class ReplayComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old, self.new = self.root / "old", self.root / "new"
        self.protocol = small_protocol()
        self.protocol["methods"] = ["ridge"]
        self.job = {"origin": 16, "lookback": 12, "folder": "origin_16_T12"}
        x, y = fixture(n=24, p=2, q=3)
        with contextlib.redirect_stdout(io.StringIO()):
            for folder in (self.old, self.new):
                R.run_window(x, y, 16, 12, self.protocol,
                             folder=folder / self.job["folder"], identity="reference")

    def compare(self):
        return V.compare_window(self.old, self.new, self.job, "reference", self.protocol)

    def test_complete_comparison_covers_all_candidates_and_folds(self):
        result = self.compare()
        self.assertEqual(result["inner_candidate_fold_comparisons"], 12)
        self.assertEqual(result["failed_comparisons"], [])
        self.assertTrue(result["selected_candidate_unchanged"])

    def test_candidate_and_prediction_changes_are_reported(self):
        folder = self.new / self.job["folder"]
        selections = S.read_json(folder / "selections.json")
        selections["ridge"]["candidate"] = {"method": "ridge", "ridge_lambda": 12345}
        R.atomic_json(folder / "selections.json", selections)
        with np.load(folder / "outer_arrays.npz") as saved:
            arrays = {k: saved[k].copy() for k in saved.files}
        arrays["ridge__prediction"][0, 0] += .01
        R.atomic_npz(folder / "outer_arrays.npz", arrays)
        result = self.compare()
        self.assertIn("outer_prediction", result["failed_comparisons"])
        self.assertIn("selected_candidate", result["failed_comparisons"])

    def test_corrupt_reference_and_missing_candidate_are_rejected(self):
        folder = self.old / self.job["folder"]
        surface = folder / "ridge_inner_surface.json"
        original = surface.read_bytes()
        surface.write_bytes(original+b" ")
        with self.assertRaisesRegex(ValueError, "Corrupted reference artifact"):
            self.compare()
        surface.write_bytes(original)
        current_path = self.new / self.job["folder"] / "ridge_inner_surface.json"
        current = S.read_json(current_path)
        current["ledger"].pop()
        R.atomic_json(current_path, current)
        with self.assertRaisesRegex(ValueError, "incomplete or duplicated"):
            self.compare()


if __name__ == "__main__":
    unittest.main()
