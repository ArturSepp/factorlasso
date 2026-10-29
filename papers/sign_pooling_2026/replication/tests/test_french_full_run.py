"""Strict full-study coverage, independent loss checks and recovery tests."""

from concurrent.futures import ThreadPoolExecutor
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from papers.sign_pooling_2026.replication import french_full_run as S
from papers.sign_pooling_2026.replication import french_finalize as P
from papers.sign_pooling_2026.replication import french_rolling_validation as R
from papers.sign_pooling_2026.replication.tests.test_french_rolling_validation import (
    fixture, small_protocol)


class NumericalProjectionTests(unittest.TestCase):
    def test_polishing_compares_objective_to_a_feasible_reference(self):
        dates = pd.date_range("2000-01-05", periods=12, freq="W-WED")
        x = pd.DataFrame({"x": np.arange(12, dtype=float)}, index=dates)
        state = R.F.prepare(x, -x.rename(columns={"x": "y"}))
        spec = R.F.Spec("lasso", np.array([0]), np.array([1.]), np.array([[1.]]), 0.)
        # The true bounded optimum is beta=0. A tiny infeasible negative beta
        # has a lower raw objective, so raw-point monotonicity is inappropriate.
        initial = np.array([[-1e-8]])
        fitted, details = R.F.polish(state, spec, initial, .1)
        self.assertTrue(details["accepted"])
        np.testing.assert_array_equal(fitted, np.zeros((1, 1)))
        self.assertGreater(details["objective_change"], 0.)
        self.assertLessEqual(details["objective_change_from_feasible_start"], 1e-12)
        self.assertEqual(R.F.audit(state, spec, fitted, .1)["kkt_residual"], 0.)


class FullRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.panel, cls.baseline = cls.root / "panel", cls.root / "baseline"
        (cls.panel / "panel").mkdir(parents=True)
        cls.baseline.mkdir()
        cls.x, cls.y = fixture(n=25, p=2, q=3)
        cls.protocol = p = small_protocol()
        p["calendar"].update(lookbacks=[12, 16], outer_burn_in=16,
                             origins=3, test_weeks=9)
        p["calendar"]["inner_train_ends"]["16"] = [10, 12, 14]
        p["calendar"]["inner_validation_ends"]["16"] = [12, 14, 16]
        cls.binding = {"source_sha256": {}, "defaults_observed": p["package_defaults"],
                       "scheduler_resources": {"workers": 2, "blas_threads_per_worker": 1}}
        cls.identity = R.digest(cls.binding)
        cls.x.to_csv(cls.panel / "panel/X_weekly.csv")
        cls.y.to_csv(cls.panel / "panel/Y_weekly.csv")
        # Freeze exact round-trip inputs, as in the production intake.
        cls.x = R.read_csv(cls.panel / "panel/X_weekly.csv")
        cls.y = R.read_csv(cls.panel / "panel/Y_weekly.csv")
        R.atomic_json(cls.panel / "protocol.json", p)
        cls.jobs = S.job_inventory(p, len(cls.x))
        R.atomic_json(cls.baseline / "run_manifest.json", {"schema": 1,
            "identity": cls.identity, "binding": cls.binding, "jobs": cls.jobs})
        with contextlib.redirect_stdout(io.StringIO()):
            for job in cls.jobs:
                R.run_window(cls.x, cls.y, job["origin"], job["lookback"], p,
                             folder=cls.baseline / job["folder"], identity=cls.identity)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.output = self.root / self._testMethodName
        shutil.copytree(self.baseline, self.output)
        self.patches = [patch.object(S, "execution_binding", return_value=self.binding),
                        patch.object(R, "verify_panel", return_value={"status": "passed"})]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def finalize(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return S.finalize(self.panel, self.output)

    def rewrite(self, name, change):
        folder = self.output / self.jobs[0]["folder"]
        path = folder / name
        value = S.read_json(path)
        change(value)
        R.atomic_json(path, value)
        receipt_path = folder / "window_receipt.json"
        receipt = S.read_json(receipt_path)
        receipt["files_sha256"][name] = R.sha256(path)
        R.atomic_json(receipt_path, receipt)

    def test_inventory_has_all_origins_lookbacks_and_terminal_week(self):
        self.assertEqual([(j["origin"], j["lookback"]) for j in self.jobs],
                         [(16, 12), (16, 16), (20, 12), (20, 16), (24, 12), (24, 16)])
        p = copy.deepcopy(self.protocol)
        p["calendar"]["origins"] = 2
        with self.assertRaisesRegex(ValueError, "origin"):
            S.job_inventory(p, 25)
        p = copy.deepcopy(self.protocol)
        p["calendar"]["lookbacks"] = [12, 12]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            S.job_inventory(p, 25)

    def test_worker_override_is_execution_only_and_validated(self):
        original = copy.deepcopy(self.protocol)
        self.assertEqual(S.execution_resources(self.protocol)["workers"], 2)
        self.assertEqual(S.execution_resources(self.protocol, 1)["workers"], 1)
        self.assertEqual(self.protocol, original)
        with self.assertRaisesRegex(ValueError, "Workers"):
            S.execution_resources(self.protocol, 0)

    def test_complete_run_independently_reconstructs_losses_and_selections(self):
        result = self.finalize()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["windows"], 6)
        self.assertEqual(result["test_weeks_per_lookback"], 9)
        audit = S.read_json(self.output / "full_validation.json")
        self.assertEqual([a["test_weeks"] for a in audit["audits"]], [4, 4, 4, 4, 1, 1])
        self.assertLess(max(a["max_inner_sse_reconstruction_error"]
                            for a in audit["audits"]), 1e-12)

    def test_missing_window_is_rejected(self):
        (self.output / self.jobs[-1]["folder"] / "window_receipt.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing/duplicate"):
            self.finalize()
        self.assertNotEqual(S.read_json(self.output / "full_validation.json")["status"], "complete")

    def test_duplicate_and_unknown_windows_are_rejected(self):
        extra = self.output / "copy"
        extra.mkdir()
        shutil.copyfile(self.output / self.jobs[0]["folder"] / "window_receipt.json",
                        extra / "window_receipt.json")
        with self.assertRaisesRegex(ValueError, "Missing/duplicate"):
            self.finalize()

    def test_duplicate_candidate_fold_is_rejected_even_with_updated_hash(self):
        self.rewrite("inner_ledger.json", lambda rows: rows.append(rows[0]))
        with self.assertRaises(ValueError):
            self.finalize()

    def test_missing_candidate_is_rejected_even_with_updated_hash(self):
        self.rewrite("ols_inner_surface.json", lambda surface: surface["rows"].clear())
        with self.assertRaises(ValueError):
            self.finalize()

    def test_wrong_inner_loss_is_rejected_even_with_updated_hash(self):
        self.rewrite("inner_ledger.json", lambda rows: rows[0].update(sse=rows[0]["sse"]+1))
        with self.assertRaisesRegex(ValueError, "loss reconstruction"):
            self.finalize()

    def test_wrong_selection_and_mixed_window_identity_are_rejected(self):
        self.rewrite("selections.json", lambda s: s["ols"].update(score=s["ols"]["score"]+1))
        with self.assertRaisesRegex(ValueError, "selection mismatch"):
            self.finalize()
        shutil.rmtree(self.output)
        shutil.copytree(self.baseline, self.output)
        path = self.output / self.jobs[0]["folder"] / "window_receipt.json"
        receipt = S.read_json(path)
        receipt["identity"] = "other-source"
        R.atomic_json(path, receipt)
        with self.assertRaisesRegex(ValueError, "Mixed window"):
            self.finalize()

    def test_corruption_and_unlisted_artifact_are_rejected(self):
        folder = self.output / self.jobs[0]["folder"]
        path = next((folder / "inner_0_fits").glob("*.npz"))
        original = path.read_bytes()
        path.write_bytes(original+b"corruption")
        with self.assertRaisesRegex(ValueError, "Corrupted window"):
            self.finalize()
        path.write_bytes(original)
        (folder / "unexpected.txt").write_text("unexpected")
        with self.assertRaisesRegex(ValueError, "unexpected window artifacts"):
            self.finalize()

    def test_prediction_corruption_is_rejected_even_with_updated_hash(self):
        folder = self.output / self.jobs[0]["folder"]
        path = folder / "outer_arrays.npz"
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        arrays["ols__prediction"][0, 0] += .01
        R.atomic_npz(path, arrays)
        receipt_path = folder / "window_receipt.json"
        receipt = S.read_json(receipt_path)
        receipt["files_sha256"][path.name] = R.sha256(path)
        R.atomic_json(receipt_path, receipt)
        with self.assertRaises(AssertionError):
            self.finalize()

    def test_unresolved_fit_status_is_rejected_even_with_updated_hash(self):
        folder = self.output / self.jobs[0]["folder"]
        row = S.read_json(folder / "inner_ledger.json")[0]
        path = folder / f"inner_{row['inner_fold']}_fits/{row['fit_key']}.npz"
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        metadata = json.loads(arrays["metadata"].tobytes())
        metadata["result"]["status"] = "unresolved"
        arrays["metadata"] = np.frombuffer(R.json_bytes(metadata), dtype=np.uint8)
        R.atomic_npz(path, arrays)
        receipt_path = folder / "window_receipt.json"
        receipt = S.read_json(receipt_path)
        receipt["files_sha256"][path.relative_to(folder).as_posix()] = R.sha256(path)
        R.atomic_json(receipt_path, receipt)
        with self.assertRaisesRegex(ValueError, "Unresolved fit status"):
            self.finalize()

    def test_mixed_run_inventory_and_source_are_rejected(self):
        path = self.output / "run_manifest.json"
        original = path.read_bytes()
        value = S.read_json(path)
        value["jobs"].append(value["jobs"][0])
        R.atomic_json(path, value)
        with self.assertRaisesRegex(ValueError, "job inventory"):
            self.finalize()
        path.write_bytes(original)
        value = S.read_json(path)
        value["binding"]["extra"] = "changed environment"
        R.atomic_json(path, value)
        with self.assertRaisesRegex(ValueError, "execution identity"):
            self.finalize()

    def test_scheduler_resume_does_not_refit_and_preserves_all_window_bytes(self):
        before = {p.relative_to(self.output).as_posix(): p.read_bytes()
                  for p in self.output.glob("origin_*/**/*") if p.is_file()}
        def resume_worker(args):
            panel, output, job, identity = args
            R.pilot_worker((panel, output, job["lookback"], job["origin"], identity))
            return {"job": job}

        with patch.object(S, "ProcessPoolExecutor", ThreadPoolExecutor), \
                patch.object(S, "worker", side_effect=resume_worker), \
                patch.object(R.F, "fit_path", side_effect=AssertionError("Unexpected refit")), \
                contextlib.redirect_stdout(io.StringIO()):
            result = S.run_full(self.panel, self.output)
        self.assertEqual(result["status"], "complete")
        after = {p.relative_to(self.output).as_posix(): p.read_bytes()
                 for p in self.output.glob("origin_*/**/*") if p.is_file()}
        self.assertEqual(before, after)
        progress = S.read_json(self.output / "progress.json")
        self.assertEqual(len(progress["completed"]), 6)
        self.assertEqual(progress["failures"], [])

    def test_exclusive_lock_and_strict_flag(self):
        with S.exclusive_run(self.output):
            with self.assertRaisesRegex(ValueError, "owns this run"):
                with S.exclusive_run(self.output):
                    self.fail("Concurrent lock acquired")
        with self.assertRaisesRegex(ValueError, "require-complete"):
            S.finalize(self.panel, self.output, require_complete=False)

    def test_parallel_finalizer_is_identical_to_serial_without_fitting(self):
        self.finalize()
        before = (self.output / "full_validation.json").read_bytes()
        with patch.object(R.F, "fit_path", side_effect=AssertionError("Unexpected refit")), \
                contextlib.redirect_stdout(io.StringIO()):
            result = P.finalize(self.panel, self.output, workers=2)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(before, (self.output / "full_validation.json").read_bytes())

    def test_parallel_finalizer_rejects_a_missing_window(self):
        (self.output / self.jobs[-1]["folder"] / "window_receipt.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing/duplicate"):
            P.finalize(self.panel, self.output, workers=2)

    def test_parallel_finalizer_seals_the_completed_estimation_receipt(self):
        R.atomic_json(self.output / "progress.json", {"identity": self.identity,
            "completed": [{"job": j} for j in self.jobs], "pending": [], "failures": [],
            "wall_seconds": 1.25})
        with contextlib.redirect_stdout(io.StringIO()):
            P.finalize(self.panel, self.output, workers=2)
        receipt = S.read_json(self.output / "full_run_receipt.json")
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(receipt["jobs"], 6)
        self.assertEqual(receipt["validation_sha256"],
                         R.sha256(self.output / "full_validation.json"))
        self.assertGreater(receipt["combined_fit_and_validation_seconds"], 1.25)

    def test_parallel_finalizer_propagates_a_window_audit_failure(self):
        self.rewrite("inner_ledger.json", lambda rows: rows[0].update(sse=rows[0]["sse"]+1))
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "loss reconstruction"):
                P.finalize(self.panel, self.output, workers=2)
        self.assertNotEqual(S.read_json(self.output / "full_validation.json")["status"], "complete")

    def test_failed_window_blocks_completion_and_is_not_discarded(self):
        def worker(args):
            if args[2] == self.jobs[0]:
                raise RuntimeError("deliberate unresolved candidate")
            return {"job": args[2]}

        with patch.object(S, "ProcessPoolExecutor", ThreadPoolExecutor), \
                patch.object(S, "worker", side_effect=worker), \
                patch.object(S, "finalize") as finalizer, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "Unresolved windows"):
                S.run_full(self.panel, self.output)
        finalizer.assert_not_called()
        progress = S.read_json(self.output / "progress.json")
        self.assertEqual(len(progress["completed"]), 5)
        self.assertEqual(len(progress["failures"]), 1)
        self.assertEqual(progress["pending"], [])
        self.assertIn("deliberate unresolved candidate", progress["failures"][0]["traceback"])


if __name__ == "__main__":
    unittest.main()
