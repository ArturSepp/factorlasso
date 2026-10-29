"""Independent monthly units, cached chronology and strict pilot completeness."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from papers.sign_pooling_2026.replication import matf_comparators as M
from papers.sign_pooling_2026.replication import matf_pilot_protocol as P
from papers.sign_pooling_2026.replication import matf_pilot_run as R
from papers.sign_pooling_2026.replication import verify_matf_pilot as V


class PilotTests(unittest.TestCase):
    def sample(self):
        rng = np.random.default_rng(42917)
        dates = pd.date_range("2000-01-31", periods=60, freq="ME")
        x = pd.DataFrame(rng.normal(size=(60, 3)), index=dates)
        y = pd.DataFrame(
            x.to_numpy() @ rng.normal(size=(3, 4)) + rng.normal(size=(60, 4)), index=dates
        )
        return x, y

    def test_known_units_and_cancellation(self):
        beta = np.array([[1.0, -1.0]])
        alpha = np.array([0.01])
        truth = np.zeros_like(beta)
        premium = np.array([0.12, 0.12])
        covariance = np.diag([0.01, 0.02])
        observed, prediction, scales = np.zeros((2, 1)), np.ones((2, 1)), np.array([2.0])
        actual, components = R.losses(
            beta, alpha, truth, premium, covariance, observed, prediction, scales
        )
        self.assertAlmostEqual(actual["attribution_mse"][0], 0.0288)
        self.assertAlmostEqual(actual["mean_mse"][0], 0)
        self.assertAlmostEqual(actual["full_mean_mse"][0], 0.0144)
        self.assertAlmostEqual(actual["annual_prediction_risk"][0], 0.36)
        self.assertAlmostEqual(actual["standardized_test_mse"][0], 0.25)
        ref, rc = V.reference_losses(
            beta, alpha, truth, premium, covariance, observed, prediction, scales
        )
        np.testing.assert_allclose(components, rc)
        for k in ref:
            np.testing.assert_allclose(actual[k], ref[k])

    def test_nonzero_total_premium_error(self):
        args = (
            np.array([[2.0, 1.0]]),
            np.array([-0.01]),
            np.array([[1.0, 0.0]]),
            np.array([0.12, 0.06]),
            np.eye(2),
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            np.ones(1),
        )
        result, _ = R.losses(*args)
        self.assertAlmostEqual(result["mean_mse"][0], 0.18**2)
        self.assertAlmostEqual(result["full_mean_mse"][0], 0.06**2)
        self.assertAlmostEqual(
            result["full_mean_mse"][0],
            result["mean_mse"][0] + result["alpha_mse"][0] + result["mean_alpha_cross"][0],
        )

    def test_ridge_extension_reuses_all_prior_folds(self):
        x, y = self.sample()
        calls, archive = [], {}

        def fake(data, spec, lam, **kwargs):
            calls.append((len(data.state.x), lam))
            beta = np.linalg.lstsq(data.weighted.x, data.weighted.y, rcond=None)[0].T
            if lam != 1000:
                beta *= 0
            return dict(beta=beta, alpha=np.zeros(4), status="fixture")

        with patch.object(R, "measured_fit", side_effect=fake):
            result = R.tuning(x, y, "ridge", None, P.design(), archive, "ridge")
        self.assertEqual(result["selected"], 1000)
        self.assertEqual(len(calls), 24)
        self.assertFalse(result["upper_exhausted"])
        self.assertEqual(len(set(calls)), len(calls))
        self.assertEqual({n for n, _ in calls}, {30, 40, 50})
        for n in (30, 40, 50):
            np.testing.assert_allclose(archive[f"ridge_fold{n}_x_mean"], x.iloc[:n].mean())

    def test_ridge_exhausted_grid_is_recorded(self):
        x, y = self.sample()

        def fake(data, spec, lam, **kwargs):
            return dict(beta=np.zeros((4, 3)), alpha=np.zeros(4), status="fixture")

        with patch.object(R, "measured_fit", side_effect=fake):
            result = R.tuning(x, y, "ridge", None, P.design(), {}, "ridge")
        self.assertEqual(result["selected"], 10000)
        self.assertEqual(len(result["folds"]), 24)
        self.assertTrue(result["upper_exhausted"])

    def test_selection_rejects_missing_duplicate_and_nonfinite(self):
        complete = [dict(candidate=0.0, train_rows=n, mse=1.0) for n in (30, 40, 50)]
        self.assertEqual(R.choose(complete, [0.0])[0], 0.0)
        for wrong in (
            complete[:2],
            complete + [complete[0]],
            [dict(r, mse=np.nan) for r in complete],
        ):
            with self.assertRaises(ValueError):
                R.choose(wrong, [0.0])

    def test_independent_gate_and_partition(self):
        x, y = self.sample()
        for span in (None, 120):
            data = M.prepare(x, y, span)
            xx, yy, w, _ = V.transforms(x.to_numpy(), y.to_numpy(), span)
            spec = M.specification(data)
            np.testing.assert_array_equal(V.partition(yy, w, 0.6), spec.labels)
            actual = M.gate(data, spec.labels)
            b, se = V.marginal_reference(xx, yy, w, spec.labels)
            np.testing.assert_allclose(actual["slopes"], b, atol=1e-10)
            np.testing.assert_allclose(actual["standard_errors"], se, atol=1e-10)

    def test_independent_stationarity_rejects_corruption(self):
        x, y = self.sample()
        data = M.prepare(x, y)
        for family in ("hcgl", "fcgl"):
            spec = M.specification(data, family)
            self.assertAlmostEqual(
                spec.scale,
                V.zero_scale(data.state.x, data.state.y, data.weights, spec.labels, family),
                places=9,
            )
            gate = M.gate(data, spec.labels)["signs"]
            lam = 0.03 * spec.scale
            result = M.fit(data, spec, lam, soft=gate, strength=1)
            args = (data.state.x, data.state.y, data.weights, spec.labels, family)
            kkt, _, objective = V.stationary(*args, result["beta"], lam, spec.signs, gate, 1, 0)
            self.assertLess(kkt, 1e-5)
            self.assertAlmostEqual(objective, result["objective"], places=8)
            broken = result["beta"].copy()
            broken[0, 0] += 0.1
            kkt, _, _ = V.stationary(*args, broken, lam, spec.signs, gate, 1, 0)
            self.assertGreater(kkt, 1e-3)

    def test_complete_inventory_rejects_missing_and_corrupt(self):
        job = dict(
            cell="fixture",
            replication=0,
            family="hcgl",
            span=None,
            arm="ungated",
            soft_strength=0.0,
            zero_strength=0.0,
        )
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            R.write(output / "run_freeze.json", dict(identity="test"))
            with self.assertRaises(ValueError):
                R.complete_inventory(output, [job])

            d = output / "datasets" / R.dataset_key("fixture", 0)
            d.mkdir(parents=True)
            R.write(d / "ledger.json", dict(fits=[dict(job, key=R.identifier(job))]))
            np.savez_compressed(d / "arrays.npz", beta=np.ones((1, 1)))
            receipt = dict(
                run_identity="test",
                planned_sha256=R.digest([job]),
                ledger_sha256=R.sha(d / "ledger.json"),
                arrays_sha256=R.sha(d / "arrays.npz"),
            )
            R.write(d / "complete.json", receipt)
            self.assertEqual(len(R.complete_inventory(output, [job])), 1)
            R.write(d / "ledger.json", dict(fits=[]))
            with self.assertRaises(ValueError):
                R.complete_inventory(output, [job])

    def test_failed_solver_recovery_requires_independent_agreement(self):
        x, y = self.sample()
        data = M.prepare(x, y)
        spec = M.specification(data)
        message = "Unresolved comparator fit: [{'status': 'solver_error', 'warnings': []}]"
        with patch.object(M, "fit", side_effect=RuntimeError(message)):
            result = R.measured_fit(data, spec, 0.1 * spec.scale)
        self.assertEqual(result["status"], "independent_fista_recovery")
        self.assertLessEqual(result["kkt_residual"], 1e-7)
        with (
            patch.object(M, "fit", side_effect=RuntimeError(message)),
            patch.object(R, "reference_fit", return_value=(np.ones_like(spec.signs) * 10, 1)),
        ):
            with self.assertRaises(ValueError):
                R.measured_fit(data, spec, 0.1 * spec.scale)


if __name__ == "__main__":
    unittest.main()
