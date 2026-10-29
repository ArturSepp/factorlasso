"""Independent comparator references, hard limits and pilot training isolation."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from papers.sign_pooling_2026.replication import matf_comparators as M
from papers.sign_pooling_2026.replication import matf_pilot_protocol as P
from papers.sign_pooling_2026.replication import verify_matf_comparators as V
from papers.sign_pooling_2026.replication import french_estimators as F


class ComparatorTests(unittest.TestCase):
    def data(self, span=None):
        rng = np.random.default_rng(12094)
        x = rng.normal(size=(60, 3))
        x[:, 1] += 0.8 * x[:, 0]
        y = x @ np.array([[1.0, -0.3, 0], [0.5, 0.1, -0.2], [-0.4, 0.8, 0.1]]).T
        y += rng.normal(size=y.shape) * 0.3
        dates = pd.date_range("2000-01-31", periods=60, freq="ME")
        return M.prepare(pd.DataFrame(x, index=dates), pd.DataFrame(y, index=dates), span)

    def scalar(self):
        # Keep raw arrays fixed: standardized slope is -1 and Gram=1.
        x = pd.DataFrame(
            {"x": [-2.0, -1.0, 0.0, 1.0, 2.0]},
            index=pd.date_range("2000-01-31", periods=5, freq="ME"),
        )
        data = M.prepare(x, -x)
        return data, M.specification(data, labels=np.zeros(1, int))

    def test_scalar_soft_closed_form(self):
        data, spec = self.scalar()
        for lam, strength in ((0, 1), (0.2, 2), (0.4, 10)):
            result = M.fit(data, spec, lam, soft=np.ones((1, 1)), strength=strength)
            self.assertAlmostEqual(result["beta"][0, 0], -(1 - lam / 2) / (1 + strength), places=6)

    def test_allowed_direction_does_not_shrink_magnitude(self):
        data, spec = self.scalar()
        base = M.fit(data, spec, 0.2)["beta"]
        soft = M.fit(data, spec, 0.2, soft=-np.ones((1, 1)), strength=10)["beta"]
        np.testing.assert_allclose(soft, base, atol=1e-7)

    def test_soft_direction_and_zero_limits_are_distinct(self):
        data, spec = self.scalar()
        released = M.fit(data, spec, 0.2, soft=np.zeros((1, 1)), strength=20)["beta"]
        self.assertAlmostEqual(released[0, 0], -0.9, places=6)
        zero = M.fit(data, spec, 0.2, soft=np.zeros((1, 1)), zero_strength=20)["beta"]
        self.assertAlmostEqual(zero[0, 0], -0.9 / 21, places=6)

    def test_soft_penalty_converges_to_hard_restriction(self):
        data, spec = self.scalar()
        hard = M.fit(data, spec, 0.2, hard=np.ones((1, 1)))["beta"]
        errors = [
            abs(M.fit(data, spec, 0.2, soft=np.ones((1, 1)), strength=v)["beta"] - hard).item()
            for v in (1, 100, 10000)
        ]
        self.assertTrue(errors[2] < errors[1] < errors[0])
        self.assertLess(errors[-1], 1e-4)
        hard_zero = M.fit(data, spec, 0.2, hard=np.zeros((1, 1)))["beta"]
        soft_zero = M.fit(data, spec, 0.2, soft=np.zeros((1, 1)), zero_strength=10000)["beta"]
        self.assertLess(abs(soft_zero - hard_zero).item(), 1e-4)

    def test_soft_gradient_by_finite_difference(self):
        b = np.array([[0.4, -0.6, 0.8, -0.2]])
        s = np.array([[1, 1, 0, np.nan]])
        _, gradient = M.soft_loss_gradient(b, s, 2, 3)
        numerical = np.zeros_like(b)
        for j in range(4):
            shift = np.zeros_like(b)
            shift[0, j] = 1e-6
            numerical[0, j] = (
                M.soft_loss_gradient(b + shift, s, 2, 3)[0]
                - M.soft_loss_gradient(b - shift, s, 2, 3)[0]
            ) / (2e-6)
        np.testing.assert_allclose(gradient, numerical, atol=1e-9)

    def test_zero_soft_strength_recovers_ungated_objective(self):
        data = self.data()
        for family in ("hcgl", "fcgl"):
            spec = M.specification(data, family, labels=np.array([0, 0, 1]))
            base = M.fit(data, spec, 0.1 * spec.scale)
            zero = M.fit(data, spec, 0.1 * spec.scale, soft=np.ones_like(spec.signs))
            np.testing.assert_allclose(base["beta"], zero["beta"], atol=1e-10)
            check = F.audit(data.weighted, spec, base["beta"], 0.1 * spec.scale)
            self.assertAlmostEqual(base["objective"], check["objective"], places=12)

    def test_conditional_date_variance_matches_fwl(self):
        for span in (None, 120):
            data = self.data(span)
            labels = np.array([0, 0, 1])
            actual = M.gate(data, labels, "conditional")
            b, se = V.partial_reference(data.state.x, data.state.y, data.weights, labels)
            np.testing.assert_allclose(actual["slopes"], b, atol=1e-12)
            np.testing.assert_allclose(actual["standard_errors"], se, atol=1e-12)

    def test_conditional_gate_corrects_known_marginal_conflict(self):
        rng = np.random.default_rng(314)
        x = rng.normal(size=(300, 2))
        x[:, 1] = 0.9 * x[:, 0] + 0.3 * x[:, 1]
        y = -x[:, 0] + 2 * x[:, 1] + 0.05 * rng.normal(size=len(x))
        dates = pd.date_range("1990-01-31", periods=len(x), freq="ME")
        data = M.prepare(pd.DataFrame(x, index=dates), pd.DataFrame({"y": y}, index=dates))
        self.assertEqual(M.gate(data, [0], "marginal")["signs"][0, 0], 1)
        self.assertEqual(M.gate(data, [0], "conditional")["signs"][0, 0], -1)

    def test_duplicate_responses_do_not_create_precision(self):
        data = self.data()
        x = pd.DataFrame(data.state.x)
        y = pd.DataFrame(data.state.y[:, :1])
        x.index = y.index = pd.date_range("2000-01-31", periods=len(x), freq="ME")
        one = M.prepare(x, y)
        repeated = M.prepare(x, pd.concat([y] * 4, axis=1).set_axis(range(4), axis=1))
        for kind in ("marginal", "conditional"):
            a = M.gate(one, [0], kind)
            b = M.gate(repeated, [0] * 4, kind)
            np.testing.assert_allclose(a["standard_errors"][0], b["standard_errors"][0], atol=1e-12)

    def test_unidentified_conditional_gate_fails(self):
        data = self.data()
        data.state.x[:, 1] = data.state.x[:, 0]
        with self.assertRaisesRegex(ValueError, "unidentified"):
            M.gate(data, [0, 0, 1], "conditional")

    def test_gate_partitions_leave_penalty_geometry_unchanged(self):
        data = self.data()
        spec = M.specification(data, "fcgl", labels=np.array([0, 0, 1]))
        before = spec.key(), spec.scale
        for labels in (spec.labels, np.arange(3)):
            gate = M.gate(data, labels)
            M.fit(data, spec, 0.1 * spec.scale, hard=M.hard_signs(gate["signs"], "release"))
        self.assertEqual(before, (spec.key(), spec.scale))

    def test_release_exclude_and_zero_only_semantics(self):
        gates = np.array([[1.0, -1.0, 0.0]])
        self.assertTrue(np.isnan(M.hard_signs(gates, "release")[0, 2]))
        self.assertEqual(M.hard_signs(gates, "exclude")[0, 2], 0)
        zero = M.hard_signs(gates, "zero_only")
        self.assertTrue(np.isnan(zero[0, :2]).all())
        self.assertEqual(zero[0, 2], 0)

    def test_ridge_augmented_reference_and_endpoints(self):
        for span in (None, 120):
            data = self.data(span)
            for lam in (0.0, 0.3, 10.0):
                a = np.vstack([data.weighted.x, np.sqrt(60 * lam) * np.eye(3)])
                y = np.vstack([data.weighted.y, np.zeros((3, 3))])
                expected = np.linalg.lstsq(a, y, rcond=None)[0].T
                np.testing.assert_allclose(M.ridge_fit(data, lam)["beta"], expected, atol=1e-11)
            self.assertTrue((M.ridge_fit(data, np.inf)["beta"] == 0).all())

    def test_invalid_soft_settings_fail(self):
        data, spec = self.scalar()
        for kwargs in (
            {"strength": -1},
            {"strength": 1},
            {"soft": np.array([[np.inf]])},
            {"soft": np.array([[2.0]])},
            {"hard": np.ones((2, 1))},
        ):
            with self.assertRaises(ValueError):
                M.fit(data, spec, 0.1, **kwargs)

    def test_chronological_selection_training_isolation(self):
        data = self.data()
        x, y = pd.DataFrame(data.state.x), pd.DataFrame(data.state.y)
        x.index = y.index = pd.date_range("2000-01-31", periods=60, freq="ME")
        base = M.chronological_select(x, y, "ridge", [0, 0.1, 1])
        y.iloc[50:] += 100
        changed = M.chronological_select(x, y, "ridge", [0, 0.1, 1])
        self.assertEqual(
            [r["training_hash"] for r in base["folds"]],
            [r["training_hash"] for r in changed["folds"]],
        )
        self.assertEqual(base["folds"][:6], changed["folds"][:6])

    def test_unverified_solver_status_is_rejected(self):
        data, spec = self.scalar()
        original = M.audit

        def invalid(*args, **kwargs):
            result = original(*args, **kwargs)
            result["kkt_residual"] = 1.0
            return result

        with patch.object(M, "audit", invalid), self.assertRaisesRegex(RuntimeError, "Unresolved"):
            M.fit(data, spec, 0.1)


class PilotProtocolTests(unittest.TestCase):
    def calibration_inputs(self):
        rng = np.random.default_rng(91470)
        dates = pd.date_range("2012-01-31", periods=84, freq="ME")
        common = rng.normal(size=(84, 1))
        x = pd.DataFrame(
            0.02 * (common + rng.normal(size=(84, 4)) * 0.3), index=dates, columns=list("ABCD")
        )
        y = pd.DataFrame(
            x.to_numpy() @ rng.normal(size=(4, 9)) + 0.01 * rng.normal(size=(84, 9)),
            index=dates,
            columns=[f"ETF{i}" for i in range(9)],
        )
        groups = pd.DataFrame(
            np.eye(3)[np.arange(9) // 3],
            index=y.columns,
            columns=["Equity", "Fixed Income", "Alternatives"],
        )
        premia = pd.DataFrame(
            0.03, index=pd.to_datetime(["2016-09-30", "2016-12-31"]), columns=x.columns
        )
        return x, y, groups, premia

    def test_future_data_and_unreleased_premia_do_not_enter_calibration(self):
        x, y, groups, premia = self.calibration_inputs()
        arrays, metadata = P.calibration(x, y, groups, premia)
        x.iloc[60:] = 1000
        y.iloc[60:] = -1000
        premia.iloc[-1] = 999
        again, second = P.calibration(x, y, groups, premia)
        self.assertEqual(metadata, second)
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], again[key])
        self.assertEqual(metadata["premium_observation"], "2016-09-30")

    def test_truth_has_verified_alignment_conflict_and_same_signal(self):
        arrays, metadata = P.calibration(*self.calibration_inputs())
        self.assertEqual(metadata["population_geometry"]["aligned"]["opposite_active_cells"], 0)
        self.assertGreaterEqual(
            metadata["population_geometry"]["conflict"]["opposite_active_cells"], 9
        )
        for kind in ("aligned", "conflict"):
            beta = arrays[kind + "_beta"]
            self.assertTrue((np.sum(beta != 0, axis=1) == 3).all())
            variance = np.diag(beta @ arrays["factor_covariance"] @ beta.T)
            np.testing.assert_allclose(variance, arrays["calibrated_signal_variance"])

    def test_missing_membership_or_month_fails(self):
        x, y, groups, premia = self.calibration_inputs()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            P.calibration(x.drop(x.index[10]), y, groups, premia)
        groups.iloc[0] = 0
        with self.assertRaisesRegex(ValueError, "one-hot"):
            P.calibration(x, y, groups, premia)

    def test_inventory_complete_paired_and_bounded(self):
        p = P.design()
        rows = P.inventory(p)
        self.assertEqual(len(p["planned_full_core"]), 16)
        self.assertEqual(len(p["pilot_cells"]), 4)
        self.assertEqual(len(rows), 1080)
        frame = pd.DataFrame(rows).fillna("equal")
        keys = [k for k in frame.columns if k != "replication"]
        self.assertTrue((frame.groupby(keys).replication.nunique() == 20).all())

    def test_draws_are_paired_and_target_is_end_of_history(self):
        arrays, _ = P.calibration(*self.calibration_inputs())
        p = P.design()
        cell = dict(history=60, alignment="aligned", strength="strong", evolution="stable")
        a = P.generate(arrays, cell, 0, p)
        b = P.generate(arrays, dict(cell, alignment="conflict", strength="weak"), 0, p)
        np.testing.assert_array_equal(a["x"], b["x"])
        np.testing.assert_allclose(
            a["y"] - a["x"] @ a["truth"].T, b["y"] - b["x"] @ b["truth"].T, atol=1e-14
        )
        drift = P.generate(arrays, dict(cell, evolution="reversal"), 0, p)
        np.testing.assert_allclose(drift["truth"], -0.5 * arrays["aligned_beta"])
        self.assertEqual(drift["loading_scales"][-1], -0.5)
        self.assertEqual(a["y_test"].shape, (512, 9))


if __name__ == "__main__":
    unittest.main()
