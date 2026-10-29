"""Published comparator identities, fair tuning, frozen counts and deliberate defects."""

import unittest
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from papers.sign_pooling_2026.replication import matf_full_engine as E
from papers.sign_pooling_2026.replication import matf_full_study as R
from papers.sign_pooling_2026.replication import matf_pilot_protocol as P
from papers.sign_pooling_2026.replication import verify_matf_full as V


class FullTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(18942)
        x = rng.normal(size=(60, 3))
        x[:, 1] += 0.7 * x[:, 0]
        y = x @ rng.normal(size=(3, 5)) + rng.normal(size=(60, 5)) * 0.4
        dates = pd.date_range("2000-01-31", periods=60, freq="ME")
        return E.Window(pd.DataFrame(x, index=dates), pd.DataFrame(y, index=dates), None)

    def test_inventory_all_cells_and_diagnostics(self):
        old = P.design()
        p = dict(diagnostic_cells=old["pilot_cells"])
        allrows = [R.record_inventory(p, c["key"]) for c in old["planned_full_core"]]
        self.assertEqual(sum(map(len, allrows)) * 200, 89600)
        for cell in old["planned_full_core"]:
            methods = E.methods(cell["key"], p["diagnostic_cells"])
            self.assertEqual(len({m["name"] for m in methods}), len(methods))
            self.assertTrue(
                {"ols", "ridge", "lasso", "cooperative", "unilasso"} <= {m["name"] for m in methods}
            )
            self.assertIn("fcgl_pooled_exclude", {m["name"] for m in methods})

    def test_unilasso_features_against_brute_leave_one_out(self):
        win = self.data()
        x, y = win.data.state.x, win.data.state.y
        eta, mean, slopes, constants = V.univariate(x, y)
        producer = win.univariate()
        np.testing.assert_allclose(producer["eta"], eta, atol=1e-11)
        for i in (0, 17, 59):
            keep = np.arange(len(x)) != i
            for j in range(x.shape[1]):
                design = np.column_stack([np.ones(keep.sum()), x[keep, j]])
                beta = np.linalg.lstsq(design, y[keep], rcond=None)[0]
                np.testing.assert_allclose(
                    eta[i, :, j] + mean[:, j], beta[0] + x[i, j] * beta[1], atol=1e-11
                )
        np.testing.assert_allclose(producer["slopes"], slopes, atol=1e-11)
        np.testing.assert_allclose(producer["constants"], constants, atol=1e-11)

    def test_unilasso_zero_endpoint_and_intercept(self):
        win = self.data()
        u = win.univariate()
        fit = E.uni_fit(win, 0)
        theta = np.array([nnls(u["eta"][:, i], win.data.state.y[:, i])[0] for i in range(5)])
        np.testing.assert_allclose(fit["parameter"], theta, atol=1e-10)
        np.testing.assert_allclose(fit["beta"], theta * u["slopes"], atol=1e-10)
        self.assertTrue(np.any(abs(fit["alpha"]) > 1e-8))
        self.assertLess(E.uni_audit(win, fit["parameter"], 0)["kkt"], 1e-7)

    def test_public_ridge_endpoints(self):
        win = self.data()
        method = next(m for m in E.methods("", []) if m["name"] == "ridge")
        results = E.path(win, method, [0.0, 0.1, "inf"])
        np.testing.assert_array_equal(results["inf"]["beta"], np.zeros((5, 3)))
        x, y = win.data.state.x, win.data.state.y
        for lam in (0.0, 0.1):
            ref = np.linalg.lstsq(
                np.vstack([x, np.sqrt(len(x) * lam) * np.eye(3)]),
                np.vstack([y, np.zeros((3, 5))]),
                rcond=None,
            )[0].T
            np.testing.assert_allclose(results[lam]["beta"], ref, atol=1e-8)

    def test_own_penalties_can_differ_with_identical_folds(self):
        for best in (0.0, 0.03, 1.0):
            rows = [
                dict(stop=n, candidate=c, mse=(c - best) ** 2)
                for n in (30, 40, 50)
                for c in (0.0, 0.03, 1.0)
            ]
            self.assertEqual(E.select(rows, [0.0, 0.03, 1.0])[0], best)
        with self.assertRaises(ValueError):
            E.select(rows[:-1], [0.0, 0.03, 1.0])
        with self.assertRaises(ValueError):
            E.select(rows + [rows[0]], [0.0, 0.03, 1.0])

    def test_ties_choose_infinite_ridge_endpoint(self):
        rows = [
            dict(stop=n, candidate=c, mse=1.0) for n in (30, 40, 50) for c in (0.0, 10000.0, "inf")
        ]
        self.assertEqual(E.select(rows, [0.0, 10000.0, "inf"])[0], "inf")

    def test_paths_match_independent_stationarity_and_scaling(self):
        win = self.data()
        s = win.data.state
        for method in E.methods("", []):
            if method["family"] in ("ols", "ridge"):
                continue
            uni = V.univariate(s.x, s.y) if method["family"] == "unilasso" else None
            expected = V.scale(s.x, s.y, win.data.weights, win.labels, method["family"], uni)
            self.assertAlmostEqual(win.spec(method["family"]).scale, expected, places=9)
            results = E.path(win, method, [0.0, 0.03, 0.3, 1.0])
            for candidate, result in results.items():
                hard, soft = win.restrictions(method)
                check = V.stationarity(
                    (s.x, s.y, win.data.weights, {}, win.labels, uni),
                    method,
                    result["beta"],
                    result["alpha"],
                    result["parameter"],
                    result["lambda"],
                    hard,
                    soft,
                )
                self.assertLess(check[0], 1e-7)
                self.assertLess(check[1], 1e-9)
            np.testing.assert_allclose(results[1.0]["beta"], 0, atol=1e-8)

    def test_corrupted_coefficient_fails_stationarity(self):
        win = self.data()
        method = next(m for m in E.methods("", []) if m["name"] == "cooperative")
        result = E.path(win, method, [0.03])[0.03]
        changed = result["beta"].copy()
        changed[0, 0] += 0.01
        s = win.data.state
        hard, soft = win.restrictions(method)
        check = V.stationarity(
            (s.x, s.y, win.data.weights, {}, win.labels, None),
            method,
            changed,
            result["alpha"],
            changed,
            result["lambda"],
            hard,
            soft,
        )
        self.assertGreater(check[0], 1e-5)

    def test_window_uses_only_supplied_training_rows(self):
        rng = np.random.default_rng(3)
        dates = pd.date_range("2000-01-31", periods=60, freq="ME")
        x = pd.DataFrame(rng.normal(size=(60, 3)), index=dates)
        y = pd.DataFrame(rng.normal(size=(60, 5)), index=dates)
        before = E.Window(x.iloc[:30], y.iloc[:30], None)
        x.iloc[30:] = 1e5
        y.iloc[30:] = -1e6
        after = E.Window(x.iloc[:30], y.iloc[:30], None)
        np.testing.assert_array_equal(before.cross, after.cross)
        np.testing.assert_array_equal(
            before.gate("marginal")["signs"], after.gate("marginal")["signs"]
        )


if __name__ == "__main__":
    unittest.main()
