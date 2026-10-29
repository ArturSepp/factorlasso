"""Independent numerical, temporal-isolation and checkpoint tests for Stage 2."""

import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from factorlasso.linear_model._solvers.lasso import solve_lasso_cvx_problem
from factorlasso.linear_model._solvers.group_lasso import solve_group_lasso_cvx_problem
from factorlasso.linear_model._solvers.cooperative import solve_cooperative_group_lasso_cvx_problem
from papers.sign_pooling_2026.replication import french_estimators as F
from papers.sign_pooling_2026.replication import french_rolling_validation as R
from papers.sign_pooling_2026.replication.french_protocol import study_protocol


def fixture(n=36, p=3, q=4):
    rng = np.random.default_rng(206)
    dates = pd.date_range("2000-01-05", periods=n, freq="W-WED")
    x = pd.DataFrame(rng.normal(size=(n, p)), index=dates,
                     columns=[f"x{k}" for k in range(p)])
    beta = rng.normal(size=(q, p))
    y = pd.DataFrame(3+x.to_numpy()@beta.T+.3*rng.normal(size=(n, q)), index=dates,
                     columns=[f"y{k}" for k in range(q)])
    return x, y


def candidate(method, fraction=.3, partition=.5, tau=.75):
    c = {"method": method}
    if method == "ridge":
        c["ridge_lambda"] = .1
    elif method != "ols":
        c.update(fraction=fraction, partition=partition if method.startswith(
            ("hcgl", "fcgl")) or method == "cooperative" else None,
            tau=tau if "exclude" in method or method == "hcgl_unrestricted" else None)
    return c


def small_protocol():
    p = copy.deepcopy(study_protocol())
    p["methods"] = ["ols", "ridge", "hcgl_exclude"]
    p["grid"]["fractions"] = [1, .3, 0]
    p["grid"]["ridge_lambdas"] = ["inf", 1, .1, 0]
    p["clustering"]["partitions"] = ["singleton", .5, "one_cluster"]
    p["gate"]["thresholds"] = [0, "inf"]
    p["grid"]["refinement"]["maximum_passes"] = 0
    p["calendar"]["outer_horizon"] = 4
    p["calendar"]["inner_train_ends"]["12"] = [6, 8, 10]
    p["calendar"]["inner_validation_ends"]["12"] = [8, 10, 12]
    return p


class NumericalTests(unittest.TestCase):
    def setUp(self):
        self.x, self.y = fixture()
        self.state = F.prepare(self.x, self.y)
        self.protocol = study_protocol()

    def test_all_twelve_methods_finite_and_optimal(self):
        for method in self.protocol["methods"]:
            with self.subTest(method=method):
                c = candidate(method)
                spec = F.specification(self.state, c)
                lam = .1 if method == "ridge" else .3*spec.scale
                r = F.fit_path(self.state, spec, [lam], self.protocol)[0]
                self.assertTrue(np.isfinite(r["beta"]).all())
                self.assertLessEqual(r["kkt_residual"], F.KKT_TOL)
                self.assertLessEqual(r["constraint_violation"], 1e-7)
                b, a = F.original_units(self.state, r)
                np.testing.assert_allclose(self.x.to_numpy()@b.T+a,
                    (self.state.x@r["beta"].T+r["alpha"])*self.state.y_sd+self.state.y_mean,
                    atol=1e-12, rtol=1e-12)

    def test_package_objective_parity(self):
        for method in ("lasso", "hcgl_exclude", "fcgl_exclude", "cooperative"):
            with self.subTest(method=method):
                spec = F.specification(self.state, candidate(method))
                lam = .3*spec.scale
                fitted = F.fit_path(self.state, spec, [lam], self.protocol)[0]
                x, y = self.state.x, self.state.y
                groups = np.eye(len(np.unique(spec.labels)))[spec.labels]
                with F.safe_cvxpy_sum_shape():
                    if method == "lasso":
                        reference = solve_lasso_cvx_problem(x, y, reg_lambda=lam)
                    elif method == "cooperative":
                        reference = solve_cooperative_group_lasso_cvx_problem(
                            x, y, groups, reg_lambda=lam, group_penalty="normalized", l1_weight=0)
                    else:
                        reference = solve_group_lasso_cvx_problem(
                            x, y, groups, reg_lambda=lam, group_penalty="normalized", l1_weight=.5,
                            factors_beta_loading_signs=spec.signs,
                            block_mode="cluster_factor" if method.startswith("fcgl") else "row")
                expected = reference.estimated_beta
                self.assertLessEqual(abs(F.audit(self.state, spec, expected, lam)["objective"]-
                                         fitted["objective"]), 1e-6)
                np.testing.assert_allclose(fitted["beta"], expected, atol=2e-4, rtol=2e-4)

    def test_zero_and_infinite_gate_endpoints(self):
        for method in self.protocol["methods"]:
            c = candidate(method, fraction=0, tau="inf")
            spec = F.specification(self.state, c)
            r = F.fit_path(self.state, spec, [0.], self.protocol)[0]
            if method in ("singleton_exclude", "hcgl_singleton_exclude",
                          "hcgl_exclude", "fcgl_exclude"):
                self.assertEqual(np.max(np.abs(r["beta"])), 0.)
            elif method != "unilasso":
                expected = np.linalg.lstsq(self.state.x, self.state.y, rcond=None)[0].T
                np.testing.assert_allclose(r["beta"], expected, atol=1e-12)
        for method in ("lasso", "hcgl_exclude", "fcgl_exclude", "cooperative", "unilasso"):
            spec = F.specification(self.state, candidate(method))
            r = F.fit_path(self.state, spec, [spec.scale], self.protocol)[0]
            self.assertEqual(np.max(np.abs(r["beta"])), 0.)

    def test_ridge_and_ols_against_augmented_least_squares(self):
        spec = F.specification(self.state, candidate("ridge"))
        for lam in (0., .01, 10.):
            r = F.fit_path(self.state, spec, [lam], self.protocol)[0]
            xx = np.vstack([self.state.x, np.sqrt(len(self.x)*lam)*np.eye(self.x.shape[1])])
            yy = np.vstack([self.state.y, np.zeros((self.x.shape[1], self.y.shape[1]))])
            expected = np.linalg.lstsq(xx, yy, rcond=None)[0].T
            np.testing.assert_allclose(r["beta"], expected, atol=1e-12)

    def test_unilasso_intercept_loo_against_explicit_refits(self):
        state = self.state
        eta, mean, slopes, constants = state.unilasso()
        reconstructed = eta+mean
        for k in range(state.y.shape[1]):
            for j in range(state.x.shape[1]):
                design = np.c_[np.ones(len(state.x)), state.x[:, j]]
                for t in range(len(state.x)):
                    keep = np.arange(len(state.x)) != t
                    ab = np.linalg.lstsq(design[keep], state.y[keep, k], rcond=None)[0]
                    self.assertAlmostEqual(reconstructed[t, k, j], design[t]@ab, places=12)
        spec = F.specification(state, candidate("unilasso", fraction=0))
        r = F.fit_path(state, spec, [0], self.protocol)[0]
        theta = np.stack([nnls(eta[:, k], state.y[:, k])[0] for k in range(state.y.shape[1])])
        np.testing.assert_allclose(r["beta"], theta*slopes, atol=1e-12)
        np.testing.assert_allclose(r["alpha"], np.sum(theta*(constants-mean), axis=1), atol=1e-12)

    def test_response_groups_are_sliced_not_passed_as_predictor_groups(self):
        x, y = fixture(n=80, p=2, q=4)
        y.iloc[:, :2] = x.iloc[:, 0].to_numpy()[:, None]+np.arange(2)*.01
        y.iloc[:, 2:] = -x.iloc[:, 0].to_numpy()[:, None]+np.arange(2)*.01
        state = F.prepare(x, y)
        labels = np.array([0, 0, 1, 1])
        signs = state.gates(labels, 0)
        np.testing.assert_array_equal(signs[:2, 0], [1, 1])
        np.testing.assert_array_equal(signs[2:, 0], [-1, -1])

    def test_same_solver_retry_and_unresolved_failure(self):
        spec = F.specification(self.state, candidate("lasso"))
        original = cp.Problem.solve
        count = [0]
        def altered(problem, *args, **kwargs):
            count[0] += 1
            value = original(problem, *args, **kwargs)
            if count[0] == 1:
                problem._status = cp.OPTIMAL_INACCURATE
            return value
        with patch.object(cp.Problem, "solve", altered):
            r = F.fit_path(self.state, spec, [.3*spec.scale], self.protocol)[0]
        self.assertEqual(len(r["attempts"]), 2)
        self.assertEqual({a["solver"] for a in r["attempts"]}, {"CLARABEL"})
        def always_bad(problem, *args, **kwargs):
            value = original(problem, *args, **kwargs)
            problem._status = cp.OPTIMAL_INACCURATE
            for variable in problem.variables():
                variable.value = np.full(variable.shape, 10.)
            return value
        with patch.object(cp.Problem, "solve", always_bad), self.assertRaises(RuntimeError):
            F.fit_path(self.state, spec, [.3*spec.scale], self.protocol)

    def test_numerical_refinement_preserves_objective_and_constraints(self):
        for method in ("lasso", "hcgl_exclude", "fcgl_exclude", "cooperative", "unilasso"):
            with self.subTest(method=method):
                spec = F.specification(self.state, candidate(method))
                lam = .3*spec.scale
                initial = np.zeros_like(spec.signs)
                result, details = F.polish(self.state, spec, initial, lam)
                checked = F.audit(self.state, spec, result, lam)
                self.assertTrue(details["accepted"])
                self.assertLessEqual(details["objective_change"], 1e-12)
                self.assertLessEqual(checked["kkt_residual"], F.KKT_TOL)
                self.assertLessEqual(checked["constraint_violation"], 1e-7)

    def test_inaccurate_status_is_retained_and_resolved_after_retry(self):
        spec = F.specification(self.state, candidate("lasso"))
        original = cp.Problem.solve
        def flagged(problem, *args, **kwargs):
            value = original(problem, *args, **kwargs)
            problem._status = cp.OPTIMAL_INACCURATE
            return value
        with patch.object(cp.Problem, "solve", flagged):
            result = F.fit_path(self.state, spec, [.3*spec.scale], self.protocol)[0]
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(result["status"], "resolved_inaccurate")
        chosen = result["attempts"][result["selected_attempt_index"]]
        self.assertEqual(chosen["status"], cp.OPTIMAL_INACCURATE)
        self.assertTrue(chosen["inaccurate_flag_resolved_by_independent_checks"])
        self.assertLessEqual(chosen["independent_relative_dual_gap"], 1e-8)


class ChronologicalTests(unittest.TestCase):
    def run_quietly(self, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return R.run_window(*args, **kwargs)

    def test_future_outcomes_cannot_change_earlier_fits_or_selection(self):
        x, y = fixture(n=24, p=2, q=3)
        p = small_protocol()
        a = self.run_quietly(x, y, 16, 12, p)
        changed = y.copy()
        changed.iloc[16:] += 10000
        b = self.run_quietly(x, changed, 16, 12, p)
        self.assertEqual(a["preprocessing"], b["preprocessing"])
        self.assertEqual(a["selections"], b["selections"])
        for method in p["methods"]:
            np.testing.assert_array_equal(a["arrays"][method+"__beta"],
                                          b["arrays"][method+"__beta"])

    def test_inner_validation_cannot_change_its_training_state(self):
        x, y = fixture(n=12, p=2, q=3)
        p = small_protocol()
        with contextlib.redirect_stdout(io.StringIO()):
            a = R.select_window(x, y, p, 12)
            changed = y.copy()
            changed.iloc[6:8] += 100
            b = R.select_window(x, changed, p, 12)
        self.assertEqual(a["evaluators"][0].state.manifest(), b["evaluators"][0].state.manifest())
        self.assertNotEqual(a["surfaces"]["ols"][0]["score"], b["surfaces"]["ols"][0]["score"])

    def test_resume_rejects_stale_or_corrupted_checkpoints(self):
        x, y = fixture(n=24, p=2, q=3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = self.run_quietly(x, y, 16, 12, small_protocol(), folder=root, identity="fixed")
            with patch.object(F, "fit_path", side_effect=AssertionError("refit")):
                b = self.run_quietly(x, y, 16, 12, small_protocol(), folder=root, identity="fixed")
            self.assertEqual(a, b)
            with self.assertRaisesRegex(ValueError, "Stale"):
                self.run_quietly(x, y, 16, 12, small_protocol(), folder=root, identity="different")
            with self.assertRaisesRegex(ValueError, "Stale"):
                self.run_quietly(x, y, 17, 12, small_protocol(), folder=root, identity="fixed")
            arrays = root / "outer_arrays.npz"
            arrays.write_bytes(arrays.read_bytes()+b"corruption")
            with self.assertRaisesRegex(ValueError, "Corrupted"):
                self.run_quietly(x, y, 16, 12, small_protocol(), folder=root, identity="fixed")

    def test_tie_order_refinements_and_no_silent_candidate_failure(self):
        p = study_protocol()
        choices = R.candidates(p, "hcgl_exclude")
        rows = [{"candidate": c, "score": 1. if c["fraction"] == .03 and
                 c["partition"] == .35 and c["tau"] == 3 else 2.} for c in choices]
        self.assertEqual(R.refinement_for(rows, p),
                         {"fraction": True, "partition": True, "tau": True})
        ties = [{"candidate": c, "score": 1.} for c in choices]
        self.assertEqual(R.choose(ties, p)[0]["candidate"], choices[0])
        state = F.prepare(*fixture())
        evaluator = R.Evaluator(state, p)
        with patch.object(F, "fit_path", side_effect=RuntimeError("failed candidate")):
            with self.assertRaisesRegex(RuntimeError, "failed candidate"):
                evaluator.evaluate([candidate("lasso")])

    def test_partial_fit_cache_resume_reproduces_uninterrupted_values(self):
        state = F.prepare(*fixture())
        choices = [candidate("lasso", fraction=f) for f in (.8, .3, .1)]
        p = study_protocol()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            before = R.Evaluator(state, p, folder, "fixed").evaluate(choices)
            removed = folder / (before[1][3]+".npz")
            removed.unlink()
            after = R.Evaluator(state, p, folder, "fixed").evaluate(choices)
            for a, b in zip(before, after):
                self.assertEqual(a[3], b[3])
                np.testing.assert_allclose(a[2]["beta"], b[2]["beta"], atol=1e-10, rtol=1e-10)
                np.testing.assert_array_equal(a[2]["alpha"], b[2]["alpha"])


if __name__ == "__main__":
    unittest.main()
