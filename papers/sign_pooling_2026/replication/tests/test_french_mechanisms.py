"""Independent mechanism references, paired truth designs and training-only budgets."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from papers.sign_pooling_2026.replication import french_estimators as F
from papers.sign_pooling_2026.replication import french_mechanisms as M
from papers.sign_pooling_2026.replication import french_semisynthetic as S
from papers.sign_pooling_2026.replication import verify_french_mechanisms as V
from papers.sign_pooling_2026.replication.french_protocol import study_protocol


class MechanismTests(unittest.TestCase):
    def state(self):
        rng = np.random.default_rng(925)
        x = rng.normal(size=(20, 2))
        y = x @ np.array([[1.0, -0.5], [0.7, 0.3], [-0.6, 0.9]]).T + rng.normal(size=(20, 3)) * 0.4
        dates = pd.date_range("2000-01-05", periods=20, freq="W-WED")
        return F.prepare(pd.DataFrame(x, index=dates), pd.DataFrame(y, index=dates))

    def test_budget_rejects_better_loss_above_ceiling(self):
        rows = [
            {"candidate": {"fraction": 0.1}, "score": 0.01, "inner_active_fraction": 0.9},
            {"candidate": {"fraction": 0.3}, "score": 0.03, "inner_active_fraction": 0.6},
            {"candidate": {"fraction": 1}, "score": 1.0, "inner_active_fraction": 0.0},
        ]
        selected = M.choose_budget(rows, 0.65, study_protocol())
        self.assertEqual(selected["candidate"]["fraction"], 0.3)
        self.assertIsNone(M.choose_budget(rows[:1], 0.5, study_protocol()))

    def test_budget_tie_order_and_endpoint(self):
        rows = [
            {"candidate": {"fraction": 0.5}, "score": 0.1, "inner_active_fraction": 0.5},
            {"candidate": {"fraction": 0.3}, "score": 0.1 - 1e-11, "inner_active_fraction": 0.4},
        ]
        self.assertEqual(M.choose_budget(rows, 0.5, study_protocol())["candidate"]["fraction"], 0.5)

    def test_pooling_identical_responses_has_no_precision_gain(self):
        state = self.state()
        state.y = np.repeat(state.y[:, [0]], 3, axis=1)
        labels = np.zeros(3, int)
        for row in M.gate_precision(state, labels):
            self.assertAlmostEqual(row["precision_ratio"], 1.0, places=12)
            self.assertAlmostEqual(row["variance_inflation"], 3.0, places=12)

    def test_gate_matches_explicit_date_score_reference(self):
        state = self.state()
        labels = np.array([0, 0, 1])
        for tau in (0, 2):
            np.testing.assert_array_equal(
                state.gates(labels, tau), V.gate(state.x, state.y, labels, tau)
            )

    def test_matched_variants_keep_geometry_and_penalty(self):
        state = self.state()
        base = F.specification(state, {"method": "fcgl_unconstrained", "partition": "one_cluster"})
        for variant in M.VARIANTS:
            spec = M.gate_spec(state, base, variant, 2)
            np.testing.assert_array_equal(spec.labels, base.labels)
            np.testing.assert_array_equal(spec.weights, base.weights)
            self.assertEqual(spec.family, base.family)
        release = M.gate_spec(state, base, "pooled_release", np.inf)
        self.assertTrue(np.isnan(release.signs).all())
        exclude = M.gate_spec(state, base, "pooled_exclude", np.inf)
        self.assertTrue((exclude.signs == 0).all())

    def test_zero_penalty_families_same_feasible_problem(self):
        state = self.state()
        labels = np.zeros(3, int)
        weights = np.repeat(np.sqrt(3), 3)
        signs = np.array([[1.0, 0.0], [np.nan, 1.0], [-1.0, 1.0]])
        answers = []
        for family in ("row", "block"):
            spec = F.Spec(family, labels, weights, signs, 1.0)
            answer = F.fit_path(state, spec, [0], study_protocol())[0]
            V.check_fit(
                state.x, state.y, answer["beta"], signs, labels, 0, family, answer["objective"]
            )
            answers.append(answer["beta"])
        np.testing.assert_allclose(*answers, atol=1e-12)

    def test_known_support_and_exclusion_definitions(self):
        truth = np.array([[1.0, -1.0], [0.0, 2.0]])
        fitted = np.array([[0.5, 0.0], [0.2, -1.0]])
        signs = np.array([[1.0, 0.0], [np.nan, -1.0]])
        metrics = S.known_metrics(
            fitted,
            np.zeros(2),
            truth,
            np.eye(2),
            signs,
            truth != 0,
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            1e-4,
        )
        self.assertAlmostEqual(metrics["correct_active_fitted_sign"], 1 / 3)
        self.assertAlmostEqual(metrics["false_exclusion"], 1 / 3)
        self.assertAlmostEqual(metrics["gate_opposite_active"], 1 / 3)
        self.assertAlmostEqual(metrics["support_precision"], 2 / 3)
        self.assertAlmostEqual(metrics["support_recall"], 2 / 3)

    def test_population_risk_matches_exact_quadrature(self):
        truth = np.array([[1.0, -0.5], [0.2, 0.3]])
        fitted = np.array([[0.7, 0.1], [0.0, 0.5]])
        intercept = np.array([0.1, -0.2])
        corr = np.array([[1.0, 0.6], [0.6, 1.0]])
        points = (
            np.vstack([np.sqrt(2) * np.eye(2), -np.sqrt(2) * np.eye(2)])
            @ np.linalg.cholesky(corr).T
        )
        target = points @ truth.T
        pred = points @ fitted.T + intercept
        result = S.known_metrics(
            fitted,
            intercept,
            truth,
            corr,
            np.full_like(truth, np.nan),
            truth != 0,
            target,
            pred,
            1e-4,
        )
        self.assertAlmostEqual(result["excess_risk"], np.square(target - pred).mean(), places=14)

    def test_common_innovations_across_factorial_conditions(self):
        beta = np.array([[1.0, 0.2], [0.5, -0.3], [0.1, 0.7]])
        corr = np.array([[1.0, 0.4], [0.4, 1.0]])
        noise = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
        cal = {}
        for fx in (0, 1):
            key = f"empirical_fx{fx}_ex0"
            cal[key + "__truth"] = beta
            cal[key + "__factor_cov"] = np.eye(2) if fx == 0 else corr
            cal[key + "__error_cov"] = noise
        p = {"seed": 20261026, "test_rows": 10}
        outputs = [
            S.generate(cal, dict(lookback=12, replication=1, coherence="empirical", fx=fx, ex=0), p)
            for fx in (0, 1)
        ]
        z0 = outputs[0][0]
        z1 = np.linalg.solve(np.linalg.cholesky(corr), outputs[1][0].T).T
        np.testing.assert_allclose(z0, z1, atol=1e-14)
        np.testing.assert_allclose(
            outputs[0][1] - outputs[0][0] @ beta.T,
            outputs[1][1] - outputs[1][0] @ beta.T,
            atol=1e-14,
        )

    def test_independent_table_check_rejects_wrong_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            pd.DataFrame([{"key": "a", "value": 2.0}]).to_csv(path / "probe.csv", index=False)
            with self.assertRaises(AssertionError):
                V.table(path, "probe", [{"key": "a", "value": 1.0}], ["key"])

    def test_training_selection_has_no_future_input(self):
        state = self.state()
        dates = pd.date_range("2000-01-05", periods=20, freq="W-WED")
        x = pd.DataFrame(state.x, index=dates)
        y = pd.DataFrame(state.y, index=dates)
        original = study_protocol()
        original["methods"] = ["lasso", "hcgl_unconstrained", "fcgl_unconstrained"]
        original["calendar"]["inner_train_ends"]["12"] = [6, 8, 10]
        original["calendar"]["inner_validation_ends"]["12"] = [8, 10, 12]
        p = M.protocol()
        p["fractions"] = [1, 0.1, 0]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            old = folder / "old"
            old.mkdir()
            first = M.train_select(
                x.iloc[:12],
                y.iloc[:12],
                original,
                p,
                old,
                {"identity": "old", "files_sha256": {}},
                folder / "new",
                "unit",
            )
            changed = y.copy()
            changed.iloc[12:] += 10000
            second = M.train_select(
                x.iloc[:12],
                changed.iloc[:12],
                original,
                p,
                old,
                {"identity": "old", "files_sha256": {}},
                folder / "new",
                "unit",
            )
            self.assertEqual(first[:3], second[:3])

    def test_inventory_complete_and_unique(self):
        p = M.protocol()["simulation"]
        jobs = S.inventory(p)
        self.assertEqual(len(jobs), 640)
        self.assertEqual(len({j["key"] for j in jobs}), 640)


if __name__ == "__main__":
    unittest.main()
