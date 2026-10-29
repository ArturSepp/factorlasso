"""Known-answer units, cancellation, training isolation and independent risk integration."""

import unittest

import numpy as np
import pandas as pd

from papers.sign_pooling_2026.replication import french_attribution as A
from papers.sign_pooling_2026.replication import verify_french_attribution as V


class AttributionTests(unittest.TestCase):
    def inputs(self):
        rng = np.random.default_rng(742)
        truth = rng.normal(size=(4, 3))
        beta = truth + rng.normal(size=(4, 3))*.2
        alpha = rng.normal(size=4)*.02
        corr = np.array([[1, .7, .2], [.7, 1, -.1], [.2, -.1, 1]])
        return beta, alpha, truth, corr, np.array([.05, -.03, .01]), np.array(
            [.01, .02, .005]), np.array([.01, .02, .03, .04])

    def calibration(self):
        rng = np.random.default_rng(193)
        dates = pd.date_range("2000-01-05", periods=24, freq="W-WED")
        x = pd.DataFrame(rng.normal(size=(24, 3))*.02 + .001, index=dates,
                         columns=["MKT_LOGREL", "SMB", "HML"])
        y = pd.DataFrame(rng.normal(size=(24, 4))*.03, index=dates)
        return x, y

    def test_offsets_cancel_mean_but_not_attribution(self):
        result = A.loss_values(np.array([[1., -1.]]), np.zeros(1), np.zeros((1, 2)),
                               np.array([[1., .99], [.99, 1.]]),
                               np.array([.04, .04]), np.ones(2), np.ones(1))
        self.assertAlmostEqual(result["attribution_mse"], 2*.04**2)
        self.assertAlmostEqual(result["mean_mse"], 0)
        self.assertAlmostEqual(result["annual_prediction_risk"], 52*.02)

    def test_original_units_and_annual_intercept(self):
        # Synthetic beta error=2; raw beta error=2*.03/.01=6.
        result = A.loss_values(np.array([[3.]]), np.array([.1]), np.array([[1.]]),
                               np.ones((1, 1)), np.array([.05]), np.array([.01]),
                               np.array([.03]))
        self.assertAlmostEqual(result["mean_mse"], .3**2)
        self.assertAlmostEqual(result["alpha_mse"], (.1*.03*52)**2)
        self.assertAlmostEqual(result["full_mean_mse"], (.3+.1*.03*52)**2)
        self.assertAlmostEqual(result["annual_prediction_risk"], 52*(2*.03)**2)

    def test_independent_random_reference(self):
        actual, expected = A.loss_values(*self.inputs()), V.reference_losses(*self.inputs())
        for key in actual:
            self.assertAlmostEqual(actual[key], expected[key], places=13, msg=key)

    def test_change_factor_units_leaves_losses_invariant(self):
        args = list(self.inputs())
        expected = A.loss_values(*args)
        scale = np.array([100, .02, 7])
        args[4], args[5] = args[4]*scale, args[5]*scale
        actual = A.loss_values(*args)
        for key in actual:
            self.assertAlmostEqual(actual[key], expected[key], places=13, msg=key)

    def test_alpha_decomposition(self):
        result = A.loss_values(*self.inputs())
        self.assertAlmostEqual(result["full_mean_mse"], result["mean_mse"]
                               + result["alpha_mse"] + result["mean_alpha_cross"])

    def test_zero_premia_leaves_only_intercept(self):
        args = list(self.inputs())
        args[4] = np.zeros(3)
        result = A.loss_values(*args)
        self.assertEqual(result["attribution_mse"], 0)
        self.assertEqual(result["mean_mse"], 0)
        self.assertEqual(result["full_mean_mse"], result["alpha_mse"])

    def test_premium_freeze_ignores_future_rows(self):
        x, y = self.calibration()
        original = A.premium_scenarios(x, y, 16)
        x.iloc[16:] = 1000
        y.iloc[16:] = -1000
        self.assertEqual(original, A.premium_scenarios(x, y, 16))

    def test_scenarios_have_equal_calibration_norm_and_canonical_sign(self):
        x, y = self.calibration()
        cal = A.premium_scenarios(x, y, 16)
        self.assertEqual(len(cal["scenarios"]), 5)
        for scenario in cal["scenarios"]:
            self.assertAlmostEqual(scenario["norm_squared_by_fx"]["1"],
                                   cal["historical_norm_squared"], places=12)
        for scenario in cal["scenarios"][-2:]:
            v = np.array(scenario["annual_premium"]) / cal["factor_weekly_sd"]
            self.assertGreater(v[np.argmax(abs(v))], 0)
        market = cal["scenarios"][1]["annual_premium"]
        self.assertGreater(market[0], 0)
        self.assertEqual(market[1:], [0, 0])

    def test_bad_calibration_rejected(self):
        x, y = self.calibration()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            A.premium_scenarios(x, y, 25)
        x.iloc[:, 0] = 0
        with self.assertRaisesRegex(ValueError, "Zero calibration scale"):
            A.premium_scenarios(x, y, 16)

    def pair_inputs(self):
        rows = []
        for variant, losses in [("pooled_exclude", [1, 4, 2]), ("unconstrained", [2, 2, 2])]:
            for replication, loss in enumerate(losses):
                rows.append(dict(lookback=52, coherence="coherent", fx=0, ex=0,
                                 partition="learned", family="hcgl", fraction=.1,
                                 scenario="training_mean", variant=variant,
                                 replication=replication, attribution_mse=loss))
        protocol = dict(source_simulation=dict(replications=3), metrics=["attribution_mse"],
                        pairs=[["pooled_exclude", "unconstrained"]])
        return pd.DataFrame(rows), protocol

    def test_paired_uncertainty_uses_within_seed_difference(self):
        data, protocol = self.pair_inputs()
        result = A.summaries(data, protocol).iloc[0]
        differences = np.array([-1., 2., 0.])
        self.assertAlmostEqual(result.difference, 1/3)
        self.assertAlmostEqual(result.mc_se, np.std(differences, ddof=1)/np.sqrt(3))

    def test_missing_or_duplicate_replication_is_rejected(self):
        data, protocol = self.pair_inputs()
        for corrupt in (data.iloc[:-1], pd.concat([data, data.iloc[:1]])):
            with self.assertRaisesRegex(ValueError, "Incomplete paired"):
                A.summaries(corrupt, protocol)

    def test_verifier_rejects_missing_and_corrupted_scores(self):
        reference = pd.DataFrame(dict(key=[1, 2], loss=[.1, .2]))
        with self.assertRaisesRegex(ValueError, "Missing or extra"):
            V.compare_tables(reference.iloc[:1], reference, ["key"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            V.compare_tables(pd.concat([reference, reference.iloc[:1]]), reference, ["key"])
        corrupt = reference.copy()
        corrupt.loc[1, "loss"] = .3
        with self.assertRaises(AssertionError):
            V.compare_tables(corrupt, reference, ["key"])

    def test_exact_exclusion_risk_against_quadrature(self):
        for beta in (0., .25, 1., 3.):
            for sd, rho in ((1, 1), (.2, .2), (.8, -.3), (1.2, 0)):
                actual = A.exclusion_risk(beta, .7, sd, rho, 2)
                expected = V.reference_exclusion_risk(beta, .7, sd, rho, 2)
                self.assertAlmostEqual(actual, expected, places=11)

    def test_gate_risk_contains_false_exclusion_cost(self):
        # Almost-certain exclusion: squared bias, not zero estimation risk.
        risk = A.exclusion_risk(.5, 1, 1, .7, 30)
        self.assertAlmostEqual(risk, .25)


if __name__ == "__main__":
    unittest.main()
