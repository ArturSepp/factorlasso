"""Cache guards and independent fold loss checks for the alpha1 extension."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from papers.sign_pooling_2026.replication import learned_pooling_expanded as E


class ExpandedTests(unittest.TestCase):
    def test_added_surface_uses_fold_training_only(self):
        rng = np.random.default_rng(941)
        x, y = rng.normal(size=(30, 4)), rng.normal(size=(30, 3))
        original_prepare = E.B.prepare
        calls = []

        def prepare(a, b, cut):
            calls.append((a.copy(), b.copy(), cut))
            return original_prepare(a, b, cut)

        def path(prep, method, tau, protocol):
            self.assertEqual(protocol.alpha, 1.)
            count = len(protocol.lambda_fractions)
            return (np.zeros((count, 3, 4)), np.zeros((count, 3)),
                    np.zeros(count), np.zeros((3, 4)))

        with patch.object(E.B, "prepare", side_effect=prepare), patch.object(
                E.B, "fit_path", side_effect=path):
            candidates, _, losses = E.additional_surface(x, y, "hcgl_pooled_zero", 17, "MOSEK")
        expected = 0.
        cursor = 0
        for train, valid in E.B.cv_splits(30, 3, 17):
            standardized = (y[valid]-y[train].mean(axis=0))/y[train].std(axis=0)
            expected += np.square(standardized).sum()/3/30
            for cut in E.L.CUTS:
                a, b, actual_cut = calls[cursor]
                np.testing.assert_array_equal(a, x[train])
                np.testing.assert_array_equal(b, y[train])
                self.assertEqual(actual_cut, cut)
                cursor += 1
        self.assertEqual(len(calls), 9)
        self.assertEqual(len(candidates), 9)
        np.testing.assert_allclose(losses, expected, atol=1e-14)

    def test_prior_cache_rejects_seed_order_grid_and_nonfinite_mutations(self):
        method, seed, scenario = "hcgl_unconstrained", 1000, "coherent"
        rows = [dict(method=method, seed=seed, scenario=scenario, cut=c.cut,
                     alpha=c.alpha, tau=c.tau, lambda_fraction=fraction, cv_mse=.2)
                for c in E.L.candidates_for(method)
                for fraction in E.B.Protocol().lambda_fractions]
        valid = pd.DataFrame(rows)
        self.assertEqual(len(E.prior_surface(valid, method, seed, scenario)), len(valid))
        bad_seed, bad_grid, bad_loss = valid.copy(), valid.copy(), valid.copy()
        bad_seed.loc[0, "seed"] = 1001
        bad_grid.loc[0, "alpha"] = 1.
        bad_loss.loc[0, "cv_mse"] = np.inf
        for mutation in (bad_seed, bad_grid, bad_loss, valid.iloc[::-1], valid.iloc[1:]):
            with self.assertRaises(ValueError):
                E.prior_surface(mutation, method, seed, scenario)


if __name__ == "__main__":
    unittest.main()
