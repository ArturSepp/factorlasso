"""Independent report references, terminal weighting, corruption and no-refit tests."""

import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from papers.sign_pooling_2026.replication import french_report as R
from papers.sign_pooling_2026.replication import verify_french_tables as V
from papers.sign_pooling_2026.replication.french_protocol import study_protocol, sha256, json_bytes


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def synthetic_run(root):
    """Known coefficients and identities, not represented as real estimator output."""
    panel, fit = root / "synthetic_panel", root / "synthetic_fit"
    (panel / "panel").mkdir(parents=True)
    (fit / "source").mkdir(parents=True)
    p = copy.deepcopy(study_protocol())
    p.update(responses=4, predictors=["SMB", "HML"], scope="Synthetic unit fixture only")
    p["calendar"].update(
        weekly_rows=21,
        lookbacks=[8, 12],
        outer_burn_in=12,
        outer_horizon=4,
        origins=3,
        test_weeks=9,
    )
    p["reporting"]["bootstrap"]["resamples"] = 80
    p["reporting"]["bootstrap"]["mean_blocks"] = {"8": 2, "12": 3}
    write_json(panel / "protocol.json", p)
    rng = np.random.default_rng(123)
    dates = pd.date_range("2000-01-05", periods=21, freq="W-WED")
    x = pd.DataFrame(rng.normal(size=(21, 2)), index=dates, columns=p["predictors"])
    truth_beta = np.array([[1.0, -0.3], [0.7, 0.2], [-0.4, 0.7], [-0.8, -0.1]])
    y = pd.DataFrame(
        2 + x.to_numpy() @ truth_beta.T + 0.2 * rng.normal(size=(21, 4)),
        index=dates,
        columns=["S01_BM01", "S01_BM02", "S02_BM01", "S02_BM02"],
    )
    # Increase the short final block's loss to make equal-window weighting wrong.
    y.iloc[-1] += 1.5
    x.to_csv(panel / "panel/X_weekly.csv")
    y.to_csv(panel / "panel/Y_weekly.csv")
    x, y = R.frame(panel / "panel/X_weekly.csv"), R.frame(panel / "panel/Y_weekly.csv")
    pd.DataFrame(
        {
            "portfolio": y.columns,
            "original_label": y.columns,
            "size_decile": [1, 1, 2, 2],
            "bm_decile": [1, 2, 1, 2],
        }
    ).to_csv(panel / "panel/portfolio_labels.csv", index=False)
    write_json(
        panel / "manifest.json",
        {
            "files_sha256": {
                name: sha256(panel / name)
                for name in (
                    "protocol.json",
                    "panel/X_weekly.csv",
                    "panel/Y_weekly.csv",
                    "panel/portfolio_labels.csv",
                )
            }
        },
    )
    (fit / "source/synthetic_fixture.txt").write_text(
        "Known synthetic coefficients, no fitted data.\n"
    )
    binding = {
        "protocol_sha256": sha256(panel / "protocol.json"),
        "panel_manifest_sha256": sha256(panel / "manifest.json"),
        "source_sha256": {"synthetic_fixture.txt": sha256(fit / "source/synthetic_fixture.txt")},
        "defaults_observed": p["package_defaults"],
    }
    identity = R.digest(binding)
    jobs, audits = [], []
    for o in (12, 16, 20):
        for t in (8, 12):
            folder = fit / f"origin_{o}_T{t}"
            folder.mkdir()
            end = min(o + 4, len(x))
            xt, yt = x.iloc[o - t : o].to_numpy(), y.iloc[o - t : o].to_numpy()
            xm, xs, ym, ys = xt.mean(0), xt.std(0), yt.mean(0), yt.std(0)
            pre = {
                "x_mean": xm.tolist(),
                "x_sd": xs.tolist(),
                "y_mean": ym.tolist(),
                "y_sd": ys.tolist(),
            }
            arrays = {"actual": y.iloc[o:end].to_numpy(), "training_mean": ym, "training_sd": ys}
            selections = {}
            for j, method in enumerate(p["methods"] + R.MATCHED):
                beta = truth_beta * (0.75 + 0.02 * j)
                grouped = method.startswith(("hcgl", "fcgl", "cooperative", "matched"))
                labels = np.array([0, 0, 1, 1]) if grouped else np.arange(4)
                gate = np.full((4, 2), np.nan)
                if method in R.GATED:
                    gate = np.sign(beta)
                    gate[0, 1] = 0
                    beta[0, 1] = 0
                    if "unrestricted" in method:
                        gate[1, 0] = np.nan
                beta[0, 0] = 5e-7 * ys[0] / xs[0]
                alpha = ym - beta @ xm
                prediction = x.iloc[o:end].to_numpy() @ beta.T + alpha
                arrays.update(
                    {
                        method + "__beta": beta,
                        method + "__gates": gate,
                        method + "__prediction": prediction,
                        method + "__weekly_standardized_loss": np.mean(
                            np.square((y.iloc[o:end].to_numpy() - prediction) / ys), axis=1
                        ),
                    }
                )
                if method in p["methods"]:
                    arrays.update(
                        {
                            method + "__intercept": alpha,
                            method + "__labels": labels,
                            method + "__beta_standardized": beta * xs / ys[:, None],
                        }
                    )
                    candidate = {"method": method}
                    if method == "ridge":
                        candidate["ridge_lambda"] = 0.1
                    elif method != "ols":
                        candidate.update(
                            fraction=0.3,
                            partition=0.5 if grouped else None,
                            tau=2 if method in R.GATED else None,
                        )
                    selections[method] = {
                        "candidate": candidate,
                        "extension": {"fraction": False, "partition": False, "tau": False},
                        "lambda": 0.2 if method != "ols" else 0.0,
                        "lambda_max": 2 / 3,
                    }
            np.savez_compressed(folder / "outer_arrays.npz", **arrays)
            write_json(folder / "selections.json", selections)
            write_json(folder / "outer_preprocessing.json", pre)
            write_json(folder / "training_diagnostics.json", {})
            files = {f.name: sha256(f) for f in folder.iterdir() if f.is_file()}
            write_json(
                folder / "window_receipt.json",
                {
                    "status": "complete",
                    "identity": identity,
                    "test_dates": x.index[o:end].strftime("%Y-%m-%d").tolist(),
                    "files_sha256": files,
                },
            )
            jobs.append({"origin": o, "lookback": t, "folder": folder.name})
            audits.append(
                {
                    "origin": o,
                    "lookback": t,
                    "window_receipt_sha256": sha256(folder / "window_receipt.json"),
                }
            )
    write_json(fit / "run_manifest.json", {"identity": identity, "binding": binding, "jobs": jobs})
    write_json(
        fit / "full_validation.json", {"status": "complete", "identity": identity, "audits": audits}
    )
    write_json(
        fit / "full_run_receipt.json",
        {
            "status": "complete",
            "identity": identity,
            "validation_sha256": sha256(fit / "full_validation.json"),
        },
    )
    return panel, fit


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.panel, cls.fit = synthetic_run(cls.root)
        cls.report = cls.root / "report"
        # Importing these only in a test permits an explicit no-refit assertion.
        from papers.sign_pooling_2026.replication import french_estimators as F

        with patch.object(F, "fit_path", side_effect=AssertionError("Unexpected refit")):
            R.run(cls.panel, cls.fit, cls.report)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def copy_report(self):
        destination = self.root / self._testMethodName
        shutil.copytree(self.report, destination)
        return destination

    def rehash(self, output, filename):
        manifest = R.read_json(output / "report_manifest.json")
        manifest["files_sha256"][filename] = sha256(output / filename)
        write_json(output / "report_manifest.json", manifest)

    def test_independent_verifier_checks_all_tables_and_printed_values(self):
        result = V.verify(self.panel, self.fit, self.report)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["windows"], 6)
        self.assertGreater(result["numeric_values_verified"], 10000)

    def test_terminal_week_uses_observation_weights_and_training_mean_baseline(self):
        table = pd.read_csv(self.report / "metrics.csv")
        blocks = pd.read_csv(self.report / "block_losses.csv")
        rows = blocks.query("lookback == 8 and method == 'ols'")
        expected = rows.standardized_loss_sum.sum() / rows.weeks.sum()
        wrong = (rows.standardized_loss_sum / rows.weeks).mean()
        actual = table.query("lookback == 8 and method == 'ols'").iloc[0]
        self.assertAlmostEqual(actual.standardized_mse, expected, places=12)
        self.assertGreater(abs(expected - wrong), 0.01)

    def test_stationary_indices_match_scalar_transition_reference(self):
        actual = R.stationary_indices(7, 20, 3, 20)
        rng = np.random.default_rng(20)
        starts, u = rng.integers(0, 7, size=(20, 7)), rng.random((20, 7))
        expected = starts.copy()
        for b in range(20):
            for k in range(1, 7):
                if u[b, k] >= 1 / 3:
                    expected[b, k] = (expected[b, k - 1] + 1) % 7
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(R.stationary_indices(7, 20, 1, 20), starts)

    def test_ari_against_pair_enumeration_and_external_reference(self):
        for a, b in (([0, 0, 1, 1], [0, 1, 0, 1]), ([0, 0, 0], [1, 1, 1]), ([0, 1, 2], [3, 4, 5])):
            self.assertAlmostEqual(R.adjusted_rand(a, b), adjusted_rand_score(a, b), places=14)
            self.assertAlmostEqual(V.pair_ari(a, b), adjusted_rand_score(a, b), places=14)

    def test_gate_exclusion_unrestricted_and_near_zero_are_distinct(self):
        gate = pd.read_csv(self.report / "gate_coverage.csv")
        release = gate.query("method == 'hcgl_unrestricted' and factor == 'SMB'")
        self.assertTrue((release.unrestricted_failed == 1).all())
        support = pd.read_csv(self.report / "support_levels.csv")
        self.assertTrue((support.active < support.cells).all())
        categories = pd.read_csv(self.report / "gate_category_summary.csv")
        row = categories.query(
            "lookback == 8 and method == 'hcgl_exclude' and factor == 'SMB' "
            "and axis == 'size_decile' and category == 1"
        ).iloc[0]
        self.assertEqual(row.cells, 6)

    def test_inapplicable_boundaries_and_non_singleton_diagnostics_are_undefined(self):
        boundary = pd.read_csv(self.report / "boundaries.csv")
        self.assertTrue(boundary.query("method == 'ols'").fraction_zero.isna().all())
        coherence = pd.read_csv(self.report / "coherence.csv")
        self.assertTrue(coherence.query("method == 'ols'").train_non_single_agreement.isna().all())
        self.assertTrue((coherence.query("method == 'ols'").train_agreement == 1).all())

    def test_corrupted_metric_is_rejected_even_with_a_matching_artifact_hash(self):
        output = self.copy_report()
        path = output / "metrics.csv"
        table = pd.read_csv(path)
        table.loc[0, "standardized_mse"] += 0.01
        table.to_csv(path, index=False)
        self.rehash(output, path.name)
        with self.assertRaisesRegex(ValueError, "metrics"):
            V.verify(self.panel, self.fit, output)

    def test_corrupted_interval_and_printed_number_are_rejected(self):
        output = self.copy_report()
        path = output / "contrasts.csv"
        table = pd.read_csv(path)
        table.loc[0, "lower"] -= 0.1
        table.to_csv(path, index=False)
        self.rehash(output, path.name)
        with self.assertRaisesRegex(ValueError, "contrasts"):
            V.verify(self.panel, self.fit, output)

    def test_wrong_printed_value_is_rejected_even_after_rehashing(self):
        output = self.copy_report()
        path = output / "table_prediction_T8.tex"
        text = path.read_text()
        line = next(v for v in text.splitlines() if v.startswith("Lasso &"))
        cells = line.split(" & ")
        cells[1] = "123.456789"
        path.write_text(text.replace(line, " & ".join(cells)))
        self.rehash(output, path.name)
        with self.assertRaisesRegex(ValueError, "printed metrics"):
            V.verify(self.panel, self.fit, output)

    def test_missing_table_row_is_rejected(self):
        output = self.copy_report()
        path = output / "gate_coverage.csv"
        pd.read_csv(path).iloc[1:].to_csv(path, index=False)
        self.rehash(output, path.name)
        with self.assertRaisesRegex(ValueError, "Missing gate_coverage"):
            V.verify(self.panel, self.fit, output)

    def test_changed_panel_labels_are_rejected_without_fallback(self):
        panel = self.root / (self._testMethodName + "_panel")
        shutil.copytree(self.panel, panel)
        path = panel / "panel/portfolio_labels.csv"
        labels = pd.read_csv(path)
        labels.loc[0, "size_decile"] = 10
        labels.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Changed panel"):
            R.load_run(panel, self.fit)
        with self.assertRaisesRegex(ValueError, "Changed raw"):
            V.verify(panel, self.fit, self.report)

    def test_intervals_regenerate_exactly_from_saved_block_data(self):
        p = R.read_json(self.panel / "protocol.json")
        with np.load(self.report / "report_data.npz", allow_pickle=False) as z:
            bundle = {key: z[key].copy() for key in z.files}
        a, _, first = R.bootstrap_tables(p, bundle)
        b, _, second = R.bootstrap_tables(p, bundle)
        pd.testing.assert_frame_equal(a, b)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        for row in a.query("metric == 'srmse'").itertuples():
            tag = f"T{row.lookback}_L{row.mean_blocks:g}__mse"
            estimates = first[tag]
            methods = p["methods"] + R.MATCHED
            delta = np.sqrt(estimates[:, methods.index(row.method_a)]) - np.sqrt(
                estimates[:, methods.index(row.method_b)]
            )
            self.assertAlmostEqual(row.lower, np.quantile(delta, 0.025), places=14)

    def test_empty_support_jaccard_and_training_mean_r2(self):
        data = R.load_run(self.panel, self.fit)
        for w in data["windows"]:
            a, pre = w["arrays"], w["pre"]
            a["lasso__beta"][:] = 0
            a["lasso__beta_standardized"][:] = 0
            a["lasso__intercept"] = np.array(pre["y_mean"])
            a["lasso__prediction"] = np.tile(pre["y_mean"], (len(a["actual"]), 1))
            a["lasso__weekly_standardized_loss"] = np.mean(
                np.square((a["actual"] - a["lasso__prediction"]) / pre["y_sd"]), axis=1
            )
        tables, _ = R.build_tables(data)
        self.assertTrue((tables["support_stability"].query("method == 'lasso'").jaccard == 1).all())
        np.testing.assert_allclose(
            tables["metrics"].query("method == 'lasso'").pooled_r2, 0, atol=1e-14
        )


if __name__ == "__main__":
    unittest.main()
