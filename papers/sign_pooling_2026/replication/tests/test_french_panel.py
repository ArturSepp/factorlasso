"""Offline French intake contracts, chronological inventory and independent returns."""

from __future__ import annotations

import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

import numpy as np
import pandas as pd

from papers.sign_pooling_2026.replication import french_panel as panel
from papers.sign_pooling_2026.replication import french_protocol as protocol


def labels() -> list[str]:
    """Use the official extreme-label aliases as well as numeric deciles."""
    result = []
    for s in range(1, 11):
        for b in range(1, 11):
            size = "SMALL" if s == 1 else "BIG" if s == 10 else f"ME{s}"
            bm = "LoBM" if b == 1 else "HiBM" if b == 10 else f"BM{b}"
            result.append(f"{size} {bm}")
    return result


def source_bytes(dates: pd.DatetimeIndex, *, sentinel: bool = False) -> bytes:
    """Build two sections with deliberately different returns on repeated dates."""
    out = io.StringIO(newline="")
    out.write("Created by test fixture, not research data\nMissing data are -99.99 or -999.\n\n")
    out.write("  Average Value Weighted Returns -- Daily\n")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow([""] + labels())
    for k, when in enumerate(dates):
        values = [1.0 + i / 100 + k / 1000 for i in range(100)]
        if sentinel and k == 0:
            values[:2] = [-99.99, -999]
        writer.writerow([when.strftime("%Y%m%d")] + values)
    out.write("\n  Average Equal Weighted Returns -- Daily\n")
    writer.writerow([""] + labels())
    writer.writerow([dates[0].strftime("%Y%m%d")] + [99.0]*100)
    return out.getvalue().encode()


def archive_bytes(content: bytes) -> bytes:
    """Make a tiny compressed source member, without network access."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("portfolios.CSV", content)
    return out.getvalue()


class ProtocolTests(unittest.TestCase):
    """Check candidate accounting and the same absolute evaluation calendars."""

    def test_candidate_counts_and_ids(self):
        p = protocol.study_protocol()
        base = protocol.candidate_inventory(p)
        extended = protocol.candidate_inventory(p, extended=True)
        self.assertEqual({m: item["count"] for m, item in base.items()}, {
            "lasso": 8, "singleton_exclude": 48, "hcgl_unconstrained": 48,
            "hcgl_singleton_exclude": 288, "hcgl_exclude": 288,
            "hcgl_unrestricted": 288, "fcgl_unconstrained": 48, "fcgl_exclude": 288,
            "ols": 1, "ridge": 10, "cooperative": 48, "unilasso": 8,
        })
        self.assertEqual(sum(v["count"] for v in base.values()), 1371)
        self.assertEqual(sum(v["count"] for v in extended.values()), 3586)
        for mode in (base, extended):
            ids = [v for item in mode.values() for v in item["candidate_ids"]]
            self.assertEqual(len(ids), len(set(ids)))
        json.dumps(p, allow_nan=False)
        self.assertEqual(p["package_defaults"], {
            "auto_sign_constraints": False, "auto_sign_threshold_t": .75,
            "auto_sign_variance": "independent"})

    def test_chronology_common_validation_and_terminal_block(self):
        p = protocol.study_protocol()
        dates = pd.date_range("2006-08-09", "2026-06-24", freq="W-WED").strftime(
            "%Y-%m-%d").tolist()
        result = protocol.calendar_inventory(dates, p)["rows"]
        self.assertEqual(len(result), 144)
        primary = [r for r in result if r["lookback"] == 52]
        self.assertEqual(len(primary), 72)
        self.assertEqual(sum(len(r["test_dates"]) for r in primary), 934)
        self.assertEqual(primary[0]["test_dates"][0], "2008-08-06")
        self.assertEqual(len(primary[-1]["test_dates"]), 11)
        for a, b in zip(result[::2], result[1::2]):
            self.assertEqual(a["test_dates"], b["test_dates"])
            for i, j in zip(a["inner"], b["inner"]):
                self.assertEqual(i["validation_dates"], j["validation_dates"])
                self.assertLess(i["train_slice"][0], i["train_slice"][1])
                self.assertEqual(i["train_slice"][1], i["validation_slice"][0])
                self.assertLessEqual(i["validation_slice"][1], a["origin"])

    def test_output_roots_reject_onedrive_and_relative(self):
        for target in (Path("relative"), protocol.REPOSITORY / "new-output",
                       Path(tempfile.gettempdir()) / "OneDrive-other" / "run"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                protocol.external_path(target)


class ParserTests(unittest.TestCase):
    """Protect section boundaries, portfolio identities, missing data and units."""

    def setUp(self):
        self.dates = pd.date_range("2020-01-06", periods=3, freq="B")

    def test_selects_only_value_weighted_section_and_percent_units(self):
        frame, mapping, meta = panel.parse_portfolios(source_bytes(self.dates))
        self.assertEqual(frame.shape, (3, 100))
        self.assertEqual(frame.iloc[0, 0], .01)
        self.assertEqual(frame.iloc[0, -1], .0199)
        self.assertEqual(mapping.iloc[0]["portfolio"], "S01_BM01")
        self.assertEqual(mapping.iloc[-1]["portfolio"], "S10_BM10")
        self.assertIn("Equal Weighted", meta["boundary_text"])

    def test_sentinels_are_missing_before_percent_conversion(self):
        frame, _, meta = panel.parse_portfolios(source_bytes(self.dates, sentinel=True))
        self.assertTrue(frame.iloc[0, :2].isna().all())
        self.assertEqual(meta["sentinel_cells_full_history"], 2)

    def test_duplicate_labels_dates_and_wrong_section_fail(self):
        content = source_bytes(self.dates)
        bad_cases = [content.replace(b"SMALL BM2", b"SMALL LoBM", 1),
                     content.replace(b"20200107,", b"20200106,", 1),
                     content.replace(b"Average Value Weighted", b"Another Value Weighted", 1)]
        for bad in bad_cases:
            with self.subTest(bad=bad[:45]), self.assertRaises(ValueError):
                panel.parse_portfolios(bad)

    def test_invalid_field_count_nonfinite_and_blank_interruption_fail(self):
        content = source_bytes(self.dates)
        for bad in (content.replace(b"20200107,1.001,", b"20200107,", 1),
                    content.replace(b"20200107,1.001,", b"20200107,nan,", 1),
                    content.replace(b"20200107,", b"\n20200107,", 1)):
            with self.assertRaises(ValueError):
                panel.parse_portfolios(bad)

    def test_calendar_gaps_and_missing_required_cells_fail(self):
        frame, _, _ = panel.parse_portfolios(source_bytes(self.dates))
        f = pd.DataFrame(.001, index=self.dates, columns=panel.FACTOR_COLUMNS)
        with self.assertRaisesRegex(ValueError, "calendar mismatch"):
            panel.align_daily(frame.drop(index=frame.index[1]), f)
        frame.iloc[1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing/nonfinite"):
            panel.align_daily(frame, f)

    def test_nonpositive_log_arguments_fail_without_clipping(self):
        f = np.full((2, 7), .001)
        p = np.full((2, 100), .01)
        for which in ("portfolio", "cash", "market", "style"):
            pp, ff = p.copy(), f.copy()
            if which == "portfolio":
                pp[0, 0] = -1
            elif which == "cash":
                ff[0, -1] = -1
            elif which == "market":
                ff[0, 0] = -1.01
            else:
                ff[0, 1] = -1
            with self.subTest(which=which), self.assertRaises(ValueError):
                panel.positive_log_arguments(pp, ff)

    def test_only_approved_missing_portfolios_are_excluded(self):
        frame, _, _ = panel.parse_portfolios(source_bytes(self.dates))
        f = pd.DataFrame(.001, index=self.dates, columns=panel.FACTOR_COLUMNS)
        frame.loc[self.dates[0], "S10_BM10"] = np.nan
        frame.loc[self.dates[1], "S09_BM10"] = np.nan
        selected, audit = panel.select_responses(frame, f, protocol.study_protocol())
        self.assertEqual(selected.shape, (3, 98))
        self.assertEqual(audit["missing_source_cells_analysis_period"], 2)
        self.assertEqual(audit["retained_missing_cells"], 0)
        self.assertTrue(np.isnan(frame.loc[self.dates[0], "S10_BM10"]))
        frame.loc[self.dates[1], "S02_BM03"] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing/nonfinite"):
            panel.select_responses(frame, f, protocol.study_protocol())


class ReturnAndArtifactTests(unittest.TestCase):
    """Use independent gross wealth calculations and a complete offline toy run."""

    def inputs(self):
        # A missing Tuesday is a legitimate holiday calendar shared by both inputs.
        dates = pd.bdate_range("2020-01-06", "2020-02-18").drop(pd.Timestamp("2020-01-21"))
        content = source_bytes(dates)
        y, _, _ = panel.parse_portfolios(content)
        f = pd.DataFrame({name: .001 + k*.0001 + np.arange(len(dates))*.00001
                          for k, name in enumerate(panel.FACTOR_COLUMNS)}, index=dates)
        f.index.name = "date"
        weekly = np.log1p(f.iloc[:, :6]).resample("W-WED").sum()
        weekly = weekly.loc[:"2020-02-12"]
        weekly.iloc[0] = np.nan
        return content, y, f, weekly

    def test_log_relative_wealth_market_styles_and_holiday_coverage(self):
        _, y, f, reference = self.inputs()
        xw, yw, coverage = panel.weekly_panel(y, f, reference.iloc[1:])
        rx, ry = panel.independent_weekly(y, f, reference.index[1:])
        np.testing.assert_allclose(xw, rx, rtol=0, atol=2e-15)
        np.testing.assert_allclose(yw, ry, rtol=0, atol=2e-15)
        self.assertIn(4, coverage["trading_days"].tolist())
        self.assertNotEqual(xw.iloc[0, 0], reference.iloc[1, 0])
        mask = (f.index > pd.Timestamp("2020-01-08")) & (f.index <= pd.Timestamp("2020-01-15"))
        expected = np.log(np.prod(1+y.loc[mask].iloc[:, 0])/np.prod(1+f.loc[mask, "RF_3M"]))
        self.assertAlmostEqual(yw.iloc[0, 0], expected, places=14)

    def toy_protocol(self):
        p = copy.deepcopy(protocol.study_protocol())
        p["responses"] = 100
        p["response_selection"]["exclude"] = []
        p["calendar"].update({
            "daily_start": "2020-01-06", "daily_end": "2020-02-18", "daily_rows": 31,
            "weekly_start": "2020-01-15", "weekly_end": "2020-02-12", "weekly_rows": 5,
            "lookbacks": [2, 3], "outer_burn_in": 3, "outer_horizon": 2,
            "origins": 1, "test_weeks": 2,
            "inner_train_ends": {"2": [1], "3": [2]},
            "inner_validation_ends": {"2": [2], "3": [3]},
        })
        return p

    def test_frozen_end_to_end_offline_and_corruption_rejection(self):
        content, _, factors, weekly = self.inputs()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            supplied = folder / "supplied"
            supplied.mkdir()
            factors.to_csv(supplied / "factor_daily.csv", float_format="%.17g")
            weekly.to_csv(supplied / "factor_weekly.csv", float_format="%.17g")
            (supplied / "factor_producer.py").write_text("# toy fixture\n")
            inputs = {name: supplied / name for name in protocol.INPUT_HASHES}
            hashes = {name: protocol.sha256(path) for name, path in inputs.items()}
            p = self.toy_protocol()
            p["input_hashes"] = hashes
            root = folder / "run"
            with patch.object(panel, "INPUT_HASHES", hashes), patch.object(panel, "study_protocol",
                    return_value=p):
                panel.freeze(root, inputs)
                with self.assertRaises(FileExistsError):
                    panel.freeze(root, inputs)
                response = MagicMock()
                response.__enter__.return_value = response
                response.geturl.return_value = panel.PORTFOLIO_URL
                response.read.return_value = archive_bytes(content)
                response.headers = {"Content-Type": "application/zip"}
                response.status = 200
                with patch.object(panel.urllib.request, "urlopen", return_value=response) as fetch:
                    panel.download(root)
                    self.assertEqual(fetch.call_count, 1)
                result = panel.build(root)
                self.assertEqual(result["weekly_rows"], 5)
                with patch.object(panel.urllib.request, "urlopen",
                                  side_effect=AssertionError("network")):
                    self.assertEqual(panel.verify(root)["verified_weekly_values"], 530)
                with self.assertRaises(FileExistsError):
                    panel.build(root)
                adopted = folder / "amended-run"
                panel.freeze(adopted, inputs)
                with patch.object(panel.urllib.request, "urlopen",
                                  side_effect=AssertionError("unexpected second download")):
                    panel.download(adopted, from_run=root)
                self.assertEqual((adopted / "raw/portfolios.zip").read_bytes(),
                                 (root / "raw/portfolios.zip").read_bytes())
                self.assertEqual((adopted / "download_receipt.json").read_bytes(),
                                 (root / "download_receipt.json").read_bytes())
                target = root / "panel/Y_weekly.csv"
                target.write_bytes(target.read_bytes().replace(b"2020-01-15", b"2020-01-16", 1))
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    panel.verify(root)
                # A modified protocol is rejected even if its JSON still parses.
                target = root / "protocol.json"
                saved = json.loads(target.read_text())
                saved["gate"]["variance_estimator"] = "independent"
                target.write_bytes(protocol.json_bytes(saved))
                with self.assertRaisesRegex(ValueError, "Frozen file changed"):
                    panel.verify(root)

    def test_zip_must_contain_exactly_one_csv(self):
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as z:
            z.writestr("one.csv", b"a")
            z.writestr("two.csv", b"b")
        with self.assertRaises(ValueError):
            panel.zip_member(out.getvalue())


if __name__ == "__main__":
    unittest.main()
