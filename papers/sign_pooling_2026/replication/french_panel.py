"""Freeze and audit 98 complete French portfolios from the 100-portfolio source.

Sources are Kenneth French's daily value-weighted 10x10 portfolios and the
explicit supplied FF6b snapshot. Responses and market are log wealth relative
to the supplied DGS3MO cash proxy; style predictors are transformed long-short
spreads. Mixed release vintages and redistribution rights remain unverified.

Run as a repository-root module or as a direct script. All generated files must
be outside the checkout and OneDrive. ``--verify`` uses frozen files only and
independently reconstructs weekly returns by products of daily gross returns.

References
----------
Kenneth R. French Data Library, 100 Portfolios Formed on Size and Book-to-Market:
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_100_port_sz.html
The supplied cash proxy is not the official French risk-free return series.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import io
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd

if __package__:
    from .french_protocol import (INPUT_HASHES, PORTFOLIO_URL, REPOSITORY,
                                  calendar_inventory, candidate_inventory, external_path,
                                  json_bytes, sha256, study_protocol)
else:
    from french_protocol import (INPUT_HASHES, PORTFOLIO_URL, REPOSITORY,
                                 calendar_inventory, candidate_inventory, external_path,
                                 json_bytes, sha256, study_protocol)

FACTOR_COLUMNS = ["MKT_RF_3M", "SMB", "HML", "RMW", "CMA", "MOM", "RF_3M"]
SECTION = "average value weighted returns -- daily"
MAX_DOWNLOAD = 64 * 1024 * 1024
MAX_MEMBER = 128 * 1024 * 1024


def utc_now() -> str:
    """Return an unambiguous retrieval/execution timestamp."""
    return datetime.now(timezone.utc).isoformat()


def write_new(path: Path, content: bytes) -> None:
    """Preserve existing artifacts, including incomplete previous attempts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def load_protocol(root: Path) -> dict:
    """Authenticate the original freeze and supplied input copies."""
    receipt = json.loads((root / "freeze_receipt.json").read_text(encoding="utf-8"))
    for name, expected in receipt["files_sha256"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"Frozen file changed: {name}")
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    if protocol != study_protocol():
        raise ValueError("Frozen protocol differs from this code; use its source snapshot")
    return protocol


def freeze(root: Path, inputs: dict[str, Path]) -> dict:
    """Write immutable protocol, inputs, source snapshot, environment and inventory."""
    root = external_path(root)
    if (root / "protocol.json").exists() or (root / "freeze_receipt.json").exists():
        raise FileExistsError("Protocol already frozen; use a new run directory")
    for name, path in inputs.items():
        if sha256(path) != INPUT_HASHES[name]:
            raise ValueError(f"Supplied input hash mismatch: {name}")
    if set(inputs) != set(INPUT_HASHES):
        raise ValueError("All three explicitly supplied input paths are required")
    # Introspection only: this never fits or changes the estimator.
    import factorlasso
    from factorlasso import LassoModel

    protocol = study_protocol()
    defaults = {name: inspect.signature(LassoModel).parameters[name].default
                for name in protocol["package_defaults"]}
    if defaults != protocol["package_defaults"]:
        raise ValueError(f"Unexpected installed estimator defaults: {defaults}")
    imported = Path(factorlasso.__file__).resolve()
    if REPOSITORY / "src" not in imported.parents:
        raise ValueError(f"Expected this checkout's src layout, imported {imported}")
    root.mkdir(parents=True, exist_ok=True)
    for name, path in inputs.items():
        write_new(root / "inputs" / name, path.read_bytes())
    weekly = read_factors(root / "inputs", protocol)[1]
    calendars = calendar_inventory(weekly.index.strftime("%Y-%m-%d").tolist(), protocol)
    base = candidate_inventory(protocol)
    extended = candidate_inventory(protocol, extended=True)
    base_count = sum(item["count"] for item in base.values())
    max_count = sum(item["count"] for item in extended.values())
    n_outer = protocol["calendar"]["origins"] * len(protocol["calendar"]["lookbacks"])
    counts = {
        "scope": "logical candidate evaluations; no fitting or response-dependent deduplication",
        "per_method_base": {m: v["count"] for m, v in base.items()},
        "per_method_extended": {m: v["count"] for m, v in extended.items()},
        "base_per_inner_split": base_count, "extended_per_inner_split_upper_bound": max_count,
        "outer_method_fits": n_outer * len(base), "matched_diagnostic_fits": n_outer * 3,
        "base_inner_evaluations_both_lookbacks": n_outer * 3 * base_count,
        "fully_extended_inner_upper_bound": n_outer * 3 * max_count,
        "note": "Exact deduplication and analytic endpoints can reduce actual solver calls",
    }
    versions = {}
    for name in ("numpy", "pandas", "scipy", "cvxpy", "clarabel", "factorlasso"):
        versions[name] = importlib.metadata.version(name)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY,
                              capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "status", "--short"], cwd=REPOSITORY,
                            capture_output=True, text=True, check=False)
    source_paths = sorted((REPOSITORY / "src/factorlasso").rglob("*.py"))
    source_paths += [REPOSITORY / p for p in (
        "pyproject.toml", "LICENSE", "NOTICE",
        "papers/sign_pooling_2026/replication/french_protocol.py",
        "papers/sign_pooling_2026/replication/french_panel.py",
        "papers/sign_pooling_2026/replication/tests/test_french_panel.py",
        "papers/sign_pooling_2026/replication/estimator_benchmark.py",
        "papers/sign_pooling_2026/replication/eqtl_validation.py",
    ) if (REPOSITORY / p).is_file()]
    source_hashes = {p.relative_to(REPOSITORY).as_posix(): sha256(p) for p in source_paths}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths:
            info = zipfile.ZipInfo(path.relative_to(REPOSITORY).as_posix(), (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    write_new(root / "source_snapshot.zip", buffer.getvalue())
    environment = {
        "created_utc": utc_now(), "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "distribution_metadata_versions": versions,
        "factorlasso_imported_from": str(imported), "defaults_observed": defaults,
        "git_revision": revision.stdout.strip(), "git_status_at_freeze": status.stdout,
        "source_sha256": source_hashes,
        "metadata_note": "Installed metadata may lag editable checkout; source hashes bind code",
        "supplied_inputs": {name: {"path": str(p.resolve()), "sha256": sha256(p)}
                            for name, p in inputs.items()},
        "no_estimator_fitted": True,
    }
    objects = {"protocol.json": protocol, "environment.json": environment,
               "candidate_inventory.json": {"base": base, "all_refinements_upper_bound": extended},
               "candidate_counts.json": counts, "calendar.json": calendars}
    for name, obj in objects.items():
        write_new(root / name, json_bytes(obj))
    files = ["protocol.json", "environment.json", "source_snapshot.zip", "candidate_inventory.json",
             "candidate_counts.json", "calendar.json"] + ["inputs/" + n for n in inputs]
    receipt = {"created_utc": utc_now(), "protocol_frozen_before_model_fitting": True,
               "files_sha256": {name: sha256(root / name) for name in files}}
    write_new(root / "freeze_receipt.json", json_bytes(receipt))
    return counts


def download(root: Path, *, from_run: Path | None = None) -> dict:
    """Retain original HTTPS response bytes and metadata; never replace a download."""
    root = external_path(root)
    load_protocol(root)
    if (root / "download_receipt.json").exists() or (root / "raw/portfolios.zip").exists():
        raise FileExistsError("Download already exists; use the frozen bytes")
    if from_run is not None:
        # Preserve the original HTTP receipt and timestamps; this is a byte copy,
        # not a second retrieval or a claim about the source's historical vintage.
        previous = json.loads((from_run / "freeze_receipt.json").read_text(encoding="utf-8"))
        for name, expected in previous["files_sha256"].items():
            if sha256(from_run / name) != expected:
                raise ValueError(f"Previous frozen file changed: {name}")
        receipt_bytes = (from_run / "download_receipt.json").read_bytes()
        receipt = json.loads(receipt_bytes)
        content = (from_run / "raw/portfolios.zip").read_bytes()
        if hashlib.sha256(content).hexdigest() != receipt["archive_sha256"]:
            raise ValueError("Previous downloaded ZIP changed")
        member, member_bytes = zip_member(content)
        if (receipt["url"] != PORTFOLIO_URL or member != receipt["member"] or
                hashlib.sha256(member_bytes).hexdigest() != receipt["member_sha256"]):
            raise ValueError("Previous download identity mismatch")
        write_new(root / "raw/portfolios.zip", content)
        write_new(root / "download_receipt.json", receipt_bytes)
        write_new(root / "download_adoption.json", json_bytes({
            "copied_utc": utc_now(), "from_run": str(from_run.resolve()),
            "original_receipt_sha256": sha256(from_run / "download_receipt.json"),
            "original_protocol_sha256": sha256(from_run / "protocol.json"),
            "note": "Original HTTPS download reused unchanged; no new retrieval",
        }))
        return receipt
    started = utc_now()
    request = urllib.request.Request(PORTFOLIO_URL, headers={
        "User-Agent": "FactorLasso-paper-replication/1.0 (research data intake)"})
    with urllib.request.urlopen(request, timeout=120) as response:
        final_url = response.geturl()
        if not final_url.startswith("https://mba.tuck.dartmouth.edu/"):
            raise ValueError(f"Unexpected download redirect: {final_url}")
        content = response.read(MAX_DOWNLOAD + 1)
        if len(content) > MAX_DOWNLOAD:
            raise ValueError("Download exceeds the declared size limit")
        headers = dict(response.headers.items())
        http_status = response.status
    # Validate a bounded single CSV member before recording a successful transfer.
    member, member_bytes = zip_member(content)
    receipt = {"started_utc": started, "completed_utc": utc_now(), "url": PORTFOLIO_URL,
               "final_url": final_url, "http_status": http_status, "response_headers": headers,
               "archive_sha256": hashlib.sha256(content).hexdigest(), "archive_bytes": len(content),
               "member": member, "member_sha256": hashlib.sha256(member_bytes).hexdigest(),
               "member_bytes": len(member_bytes), "original_retrieval_vintage": "not inferred",
               "distribution_permission": "not established; local research copy only"}
    write_new(root / "raw/portfolios.zip", content)
    write_new(root / "download_receipt.json", json_bytes(receipt))
    return receipt


def zip_member(content: bytes) -> tuple[str, bytes]:
    """Read one bounded CSV member without extracting archive-controlled paths."""
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        members = [item for item in archive.infolist() if item.filename.lower().endswith(".csv")]
        if len(members) != 1 or members[0].file_size > MAX_MEMBER:
            raise ValueError("Expected exactly one bounded CSV member")
        member = members[0]
        return member.filename, archive.read(member)


def portfolio_mapping(columns: list[str]) -> pd.DataFrame:
    """Validate all 100 size/BM intersections, preserving the official labels."""
    rows = []
    for original in columns:
        label = " ".join(original.strip().upper().split())
        match = re.fullmatch(r"(SMALL|BIG|ME\d{1,2}) (LOBM|HIBM|BM\d{1,2})", label)
        if match is None:
            raise ValueError(f"Unknown portfolio label: {original}")
        size, bm = match.groups()
        size_n = {"SMALL": 1, "BIG": 10}.get(size, int(size[2:]) if size.startswith("ME") else 0)
        bm_n = {"LOBM": 1, "HIBM": 10}.get(bm, int(bm[2:]) if bm.startswith("BM") else 0)
        rows.append({"portfolio": f"S{size_n:02d}_BM{bm_n:02d}", "original_label": original,
                     "size_decile": size_n, "bm_decile": bm_n})
    result = pd.DataFrame(rows)
    expected = {(s, b) for s in range(1, 11) for b in range(1, 11)}
    pairs = set(zip(result["size_decile"], result["bm_decile"]))
    if len(rows) != 100 or len(set(columns)) != 100 or pairs != expected:
        raise ValueError("Expected each of the 100 distinct size/BM intersections exactly once")
    return result


def parse_portfolios(content: bytes) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Parse only the value-weighted daily section; missing sentinels remain missing."""
    try:
        text, encoding = content.decode("utf-8-sig"), "utf-8-sig"
    except UnicodeDecodeError:
        text, encoding = content.decode("latin-1"), "latin-1"
    lines = text.splitlines()
    matches = [i for i, line in enumerate(lines) if " ".join(line.lower().split()) == SECTION]
    if len(matches) != 1:
        raise ValueError("Expected one explicitly titled value-weighted daily section")
    title = matches[0]
    header_line = title + 1
    while header_line < len(lines) and not lines[header_line].strip():
        header_line += 1
    if header_line == len(lines):
        raise ValueError("Truncated source section: missing column header")
    header = next(csv.reader([lines[header_line]]))
    if len(header) != 101 or header[0].strip().lower() not in ("", "date"):
        raise ValueError("Expected a date and 100 portfolio columns")
    mapping = portfolio_mapping([s.strip() for s in header[1:]])
    records, dates, sentinels = [], [], 0
    line_number = header_line + 1
    while line_number < len(lines):
        row = next(csv.reader([lines[line_number]]))
        if not row or not re.fullmatch(r"\d{8}", row[0].strip()):
            break
        if len(row) != 101:
            raise ValueError(f"Wrong field count at source line {line_number+1}")
        try:
            when = datetime.strptime(row[0].strip(), "%Y%m%d")
            values = np.asarray([float(value.strip()) for value in row[1:]], dtype=float)
        except ValueError as exc:
            raise ValueError(f"Invalid daily row at source line {line_number+1}") from exc
        missing = np.isin(values, [-99.99, -999.0])
        sentinels += int(missing.sum())
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite source value at source line {line_number+1}")
        values[missing] = np.nan
        records.append(values / 100.0)
        dates.append(when)
        line_number += 1
    if not records:
        raise ValueError("No daily rows in the selected section")
    index = pd.DatetimeIndex(dates, name="date")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Selected section dates must be unique and increasing")
    # An unexplained blank inside a section cannot hide later daily rows.
    following = line_number
    while following < len(lines) and not lines[following].strip():
        following += 1
    if following < len(lines) and re.match(r"\s*\d{8}\s*,", lines[following]):
        raise ValueError("Unexpected blank interruption within the daily section")
    if following < len(lines) and " ".join(lines[following].lower().split()) != (
            "average equal weighted returns -- daily"):
        raise ValueError("Unexpected daily section boundary")
    panel = pd.DataFrame(records, index=index, columns=mapping["portfolio"].tolist())
    metadata = {"encoding": encoding, "section": lines[title].strip(),
                "section_title_line": title+1, "column_header_line": header_line+1,
                "first_data_line": header_line+2, "last_data_line": line_number,
                "boundary_line": line_number+1, "boundary_text": lines[following].strip()
                if following < len(lines) else "EOF", "source_preamble": lines[:title],
                "source_rows": len(panel), "source_first": index[0].strftime("%Y-%m-%d"),
                "source_last": index[-1].strftime("%Y-%m-%d"),
                "sentinel_cells_full_history": sentinels, "units_conversion": "percent / 100"}
    return panel, mapping, metadata


def read_csv(path: Path) -> pd.DataFrame:
    """Read frozen numeric CSVs without losing a round-trip float digit."""
    frame = pd.read_csv(path, index_col=0, float_precision="round_trip")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index), name="date")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f"Nonunique or unordered dates: {path.name}")
    return frame


def read_factors(folder: Path, protocol: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check supplied factor bytes and the declared complete-week reference calendar."""
    for name, expected in protocol["input_hashes"].items():
        if sha256(folder / name) != expected:
            raise ValueError(f"Factor input changed: {name}")
    daily = read_csv(folder / "factor_daily.csv")
    reference = read_csv(folder / "factor_weekly.csv")
    c = protocol["calendar"]
    if list(daily.columns) != FACTOR_COLUMNS or len(daily) != c["daily_rows"]:
        raise ValueError("Unexpected daily factor schema")
    if not np.isfinite(daily.to_numpy()).all():
        raise ValueError("Missing/nonfinite daily factor")
    if list(reference.columns) != FACTOR_COLUMNS[:-1] or not reference.iloc[0].isna().all():
        raise ValueError("Unexpected weekly factor reference schema/initialization row")
    reference = reference.iloc[1:]
    expected = pd.date_range(c["weekly_start"], c["weekly_end"], freq="W-WED", name="date")
    if not reference.index.equals(expected) or len(reference) != c["weekly_rows"]:
        raise ValueError("Weekly reference calendar differs from protocol")
    if not np.isfinite(reference.to_numpy()).all():
        raise ValueError("Nonfinite complete-week factor reference")
    if daily.index[0].strftime("%Y-%m-%d") != c["daily_start"] or (
            daily.index[-1].strftime("%Y-%m-%d") != c["daily_end"]):
        raise ValueError("Daily factor endpoints differ from protocol")
    return daily, reference


def positive_log_arguments(portfolios: np.ndarray, factors: np.ndarray) -> None:
    """Fail on invalid gross returns before applying logarithms."""
    arrays = [1 + portfolios, 1 + factors[:, -1], 1 + factors[:, 0] + factors[:, -1],
              1 + factors[:, 1:6]]
    if any(not np.isfinite(a).all() or np.any(a <= 0) for a in arrays):
        raise ValueError("Every log argument must be finite and strictly positive")


def align_daily(portfolios: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Reject hidden date gaps and required missing cells; never repair returns."""
    selected = portfolios.loc[(portfolios.index >= factors.index[0]) &
                              (portfolios.index <= factors.index[-1])]
    if not selected.index.equals(factors.index):
        missing = factors.index.difference(selected.index).strftime("%Y-%m-%d").tolist()
        extra = selected.index.difference(factors.index).strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Daily calendar mismatch: missing={missing[:8]}, extra={extra[:8]}")
    if not np.isfinite(selected.to_numpy()).all():
        raise ValueError("Missing/nonfinite portfolio return in the required analysis period")
    positive_log_arguments(selected.to_numpy(), factors.to_numpy())
    return selected


def select_responses(portfolios: pd.DataFrame, factors: pd.DataFrame,
                     protocol: dict) -> tuple[pd.DataFrame, dict]:
    """Apply only the explicitly approved availability exclusions and retain an audit."""
    selection = protocol["response_selection"]
    period = portfolios.loc[(portfolios.index >= factors.index[0]) &
                            (portfolios.index <= factors.index[-1])]
    excluded = selection["exclude"]
    if len(portfolios.columns) != selection["source_portfolios"] or any(
            name not in portfolios.columns for name in excluded):
        raise ValueError("Source universe does not match the frozen response selection")
    missing = period.isna()
    missing_by_portfolio = {}
    for name in period.columns[missing.any()]:
        dates = period.index[missing[name]].strftime("%Y-%m-%d").tolist()
        missing_by_portfolio[name] = {"missing_days": len(dates),
                                      "first_missing": dates[0], "last_missing": dates[-1],
                                      "missing_dates": dates}
    retained = period.drop(columns=excluded)
    if retained.shape[1] != protocol["responses"]:
        raise ValueError("Unexpected retained response count")
    retained = align_daily(retained, factors)
    audit = {"source_portfolios": len(portfolios.columns),
             "retained_portfolios": len(retained.columns),
             "excluded": excluded, "missing_source_cells_analysis_period": int(missing.sum().sum()),
             "missing_by_portfolio": missing_by_portfolio, "retained_missing_cells": 0,
             "selection_rule": selection, "imputation_performed": False}
    return retained, audit


def weekly_panel(portfolios: pd.DataFrame, factors: pd.DataFrame,
                 reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate the explicitly aligned daily log-relative/spread transformations."""
    portfolios = align_daily(portfolios, factors)
    rf = np.log1p(factors["RF_3M"])
    y_daily = np.log1p(portfolios).sub(rf, axis=0)
    x_daily = np.log1p(factors.iloc[:, :6]).copy()
    x_daily["MKT_RF_3M"] = np.log1p(factors["MKT_RF_3M"] + factors["RF_3M"]) - rf
    x_daily = x_daily.rename(columns={"MKT_RF_3M": "MKT_LOGREL"})
    x = x_daily.resample("W-WED", closed="right", label="right").sum(min_count=1)
    y = y_daily.resample("W-WED", closed="right", label="right").sum(min_count=1)
    counts = pd.Series(1, index=factors.index).resample(
        "W-WED", closed="right", label="right").sum()
    if not reference.index.isin(x.index).all():
        raise ValueError("Missing required complete weeks")
    x, y = x.loc[reference.index], y.loc[reference.index]
    if not np.allclose(x.iloc[:, 1:], reference.iloc[:, 1:], rtol=0, atol=1e-12):
        raise ValueError("Style factors do not reproduce the supplied weekly reference")
    coverage = pd.DataFrame({"trading_days": counts.loc[reference.index]})
    return x, y, coverage


def independent_weekly(portfolios: pd.DataFrame, factors: pd.DataFrame,
                       labels: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Reference: explicit date intervals and products of gross returns, not resample/log sums."""
    factors_array = factors.to_numpy()
    portfolio_array = portfolios.to_numpy()
    xs, ys = [], []
    for when in labels:
        mask = (factors.index > when - pd.Timedelta(days=7)) & (factors.index <= when)
        f, p = factors_array[mask], portfolio_array[mask]
        if not len(f):
            raise ValueError("Empty complete-week interval")
        cash_gross = np.prod(1 + f[:, -1])
        ys.append(np.log(np.prod(1 + p, axis=0) / cash_gross))
        market = np.log(np.prod(1 + f[:, 0] + f[:, -1]) / cash_gross)
        xs.append(np.r_[market, np.log(np.prod(1 + f[:, 1:6], axis=0))])
    return np.asarray(xs), np.asarray(ys)


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """Save complete precision with stable date labels and LF line endings."""
    return frame.to_csv(index_label="date", date_format="%Y-%m-%d",
                        float_format="%.17g", lineterminator="\n").encode()


def build(root: Path) -> dict:
    """Build audited outputs from frozen local input bytes, without any refitting."""
    root = external_path(root)
    protocol = load_protocol(root)
    if (root / "manifest.json").exists() or (root / "panel").exists():
        raise FileExistsError("Panel outputs already exist; verify them or use a new run")
    factors, reference = read_factors(root / "inputs", protocol)
    receipt = json.loads((root / "download_receipt.json").read_text(encoding="utf-8"))
    raw = (root / "raw/portfolios.zip").read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["archive_sha256"]:
        raise ValueError("Downloaded archive changed")
    member, content = zip_member(raw)
    if (member != receipt["member"] or
            hashlib.sha256(content).hexdigest() != receipt["member_sha256"]):
        raise ValueError("Downloaded member changed")
    portfolios, all_mapping, parser = parse_portfolios(content)
    selected, missingness = select_responses(portfolios, factors, protocol)
    mapping = all_mapping.loc[all_mapping["portfolio"].isin(selected.columns)].reset_index(
        drop=True)
    x, y, coverage = weekly_panel(selected, factors, reference)
    ref_x, ref_y = independent_weekly(selected, factors, reference.index)
    if (not np.allclose(x, ref_x, rtol=0, atol=1e-12) or
            not np.allclose(y, ref_y, rtol=0, atol=1e-12)):
        raise ValueError("Independent gross-return reconstruction failed")
    c = protocol["calendar"]
    if x.shape != (c["weekly_rows"], 6) or y.shape != (c["weekly_rows"], protocol["responses"]):
        raise ValueError("Unexpected final weekly panel shape")
    parser.update({"archive_member": member, "member_sha256": receipt["member_sha256"],
                   "required_daily_rows": len(selected), "required_sentinel_or_missing_cells": 0,
                   "excluded_source_rows_outside_factor_period": len(portfolios)-len(selected)})
    validation = {
        "status": "passed", "no_estimator_fitted": True, "weekly_rows": len(x),
        "response_columns": y.shape[1], "factor_columns": x.shape[1],
        "first_week": x.index[0].strftime("%Y-%m-%d"),
        "last_week": x.index[-1].strftime("%Y-%m-%d"),
        "required_daily_rows": len(selected), "missing_cells": 0,
        "max_independent_x_difference": float(np.max(np.abs(x.to_numpy()-ref_x))),
        "max_independent_y_difference": float(np.max(np.abs(y.to_numpy()-ref_y))),
        "max_style_reference_difference": float(np.max(np.abs(
            x.iloc[:, 1:].to_numpy()-reference.iloc[:, 1:].to_numpy()))),
        "max_new_market_difference_from_old": float(np.max(np.abs(
            x.iloc[:, 0].to_numpy()-reference.iloc[:, 0].to_numpy()))),
        "complete_week_trading_day_counts": {str(k): int(v) for k, v in
            coverage["trading_days"].value_counts().sort_index().items()},
        "omitted_initial_partial_label": (reference.index[0] - pd.Timedelta(days=7)).strftime(
            "%Y-%m-%d"),
        "omitted_final_incomplete_label": (reference.index[-1] + pd.Timedelta(days=7)).strftime(
            "%Y-%m-%d"),
        "distribution_rights": "unverified; not approved for public redistribution",
        "mixed_release_vintage": "not verified to match supplied factor snapshot",
        "parser_source_sha256": sha256(Path(__file__)),
    }
    for name, frame in {"X_weekly.csv": x, "Y_weekly.csv": y,
                        "weekly_panel.csv": pd.concat([x, y], axis=1),
                        "weekly_coverage.csv": coverage}.items():
        write_new(root / "panel" / name, csv_bytes(frame))
    write_new(root / "panel/portfolio_labels.csv", mapping.to_csv(
        index=False, lineterminator="\n").encode())
    write_new(root / "panel/source_portfolio_labels.csv", all_mapping.to_csv(
        index=False, lineterminator="\n").encode())
    write_new(root / "panel/missingness_audit.json", json_bytes(missingness))
    write_new(root / "panel/parser_receipt.json", json_bytes(parser))
    write_new(root / "panel/validation.json", json_bytes(validation))
    freeze_receipt = json.loads((root / "freeze_receipt.json").read_text(encoding="utf-8"))
    files = list(freeze_receipt["files_sha256"]) + ["freeze_receipt.json",
            "download_receipt.json", "raw/portfolios.zip"]
    if (root / "download_adoption.json").is_file():
        files.append("download_adoption.json")
    files += [p.relative_to(root).as_posix() for p in sorted((root / "panel").iterdir())]
    manifest = {"schema_version": 1, "created_utc": utc_now(),
                "study_id": protocol["study_id"], "protocol_sha256": sha256(root / "protocol.json"),
                "rights": protocol["rights_and_provenance"],
                "files_sha256": {name: sha256(root / name) for name in files}}
    write_new(root / "manifest.json", json_bytes(manifest))
    return validation


def verify(root: Path) -> dict:
    """Offline audit of hashes, labels, calendars and all weekly output values."""
    protocol = load_protocol(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"Output/input checksum mismatch: {name}")
    raw = (root / "raw/portfolios.zip").read_bytes()
    _, content = zip_member(raw)
    portfolios, all_mapping, parser = parse_portfolios(content)
    factors, reference = read_factors(root / "inputs", protocol)
    selected, missingness = select_responses(portfolios, factors, protocol)
    mapping = all_mapping.loc[all_mapping["portfolio"].isin(selected.columns)].reset_index(
        drop=True)
    saved_missingness = json.loads(
        (root / "panel/missingness_audit.json").read_text(encoding="utf-8"))
    if missingness != saved_missingness:
        raise ValueError("Missingness/selection audit changed")
    if not all_mapping.equals(pd.read_csv(root / "panel/source_portfolio_labels.csv")):
        raise ValueError("Full source portfolio mapping changed")
    x = read_csv(root / "panel/X_weekly.csv")
    y = read_csv(root / "panel/Y_weekly.csv")
    combined = read_csv(root / "panel/weekly_panel.csv")
    expected_mapping = pd.read_csv(root / "panel/portfolio_labels.csv")
    if not mapping.equals(expected_mapping):
        raise ValueError("Output portfolio mapping does not match source labels")
    if not x.index.equals(reference.index) or not y.index.equals(reference.index):
        raise ValueError("Output weekly calendar changed")
    if (list(x.columns) != protocol["predictors"] or
            list(y.columns) != mapping["portfolio"].tolist()):
        raise ValueError("Output columns changed")
    pd.testing.assert_frame_equal(combined, pd.concat([x, y], axis=1))
    ref_x, ref_y = independent_weekly(selected, factors, reference.index)
    if (not np.allclose(x, ref_x, atol=1e-12, rtol=0) or
            not np.allclose(y, ref_y, atol=1e-12, rtol=0)):
        raise ValueError("Independent output reconstruction failed")
    if not np.allclose(x.iloc[:, 1:], reference.iloc[:, 1:], atol=1e-12, rtol=0):
        raise ValueError("Style reference reconciliation failed")
    calendar = json.loads((root / "calendar.json").read_text(encoding="utf-8"))
    if calendar != calendar_inventory(x.index.strftime("%Y-%m-%d").tolist(), protocol):
        raise ValueError("Calendar inventory changed")
    with zipfile.ZipFile(root / "source_snapshot.zip") as archive:
        environment = json.loads((root / "environment.json").read_text(encoding="utf-8"))
        for name, expected in environment["source_sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"Source snapshot hash mismatch: {name}")
    return {"status": "passed", "offline": True, "verified_files": len(manifest["files_sha256"]),
            "verified_weekly_values": int(x.size+y.size), "weekly_rows": len(x),
            "responses": y.shape[1], "predictors": x.shape[1],
            "source_full_history_rows": parser["source_rows"],
            "protocol_sha256": sha256(root / "protocol.json"), "no_estimator_fitted": True}


def main() -> None:
    """Expose explicit freeze/download/build/verify steps without implicit network calls."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ("freeze", "download", "build", "verify"):
        action.add_argument("--" + name, action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--factor-daily", type=Path)
    parser.add_argument("--factor-weekly", type=Path)
    parser.add_argument("--factor-producer", type=Path)
    parser.add_argument("--reuse-download", type=Path,
                        help="--download: copy a previous frozen run's original download")
    args = parser.parse_args()
    root = external_path(args.out)
    if args.freeze:
        if any(v is None for v in (args.factor_daily, args.factor_weekly, args.factor_producer)):
            parser.error("--freeze requires all three explicit factor input paths")
        result = freeze(root, {"factor_daily.csv": args.factor_daily,
                               "factor_weekly.csv": args.factor_weekly,
                               "factor_producer.py": args.factor_producer})
    elif args.download:
        result = download(root, from_run=args.reuse_download)
    elif args.build:
        result = build(root)
    else:
        result = verify(root)
    print(json_bytes(result).decode(), end="")


if __name__ == "__main__":
    main()
