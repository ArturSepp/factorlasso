"""Replay every frozen French ridge candidate through the public package estimator.

This is a migration audit, not a new statistical protocol. It preserves the
historical run, repeats only ridge across every origin/lookback, and checks inner
coefficients/losses, tuning choices, outer coefficients, predictions and losses.
Outputs require a fresh directory outside OneDrive. No archived shard is reused
as a newly estimated fit, and no other estimator family is rerun.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import time

import numpy as np

if __package__:
    from . import french_full_run as S
    from . import french_rolling_validation as R
else:
    import french_full_run as S
    import french_rolling_validation as R


def verified_file(folder, receipt, name):
    """Authenticate one consumed artifact against its sealed window receipt."""
    path = folder / name
    S.require(name in receipt["files_sha256"], f"Reference artifact is unlisted: {name}")
    S.require(R.sha256(path) == receipt["files_sha256"][name],
              f"Corrupted reference artifact: {path}")
    return path


def max_difference(actual, expected):
    """Require finite equal-shaped arrays and return their largest discrepancy."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    S.require(actual.shape == expected.shape, "Migration array shape mismatch")
    S.require(np.isfinite(actual).all() and np.isfinite(expected).all(),
              "Nonfinite migration comparison")
    return float(np.max(np.abs(actual-expected), initial=0.))


def compare_window(reference, output, job, reference_identity, protocol):
    """Compare all ridge checkpoints for one origin, not just the selected fit."""
    old, new = reference / job["folder"], output / job["folder"]
    receipt = S.read_json(old / "window_receipt.json")
    S.require(receipt["identity"] == reference_identity, "Mixed historical window identity")
    surface = S.read_json(verified_file(old, receipt, "ridge_inner_surface.json"))
    current = S.read_json(new / "ridge_inner_surface.json")
    expected_candidates = R.candidates(protocol, "ridge")
    S.require([row["candidate"] for row in surface["rows"]] == expected_candidates,
              "Historical ridge inventory differs from frozen protocol")
    S.require([row["candidate"] for row in current["rows"]] == expected_candidates,
              "Migrated ridge inventory differs from frozen protocol")
    selected = S.read_json(verified_file(old, receipt, "selections.json"))["ridge"]
    new_selected = S.read_json(new / "selections.json")["ridge"]
    errors = {"inner_beta": 0., "inner_alpha": 0., "inner_sse": 0.,
              "candidate_score": 0., "outer_beta": 0., "outer_intercept": 0.,
              "outer_beta_standardized": 0., "outer_alpha_standardized": 0.,
              "outer_prediction": 0., "outer_weekly_standardized_loss": 0.}
    failures = []

    def compare(key, actual, expected):
        errors[key] = max(errors[key], max_difference(actual, expected))
        if not np.allclose(actual, expected, atol=1e-10, rtol=1e-9):
            failures.append(key)

    expected_pairs = {(R.digest(c), f) for c in expected_candidates for f in range(
        len(protocol["calendar"]["inner_train_ends"][str(job["lookback"])]))}
    old_ledger = {(r["candidate_id"], r["inner_fold"]): r for r in surface["ledger"]}
    new_ledger = {(r["candidate_id"], r["inner_fold"]): r for r in current["ledger"]}
    S.require(len(old_ledger) == len(surface["ledger"]) and set(old_ledger) == expected_pairs,
              "Historical ridge candidate/fold inventory is incomplete or duplicated")
    S.require(len(new_ledger) == len(current["ledger"]) and set(new_ledger) == expected_pairs,
              "Migrated ridge candidate/fold inventory is incomplete or duplicated")
    for key in sorted(expected_pairs):
        old_row, new_row = old_ledger[key], new_ledger[key]
        S.require(old_row["fit_key"] == new_row["fit_key"], "Objective/preprocessing key changed")
        name = f"inner_{key[1]}_fits/{old_row['fit_key']}.npz"
        with np.load(verified_file(old, receipt, name), allow_pickle=False) as a, \
                np.load(new / name, allow_pickle=False) as b:
            compare("inner_beta", b["beta"], a["beta"])
            compare("inner_alpha", b["alpha"], a["alpha"])
            metadata = json.loads(b["metadata"].tobytes())
            S.require(metadata["result"]["kind"] == "package_ridge",
                      "Migration did not use package ridge")
        compare("inner_sse", new_row["sse"], old_row["sse"])
    for a, b in zip(surface["rows"], current["rows"]):
        compare("candidate_score", b["score"], a["score"])
    for name in ["outer_preprocessing.json"] + [f"inner_{f}_preprocessing.json" for f in
            range(len(protocol["calendar"]["inner_train_ends"][str(job["lookback"])]))]:
        S.require(S.read_json(verified_file(old, receipt, name)) == S.read_json(new / name),
                  "Training transformation changed")
    with np.load(verified_file(old, receipt, "outer_arrays.npz"), allow_pickle=False) as a, \
            np.load(new / "outer_arrays.npz", allow_pickle=False) as b:
        for field in ("beta", "intercept", "beta_standardized", "alpha_standardized",
                      "prediction", "weekly_standardized_loss"):
            compare("outer_"+field, b["ridge__"+field], a["ridge__"+field])
        weeks = len(b["ridge__weekly_standardized_loss"])
        old_mse = float(a["ridge__weekly_standardized_loss"].mean())
        new_mse = float(b["ridge__weekly_standardized_loss"].mean())
    same_choice = (new_selected["candidate"] == selected["candidate"] and
                   new_selected["lambda"] == selected["lambda"])
    if not same_choice:
        failures.append("selected_candidate")
    return {**job, "inner_candidate_fold_comparisons": len(expected_pairs),
            "candidates": len(expected_candidates), "selected_candidate_unchanged": same_choice,
            "old_selected_candidate": selected["candidate"],
            "new_selected_candidate": new_selected["candidate"],
            "max_absolute_differences": errors, "test_weeks": weeks,
            "old_outer_mse": old_mse, "new_outer_mse": new_mse,
            "failed_comparisons": sorted(set(failures))}


def replay(panel, reference, output):
    """Replay every ridge grid and issue a source-bound complete migration receipt."""
    started = time.perf_counter()
    output = R.external_path(output)
    S.require(not output.exists() or not any(output.iterdir()), "Use a fresh migration directory")
    protocol = S.read_json(panel / "protocol.json")
    S.configure_threads(protocol)
    R.verify_panel(panel)
    x = R.read_csv(panel / "panel/X_weekly.csv")
    y = R.read_csv(panel / "panel/Y_weekly.csv")
    old_manifest = S.read_json(reference / "run_manifest.json")
    old_certificate = S.read_json(reference / "full_validation.json")
    old_receipt = S.read_json(reference / "full_run_receipt.json")
    S.require(old_manifest["identity"] == R.digest(old_manifest["binding"]),
              "Historical execution identity mismatch")
    S.require(old_certificate["status"] == old_receipt["status"] == "complete" and
              old_certificate["identity"] == old_receipt["identity"] == old_manifest["identity"],
              "Historical run is not certified complete")
    S.require(old_receipt["validation_sha256"] == R.sha256(reference / "full_validation.json"),
              "Historical validation certificate hash mismatch")
    for key, name in (("panel_manifest_sha256", "manifest.json"),
                      ("protocol_sha256", "protocol.json")):
        S.require(old_manifest["binding"][key] == R.sha256(panel / name),
                  "Historical panel/protocol mismatch")
    jobs = S.job_inventory(protocol, len(x))
    S.require(jobs == old_manifest["jobs"], "Historical job inventory mismatch")
    certified_receipts = {f"origin_{r['origin']}_T{r['lookback']}": r["window_receipt_sha256"]
                          for r in old_certificate["audits"]}
    S.require(len(certified_receipts) == len(old_certificate["audits"]) and
              set(certified_receipts) == {job["folder"] for job in jobs},
              "Historical certificate has incomplete or duplicate windows")
    binding = R.execution_identity(panel, protocol)
    binding["source_sha256"][Path(__file__).relative_to(R.REPOSITORY).as_posix()] = R.sha256(
        Path(__file__))
    binding["source_sha256"][Path(S.__file__).relative_to(R.REPOSITORY).as_posix()] = R.sha256(
        Path(S.__file__))
    binding.update(reference_identity=old_manifest["identity"],
                   reference_manifest_sha256=R.sha256(reference / "run_manifest.json"),
                   reference_certificate_sha256=R.sha256(reference / "full_validation.json"),
                   reference_window_receipts_sha256=certified_receipts,
                   executed_methods=["ridge"], frozen_methods=protocol["methods"],
                   comparison_atol=1e-10, comparison_rtol=1e-9)
    identity = R.digest(binding)
    output.mkdir(parents=True, exist_ok=True)
    R.atomic_json(output / "migration_manifest.json", {"identity": identity, "binding": binding,
                                                       "jobs": jobs})
    for name in binding["source_sha256"]:
        R.atomic_write(output / "source" / name, (R.REPOSITORY / name).read_bytes())
    ridge_protocol = copy.deepcopy(protocol)
    ridge_protocol["methods"] = ["ridge"]
    comparisons = []
    try:
        for i, job in enumerate(jobs, 1):
            S.require(R.sha256(reference / job["folder"] / "window_receipt.json")
                      == certified_receipts[job["folder"]], "Historical window seal changed")
            with redirect_stdout(io.StringIO()):
                R.run_window(x, y, job["origin"], job["lookback"], ridge_protocol,
                             folder=output / job["folder"], identity=identity)
            comparisons.append(compare_window(reference, output, job, old_manifest["identity"],
                                              protocol))
            R.atomic_json(output / "migration_progress.json", {"completed": i, "total": len(jobs),
                          "elapsed_seconds": time.perf_counter()-started})
            print(f"Ridge migration {i}/{len(jobs)}: {job['folder']}", flush=True)
        for name, expected in binding["source_sha256"].items():
            S.require(R.sha256(R.REPOSITORY / name) == expected,
                      "Sources changed during migration replay; repeat in a fresh directory")
        R.verify_panel(panel)
        S.require(R.sha256(reference / "run_manifest.json")
                  == binding["reference_manifest_sha256"] and
                  R.sha256(reference / "full_validation.json")
                  == binding["reference_certificate_sha256"],
                  "Historical run changed during replay")
        summary = {}
        for lookback in protocol["calendar"]["lookbacks"]:
            rows = [r for r in comparisons if r["lookback"] == lookback]
            weeks = sum(r["test_weeks"] for r in rows)
            S.require(weeks == protocol["calendar"]["test_weeks"], "Incomplete replay calendar")
            summary[str(lookback)] = {
                "origins": len(rows), "test_weeks": weeks,
                "old_standardized_mse": sum(r["old_outer_mse"]*r["test_weeks"] for r in rows)/weeks,
                "new_standardized_mse": sum(r["new_outer_mse"]*r["test_weeks"] for r in rows)/weeks,
                "changed_selections": sum(not r["selected_candidate_unchanged"] for r in rows)}
        failed = [r["folder"] for r in comparisons if r["failed_comparisons"]]
        report = {"status": "complete" if not failed else "discrepancies",
                  "identity": identity, "reference_identity": old_manifest["identity"],
                  "windows": len(comparisons), "lookbacks": summary,
                  "inner_candidate_fold_comparisons": sum(
                      r["inner_candidate_fold_comparisons"] for r in comparisons),
                  "outer_refits": len(comparisons), "failed_windows": failed,
                  "max_absolute_differences": {key: max(r["max_absolute_differences"][key]
                      for r in comparisons) for key in comparisons[0]["max_absolute_differences"]},
                  "wall_seconds": time.perf_counter()-started, "comparisons": comparisons,
                  "scope": "All ridge candidates and selections only; other families not rerun",
                  "historical_outputs_modified": False, "defaults_changed": False}
        R.atomic_json(output / "migration_report.json", report)
        return {k: v for k, v in report.items() if k != "comparisons"}
    except Exception as exc:
        R.atomic_json(output / "migration_failure.json", {"identity": identity, "error": repr(exc),
                      "completed_windows": len(comparisons)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel-run", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.panel_run, args.reference_run, args.out)
    print(R.json_bytes(result).decode(), flush=True)
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
