"""Parallel read-only checkpoint audits using the unchanged strict window validator.

Validation has its own source receipt, separate from the estimation identity.
No fitting is performed and no estimator or protocol setting is changed.
The report schema and window checks match french_full_run.finalize exactly.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
import traceback

if __package__:
    from . import french_full_run as S
else:
    import french_full_run as S


def finalize(panel, output, *, workers=None, require_complete=True):
    """Check every sealed window concurrently, then write the strict certificate."""
    S.require(require_complete, "Finalization requires --require-complete")
    started = time.perf_counter()
    protocol = S.read_json(panel / "protocol.json")
    S.configure_threads(protocol)
    S.R.verify_panel(panel)
    x = S.R.read_csv(panel / "panel/X_weekly.csv")
    y = S.R.read_csv(panel / "panel/Y_weekly.csv")
    manifest = S.read_json(output / "run_manifest.json")
    binding = S.execution_binding(panel, protocol,
        workers=manifest["binding"]["scheduler_resources"]["workers"],
        source_upgrade_reason=manifest["binding"].get("core_source_upgrade", {}).get("reason"))
    S.require(manifest["binding"] == binding and manifest["identity"] == S.R.digest(binding),
              "Stale/mixed execution identity")
    S.R.atomic_json(output / "full_validation.json", {"status": "validating",
                                                     "identity": manifest["identity"]})
    expected = S.job_inventory(protocol, len(x))
    S.require(manifest["jobs"] == expected, "Unexpected/duplicate job inventory")
    for name, value in binding["source_sha256"].items():
        S.require(S.R.sha256(output / "source" / name) == value, "Corrupted source snapshot")
    paths = {p.parent.relative_to(output).as_posix() for p in output.rglob("window_receipt.json")}
    S.require(paths == {j["folder"] for j in expected},
              "Missing/duplicate/unexpected completed windows")
    S.require({p.name for p in output.glob("origin_*") if p.is_dir()}
              == {j["folder"] for j in expected}, "Unexpected/incomplete window directories")
    progress = None
    progress_path = output / "progress.json"
    if progress_path.exists() and not (output / "full_run_receipt.json").exists():
        progress = S.read_json(progress_path)
        S.require(progress["identity"] == manifest["identity"] and not progress["failures"]
                  and not progress["pending"] and len(progress["completed"]) == len(expected)
                  and {r["job"]["folder"] for r in progress["completed"]}
                  == {j["folder"] for j in expected}, "Estimation progress is incomplete or mixed")
    effective = S.execution_resources(protocol, workers if workers is not None else
                                      binding["scheduler_resources"]["workers"])["workers"]
    source = Path(__file__)
    provenance = {"estimation_identity": manifest["identity"],
                  "verifier_source_sha256": S.R.sha256(source),
                  "window_validator_source_sha256": S.R.sha256(Path(S.__file__)),
                  "workers": effective, "fitting_performed": False}
    S.R.atomic_write(output / "validation_source" / source.name, source.read_bytes())
    S.R.atomic_json(output / "parallel_finalization_manifest.json", provenance)
    results = {}
    try:
        with ThreadPoolExecutor(max_workers=effective) as pool:
            pending = {pool.submit(S.audit_window, output / j["folder"], x, y,
                                   protocol, j, manifest["identity"]): j for j in expected}
            for future in as_completed(pending):
                job = pending[future]
                results[job["folder"]] = future.result()
                print(f"Validated {len(results)}/{len(expected)} windows", flush=True)
                S.R.atomic_json(output / "validation_progress.json", {
                    "identity": manifest["identity"], "validated": len(results),
                    "expected": len(expected), "elapsed_seconds": time.perf_counter()-started})
    except Exception:
        S.R.atomic_json(output / "parallel_finalization_failure.json", {
            "identity": manifest["identity"], "traceback": traceback.format_exc()})
        raise
    # Recheck source/input identity after the audit, before issuing a certificate.
    S.require(S.execution_binding(panel, protocol,
              workers=binding["scheduler_resources"]["workers"],
              source_upgrade_reason=binding.get("core_source_upgrade", {}).get("reason"))
              == binding,
              "Source/input/runtime changed during validation")
    S.require(S.R.sha256(source) == provenance["verifier_source_sha256"],
              "Verifier source changed during validation")
    S.R.verify_panel(panel)
    audits = [results[j["folder"]] for j in expected]
    for t in protocol["calendar"]["lookbacks"]:
        S.require(sum(a["test_weeks"] for a in audits if a["lookback"] == t)
                  == protocol["calendar"]["test_weeks"], "Incomplete test-week coverage")
    report = {"status": "complete", "identity": manifest["identity"],
              "windows": len(audits), "lookbacks": protocol["calendar"]["lookbacks"],
              "test_weeks_per_lookback": protocol["calendar"]["test_weeks"],
              "methods": protocol["methods"], "audits": audits,
              "inner_evaluations": sum(a["inner_evaluations"] for a in audits),
              "strict_completeness": True, "all_inner_losses_and_selections_reconstructed": True,
              "all_window_hashes_verified": True,
              "estimator_defaults": binding["defaults_observed"],
              "outer_performance_rankings_inspected": False}
    S.R.atomic_json(output / "full_validation.json", report)
    seconds = time.perf_counter()-started
    S.R.atomic_json(output / "parallel_finalization_receipt.json", {
        **provenance, "status": "complete", "elapsed_seconds": seconds,
        "validation_sha256": S.R.sha256(output / "full_validation.json")})
    # A standalone audit can complete a sealed estimation phase whose original
    # serial finalizer was interrupted. Preserve an existing producer receipt.
    if progress is not None:
        S.R.atomic_json(output / "full_run_receipt.json", {
            "identity": manifest["identity"], "status": "complete", "jobs": len(expected),
            "workers": binding["scheduler_resources"]["workers"],
            "wall_seconds_this_invocation": seconds,
            "fit_wall_seconds": progress["wall_seconds"],
            "combined_fit_and_validation_seconds": progress["wall_seconds"]+seconds,
            "validation_sha256": S.R.sha256(output / "full_validation.json"),
            "finalization_mode": "standalone parallel audit; no refitting",
            "verifier_source_sha256": provenance["verifier_source_sha256"]})
    return {k: v for k, v in report.items() if k != "audits"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--finalize", action="store_true", required=True)
    parser.add_argument("--require-complete", action="store_true", required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--panel-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = S.R.external_path(args.out)
    with S.exclusive_run(output):
        result = finalize(args.panel_run, output, workers=args.workers,
                          require_complete=args.require_complete)
    print(S.R.json_bytes(result).decode(), flush=True)


if __name__ == "__main__":
    main()
