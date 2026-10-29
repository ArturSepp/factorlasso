"""Hash-locked full French study scheduler and strict, no-refit finalization.

Uses the unchanged Stage 2 estimation engine. Finalization independently
reconstructs inner losses and selections, checks exact candidate/fold and
outer-calendar coverage, validates every checkpoint, and rejects mixed sources.
Scientific aggregation and uncertainty intervals belong to Stage 4.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

if __package__:
    from . import french_rolling_validation as R
else:
    import french_rolling_validation as R


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def job_inventory(protocol, rows):
    """Enumerate every frozen origin/lookback and verify complete test coverage."""
    c = protocol["calendar"]
    origins = list(range(c["outer_burn_in"], rows, c["outer_horizon"]))
    require(len(origins) == c["origins"], "Outer-origin count differs from protocol")
    require(rows-c["outer_burn_in"] == c["test_weeks"], "Test-week count differs from protocol")
    require(len(c["lookbacks"]) == len(set(c["lookbacks"])), "Duplicate lookbacks")
    require(min(origins) >= max(c["lookbacks"]), "Insufficient outer training history")
    return [{"origin": o, "lookback": t, "folder": f"origin_{o}_T{t}"}
            for o in origins for t in c["lookbacks"]]


def configure_threads(protocol):
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(protocol["solver"]["blas_threads_per_worker"])


def execution_resources(protocol, workers=None):
    """Record a compute-only override without editing the frozen protocol."""
    effective = protocol["solver"]["workers"] if workers is None else workers
    require(isinstance(effective, int) and 1 <= effective <= (os.cpu_count() or 1),
            "Workers must be between one and the available logical processor count")
    return {"workers": effective,
            "blas_threads_per_worker": protocol["solver"]["blas_threads_per_worker"]}


def execution_binding(panel, protocol, *, workers=None, source_upgrade_reason=None):
    """Bind unchanged sources, or an explicitly acknowledged new-source experiment.

    Acknowledging a package upgrade never permits mixing historical checkpoints:
    the old/new hashes and reason enter a new execution identity.
    """
    binding = R.execution_identity(panel, protocol)
    frozen = read_json(panel / "environment.json")["source_sha256"]
    old = {k: v for k, v in frozen.items() if k.startswith("src/factorlasso/")}
    new = {k: v for k, v in binding["source_sha256"].items()
           if k.startswith("src/factorlasso/")}
    changed = {k: {"stage1_sha256": old.get(k), "current_sha256": new.get(k)}
               for k in sorted(old.keys() | new.keys()) if old.get(k) != new.get(k)}
    if source_upgrade_reason is not None:
        require(isinstance(source_upgrade_reason, str) and source_upgrade_reason.strip(),
                "Source upgrade requires a nonempty reason")
        binding["core_source_upgrade"] = {
            "reason": source_upgrade_reason.strip(),
            "stage1_environment_sha256": R.sha256(panel / "environment.json"),
            "stage1_core_sha256": old, "current_core_sha256": new,
            "changed_sources": changed,
        }
    else:
        require(not changed, "Core sources differ from Stage 1; a fresh run requires "
                "explicit --source-upgrade-reason")
    name = Path(__file__).relative_to(R.REPOSITORY).as_posix()
    binding["source_sha256"][name] = R.sha256(Path(__file__))
    binding["scheduler_resources"] = execution_resources(protocol, workers)
    return binding


@contextmanager
def exclusive_run(output):
    """OS-released lock prevents concurrent schedulers/finalizers, even after a crash."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("Another scheduler/finalizer owns this run") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def initialize(panel, output, protocol, x, *, workers=None, source_upgrade_reason=None):
    binding = execution_binding(panel, protocol, workers=workers,
                                source_upgrade_reason=source_upgrade_reason)
    manifest = {"schema": 1, "identity": R.digest(binding), "binding": binding,
                "jobs": job_inventory(protocol, len(x))}
    path = output / "run_manifest.json"
    if path.exists():
        require(read_json(path) == manifest, "Stale or mixed source/input/protocol/runtime run")
    else:
        require(not list(output.glob("origin_*")), "Unbound existing window outputs")
        R.atomic_json(path, manifest)
        for name in binding["source_sha256"]:
            R.atomic_write(output / "source" / name, (R.REPOSITORY / name).read_bytes())
    return manifest


def load_fit(path, identity, state_hash, expected_key):
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == {"metadata", "beta", "alpha"}, "Unexpected fit arrays")
        meta = json.loads(saved["metadata"].tobytes())
        beta, alpha = saved["beta"].copy(), saved["alpha"].copy()
    require(meta["identity"] == identity and meta["state_hash"] == state_hash
            and meta["fit_key"] == expected_key, "Stale/mixed fit checkpoint")
    require(np.isfinite(beta).all() and np.isfinite(alpha).all(), "Nonfinite fit")
    require(R.F.array_hash(beta) == meta["beta_hash"]
            and R.F.array_hash(alpha) == meta["alpha_hash"], "Fit coefficient hash mismatch")
    result = meta["result"]
    require(result["status"] in ("optimal", "resolved_inaccurate"), "Unresolved fit status")
    require(np.isfinite(result["objective"]) and result["kkt_residual"] <= R.F.KKT_TOL
            and result["constraint_violation"] <= 1e-7, "Fit numerical checks failed")
    if result["status"] == "resolved_inaccurate":
        attempts = result["attempts"]
        require(len(attempts) == 2, "Inaccurate fit lacks prescribed retry")
        a = attempts[result["selected_attempt_index"]]
        require(a["status"] == "optimal_inaccurate"
                and a["inaccurate_flag_resolved_by_independent_checks"]
                and a["kkt_residual"] <= R.F.POLISH_TOL
                and a["constraint_violation"] <= 1e-7
                and a["independent_relative_dual_gap"] <= 1e-8
                and a["dual_residual"] <= 1e-7
                and a["objective_discrepancy"] <= R.F.OBJECTIVE_TOL
                and a["raw_objective_discrepancy"] <= R.F.OBJECTIVE_TOL,
                "Inaccurate fit fails stricter resolution criteria")
    return beta, alpha, result


def audit_window(folder, x, y, protocol, job, identity):
    """No fitting: validate hashes, coverage, losses, selection and raw predictions."""
    receipt = read_json(folder / "window_receipt.json")
    o, t = job["origin"], job["lookback"]
    end = min(o+protocol["calendar"]["outer_horizon"], len(x))
    require(receipt["status"] == "complete" and receipt["identity"] == identity
            and receipt["origin"] == o and receipt["lookback"] == t, "Mixed window receipt")
    job_key = R.digest({"origin": o, "lookback": t, "protocol": R.digest(protocol),
                        "x_window": R.F.array_hash(x.iloc[o-t:end].to_numpy()),
                        "y_window": R.F.array_hash(y.iloc[o-t:end].to_numpy())})
    require(receipt["job_key"] == job_key, "Window input identity mismatch")
    actual_files = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()
                    and p.name != "window_receipt.json"}
    require(actual_files == set(receipt["files_sha256"]), "Missing/unexpected window artifacts")
    for name, value in receipt["files_sha256"].items():
        path = folder / name
        require(path.resolve().is_relative_to(folder.resolve()), "Unsafe artifact path")
        require(R.sha256(path) == value, f"Corrupted window artifact: {name}")
    def dates(frame):
        return frame.index.strftime("%Y-%m-%d").tolist()
    require(receipt["train_dates"] == dates(x.iloc[o-t:o])
            and receipt["test_dates"] == dates(x.iloc[o:end]), "Window date mismatch")
    selections = read_json(folder / "selections.json")
    require(set(selections) == set(protocol["methods"]), "Missing/duplicate method selections")
    ledger = read_json(folder / "inner_ledger.json")
    fold_ends = protocol["calendar"]["inner_train_ends"][str(t)]
    val_ends = protocol["calendar"]["inner_validation_ends"][str(t)]
    transforms, state_hashes = {}, {}
    for fold, (stop, val_end) in enumerate(zip(fold_ends, val_ends)):
        state = R.F.prepare(x.iloc[o-t:o-t+stop], y.iloc[o-t:o-t+stop])
        stored = read_json(folder / f"inner_{fold}_preprocessing.json")
        require(stored == state.manifest(), "Inner preprocessing mismatch")
        state_hashes[fold] = R.digest(stored)
        xv, yv = x.iloc[o-t+stop:o-t+val_end], y.iloc[o-t+stop:o-t+val_end]
        transforms[fold] = ((xv.to_numpy()-state.x_mean)/state.x_sd,
                            (yv.to_numpy()-state.y_mean)/state.y_sd, state.dates, dates(xv))
    expected_pairs, combined = set(), []
    losses, cache = {}, {}
    max_loss_error = 0.
    for row in ledger:
        cid, fold, key = row["candidate_id"], row["inner_fold"], row["fit_key"]
        require(cid == R.digest(row["candidate"]), "Incorrect candidate identity")
        require(fold in transforms, "Unexpected inner fold")
        xv, yv, train_dates, val_dates = transforms[fold]
        require(row["train_dates"] == train_dates and row["validation_dates"] == val_dates
                and max(train_dates) < min(val_dates), "Leaked/misdated inner rows")
        ck = (fold, key)
        if ck not in cache:
            cache[ck] = load_fit(folder / f"inner_{fold}_fits/{key}.npz", identity,
                                 state_hashes[fold], key)
        beta, alpha, result = cache[ck]
        for name in ("status", "kind", "lambda", "objective", "kkt_residual",
                     "constraint_violation", "attempts"):
            require(row[name] == result[name], f"Ledger/fit mismatch: {name}")
        sse = float(np.square(yv-(xv@beta.T+alpha)).sum())
        max_loss_error = max(max_loss_error, abs(sse-row["sse"]))
        require(abs(sse-row["sse"]) <= 1e-9*(1+abs(sse)), "Inner loss reconstruction failed")
        require(row["validation_cells"] == yv.size, "Validation denominator mismatch")
        total = losses.setdefault(cid, [0., 0])
        total[0] += sse
        total[1] += yv.size
    for method in protocol["methods"]:
        surface = read_json(folder / f"{method}_inner_surface.json")
        rows = surface["rows"]
        base = {R.digest(c) for c in R.candidates(protocol, method)}
        base_rows = [r for r in rows if r["candidate_id"] in base]
        require(len(base_rows) == len(base), "Missing/duplicate base candidates")
        extension = R.refinement_for(base_rows, protocol)
        expected = R.candidates(protocol, method, extension=extension)
        require([r["candidate"] for r in rows] == expected,
                "Missing/duplicate/reordered candidates")
        require(selections[method]["extension"] == extension, "Unexpected refinement decision")
        for row in rows:
            cid = R.digest(row["candidate"])
            require(row["candidate_id"] == cid and cid in losses, "Missing candidate loss")
            total, count = losses[cid]
            require(abs(total/count-row["score"]) <= 1e-11*(1+abs(row["score"])),
                    "Surface score mismatch")
            expected_pairs.update((cid, f) for f in range(len(fold_ends)))
        # Independent tie selection from the stored, ordered complete surface.
        minimum = min(r["score"] for r in rows)
        tol = protocol["selection"]["tie_atol"]+protocol["selection"]["tie_rtol"]*abs(minimum)
        winner = next(r for r in rows if r["score"] <= minimum+tol)
        require(all(selections[method][k] == winner[k] for k in winner),
                "Tuning selection mismatch")
        combined.extend(surface["ledger"])
    pairs = [(r["candidate_id"], r["inner_fold"]) for r in ledger]
    require(len(pairs) == len(set(pairs)) and set(pairs) == expected_pairs
            and len(pairs) == receipt["logical_inner_evaluations"],
            "Incomplete/duplicate inner ledger")
    require(combined == ledger, "Surface/combined ledger mismatch")
    for fold in range(len(fold_ends)):
        actual = {p.stem for p in (folder / f"inner_{fold}_fits").glob("*.npz")}
        require(actual == {key for f, key in cache if f == fold}, "Unreferenced/missing inner fits")
    outer_state = R.F.prepare(x.iloc[o-t:o], y.iloc[o-t:o])
    require(read_json(folder / "outer_preprocessing.json") == outer_state.manifest(),
            "Outer preprocessing mismatch")
    outer_hash = R.digest(outer_state.manifest())
    evaluator = R.Evaluator(outer_state, protocol, identity=identity)
    outer_keys, outer_results = set(), {}
    with np.load(folder / "outer_arrays.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["actual"], y.iloc[o:end].to_numpy())
        np.testing.assert_array_equal(arrays["training_mean"], outer_state.y_mean)
        np.testing.assert_array_equal(arrays["training_sd"], outer_state.y_sd)
        for method in protocol["methods"]:
            selected = selections[method]
            candidate = selected["candidate"]
            spec = evaluator.spec(candidate)
            lam = np.inf if selected["lambda"] == "inf" else selected["lambda"]
            key = evaluator.fit_key(spec, lam)
            require(selected["fit_key"] == key, "Selected fit key mismatch")
            beta, alpha, result = load_fit(folder / f"outer_fits/{key}.npz", identity,
                                           outer_hash, key)
            raw_beta = beta*outer_state.y_sd[:, None]/outer_state.x_sd[None, :]
            intercept = outer_state.y_mean+outer_state.y_sd*alpha-raw_beta@outer_state.x_mean
            for suffix, value in (("beta", raw_beta), ("intercept", intercept),
                                  ("beta_standardized", beta), ("alpha_standardized", alpha),
                                  ("labels", spec.labels), ("gates", spec.signs)):
                np.testing.assert_allclose(arrays[method+"__"+suffix], value,
                                           atol=1e-12, rtol=1e-12)
            outer_keys.add(key)
            outer_results[key] = result
        matched = []
        if "hcgl_unconstrained" in selections:
            reference = selections["hcgl_unconstrained"]
            for method in ("hcgl_singleton_exclude", "hcgl_exclude", "hcgl_unrestricted"):
                candidate = {"method": method, "partition": reference["candidate"]["partition"],
                             "tau": 2, "fraction": 0}
                spec = evaluator.spec(candidate)
                key = evaluator.fit_key(spec, reference["lambda"])
                beta, alpha, result = load_fit(folder / f"outer_fits/{key}.npz", identity,
                                               outer_hash, key)
                raw_beta = beta*outer_state.y_sd[:, None]/outer_state.x_sd[None, :]
                name = "matched__"+method
                np.testing.assert_allclose(arrays[name+"__beta"], raw_beta, atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(arrays[name+"__gates"], spec.signs, atol=0, rtol=0)
                outer_keys.add(key)
                matched.append(name)
                outer_results[key] = result
        for name in list(protocol["methods"])+matched:
            beta = arrays[name+"__beta"]
            intercept = (arrays[name+"__intercept"] if name in selections else
                         outer_state.y_mean-beta@outer_state.x_mean)
            prediction = x.iloc[o:end].to_numpy()@beta.T+intercept
            np.testing.assert_allclose(arrays[name+"__prediction"], prediction,
                                       atol=1e-12, rtol=1e-12)
            loss = np.square((y.iloc[o:end].to_numpy()-prediction)/outer_state.y_sd).mean(1)
            np.testing.assert_allclose(arrays[name+"__weekly_standardized_loss"], loss,
                                       atol=1e-12, rtol=1e-12)
        expected_arrays = {"actual", "training_mean", "training_sd"}
        for method in protocol["methods"]:
            expected_arrays.update(method+"__"+s for s in ("beta", "intercept",
                "beta_standardized", "alpha_standardized", "prediction", "labels", "gates",
                "weekly_standardized_loss"))
        for name in matched:
            expected_arrays.update(name+"__"+s for s in
                                   ("beta", "prediction", "gates", "weekly_standardized_loss"))
        require(set(arrays.files) == expected_arrays, "Missing/unexpected outer arrays")
    require({p.stem for p in (folder / "outer_fits").glob("*.npz")} == outer_keys,
            "Missing/unreferenced outer fits")
    results = [v[2] for v in cache.values()]+list(outer_results.values())
    for name, value in {
        "unique_fits": len(results),
        "conic_fits": sum(r["kind"] == "conic" for r in results),
        "conic_attempts": sum(len(r["attempts"]) for r in results),
        "retry_attempts": sum(max(len(r["attempts"])-1, 0) for r in results),
        "independently_resolved_inaccurate_fits": sum(r["status"] == "resolved_inaccurate"
                                                      for r in results),
        "max_kkt_residual": max(r["kkt_residual"] for r in results),
        "max_constraint_violation": max(r["constraint_violation"] for r in results),
    }.items():
        require(receipt["counts"][name] == value, f"Fit count/diagnostic mismatch: {name}")
    return {"origin": o, "lookback": t, "test_weeks": end-o,
            "inner_evaluations": len(pairs), "unique_inner_fits": len(cache),
            "unique_outer_fits": len(outer_keys), "counts": receipt["counts"],
            "max_inner_sse_reconstruction_error": max_loss_error,
            "elapsed_seconds": receipt["elapsed_seconds"],
            "peak_working_set_mib": receipt["peak_working_set_mib"],
            "window_receipt_sha256": R.sha256(folder / "window_receipt.json")}


def worker(arguments):
    panel, output, job, identity = arguments
    panel, output = Path(panel), Path(output)
    log = output / "logs" / (job["folder"]+".log")
    log.parent.mkdir(exist_ok=True)
    with log.open("a", encoding="utf-8", buffering=1) as handle:
        with redirect_stdout(handle), redirect_stderr(handle):
            started = time.perf_counter()
            receipt = R.pilot_worker((str(panel), str(output), job["lookback"],
                                      job["origin"], identity))
            return {"job": job, "counts": receipt["counts"],
                    "elapsed_seconds_this_invocation": time.perf_counter()-started,
                    "window_receipt_sha256": R.sha256(output / job["folder"] /
                                                      "window_receipt.json")}


def finalize(panel, output, *, require_complete=True):
    """A complete receipt is written only after every expected job passes audit."""
    require(require_complete, "Finalization requires --require-complete")
    protocol = read_json(panel / "protocol.json")
    configure_threads(protocol)
    R.verify_panel(panel)
    x, y = R.read_csv(panel / "panel/X_weekly.csv"), R.read_csv(panel / "panel/Y_weekly.csv")
    manifest = read_json(output / "run_manifest.json")
    binding = execution_binding(panel, protocol,
        workers=manifest["binding"]["scheduler_resources"]["workers"],
        source_upgrade_reason=manifest["binding"].get("core_source_upgrade", {}).get("reason"))
    require(manifest["binding"] == binding and manifest["identity"] == R.digest(binding),
            "Stale/mixed execution identity")
    # An old certificate must never remain complete after a failing re-audit.
    R.atomic_json(output / "full_validation.json", {"status": "validating",
                                                   "identity": manifest["identity"]})
    expected = job_inventory(protocol, len(x))
    require(manifest["jobs"] == expected, "Unexpected/duplicate job inventory")
    for name, value in binding["source_sha256"].items():
        require(R.sha256(output / "source" / name) == value, "Corrupted source snapshot")
    paths = {p.parent.relative_to(output).as_posix() for p in output.rglob("window_receipt.json")}
    require(paths == {j["folder"] for j in expected},
            "Missing/duplicate/unexpected completed windows")
    require({p.name for p in output.glob("origin_*") if p.is_dir()}
            == {j["folder"] for j in expected}, "Unexpected/incomplete window directories")
    audits = []
    for job in expected:
        audit = audit_window(output / job["folder"], x, y, protocol, job, manifest["identity"])
        audits.append(audit)
        print(f"Validated {len(audits)}/{len(expected)} windows", flush=True)
    for t in protocol["calendar"]["lookbacks"]:
        require(sum(a["test_weeks"] for a in audits if a["lookback"] == t)
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
    R.atomic_json(output / "full_validation.json", report)
    return {k: v for k, v in report.items() if k != "audits"}


def run_full(panel, output, *, workers=None, source_upgrade_reason=None):
    started = time.perf_counter()
    protocol = read_json(panel / "protocol.json")
    configure_threads(protocol)
    R.verify_panel(panel)
    x = R.read_csv(panel / "panel/X_weekly.csv")
    manifest = initialize(panel, output, protocol, x, workers=workers,
                          source_upgrade_reason=source_upgrade_reason)
    resources = manifest["binding"]["scheduler_resources"]
    identity = manifest["identity"]
    completed, failures = [], []
    jobs = [(str(panel), str(output), job, identity) for job in manifest["jobs"]]
    with ProcessPoolExecutor(max_workers=resources["workers"]) as pool:
        pending = {pool.submit(worker, args): args[2] for args in jobs}
        while pending:
            done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                try:
                    completed.append(future.result())
                except Exception:
                    failures.append({"job": job, "traceback": traceback.format_exc()})
                print(f"Full run: {len(completed)}/{len(jobs)} complete, {len(failures)} failed; "
                      f"{time.perf_counter()-started:.0f}s", flush=True)
            R.atomic_json(output / "progress.json", {"identity": identity,
                "completed": completed, "failures": failures, "pending": list(pending.values()),
                "wall_seconds": time.perf_counter()-started})
    require(not failures, "Unresolved windows block finalization; see progress.json")
    result = finalize(panel, output)
    R.atomic_json(output / "full_run_receipt.json", {"identity": identity, "status": "complete",
                  "wall_seconds_this_invocation": time.perf_counter()-started,
                  "validation_sha256": R.sha256(output / "full_validation.json"),
                  "jobs": len(jobs), "workers": resources["workers"]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full-run", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--workers", type=int, default=None,
                        help="Compute-only override; default is the frozen two-worker recipe")
    parser.add_argument("--source-upgrade-reason", default=None,
                        help="Explicitly bind a fresh run to changed package sources, preserving "
                             "old/new hashes; never reuses historical checkpoints")
    parser.add_argument("--panel-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.finalize and not args.require_complete:
        parser.error("--finalize requires --require-complete")
    if args.finalize and args.workers is not None:
        parser.error("Finalization reads the execution resources from the saved manifest")
    if args.finalize and args.source_upgrade_reason is not None:
        parser.error("Finalization reads the source-upgrade record from the saved manifest")
    output = R.external_path(args.out)
    with exclusive_run(output):
        result = (run_full(args.panel_run, output, workers=args.workers,
                          source_upgrade_reason=args.source_upgrade_reason) if args.full_run else
                  finalize(args.panel_run, output, require_complete=args.require_complete))
    print(R.json_bytes(R.safe_json(result)).decode(), flush=True)


if __name__ == "__main__":
    main()
