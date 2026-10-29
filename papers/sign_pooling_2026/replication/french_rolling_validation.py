"""Chronological estimation engine and measured first-origin French pilot.

Reads the immutable Stage 1 panel. Selection receives only an outer training
window; every inner transformation is fitted afresh on an earlier prefix.
The pilot covers origin 104 for both lookbacks, all methods and the prescribed
one-pass refinements. Outer scores are saved for audit, not printed or used to
modify the protocol. This module does not launch the full empirical study.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

from factorlasso import LassoModel

if __package__:
    from . import french_estimators as F
    from .french_panel import read_csv, verify as verify_panel
    from .french_protocol import (REPOSITORY, candidate_inventory, external_path,
                                  json_bytes, sha256)
    from .computational_validation import peak_working_set_mb
else:
    import french_estimators as F
    from french_panel import read_csv, verify as verify_panel
    from french_protocol import (REPOSITORY, candidate_inventory, external_path,
                                 json_bytes, sha256)
    from computational_validation import peak_working_set_mb


def safe_json(value):
    """Represent infinite endpoints explicitly without nonstandard JSON values."""
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            raise ValueError("Unexpected NaN in a result receipt")
        return "inf" if value > 0 else "-inf"
    return value


def digest(value) -> str:
    return hashlib.sha256(json_bytes(safe_json(value))).hexdigest()


def atomic_write(path: Path, content: bytes):
    """Atomic checkpoint replacement; callers validate identity before resuming."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("xb") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value):
    atomic_write(path, json_bytes(safe_json(value)))


def atomic_npz(path: Path, arrays: dict):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    atomic_write(path, buffer.getvalue())


def candidates(protocol: dict, method: str, *, extension=None) -> list[dict]:
    """Enumerate only applicable axes and only the permitted triggered extensions."""
    inventory = candidate_inventory(protocol, extended=bool(extension))[method]["candidates"]
    if not extension:
        return inventory
    refinements = protocol["grid"]["refinement"]
    partitions = protocol["clustering"]["partitions"] + (
        refinements["cuts_if_lowest_or_highest_finite_wins"] if extension["partition"] else [])
    taus = protocol["gate"]["thresholds"] + (
        refinements["thresholds_if_highest_finite_wins"] if extension["tau"] else [])
    fractions = protocol["grid"]["fractions"] + (
        refinements["fractions_if_smallest_positive_wins"] if extension["fraction"] else [])
    return [c for c in inventory if (c.get("partition") is None or c["partition"] in partitions)
            and (c.get("tau") is None or c["tau"] in taus)
            and ("fraction" not in c or c["fraction"] in fractions)]


def choose(rows: list[dict], protocol: dict) -> tuple[dict, list[dict]]:
    """The inventory's order supplies the frozen tie order."""
    scores = np.asarray([r["score"] for r in rows])
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Every candidate requires a finite score")
    minimum = float(scores.min())
    tol = protocol["selection"]["tie_atol"] + protocol["selection"]["tie_rtol"]*abs(minimum)
    winners = [r for r in rows if r["score"] <= minimum+tol]
    return winners[0], winners


def refinement_for(rows: list[dict], protocol: dict) -> dict:
    method = rows[0]["candidate"]["method"]
    if method in ("ols", "ridge") or protocol["grid"]["refinement"]["maximum_passes"] == 0:
        return {"fraction": False, "partition": False, "tau": False}
    _, winners = choose(rows, protocol)
    smallest = min(v for v in protocol["grid"]["fractions"] if v > 0)
    finite_cuts = [v for v in protocol["clustering"]["partitions"] if isinstance(v, (int, float))]
    finite_taus = [v for v in protocol["gate"]["thresholds"] if isinstance(v, (int, float))]
    return {
        "fraction": any(r["candidate"].get("fraction") == smallest for r in winners),
        "partition": any(r["candidate"].get("partition") in (min(finite_cuts), max(finite_cuts))
                         for r in winners),
        "tau": any(r["candidate"].get("tau") == max(finite_taus) for r in winners),
    }


class Evaluator:
    """Cache only identical training objectives; save one authenticated fit per key."""

    def __init__(self, state, protocol, folder=None, identity="test"):
        self.state, self.protocol = state, protocol
        self.folder, self.identity = folder, identity
        self.state_hash = digest(state.manifest())
        self.specs, self.results = {}, {}

    def spec(self, candidate):
        key = (candidate["method"], candidate.get("partition"), candidate.get("tau"))
        if key not in self.specs:
            self.specs[key] = F.specification(self.state, candidate)
        return self.specs[key]

    def fit_key(self, spec, lam):
        # At lambda=0 group weights/geometry drop out of the direct regression objective.
        geometry = ("least_squares" + F.array_hash(np.nan_to_num(spec.signs, nan=2.))
                    if lam == 0 and spec.family not in ("unilasso", "ridge") else spec.key())
        return digest({"state": self.state_hash, "objective": geometry, "lambda": float(lam).hex()})

    def _load(self, key):
        if self.folder is None or not (self.folder / f"{key}.npz").is_file():
            return None
        with np.load(self.folder / f"{key}.npz", allow_pickle=False) as saved:
            meta = json.loads(saved["metadata"].tobytes().decode())
            beta, alpha = saved["beta"].copy(), saved["alpha"].copy()
        if (meta["identity"] != self.identity or meta["fit_key"] != key or
                meta["state_hash"] != self.state_hash or
                meta["beta_hash"] != F.array_hash(beta) or
                meta["alpha_hash"] != F.array_hash(alpha)):
            raise ValueError("Checkpoint identity or coefficient hash mismatch")
        result = meta["result"]
        result["lambda"] = np.inf if result["lambda"] == "inf" else float(result["lambda"])
        return {**result, "beta": beta, "alpha": alpha}

    def _save(self, key, result):
        if self.folder is None:
            return
        meta = {"identity": self.identity, "fit_key": key, "state_hash": self.state_hash,
                "beta_hash": F.array_hash(result["beta"]),
                "alpha_hash": F.array_hash(result["alpha"]),
                "result": {k: v for k, v in result.items() if k not in ("beta", "alpha")}}
        atomic_npz(self.folder / f"{key}.npz", {"beta": result["beta"], "alpha": result["alpha"],
                   "metadata": np.frombuffer(json_bytes(safe_json(meta)), dtype=np.uint8)})

    def evaluate(self, choices, *, numerical_lambdas=None):
        descriptors, pending = [], {}
        for i, candidate in enumerate(choices):
            spec = self.spec(candidate)
            lam = (numerical_lambdas[i] if numerical_lambdas is not None else
                   (np.inf if candidate.get("ridge_lambda") == "inf" else
                    float(candidate.get("ridge_lambda", candidate.get("fraction", 0)*spec.scale))))
            key = self.fit_key(spec, lam)
            descriptors.append((candidate, spec, lam, key))
            if key not in self.results:
                loaded = self._load(key)
                if loaded is not None:
                    self.results[key] = loaded
                elif key not in pending:
                    pending[key] = (spec, lam)
        paths = {}
        for key, (spec, lam) in pending.items():
            paths.setdefault(spec.key(), []).append((key, spec, lam))
        for path in paths.values():
            path.sort(key=lambda v: -v[2])
            def accept(index, result):
                key = path[index][0]
                self._save(key, result)
                self.results[key] = result
            try:
                F.fit_path(self.state, path[0][1], [v[2] for v in path], self.protocol,
                           on_result=accept)
            except Exception as exc:
                if self.folder is not None:
                    atomic_json(self.folder.parent / f"failure_{path[0][0]}.json",
                                {"identity": self.identity, "state": self.state.manifest(),
                                 "path_keys": [v[0] for v in path], "error": repr(exc),
                                 "attempts": getattr(exc, "attempts", [])})
                raise
        return [(candidate, spec, self.results[key], key)
                for candidate, spec, _, key in descriptors]


def evaluate_surface(evaluators, validations, choices, protocol):
    """Return pooled inner loss and a complete candidate-to-fit alias ledger."""
    totals = {digest(c): 0. for c in choices}
    count = sum(y.size for _, y in validations)
    ledger = []
    for fold, (evaluator, (xv, yv)) in enumerate(zip(evaluators, validations)):
        state = evaluator.state
        xz = (xv.to_numpy()-state.x_mean)/state.x_sd
        yz = (yv.to_numpy()-state.y_mean)/state.y_sd
        for candidate, spec, fitted, key in evaluator.evaluate(choices):
            prediction = xz@fitted["beta"].T+fitted["alpha"]
            sse = float(np.square(yz-prediction).sum())
            totals[digest(candidate)] += sse
            ledger.append({"candidate": candidate, "candidate_id": digest(candidate),
                           "inner_fold": fold, "fit_key": key,
                           "train_dates": state.dates,
                           "validation_dates": xv.index.strftime("%Y-%m-%d").tolist(),
                           "partition_hash": F.array_hash(spec.labels),
                           "gate_hash": F.array_hash(np.nan_to_num(spec.signs, nan=2.)),
                           "lambda_max": spec.scale, "lambda": fitted["lambda"],
                           "sse": sse, "validation_cells": yv.size,
                           "status": fitted["status"], "kind": fitted["kind"],
                           "objective": fitted["objective"],
                           "constraint_violation": fitted["constraint_violation"],
                           "kkt_residual": fitted["kkt_residual"],
                           "fit_seconds": fitted["seconds"], "attempts": fitted["attempts"]})
    rows = [{"candidate": c, "candidate_id": digest(c), "score": totals[digest(c)]/count}
            for c in choices]
    return rows, ledger


def select_window(x_train, y_train, protocol, lookback, *, folder=None, identity="test"):
    """Selection API deliberately cannot receive outer test data."""
    ends = protocol["calendar"]["inner_train_ends"][str(lookback)]
    val_ends = protocol["calendar"]["inner_validation_ends"][str(lookback)]
    evaluators, validations = [], []
    for fold, (end, val_end) in enumerate(zip(ends, val_ends)):
        if not 5 <= end < val_end <= len(x_train):
            raise ValueError("Invalid chronological inner split")
        state = F.prepare(x_train.iloc[:end], y_train.iloc[:end])
        if folder is not None:
            atomic_json(folder / f"inner_{fold}_preprocessing.json", state.manifest())
        evaluators.append(Evaluator(state, protocol,
                          folder / f"inner_{fold}_fits" if folder else None, identity))
        validations.append((x_train.iloc[end:val_end], y_train.iloc[end:val_end]))
    all_rows, all_ledger, selected, extensions = {}, [], {}, {}
    for method in protocol["methods"]:
        base_choices = candidates(protocol, method)
        rows, ledger = evaluate_surface(evaluators, validations, base_choices, protocol)
        extension = refinement_for(rows, protocol)
        if any(extension.values()):
            full_choices = candidates(protocol, method, extension=extension)
            rows, ledger = evaluate_surface(evaluators, validations, full_choices, protocol)
        winner, _ = choose(rows, protocol)
        selected[method] = winner
        all_rows[method] = rows
        all_ledger.extend(ledger)
        extensions[method] = extension
        if folder is not None:
            atomic_json(folder / f"{method}_inner_surface.json", {"rows": rows, "ledger": ledger})
        print(f"T{lookback} {method}: {len(rows)} candidates validated", flush=True)
    # Final ledger has each applicable candidate/fold once, even after refinement.
    ids = [(r["candidate_id"], r["inner_fold"]) for r in all_ledger]
    expected = sum(len(rows) for rows in all_rows.values())*len(evaluators)
    if len(ids) != len(set(ids)) or len(ids) != expected:
        raise ValueError("Duplicate or incomplete inner candidate ledger")
    return {"selected": selected, "surfaces": all_rows, "ledger": all_ledger,
            "extensions": extensions, "evaluators": evaluators}


def training_diagnostics(state, spec):
    """Descriptive marginal-residual dependence; no reference-distribution claim."""
    rows = []
    for g in F.group_indices(spec.labels):
        slopes = np.mean(state.y[:, g], axis=1)@state.x / np.sum(state.x**2, axis=0)
        for j, factor in enumerate(state.factors):
            residual = state.y[:, g]-state.x[:, j, None]*slopes[j]
            scores = state.x[:, j]*residual.sum(axis=1)
            scores = scores-scores.mean()
            denom = float(scores@scores)
            acf = {str(lag): float(scores[:-lag]@scores[lag:]/denom)
                   if lag < len(scores) and denom > 0 else None for lag in (1, 4, 13)}
            sd = residual.std(axis=0)
            good = sd > 1e-12
            if good.sum() >= 2:
                corr = np.corrcoef(residual[:, good], rowvar=False)
                correlation = float(corr[np.triu_indices_from(corr, 1)].mean())
            else:
                correlation = None
            rows.append({"cluster": int(spec.labels[g[0]]), "members": len(g), "factor": factor,
                         "marginal_score_acf": acf,
                         "mean_marginal_residual_correlation": correlation})
    return {"clusters": len(np.unique(spec.labels)),
            "singleton_response_fraction": float(np.mean(
                [np.sum(spec.labels == v) == 1 for v in spec.labels])), "by_cluster_factor": rows}


def cost_counts(evaluators):
    results = [r for e in evaluators for r in e.results.values()]
    return {"unique_fits": len(results),
            "conic_fits": sum(r["kind"] == "conic" for r in results),
            "conic_attempts": sum(len(r["attempts"]) for r in results),
            "retry_attempts": sum(max(len(r["attempts"])-1, 0) for r in results),
            "independently_resolved_inaccurate_fits": sum(r["status"] == "resolved_inaccurate"
                                                          for r in results),
            "numerically_refined_fits": sum(any(a.get("numerical_refinement") is not None
                                                for a in r["attempts"]) for r in results),
            "refinement_iterations": sum(a["numerical_refinement"]["steps"] for r in results
                                         for a in r["attempts"] if a.get("numerical_refinement")),
            "package_ridge_fits": sum(r["kind"] == "package_ridge" for r in results),
            "direct_or_analytic_fits": sum(r["kind"] not in ("conic", "package_ridge")
                                           for r in results),
            "sum_unique_fit_seconds": sum(r["seconds"] for r in results),
            "max_kkt_residual": max((r["kkt_residual"] for r in results), default=0.),
            "max_constraint_violation": max((r["constraint_violation"] for r in results),
                                            default=0.)}


def run_window(x, y, origin, lookback, protocol, *, folder=None, identity="test"):
    """Tune on earlier rows, refit, then produce frozen-block predictions."""
    if origin < lookback or origin >= len(x) or not x.index.equals(y.index):
        raise ValueError("Invalid outer calendar")
    started = time.perf_counter()
    end = min(origin+protocol["calendar"]["outer_horizon"], len(x))
    job_key = digest({"origin": origin, "lookback": lookback, "protocol": digest(protocol),
                      "x_window": F.array_hash(x.iloc[origin-lookback:end].to_numpy()),
                      "y_window": F.array_hash(y.iloc[origin-lookback:end].to_numpy())})
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
        complete = folder / "window_receipt.json"
        if complete.is_file():
            receipt = json.loads(complete.read_text(encoding="utf-8"))
            if receipt["identity"] != identity or receipt.get("job_key") != job_key:
                raise ValueError("Stale outer checkpoint identity")
            for name, expected in receipt["files_sha256"].items():
                if sha256(folder / name) != expected:
                    raise ValueError(f"Corrupted completed window: {name}")
            return receipt
    xx, yy = x.iloc[origin-lookback:origin], y.iloc[origin-lookback:origin]
    selection = select_window(xx, yy, protocol, lookback, folder=folder, identity=identity)
    state = F.prepare(xx, yy)
    outer = Evaluator(state, protocol, folder / "outer_fits" if folder else None, identity)
    if folder is not None:
        atomic_json(folder / "outer_preprocessing.json", state.manifest())
    choices = [selection["selected"][method]["candidate"] for method in protocol["methods"]]
    fitted = outer.evaluate(choices)
    arrays, selections, diagnostics = {}, {}, {}
    for candidate, spec, result, key in fitted:
        method = candidate["method"]
        beta, intercept = F.original_units(state, result)
        predicted = x.iloc[origin:end].to_numpy()@beta.T+intercept
        reconstructed = (((x.iloc[origin:end].to_numpy()-state.x_mean)/state.x_sd) @
                         result["beta"].T+result["alpha"])*state.y_sd+state.y_mean
        if not np.allclose(predicted, reconstructed, atol=1e-12, rtol=1e-12):
            raise ValueError("Intercept/unscaling identity failed")
        arrays.update({method+"__beta": beta, method+"__intercept": intercept,
                       method+"__beta_standardized": result["beta"],
                       method+"__alpha_standardized": result["alpha"],
                       method+"__prediction": predicted, method+"__labels": spec.labels,
                       method+"__gates": spec.signs})
        selections[method] = {**selection["selected"][method], "lambda": result["lambda"],
                              "lambda_max": spec.scale, "fit_key": key,
                              "clusters": len(np.unique(spec.labels)),
                              "extension": selection["extensions"][method],
                              "objective": result["objective"],
                              "kkt_residual": result["kkt_residual"],
                              "constraint_violation": result["constraint_violation"]}
        if candidate.get("partition") is not None or candidate.get("tau") is not None:
            diagnostics[method] = training_diagnostics(state, spec)
    if "hcgl_unconstrained" in selections:
        matched = []
        part = selections["hcgl_unconstrained"]["candidate"]["partition"]
        for method in ("hcgl_singleton_exclude", "hcgl_exclude", "hcgl_unrestricted"):
            matched.append({"method": method, "partition": part, "tau": 2, "fraction": 0})
        common_lambda = selections["hcgl_unconstrained"]["lambda"]
        for candidate, spec, result, _ in outer.evaluate(
                matched, numerical_lambdas=[common_lambda]*len(matched)):
            beta, intercept = F.original_units(state, result)
            name = "matched__"+candidate["method"]
            arrays[name+"__beta"] = beta
            arrays[name+"__prediction"] = x.iloc[origin:end].to_numpy()@beta.T+intercept
            arrays[name+"__gates"] = spec.signs
    # Outcomes are first accessed here, after all selection/refit decisions.
    truth = y.iloc[origin:end].to_numpy()
    arrays["actual"] = truth
    arrays["training_mean"] = state.y_mean
    arrays["training_sd"] = state.y_sd
    outer_metrics = {}
    for name in [m for m in protocol["methods"]] + ["matched__"+m for m in
            ("hcgl_singleton_exclude", "hcgl_exclude", "hcgl_unrestricted")
            if "hcgl_unconstrained" in selections]:
        error = truth-arrays[name+"__prediction"]
        losses = np.square(error/state.y_sd).mean(axis=1)
        arrays[name+"__weekly_standardized_loss"] = losses
        outer_metrics[name] = {"standardized_mse": float(losses.mean()),
                               "standardized_rmse": float(np.sqrt(losses.mean())),
                               "raw_rmse": float(np.sqrt(np.square(error).mean()))}
    counts = cost_counts(selection["evaluators"]+[outer])
    receipt = {"status": "complete", "identity": identity, "job_key": job_key, "origin": origin,
               "lookback": lookback, "train_dates": state.dates,
               "test_dates": x.index[origin:end].strftime("%Y-%m-%d").tolist(),
               "methods": list(selections), "logical_inner_evaluations": len(selection["ledger"]),
               "counts": counts, "elapsed_seconds": time.perf_counter()-started,
               "compute_seconds_before_export": time.perf_counter()-started,
               "peak_working_set_mib": peak_working_set_mb(),
               "performance_withheld_from_protocol_selection": True,
               "all_selected_predictions_finite": all(np.isfinite(v).all()
                                                       for k, v in arrays.items()
                                                       if "gates" not in k),
               "files_sha256": {}}
    if folder is not None:
        atomic_npz(folder / "outer_arrays.npz", arrays)
        atomic_json(folder / "selections.json", selections)
        atomic_json(folder / "training_diagnostics.json", diagnostics)
        atomic_json(folder / "outer_metrics_audit_only.json", outer_metrics)
        atomic_json(folder / "inner_ledger.json", selection["ledger"])
        receipt["files_sha256"] = {p.relative_to(folder).as_posix(): sha256(p)
                                   for p in sorted(folder.rglob("*")) if p.is_file()
                                   and p.name != "window_receipt.json"}
        receipt["elapsed_seconds"] = time.perf_counter()-started
        receipt["peak_working_set_mib"] = peak_working_set_mb()
        atomic_json(folder / "window_receipt.json", receipt)
    else:
        receipt.update({"selections": selections, "arrays": arrays,
                        "preprocessing": state.manifest()})
    return receipt


def execution_identity(panel_run, protocol):
    """Bind a new engine separately from the Stage 1 source freeze."""
    files = [Path(__file__), Path(F.__file__), Path(F.unilasso_features.__code__.co_filename),
             Path(inspect.getsourcefile(inspect.unwrap(F.safe_cvxpy_sum_shape))),
             Path(candidate_inventory.__code__.co_filename), Path(read_csv.__code__.co_filename),
             Path(peak_working_set_mb.__code__.co_filename)]
    files += sorted((REPOSITORY / "src/factorlasso").rglob("*.py"))
    sources = {p.relative_to(REPOSITORY).as_posix(): sha256(p) for p in files}
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "pandas", "scipy", "cvxpy", "clarabel", "factorlasso")}
    defaults = {name: inspect.signature(LassoModel).parameters[name].default
                for name in protocol["package_defaults"]}
    if defaults != protocol["package_defaults"]:
        raise ValueError("Estimator defaults differ from frozen protocol")
    return {"panel_manifest_sha256": sha256(panel_run / "manifest.json"),
            "protocol_sha256": sha256(panel_run / "protocol.json"),
            "source_sha256": sources, "runtime_versions": versions,
            "python": sys.version, "platform": platform.platform(),
            "defaults_observed": defaults, "kkt_check_tolerance": F.KKT_TOL,
            "objective_check_tolerance": F.OBJECTIVE_TOL,
            "conic_initialization": "cold numerical starts; DPP canonicalization cache retained",
            "numerical_refinement": {"tolerance": F.POLISH_TOL,
                                     "max_steps": F.POLISH_MAX_STEPS,
                                     "objective": "unchanged; recorded per conic attempt"},
            "inaccurate_resolution": {"requires_prescribed_retry": True,
                                      "independent_kkt_tolerance": F.POLISH_TOL,
                                      "independent_relative_dual_gap": 1e-8,
                                      "retain_original_status_and_warning": True},
            "worker_threads": {name: os.environ.get(name) for name in
                               ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")}}


def pilot_worker(job):
    panel_run, output, lookback, origin, identity = job
    panel_run, output = Path(panel_run), Path(output)
    protocol = json.loads((panel_run / "protocol.json").read_text(encoding="utf-8"))
    x, y = read_csv(panel_run / "panel/X_weekly.csv"), read_csv(panel_run / "panel/Y_weekly.csv")
    folder = output / f"origin_{origin}_T{lookback}"
    return run_window(x, y, origin, lookback, protocol, folder=folder, identity=identity)


def run_pilot(panel_run: Path, output: Path):
    """Run only the prespecified earliest origin, covering both lookbacks."""
    started = time.perf_counter()
    output = external_path(output)
    validation = verify_panel(panel_run)
    protocol = json.loads((panel_run / "protocol.json").read_text(encoding="utf-8"))
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(protocol["solver"]["blas_threads_per_worker"])
    binding = execution_identity(panel_run, protocol)
    identity = digest(binding)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "execution_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != binding:
            raise ValueError("Source/protocol/runtime changed; choose a fresh pilot directory")
    else:
        atomic_json(manifest_path, binding)
        for name in binding["source_sha256"]:
            target = output / "source" / name
            atomic_write(target, (REPOSITORY / name).read_bytes())
    origin = protocol["calendar"]["outer_burn_in"]
    jobs = [(str(panel_run), str(output), lookback, origin, identity)
            for lookback in protocol["calendar"]["lookbacks"]]
    completed = []
    try:
        with ProcessPoolExecutor(max_workers=protocol["solver"]["workers"]) as pool:
            futures = [pool.submit(pilot_worker, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                completed.append(result)
                print(f"Pilot T{result['lookback']} completed: "
                      f"{result['elapsed_seconds']:.1f}s, {result['counts']['conic_attempts']} "
                      "conic attempts", flush=True)
    except Exception as exc:
        atomic_json(output / "pilot_failure.json", {"identity": identity, "error": repr(exc),
                                                    "completed_lookbacks": [r["lookback"]
                                                                           for r in completed]})
        raise
    if {r["lookback"] for r in completed} != set(protocol["calendar"]["lookbacks"]):
        raise ValueError("Incomplete pilot")
    summed_seconds = sum(r["elapsed_seconds"] for r in completed)
    workers = protocol["solver"]["workers"]
    nominal = summed_seconds*protocol["calendar"]["origins"]/workers/3600
    report = {"status": "complete", "identity": identity, "stage1_validation": validation,
              "pilot_origin": origin, "lookbacks": sorted(r["lookback"] for r in completed),
              "wall_seconds_including_startup": time.perf_counter()-started,
              "windows": sorted(completed, key=lambda r: r["lookback"]),
              "full_run_cost": {"workers": workers, "origins_per_lookback": 72,
                                "nominal_wall_hours_from_measured_windows": nominal,
                                "twofold_runtime_scenario_hours": 2*nominal,
                                "fourfold_runtime_scenario_hours": 4*nominal,
                                "scope": "single origin; later regimes/partitions may cost more",
                                "assumptions": "same host, two workers, one BLAS thread each",
                                "outer_metrics_used_for_cost_or_protocol": False},
              "estimator_defaults_unchanged": True,
              "outer_scores": "saved in audit-only files; not printed or used to alter protocol"}
    atomic_json(output / "pilot_report.json", report)
    return {k: report[k] for k in ("status", "lookbacks", "wall_seconds_including_startup",
                                  "full_run_cost", "estimator_defaults_unchanged")}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pilot", action="store_true", required=True)
    parser.add_argument("--panel-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json_bytes(safe_json(run_pilot(args.panel_run, args.out))).decode())


if __name__ == "__main__":
    main()
