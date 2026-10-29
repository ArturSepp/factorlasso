"""Execute the source-bound Stage 1B ETF/MATF pilot, with strict per-dataset receipts.

No estimator or scientific protocol is changed here. Training-only folds select
the ungated sparse penalty shared by gate arms; ridge alone has the frozen upper
grid extension. Scores are annual additive premium errors, never realised CAGR.
Outputs and resume checkpoints must be outside OneDrive.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

if __package__:
    from . import matf_comparators as M, matf_pilot_protocol as P
    from .verify_matf_comparators import reference_fit
    from .french_attribution import require, read, write, sha, digest, external, REPO
else:
    import matf_comparators as M
    import matf_pilot_protocol as P
    from verify_matf_comparators import reference_fit
    from french_attribution import require, read, write, sha, digest, external, REPO

HERE = Path(__file__).resolve().parent
METHOD_KEYS = ["cell", "family", "span", "arm", "soft_strength", "zero_strength"]


def identifier(row):
    return digest(row)[:24]


def dataset_key(cell, replication):
    return f"{cell}__{replication:03d}"


def choose(ledger, grid):
    """Equally weighted complete prefix folds; ties favour larger penalty."""
    scores = {}
    for v in grid:
        rows = [r for r in ledger if r["candidate"] == v]
        require(
            len(rows) == 3 and len({r["train_rows"] for r in rows}) == 3,
            "Missing or duplicate tuning fold",
        )
        scores[v] = float(np.mean([r["mse"] for r in rows]))
    require(np.isfinite(list(scores.values())).all(), "Nonfinite tuning score")
    minimum = min(scores.values())
    return max(v for v in grid if scores[v] <= minimum + 1e-12), scores


def measured_fit(data, spec, lam, **kwargs):
    """Retain every solver attempt and warning; independently audit ridge stationarity."""
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = M.ridge_fit(data, lam) if spec is None else M.fit(data, spec, lam, **kwargs)
        except RuntimeError as error:
            if spec is None or not str(error).startswith("Unresolved comparator fit:"):
                raise
            result = recover_fit(data, spec, lam, error, **kwargs)
    require(
        not any(issubclass(w.category, RuntimeWarning) for w in caught),
        "RuntimeWarning in fitted comparator",
    )
    result["outer_warnings"] = [str(w.message) for w in caught]
    result["elapsed_seconds"] = time.perf_counter() - start
    require(np.isfinite(result["beta"]).all(), "Nonfinite fitted beta")
    if spec is None:
        x, y = data.weighted.x, data.weighted.y
        gradient = 2 * (
            result["beta"] @ (x.T @ x / len(x)) - y.T @ x / len(x) + lam * result["beta"]
        )
        result["kkt_residual"] = float(abs(gradient).max() / (1 + abs(result["beta"]).max()))
        result["constraint_violation"] = 0.0
        result["attempts"] = []
        require(result["kkt_residual"] <= 1e-7, "Ridge stationarity failed")
    return result


def recover_fit(data, spec, lam, original_error, **kwargs):
    """Recorded numerical amendment: two independent solvers for the identical objective.

    Invoked only after both frozen CLARABEL attempts fail acceptance. The Stage
    1B sources/protocol remain immutable; this execution amendment and failed
    attempt ledger are bound separately before rerunning the complete pilot.
    """
    hard = kwargs.get("hard", np.full_like(spec.signs, np.nan))
    soft = kwargs.get("soft", np.full_like(spec.signs, np.nan))
    eta, eta0 = kwargs.get("strength", 0.0), kwargs.get("zero_strength", 0.0)
    restricted = replace(spec, signs=hard)
    beta, refinement = M.refine(data, restricted, np.zeros_like(spec.signs), lam, soft, eta, eta0)
    checks = M.audit(data, restricted, beta, lam, soft, eta, eta0)
    reference, steps = reference_fit(
        data.state.x,
        data.state.y,
        data.weights,
        spec.labels,
        "hcgl" if spec.family == "row" else "fcgl",
        lam,
        hard,
        soft,
        eta,
        eta0,
    )
    reference_check = M.audit(data, restricted, reference, lam, soft, eta, eta0)
    discrepancy = float(abs(reference - beta).max())
    objective_gap = abs(checks["objective"] - reference_check["objective"]) / (
        1 + abs(reference_check["objective"])
    )
    require(
        checks["kkt_residual"] <= 1e-7
        and checks["constraint_violation"] <= 1e-7
        and reference_check["kkt_residual"] <= 1e-7
        and discrepancy <= 3e-6
        and objective_gap <= 1e-8,
        "Independent numerical recovery failed",
    )
    try:
        attempts = ast.literal_eval(str(original_error).split(": ", 1)[1])
    except (ValueError, SyntaxError):
        attempts = [dict(status="unresolved_original", warnings=[], error=str(original_error))]
    return dict(
        beta=beta,
        alpha=np.zeros(len(beta)),
        status="independent_fista_recovery",
        attempts=attempts,
        original_failure=str(original_error),
        **checks,
        recovery=dict(
            refinement=refinement,
            reference_steps=steps,
            reference_coefficient_error=discrepancy,
            reference_relative_objective_gap=objective_gap,
            reference_audit=reference_check,
        ),
    )


def store_fit(archive, key, data, result, prediction_x):
    beta, alpha = M.F.original_units(data.state, result)
    archive[key + "_beta_std"] = result["beta"]
    archive[key + "_beta"] = beta
    archive[key + "_alpha"] = alpha
    archive[key + "_prediction"] = prediction_x @ beta.T + alpha
    return {k: v for k, v in result.items() if k not in ("beta", "alpha")}


def store_training(archive, key, data, spec):
    for name in ("x_mean", "x_sd", "y_mean", "y_sd"):
        archive[key + "_" + name] = getattr(data.state, name)
    archive[key + "_weights"] = data.weights
    if spec is not None:
        archive[key + "_labels"] = spec.labels
        archive[key + "_group_weights"] = spec.weights
        archive[key + "_scale"] = np.asarray(spec.scale)


def tuning(x, y, family, span, p, archive, key):
    """Cache every candidate/fold exactly once, including boundary extensions."""
    n, step = len(x), len(x) // 6
    require(n in (60, 120), "Unexpected pilot history")
    folds = []
    for stop in (n // 2, n // 2 + step, n // 2 + 2 * step):
        data = M.prepare(x.iloc[:stop], y.iloc[:stop], span)
        spec = None if family == "ridge" else M.specification(data, family, p["partition_cut"])
        fkey = key + f"_fold{stop}"
        store_training(archive, fkey, data, spec)
        folds.append((stop, data, spec, fkey))
    ledger = []
    grid = list(p["ridge_grid"] if family == "ridge" else p["sparse_fraction_grid"])
    extensions = iter(p["ridge_upper_extensions"] if family == "ridge" else [])
    pending = grid.copy()
    while True:
        for stop, data, spec, fkey in folds:
            xx = x.iloc[stop : stop + step].to_numpy()
            yy = y.iloc[stop : stop + step].to_numpy()
            for candidate in pending:
                lam = candidate if spec is None else candidate * spec.scale
                fit_key = fkey + "_c" + str(candidate)
                result = measured_fit(data, spec, lam)
                diagnostics = store_fit(archive, fit_key, data, result, xx)
                error = (yy - archive[fit_key + "_prediction"]) / data.state.y_sd
                ledger.append(
                    dict(
                        train_rows=stop,
                        validation_rows=step,
                        candidate=candidate,
                        lambda_=lam,
                        mse=float(np.mean(error**2)),
                        key=fit_key,
                        training_key=fkey,
                        diagnostics=diagnostics,
                    )
                )
        selected, scores = choose(ledger, grid)
        next_value = next(extensions, None) if selected == max(grid) else None
        if next_value is None:
            break
        grid.append(next_value)
        pending = [next_value]
    return dict(
        family=family,
        span=span,
        selected=selected,
        grid=grid,
        scores={str(k): v for k, v in scores.items()},
        folds=ledger,
        boundary=bool(selected in (min(grid), max(grid))),
        upper_exhausted=bool(family == "ridge" and selected == 10000.0),
    )


def gate_history(x, y, span, family, kind, p, full, archive, key):
    """Save training-prefix gate crossings; partition changes are part of this diagnostic."""
    snapshots = []
    n, step = len(x), len(x) // 6
    for stop in (n // 2, n // 2 + step, n // 2 + 2 * step, n):
        data, spec = full if stop == n else (M.prepare(x.iloc[:stop], y.iloc[:stop], span), None)
        if spec is None:
            spec = M.specification(data, family, p["partition_cut"])
        labels = np.arange(y.shape[1]) if kind == "singleton" else spec.labels
        gate = M.gate(
            data,
            labels,
            "conditional" if kind == "conditional" else "marginal",
            p["gate_threshold"],
        )
        for name in ("signs", "slopes", "standard_errors", "t_stats", "labels"):
            archive[f"{key}_{stop}_{name}"] = gate[name]
        snapshots.append(gate)
    signs = np.array([g["signs"] for g in snapshots])
    crossing = float(np.mean((signs[1:] != 0) != (signs[:-1] != 0)))
    changes = float(np.mean(signs[1:] != signs[:-1]))
    return snapshots[-1], crossing, changes


def losses(beta, alpha, truth, premia, covariance, y_test, prediction, y_sd):
    """Per-asset annual additive mean losses; risk uses annual covariance, not squared means."""
    delta = beta - truth
    component = delta * premia
    mu = component.sum(axis=1)
    return dict(
        attribution_mse=np.sum(component**2, axis=1),
        mean_mse=mu**2,
        full_mean_mse=(mu + 12 * alpha) ** 2,
        alpha_mse=(12 * alpha) ** 2,
        mean_alpha_cross=2 * mu * 12 * alpha,
        annual_prediction_risk=12 * np.sum((delta @ covariance) * delta, axis=1),
        loading_mse=np.mean(delta**2, axis=1),
        test_mse=np.mean((prediction - y_test) ** 2, axis=0),
        standardized_test_mse=np.mean(((prediction - y_test) / y_sd) ** 2, axis=0),
    ), component


def run_dataset(protocol_dir, output, cell_key, replication):
    """Atomic complete unit: all frozen arms for one paired seed and condition."""
    start = time.perf_counter()
    protocol_dir, output = Path(protocol_dir), Path(output)
    p = read(protocol_dir / "protocol.json")
    planned = read(protocol_dir / "planned_fits.json")
    jobs = [j for j in planned if j["cell"] == cell_key and j["replication"] == replication]
    key = dataset_key(cell_key, replication)
    dest = output / "datasets" / key
    dest.mkdir(parents=True, exist_ok=True)
    require(not (dest / "complete.json").exists(), "Already completed dataset")
    archive, rows, selectors = {}, [], []
    try:
        with np.load(protocol_dir / "calibration.npz", allow_pickle=False) as data:
            calibration = dict(data)
        cell = next(c for c in p["planned_full_core"] if c["key"] == cell_key)
        generated = P.generate(calibration, cell, replication, p)
        archive.update(generated)
        dates = pd.date_range("2000-01-31", periods=cell["history"], freq="ME")
        x = pd.DataFrame(generated["x"], index=dates, columns=p["calibration"]["factors"])
        y = pd.DataFrame(generated["y"], index=dates, columns=p["calibration"]["assets"])
        configs = sorted({(j["family"], j["span"] or 0) for j in jobs})
        for family, raw_span in configs:
            span = raw_span or None
            config_key = f"{family}_{raw_span}"
            data = M.prepare(x, y, span)
            spec = M.specification(data, family, p["partition_cut"])
            store_training(archive, config_key, data, spec)
            selected_jobs = [j for j in jobs if j["family"] == family and j["span"] == span]
            select = tuning(x, y, family, span, p, archive, config_key)
            selectors.append(dict(key=config_key, **select))
            ridge_select = None
            if any(j["arm"] == "ridge" for j in selected_jobs):
                rkey = "ridge_" + str(raw_span)
                ridge_select = tuning(x, y, "ridge", span, p, archive, rkey)
                selectors.append(dict(key=rkey, **ridge_select))
            gates = {}
            kinds = {"marginal"}
            if any(j["arm"].startswith("conditional") for j in selected_jobs):
                kinds.add("conditional")
            if any(j["arm"].startswith("singleton") for j in selected_jobs):
                kinds.add("singleton")
            for kind in sorted(kinds):
                gates[kind] = gate_history(
                    x, y, span, family, kind, p, (data, spec), archive, config_key + "_" + kind
                )
            for job in selected_jobs:
                fkey, arm = identifier(job), job["arm"]
                hard = np.full_like(spec.signs, np.nan)
                soft = np.full_like(spec.signs, np.nan)
                used_gate = None
                if arm not in ("ridge", "ungated"):
                    kind = (
                        "conditional"
                        if arm.startswith("conditional")
                        else "singleton"
                        if arm.startswith("singleton")
                        else "marginal"
                    )
                    used_gate, crossing, changes = gates[kind]
                    if arm.startswith("soft"):
                        soft = used_gate["signs"]
                    else:
                        policy = (
                            "zero_only"
                            if arm == "pooled_zero_only"
                            else "release"
                            if arm.endswith("release")
                            else "exclude"
                        )
                        hard = M.hard_signs(used_gate["signs"], policy)
                lam = (
                    ridge_select["selected"] if arm == "ridge" else select["selected"] * spec.scale
                )
                result = measured_fit(
                    data,
                    None if arm == "ridge" else spec,
                    lam,
                    hard=hard,
                    soft=soft,
                    strength=job["soft_strength"],
                    zero_strength=job["zero_strength"],
                )
                diagnostic = store_fit(archive, fkey, data, result, generated["x_test"])
                archive[fkey + "_hard"], archive[fkey + "_soft"] = hard, soft
                beta, alpha = archive[fkey + "_beta"], archive[fkey + "_alpha"]
                per_asset, component = losses(
                    beta,
                    alpha,
                    generated["truth"],
                    generated["annual_premia"],
                    calibration["factor_covariance"],
                    generated["y_test"],
                    archive[fkey + "_prediction"],
                    data.state.y_sd,
                )
                archive[fkey + "_component_error"] = component
                for metric, value in per_asset.items():
                    archive[fkey + "_" + metric] = value
                active = generated["truth"] != 0
                retained = abs(result["beta"]) > 1e-6
                metrics = {k: float(v.mean()) for k, v in per_asset.items()}
                metrics.update(
                    retention=float(retained.mean()),
                    active_retention=float(retained[active].mean()),
                    wrong_fitted_direction=float(
                        np.mean((beta[active] * generated["truth"][active] < 0) & retained[active])
                    ),
                    false_exclusion=float(np.mean((hard == 0)[active])),
                    wrong_hard_restriction=float(
                        np.mean((hard[active] * generated["truth"][active]) < 0)
                    ),
                    cluster_ari=float(
                        adjusted_rand_score(calibration["truth_labels"], spec.labels)
                    ),
                    clusters=int(len(np.unique(spec.labels))),
                )
                if used_gate is not None:
                    signs = used_gate["signs"]
                    metrics.update(
                        gate_failed_active=float(np.mean(signs[active] == 0)),
                        gate_wrong_active=float(
                            np.mean(signs[active] * generated["truth"][active] < 0)
                        ),
                        gate_threshold_crossing=crossing,
                        gate_direction_change=changes,
                    )
                rows.append(
                    dict(
                        **job,
                        key=fkey,
                        training_key=config_key,
                        lambda_=lam,
                        selected=ridge_select["selected"] if arm == "ridge" else select["selected"],
                        metrics=metrics,
                        diagnostics=diagnostic,
                    )
                )
        require({identifier(j) for j in jobs} == {r["key"] for r in rows}, "Fit inventory differs")
        archive_path = dest / "arrays.npz"
        np.savez_compressed(archive_path, **archive)
        ledger_path = dest / "ledger.json"
        write(
            ledger_path,
            dict(
                cell=cell_key,
                replication=replication,
                selectors=selectors,
                fits=rows,
                seconds=time.perf_counter() - start,
            ),
        )
        complete = dict(
            dataset=key,
            planned_sha256=digest(jobs),
            fits=len(rows),
            inner_fits=sum(len(s["folds"]) for s in selectors),
            ledger_sha256=sha(ledger_path),
            arrays_sha256=sha(archive_path),
            run_identity=read(output / "run_freeze.json")["identity"],
        )
        write(dest / "complete.json", complete)
        return dict(**complete, seconds=time.perf_counter() - start)
    except Exception:
        np.savez_compressed(dest / "partial_arrays.npz", **archive)
        write(dest / "partial_ledger.json", dict(fits=rows, selectors=selectors))
        write(
            dest / "failure.json",
            dict(
                cell=cell_key,
                replication=replication,
                traceback=traceback.format_exc(),
                seconds=time.perf_counter() - start,
            ),
        )
        raise


def authenticate(protocol_dir):
    freeze = read(protocol_dir / "freeze.json")
    binding = freeze["binding"]
    require(digest(binding) == freeze["identity"], "Changed Stage 1B freeze")
    for name in ("protocol", "calibration", "planned_fits"):
        suffix = ".npz" if name == "calibration" else ".json"
        require(
            sha(protocol_dir / (name + suffix)) == binding[name + "_sha256"],
            "Changed frozen pilot input: " + name,
        )
    for name, expected in {
        **binding["source_sha256"],
        **binding["unchanged_estimator_sha256"],
    }.items():
        require(sha(REPO / name) == expected, "Changed Stage 1B source: " + name)
    for path, expected in binding["inputs_sha256"].items():
        require(sha(Path(path)) == expected, "Changed authenticated financial input")
    import factorlasso

    require(
        Path(factorlasso.__file__).resolve() == (REPO / "src/factorlasso/__init__.py").resolve(),
        "Unexpected factorlasso import",
    )
    return freeze


def complete_inventory(output, jobs):
    """Reject missing, extra, duplicate or corrupted output units before any aggregation."""
    expected = {dataset_key(j["cell"], j["replication"]) for j in jobs}
    directory = output / "datasets"
    actual = {p.name for p in directory.iterdir() if p.is_dir()} if directory.exists() else set()
    require(actual == expected, "Missing or extra dataset directory")
    rows = []
    identity = read(output / "run_freeze.json")["identity"]
    for key in sorted(expected):
        path = directory / key
        require((path / "complete.json").exists(), "Incomplete dataset: " + key)
        receipt = read(path / "complete.json")
        require(receipt["run_identity"] == identity, "Receipt from a different execution freeze")
        for name, ext in (("arrays", ".npz"), ("ledger", ".json")):
            require(sha(path / (name + ext)) == receipt[name + "_sha256"], "Changed result: " + key)
        data = read(path / "ledger.json")
        planned = [j for j in jobs if dataset_key(j["cell"], j["replication"]) == key]
        require(receipt["planned_sha256"] == digest(planned), "Changed dataset plan")
        rows.extend(data["fits"])
    require(len(rows) == len(jobs), "Missing or duplicate final fits")
    require({r["key"] for r in rows} == {identifier(j) for j in jobs}, "Unexpected final fits")
    return rows


def initialize(protocol_dir, output, workers):
    """Bind execution and score definitions before the first comparative fit."""
    frozen = authenticate(protocol_dir)
    sources = {
        ("papers/sign_pooling_2026/replication/" + name): sha(HERE / name)
        for name in ("matf_pilot_run.py", "verify_matf_pilot.py", "tests/test_matf_pilot.py")
    }
    binding = dict(
        protocol_dir=str(protocol_dir),
        protocol_identity=frozen["identity"],
        sources=sources,
        workers=workers,
        blas_threads=1,
        python=sys.executable,
        retention_threshold_standardized=1e-6,
        score_units="Annual additive premium errors; covariance times 12; means squared",
        primary="mean_asset sum_factor ((beta_hat-beta_true)*annual_premia)^2",
        secondary="mean_asset (sum_factor ((beta_hat-beta_true)*annual_premia))^2",
        full_mean="same with 12*monthly fitted intercept added before squaring",
        gate_crossing="Mean pass/fail toggles over three successive prefix-to-full pairs",
        inference="Within-cell paired seeds; 1.96 normal MC bands descriptive only",
        scheduling="One process per condition/seed; atomic complete receipts; stop failures",
        numerical_amendment=dict(
            reason="CLARABEL attempts failed in run01; preserve it and rerun all fits",
            scientific_changes="None: same inputs, seeds, objectives, gates and grids",
            recovery="After both failed attempts only: frozen FISTA refinement from zero and "
            "independent Gram/prox optimizer; retain original failed attempt ledger",
            max_kkt=1e-7,
            max_constraint_violation=1e-7,
            max_independent_coefficient_discrepancy=3e-6,
            max_relative_objective_discrepancy=1e-8,
        ),
    )
    target = output / "run_freeze.json"
    if target.exists():
        old = read(target)
        require(
            old["binding"] == binding and old["identity"] == digest(binding),
            "Changed execution source/protocol; use a fresh run directory",
        )
    else:
        require(
            not output.exists() or not any(output.iterdir()), "Fresh execution directory required"
        )
        output.mkdir(parents=True, exist_ok=True)
        write(
            target,
            dict(
                identity=digest(binding),
                binding=binding,
                created_utc=datetime.now(timezone.utc).isoformat(),
            ),
        )
        for name in sources:
            dest = output / "source" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, dest)
    return binding


def run(protocol_dir, output, workers=4, limit=None):
    protocol_dir, output = Path(protocol_dir).resolve(), external(output)
    require(1 <= workers <= 8, "Use one to eight workers")
    for env in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        require(os.environ.get(env) == "1", "Set " + env + "=1 before starting Python")
    initialize(protocol_dir, output, workers)
    jobs = read(protocol_dir / "planned_fits.json")
    datasets = sorted({(j["cell"], j["replication"]) for j in jobs})
    pending = []
    for cell, replication in datasets:
        dest = output / "datasets" / dataset_key(cell, replication)
        if (dest / "complete.json").exists():
            receipt = read(dest / "complete.json")
            require(
                receipt["run_identity"] == read(output / "run_freeze.json")["identity"],
                "Invalid resume identity",
            )
            require(
                sha(dest / "ledger.json") == receipt["ledger_sha256"]
                and sha(dest / "arrays.npz") == receipt["arrays_sha256"],
                "Corrupt checkpoint",
            )
        else:
            require(
                not (dest / "failure.json").exists(),
                "Unresolved previous failure; preserve it and investigate before a new run",
            )
            pending.append((cell, replication))
    if limit is not None:
        pending = pending[:limit]
    started = time.perf_counter()
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_dataset, str(protocol_dir), str(output), c, r): (c, r)
            for c, r in pending
        }
        try:
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                print(
                    json.dumps(dict(completed=completed, scheduled=len(pending), **result)),
                    flush=True,
                )
        except Exception:
            for f in futures:
                f.cancel()
            raise
    elapsed = time.perf_counter() - started
    sessions = output / "sessions"
    sessions.mkdir(exist_ok=True)
    write(
        sessions / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json"),
        dict(scheduled=len(pending), completed=completed, wall_seconds=elapsed, workers=workers),
    )
    if all(
        (output / "datasets" / dataset_key(c, r) / "complete.json").exists() for c, r in datasets
    ):
        rows = complete_inventory(output, jobs)
        write(
            output / "execution_complete.json",
            dict(
                status="complete; independent audit pending",
                datasets=len(datasets),
                fits=len(rows),
                run_identity=read(output / "run_freeze.json")["identity"],
            ),
        )
    return dict(scheduled=len(pending), completed=completed, wall_seconds=elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.protocol, args.out, args.workers, args.limit), indent=2))
