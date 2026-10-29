"""Freeze and execute all 16 ETF/MATF cells with 200 fresh paired replications.

All regularized procedures select their own penalty on identical chronological
folds. Matched-penalty gate records are separate diagnostics. Frozen pilot sources,
calibration and production defaults are never edited. Saved coefficient paths,
transformations and deterministic draws reconstruct predictions without a large
redundant prediction archive. No failed dataset may enter aggregate results.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd

if __package__:
    from . import matf_full_engine as E, matf_pilot_protocol as P
    from .matf_pilot_run import authenticate, losses
    from .french_attribution import require, read, write, sha, digest, external, REPO
else:
    import matf_full_engine as E
    import matf_pilot_protocol as P
    from matf_pilot_run import authenticate, losses
    from french_attribution import require, read, write, sha, digest, external, REPO

HERE = Path(__file__).resolve().parent


def configs(p, cell):
    return E.methods(cell, p["diagnostic_cells"])


def record_inventory(p, cell):
    rows = []
    for method in configs(p, cell):
        rows.append(dict(method=method["name"], tuning="own"))
        if method["family"] in ("hcgl", "fcgl") and method["gate"] != "ungated":
            rows.append(dict(method=method["name"], tuning="matched"))
    return rows


def freeze(pilot_dir, output, validation, workers=8):
    output = external(output)
    require(not output.exists() or not any(output.iterdir()), "Use a fresh full-study directory")
    parent = authenticate(pilot_dir)
    old = read(pilot_dir / "protocol.json")
    v = read(validation / "validation.json")
    require(v["status"] == "passed", "Independent implementation validation required")
    require(sha(validation / "cases.json") == v["cases_sha256"], "Changed validation cases")
    for name, h in v["sources"].items():
        require(sha(REPO / name) == h, "Unvalidated full-study source")
    require(
        shutil.disk_usage(output.parent).free > 100 * 1024**3,
        "Full coefficient-path archive needs at least 100 GiB available",
    )
    generator = dict(old, pilot_seed=old["reserved_full_seed"], pilot_replications=200)
    p = dict(
        study="etf_matf_full_v1",
        replications=200,
        full_seed=old["reserved_full_seed"],
        cells=old["planned_full_core"],
        generator=generator,
        diagnostic_cells=old["pilot_cells"],
        sparse_grid=[0.0, 1e-6, 1e-5, 1e-4, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
        ridge_grid=[
            0.0,
            1e-4,
            0.001,
            0.003,
            0.01,
            0.03,
            0.1,
            0.3,
            1.0,
            3.0,
            10.0,
            30.0,
            100.0,
            1000.0,
            10000.0,
            "inf",
        ],
        ridge_extensions=[1e5, 1e6, 1e7, 1e8],
        ridge_extension_rule="If largest finite lambda wins, append next finite value; "
        "infinite zero endpoint always available; flag exhaustion",
        own_tuning="Three earlier prefix folds, standardized held-out MSE, ties larger lambda",
        matched_tuning="Same family/span ungated chosen fraction and numerical full-window lambda",
        selection_scope="Penalty selected separately for every procedure. Cut=.6, threshold=2, "
        "mixture=.5 fixed; eta=1 primary. Structural tuning is not claimed.",
        folds="60: train 30/40/50, validate next 10. 120: train 60/80/100, validate next 20.",
        factor_basis=old["calibration"]["factors"],
        assets=old["calibration"]["assets"],
        weighting="Equal throughout primary; EW span120 bounded four-cell core/ridge slice",
        units="Monthly excess-log DGP, supplied annual additive premia; primary alpha=0; no RF",
        precision="Exactly 200 draws per cell, fixed by user; no significance stopping. "
        "Report paired MCSE and 1.96 bands; across-cell summaries average within seed.",
        numerical=dict(
            primary="Cached-Gram FISTA; UniLasso positive coordinate descent; "
            "public FactorLasso ridge including zero/infinite endpoints",
            target_kkt=2e-8,
            acceptance_kkt=1e-7,
            max_iterations=20000,
            feasibility=1e-9,
            verification="Independent KKT for all paths, separate CVXPY references "
            "on fixed replication-zero final fits",
            failure="Stop unit/run; preserve failure and partial arrays; no survivors",
        ),
        storage="All inner standardized betas/UniLasso theta, transformations and evidence; "
        "all unique final coefficients and per-asset losses. Predictions reconstructed "
        "from saved coefficients and saved train/test draws. Float64 throughout.",
        dependencies="No package changes or added runtime dependencies",
        references=dict(
            unilasso="https://hdsr.mitpress.mit.edu/pub/3i97j340",
            cooperative="https://doi.org/10.1214/11-AOAS520",
        ),
        deferred=[
            "No nonzero alpha or noisy premia",
            "No serial/heavy-tail DGP",
            "No oracle/misspecified/minority-group extension",
            "No universe bootstrap",
            "No new portfolio optimization or CMA rebalancing",
            "No structural threshold/cut/mixing grid search",
        ],
    )
    expected = {c["key"]: record_inventory(p, c["key"]) for c in p["cells"]}
    output.mkdir(parents=True)
    shutil.copy2(pilot_dir / "calibration.npz", output / "calibration.npz")
    write(output / "protocol.json", p)
    write(output / "planned_records.json", expected)
    files = [
        "matf_full_engine.py",
        "matf_full_study.py",
        "verify_matf_full.py",
        "tests/test_matf_full.py",
        "matf_pilot_run.py",
        "verify_matf_pilot.py",
    ]
    sources = {("papers/sign_pooling_2026/replication/" + name): sha(HERE / name) for name in files}
    sources.update(parent["binding"]["source_sha256"])
    sources.update(parent["binding"]["unchanged_estimator_sha256"])
    binding = dict(
        protocol_sha256=sha(output / "protocol.json"),
        inventory_sha256=sha(output / "planned_records.json"),
        calibration_sha256=sha(output / "calibration.npz"),
        sources=sources,
        parent_identity=parent["identity"],
        parent_dir=str(pilot_dir),
        validation_dir=str(validation),
        validation_sha256=sha(validation / "validation.json"),
        workers=workers,
        blas_threads=1,
        python=sys.executable,
        versions={
            n: importlib.metadata.version(n)
            for n in (
                "numpy",
                "pandas",
                "scipy",
                "cvxpy",
                "clarabel",
                "scs",
                "scikit-learn",
                "factorlasso",
            )
        },
    )
    write(
        output / "freeze.json",
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
    return dict(
        identity=digest(binding),
        datasets=3200,
        final_records=sum(map(len, expected.values())) * 200,
        output=str(output),
    )


def authenticate_full(output):
    frozen = read(output / "freeze.json")
    b = frozen["binding"]
    require(digest(b) == frozen["identity"], "Changed full freeze")
    for file, key in [
        ("protocol.json", "protocol_sha256"),
        ("calibration.npz", "calibration_sha256"),
        ("planned_records.json", "inventory_sha256"),
    ]:
        require(sha(output / file) == b[key], "Changed frozen " + file)
    for name, h in b["sources"].items():
        require(sha(REPO / name) == h, "Changed frozen source " + name)
    require(
        sha(Path(b["validation_dir"]) / "validation.json") == b["validation_sha256"],
        "Changed implementation validation",
    )
    authenticate(Path(b["parent_dir"]))
    return frozen


def store_window(arrays, key, window):
    state = window.data.state
    for name in ("x_mean", "x_sd", "y_mean", "y_sd"):
        arrays[key + "_" + name] = getattr(state, name)
    arrays[key + "_weights"] = window.data.weights
    arrays[key + "_labels"] = window.labels
    for kind, g in window.gates.items():
        for name in ("signs", "slopes", "standard_errors", "t_stats", "labels"):
            arrays[key + "_" + kind + "_" + name] = g[name]


def store_result(arrays, key, result):
    arrays[key + "_beta"] = result["beta"]
    arrays[key + "_alpha"] = result["alpha"]
    # Parameters differ from beta only for UniLasso; shared named arrays avoid copying otherwise.
    if not np.array_equal(result["parameter"], result["beta"]):
        arrays[key + "_parameter"] = result["parameter"]


def dataset(output, cell_key, replication):
    started = time.perf_counter()
    output = Path(output)
    p = read(output / "protocol.json")
    frozen = read(output / "freeze.json")
    cell = next(c for c in p["cells"] if c["key"] == cell_key)
    key = f"{cell_key}__{replication:03d}"
    dest = output / "datasets" / key
    dest.mkdir(parents=True, exist_ok=True)
    require(not (dest / "complete.json").exists(), "Dataset already complete")
    arrays, selectors, records, finals = {}, [], [], {}
    try:
        with np.load(output / "calibration.npz", allow_pickle=False) as archive:
            cal = dict(archive)
        generated = P.generate(cal, cell, replication, p["generator"])
        arrays.update(generated)
        n = cell["history"]
        step = n // 6
        dates = pd.date_range("2000-01-31", periods=n, freq="ME")
        x = pd.DataFrame(generated["x"], index=dates, columns=p["factor_basis"])
        y = pd.DataFrame(generated["y"], index=dates, columns=p["assets"])
        methods = configs(p, cell_key)
        windows = {}
        for span in sorted({m["span"] or 0 for m in methods}):
            for stop in (n // 2, n // 2 + step, n // 2 + 2 * step, n):
                windows[(span, stop)] = E.Window(x.iloc[:stop], y.iloc[:stop], span or None)
        selected = {}
        for method in methods:
            name, family, span = method["name"], method["family"], method["span"] or 0
            grid = (
                [0.0]
                if family == "ols"
                else list(p["ridge_grid"] if family == "ridge" else p["sparse_grid"])
            )
            pending = grid.copy()
            fold_records = []
            extension = iter(p["ridge_extensions"])
            while True:
                for stop in (n // 2, n // 2 + step, n // 2 + 2 * step):
                    window = windows[(span, stop)]
                    for candidate, result in E.path(window, method, pending).items():
                        fkey = f"{name}_n{stop}_c{candidate}"
                        store_result(arrays, fkey, result)
                        xs = (generated["x"][stop : stop + step] - window.data.state.x_mean) / (
                            window.data.state.x_sd
                        )
                        ys = (generated["y"][stop : stop + step] - window.data.state.y_mean) / (
                            window.data.state.y_sd
                        )
                        error = ys - xs @ result["beta"].T - result["alpha"]
                        fold_records.append(
                            dict(
                                stop=stop,
                                candidate=candidate,
                                key=fkey,
                                lambda_=result["lambda"],
                                mse=float(np.mean(error**2)),
                                diagnostics=result["diagnostics"],
                            )
                        )
                choice, scores = E.select(fold_records, grid)
                finite = [v for v in grid if v != "inf"]
                value = (
                    next(extension, None) if family == "ridge" and choice == max(finite) else None
                )
                if value is None:
                    break
                grid.insert(-1, value)
                pending = [value]
            selected[name] = choice
            selectors.append(
                dict(
                    method=method,
                    grid=grid,
                    selected=choice,
                    scores={str(k): v for k, v in scores.items()},
                    folds=fold_records,
                    upper_exhausted=bool(family == "ridge" and choice == 1e8),
                )
            )
        for method in methods:
            name, family, span = method["name"], method["family"], method["span"] or 0
            requested = [("own", selected[name])]
            if family in ("hcgl", "fcgl") and method["gate"] != "ungated":
                base = family + "_ungated" + ("_ew120" if span else "")
                requested.append(("matched", selected[base]))
            window = windows[(span, n)]
            values = list(dict.fromkeys(v for _, v in requested))
            fits = E.path(window, method, values)
            for candidate, result in fits.items():
                fkey = f"{name}_full_c{candidate}"
                store_result(arrays, fkey, result)
                raw_beta, raw_alpha = E.F.original_units(window.data.state, result)
                arrays[fkey + "_raw_beta"], arrays[fkey + "_raw_alpha"] = raw_beta, raw_alpha
                prediction = generated["x_test"] @ raw_beta.T + raw_alpha
                scores, component = losses(
                    raw_beta,
                    raw_alpha,
                    generated["truth"],
                    cal["annual_premia"],
                    cal["factor_covariance"],
                    generated["y_test"],
                    prediction,
                    window.data.state.y_sd,
                )
                arrays[fkey + "_component_error"] = component
                for metric, value in scores.items():
                    arrays[fkey + "_" + metric] = value
                retained = abs(result["beta"]) > 1e-6
                active = generated["truth"] != 0
                hard, soft = window.restrictions(method)
                metrics = {metric: float(value.mean()) for metric, value in scores.items()}
                metrics.update(
                    retention=float(retained.mean()),
                    active_retention=float(retained[active].mean()),
                    false_exclusion=float(np.mean((hard == 0)[active])),
                    wrong_hard_restriction=float(
                        np.mean(hard[active] * generated["truth"][active] < 0)
                    ),
                    wrong_fitted_direction=float(
                        np.mean(
                            (raw_beta[active] * generated["truth"][active] < 0) & retained[active]
                        )
                    ),
                )
                finals[fkey] = dict(
                    method=method,
                    candidate=candidate,
                    lambda_=result["lambda"],
                    diagnostics=result["diagnostics"],
                    metrics=metrics,
                )
            for mode, candidate in requested:
                records.append(dict(method=name, tuning=mode, key=f"{name}_full_c{candidate}"))
        for (span, stop), window in windows.items():
            store_window(arrays, f"w{span}_n{stop}", window)
        require(
            {(r["method"], r["tuning"]) for r in records}
            == {(r["method"], r["tuning"]) for r in record_inventory(p, cell_key)},
            "Incomplete records",
        )
        np.savez_compressed(dest / "arrays.npz", **arrays)
        ledger = dict(
            cell=cell_key,
            replication=replication,
            selectors=selectors,
            records=records,
            finals=finals,
            seconds=time.perf_counter() - started,
        )
        write(dest / "ledger.json", ledger)
        receipt = dict(
            run_identity=frozen["identity"],
            cell=cell_key,
            replication=replication,
            final_records=len(records),
            unique_final_fits=len(finals),
            inner_fits=sum(len(s["folds"]) for s in selectors),
            ledger_sha256=sha(dest / "ledger.json"),
            arrays_sha256=sha(dest / "arrays.npz"),
        )
        write(dest / "complete.json", receipt)
        if __package__:
            from .verify_matf_full import verified_dataset
        else:
            from verify_matf_full import verified_dataset
        verified_dataset(output, key)
        return dict(**receipt, seconds=time.perf_counter() - started)
    except Exception:
        np.savez_compressed(dest / "partial_arrays.npz", **arrays)
        write(
            dest / "partial_ledger.json", dict(selectors=selectors, records=records, finals=finals)
        )
        write(
            dest / "failure.json",
            dict(traceback=traceback.format_exc(), seconds=time.perf_counter() - started),
        )
        raise


def execute(output, limit=None):
    output = external(output)
    frozen = authenticate_full(output)
    p = read(output / "protocol.json")
    for env in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        require(os.environ.get(env) == "1", "Set " + env + "=1 before Python")
    units = [(c["key"], r) for r in range(p["replications"]) for c in p["cells"]]
    pending = []
    done = 0
    for cell, r in units:
        dest = output / "datasets" / f"{cell}__{r:03d}"
        require(not (dest / "failure.json").exists(), "Unresolved prior failure; investigate first")
        if (dest / "complete.json").exists() and (dest / "verified.json").exists():
            record = read(dest / "complete.json")
            require(record["run_identity"] == frozen["identity"], "Wrong checkpoint identity")
            for file in ("ledger", "arrays"):
                ext = ".json" if file == "ledger" else ".npz"
                require(sha(dest / (file + ext)) == record[file + "_sha256"], "Corrupt checkpoint")
            verified = read(dest / "verified.json")
            require(
                verified["status"] == "passed"
                and verified["run_identity"] == frozen["identity"]
                and verified["receipt_sha256"] == sha(dest / "complete.json"),
                "Invalid verification checkpoint",
            )
            done += 1
        else:
            require(
                not (dest / "complete.json").exists(),
                "Completed unit lacks independent audit; verify it before resuming",
            )
            pending.append((cell, r))
    if limit is not None:
        pending = pending[:limit]
    started = time.perf_counter()
    finished = 0
    workers = frozen["binding"]["workers"]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = iter(pending)
        running = {}

        def refill():
            while len(running) < workers * 2:
                unit = next(tasks, None)
                if unit is None:
                    break
                running[pool.submit(dataset, str(output), *unit)] = unit

        refill()
        try:
            while running:
                ready, _ = wait(running, timeout=45, return_when=FIRST_COMPLETED)
                for future in ready:
                    result = future.result()
                    running.pop(future)
                    finished += 1
                    if finished % 16 == 0 or finished == len(pending):
                        progress = dict(
                            completed=done + finished,
                            total=len(units),
                            session_completed=finished,
                            wall_seconds=time.perf_counter() - started,
                            last_cell=result["cell"],
                            last_replication=result["replication"],
                        )
                        write(output / "progress.json", progress)
                        print(json.dumps(progress), flush=True)
                refill()
        except Exception:
            for future in running:
                future.cancel()
            raise
    sessions = output / "sessions"
    sessions.mkdir(exist_ok=True)
    timing = dict(completed=finished, wall_seconds=time.perf_counter() - started, workers=workers)
    write(sessions / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json"), timing)
    if done + finished == len(units):
        write(
            output / "execution_complete.json",
            dict(
                status="complete; all units independently audited; aggregate audit pending",
                datasets=len(units),
                identity=frozen["identity"],
            ),
        )
    return timing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "run"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze(args.pilot, args.out, args.validation, args.workers)
            if args.action == "freeze"
            else execute(args.out, args.limit),
            indent=2,
        )
    )
