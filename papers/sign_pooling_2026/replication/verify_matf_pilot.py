"""Independent readback of the frozen monthly ETF/MATF pilot.

No producer scoring, preprocessing, tuning-selection or proximal code is used.
Reconstruct paired Gaussian draws, units, gates, losses and sparse stationarity
from saved arrays. Ridge is checked through augmented least squares. Every fit
and every inner fold is required before descriptive paired Monte Carlo tables.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.special import comb
from scipy.optimize import brentq

if __package__:
    from .verify_matf_comparators import partial_reference, reference_fit
else:
    from verify_matf_comparators import partial_reference, reference_fit

REPO = Path(__file__).resolve().parents[3]
KEYS = ["cell", "family", "span", "arm", "soft_strength", "zero_strength"]


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def identity(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def close(actual, expected, name="", rtol=2e-9, atol=2e-11):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)


def reference_losses(beta, alpha, truth, premium, covariance, observed, prediction, scales):
    q = len(beta)
    component = np.empty_like(beta)
    values = {
        name: np.empty(q)
        for name in (
            "attribution_mse",
            "mean_mse",
            "full_mean_mse",
            "alpha_mse",
            "mean_alpha_cross",
            "annual_prediction_risk",
            "loading_mse",
            "test_mse",
            "standardized_test_mse",
        )
    }
    for i in range(q):
        error = beta[i] - truth[i]
        monthly = error * (premium / 12)
        component[i] = 12 * monthly
        mean_error = np.dot(error, premium)
        values["attribution_mse"][i] = 144 * np.dot(monthly, monthly)
        values["mean_mse"][i] = mean_error**2
        values["full_mean_mse"][i] = (mean_error + alpha[i] * 12) ** 2
        values["alpha_mse"][i] = 144 * alpha[i] ** 2
        values["mean_alpha_cross"][i] = 24 * mean_error * alpha[i]
        values["annual_prediction_risk"][i] = error @ (12 * covariance) @ error
        values["loading_mse"][i] = np.dot(error, error) / len(error)
        residual = prediction[:, i] - observed[:, i]
        values["test_mse"][i] = np.dot(residual, residual) / len(residual)
        values["standardized_test_mse"][i] = values["test_mse"][i] / scales[i] ** 2
    return values, component


def transforms(x, y, span):
    n = len(x)
    w = (
        np.ones(n)
        if span is None
        else np.exp(np.arange(n - 1, -1, -1) * np.log((span - 1) / (span + 1)))
    )
    w *= n / w.sum()
    xm, ym = w @ x / n, w @ y / n
    xs = np.sqrt(w @ ((x - xm) ** 2) / n)
    ys = np.sqrt(w @ ((y - ym) ** 2) / n)
    return (x - xm) / xs, (y - ym) / ys, w, dict(x_mean=xm, x_sd=xs, y_mean=ym, y_sd=ys)


def partition(y, w, cut):
    tree = linkage(pdist((np.sqrt(w[:, None]) * y).T) / np.sqrt(len(y)), method="ward")
    labels = fcluster(tree, cut * tree[-1, 2], criterion="distance")
    mapping = {}
    return np.array([mapping.setdefault(int(v), len(mapping)) for v in labels])


def marginal_reference(x, y, w, labels):
    b, se = np.zeros((y.shape[1], x.shape[1])), np.zeros((y.shape[1], x.shape[1]))
    effective = w.sum() ** 2 / np.dot(w, w)
    for group in np.unique(labels):
        ids = np.where(labels == group)[0]
        response = y[:, ids].mean(axis=1)
        for j in range(x.shape[1]):
            denominator = np.dot(w, x[:, j] ** 2)
            slope = np.dot(w * x[:, j], response) / denominator
            influence = w * x[:, j] * (response - slope * x[:, j]) / denominator
            b[ids, j] = slope
            se[ids, j] = np.sqrt(np.dot(influence, influence) * effective / (effective - 1))
    return b, se


def proximal(b, labels, family, lam):
    """Independent vector/block norm soft threshold after element threshold."""
    b = np.sign(b) * np.maximum(abs(b) - lam / 2, 0)
    groups = np.unique(labels)
    for group in groups:
        mask = labels == group
        weight = np.sqrt(mask.sum() / len(groups))
        block = b[mask]
        norms = np.linalg.norm(block, axis=1 if family == "hcgl" else 0, keepdims=True)
        multiplier = np.maximum(0, 1 - lam * weight / (2 * np.maximum(norms, 1e-300)))
        b[mask] = block * multiplier
    return b


def zero_scale(x, y, w, labels, family):
    """Independent root of the largest sparse-group zero-stationarity slack."""
    cross = 2 * (y.T * w) @ x / len(x)
    groups = np.unique(labels)

    def slack(lam):
        soft = np.maximum(abs(cross) - lam / 2, 0)
        terms = []
        for group in groups:
            mask = labels == group
            norms = np.linalg.norm(soft[mask], axis=1 if family == "hcgl" else 0)
            terms.append(np.max(norms - lam / 2 * np.sqrt(mask.sum() / len(groups))))
        return max(terms)

    return brentq(slack, 0, 2 * abs(cross).max(), xtol=1e-12)


def stationary(x, y, w, labels, family, beta, lam, hard, soft, eta, eta0):
    gram = (x.T * w) @ x / len(x)
    cross = (y.T * w) @ x / len(x)
    grad = 2 * (beta @ gram - cross)
    grad += 2 * eta * (np.nan_to_num(soft) * beta < 0) * beta
    grad += 2 * eta0 * (soft == 0) * beta
    lipschitz = max(1, 2 * np.linalg.eigvalsh(gram)[-1] + 2 * max(eta, eta0))
    b = beta - grad / lipschitz
    b[hard == 0] = 0
    b[hard == 1] = np.maximum(b[hard == 1], 0)
    b[hard == -1] = np.minimum(b[hard == -1], 0)
    update = proximal(b, labels, family, lam / lipschitz)
    residual = np.max(abs(beta - update)) * lipschitz / (1 + abs(beta).max() + abs(grad).max())
    violation = max(
        np.max(abs(beta[hard == 0]), initial=0),
        np.max(-beta[hard == 1], initial=0),
        np.max(beta[hard == -1], initial=0),
    )
    groups = np.unique(labels)
    penalty = np.sum(abs(beta)) / 2
    for group in groups:
        mask = labels == group
        norms = np.linalg.norm(beta[mask], axis=1 if family == "hcgl" else 0)
        penalty += np.sqrt(mask.sum() / len(groups)) * norms.sum() / 2
    objective = np.sum(w[:, None] * (y - x @ beta.T) ** 2) / len(x) + lam * penalty
    objective += eta * np.sum(np.minimum(np.nan_to_num(soft) * beta, 0) ** 2)
    objective += eta0 * np.sum(beta[soft == 0] ** 2)
    return float(residual), float(violation), float(objective)


def ari(a, b):
    contingency = np.array(
        [[np.sum((a == i) & (b == j)) for j in np.unique(b)] for i in np.unique(a)]
    )
    n = len(a)
    index = comb(contingency, 2).sum()
    row, col = comb(contingency.sum(1), 2).sum(), comb(contingency.sum(0), 2).sum()
    expected = row * col / comb(n, 2)
    denominator = (row + col) / 2 - expected
    return 1.0 if denominator == 0 else (index - expected) / denominator


def verify(output):
    started = time.perf_counter()
    output = Path(output).resolve()
    freeze = read(output / "run_freeze.json")
    check(identity(freeze["binding"]) == freeze["identity"], "Execution freeze changed")
    for name, expected in freeze["binding"]["sources"].items():
        check(sha(REPO / name) == expected, "Execution source changed: " + name)
        check(sha(output / "source" / name) == expected, "Execution snapshot changed")
    protocol_dir = Path(freeze["binding"]["protocol_dir"])
    previous = read(protocol_dir / "freeze.json")
    check(
        identity(previous["binding"])
        == previous["identity"]
        == freeze["binding"]["protocol_identity"],
        "Stage 1B identity changed",
    )
    for name, ext in (("protocol", ".json"), ("calibration", ".npz"), ("planned_fits", ".json")):
        check(
            sha(protocol_dir / (name + ext)) == previous["binding"][name + "_sha256"],
            "Stage 1B artifact changed",
        )
    for name, expected in {
        **previous["binding"]["source_sha256"],
        **previous["binding"]["unchanged_estimator_sha256"],
    }.items():
        check(sha(REPO / name) == expected, "Estimator source changed")
    p, jobs = read(protocol_dir / "protocol.json"), read(protocol_dir / "planned_fits.json")
    with np.load(protocol_dir / "calibration.npz", allow_pickle=False) as f:
        calibration = dict(f)
    dataset_jobs = {}
    for j in jobs:
        key = f"{j['cell']}__{j['replication']:03d}"
        dataset_jobs.setdefault(key, []).append(j)
    check(
        {d.name for d in (output / "datasets").iterdir() if d.is_dir()} == set(dataset_jobs),
        "Missing or extra datasets",
    )
    all_rows, sleeve_rows, factor_rows, all_selectors = [], [], [], []
    maxima = dict(
        kkt=0.0, violation=0.0, ridge_coefficient_error=0.0, reference_coefficient_error=0.0
    )
    counts = Counter()
    runtimes = []
    statuses = Counter()
    for dataset, expected_jobs in sorted(dataset_jobs.items()):
        dest = output / "datasets" / dataset
        receipt = read(dest / "complete.json")
        check(receipt["run_identity"] == freeze["identity"], "Wrong checkpoint identity")
        check(receipt["planned_sha256"] == identity(expected_jobs), "Changed planned subset")
        check(sha(dest / "ledger.json") == receipt["ledger_sha256"], "Changed ledger")
        check(sha(dest / "arrays.npz") == receipt["arrays_sha256"], "Changed coefficient arrays")
        ledger = read(dest / "ledger.json")
        rows, selectors = ledger["fits"], ledger["selectors"]
        check(
            len(rows) == len(expected_jobs)
            and {identity({k: r[k] for k in expected_jobs[0]}) for r in rows}
            == {identity(j) for j in expected_jobs},
            "Missing/duplicate/changed final record",
        )
        cell = next(c for c in p["planned_full_core"] if c["key"] == ledger["cell"])
        rep, n = ledger["replication"], cell["history"]
        with np.load(dest / "arrays.npz", allow_pickle=False) as f:
            a = dict(f)
        rng = np.random.default_rng(np.random.SeedSequence([p["pilot_seed"], rep]))
        xx = rng.standard_normal((120 + p["test_rows"], 11))
        xx = xx @ np.linalg.cholesky(calibration["factor_covariance"]).T
        xx += calibration["annual_premia"] / 12
        eps = rng.standard_normal((120 + p["test_rows"], 247))
        eps = eps @ np.linalg.cholesky(calibration["residual_covariance"]).T
        beta = calibration[cell["alignment"] + "_beta"] * (0.5 if cell["strength"] == "weak" else 1)
        scales = np.ones(n)
        if cell["evolution"] == "reversal":
            scales[n // 2 :] = 1 - 1.5 * np.arange(n - n // 2) / (n - n // 2 - 1)
        truth = beta * scales[-1]
        close(a["x"], xx[120 - n : 120], "paired factor draws")
        close(a["x_test"], xx[120:], "paired test factors")
        close(a["y"], (xx[120 - n : 120] @ beta.T) * scales[:, None] + eps[120 - n : 120])
        close(a["y_test"], xx[120:] @ truth.T + eps[120:])
        close(a["truth"], truth)
        close(a["training_beta"], beta)
        close(a["loading_scales"], scales)
        close(a["annual_premia"], calibration["annual_premia"])
        cache = {}

        def training(key, stop, span, sparse=True):
            if key not in cache:
                x, y, w, meta = transforms(a["x"][:stop], a["y"][:stop], span)
                for field, value in meta.items():
                    close(a[key + "_" + field], value, key + field)
                close(a[key + "_weights"], w)
                labels = partition(y, w, p["partition_cut"]) if sparse else None
                if sparse:
                    close(a[key + "_labels"], labels)
                    gw = np.array(
                        [np.sqrt(np.sum(labels == i) / len(np.unique(labels))) for i in labels]
                    )
                    close(a[key + "_group_weights"], gw)
                    close(
                        a[key + "_scale"],
                        zero_scale(x, y, w, labels, "fcgl" if key.startswith("fcgl") else "hcgl"),
                        "Independent KKT penalty scale",
                        rtol=1e-8,
                        atol=1e-9,
                    )
                cache[key] = (x, y, w, meta, labels)
            return cache[key]

        def fit_check(key, state, family, lam, diagnostics, hard, soft, eta, eta0, predict_x):
            x, y, w, meta, labels = state
            b = a[key + "_beta_std"]
            check(np.isfinite(b).all(), "Nonfinite fit")
            close(a[key + "_beta"], b * meta["y_sd"][:, None] / meta["x_sd"])
            close(a[key + "_alpha"], meta["y_mean"] - a[key + "_beta"] @ meta["x_mean"])
            close(a[key + "_prediction"], predict_x @ a[key + "_beta"].T + a[key + "_alpha"])
            if family == "ridge":
                design = np.vstack(
                    [np.sqrt(w[:, None]) * x, np.sqrt(len(x) * lam) * np.eye(x.shape[1])]
                )
                target = np.vstack([np.sqrt(w[:, None]) * y, np.zeros((x.shape[1], y.shape[1]))])
                reference = np.linalg.lstsq(design, target, rcond=None)[0].T
                discrepancy = float(np.max(abs(reference - b)))
                maxima["ridge_coefficient_error"] = max(
                    maxima["ridge_coefficient_error"], discrepancy
                )
                close(b, reference, "ridge augmented reference", rtol=1e-7, atol=1e-8)
                counts["ridge_references"] += 1
            else:
                kkt, violation, objective = stationary(
                    x, y, w, labels, family, b, lam, hard, soft, eta, eta0
                )
                maxima["kkt"] = max(maxima["kkt"], kkt)
                maxima["violation"] = max(maxima["violation"], violation)
                check(kkt <= 1.01e-5 and violation <= 1.01e-7, "Independent stationarity failed")
                close(objective, diagnostics["objective"], "objective", rtol=1e-8, atol=1e-8)
                counts["sparse_stationarity"] += 1
            check(
                diagnostics["status"]
                in (
                    "package_ridge",
                    "optimal",
                    "resolved_inaccurate",
                    "exact_zero",
                    "independent_fista_recovery",
                ),
                "Unresolved fit",
            )
            if diagnostics["status"] == "independent_fista_recovery":
                check(kkt <= 1.01e-7 and violation <= 1e-7, "Recovery stationarity failed")
                recovery = diagnostics["recovery"]
                check(
                    recovery["reference_coefficient_error"] <= 3e-6
                    and recovery["reference_relative_objective_gap"] <= 1e-8
                    and bool(diagnostics["original_failure"]),
                    "Recovery audit missing",
                )
            statuses[diagnostics["status"]] += 1
            counts["warnings"] += len(diagnostics["outer_warnings"])
            for attempt in diagnostics.get("attempts", []):
                counts["warnings"] += len(attempt.get("warnings", []))
                counts["solver_attempts"] += 1
                counts["solver_error_attempts"] += attempt["status"] == "solver_error"
                counts["inaccurate_attempts"] += attempt["status"] == "optimal_inaccurate"
            counts["independently_checked_fits"] += 1

        for selector in selectors:
            family, span = selector["family"], selector["span"]
            grid = selector["grid"]
            base = p["ridge_grid"] if family == "ridge" else p["sparse_fraction_grid"]
            valid_grid = list(base)
            entries = selector["folds"]
            check(len(entries) == 3 * len(grid), "Missing or duplicate tuning candidates")
            check(
                {(v["train_rows"], v["candidate"]) for v in entries}
                == {
                    (stop, v)
                    for stop in (n // 2, n // 2 + n // 6, n // 2 + 2 * n // 6)
                    for v in grid
                },
                "Tuning fold coverage differs",
            )
            scores = {}
            for candidate in grid:
                scores[candidate] = np.mean(
                    [v["mse"] for v in entries if v["candidate"] == candidate]
                )

            def selected(g):
                return max(v for v in g if scores[v] <= min(scores[c] for c in g) + 1e-12)

            if family == "ridge":
                for extension in p["ridge_upper_extensions"]:
                    if selected(valid_grid) != max(valid_grid):
                        break
                    valid_grid.append(extension)
                    check(extension in grid, "Missing required ridge boundary extension")
            check(
                grid == valid_grid and selector["selected"] == selected(grid),
                "Wrong selection or unapproved grid extension",
            )
            for entry in entries:
                stop, key = entry["train_rows"], entry["key"]
                state = training(entry["training_key"], stop, span, family != "ridge")
                lam = (
                    entry["candidate"]
                    if family == "ridge"
                    else (entry["candidate"] * float(a[entry["training_key"] + "_scale"]))
                )
                close(entry["lambda_"], lam)
                unrestricted = np.full_like(a[key + "_beta_std"], np.nan)
                fit_check(
                    key,
                    state,
                    family,
                    lam,
                    entry["diagnostics"],
                    unrestricted,
                    unrestricted,
                    0,
                    0,
                    a["x"][stop : stop + n // 6],
                )
                error = a["y"][stop : stop + n // 6] - a[key + "_prediction"]
                expected_mse = np.mean(error**2 / state[3]["y_sd"] ** 2)
                close(entry["mse"], expected_mse, "fold MSE")
                counts["inner_fits"] += 1
            all_selectors.append(
                dict(
                    cell=cell["key"],
                    replication=rep,
                    family=family,
                    span=span or 0,
                    selected=selector["selected"],
                    candidates=len(grid),
                    upper_exhausted=selector["upper_exhausted"],
                    seconds=sum(v["diagnostics"]["elapsed_seconds"] for v in entries),
                )
            )
        gate_cache = {}
        for row in rows:
            key, arm, family, span = row["key"], row["arm"], row["family"], row["span"]
            check(
                key == identity({k: row[k] for k in expected_jobs[0]})[:24], "Fit identity changed"
            )
            tkey = row["training_key"]
            state = training(tkey, n, span)
            x, y, w, meta, labels = state
            selected_family = "ridge" if arm == "ridge" else family
            selector = next(
                s
                for s in selectors
                if s["key"] == (("ridge_" + str(span or 0)) if arm == "ridge" else tkey)
            )
            expected_lambda = selector["selected"] * (
                1 if arm == "ridge" else float(a[tkey + "_scale"])
            )
            close(row["lambda_"], expected_lambda)
            close(row["selected"], selector["selected"])
            hard, soft = np.full_like(truth, np.nan), np.full_like(truth, np.nan)
            gate_metrics = {}
            if arm not in ("ridge", "ungated"):
                kind = (
                    "conditional"
                    if arm.startswith("conditional")
                    else "singleton"
                    if arm.startswith("singleton")
                    else "marginal"
                )
                gkey = tkey + "_" + kind
                if gkey not in gate_cache:
                    history = []
                    for stop in (n // 2, n // 2 + n // 6, n // 2 + 2 * n // 6, n):
                        gx, gy, gw, _ = transforms(a["x"][:stop], a["y"][:stop], span)
                        gl = (
                            np.arange(len(truth))
                            if kind == "singleton"
                            else partition(gy, gw, p["partition_cut"])
                        )
                        slope, se = (
                            partial_reference(gx, gy, gw, gl)
                            if kind == "conditional"
                            else marginal_reference(gx, gy, gw, gl)
                        )
                        stat = np.divide(slope, se, out=np.zeros_like(slope), where=se > 0)
                        stat[(se == 0) & (slope != 0)] = (
                            np.sign(slope[(se == 0) & (slope != 0)]) * np.inf
                        )
                        signs = np.where(abs(stat) >= p["gate_threshold"], np.sign(slope), 0)
                        for field, value in dict(
                            slopes=slope, standard_errors=se, t_stats=stat, signs=signs, labels=gl
                        ).items():
                            close(a[f"{gkey}_{stop}_{field}"], value, gkey + field)
                        history.append(signs)
                        counts["gate_snapshots"] += 1
                    history = np.array(history)
                    gate_cache[gkey] = (
                        history[-1],
                        np.mean((history[1:] != 0) != (history[:-1] != 0)),
                        np.mean(history[1:] != history[:-1]),
                    )
                signs, crossing, changes = gate_cache[gkey]
                if arm.startswith("soft"):
                    soft = signs.copy()
                else:
                    hard = signs.copy()
                    if arm.endswith("release"):
                        hard[signs == 0] = np.nan
                    if arm == "pooled_zero_only":
                        hard[signs != 0] = np.nan
                active = truth != 0
                gate_metrics = dict(
                    gate_failed_active=float(np.mean(signs[active] == 0)),
                    gate_wrong_active=float(np.mean(signs[active] * truth[active] < 0)),
                    gate_threshold_crossing=float(crossing),
                    gate_direction_change=float(changes),
                )
            close(a[key + "_hard"], hard)
            close(a[key + "_soft"], soft)
            fit_check(
                key,
                state,
                selected_family,
                row["lambda_"],
                row["diagnostics"],
                hard,
                soft,
                row["soft_strength"],
                row["zero_strength"],
                a["x_test"],
            )
            if rep == 0 and arm != "ridge":
                reference, _ = reference_fit(
                    x,
                    y,
                    w,
                    labels,
                    family,
                    row["lambda_"],
                    hard,
                    soft,
                    row["soft_strength"],
                    row["zero_strength"],
                )
                discrepancy = float(np.max(abs(reference - a[key + "_beta_std"])))
                maxima["reference_coefficient_error"] = max(
                    maxima["reference_coefficient_error"], discrepancy
                )
                close(
                    reference,
                    a[key + "_beta_std"],
                    "full reference optimization",
                    rtol=1e-5,
                    atol=3e-6,
                )
                counts["full_optimization_references"] += 1
            values, components = reference_losses(
                a[key + "_beta"],
                a[key + "_alpha"],
                truth,
                calibration["annual_premia"],
                calibration["factor_covariance"],
                a["y_test"],
                a[key + "_prediction"],
                meta["y_sd"],
            )
            close(a[key + "_component_error"], components)
            for metric, vector in values.items():
                close(a[key + "_" + metric], vector, metric)
                close(row["metrics"][metric], vector.mean(), metric)
                counts["scalar_loss_values"] += len(vector) + 1
            active, retained = truth != 0, abs(a[key + "_beta_std"]) > 1e-6
            expected_metrics = dict(
                retention=float(retained.mean()),
                active_retention=float(retained[active].mean()),
                wrong_fitted_direction=float(
                    np.mean((a[key + "_beta"][active] * truth[active] < 0) & retained[active])
                ),
                false_exclusion=float(np.mean((hard == 0)[active])),
                wrong_hard_restriction=float(np.mean(hard[active] * truth[active] < 0)),
                cluster_ari=float(ari(calibration["truth_labels"], labels)),
                clusters=len(np.unique(labels)),
                **gate_metrics,
            )
            for metric, value in expected_metrics.items():
                close(row["metrics"][metric], value, metric)
            flat = {k: row[k] for k in KEYS + ["replication"]}
            flat["span"] = span or 0
            flat.update(row["metrics"])
            flat.update(lambda_=row["lambda_"], seconds=row["diagnostics"]["elapsed_seconds"])
            all_rows.append(flat)
            for label, name in enumerate(("Equity", "Fixed Income", "Alternatives")):
                sleeve_rows.append(
                    dict(
                        **{k: flat[k] for k in KEYS + ["replication"]},
                        sleeve=name,
                        **{
                            metric: float(vector[calibration["truth_labels"] == label].mean())
                            for metric, vector in values.items()
                        },
                    )
                )
            for j, factor in enumerate(p["calibration"]["factors"]):
                factor_rows.append(
                    dict(
                        **{k: flat[k] for k in KEYS + ["replication"]},
                        factor=factor,
                        contribution_mse=float(np.mean(components[:, j] ** 2)),
                        contribution_bias=float(components[:, j].mean()),
                    )
                )
            counts["final_fits"] += 1
        runtimes.append(dict(cell=cell["key"], replication=rep, seconds=ledger["seconds"]))
        print(json.dumps(dict(verified_dataset=dataset, fits=counts["final_fits"])), flush=True)
    check(counts["final_fits"] == 1080 and len(dataset_jobs) == 80, "Incomplete pilot")
    frame = pd.DataFrame(all_rows)
    tables = output / "verified_tables"
    tables.mkdir(exist_ok=True)
    frame.to_csv(tables / "fit_scores.csv", index=False)
    pd.DataFrame(sleeve_rows).to_csv(tables / "sleeve_scores.csv", index=False)
    pd.DataFrame(factor_rows).to_csv(tables / "factor_scores.csv", index=False)
    pd.DataFrame(all_selectors).to_csv(tables / "tuning.csv", index=False)
    pd.DataFrame(runtimes).to_csv(tables / "runtime.csv", index=False)
    summarize(frame, tables)
    sessions = [read(f) for f in (output / "sessions").glob("*.json")]
    runtime = dict(
        worker_seconds=sum(r["seconds"] for r in runtimes),
        session_wall_seconds=sum(s["wall_seconds"] for s in sessions),
        workers=freeze["binding"]["workers"],
        verifier_seconds=time.perf_counter() - started,
    )
    result = dict(
        status="passed",
        run_identity=freeze["identity"],
        counts=dict(counts),
        maxima=maxima,
        statuses=dict(statuses),
        runtime=runtime,
        unchanged_core_files=len(previous["binding"]["unchanged_estimator_sha256"]),
        files_sha256={f.name: sha(f) for f in sorted(tables.glob("*.csv"))},
    )
    write(output / "verification.json", result)
    return result


def summarize(frame, output):
    """Within-cell Monte Carlo error; the same seed is never treated as an independent cell."""
    summaries, pairs = [], []
    metrics = [
        "attribution_mse",
        "mean_mse",
        "full_mean_mse",
        "annual_prediction_risk",
        "test_mse",
        "loading_mse",
        "retention",
        "active_retention",
        "cluster_ari",
    ]
    for keys, group in frame.groupby(KEYS, dropna=False):
        check(sorted(group.replication) == list(range(20)), "Incomplete paired replications")
        method = dict(zip(KEYS, keys))
        for metric in metrics:
            values = group.sort_values("replication")[metric].to_numpy()
            mean, se = float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))
            summaries.append(
                dict(
                    **method,
                    metric=metric,
                    mean=mean,
                    mcse=se,
                    low=mean - 1.96 * se,
                    high=mean + 1.96 * se,
                )
            )
        base = frame[
            (frame.cell == method["cell"])
            & (frame.family == method["family"])
            & (frame.span == method["span"])
            & (frame.arm == "ungated")
        ]
        for metric in metrics[:6]:
            left = group.set_index("replication")[metric].sort_index()
            right = base.set_index("replication")[metric].sort_index()
            check(left.index.equals(right.index), "Unpaired method contrast")
            diff = left.to_numpy() - right.to_numpy()
            mean, se = float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(len(diff)))
            pairs.append(
                dict(
                    **method,
                    metric=metric,
                    mean_difference=mean,
                    mcse=se,
                    low=mean - 1.96 * se,
                    high=mean + 1.96 * se,
                    percent_change=100 * mean / float(right.mean()),
                    wins=int(np.sum(diff < 0)),
                    replications=len(diff),
                )
            )
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(pairs).to_csv(output / "paired_differences.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.out), indent=2))
