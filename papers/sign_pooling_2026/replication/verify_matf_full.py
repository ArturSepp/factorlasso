"""Independent full-study verifier and tables; no producer optimization/scoring code.

An independent leverage-deletion identity checks intercepted UniLasso features.
Convex references use CVXPY on raw weighted designs; ridge uses augmented least
squares. Every stored candidate is checked for stationarity and fold loss.
The fixed 200-seed inventory is mandatory before any aggregate publication.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
import warnings

import cvxpy as cp
import numpy as np
import pandas as pd

if __package__:
    from . import verify_matf_pilot as V
    from .verify_matf_comparators import partial_reference
else:
    import verify_matf_pilot as V
    from verify_matf_comparators import partial_reference

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
read, write, sha, identity, check, close = V.read, V.write, V.sha, V.identity, V.check, V.close


def univariate(x, y):
    """LOO prediction through full-fit leverage deletion, not the producer's n-1 sums."""
    xc = x - x.mean(0)
    yc = y - y.mean(0)
    slope = (yc.T @ xc) / np.sum(xc**2, axis=0)
    constant = y.mean(0)[:, None] - slope * x.mean(0)
    full = x[:, None, :] * slope[None, :, :] + constant[None, :, :]
    leverage = 1 / len(x) + xc**2 / np.sum(xc**2, axis=0)
    loo = (full - leverage[:, None, :] * y[:, :, None]) / (1 - leverage[:, None, :])
    mean = loo.mean(0)
    return loo - mean, mean, slope, constant


def penalty(beta, labels, family):
    if family == "lasso":
        return float(np.sum(abs(beta)))
    result = 0.0
    for group in np.unique(labels):
        mask = labels == group
        weight = np.sqrt(mask.sum() / len(np.unique(labels)))
        if family == "cooperative":
            value = (
                np.linalg.norm(np.maximum(beta[mask], 0), axis=0).sum()
                + np.linalg.norm(np.minimum(beta[mask], 0), axis=0).sum()
            )
        else:
            value = np.linalg.norm(beta[mask], axis=1 if family == "hcgl" else 0).sum()
        result += weight * value
    return result if family == "cooperative" else 0.5 * (np.sum(abs(beta)) + result)


def prox(value, labels, family, lam):
    if family in ("hcgl", "fcgl"):
        return V.proximal(value, labels, family, lam)
    if family == "lasso":
        return np.sign(value) * np.maximum(abs(value) - lam, 0)
    out = value.copy()
    for group in np.unique(labels):
        mask = labels == group
        weight = np.sqrt(mask.sum() / len(np.unique(labels)))
        positive = np.maximum(value[mask], 0)
        negative = np.minimum(value[mask], 0)
        for part in (positive, negative):
            norm = np.linalg.norm(part, axis=0)
            part *= np.maximum(1 - lam * weight / np.maximum(norm, 1e-300), 0)
        out[mask] = positive + negative
    return out


def scale(x, y, w, labels, family, uni=None):
    cross = 2 * (y.T * w) @ x / len(x)
    if family in ("hcgl", "fcgl"):
        return V.zero_scale(x, y, w, labels, family)
    if family == "lasso":
        return float(np.max(abs(cross)))
    if family == "unilasso":
        return float(max(0, np.max(2 * np.einsum("nqp,nq->qp", uni[0], y) / len(x))))
    if family == "cooperative":
        candidates = []
        for group in np.unique(labels):
            mask = labels == group
            weight = np.sqrt(mask.sum() / len(np.unique(labels)))
            candidates.extend(
                [
                    np.max(np.linalg.norm(np.maximum(cross[mask], 0), axis=0)) / weight,
                    np.max(np.linalg.norm(np.minimum(cross[mask], 0), axis=0)) / weight,
                ]
            )
        return float(max(candidates))
    return 1.0


def restrictions(method, gates):
    sample = next(iter(gates.values()))["signs"]
    hard, soft = np.full_like(sample, np.nan), np.full_like(sample, np.nan)
    action = method["gate"]
    if action not in ("none", "ungated"):
        kind = (
            "conditional"
            if action.startswith("conditional")
            else "singleton"
            if action.startswith("singleton")
            else "marginal"
        )
        signs = gates[kind]["signs"]
        if action == "soft_direction":
            soft = signs.copy()
        else:
            hard = signs.copy()
            if action.endswith("release"):
                hard[hard == 0] = np.nan
            elif action.endswith("zero_only"):
                hard[hard != 0] = np.nan
    return hard, soft


def stationarity(state, method, beta, alpha, parameter, lam, hard, soft):
    x, y, w, meta, labels, uni = state
    family = method["family"]
    n = len(x)
    if family in ("ridge", "ols"):
        if lam == "inf":
            close(beta, np.zeros_like(beta), atol=0, rtol=0)
            return 0.0, 0.0, None
        # Independent full normal equations; augmented LS is used in fixed reference checks.
        gradient = 2 * ((x.T * w) @ x @ beta.T - (x.T * w) @ y) / n + 2 * lam * beta.T
        return float(abs(gradient).max() / (1 + abs(beta).max())), 0.0, None
    if family == "unilasso":
        eta, mean, slope, constant = uni
        close(beta, parameter * slope, atol=1e-10)
        close(alpha, np.sum(parameter * (constant - mean), axis=1), atol=1e-10)
        residual = np.einsum("nqp,qp->nq", eta, parameter) - y
        grad = 2 * np.einsum("nqp,nq->qp", eta, residual) / n + lam
        cross = 2 * np.einsum("nqp,nq->qp", eta, y) / n
        error = np.where(parameter > 1e-10, abs(grad), np.maximum(-grad, 0))
        kkt = float(np.max(error / (1 + abs(cross))))
        return (
            kkt,
            float(np.maximum(-parameter, 0).max()),
            float(np.sum(residual**2) / n + lam * parameter.sum()),
        )
    grad = 2 * ((x.T * w) @ x @ beta.T - (x.T * w) @ y).T / n
    grad += 2 * method["eta"] * (np.nan_to_num(soft) * beta < 0) * beta
    grad += 2 * method["eta0"] * (soft == 0) * beta
    lip = max(
        1, 2 * np.linalg.eigvalsh((x.T * w) @ x / n)[-1] + 2 * max(method["eta"], method["eta0"])
    )
    value = beta - grad / lip
    value[hard == 0] = 0
    value[hard == 1] = np.maximum(value[hard == 1], 0)
    value[hard == -1] = np.minimum(value[hard == -1], 0)
    updated = prox(value, labels, family, lam / lip)
    kkt = float(lip * abs(beta - updated).max() / (1 + abs(beta).max() + abs(grad).max()))
    violation = max(
        float(np.max(abs(beta[hard == 0]), initial=0)),
        float(np.max(-beta[hard == 1], initial=0)),
        float(np.max(beta[hard == -1], initial=0)),
    )
    obj = np.sum(w[:, None] * (y - x @ beta.T) ** 2) / n + lam * penalty(beta, labels, family)
    obj += method["eta"] * np.sum(np.minimum(np.nan_to_num(soft) * beta, 0) ** 2)
    obj += method["eta0"] * np.sum(beta[soft == 0] ** 2)
    return kkt, violation, float(obj)


def conic_reference(x, y, w, labels, method, lam, hard, soft, uni=None):
    """Independent CVXPY objectives; UniLasso is evaluated on fixed independent response indices."""
    family = method["family"]
    n = len(x)
    q, k = y.shape[1], x.shape[1]
    if family in ("ridge", "ols"):
        if lam == "inf":
            return np.zeros((q, k)), np.zeros(q), dict(status="exact_inf")
        design = np.vstack([np.sqrt(w[:, None]) * x, np.sqrt(n * lam) * np.eye(k)])
        target = np.vstack([np.sqrt(w[:, None]) * y, np.zeros((k, q))])
        return (
            np.linalg.lstsq(design, target, rcond=None)[0].T,
            np.zeros(q),
            dict(status="augmented_ls"),
        )
    variable = cp.Variable((q, k))
    constraints = []
    if family == "unilasso":
        eta, mean, slope, constant = uni
        loss = sum(cp.sum_squares(eta[:, i] @ variable[i] - y[:, i]) / n for i in range(q))
        regularizer = cp.sum(variable)
        constraints = [variable >= 0]
    else:
        loss = cp.sum_squares(cp.multiply(np.sqrt(w[:, None]), x @ variable.T - y)) / n
        if family == "lasso":
            regularizer = cp.norm1(variable)
        else:
            terms = []
            for group in np.unique(labels):
                ids = np.flatnonzero(labels == group)
                weight = np.sqrt(len(ids) / len(np.unique(labels)))
                if family == "cooperative":
                    terms.append(
                        weight
                        * cp.sum(
                            cp.norm(cp.pos(variable[ids]), 2, axis=0)
                            + cp.norm(cp.neg(variable[ids]), 2, axis=0)
                        )
                    )
                else:
                    terms.append(
                        weight
                        * cp.sum(cp.norm(variable[ids], 2, axis=1 if family == "hcgl" else 0))
                    )
            regularizer = (
                sum(terms) if family == "cooperative" else 0.5 * (cp.norm1(variable) + sum(terms))
            )
        for value in (-1, 0, 1):
            mask = hard == value
            if np.any(mask):
                constraints.append(
                    variable[mask] == 0 if value == 0 else value * variable[mask] >= 0
                )
        direction = np.flatnonzero(np.isfinite(soft.ravel()) & (soft.ravel() != 0))
        zero = np.flatnonzero(soft.ravel() == 0)
        flat = cp.vec(variable, order="C")
        if method["eta"] and len(direction):
            loss += method["eta"] * cp.sum_squares(
                cp.pos(-cp.multiply(soft.ravel()[direction], flat[direction]))
            )
        if method["eta0"] and len(zero):
            loss += method["eta0"] * cp.sum_squares(flat[zero])
    problem = cp.Problem(cp.Minimize(loss if lam == 0 else loss + lam * regularizer), constraints)
    attempts = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for solver, options in (
            ("CLARABEL", dict(tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9, max_iter=2000)),
            ("SCS", dict(eps=1e-8, max_iters=100000)),
        ):
            try:
                problem.solve(solver=solver, **options)
                attempts.append(dict(solver=solver, status=problem.status))
                if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    break
            except cp.error.SolverError as exc:
                attempts.append(dict(solver=solver, error=str(exc)))
    check(
        problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        and variable.value is not None
        and np.isfinite(variable.value).all(),
        "Reference solver failed",
    )
    value = np.asarray(variable.value)
    if family == "unilasso":
        beta, alpha = value * slope, np.sum(value * (constant - mean), axis=1)
    else:
        beta, alpha = value, np.zeros(q)
    return (
        beta,
        alpha,
        dict(
            status=problem.status,
            objective=float(problem.value),
            warnings=[str(w.message) for w in caught],
            attempts=attempts,
        ),
    )


def validation(output):
    """Pre-freeze numerical checks on non-study fixtures."""
    if __package__:
        from . import matf_full_engine as E
    else:
        import matf_full_engine as E
    output = Path(output)
    check(not output.exists(), "Use fresh validation directory")
    rng = np.random.default_rng(429419)
    cases = []
    maximum = 0.0
    for span in (None, 120):
        x = rng.normal(size=(60, 4))
        x[:, 1] += 0.8 * x[:, 0]
        y = x @ rng.normal(size=(4, 7)) + rng.normal(size=(60, 7)) * 0.5
        dates = pd.date_range("2000-01-31", periods=60, freq="ME")
        win = E.Window(pd.DataFrame(x, index=dates), pd.DataFrame(y, index=dates), span)
        for method in E.methods("fixture", []):
            if span and method["family"] == "unilasso":
                continue
            for candidate in (
                [0.0, 0.1, "inf"]
                if method["family"] == "ridge"
                else [0.0]
                if method["family"] == "ols"
                else [0.0, 0.03, 0.3, 1.0]
            ):
                result = E.path(win, method, [candidate])[candidate]
                sx, sy, w, meta = V.transforms(x, y, span)
                labels = V.partition(sy, w, 0.6)
                hard, soft = win.restrictions(method)
                uni = univariate(sx, sy) if method["family"] == "unilasso" else None
                state = (sx, sy, w, meta, labels, uni)
                check_result = stationarity(
                    state,
                    method,
                    result["beta"],
                    result["alpha"],
                    result["parameter"],
                    result["lambda"],
                    hard,
                    soft,
                )
                check(check_result[0] < 1e-7 and check_result[1] < 1e-8, "Independent KKT failed")
                rb, ra, details = conic_reference(
                    sx, sy, w, labels, method, result["lambda"], hard, soft, uni
                )
                error = float(max(abs(rb - result["beta"]).max(), abs(ra - result["alpha"]).max()))
                check(
                    error < 3e-4,
                    f"Reference disagreement: {span}, {method['name']}, {candidate}, {error}",
                )
                maximum = max(maximum, error)
                cases.append(
                    dict(
                        span=span,
                        method=method["name"],
                        candidate=candidate,
                        error=error,
                        kkt=check_result[0],
                        reference=details,
                    )
                )
    sources = {
        ("papers/sign_pooling_2026/replication/" + n): sha(HERE / n)
        for n in (
            "matf_full_engine.py",
            "matf_full_study.py",
            "verify_matf_full.py",
            "tests/test_matf_full.py",
        )
    }
    output.mkdir(parents=True)
    write(output / "cases.json", cases)
    result = dict(
        status="passed",
        cases=len(cases),
        maximum_coefficient_error=maximum,
        sources=sources,
        cases_sha256=sha(output / "cases.json"),
    )
    write(output / "validation.json", result)
    return result


def verified_dataset(output, key):
    """Read a complete unit and independently verify every saved candidate and final fit."""
    start = time.perf_counter()
    output = Path(output)
    dest = output / "datasets" / key
    freeze = read(output / "freeze.json")
    p = read(output / "protocol.json")
    receipt = read(dest / "complete.json")
    check(receipt["run_identity"] == freeze["identity"], "Incorrect unit source identity")
    for name, ext in [("ledger", ".json"), ("arrays", ".npz")]:
        check(sha(dest / (name + ext)) == receipt[name + "_sha256"], "Corrupted unit artifact")
    ledger = read(dest / "ledger.json")
    cell = next(c for c in p["cells"] if c["key"] == ledger["cell"])
    r = ledger["replication"]
    n = cell["history"]
    step = n // 6
    with np.load(dest / "arrays.npz", allow_pickle=False) as f:
        a = dict(f)
    with np.load(output / "calibration.npz", allow_pickle=False) as f:
        cal = dict(f)
    rng = np.random.default_rng(np.random.SeedSequence([p["full_seed"], r]))
    xx = (
        rng.standard_normal((632, 11)) @ np.linalg.cholesky(cal["factor_covariance"]).T
        + cal["annual_premia"] / 12
    )
    eps = rng.standard_normal((632, 247)) @ np.linalg.cholesky(cal["residual_covariance"]).T
    beta = cal[cell["alignment"] + "_beta"] * (0.5 if cell["strength"] == "weak" else 1)
    evolution = np.ones(n)
    if cell["evolution"] == "reversal":
        evolution[n // 2 :] = 1 - 1.5 * np.arange(n - n // 2) / (n - n // 2 - 1)
    truth = beta * evolution[-1]
    close(a["x"], xx[120 - n : 120])
    close(a["x_test"], xx[120:])
    close(a["truth"], truth)
    close(a["y"], (xx[120 - n : 120] @ beta.T) * evolution[:, None] + eps[120 - n : 120])
    close(a["y_test"], xx[120:] @ truth.T + eps[120:])
    close(a["annual_premia"], cal["annual_premia"])
    planned = read(output / "planned_records.json")[cell["key"]]
    check(
        len(ledger["records"]) == len(planned)
        and {(v["method"], v["tuning"]) for v in ledger["records"]}
        == {(v["method"], v["tuning"]) for v in planned},
        "Missing/duplicate final records",
    )
    counts = Counter()
    maxima = dict(kkt=0.0, violation=0.0, reference_coefficient_error=0.0)
    cache = {}
    scales = {}
    gcache = {}
    refdetails = []

    def getstate(span, stop, uni_needed=False):
        wkey = f"w{span or 0}_n{stop}"
        if wkey not in cache:
            x, y, w, meta = V.transforms(a["x"][:stop], a["y"][:stop], span)
            labels = V.partition(y, w, 0.6)
            for name, val in meta.items():
                close(a[wkey + "_" + name], val)
            close(a[wkey + "_weights"], w)
            close(a[wkey + "_labels"], labels)
            cache[wkey] = (x, y, w, meta, labels, None)
        state = cache[wkey]
        if uni_needed and state[-1] is None:
            state = (*state[:-1], univariate(state[0], state[1]))
            cache[wkey] = state
        return state, wkey

    def gates(state, wkey):
        if wkey not in gcache:
            x, y, w, _, labels, _ = state
            found = {}
            for kind in ("marginal", "conditional", "singleton"):
                prefix = wkey + "_" + kind
                if prefix + "_signs" not in a:
                    continue
                gl = np.arange(len(labels)) if kind == "singleton" else labels
                b, se = (
                    partial_reference(x, y, w, gl)
                    if kind == "conditional"
                    else V.marginal_reference(x, y, w, gl)
                )
                stat = np.divide(b, se, out=np.zeros_like(b), where=se > 0)
                signs = np.where(abs(stat) >= 2, np.sign(b), 0.0)
                for name, val in dict(
                    signs=signs, slopes=b, standard_errors=se, t_stats=stat, labels=gl
                ).items():
                    close(a[prefix + "_" + name], val, kind + " gate")
                found[kind] = dict(signs=signs)
                counts["gate_snapshots"] += 1
            if not found:
                found["marginal"] = dict(signs=np.zeros_like(truth))
            gcache[wkey] = found
        return gcache[wkey]

    def checkfit(fkey, method, candidate, stored_lambda, diag, stop, reference=False):
        state, wkey = getstate(method["span"], stop, method["family"] == "unilasso")
        x, y, w, meta, labels, uni = state
        skey = (wkey, method["family"])
        if skey not in scales:
            scales[skey] = scale(x, y, w, labels, method["family"], uni)
        lam = "inf" if candidate == "inf" else candidate * scales[skey]
        if lam == "inf":
            check(stored_lambda == "inf", "Infinite endpoint missing")
        else:
            close(stored_lambda, lam, rtol=1e-8, atol=1e-10)
        hard, soft = restrictions(method, gates(state, wkey))
        parameter = a.get(fkey + "_parameter", a[fkey + "_beta"])
        check(np.isfinite(parameter).all(), "Nonfinite parameter")
        kkt, violation, obj = stationarity(
            state, method, a[fkey + "_beta"], a[fkey + "_alpha"], parameter, lam, hard, soft
        )
        check(kkt <= 1.01e-7 and violation <= 1e-8, "Independent stationarity failed " + fkey)
        if obj is not None:
            close(diag["objective"], obj, rtol=1e-8, atol=1e-8)
        maxima["kkt"] = max(maxima["kkt"], kkt)
        maxima["violation"] = max(maxima["violation"], violation)
        counts["checked_fits"] += 1
        if reference:
            # UniLasso/OLS/ridge are response-separable; deterministic representative indices.
            indices = (
                np.unique(np.linspace(0, len(labels) - 1, 12, dtype=int))
                if method["family"] in ("unilasso", "ridge", "ols", "lasso")
                else np.arange(len(labels))
            )
            u = (
                None
                if uni is None
                else (uni[0][:, indices], uni[1][indices], uni[2][indices], uni[3][indices])
            )
            rb, ra, details = conic_reference(
                x, y[:, indices], w, labels[indices], method, lam, hard[indices], soft[indices], u
            )
            error = float(
                max(
                    abs(rb - a[fkey + "_beta"][indices]).max(),
                    abs(ra - a[fkey + "_alpha"][indices]).max(),
                )
            )
            check(error <= 5e-4, "Full conic reference coefficient discrepancy " + fkey)
            maxima["reference_coefficient_error"] = max(
                maxima["reference_coefficient_error"], error
            )
            refdetails.append(dict(key=fkey, error=error, **details))
            counts["reference_fits"] += 1
        return state, hard, soft

    selections = {}
    method_map = {}
    for selector in ledger["selectors"]:
        method = selector["method"]
        method_map[method["name"]] = method
        grid = selector["grid"]
        entries = selector["folds"]
        check(len(entries) == 3 * len(grid), "Missing tuning candidates")
        check(
            {(v["stop"], v["candidate"]) for v in entries}
            == {(stop, c) for stop in (n // 2, n // 2 + step, n // 2 + 2 * step) for c in grid},
            "Invalid tuning folds",
        )
        scores = {}
        for entry in entries:
            state, _, _ = checkfit(
                entry["key"],
                method,
                entry["candidate"],
                entry["lambda_"],
                entry["diagnostics"],
                entry["stop"],
            )
            meta = state[3]
            stop = entry["stop"]
            fkey = entry["key"]
            rawb = a[fkey + "_beta"] * meta["y_sd"][:, None] / meta["x_sd"]
            rawa = meta["y_mean"] + a[fkey + "_alpha"] * meta["y_sd"] - rawb @ meta["x_mean"]
            pred = a["x"][stop : stop + step] @ rawb.T + rawa
            mse = float(np.mean(((pred - a["y"][stop : stop + step]) / meta["y_sd"]) ** 2))
            close(entry["mse"], mse)
            scores.setdefault(entry["candidate"], []).append(mse)
        scores = {c: np.mean(v) for c, v in scores.items()}

        def choose(g):
            return max(
                (c for c in g if scores[c] <= min(scores[v] for v in g) + 1e-12),
                key=lambda c: np.inf if c == "inf" else c,
            )

        family = method["family"]
        expected = (
            [0.0]
            if family == "ols"
            else list(p["ridge_grid"] if family == "ridge" else p["sparse_grid"])
        )
        if family == "ridge":
            for value in p["ridge_extensions"]:
                if choose(expected) != max(c for c in expected if c != "inf"):
                    break
                expected.insert(-1, value)
                check(value in grid, "Missing ridge extension")
        check(
            grid == expected and selector["selected"] == choose(grid), "Invalid penalty selection"
        )
        selections[method["name"]] = selector["selected"]
        counts["inner_fits"] += len(entries)
    expected_methods = {v["method"] for v in planned}
    check(set(method_map) == expected_methods, "Missing estimator selector")
    scores_out = {}
    sleeves_out = {}
    factors_out = {}
    for fkey, item in ledger["finals"].items():
        method = item["method"]
        family = method["family"]
        state, hard, soft = checkfit(
            fkey,
            method,
            item["candidate"],
            item["lambda_"],
            item["diagnostics"],
            n,
            reference=(r == 0),
        )
        meta = state[3]
        b = a[fkey + "_beta"] * meta["y_sd"][:, None] / meta["x_sd"]
        alpha = meta["y_mean"] + a[fkey + "_alpha"] * meta["y_sd"] - b @ meta["x_mean"]
        close(a[fkey + "_raw_beta"], b)
        close(a[fkey + "_raw_alpha"], alpha)
        pred = a["x_test"] @ b.T + alpha
        values, component = V.reference_losses(
            b,
            alpha,
            truth,
            cal["annual_premia"],
            cal["factor_covariance"],
            a["y_test"],
            pred,
            meta["y_sd"],
        )
        close(a[fkey + "_component_error"], component)
        for metric, value in values.items():
            close(a[fkey + "_" + metric], value)
            close(item["metrics"][metric], value.mean())
            counts["loss_values"] += len(value) + 1
        active = truth != 0
        retained = abs(a[fkey + "_beta"]) > 1e-6
        extra = dict(
            retention=retained.mean(),
            active_retention=retained[active].mean(),
            false_exclusion=np.mean((hard == 0)[active]),
            wrong_hard_restriction=np.mean(hard[active] * truth[active] < 0),
            wrong_fitted_direction=np.mean((b[active] * truth[active] < 0) & retained[active]),
        )
        for name, value in extra.items():
            close(item["metrics"][name], value)
        scores_out[fkey] = dict(
            item["metrics"],
            cluster_ari=float(V.ari(cal["truth_labels"], state[4])),
            clusters=int(len(np.unique(state[4]))),
        )
        sleeves_out[fkey] = {
            name: {
                metric: float(value[cal["truth_labels"] == i].mean())
                for metric, value in values.items()
            }
            for i, name in enumerate(("Equity", "Fixed Income", "Alternatives"))
        }
        factors_out[fkey] = {
            factor: dict(
                mse=float(np.mean(component[:, j] ** 2)), bias=float(np.mean(component[:, j]))
            )
            for j, factor in enumerate(p["factor_basis"])
        }
        counts["unique_final_fits"] += 1
    rows = []
    for record in ledger["records"]:
        method = method_map[record["method"]]
        fkey = record["key"]
        item = ledger["finals"][fkey]
        baseline = method["family"] + "_ungated" + ("_ew120" if method["span"] else "")
        expected = (
            selections[record["method"]] if record["tuning"] == "own" else selections[baseline]
        )
        check(item["candidate"] == expected, "Invalid matched/own final penalty")
        rows.append(dict(cell=cell["key"], replication=r, **record, **scores_out[fkey]))
    counts["final_records"] = len(rows)
    result = dict(
        status="passed",
        run_identity=freeze["identity"],
        dataset=key,
        receipt_sha256=sha(dest / "complete.json"),
        counts=dict(counts),
        maxima=maxima,
        seconds=time.perf_counter() - start,
        fit_seconds=ledger["seconds"],
        rows=rows,
        sleeves=sleeves_out,
        factors=factors_out,
        references=refdetails,
    )
    write(dest / "verified.json", result)
    return dict(dataset=key, counts=dict(counts), seconds=result["seconds"])


def aggregate(output):
    output = Path(output)
    p = read(output / "protocol.json")
    freeze = read(output / "freeze.json")
    check(identity(freeze["binding"]) == freeze["identity"], "Freeze identity changed")
    for name, h in freeze["binding"]["sources"].items():
        check(sha(REPO / name) == h, "Changed source " + name)
    expected = {f"{c['key']}__{r:03d}" for c in p["cells"] for r in range(200)}
    check(
        {d.name for d in (output / "datasets").iterdir() if d.is_dir()} == expected,
        "Dataset inventory",
    )
    rows = []
    sleeves = []
    factors = []
    tuning = []
    counts = Counter()
    maxima = {}
    runtime = []
    for key in sorted(expected):
        d = output / "datasets" / key
        v = read(d / "verified.json")
        receipt = read(d / "complete.json")
        check(
            v["status"] == "passed" and v["run_identity"] == freeze["identity"], "Unverified unit"
        )
        check(sha(d / "complete.json") == v["receipt_sha256"], "Changed unit receipt")
        for name, ext in [("ledger", ".json"), ("arrays", ".npz")]:
            check(sha(d / (name + ext)) == receipt[name + "_sha256"], "Changed unit data")
        rows.extend(v["rows"])
        counts.update(v["counts"])
        for name, value in v["maxima"].items():
            maxima[name] = max(maxima.get(name, 0), value)
        runtime.append(dict(dataset=key, fit_seconds=v["fit_seconds"], verify_seconds=v["seconds"]))
        unit_ledger = read(d / "ledger.json")
        for row in v["rows"]:
            fields = {k: row[k] for k in ("cell", "replication", "method", "tuning")}
            for sleeve, metrics in v["sleeves"][row["key"]].items():
                sleeves.append(dict(**fields, sleeve=sleeve, **metrics))
            for factor, metrics in v["factors"][row["key"]].items():
                factors.append(dict(**fields, factor=factor, **metrics))
        for selector in unit_ledger["selectors"]:
            tuning.append(
                dict(
                    cell=unit_ledger["cell"],
                    replication=unit_ledger["replication"],
                    method=selector["method"]["name"],
                    selected=selector["selected"],
                    candidates=len(selector["grid"]),
                    upper_exhausted=selector["upper_exhausted"],
                )
            )
    frame = pd.DataFrame(rows)
    for _, group in frame.groupby(["cell", "method", "tuning"]):
        check(sorted(group.replication) == list(range(200)), "Incomplete 200 paired replications")
    tables = output / "tables"
    tables.mkdir(exist_ok=True)
    for name, data in [
        ("scores", frame),
        ("sleeves", sleeves),
        ("factors", factors),
        ("tuning", tuning),
        ("runtime", runtime),
    ]:
        pd.DataFrame(data).to_csv(tables / (name + ".csv"), index=False)
    metrics = [
        "attribution_mse",
        "mean_mse",
        "full_mean_mse",
        "annual_prediction_risk",
        "test_mse",
        "standardized_test_mse",
        "loading_mse",
        "retention",
        "active_retention",
    ]
    summaries = []
    contrasts = []
    for (cell, method, tune), g in frame.groupby(["cell", "method", "tuning"]):
        base_method = ("fcgl_ungated" if method.startswith("fcgl") else "hcgl_ungated") + (
            "_ew120" if method.endswith("_ew120") else ""
        )
        baseline = frame[
            (frame.cell == cell) & (frame.method == base_method) & (frame.tuning == "own")
        ]
        for metric in metrics:
            values = g[metric].to_numpy()
            mean = float(values.mean())
            se = float(values.std(ddof=1) / np.sqrt(200))
            summaries.append(
                dict(
                    cell=cell,
                    method=method,
                    tuning=tune,
                    metric=metric,
                    mean=mean,
                    mcse=se,
                    low=mean - 1.96 * se,
                    high=mean + 1.96 * se,
                )
            )
            diff = (
                g.set_index("replication")[metric].sort_index()
                - baseline.set_index("replication")[metric].sort_index()
            ).to_numpy()
            dm = float(diff.mean())
            ds = float(diff.std(ddof=1) / np.sqrt(200))
            contrasts.append(
                dict(
                    cell=cell,
                    method=method,
                    tuning=tune,
                    metric=metric,
                    baseline=base_method,
                    mean_difference=dm,
                    mcse=ds,
                    low=dm - 1.96 * ds,
                    high=dm + 1.96 * ds,
                    percent_change=100 * dm / float(baseline[metric].mean()),
                    wins=int(np.sum(diff < 0)),
                )
            )
    pd.DataFrame(summaries).to_csv(tables / "summary.csv", index=False)
    pd.DataFrame(contrasts).to_csv(tables / "paired.csv", index=False)
    result = dict(
        status="passed",
        run_identity=freeze["identity"],
        datasets=3200,
        replications=200,
        counts=dict(counts),
        maxima=maxima,
        fit_worker_seconds=sum(v["fit_seconds"] for v in runtime),
        verification_worker_seconds=sum(v["verify_seconds"] for v in runtime),
        files={f.name: sha(f) for f in tables.glob("*.csv")},
    )
    write(output / "verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["validate", "dataset", "aggregate"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--key")
    args = parser.parse_args()
    result = (
        validation(args.out)
        if args.action == "validate"
        else verified_dataset(args.out, args.key)
        if args.action == "dataset"
        else aggregate(args.out)
    )
    print(json.dumps(result, indent=2))
