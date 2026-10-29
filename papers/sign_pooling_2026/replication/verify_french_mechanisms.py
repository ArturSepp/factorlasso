"""Independent references for the FF mechanism, support-budget and truth studies.

Does not import the producers or the fitting engine. Reconstructs chronological
selection, score variances, restrictions, losses and known-truth metrics directly.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

CHECKED = 0


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, atol=2e-10):
    global CHECKED
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=2e-9, equal_nan=True)
    CHECKED += np.asarray(expected).size


def canonical(a):
    result, seen = [], {}
    for v in a:
        if int(v) not in seen:
            seen[int(v)] = len(seen)
        result.append(seen[int(v)])
    return np.array(result)


def ward(y, cut):
    tree = linkage(pdist(y.T) / np.sqrt(len(y)), method="ward")
    return canonical(fcluster(tree, cut * tree[-1, 2], criterion="distance"))


def gate(x, y, labels, tau):
    result = np.zeros((y.shape[1], x.shape[1]))
    for group in np.unique(labels):
        members = np.flatnonzero(labels == group)
        for j in range(x.shape[1]):
            xx = x[:, j]
            b = sum(np.dot(xx, y[:, i]) for i in members) / (len(members) * np.dot(xx, xx))
            scores = np.array(
                [xx[k] * sum(y[k, i] - b * xx[k] for i in members) for k in range(len(x))]
            )
            var = (
                len(x)
                / (len(x) - 1)
                * np.dot(scores, scores)
                / (len(members) * np.dot(xx, xx)) ** 2
            )
            stat = abs(b) / np.sqrt(var) if var > 0 else (np.inf if b != 0 else 0)
            result[members, j] = np.sign(b) if stat >= tau else 0
    return result


def check_fit(x, y, beta, signs, labels, lam, family, objective=None):
    """Explicit cone and proximal optimality reference for row/block objectives."""
    projected = beta.copy()
    for i, j in np.ndindex(beta.shape):
        g = signs[i, j]
        if g == 0 or (g == 1 and projected[i, j] < 0) or (g == -1 and projected[i, j] > 0):
            projected[i, j] = 0
    close(beta, projected, atol=1.1e-7)
    weights = np.array([np.sqrt(np.sum(labels == v) / len(np.unique(labels))) for v in labels])
    residual = x @ beta.T - y
    value = float(np.sum(residual * residual) / len(x))
    pen = 0.5 * sum(abs(v) for v in beta.ravel())
    if family == "row":
        pen += 0.5 * sum(weights[i] * np.sqrt(np.dot(beta[i], beta[i])) for i in range(len(beta)))
    else:
        for label in np.unique(labels):
            group = np.flatnonzero(labels == label)
            pen += (
                0.5
                * weights[group[0]]
                * sum(np.linalg.norm(beta[group, j]) for j in range(beta.shape[1]))
            )
    value += lam * pen
    if objective is not None:
        close(value, objective, atol=2e-7 * (1 + abs(value)))
    grad = 2 * residual.T @ x / len(x)
    step = 1 / max(1.0, np.linalg.norm(2 * x.T @ x / len(x), 2))
    prox = beta - step * grad
    for i, j in np.ndindex(beta.shape):
        g = signs[i, j]
        if g == 0:
            prox[i, j] = 0
        elif g == 1:
            prox[i, j] = max(0.0, prox[i, j])
        elif g == -1:
            prox[i, j] = min(0.0, prox[i, j])
    prox = np.sign(prox) * np.maximum(abs(prox) - 0.5 * lam * step, 0)
    if family == "row":
        for i in range(len(prox)):
            norm = np.linalg.norm(prox[i])
            prox[i] *= max(0.0, 1 - 0.5 * weights[i] * lam * step / norm) if norm else 0
    else:
        for label in np.unique(labels):
            group = np.flatnonzero(labels == label)
            for j in range(prox.shape[1]):
                norm = np.linalg.norm(prox[group, j])
                prox[group, j] *= (
                    max(0.0, 1 - 0.5 * weights[group[0]] * lam * step / norm) if norm else 0
                )
    scaled = np.max(abs(beta - prox)) / step / (1 + np.max(abs(beta)) + np.max(abs(grad)))
    if scaled > 1.1e-5:
        raise ValueError(f"Independent stationarity failed: {scaled}")


def table(out, name, expected, keys):
    actual = pd.read_csv(out / (name + ".csv"), float_precision="round_trip")
    expected = pd.DataFrame(expected, columns=actual.columns if not len(expected) else None)
    if set(actual.columns) != set(expected.columns) or len(actual) != len(expected):
        raise ValueError("Table schema/coverage mismatch: " + name)
    actual = actual.sort_values(keys, kind="stable").reset_index(drop=True)
    expected = expected.sort_values(keys, kind="stable").reset_index(drop=True)
    for col in actual.columns:
        try:
            a, b = (
                pd.to_numeric(actual[col], errors="raise"),
                pd.to_numeric(expected[col], errors="raise"),
            )
        except (ValueError, TypeError):
            if actual[col].fillna("").tolist() != expected[col].fillna("").tolist():
                raise ValueError(f"Text/key mismatch: {name}/{col}")
        else:
            close(a.to_numpy(), b.to_numpy())


def receipt(folder, identity, name="receipt.json"):
    r = read(folder / name)
    if r["identity"] != identity or r["status"] != "complete":
        raise ValueError("Mixed or incomplete job")
    for file, value in r["files_sha256"].items():
        if sha(folder / file) != value:
            raise ValueError("Changed job artifact: " + str(folder / file))
    return r


def verify_real(out, frozen, p):
    binding = frozen["binding"]
    panel, fits = Path(binding["panel"]), Path(binding["fits"])
    original = read(panel / "protocol.json")
    x = pd.read_csv(
        panel / "panel/X_weekly.csv", index_col=0, parse_dates=True, float_precision="round_trip"
    )
    y = pd.read_csv(
        panel / "panel/Y_weekly.csv", index_col=0, parse_dates=True, float_precision="round_trip"
    )
    jobs = read(fits / "run_manifest.json")["jobs"]
    certificate = read(fits / "full_validation.json")
    if (
        sha(fits / "full_validation.json")
        != read(fits / "full_run_receipt.json")["validation_sha256"]
    ):
        raise ValueError("Changed original completeness certificate")
    signed = {
        (v["origin"], v["lookback"]): v["window_receipt_sha256"] for v in certificate["audits"]
    }
    manifest = read(out / "real_report_manifest.json")
    if manifest["identity"] != frozen["identity"] or set(manifest["window_receipt_sha256"]) != {
        j["folder"] for j in jobs
    }:
        raise ValueError("Incomplete real-study report")
    for name, value in manifest["files_sha256"].items():
        if sha(out / name) != value:
            raise ValueError("Changed real report")
    blocks, changes, precision, disagreement, conflicts, coherence, missing = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    previous = {}
    for job in jobs:
        folder = out / job["folder"]
        r = receipt(folder, frozen["identity"])
        if sha(folder / "receipt.json") != manifest["window_receipt_sha256"][job["folder"]]:
            raise ValueError("Changed real receipt")
        o, t = job["origin"], job["lookback"]
        old = fits / job["folder"]
        if sha(old / "window_receipt.json") != signed[o, t]:
            raise ValueError("Changed original window receipt")
        old_receipt = read(old / "window_receipt.json")
        for name, value in r["used_frozen_fits"].items():
            if old_receipt["files_sha256"].get(name) != value or sha(old / name) != value:
                raise ValueError("Changed reused original fit")
        a = dict(np.load(folder / "arrays.npz", allow_pickle=False))
        trainx, trainy = x.iloc[o - t : o].to_numpy(), y.iloc[o - t : o].to_numpy()
        xx = (trainx - trainx.mean(0)) / trainx.std(0)
        yy = (trainy - trainy.mean(0)) / trainy.std(0)
        close(a["x_train"], xx)
        close(a["y_train"], yy)
        close(a["training_sd"], trainy.std(0))
        close(a["training_mean"], trainy.mean(0))
        end = min(o + original["calendar"]["outer_horizon"], len(x))
        close(a["x_test"], x.iloc[o:end].to_numpy())
        close(a["actual"], y.iloc[o:end].to_numpy())
        if r["test_dates"] != x.index[o:end].strftime("%Y-%m-%d").tolist():
            raise ValueError("Wrong test calendar")
        selection, meta = read(folder / "selection.json"), read(folder / "fit_metadata.json")
        surface = selection["surface"]
        expected_candidates = []
        for method in original["methods"]:
            grouped = method.startswith(("hcgl", "fcgl")) or method == "cooperative"
            c = {
                "method": method,
                "partition": p["ward_cut"] if grouped else None,
                "tau": p["gate_tau"]
                if method
                in (
                    "singleton_exclude",
                    "hcgl_singleton_exclude",
                    "hcgl_exclude",
                    "hcgl_unrestricted",
                    "fcgl_exclude",
                )
                else None,
            }
            if method == "ols":
                expected_candidates.append(c)
            elif method == "ridge":
                expected_candidates.extend(
                    {**c, "ridge_lambda": v} for v in original["grid"]["ridge_lambdas"]
                )
            else:
                expected_candidates.extend({**c, "fraction": v} for v in p["fractions"])
        if [v["candidate"] for v in surface] != expected_candidates:
            raise ValueError("Missing/extra/reordered support-budget candidates")
        if len(selection["ledger"]) != 3 * len(surface):
            raise ValueError("Incomplete inner evaluation ledger")
        for row in surface:
            c = row["candidate"]
            records = [v for v in selection["ledger"] if v["candidate"] == c]
            if sorted(v["fold"] for v in records) != [0, 1, 2]:
                raise ValueError("Duplicate/missing inner fold")
            sse = cells = active = 0
            for v in records:
                fold = v["fold"]
                tr = original["calendar"]["inner_train_ends"][str(t)][fold]
                ve = original["calendar"]["inner_validation_ends"][str(t)][fold]
                if v["train_end"] != tr or v["validation_end"] != ve:
                    raise ValueError("Inner calendar leakage")
                with np.load(
                    folder / f"inner_{fold}_fits" / (v["fit_key"] + ".npz"), allow_pickle=False
                ) as cache:
                    beta, alpha = cache["beta"], cache["alpha"]
                    cm = json.loads(cache["metadata"].tobytes().decode())
                if cm["identity"] != frozen["identity"]:
                    raise ValueError("Mixed inner fit")
                xm, xs = trainx[:tr].mean(0), trainx[:tr].std(0)
                ym, ys = trainy[:tr].mean(0), trainy[:tr].std(0)
                pred = np.column_stack(
                    [((trainx[tr:ve] - xm) / xs) @ beta[i] + alpha[i] for i in range(len(beta))]
                )
                actual = (trainy[tr:ve] - ym) / ys
                val = float(np.einsum("ij,ij->", actual - pred, actual - pred))
                complexity = sum(abs(b) > p["support_tolerance"] for b in beta.ravel()) / beta.size
                close(v["sse"], val)
                close(v["active_fraction"], complexity)
                sse += val
                cells += actual.size
                active += complexity
            close(row["score"], sse / cells)
            close(row["inner_active_fraction"], active / 3)
        for name, value in selection["selected"].items():
            method = name.split("__")[1]
            candidates = [v for v in surface if v["candidate"]["method"] == method]
            if name.startswith("budget_"):
                budget = float(name.split("__")[0].replace("budget_", ""))
                candidates = [v for v in candidates if v["inner_active_fraction"] <= budget + 1e-12]
            if not candidates:
                if value is not None:
                    raise ValueError("Infeasible budget selection")
                missing.append(
                    {
                        "lookback": t,
                        "origin": o,
                        "method": name,
                        "reason": "no inner-budget-feasible candidate",
                    }
                )
            else:
                minimum = min(v["score"] for v in candidates)
                tol = original["selection"]["tie_atol"] + original["selection"]["tie_rtol"] * abs(
                    minimum
                )
                winner = next(v for v in candidates if v["score"] <= minimum + tol)
                if value["candidate"] != winner["candidate"]:
                    raise ValueError("Wrong training-only support-budget selection")
        labels = ward(yy, p["ward_cut"])
        ols = np.linalg.solve(xx.T @ xx, xx.T @ yy).T
        close(a["reference_ols_standardized"], ols)
        marginal = np.array(
            [
                [np.dot(xx[:, j], yy[:, i]) / len(xx) for j in range(xx.shape[1])]
                for i in range(yy.shape[1])
            ]
        )
        for threshold in p["reference_thresholds"]:
            for j, factor in enumerate(original["predictors"]):
                active = abs(ols[:, j]) > threshold
                disagreement.append(
                    dict(
                        lookback=t,
                        origin=o,
                        factor=factor,
                        tolerance=threshold,
                        reference_active=int(active.sum()),
                        opposite_marginal=int((active & (ols[:, j] * marginal[:, j] < 0)).sum()),
                    )
                )
        for group in np.unique(labels):
            members = np.flatnonzero(labels == group)
            if len(members) < 2:
                continue
            for j, factor in enumerate(original["predictors"]):
                z = xx[:, j]
                d = z @ z
                q = len(members)
                bs = np.array([z @ yy[:, i] / d for i in members])
                b = bs.mean()
                scores = np.column_stack([z * (yy[:, i] - b * z) for i in members])
                sums = np.sum(scores, axis=1)
                variance = (len(z) / (len(z) - 1)) * (sums @ sums) / (q * d) ** 2
                independent = (
                    (len(z) / (len(z) - 1))
                    * sum(np.dot(scores[:, k], scores[:, k]) for k in range(q))
                    / (q * d) ** 2
                )
                singleton = np.mean(
                    [
                        len(z)
                        / (len(z) - 1)
                        * np.dot(z * (yy[:, i] - bs[k] * z), z * (yy[:, i] - bs[k] * z))
                        / d**2
                        for k, i in enumerate(members)
                    ]
                )
                precision.append(
                    dict(
                        lookback=t,
                        origin=o,
                        cluster=int(group),
                        factor=factor,
                        size=q,
                        pooled_slope=b,
                        date_variance=variance,
                        independent_score_variance=independent,
                        mean_singleton_variance=singleton,
                        precision_ratio=np.sqrt(singleton / variance),
                        variance_inflation=variance / independent,
                    )
                )
        for name, info in meta.items():
            beta = a[name + "__beta"]
            alpha = a[name + "__intercept"]
            b = beta * trainx.std(0)[None, :] / trainy.std(0)[:, None]
            close(b, a[name + "__beta_standardized"])
            pred = np.column_stack([a["x_test"] @ beta[i] + alpha[i] for i in range(len(beta))])
            close(pred, a[name + "__prediction"])
            error = a["actual"] - pred
            loss = float(np.sum(np.square(error / trainy.std(0))) / len(beta))
            blocks.append(
                dict(
                    lookback=t,
                    origin=o,
                    method=name,
                    weeks=len(pred),
                    loss_sum=loss,
                    raw_sse=float(np.linalg.norm(error) ** 2),
                    active_fraction=float((abs(b) > p["support_tolerance"]).mean()),
                    clusters=len(np.unique(a[name + "__labels"])),
                    lambda_=info["lambda"],
                )
            )
            blocks[-1]["lambda"] = blocks[-1].pop("lambda_")
            if name.startswith("matched__"):
                _, family, variant = name.split("__")
                glabels = np.arange(len(beta)) if variant == "singleton_exclude" else labels
                signs = (
                    np.full_like(beta, np.nan)
                    if variant == "unconstrained"
                    else gate(
                        xx, yy, glabels, 0 if variant == "pooled_direction" else p["gate_tau"]
                    )
                )
                if variant == "pooled_release":
                    signs[signs == 0] = np.nan
                close(signs, a[name + "__gates"])
                close(info["lambda"], meta[f"matched__{family}__unconstrained"]["lambda"])
                check_fit(
                    xx,
                    yy,
                    b,
                    signs,
                    labels,
                    float(info["lambda"]),
                    "row" if family == "hcgl" else "block",
                    info["objective"],
                )
                for threshold in p["reference_thresholds"]:
                    for j, factor in enumerate(original["predictors"]):
                        active = abs(ols[:, j]) > threshold
                        conflicts.append(
                            dict(
                                lookback=t,
                                origin=o,
                                method=name,
                                factor=factor,
                                tolerance=threshold,
                                reference_active=int(active.sum()),
                                opposite_signed_gate=int(
                                    (active & (signs[:, j] * ols[:, j] < 0)).sum()
                                ),
                                excluded_reference_active=int((active & (signs[:, j] == 0)).sum()),
                            )
                        )
                        total = minority = eligible = mixed = 0
                        for g in np.unique(labels):
                            members = np.flatnonzero(labels == g)
                            if len(members) < 2:
                                continue
                            pos = sum(ols[i, j] > threshold for i in members)
                            neg = sum(ols[i, j] < -threshold for i in members)
                            total += pos + neg
                            minority += min(pos, neg)
                            eligible += int(pos + neg > 0)
                            mixed += int(pos > 0 and neg > 0)
                        coherence.append(
                            dict(
                                lookback=t,
                                origin=o,
                                method=name,
                                factor=factor,
                                tolerance=threshold,
                                active_non_singleton_cells=total,
                                minority_cells=minority,
                                eligible_groups=eligible,
                                mixed_groups=mixed,
                            )
                        )
            key = t, name
            if key in previous:
                pb, raw = previous[key]
                changes.append(
                    dict(
                        lookback=t,
                        origin=o,
                        method=name,
                        support_change=float(
                            np.mean(
                                (abs(b) > p["support_tolerance"])
                                != (abs(pb) > p["support_tolerance"])
                            )
                        ),
                        raw_loading_change=float(abs(beta - raw).mean()),
                    )
                )
            previous[key] = b, beta
    df = pd.DataFrame(blocks)
    table(out, "real_blocks", blocks, ["lookback", "origin", "method"])
    table(out, "stability", changes, ["lookback", "origin", "method"])
    table(out, "precision", precision, ["lookback", "origin", "cluster", "factor"])
    table(out, "marginal_partial", disagreement, ["lookback", "origin", "factor", "tolerance"])
    table(out, "gate_conflicts", conflicts, ["lookback", "origin", "method", "factor", "tolerance"])
    table(
        out, "partial_coherence", coherence, ["lookback", "origin", "method", "factor", "tolerance"]
    )
    table(out, "budget_unavailable", missing, ["lookback", "origin", "method"])
    metrics = []
    for (t, name), g in df.groupby(["lookback", "method"], sort=False):
        metrics.append(
            dict(
                lookback=t,
                method=name,
                windows=len(g),
                weeks=int(g.weeks.sum()),
                standardized_mse=g.loss_sum.sum() / g.weeks.sum(),
                standardized_rmse=np.sqrt(g.loss_sum.sum() / g.weeks.sum()),
                raw_rmse=np.sqrt(g.raw_sse.sum() / g.weeks.sum() / y.shape[1]),
                mean_active_fraction=g.active_fraction.mean(),
                non_singleton_windows=int((g.clusters < y.shape[1]).sum()),
            )
        )
    table(out, "real_metrics", metrics, ["lookback", "method"])
    stab = (
        pd.DataFrame(changes)
        .groupby(["lookback", "method"])[["support_change", "raw_loading_change"]]
        .mean()
        .reset_index()
        .merge(pd.DataFrame(metrics), on=["lookback", "method"])
    )
    table(out, "stability_error", stab.to_dict("records"), ["lookback", "method"])
    expected = []
    for t in original["calendar"]["lookbacks"]:
        groups = {m: g.sort_values("origin") for m, g in df[df.lookback == t].groupby("method")}
        n = len(next(iter(groups.values())))
        rng = np.random.default_rng(p["bootstrap"]["seed"])
        starts = rng.integers(0, n, size=(p["bootstrap"]["resamples"], n))
        uniforms = rng.random(starts.shape)
        multiplicity = []
        for r in range(len(starts)):
            path = [int(starts[r, 0])]
            for k in range(1, n):
                path.append(
                    int(starts[r, k])
                    if uniforms[r, k] < 1 / p["bootstrap"]["mean_blocks"][str(t)]
                    else (path[-1] + 1) % n
                )
            multiplicity.append(np.bincount(path, minlength=n))
        weights = np.array(multiplicity)
        for family in p["matched_families"]:
            for variant, baseline in p["matched_pairs"]:
                reference = f"matched__{family}__{baseline}"
                base = groups[reference]
                name = f"matched__{family}__{variant}"
                g = groups[name]
                delta = g.loss_sum.to_numpy() - base.loss_sum.to_numpy()
                counts = g.weeks.to_numpy()
                sample = (weights @ delta) / (weights @ counts)
                lo, hi = np.percentile(sample, [2.5, 97.5])
                expected.append(
                    dict(
                        lookback=t,
                        method=name,
                        reference=reference,
                        difference=delta.sum() / counts.sum(),
                        lower=lo,
                        upper=hi,
                    )
                )
    table(out, "real_contrasts", expected, ["lookback", "method", "reference"])
    return len(jobs)


def ari(a, b):
    # Pair counts rather than the producer's contingency formulation.
    aa = bb = both = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        samea, sameb = a[i] == a[j], b[i] == b[j]
        aa += samea
        bb += sameb
        both += samea and sameb
    total = len(a) * (len(a) - 1) / 2
    expected = aa * bb / total
    den = (aa + bb) / 2 - expected
    return (both - expected) / den if den else 1.0


def verify_simulation(out, frozen, p):
    sim = p["simulation"]
    cal = dict(np.load(out / "calibration.npz", allow_pickle=False))
    rawx, rawy = cal["calibration_x"], cal["calibration_y"]
    panel = Path(frozen["binding"]["panel"])
    sourcex = (
        pd.read_csv(panel / "panel/X_weekly.csv", index_col=0, float_precision="round_trip")
        .iloc[: sim["calibration_rows"]]
        .to_numpy()
    )
    sourcey = (
        pd.read_csv(panel / "panel/Y_weekly.csv", index_col=0, float_precision="round_trip")
        .iloc[: sim["calibration_rows"]]
        .to_numpy()
    )
    close(rawx, (sourcex - sourcex.mean(0)) / sourcex.std(0))
    close(rawy, (sourcey - sourcey.mean(0)) / sourcey.std(0))
    labels = pd.read_csv(panel / "panel/portfolio_labels.csv")
    expected_labels = canonical(
        2 * (labels.size_decile.to_numpy() > 5) + (labels.bm_decile.to_numpy() > 5)
    )
    np.testing.assert_array_equal(cal["truth_labels"], expected_labels)
    seed = np.linalg.solve(rawx.T @ rawx, rawx.T @ rawy).T
    residual = rawy - rawx @ seed.T
    sigma = (residual.T @ residual) / len(residual)
    close(cal["noise_covariance"], 0.9 * sigma + 0.1 * np.diag(np.diag(sigma)))
    seed[abs(seed) < 0.05] = 0
    close(cal["beta_seed"], seed)
    manifest = read(out / "simulation_manifest.json")
    if manifest["calibration_sha256"] != sha(out / "calibration.npz"):
        raise ValueError("Changed calibration artifact")
    for name, value in manifest["files_sha256"].items():
        if sha(out / name) != value:
            raise ValueError("Changed simulation report")
    expected_jobs = [
        (t, r, c, fx, ex)
        for t, r, c, fx, ex in itertools.product(
            sim["lookbacks"],
            range(sim["replications"]),
            sim["coherence"],
            sim["factor_correlation_mix"],
            sim["error_dependence_mix"],
        )
    ]
    if manifest["jobs"] != len(expected_jobs) or len(manifest["receipt_sha256"]) != len(
        expected_jobs
    ):
        raise ValueError("Incomplete simulation")
    rows = []
    for t, rep, coh, fx, ex in expected_jobs:
        key = f"T{t}_r{rep:03d}_{coh}_fx{fx}_ex{ex}"
        folder = out / "simulation" / key
        receipt(folder, frozen["identity"])
        if sha(folder / "receipt.json") != manifest["receipt_sha256"].get(key):
            raise ValueError("Changed simulation receipt")
        a = dict(np.load(folder / "arrays.npz", allow_pickle=False))
        beta = seed.copy()
        if coh == "coherent":
            for g in np.unique(cal["truth_labels"]):
                members = np.flatnonzero(cal["truth_labels"] == g)
                for j in range(beta.shape[1]):
                    direction = -1 if beta[members, j].sum() < 0 else 1
                    beta[members, j] = direction * abs(beta[members, j])
        corr = (1 - fx) * np.eye(seed.shape[1]) + fx * cal["factor_corr"]
        for i in range(len(beta)):
            beta[i] *= np.sqrt(cal["signal_variance"][i] / (beta[i] @ corr @ beta[i]))
        noise = (1 - ex) * np.diag(np.diag(cal["noise_covariance"])) + ex * cal["noise_covariance"]
        close(a["truth"], beta)
        close(a["factor_cov"], corr)
        close(a["error_cov"], noise)
        rng = np.random.default_rng(np.random.SeedSequence([sim["seed"], t, rep]))
        zx, ze, zt, zet = [
            rng.standard_normal(shape)
            for shape in (
                (t, beta.shape[1]),
                (t, len(beta)),
                (sim["test_rows"], beta.shape[1]),
                (sim["test_rows"], len(beta)),
            )
        ]
        xx = zx @ np.linalg.cholesky(corr).T
        xt = zt @ np.linalg.cholesky(corr).T
        yy = xx @ beta.T + ze @ np.linalg.cholesky(noise).T
        yt = xt @ beta.T + zet @ np.linalg.cholesky(noise).T
        close(a["x"], xx)
        close(a["y"], yy)
        close(a["x_test"], xt)
        close(a["y_test"], yt)
        xs = (xx - xx.mean(0)) / xx.std(0)
        ys = (yy - yy.mean(0)) / yy.std(0)
        active = beta != 0
        na = int(active.sum())
        nn = int((~active).sum())
        saved = read(folder / "metrics.json")
        ledger = read(folder / "solver_ledger.json")
        expected_names = {
            f"{part}__{fam}__{frac}__{variant}"
            for part, fam, frac, variant in itertools.product(
                sim["partitions"], sim["families"], sim["penalty_fractions"], sim["variants"]
            )
        }
        actual_names = [
            f"{v['partition']}__{v['family']}__{v['fraction']}__{v['variant']}" for v in saved
        ]
        if (
            set(actual_names) != expected_names
            or len(actual_names) != len(expected_names)
            or set(ledger) != expected_names
        ):
            raise ValueError("Missing/duplicate simulation variant")
        if len(saved) != len(sim["partitions"]) * len(sim["families"]) * len(
            sim["penalty_fractions"]
        ) * len(sim["variants"]):
            raise ValueError("Missing simulation variant")
        signs_cache = {}
        labels_cache = {
            "known_quadrants": cal["truth_labels"],
            "learned_ward": ward(ys, p["ward_cut"]),
        }
        ari_cache = {name: ari(labs, cal["truth_labels"]) for name, labs in labels_cache.items()}
        for row in saved:
            partition, family, fraction, variant = (
                row[k] for k in ("partition", "family", "fraction", "variant")
            )
            name = f"{partition}__{family}__{fraction}__{variant}"
            labels = labels_cache[partition]
            np.testing.assert_array_equal(a[name + "__labels"], labels)
            gc = (partition, variant)
            if gc not in signs_cache:
                signs = (
                    np.full_like(beta, np.nan)
                    if variant == "unconstrained"
                    else gate(
                        xs,
                        ys,
                        np.arange(len(beta)) if variant == "singleton_exclude" else labels,
                        0 if variant == "pooled_direction" else p["gate_tau"],
                    )
                )
                if variant == "pooled_release":
                    signs[signs == 0] = np.nan
                signs_cache[gc] = signs
            signs = signs_cache[gc]
            close(a[name + "__gates"], signs)
            bh, intercept = a[name + "__beta"], a[name + "__intercept"]
            delta = bh - beta
            prediction = np.column_stack([xt @ bh[i] + intercept[i] for i in range(len(beta))])
            selected = abs(bh) > p["support_tolerance"]
            tp = int((selected & active).sum())
            fp = int((selected & ~active).sum())
            expected = {
                "loading_mse": float(np.linalg.norm(delta) ** 2 / delta.size),
                "excess_risk": float(
                    (np.linalg.norm(delta @ np.linalg.cholesky(corr)) ** 2 + sum(intercept**2))
                    / len(beta)
                ),
                "test_mse": float(np.linalg.norm(yt - prediction) ** 2 / yt.size),
                "correct_active_fitted_sign": float(
                    (active & selected & (bh * beta > 0)).sum() / na
                ),
                "support_precision": tp / (tp + fp) if tp + fp else np.nan,
                "support_recall": tp / na,
                "support_false_positive_rate": fp / nn if nn else np.nan,
                "active_fraction": selected.mean(),
                "gate_correct_active": float((active & (signs * beta > 0)).sum() / na),
                "gate_opposite_active": float((active & (signs * beta < 0)).sum() / na),
                "false_exclusion": float((active & (signs == 0)).sum() / na),
                "active_unrestricted": float((active & np.isnan(signs)).sum() / na),
                "true_active_cells": na,
                "true_null_cells": nn,
                "ari": ari_cache[partition],
            }
            for metric, value in expected.items():
                close(np.nan if row[metric] is None else row[metric], value)
            bstd = bh * xx.std(0)[None, :] / yy.std(0)[:, None]
            check_fit(
                xs,
                ys,
                bstd,
                signs,
                labels,
                row["lambda"],
                "row" if family == "hcgl" else "block",
                ledger[name]["objective"],
            )
            rows.append({**row, **expected})
    table(
        out,
        "simulation_replications",
        rows,
        [
            "lookback",
            "replication",
            "coherence",
            "fx",
            "ex",
            "partition",
            "family",
            "fraction",
            "variant",
        ],
    )
    df = pd.DataFrame(rows)
    summary = pd.read_csv(out / "simulation_summary.csv", float_precision="round_trip")
    keys = ["lookback", "coherence", "fx", "ex", "partition", "family", "fraction"]
    metric_names = [
        "loading_mse",
        "excess_risk",
        "test_mse",
        "correct_active_fitted_sign",
        "support_precision",
        "support_recall",
        "support_false_positive_rate",
        "active_fraction",
        "gate_correct_active",
        "gate_opposite_active",
        "false_exclusion",
        "active_unrestricted",
    ]
    group_cache = {key: g for key, g in df.groupby(keys)}
    expected_summary = {
        (*key, variant, metric)
        for key in group_cache
        for variant in sim["variants"]
        for metric in metric_names
    }
    actual_summary = [
        tuple(r[k] for k in keys + ["variant", "metric"]) for r in summary.to_dict("records")
    ]
    if set(actual_summary) != expected_summary or len(actual_summary) != len(expected_summary):
        raise ValueError("Missing/duplicate Monte Carlo summary row")
    for row in summary.to_dict("records"):
        g = group_cache[tuple(row[k] for k in keys)]
        value = g[g.variant == row["variant"]].set_index("replication")[row["metric"]].sort_index()
        reference = (
            g[g.variant == "unconstrained"].set_index("replication")[row["metric"]].sort_index()
        )
        if len(value) != sim["replications"] or row["replications"] != len(value):
            raise ValueError("Wrong Monte Carlo replication unit")
        difference = (value - reference).dropna()
        value = value.dropna()
        close(row["defined"], len(value))
        mean = value.sum() / len(value) if len(value) else np.nan
        se = (
            np.sqrt(sum((value - mean) ** 2) / (len(value) * (len(value) - 1)))
            if len(value) > 1
            else np.nan
        )
        dm = difference.sum() / len(difference) if len(difference) else np.nan
        dse = (
            np.sqrt(sum((difference - dm) ** 2) / (len(difference) * (len(difference) - 1)))
            if len(difference) > 1
            else np.nan
        )
        for name, v in dict(
            mean=mean,
            mc_se=se,
            paired_difference=dm,
            paired_mc_se=dse,
            paired_lower=dm - 1.96 * dse,
            paired_upper=dm + 1.96 * dse,
        ).items():
            close(row[name], v)
    paired = []
    for key, g in group_cache.items():
        for variant, reference in p["pooling_pairs"]:
            left = g[g.variant == variant].set_index("replication").sort_index()
            right = g[g.variant == reference].set_index("replication").sort_index()
            for metric in metric_names:
                difference = (left[metric] - right[metric]).dropna()
                mean = sum(difference) / len(difference) if len(difference) else np.nan
                se = (
                    np.sqrt(
                        sum((difference - mean) ** 2) / (len(difference) * (len(difference) - 1))
                    )
                    if len(difference) > 1
                    else np.nan
                )
                paired.append(
                    {
                        **dict(zip(keys, key)),
                        "variant": variant,
                        "reference": reference,
                        "metric": metric,
                        "defined": len(difference),
                        "difference": mean,
                        "mc_se": se,
                        "lower": mean - 1.96 * se,
                        "upper": mean + 1.96 * se,
                    }
                )
    table(out, "simulation_pooling_contrasts", paired, keys + ["variant", "reference", "metric"])
    return len(expected_jobs)


def verify(out):
    global CHECKED
    CHECKED = 0
    frozen = read(out / "freeze.json")
    p = read(out / "protocol.json")
    if sha(out / "protocol.json") != frozen["binding"]["protocol_sha256"]:
        raise ValueError("Changed protocol")
    repo = Path(__file__).resolve().parents[3]
    for name, value in frozen["binding"]["source_sha256"].items():
        if sha(repo / name) != value or sha(out / "source" / name) != value:
            raise ValueError("Source differs from mechanism freeze")
    panel = Path(frozen["binding"]["panel"])
    if sha(panel / "manifest.json") != frozen["binding"]["panel_manifest_sha256"]:
        raise ValueError("Changed frozen panel manifest")
    panel_manifest = read(panel / "manifest.json")
    for name in ("panel/X_weekly.csv", "panel/Y_weekly.csv", "panel/portfolio_labels.csv"):
        if sha(panel / name) != panel_manifest["files_sha256"][name]:
            raise ValueError("Changed panel input")
    real = verify_real(out, frozen, p)
    sim = verify_simulation(out, frozen, p)
    result = {
        "status": "passed",
        "identity": frozen["identity"],
        "real_windows": real,
        "simulation_datasets": sim,
        "checked_values": CHECKED,
        "selection_losses_gates_precision_truth_metrics_independently_recomputed": True,
        "source_sha256": sha(Path(__file__)),
        "report_manifest_sha256": {
            name: sha(out / name)
            for name in ("real_report_manifest.json", "simulation_manifest.json")
        },
    }
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().out), indent=2))
