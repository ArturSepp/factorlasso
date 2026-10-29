"""Independent references for Stage 1B, without importing comparator internals.

FWL residual regressions reconstruct conditional slopes and date-cluster scores.
A direct normal-equation/FISTA implementation checks the convex objectives.
The validation command imports the producer only to obtain the points to check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd


def partial_reference(x, y, weights, labels):
    """FWL per factor, with stacked response scores summed before squaring."""
    n, k = x.shape
    slopes, ses = np.zeros((y.shape[1], k)), np.zeros((y.shape[1], k))
    root = np.sqrt(weights)
    for group in np.unique(labels):
        ids = np.flatnonzero(labels == group)
        mean = y[:, ids].mean(1)
        full = np.column_stack([np.ones(n), x])
        fit = np.linalg.lstsq(root[:, None] * full, root * mean, rcond=None)[0]
        # Sum individual member residuals, not their squares.
        residual_sum = (y[:, ids] - full @ fit[:, None]).sum(1)
        for j in range(k):
            other = np.column_stack([np.ones(n), np.delete(x, j, axis=1)])
            projection = np.linalg.lstsq(root[:, None] * other, root * x[:, j], rcond=None)[0]
            partial = x[:, j] - other @ projection
            denominator = np.dot(weights, partial**2)
            slope = np.dot(weights * partial, mean) / denominator
            influence = weights * partial * residual_sum / (len(ids) * denominator)
            se = np.sqrt(np.dot(influence, influence) * n / (n - k - 1))
            slopes[ids, j], ses[ids, j] = slope, se
    return slopes, ses


def reference_fit(x, y, w, labels, family, lam, hard, soft, strength, zero_strength):
    """Independent Gram-matrix accelerated proximal solver; no package prox import."""
    n, k = x.shape
    q = y.shape[1]
    gram = (x.T * w) @ x / n
    cross = (y.T * w) @ x / n
    groups = [np.flatnonzero(labels == g) for g in np.unique(labels)]
    group_weights = [np.sqrt(len(ids) / len(groups)) for ids in groups]
    step = 1 / (2 * np.linalg.eigvalsh(gram)[-1] + 2 * max(strength, zero_strength))
    direction = np.nan_to_num(soft, nan=0.0)
    zero_mask = np.isfinite(soft) & (soft == 0)

    def prox(b):
        b = b.copy()
        b[hard == 0] = 0
        b[hard == 1] = np.maximum(b[hard == 1], 0)
        b[hard == -1] = np.minimum(b[hard == -1], 0)
        b = np.sign(b) * np.maximum(abs(b) - step * lam / 2, 0)
        for ids, gw in zip(groups, group_weights):
            pieces = (
                [((i,), tuple(range(k))) for i in ids]
                if family == "hcgl"
                else [(tuple(ids), (j,)) for j in range(k)]
            )
            for rows, cols in pieces:
                ix = np.ix_(rows, cols)
                norm = np.linalg.norm(b[ix])
                b[ix] = 0 if norm == 0 else b[ix] * max(0, 1 - step * lam * gw / (2 * norm))
        return b

    def gradient(b):
        wrong = direction * b < 0
        return 2 * (b @ gram - cross) + 2 * strength * wrong * b + 2 * zero_strength * zero_mask * b

    current, accelerated, momentum = np.zeros((q, k)), np.zeros((q, k)), 1.0
    for iteration in range(1, 100001):
        updated = prox(accelerated - step * gradient(accelerated))
        mapping = (updated - prox(updated - step * gradient(updated))) / step
        if np.max(abs(mapping)) < 2e-9:
            return updated, iteration
        next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        accelerated = updated + (momentum - 1) / next_momentum * (updated - current)
        current, momentum = updated, next_momentum
    raise AssertionError("Independent reference failed to converge")


def validate(output):
    """Synthetic implementation validation only; no financial pilot outcomes."""
    if __package__:
        from . import matf_comparators as M
        from .french_attribution import external, sha, write
    else:
        import matf_comparators as M
        from french_attribution import external, sha, write
    output = external(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh validation directory")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(712022)
    x = rng.normal(size=(90, 3)) @ np.array([[1, 0.7, 0.3], [0, 0.7, -0.2], [0, 0, 0.8]])
    true = np.array([[0.7, -0.3, 0.1], [0.6, -0.2, 0], [-0.2, 0.8, -0.4], [0, 0.7, -0.5]])
    common = rng.normal(size=(90, 1)) * 0.2
    y = x @ true.T + common + rng.normal(size=(90, 4)) * 0.1
    dates = pd.date_range("2000-01-31", periods=90, freq="ME")
    x, y = pd.DataFrame(x, index=dates), pd.DataFrame(y, index=dates)
    records, arrays = [], {}
    started = time.perf_counter()
    for span in (None, 120):
        data = M.prepare(x, y, span)
        labels = np.array([0, 0, 1, 1])
        conditional = M.gate(data, labels, "conditional")
        slopes, se = partial_reference(data.state.x, data.state.y, data.weights, labels)
        np.testing.assert_allclose(conditional["slopes"], slopes, atol=1e-12, rtol=1e-11)
        np.testing.assert_allclose(conditional["standard_errors"], se, atol=1e-12, rtol=1e-11)
        for lam in (0, 0.3, 3):
            fitted = M.ridge_fit(data, lam)["beta"]
            a = np.vstack([data.weighted.x, np.sqrt(len(x) * lam) * np.eye(3)])
            target = np.vstack([data.weighted.y, np.zeros((3, 4))])
            reference = np.linalg.lstsq(a, target, rcond=None)[0].T
            np.testing.assert_allclose(fitted, reference, atol=1e-11)
        directions = np.array([[1, 1, 0], [1, 1, 0], [-1, -1, 1], [-1, 0, 1.0]])
        for family in ("hcgl", "fcgl"):
            spec = M.specification(data, family, labels=labels)
            for case in (
                "ungated",
                "soft_direction",
                "soft_exclusion",
                "hard_release",
                "hard_exclude",
                "conditional_release",
            ):
                hard = np.full_like(directions, np.nan)
                soft, strength, zero_strength = directions, 0.0, 0.0
                if case.startswith("soft"):
                    strength = 2.0
                    zero_strength = 3.0 if case == "soft_exclusion" else 0.0
                if case.startswith("hard"):
                    hard = M.hard_signs(directions, case.split("_")[1])
                if case == "conditional_release":
                    hard = M.hard_signs(conditional["signs"], "release")
                lam = 0.1 * spec.scale
                fitted = M.fit(
                    data,
                    spec,
                    lam,
                    hard=hard,
                    soft=soft,
                    strength=strength,
                    zero_strength=zero_strength,
                )
                reference, iterations = reference_fit(
                    data.state.x,
                    data.state.y,
                    data.weights,
                    labels,
                    family,
                    lam,
                    hard,
                    soft,
                    strength,
                    zero_strength,
                )
                np.testing.assert_allclose(fitted["beta"], reference, atol=3e-5, rtol=1e-5)
                key = f"span{span}_{family}_{case}"
                arrays[key + "__beta"], arrays[key + "__reference"] = fitted["beta"], reference
                records.append(
                    dict(
                        case=key,
                        max_coefficient_error=float(np.max(abs(fitted["beta"] - reference))),
                        reference_iterations=iterations,
                        **{
                            k: fitted[k]
                            for k in ("attempts", "kkt_residual", "constraint_violation")
                        },
                    )
                )
    # Match the intended panel dimensions without using financial pilot draws or outcomes.
    rng = np.random.default_rng(809631)
    xbig = rng.normal(size=(60, 11))
    xbig[:, 1:] += 0.4 * xbig[:, :1]
    ybig = xbig @ rng.normal(size=(11, 247)) * 0.2 + rng.normal(size=(60, 247)) * 0.4
    dates = pd.date_range("2000-01-31", periods=60, freq="ME")
    data = M.prepare(pd.DataFrame(xbig, index=dates), pd.DataFrame(ybig, index=dates))
    for family in ("hcgl", "fcgl"):
        spec = M.specification(data, family)
        evidence = M.gate(data, spec.labels, "conditional")
        b, se = partial_reference(data.state.x, data.state.y, data.weights, spec.labels)
        np.testing.assert_allclose(evidence["slopes"], b, atol=2e-12)
        np.testing.assert_allclose(evidence["standard_errors"], se, atol=2e-12)
        lam = 0.1 * spec.scale
        fitted = M.fit(data, spec, lam, soft=evidence["signs"], strength=1.0)
        reference, iterations = reference_fit(
            data.state.x,
            data.state.y,
            data.weights,
            spec.labels,
            family,
            lam,
            np.full_like(spec.signs, np.nan),
            evidence["signs"],
            1.0,
            0.0,
        )
        np.testing.assert_allclose(fitted["beta"], reference, atol=3e-5, rtol=1e-5)
        key = f"panel247x11_{family}_soft"
        arrays[key + "__beta"], arrays[key + "__reference"] = fitted["beta"], reference
        records.append(
            dict(
                case=key,
                max_coefficient_error=float(np.max(abs(fitted["beta"] - reference))),
                reference_iterations=iterations,
                **{k: fitted[k] for k in ("attempts", "kkt_residual", "constraint_violation")},
            )
        )
    np.savez_compressed(output / "validation_arrays.npz", **arrays)
    write(output / "validation_cases.json", records)
    source = {p.name: sha(p) for p in (Path(M.__file__), Path(__file__))}
    result = dict(
        status="passed",
        scope="Synthetic implementation validation; no pilot run",
        comparisons=len(records),
        conditional_reference_cases=4,
        ridge_reference_cases=6,
        max_coefficient_error=max(r["max_coefficient_error"] for r in records),
        wall_seconds=time.perf_counter() - started,
        source_sha256=source,
        files_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()},
    )
    write(output / "verification.json", result)
    return result


def verify_pilot(output):
    """Authenticate the freeze and independently rebuild its financial truth parameters."""
    output = Path(output).resolve()
    root = Path(__file__).resolve().parents[3]
    import factorlasso

    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def sha(path):
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

    def check(condition, message):
        if not condition:
            raise ValueError(message)

    check(
        not any(p.casefold().startswith("onedrive") for p in output.parts),
        "Verification outputs must be outside OneDrive",
    )
    freeze = read(output / "freeze.json")
    b = freeze["binding"]
    check(
        Path(factorlasso.__file__).resolve()
        == Path(b["factorlasso_import"])
        == (root / "src/factorlasso/__init__.py").resolve(),
        "Changed estimator import source",
    )
    identity = hashlib.sha256(json.dumps(b, sort_keys=True, allow_nan=False).encode()).hexdigest()
    check(identity == freeze["identity"], "Changed freeze identity")
    for name in ("protocol", "calibration", "planned_fits"):
        suffix = ".npz" if name == "calibration" else ".json"
        check(sha(output / (name + suffix)) == b[name + "_sha256"], "Changed frozen artifact")
    for prefix, files in (
        (root, b["source_sha256"]),
        (output / "source", b["source_sha256"]),
        (root, b["unchanged_estimator_sha256"]),
    ):
        for name, expected in files.items():
            check(sha(prefix / name) == expected, "Changed source: " + name)
    for path, expected in b["inputs_sha256"].items():
        check(sha(Path(path)) == expected, "Changed financial input")
    validation = Path(b["validation_path"])
    check(
        sha(validation / "verification.json") == b["validation_sha256"],
        "Changed implementation validation",
    )
    p, jobs = read(output / "protocol.json"), read(output / "planned_fits.json")
    check(not p["comparative_pilot_run"], "Protocol falsely labels pilot outcomes")
    files = {Path(path).name: Path(path) for path in b["inputs_sha256"]}
    x, y = [
        pd.read_csv(files[name], index_col=0, parse_dates=True, float_precision="round_trip").loc[
            "2012-01-31":"2016-12-31"
        ]
        for name in ("factor_logreturns.csv", "asset_excess_logreturns.csv")
    ]
    membership = pd.read_csv(files["explicit_class_membership.csv"], index_col=0)
    premiums = pd.read_csv(
        files["premia.csv"], index_col=0, parse_dates=True, float_precision="round_trip"
    )
    selected = premiums.loc[:"2016-11-30"].iloc[-1]
    cal = p["calibration"]
    check(cal["assets"] == list(y.columns) and cal["factors"] == list(x.columns), "Basis differs")
    check(x.index.equals(y.index) and len(x) == 60, "Calibration window differs")
    check(str(selected.name.date()) == cal["premium_observation"], "Premium availability differs")
    xx, yy = x.to_numpy(), y.to_numpy()
    xs, ys = np.std(xx, axis=0), np.std(yy, axis=0)
    cov = np.cov(xx, rowvar=False, bias=True)
    corr = cov / np.outer(xs, xs)
    design = np.column_stack([np.ones(len(x)), xx])
    coefficients = np.linalg.lstsq(design, yy, rcond=None)[0]
    proxy = coefficients[1:].T
    residuals = yy - design @ coefficients
    rcov = residuals.T @ residuals / len(x)
    residual_cov = 0.9 * rcov + 0.1 * np.diag(np.diag(rcov))
    signal = np.diag(proxy @ cov @ proxy.T)
    standardized_proxy = proxy * xs / ys[:, None]
    reference = dict(
        calibration_x=xx,
        calibration_y=yy,
        annual_premia=selected.to_numpy(),
        factor_sd=xs,
        response_sd=ys,
        factor_covariance=cov,
        residual_covariance=residual_cov,
        calibrated_signal_variance=signal,
        proxy_ols=standardized_proxy,
        truth_labels=np.argmax(membership.to_numpy(), axis=1),
    )
    aligned, conflict = np.zeros_like(proxy), np.zeros_like(proxy)
    for group in range(3):
        ids = np.flatnonzero(reference["truth_labels"] == group)
        median = np.median(standardized_proxy[ids], axis=0)
        support = sorted(range(len(xs)), key=lambda j: (-abs(median[j]), j))[:3]
        local = corr[np.ix_(support, support)]
        _, vecs = np.linalg.eigh(local)
        a = vecs[:, -1]
        a *= 1 if a @ median[support] >= 0 else -1
        pairs = [(i, j) for i in range(3) for j in range(i + 1, 3)]
        first, second = max(pairs, key=lambda ij: abs(local[ij]))
        c = 0.1 * np.sign(a)
        c[first], c[second] = 1.0, -2 / local[first, second]
        c *= np.sign(a[first])
        for matrix, vector in ((aligned, a), (conflict, c)):
            raw = np.zeros(len(xs))
            raw[support] = vector / xs[support]
            matrix[ids] = np.sqrt(signal[ids] / (raw @ cov @ raw))[:, None] * raw
    reference["aligned_beta"], reference["conflict_beta"] = aligned, conflict
    with np.load(output / "calibration.npz", allow_pickle=False) as stored:
        check(set(stored.files) == set(reference), "Calibration inventory differs")
        for key, expected in reference.items():
            np.testing.assert_allclose(stored[key], expected, rtol=1e-10, atol=1e-12, err_msg=key)
    for kind, beta in (("aligned", aligned), ("conflict", conflict)):
        check((np.sum(beta != 0, axis=1) == 3).all(), "Changed sparsity")
        opposite = np.count_nonzero(beta * (beta @ cov) < 0)
        check(
            opposite == cal["population_geometry"][kind]["opposite_active_cells"],
            "Population alignment diagnostic differs",
        )
        check(
            opposite == 0 if kind == "aligned" else opposite >= len(beta),
            "Incorrect population alignment/conflict",
        )
    # Audit inventory independently by cell/configuration, including all sensitivity values.
    check(
        len(jobs) == 1080 and len({json.dumps(j, sort_keys=True) for j in jobs}) == 1080,
        "Missing or duplicate planned fits",
    )
    frame = pd.DataFrame(jobs).fillna("equal")
    keys = [c for c in frame if c != "replication"]
    check(
        all(sorted(g.replication) == list(range(20)) for _, g in frame.groupby(keys)),
        "Unpaired seed inventory",
    )
    check(set(frame.cell) == set(p["pilot_cells"]), "Pilot cells differ")
    for cell in p["pilot_cells"]:
        actual = frame[(frame.cell == cell) & (frame.replication == 0)]
        expected = set()
        for family, span in (
            [("hcgl", "equal")]
            + ([("fcgl", "equal")] if cell in p["fcgl_cells"] else [])
            + ([("hcgl", 120)] if cell in p["ew_cells"] else [])
        ):
            for arm in p["core_arms"]:
                if arm == "ridge" and family == "fcgl":
                    continue
                expected.add((family, span, arm, 1.0 if arm == "soft_direction" else 0.0, 0.0))
        for list_key, arms_key in (
            ("conditional_cells", "conditional_arms"),
            ("gate_diagnostic_cells", "gate_diagnostic_arms"),
        ):
            if cell in p[list_key]:
                expected.update(("hcgl", "equal", arm, 0.0, 0.0) for arm in p[arms_key])
        if cell in p["soft_sensitivity_cells"]:
            expected.update(("hcgl", "equal", "soft_direction", eta, 0.0) for eta in (0.1, 10.0))
            expected.add(("hcgl", "equal", "soft_exclusion", 1.0, 1.0))
        observed = set(
            actual[["family", "span", "arm", "soft_strength", "zero_strength"]].itertuples(
                index=False, name=None
            )
        )
        check(expected == observed, "Unexpected or missing pilot comparator")
    result = dict(
        status="passed",
        identity=identity,
        calibration_months=60,
        assets=len(y.columns),
        factors=len(x.columns),
        known_truth_datasets=80,
        planned_final_fits=len(jobs),
        comparative_pilot_run=False,
        independent_calibration_values=sum(np.size(v) for v in reference.values()),
        unchanged_estimator_files=len(b["unchanged_estimator_sha256"]),
    )
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pilot-protocol", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(verify_pilot(args.out) if args.pilot_protocol else validate(args.out), indent=2)
    )
