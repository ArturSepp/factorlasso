"""Freeze premium scenarios, then rescore archived FF known-truth fits without refitting.

This is a retrospective attribution experiment conditional on supplied premia.
It does not estimate true attribution from real FF portfolio histories. Annual
quantities use 52 times the weekly log-relative-return mean/covariance; root
mean losses are annual log-return units, not realised CAGR or portfolio returns.
Only numpy/pandas/scipy and the standard library are used: no estimator import.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import ndtr

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
KEYS = ["lookback", "coherence", "fx", "ex", "partition", "family", "fraction"]
VARIANTS = ["unconstrained", "singleton_exclude", "pooled_exclude",
            "pooled_release", "pooled_direction"]
PAIRS = [(v, "unconstrained") for v in VARIANTS[1:]] + [
    ("pooled_exclude", "singleton_exclude"), ("pooled_exclude", "pooled_release")]
METRICS = ["attribution_mse", "mean_mse", "full_mean_mse", "alpha_mse",
           "mean_alpha_cross", "annual_prediction_risk", "legacy_loading_mse",
           "legacy_excess_risk", "legacy_test_mse"]


def require(condition, message):
    """Fail explicitly on incomplete, changed or nonfinite inputs."""
    if not condition:
        raise ValueError(message)


def sha(path):
    """Hash bytes, including large immutable coefficient archives."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    """Read strict research JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    """Write strict JSON; the caller enforces a fresh output or matching freeze."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(value):
    """Canonical identity for this new protocol, independent of JSON whitespace."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def external(path):
    """Never generate outputs in a checkout or OneDrive."""
    path = Path(path)
    require(path.is_absolute(), "Use an absolute external output directory")
    path = path.resolve()
    require(REPO != path and REPO not in path.parents and not any(
        part.casefold().startswith("onedrive") for part in path.parts), "Output inside OneDrive")
    return path


def inventory(p):
    """Preserve every previously frozen simulation condition and paired seed."""
    return [dict(lookback=t, replication=r, coherence=c, fx=fx, ex=ex,
                 key=f"T{t}_r{r:03d}_{c}_fx{fx}_ex{ex}")
            for t, r, c, fx, ex in itertools.product(
                p["lookbacks"], range(p["replications"]), p["coherence"],
                p["factor_correlation_mix"], p["error_dependence_mix"])]


def premium_scenarios(x, y, rows):
    """Training-only directions, fixed before examining any new attribution score."""
    require(x.index.equals(y.index), "Calibration dates differ")
    xx, yy = x.iloc[:rows].to_numpy(), y.iloc[:rows].to_numpy()
    require(len(xx) == rows and np.isfinite(xx).all() and np.isfinite(yy).all(),
            "Incomplete calibration")
    xs, ys = xx.std(0), yy.std(0)
    require(np.all(xs > 0) and np.all(ys > 0), "Zero calibration scale")
    z = (xx - xx.mean(0)) / xs
    corr = z.T @ z / rows
    eigenvalues, vectors = np.linalg.eigh(corr)
    require(eigenvalues[0] > 1e-10, "Premium geometry requires positive definite covariance")
    historical = 52 * xx.mean(0)
    standardized = historical / xs
    norm2 = float(standardized @ np.linalg.solve(corr, standardized) / 52)
    require(norm2 > 0, "Zero historical premium norm")

    def normalized(direction):
        direction = np.asarray(direction, dtype=float)
        norm = float(direction @ np.linalg.solve(corr, direction) / 52)
        return xs * direction * np.sqrt(norm2 / norm)

    def oriented(vector):
        # Canonical orientation uses factor coordinates, never fitted-method errors.
        return vector if vector[np.argmax(abs(vector))] >= 0 else -vector

    market = np.zeros(len(xs))
    market[list(x.columns).index("MKT_LOGREL")] = 1
    values = {
        "training_mean": historical,
        "market_only": normalized(market),
        "equal_positive_risk": normalized(np.ones(len(xs))),
        "weak_covariance_direction": normalized(oriented(vectors[:, 0])),
        "strong_covariance_direction": normalized(oriented(vectors[:, -1])),
    }
    scenarios = []
    for name, premium in values.items():
        norms = {str(fx): float((premium / xs) @ np.linalg.solve(
            np.eye(len(xs)) if fx == 0 else corr, premium / xs) / 52) for fx in (0, 1)}
        scenarios.append({"name": name, "annual_premium": premium.tolist(),
                          "primary": name == "training_mean", "norm_squared_by_fx": norms})
    return {"rows": rows, "first_date": str(x.index[0]), "last_date": str(x.index[rows-1]),
            "factors": list(x.columns), "responses": list(y.columns),
            "factor_weekly_sd": xs.tolist(), "response_weekly_sd": ys.tolist(),
            "factor_corr": corr.tolist(), "historical_norm_squared": norm2,
            "scenarios": scenarios}


def authenticated_archive(archive):
    """Authenticate the prior independent certificate, metadata and all dataset receipts."""
    frozen = read(archive / "freeze.json")
    certificate = read(archive / "verification.json")
    manifest = read(archive / "simulation_manifest.json")
    old_protocol = read(archive / "protocol.json")
    require(certificate["status"] == "passed" and certificate["identity"]
            == manifest["identity"] == frozen["identity"], "Unverified historical simulation")
    require(certificate["report_manifest_sha256"]["simulation_manifest.json"]
            == sha(archive / "simulation_manifest.json"), "Historical manifest changed")
    require(frozen["binding"]["protocol_sha256"] == sha(archive / "protocol.json"),
            "Historical protocol changed")
    require(manifest["calibration_sha256"] == sha(archive / "calibration.npz"),
            "Historical calibration changed")
    jobs = inventory(old_protocol["simulation"])
    require(set(manifest["receipt_sha256"]) == {j["key"] for j in jobs}
            and manifest["jobs"] == len(jobs), "Incomplete historical inventory")
    for job in jobs:
        folder = archive / "simulation" / job["key"]
        require(sha(folder / "receipt.json") == manifest["receipt_sha256"][job["key"]],
                "Historical dataset seal changed")
        receipt = read(folder / "receipt.json")
        require(receipt["identity"] == frozen["identity"] and receipt["status"] == "complete",
                "Mixed or incomplete dataset")
        require(all(receipt["design"][k] == v for k, v in job.items()), "Dataset design mismatch")
    return frozen, old_protocol, manifest, jobs


def freeze(archive, output):
    """Persist choices and exact premia before opening saved coefficient arrays for scoring."""
    archive, output = Path(archive).resolve(), external(output)
    require(not output.exists() or not any(output.iterdir()), "Freeze requires a fresh directory")
    old, p, manifest, jobs = authenticated_archive(archive)
    panel = Path(old["binding"]["panel"])
    require(sha(panel / "manifest.json") == old["binding"]["panel_manifest_sha256"],
            "Calibration panel manifest changed")
    panel_manifest = read(panel / "manifest.json")
    for name in ("panel/X_weekly.csv", "panel/Y_weekly.csv"):
        require(sha(panel / name) == panel_manifest["files_sha256"][name],
                "Calibration panel bytes changed")
    x, y = [pd.read_csv(panel / f"panel/{name}_weekly.csv", index_col=0,
                       float_precision="round_trip") for name in ("X", "Y")]
    calibration = premium_scenarios(x, y, p["simulation"]["calibration_rows"])
    protocol = {
        "study": "ff_attribution_stage1a_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "retrospective_exploratory": True, "new_scores_computed_before_freeze": False,
        "scope": "All saved known-truth fits; no refitting or tuning; no real-data truth claim",
        "calibration": calibration, "source_simulation": p["simulation"],
        "annualization": 52, "primary_loss": "attribution_mse",
        "key_secondary_loss": "mean_mse", "metrics": METRICS,
        "pairs": PAIRS, "legacy_losses_preserved": True,
        "units": "Annual log-relative-return means; annual covariance=52*weekly covariance",
        "de_standardization": "B_raw=diag(calibration_y_sd)*B_saved*diag(1/calibration_x_sd)",
        "alpha": "Primary fixes true/fitted alpha to zero; full_mean_mse adds saved fitted alpha",
        "premium_interpretation": "Hypothetical future premia on centered training innovations",
        "geometry_controls": "Same calibration covariance norm; vectors fixed across fx conditions",
        "inference": "40 paired seeds/cell; mean difference +/-1.96 MC SE; no multiplicity claim",
        "aggregation": "Equal-cell losses; condition/family cells are not independent trials",
        "toy_risk": {"beta_over_fit_sd": [0, .25, .5, 1, 2, 3, 4], "m": [1, 5, 25],
                     "error_correlation": [0, .5, .9], "gate_threshold": 2,
                     "scope": "Equicorrelated Gaussian scalar exclusion only, same-sample gate"},
    }
    source = {path.relative_to(REPO).as_posix(): sha(path) for path in (
        Path(__file__), HERE / "verify_french_attribution.py",
        HERE / "tests/test_french_attribution.py")}
    archive_files = {name: sha(archive / name) for name in (
        "freeze.json", "verification.json", "protocol.json", "simulation_manifest.json",
        "calibration.npz", "calibration.json")}
    # Full estimator source fingerprints document that scoring leaves defaults/code untouched.
    core = {p.relative_to(REPO).as_posix(): sha(p) for p in
            sorted((REPO / "src/factorlasso").rglob("*.py"))}
    output.mkdir(parents=True, exist_ok=True)
    write(output / "protocol.json", protocol)
    binding = {"protocol_sha256": sha(output / "protocol.json"), "archive": str(archive),
               "archive_files_sha256": archive_files, "source_sha256": source,
               "unchanged_estimator_sha256": core, "panel": str(panel),
               "panel_files_sha256": {name: sha(panel / name) for name in (
                   "manifest.json", "panel/X_weekly.csv", "panel/Y_weekly.csv")},
               "python": sys.executable, "python_version": platform.python_version(),
               "numpy": np.__version__, "pandas": pd.__version__, "jobs": jobs,
               "expected_saved_fits": manifest["rows"]}
    write(output / "freeze.json", {"identity": digest(binding), "binding": binding})
    for name in source:
        dest = output / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((REPO / name).read_bytes())
    return {"identity": digest(binding), "jobs": len(jobs), "saved_fits": manifest["rows"],
            "premia": calibration["scenarios"], "output": str(output)}


def validate_freeze(output):
    """Check content bindings, including core preservation, before and after scoring."""
    f = read(output / "freeze.json")
    b = f["binding"]
    require(digest(b) == f["identity"], "Freeze identity mismatch")
    require(sha(output / "protocol.json") == b["protocol_sha256"], "Premium protocol changed")
    for prefix, files in ((REPO, b["source_sha256"]), (REPO, b["unchanged_estimator_sha256"]),
                          (Path(b["archive"]), b["archive_files_sha256"]),
                          (Path(b["panel"]), b["panel_files_sha256"])):
        for name, expected in files.items():
            require(sha(prefix / name) == expected, f"Bound input/source changed: {name}")
    for name, expected in b["source_sha256"].items():
        require(sha(output / "source" / name) == expected, "Archived scoring source changed")
    return f, read(output / "protocol.json")


def loss_values(beta, alpha, truth, corr, premium, xs, ys):
    """Primary fixed-alpha attribution and mean errors in explicit original return units."""
    delta = (beta - truth) * ys[:, None] / xs[None, :]
    contribution = delta * premium[None, :]
    error = contribution.sum(1)
    alpha_error = 52 * ys * alpha
    covariance = 52 * corr * xs[:, None] * xs[None, :]
    prediction = float(np.einsum("ij,jk,ik->", delta, covariance, delta) / len(delta))
    mean_bound = float(premium @ np.linalg.solve(covariance, premium))
    eigen, vectors = np.linalg.eigh(covariance)
    invroot = (vectors / np.sqrt(eigen)) @ vectors.T
    attr_bound = float(np.linalg.eigvalsh(invroot @ np.diag(premium**2) @ invroot)[-1])
    attr = float(np.square(contribution).sum() / len(delta))
    mean = float(np.square(error).mean())
    require(mean <= mean_bound * prediction + 1e-12, "Mean-risk bound fails")
    require(attr <= attr_bound * prediction + 1e-12, "Attribution-risk bound fails")
    values = {"attribution_mse": attr, "mean_mse": mean,
              "full_mean_mse": float(np.square(error + alpha_error).mean()),
              "alpha_mse": float(np.square(alpha_error).mean()),
              "mean_alpha_cross": float(2 * np.mean(error * alpha_error)),
              "annual_prediction_risk": prediction, "mean_bias": float(error.mean()),
              "full_mean_bias": float((error + alpha_error).mean())}
    for k in range(len(premium)):
        values[f"factor_{k}_mse"] = float(np.square(contribution[:, k]).mean())
        values[f"factor_{k}_bias"] = float(contribution[:, k].mean())
    return values


def exclusion_risk(beta, fit_sd, gate_sd, correlation, threshold=2):
    """Exact scalar Gaussian risk for zero if the gate fails, unchanged fit otherwise.

    Conditional on standardized gate Z, the fit error has mean rho*fit_sd*Z
    and variance fit_sd**2*(1-rho**2). Integrating outside the gate interval
    gives this expression. Passing-gate sign clipping is deliberately absent.
    """
    lower, upper = -threshold - beta / gate_sd, threshold - beta / gate_sd
    probability = ndtr(upper) - ndtr(lower)
    def phi(z):
        return np.exp(-z*z/2) / np.sqrt(2*np.pi)
    return float(beta*beta*probability + fit_sd*fit_sd * (
        1 - probability + correlation**2*(upper*phi(upper)-lower*phi(lower))))


def summaries(df, protocol):
    """All paired contrasts, preserving dependence within every simulation seed."""
    keys = KEYS + ["scenario"]
    summary = []
    for key, group in df.groupby(keys, sort=False):
        for variant, reference in protocol["pairs"]:
            left = group[group.variant == variant].set_index("replication").sort_index()
            right = group[group.variant == reference].set_index("replication").sort_index()
            n = protocol["source_simulation"]["replications"]
            require(list(left.index) == list(right.index) == list(range(n)),
                    "Incomplete paired replications")
            for metric in protocol["metrics"]:
                a, b = left[metric].to_numpy(), right[metric].to_numpy()
                difference = a-b
                se = float(difference.std(ddof=1) / np.sqrt(n))
                summary.append({**dict(zip(keys, key)), "variant": variant,
                                "reference": reference, "metric": metric, "replications": n,
                                "mean": float(a.mean()), "reference_mean": float(b.mean()),
                                "difference": float(difference.mean()), "mc_se": se,
                                "lower": float(difference.mean()-1.96*se),
                                "upper": float(difference.mean()+1.96*se)})
    return pd.DataFrame(summary)


def score(output):
    """Read every immutable fit once; write new scores without modifying the original run."""
    output = external(output)
    require(not (output / "results_manifest.json").exists(), "Results already exist; use verifier")
    f, p = validate_freeze(output)
    archive = Path(f["binding"]["archive"])
    _, _, manifest, jobs = authenticated_archive(archive)
    require(jobs == f["binding"]["jobs"], "Job inventory changed")
    cal, sim = p["calibration"], p["source_simulation"]
    xs, ys = np.array(cal["factor_weekly_sd"]), np.array(cal["response_weekly_sd"])
    fit_names = [f"{part}__{family}__{fraction}__{variant}"
                 for part, family, fraction, variant in itertools.product(
                     sim["partitions"], sim["families"], sim["penalty_fractions"], VARIANTS)]
    rows, saved_hashes = [], {}
    started = time.perf_counter()
    for number, job in enumerate(jobs, 1):
        folder = archive / "simulation" / job["key"]
        receipt = read(folder / "receipt.json")
        for name, expected in receipt["files_sha256"].items():
            require(sha(folder / name) == expected, "Saved fit/metric bytes changed")
        saved_hashes[job["key"]] = receipt["files_sha256"]
        metrics = read(folder / "metrics.json")
        require(len(metrics) == len(fit_names), "Missing legacy metrics")
        with np.load(folder / "arrays.npz", allow_pickle=False) as arrays:
            actual = {k[:-6] for k in arrays.files if k.endswith("__beta")}
            require(actual == set(fit_names), "Missing/extra saved fits")
            truth, corr = arrays["truth"], arrays["factor_cov"]
            require(truth.shape == (len(ys), len(xs)), "Calibration/fit basis differs")
            for name in fit_names:
                part, family, fraction, variant = name.split("__")
                beta, alpha = arrays[name+"__beta"], arrays[name+"__intercept"]
                require(np.isfinite(beta).all() and np.isfinite(alpha).all(), "Nonfinite fit")
                prior = [r for r in metrics if r["partition"] == part and r["family"] == family
                         and r["fraction"] == float(fraction) and r["variant"] == variant]
                require(len(prior) == 1, "Duplicate/missing legacy metric")
                prior = prior[0]
                delta = beta-truth
                legacy_loading = float(np.square(delta).mean())
                legacy_risk = float((np.einsum("ij,jk,ik->", delta, corr, delta)
                                     + alpha @ alpha) / len(beta))
                np.testing.assert_allclose([legacy_loading, legacy_risk],
                                           [prior["loading_mse"], prior["excess_risk"]],
                                           atol=1e-12, rtol=1e-10)
                metadata = {k: job[k] for k in ("lookback", "replication", "coherence", "fx", "ex")}
                for scenario in cal["scenarios"]:
                    values = loss_values(beta, alpha, truth, corr,
                                         np.array(scenario["annual_premium"]), xs, ys)
                    rows.append({**metadata, "partition": part, "family": family,
                                 "fraction": float(fraction), "variant": variant,
                                 "scenario": scenario["name"], **values,
                                 "legacy_loading_mse": legacy_loading,
                                 "legacy_excess_risk": legacy_risk,
                                 "legacy_test_mse": prior["test_mse"]})
        if number % 80 == 0:
            print(f"Rescored {number}/{len(jobs)} saved datasets", flush=True)
    df = pd.DataFrame(rows)
    require(len(df) == manifest["rows"]*len(cal["scenarios"]), "Incomplete rescore")
    require(not df.duplicated(KEYS + ["replication", "variant", "scenario"]).any(),
            "Duplicate result")
    require(np.isfinite(df.select_dtypes(include=np.number)).all().all(), "Nonfinite output")
    df.to_csv(output / "replication_losses.csv", index=False, float_format="%.17g")
    contrasts = summaries(df, p)
    contrasts.to_csv(output / "paired_contrasts.csv", index=False, float_format="%.17g")
    overview = []
    overview_keys = ["scenario", "partition", "variant", "reference", "metric"]
    for key, group in contrasts.groupby(overview_keys):
        a, b = float(group["mean"].mean()), float(group.reference_mean.mean())
        overview.append(dict(zip(["scenario", "partition", "variant", "reference", "metric"], key))
                        | {"cells": len(group), "mean": a, "reference_mean": b,
                           "relative_change_percent": 100*(a/b-1) if b != 0 else None,
                           "improved_cells": int((group.difference < 0).sum()),
                           "worse_cells": int((group.difference > 0).sum()),
                           "interval_below_zero": int((group.upper < 0).sum()),
                           "interval_above_zero": int((group.lower > 0).sum())})
    pd.DataFrame(overview).to_csv(output / "overview.csv", index=False, float_format="%.17g")
    toy = p["toy_risk"]
    risks = []
    for beta, m, rho in itertools.product(toy["beta_over_fit_sd"], toy["m"],
                                         toy["error_correlation"]):
        sd = np.sqrt((1+(m-1)*rho)/m)
        risk = exclusion_risk(beta, 1, sd, sd, toy["gate_threshold"])
        risks.append(dict(beta_over_fit_sd=beta, m=m, error_correlation=rho,
                          gate_sd=sd, fit_gate_correlation=sd, risk=risk,
                          difference_from_ungated=risk-1))
    pd.DataFrame(risks).to_csv(output / "exclusion_reference.csv", index=False,
                             float_format="%.17g")
    validate_freeze(output)
    for job in jobs:
        for name, expected in saved_hashes[job["key"]].items():
            require(sha(archive / "simulation" / job["key"] / name) == expected,
                    "Historical fit changed during rescore")
    outputs = ["replication_losses.csv", "paired_contrasts.csv", "overview.csv",
               "exclusion_reference.csv"]
    result = {"identity": f["identity"], "status": "produced; independent verification required",
              "datasets": len(jobs), "saved_fits": manifest["rows"],
              "scenarios": len(cal["scenarios"]),
              "rows": len(df), "paired_rows": len(contrasts), "refits": 0,
              "wall_seconds": time.perf_counter()-started,
              "files_sha256": {name: sha(output / name) for name in outputs}}
    write(output / "results_manifest.json", result)
    return result


def main():
    """Require distinct freeze and scoring invocations."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--score", action="store_true")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.freeze and args.archive is None:
        parser.error("--freeze requires --archive")
    print(json.dumps(freeze(args.archive, args.out) if args.freeze else score(args.out), indent=2))


if __name__ == "__main__":
    main()
