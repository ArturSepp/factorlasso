"""Independent FF attribution verifier: no producer, estimator or solver imports.

Reconstructs premiums from the first training window, uses factor-by-factor
financial-unit errors and Gram traces, and checks exclusion risk by quadrature.
All saved fits and paired summaries are checked, not a favourable subset.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.linalg import eigvalsh

ROOT = Path(__file__).resolve().parents[3]
CELL = ["lookback", "coherence", "fx", "ex", "partition", "family", "fraction", "scenario"]
VARIANTS = ["unconstrained", "singleton_exclude", "pooled_exclude",
            "pooled_release", "pooled_direction"]


def read(path):
    """Read a JSON artifact without relying on the producer."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def checksum(path):
    """Hash an input using file_digest's independent streaming implementation."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(condition, message):
    """Reject an incomplete or corrupted experiment."""
    if not condition:
        raise ValueError(message)


def reference_losses(beta, alpha, truth, corr, premium, xs, ys):
    """Use per-factor contributions and a Gram trace in standardized coordinates."""
    # Convert each contribution directly, without constructing a raw coefficient matrix.
    scaled_error = (beta - truth) * ys[:, None]
    components = np.array([scaled_error[:, j] * (p / xs[j])
                           for j, p in enumerate(premium)])
    mu = np.add.reduce(components, axis=0)
    intercept = alpha * ys * 52
    component_mse = np.mean(components * components, axis=1)
    values = {"attribution_mse": float(sum(component_mse)),
              "mean_mse": float(np.dot(mu, mu) / len(mu)),
              "full_mean_mse": float(np.dot(mu + intercept, mu + intercept) / len(mu)),
              "alpha_mse": float(np.dot(intercept, intercept) / len(mu)),
              "mean_alpha_cross": float(2 * np.dot(mu, intercept) / len(mu)),
              "annual_prediction_risk": float(52 * np.trace(
                  corr @ (scaled_error.T @ scaled_error)) / len(mu)),
              "mean_bias": float(sum(mu) / len(mu)),
              "full_mean_bias": float(sum(mu + intercept) / len(mu))}
    for j, loss in enumerate(component_mse):
        values[f"factor_{j}_mse"] = float(loss)
        values[f"factor_{j}_bias"] = float(sum(components[j]) / len(mu))
    covariance = 52 * corr * np.outer(xs, xs)
    mean_constant = np.linalg.norm(np.linalg.solve(np.linalg.cholesky(covariance), premium))**2
    attr_constant = eigvalsh(np.diag(premium**2), covariance)[-1]
    check(values["mean_mse"] <= mean_constant * values["annual_prediction_risk"] + 1e-12,
          "Independent mean bound failed")
    check(values["attribution_mse"] <= attr_constant * values["annual_prediction_risk"] + 1e-12,
          "Independent attribution bound failed")
    return values


def reference_exclusion_risk(beta, fit_sd, gate_sd, correlation, threshold):
    """Integrate conditional squared error across the two gate regions."""
    low, high = -threshold - beta / gate_sd, threshold - beta / gate_sd

    def density(z):
        return np.exp(-z*z/2) / np.sqrt(2*np.pi)

    def kept(z):
        return fit_sd**2 * (1 - correlation**2 + correlation**2*z*z) * density(z)

    return (quad(kept, -np.inf, low, epsabs=1e-12)[0]
            + beta**2 * quad(density, low, high, epsabs=1e-12)[0]
            + quad(kept, high, np.inf, epsabs=1e-12)[0])


def compare_tables(actual, expected, keys):
    """Compare complete keyed inventories, every numeric field and exact metadata."""
    check(set(actual.columns) == set(expected.columns), "Table columns differ")
    check(not actual.duplicated(keys).any(), "Duplicate table row")
    check(not expected.duplicated(keys).any(), "Duplicate independent reference row")
    a, e = [frame.set_index(keys).sort_index() for frame in (actual, expected)]
    check(a.index.equals(e.index), "Missing or extra table rows")
    e = e[a.columns]
    for column in a:
        np.testing.assert_allclose(a[column], e[column], rtol=2e-10, atol=2e-13,
                                   equal_nan=True, err_msg=column)
    return a.size


def verify(output):
    """Check all inputs and independent numerical references, then issue a certificate."""
    started = time.perf_counter()
    output = Path(output).resolve()
    check(not any(p.casefold().startswith("onedrive") for p in output.parts),
          "Verification output must be outside OneDrive")
    freeze, p, result = [read(output / f"{name}.json") for name in
                         ("freeze", "protocol", "results_manifest")]
    b = freeze["binding"]
    identity = hashlib.sha256(json.dumps(b, sort_keys=True, allow_nan=False).encode()).hexdigest()
    check(identity == freeze["identity"] == result["identity"], "Identity mismatch")
    check(checksum(output / "protocol.json") == b["protocol_sha256"], "Changed protocol")
    archive, panel = Path(b["archive"]), Path(b["panel"])
    checked_files = []
    for prefix, mapping in ((ROOT, b["source_sha256"]),
                            (output / "source", b["source_sha256"]),
                            (ROOT, b["unchanged_estimator_sha256"]),
                            (archive, b["archive_files_sha256"]),
                            (panel, b["panel_files_sha256"]),
                            (output, result["files_sha256"])):
        for name, expected in mapping.items():
            path = prefix / name
            check(checksum(path) == expected, f"Changed artifact: {path}")
            checked_files.append((path, expected))
    legacy_manifest = read(archive / "simulation_manifest.json")
    legacy_freeze = read(archive / "freeze.json")
    legacy_certificate = read(archive / "verification.json")
    check(legacy_certificate["status"] == "passed", "Unverified archive")
    check(legacy_certificate["identity"] == legacy_freeze["identity"]
          == legacy_manifest["identity"], "Mixed archive")
    check(legacy_certificate["report_manifest_sha256"]["simulation_manifest.json"]
          == checksum(archive / "simulation_manifest.json"), "Changed historical manifest")
    x, y = [pd.read_csv(panel / f"panel/{name}_weekly.csv", index_col=0,
                       float_precision="round_trip") for name in ("X", "Y")]
    cal, sim = p["calibration"], p["source_simulation"]
    check(sim == read(archive / "protocol.json")["simulation"], "Simulation design changed")
    check(list(x.columns) == cal["factors"] and list(y.columns) == cal["responses"],
          "Factor/response basis changed")
    n = sim["calibration_rows"]
    xx, yy = x.iloc[:n].to_numpy(), y.iloc[:n].to_numpy()
    xs, ys = np.sqrt(np.var(xx, axis=0)), np.sqrt(np.var(yy, axis=0))
    cov = np.cov(xx, rowvar=False, bias=True)
    corr = cov / np.outer(xs, xs)
    for actual, expected in ((xs, cal["factor_weekly_sd"]), (ys, cal["response_weekly_sd"]),
                             (corr, cal["factor_corr"])):
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-14)
    with np.load(archive / "calibration.npz", allow_pickle=False) as original:
        np.testing.assert_allclose(original["calibration_x"], (xx-xx.mean(0))/xs,
                                   atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(original["calibration_y"], (yy-yy.mean(0))/ys,
                                   atol=1e-12, rtol=1e-12)
    hist = np.sum(xx, axis=0) * 52 / n
    target = float(hist @ np.linalg.solve(52*cov, hist))
    eigen, vectors = np.linalg.eigh(corr)
    market = np.zeros(len(xs))
    market[list(x.columns).index("MKT_LOGREL")] = 1
    raw_directions = [market*xs, xs, vectors[:, 0]*xs, vectors[:, -1]*xs]
    for j in (2, 3):
        raw = raw_directions[j]
        raw_directions[j] = raw * np.sign((raw / xs)[np.argmax(abs(raw / xs))])
    premiums = [hist] + [raw * np.sqrt(target/(raw @ np.linalg.solve(52*cov, raw)))
                         for raw in raw_directions]
    names = ["training_mean", "market_only", "equal_positive_risk",
             "weak_covariance_direction", "strong_covariance_direction"]
    check([s["name"] for s in cal["scenarios"]] == names, "Scenario inventory changed")
    for scenario, premium in zip(cal["scenarios"], premiums):
        np.testing.assert_allclose(scenario["annual_premium"], premium, atol=1e-13)
        for fx in (0, 1):
            covariance = 52 * (np.eye(len(xs)) if fx == 0 else corr) * np.outer(xs, xs)
            norm = premium @ np.linalg.solve(covariance, premium)
            np.testing.assert_allclose(scenario["norm_squared_by_fx"][str(fx)], norm)
    jobs = list(itertools.product(sim["lookbacks"], range(sim["replications"]),
                                 sim["coherence"], sim["factor_correlation_mix"],
                                 sim["error_dependence_mix"]))
    expected_jobs = {f"T{t}_r{r:03d}_{c}_fx{fx}_ex{ex}" for t, r, c, fx, ex in jobs}
    check(expected_jobs == set(legacy_manifest["receipt_sha256"])
          == {j["key"] for j in b["jobs"]}, "Dataset inventory differs")
    reference_rows = []
    fits = 0
    for job_number, (t, r, c, fx, ex) in enumerate(jobs, 1):
        job_key = f"T{t}_r{r:03d}_{c}_fx{fx}_ex{ex}"
        folder = archive / "simulation" / job_key
        receipt = read(folder / "receipt.json")
        check(checksum(folder / "receipt.json") == legacy_manifest["receipt_sha256"][job_key],
              "Changed saved dataset receipt")
        for name, expected in receipt["files_sha256"].items():
            path = folder / name
            check(checksum(path) == expected, "Changed saved fit")
            checked_files.append((path, expected))
        with np.load(folder / "arrays.npz", allow_pickle=False) as a:
            truth, sigma = a["truth"], a["factor_cov"]
            xt, yt = a["x_test"], a["y_test"]
            np.testing.assert_allclose(sigma, np.eye(len(xs)) if fx == 0 else corr, atol=1e-12)
            for part, family, fraction, variant in itertools.product(
                    sim["partitions"], sim["families"], sim["penalty_fractions"], VARIANTS):
                fit_name = f"{part}__{family}__{fraction}__{variant}"
                beta, alpha = a[fit_name+"__beta"], a[fit_name+"__intercept"]
                d = beta-truth
                legacy_loading = np.mean(d*d)
                legacy_risk = (np.trace(sigma @ (d.T @ d)) + sum(alpha*alpha))/len(alpha)
                legacy_test = np.mean((yt - xt @ beta.T - alpha)**2)
                for name, premium in zip(names, premiums):
                    values = reference_losses(beta, alpha, truth, sigma, premium, xs, ys)
                    reference_rows.append(dict(lookback=t, replication=r, coherence=c, fx=fx,
                                               ex=ex, partition=part, family=family,
                                               fraction=fraction, variant=variant, scenario=name,
                                               legacy_loading_mse=legacy_loading,
                                               legacy_excess_risk=legacy_risk,
                                               legacy_test_mse=legacy_test, **values))
                fits += 1
        if job_number % 80 == 0:
            print(f"Independently checked {job_number}/{len(jobs)} datasets", flush=True)
    reference = pd.DataFrame(reference_rows)
    actual = pd.read_csv(output / "replication_losses.csv", float_precision="round_trip")
    values_checked = compare_tables(actual, reference, CELL + ["replication", "variant"])
    pairs = [(v, "unconstrained") for v in VARIANTS[1:]] + [
        ("pooled_exclude", "singleton_exclude"), ("pooled_exclude", "pooled_release")]
    check(p["pairs"] == [list(pair) for pair in pairs], "Contrast inventory changed")
    metrics = ["attribution_mse", "mean_mse", "full_mean_mse", "alpha_mse",
               "mean_alpha_cross", "annual_prediction_risk", "legacy_loading_mse",
               "legacy_excess_risk", "legacy_test_mse"]
    check(metrics == p["metrics"], "Metric inventory changed")
    comparison_rows = []
    for variant, baseline in pairs:
        a = reference[reference.variant == variant].set_index(CELL + ["replication"])[metrics]
        z = reference[reference.variant == baseline].set_index(CELL + ["replication"])[metrics]
        delta = a-z
        means, refs, diffs = [frame.groupby(level=CELL).mean() for frame in (a, z, delta)]
        counts = delta.groupby(level=CELL).size()
        check((counts == sim["replications"]).all(), "Unpaired replication inventory")
        se = np.sqrt(delta.groupby(level=CELL).var(ddof=1).div(counts, axis=0))
        for key in means.index:
            for metric in metrics:
                difference, mcse = diffs.loc[key, metric], se.loc[key, metric]
                comparison_rows.append(dict(zip(CELL, key)) | dict(
                    variant=variant, reference=baseline, metric=metric,
                    replications=int(counts[key]), mean=means.loc[key, metric],
                    reference_mean=refs.loc[key, metric], difference=difference, mc_se=mcse,
                    lower=difference-1.96*mcse, upper=difference+1.96*mcse))
    contrasts = pd.DataFrame(comparison_rows)
    actual_pairs = pd.read_csv(output / "paired_contrasts.csv", float_precision="round_trip")
    values_checked += compare_tables(actual_pairs, contrasts, CELL+[
        "variant", "reference", "metric"])
    overview_rows = []
    overview_keys = ["scenario", "partition", "variant", "reference", "metric"]
    for key, group in contrasts.groupby(overview_keys):
        mean, base = group["mean"].mean(), group.reference_mean.mean()
        overview_rows.append(dict(zip(overview_keys, key)) | dict(
            cells=len(group), mean=mean, reference_mean=base,
            relative_change_percent=100*(mean-base)/base if base else np.nan,
            improved_cells=sum(group.difference < 0), worse_cells=sum(group.difference > 0),
            interval_below_zero=sum(group.upper < 0), interval_above_zero=sum(group.lower > 0)))
    values_checked += compare_tables(pd.read_csv(output / "overview.csv",
                                                float_precision="round_trip"),
                                     pd.DataFrame(overview_rows), overview_keys)
    toy = pd.read_csv(output / "exclusion_reference.csv", float_precision="round_trip")
    check(len(toy) == 63 and not toy.duplicated(
        ["beta_over_fit_sd", "m", "error_correlation"]).any(), "Incomplete toy grid")
    for row in toy.itertuples():
        sd = np.sqrt((1+(row.m-1)*row.error_correlation)/row.m)
        risk = reference_exclusion_risk(row.beta_over_fit_sd, 1, sd, sd, 2)
        np.testing.assert_allclose([row.gate_sd, row.fit_gate_correlation, row.risk,
                                    row.difference_from_ungated], [sd, sd, risk, risk-1],
                                   rtol=1e-10, atol=1e-12)
        values_checked += 4
    check(result["datasets"] == len(jobs) and result["saved_fits"] == fits
          and result["rows"] == len(reference) and result["paired_rows"] == len(contrasts)
          and result["scenarios"] == len(names) and result["refits"] == 0,
          "Manifest counts differ")
    for path, expected in checked_files:
        check(checksum(path) == expected, "Inputs changed during verification")
    certificate = dict(status="passed", identity=identity, datasets=len(jobs), saved_fits=fits,
                       scenarios=len(names), rows=len(reference), paired_rows=len(contrasts),
                       scalar_values_checked=values_checked, exclusion_reference_cases=len(toy),
                       refits=0, unchanged_estimator_files=len(b["unchanged_estimator_sha256"]),
                       verifier_sha256=checksum(Path(__file__)),
                       results_manifest_sha256=checksum(output / "results_manifest.json"),
                       wall_seconds=time.perf_counter()-started)
    (output / "verification.json").write_text(json.dumps(certificate, indent=2, allow_nan=False)
                                              + "\n", encoding="utf-8")
    return certificate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().out), indent=2))
