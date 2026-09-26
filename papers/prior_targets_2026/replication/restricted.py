"""Prespecified E3 restricted-target sensitivity on fresh seeds and credit cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

from papers.prior_targets_2026.replication.run import (
    HERE, check_snapshot, digest, lambdas, read_json, save_json, validate_root,
)
from papers.prior_targets_2026.replication.multiasset import calibration, check_manifest, samples_for


def fit(x, y, xv, yv, factors, grid):
    """Build restricted targets from training rows and select validation loss only."""
    from papers.prior_targets_2026.replication.estimators import (
        fit_paths, prediction_losses, select_index,
    )
    targets = {}
    for name, names in [("small2", ["Rates", "Credit"]),
                        ("small3", ["Equity", "Rates", "Credit"])]:
        indices = [factors.index(f) for f in names]
        design = np.column_stack([np.ones(len(y)), x[:, indices]])
        b = np.linalg.lstsq(design, y, rcond=None)[0][1:]
        centred = x[:, indices] - x[:, indices].mean(0)
        reference = np.linalg.lstsq(centred, y - y.mean(), rcond=None)[0]
        np.testing.assert_allclose(b, reference, atol=1e-11)
        targets[name] = np.zeros(x.shape[1])
        targets[name][indices] = b
    paths = fit_paths(x, y, grid, external_targets=targets, include_relaxed=False)
    result = {}
    for name in ["M1_zero", "small2", "small3"]:
        path = paths[name]
        k = select_index(prediction_losses(xv, yv, path["beta"], path["intercept"]))
        result[name] = dict(beta=path["beta"][k], target=path["target"],
                            intercept=path["intercept"][k], reg_lambda=grid[k])
    return result


def numerical_checks(data, config):
    """Compare newly introduced targets against independent package objectives."""
    from factorlasso import solve_lasso_cvx_problem
    from papers.prior_targets_2026.replication.estimators import prepare
    maximum = 0.
    for profile in config["profiles"]:
        samples, _, _, _, _, _ = samples_for(data, profile, 112, 979999, config)
        (xt, yt), (xv, yv), _ = samples
        a = int(data["credit_indices"][0])
        selected = fit(xt, yt[:, a], xv, yv[:, a], list(data["factors"]), lambdas(config))
        z, v, _, _, xs, ys = prepare(xt, yt[:, a])
        for row in selected.values():
            ref = solve_lasso_cvx_problem(
                z, v[:, None], reg_lambda=float(row["reg_lambda"]), solver="CLARABEL",
                factors_beta_prior=(row["target"] * xs / ys)[None])
            error = float(np.max(abs(ref.estimated_beta[0] - row["beta"] * xs / ys)))
            maximum = max(maximum, error)
    assert maximum < 3e-4
    return dict(package_solver_comparisons=9, maximum_standardised_error=maximum,
                tolerance=.0003, restricted_target_reference_verified=True)


def paired(frame):
    """Aggregate within seed before comparing targets, preserving paired outcomes."""
    panels = frame.groupby(["profile", "seed", "method"])[
        ["credit_mse", "scenario_mse", "prediction_nmse"]].mean().reset_index()
    rows = []
    for profile, group in panels.groupby("profile"):
        for comparator in ["M1_zero", "small2"]:
            base = group[group.method == comparator].set_index("seed").sort_index()
            for method, part in group.groupby("method"):
                part = part.set_index("seed").sort_index()
                assert part.index.equals(base.index)
                row = dict(profile=profile, method=method, comparator=comparator, seeds=len(part))
                for metric in ["credit_mse", "scenario_mse", "prediction_nmse"]:
                    d = part[metric] - base[metric]
                    se = d.std(ddof=1) / np.sqrt(len(d))
                    row.update({metric: part[metric].mean(), metric+"_delta": d.mean(),
                                metric+"_low": d.mean()-1.96*se,
                                metric+"_high": d.mean()+1.96*se})
                rows.append(row)
    return panels, pd.DataFrame(rows)


def validate(frame, config):
    """Reject missing cells, inaccurate saved errors and non-fresh seeds."""
    assert len(frame) == 3 * 50 * 17 * 3
    assert set(frame.seed) == set(range(980000, 980050))
    assert set(frame.profile) == set(config["profiles"])
    assert set(frame.method) == {"M1_zero", "small2", "small3"}
    assert not frame.duplicated(["profile", "seed", "asset", "method"]).any()
    assert frame.groupby(["profile", "seed", "method"]).asset.nunique().eq(17).all()
    np.testing.assert_allclose(frame.credit_mse, (frame.beta2-frame.truth2)**2, atol=1e-12)
    assert np.isfinite(frame[["credit_mse", "scenario_mse", "prediction_nmse"]]).all().all()


def verify(out, config):
    """Independently recompute paired results and reject deliberate corruptions."""
    from papers.prior_targets_2026.replication.robustness import reject_assertion
    frame = pd.read_csv(out / "asset_results.csv")
    validate(frame, config)
    _, expected = paired(frame)
    table = pd.read_csv(out / "paired_summary.csv")
    numeric = expected.select_dtypes(include="number").columns
    np.testing.assert_allclose(table[numeric], expected[numeric], atol=1e-11)
    bad = frame.copy()
    bad.loc[0, "credit_mse"] += .01
    reject_assertion(lambda: validate(bad, config))
    reject_assertion(lambda: validate(frame.iloc[1:], config))
    assert read_json(out / "failures.json") == []
    return dict(rows=len(frame), panels=150, panel_method_rows=450, failures=0,
                corrupted_metric_rejected=True, missing_row_rejected=True)


def main():
    """Run the frozen sensitivity without editing previous study artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "verify"])
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = validate_root(args.output_root)
    check_snapshot(root)
    sys.path.insert(0, str(root / "source_snapshot/src"))
    check_manifest(root / "E3_L1_confirmation_v1")
    config = read_json(HERE / "restricted_protocol.json")
    out = root / "E3_restricted_v2"
    if args.command == "verify":
        check_manifest(out)
        print(json.dumps(verify(out, config), indent=2))
        return
    out.mkdir()
    (out / "archive").mkdir()
    for name in ["restricted.py", "restricted_protocol.json", "estimators.py",
                 "multiasset.py", "run.py", "robustness.py"]:
        shutil.copy2(HERE / name, out / "archive" / name)
    data = calibration(root)
    save_json(out / "numerical_checks.json", numerical_checks(data, config))
    records, failures = [], []
    started = time.perf_counter()
    for profile in config["profiles"]:
        for seed in range(980000, 980050):
            try:
                samples, _, truth, sigma, _, variance = samples_for(
                    data, profile, 112, seed, config)
                (xt, yt), (xv, yv), (xe, ye) = samples
                for a in data["credit_indices"]:
                    fitted = fit(xt, yt[:, a], xv, yv[:, a],
                                 list(data["factors"]), lambdas(config))
                    for method, row in fitted.items():
                        error = row["beta"]-truth[a]
                        record = dict(
                            profile=profile, seed=seed, asset=int(a), ticker=str(data["tickers"][a]),
                            method=method, credit_mse=error[2]**2,
                            scenario_mse=np.mean(error**2*np.diag(sigma))/variance[a],
                            prediction_nmse=np.mean((ye[:, a]-xe@row["beta"]
                                                     - row["intercept"])**2)/variance[a],
                            reg_lambda=row["reg_lambda"])
                        for f in range(9):
                            record.update({f"beta{f}": row["beta"][f],
                                           f"truth{f}": truth[a, f], f"target{f}": row["target"][f]})
                        records.append(record)
            except Exception as error:
                failures.append(dict(profile=profile, seed=seed, error=repr(error)))
        print(profile+" complete", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(out / "asset_results.csv", index=False)
    panels, table = paired(frame)
    panels.to_csv(out / "panel_results.csv", index=False)
    table.to_csv(out / "paired_summary.csv", index=False)
    save_json(out / "failures.json", failures)
    checks = verify(out, config)
    save_json(out / "summary.json", {**checks, "seconds": time.perf_counter()-started,
                                    "protocol_sha256": digest(HERE / "restricted_protocol.json")})
    save_json(out / "manifest.json",
              {str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()})
    print(json.dumps(read_json(out / "summary.json"), indent=2))


if __name__ == "__main__":
    main()


