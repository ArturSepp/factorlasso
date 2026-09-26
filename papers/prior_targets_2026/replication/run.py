"""Reproducible evidence runner; generated outputs must remain C-local.

Commands: run/verify with --stage E0/E1/E2 and --output-root.
E2 currently implements the frozen base pilot/confirmation grid. Additional
stress designs require a recorded protocol amendment before they are run.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tomllib

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PAPER_ROOT = HERE.parent
DATA = REPO / "papers/jss_2026/applications/data"
INPUTS = ["etf_excess_logreturns.csv", "futures_risk_factors.csv", "riskfree_monthly.csv"]


def digest(path):
    """Return a content hash for one file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    """Write a human-readable receipt without nonstandard numeric values."""
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n",
                          encoding="utf-8")


def read_json(path):
    """Read UTF-8 configuration, accepting a Windows BOM."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def validate_root(root):
    """Reject output outside governed storage, including all OneDrive paths."""
    allowed = Path(os.environ["AGENT_LOCAL_ROOT"]).resolve()
    root = Path(root).resolve()
    if "onedrive" in str(root).lower() or not root.is_relative_to(allowed):
        raise ValueError("Output must be inside AGENT_LOCAL_ROOT and outside OneDrive")
    return root


def validate_protocol(config):
    """Fail closed on empty designs, leaked seeds or unspecified scenario units."""
    for name in ["pilot", "confirmation"]:
        design = config[name]
        for axis in ["rho", "secondary", "n", "snr"]:
            if not design[axis]:
                raise ValueError("Empty design")
        if design["seeds"][1] < design["seeds"][0]:
            raise ValueError("Invalid seed range")
    a, b = config["pilot"]["seeds"], config["confirmation"]["seeds"]
    if max(a[0], b[0]) <= min(a[1], b[1]):
        raise ValueError("Seed ranges overlap")
    if not config["scenarios"]["units"]:
        raise ValueError("Missing scenario units")
    if config["lambda"]["minimum"] <= 0 or config["lambda"]["maximum"] <= 0:
        raise ValueError("Invalid lambda grid")
    if not config["data_status"] or len(config["empirical_assets"]) < 3:
        raise ValueError("Missing data provenance")


def protocol_negative_controls(config):
    """Prove protocol verification detects intentional omissions and leakage."""
    rejected = []
    for defect in ["seed_overlap", "empty_grid", "missing_units", "missing_provenance"]:
        broken = deepcopy(config)
        if defect == "seed_overlap":
            broken["confirmation"]["seeds"] = broken["pilot"]["seeds"]
        elif defect == "empty_grid":
            broken["pilot"]["rho"] = []
        elif defect == "missing_units":
            broken["scenarios"]["units"] = ""
        else:
            broken["data_status"] = {}
        try:
            validate_protocol(broken)
        except ValueError:
            rejected.append(defect)
        else:
            raise AssertionError(f"Failed to detect {defect}")
    try:
        validate_root(REPO / "generated")
    except ValueError:
        rejected.append("onedrive_output")
    else:
        raise AssertionError("OneDrive output accepted")
    return rejected


def lambdas(config):
    """Return the shared descending grid, including exact unregularised fitting."""
    spec = config["lambda"]
    return np.r_[np.geomspace(spec["maximum"], spec["minimum"], spec["positive_points"]), 0.]


def load_panels(root, config):
    """Apply the existing paper's explicit monthly log-return input convention."""
    source = root / "source_snapshot/data"
    nav = pd.read_csv(source / INPUTS[1], index_col=0, parse_dates=True)
    nav = nav.rename(columns={"Private Equity": "PrivateEquity", "Rates Vol": "RatesVol"})
    x = np.log(nav[config["factors"]].resample("ME").last()).diff()
    y = pd.read_csv(source / INPUTS[0], index_col=0, parse_dates=True)
    ix = x.index.intersection(y.index)
    return x.loc[ix], y.loc[ix]


def snapshot(root):
    """Freeze source and data so concurrent checkout work cannot change the fits."""
    target = root / "source_snapshot"
    target.mkdir(parents=True)
    shutil.copytree(REPO / "src/factorlasso", target / "src/factorlasso",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (target / "data").mkdir()
    for name in INPUTS:
        shutil.copy2(DATA / name, target / "data" / name)
    shutil.copy2(DATA.parent / "etf_universe.csv", target / "data/etf_universe.csv")
    for name in ["pyproject.toml", "uv.lock"]:
        shutil.copy2(REPO / name, target / name)
    shutil.copy2(HERE / "protocol.json", target / "protocol.json")
    manifest = {str(p.relative_to(target)): digest(p) for p in target.rglob("*")
                if p.is_file()}
    save_json(root / "snapshot_hashes.json", manifest)


def check_snapshot(root):
    """Reject a changed source or input in the frozen experiment."""
    for name, expected in read_json(root / "snapshot_hashes.json").items():
        if digest(root / "source_snapshot" / name) != expected:
            raise AssertionError(f"Snapshot drift: {name}")


def finish(stage_dir, report):
    """Save a stage receipt and content hashes for all generated artifacts."""
    save_json(stage_dir / "summary.json", report)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {str(p.relative_to(stage_dir)): digest(p)
                      for p in stage_dir.rglob("*") if p.is_file()},
        "research_source": {p.name: digest(p) for p in HERE.glob("*.py")},
    }
    save_json(stage_dir / "manifest.json", manifest)


def run_e0(root, destination, config):
    """Audit the frozen input panel, runtime and protocol before comparisons."""
    import factorlasso
    import cvxpy
    from papers.prior_targets_2026.replication.estimators import reference_checks

    x, y = load_panels(root, config)
    complete = pd.concat([x, y[config["empirical_assets"]]], axis=1).dropna()
    if len(complete) <= config["empirical_training_months"] + 24:
        raise AssertionError("Insufficient empirical rows")
    audit = []
    for name in INPUTS:
        frame = pd.read_csv(root / "source_snapshot/data" / name, index_col=0)
        audit.append({
            "file": name, "sha256": digest(root / "source_snapshot/data" / name),
            "rows": len(frame), "columns": len(frame.columns),
            "start": str(frame.index[0]), "end": str(frame.index[-1]),
            "missing_cells": int(frame.isna().sum().sum()),
        })
    pd.DataFrame(audit).to_csv(destination / "input_audit.csv", index=False)
    checks = reference_checks(lambdas(config))
    checks["protocol_negative_controls"] = protocol_negative_controls(config)
    save_json(destination / "checks.json", checks)
    versions = {}
    for name in ["factorlasso", "numpy", "pandas", "scipy", "cvxpy",
                 "scikit-learn", "clarabel", "pytest"]:
        versions[name] = metadata.version(name)
    status = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--short"], text=True)
    revision = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    universe = pd.read_csv(root / "source_snapshot/data/etf_universe.csv")
    credit = universe[universe.sub_asset_class.isin(
        ["Global IG Bonds", "Global HY Bonds", "EM Bonds"])]
    report = {
        "stage": "E0", "status": "complete", "python": sys.executable,
        "versions": versions, "factorlasso_import": factorlasso.__file__,
        "source_declared_version": tomllib.loads(
            (root / "source_snapshot/pyproject.toml").read_text(encoding="utf-8")
        )["project"]["version"],
        "version_caveat": "Editable distribution metadata differs from frozen source declaration",
        "git_revision": revision, "checkout_status_at_snapshot": status,
        "solvers": cvxpy.installed_solvers(), "data_status": config["data_status"],
        "all_etfs": len(y.columns), "credit_cohort": credit.ticker.tolist(),
        "complete_subset_n": len(complete),
        "complete_subset_start": str(complete.index[0]),
        "complete_subset_end": str(complete.index[-1]),
        "empirical_training_end": str(complete.index[59]),
        "empirical_validation_end": str(complete.index[83]),
        "empirical_test_start": str(complete.index[84]),
        "pilot_cells": 12, "pilot_datasets": 240, "pilot_methods": 8,
        "snapshot_frozen": True,
        "practical_effect": config["practical_effect"],
    }
    finish(destination, report)
    return report


def empirical_target(x, y, asset, factor_names):
    """Fit a small declared economic model using the current training slice."""
    names = ["Equity"] if asset == "SPY" else (
        ["Rates"] if asset == "TLT" else ["Rates", "Credit"])
    selected = [factor_names.index(n) for n in names]
    design = np.column_stack([np.ones(len(x)), x[:, selected]])
    beta = np.linalg.lstsq(design, y, rcond=None)[0][1:]
    result = np.zeros(x.shape[1])
    result[selected] = beta
    return result


def package_ablations(x, y, config):
    """Isolate target, signs, adaptive weights, grouping and weighting changes."""
    from factorlasso import LassoModel, LassoModelType as MT

    signs = pd.DataFrame(np.nan, index=y.columns, columns=x.columns)
    signs.loc[:, ["Equity", "Rates", "Credit", "Carry"]] = 1.
    signs.loc[:, "PrivateEquity"] = 0.
    signs.loc[:, "RatesVol"] = -1.
    settings = [
        ("L1", MT.LASSO, False, False),
        ("L1_sign", MT.LASSO, True, False),
        ("L1_sign_adaptive", MT.LASSO, True, True),
        ("HCGL", MT.HIERARCHICAL_CLUSTER_GROUP_LASSO, False, False),
        ("HCGL_sign", MT.HIERARCHICAL_CLUSTER_GROUP_LASSO, True, False),
        ("HCGL_sign_adaptive", MT.HIERARCHICAL_CLUSTER_GROUP_LASSO, True, True),
        ("FCGL", MT.FACTOR_CLUSTER_GROUP_LASSO, False, False),
        ("FCGL_sign", MT.FACTOR_CLUSTER_GROUP_LASSO, True, False),
        ("FCGL_sign_adaptive", MT.FACTOR_CLUSTER_GROUP_LASSO, True, True),
    ]
    records, warnings_log = [], []
    import warnings

    for span, (label, mode, signed, adaptive), auto, lam in itertools.product(
            [None, 36], settings, [False, True], [1e-6, 1e-5, 1e-4]):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = LassoModel(
                model_type=mode, span=span, reg_lambda=lam, cutoff_fraction=.4,
                factors_beta_loading_signs=signs if signed else None,
                auto_sign_adaptive_weights=adaptive, auto_sign_adaptive_floor=.5,
                apply_ols_prior=auto, solver="CLARABEL", warmup_period=None,
            ).fit(x, y)
        warnings_log.extend([{"setting": label, "span": span, "lambda": lam,
                              "automatic": auto, "warning": str(w.message)}
                             for w in caught])
        if not np.isfinite(model.coef_.to_numpy()).all():
            raise AssertionError("Nonfinite package fit")
        if signed:
            b = model.coef_.to_numpy()
            s = signs.to_numpy()
            assert np.max(np.abs(b[s == 0]), initial=0.) < 1e-5
            assert np.min((b * s)[np.isfinite(s) & (s != 0)], initial=0.) > -1e-5
        for asset in y.columns:
            target = model.effective_beta_prior_
            raw = model.ols_beta_prior_
            winner = str(model.ols_r2_.loc[asset].idxmax()) if auto else ""
            row = {
                "asset": asset, "setting": label, "span": span or 0, "lambda": lam,
                "automatic": auto, "winner": winner,
                "r2": float(np.asarray(model.estimation_result_.r2)[y.columns.get_loc(asset)]),
                "fit_n": len(x), "training_start": str(x.index[0]),
                "training_end": str(x.index[-1]),
            }
            for factor in x.columns:
                row["beta_" + factor] = model.coef_.loc[asset, factor]
                row["target_" + factor] = 0. if target is None else target.loc[asset, factor]
                row["raw_target_" + factor] = 0. if raw is None else raw.loc[asset, factor]
            records.append(row)
    return pd.DataFrame(records), warnings_log


def plot_empirical(frame, destination):
    """Render the prespecified assets' native Credit-loading paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, asset in zip(axes, ["HYG", "LQD", "VCSH"]):
        part = frame[(frame.asset == asset) & (frame["lambda"] > 0)]
        for method in ["M1_zero", "M2_auto", "M3_small_model", "M4_free_winner"]:
            rows = part[part.method == method]
            ax.semilogx(rows["lambda"], rows.beta_Credit, label=method, linewidth=1.6)
        ax.set(title=asset, xlabel="Standardised loss penalty", ylabel="Native Credit beta")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle("Frozen ETF snapshot: full-sample diagnostic paths")
    fig.savefig(destination / "credit_paths.png", dpi=160)
    plt.close(fig)


def run_e1(root, destination, config):
    """Run reference checks, empirical paths and actual package ablations."""
    from papers.prior_targets_2026.replication.estimators import (
        fit_paths, prediction_losses, reference_checks, select_index,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "source_snapshot/src") + os.pathsep + str(REPO)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(REPO / "tests/test_ols_prior.py"),
         str(REPO / "tests/test_cv_lambda_path.py"), "-q", "-o", "addopts="],
        cwd=REPO, env=env, capture_output=True, text=True,
    )
    (destination / "existing_tests.log").write_text(
        result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError("Existing numerical contract tests failed")
    x, y = load_panels(root, config)
    panel = pd.concat([x, y[config["empirical_assets"]]], axis=1).dropna()
    x, y = panel[x.columns], panel[config["empirical_assets"]]
    grid = lambdas(config)
    paths_rows, selected_rows = [], []
    train, valid = config["empirical_training_months"], config["empirical_validation_months"]
    for asset in y:
        arrays = x.to_numpy(), y[asset].to_numpy()
        full = fit_paths(*arrays, grid, external_targets={"M3_small_model":
            empirical_target(*arrays, asset, list(x.columns))})
        training = arrays[0][:train], arrays[1][:train]
        validation = arrays[0][train:train + valid], arrays[1][train:train + valid]
        outer_train = arrays[0][:train + valid], arrays[1][:train + valid]
        test = arrays[0][train + valid:], arrays[1][train + valid:]
        fitted = fit_paths(*training, grid, external_targets={"M3_small_model":
            empirical_target(*training, asset, list(x.columns))})
        refitted = fit_paths(*outer_train, grid, external_targets={"M3_small_model":
            empirical_target(*outer_train, asset, list(x.columns))})
        for method, path in full.items():
            residual_losses = prediction_losses(*arrays, path["beta"], path["intercept"])
            for k, lam in enumerate(grid):
                row = {"asset": asset, "method": method, "lambda": lam,
                       "winner": x.columns[path["winner"]], "mse": residual_losses[k]}
                for j, name in enumerate(x.columns):
                    row["beta_" + name] = path["beta"][k, j]
                    row["target_" + name] = path["target"][j]
                paths_rows.append(row)
            candidate = fitted[method]
            scores = prediction_losses(*validation, candidate["beta"], candidate["intercept"])
            k = len(grid) - 1 if method == "M0_ols" else select_index(scores)
            final = refitted[method]
            losses = prediction_losses(*test, final["beta"], final["intercept"])
            row = {
                "asset": asset, "method": method, "lambda": grid[k],
                "validation_mse": scores[k], "test_mse": losses[k],
                "test_n": len(test[1]), "test_start": str(x.index[train + valid]),
                "winner": x.columns[final["winner"]], "grid_edge": k in [0, len(grid)-1],
            }
            for j, factor in enumerate(x.columns):
                row["beta_" + factor] = final["beta"][k, j]
            selected_rows.append(row)
    paths = pd.DataFrame(paths_rows)
    paths.to_csv(destination / "empirical_paths.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(destination / "empirical_selected.csv", index=False)
    ablations, warnings_log = package_ablations(x, y, config)
    ablations.to_csv(destination / "package_ablations.csv", index=False)
    save_json(destination / "solver_warnings.json", warnings_log)
    checks = reference_checks(grid)
    save_json(destination / "checks.json", checks)
    plot_empirical(paths, destination)
    report = {
        "stage": "E1", "status": "complete", "assets": list(y.columns),
        "factors": list(x.columns), "n": len(x), "paths_rows": len(paths),
        "selected_rows": len(selected_rows), "ablation_rows": len(ablations),
        "package_fit_count": len(ablations) // len(y.columns),
        "test_exit_code": result.returncode,
        "empirical_status": "Exploratory frozen-snapshot diagnosis; no empirical true betas",
        "M3": "Training-only small economic regression, not independent holdings information",
        "package_subset_caveat": "Six responses; group results are not the original 102-asset fit",
        "warnings": len(warnings_log),
    }
    finish(destination, report)
    return report


def plot_pilot(summary, destination):
    """Render error and prediction trade-offs on the pilot's two sample sizes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for column, n in enumerate([60, 240]):
        part = summary[(summary.n == n) & (summary.secondary == .3)]
        for method in ["M0_ols", "M1_zero", "M2_auto", "M3_fixed",
                       "M3_noisy", "M4_free_winner", "M5_post_lasso"]:
            rows = part[part.method == method].sort_values("rho")
            axes[0, column].plot(rows.rho, rows.secondary_mse, marker="o",
                                 label=method, linewidth=1.5)
            axes[1, column].plot(rows.rho, rows.prediction_mse_delta, marker="o",
                                 linewidth=1.5)
        axes[0, column].set(title=f"Training n={n}", ylabel="Secondary loading MSE")
        axes[1, column].set(xlabel="Factor correlation", ylabel="Prediction MSE minus zero target")
        axes[1, column].axhline(0, color="grey", linewidth=.7)
        for ax in axes[:, column]:
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Exploratory pilot: true secondary loading 0.3; 20 paired seeds")
    fig.savefig(destination / "pilot_comparison.png", dpi=160)
    plt.close(fig)


def run_e2(root, destination, config, profile):
    """Run paired synthetic comparisons without using test data for selection."""
    from papers.prior_targets_2026.replication.estimators import (
        fit_paths, paired_summary, prediction_losses, select_index,
    )
    design, grid = config[profile], lambdas(config)
    scenarios = np.array(config["scenarios"]["vectors"], float)
    cells = list(itertools.product(design["rho"], design["secondary"],
                                   design["n"], design["snr"]))
    records, failures = [], []
    started = time.perf_counter()
    for cell_id, (rho, secondary, n, snr) in enumerate(cells):
        sigma = np.array([[1., rho], [rho, 1.]])
        factor = np.linalg.cholesky(sigma)
        np.testing.assert_allclose(factor @ factor.T, sigma, atol=1e-14)
        beta = np.array([1., secondary])
        signal_var = float(1 + secondary**2 + 2 * rho * secondary)
        np.testing.assert_allclose(beta @ sigma @ beta, signal_var, atol=1e-14)
        noise_var = signal_var / snr
        response_var = signal_var + noise_var
        for seed in range(design["seeds"][0], design["seeds"][1] + 1):
            samples = []
            for stream, size in enumerate([n, n, design["test_size"]]):
                rng = np.random.default_rng(np.random.SeedSequence([seed, cell_id, stream]))
                x = rng.normal(size=(size, 2)) @ factor.T
                y = x @ beta + np.sqrt(noise_var) * rng.normal(size=size)
                samples.append((x, y))
            target_rng = np.random.default_rng(np.random.SeedSequence([seed, cell_id, 3]))
            external = {
                "M3_fixed": np.array([1., .3]),
                "M3_noisy": beta + target_rng.normal(scale=.25, size=2),
                "M3_oracle": beta.copy(),
            }
            try:
                paths = fit_paths(*samples[0], grid, external_targets=external)
                for method, path in paths.items():
                    losses = prediction_losses(*samples[1], path["beta"], path["intercept"])
                    chosen = len(grid) - 1 if method == "M0_ols" else select_index(losses)
                    fitted = path["beta"][chosen]
                    error = fitted - beta
                    scenario_errors = scenarios @ error
                    prediction = prediction_losses(*samples[2], path["beta"], path["intercept"])
                    row = {
                        "rho": rho, "secondary": secondary, "n": n, "snr": snr,
                        "seed": seed, "cell_id": cell_id, "method": method,
                        "lambda": grid[chosen], "validation_mse": losses[chosen],
                        "prediction_mse": prediction[chosen],
                        "beta_mse": float(np.mean(error**2)),
                        "secondary_mse": float(error[1]**2),
                        "scenario_mse": float(np.mean(scenario_errors**2) / response_var),
                        "isolated_credit_scenario_mse": float(error[1]**2 / response_var),
                        "beta_dominant": fitted[0], "beta_secondary": fitted[1],
                        "target_dominant": path["target"][0],
                        "target_secondary": path["target"][1],
                        "winner": path["winner"], "winner_r2": path["winner_r2"],
                        "secondary_zero": abs(fitted[1]) <= config["zero_tolerance"],
                        "grid_edge": chosen in [0, len(grid) - 1],
                        "dual_gap": path["gap"], "status": "ok",
                        "oracle_target": method == "M3_oracle",
                        "independent_measurement_target": method == "M3_noisy",
                        "signal_prediction_mse_population": float(error @ sigma @ error),
                    }
                    records.append(row)
            except Exception as error:
                failures.append({"cell": cell_id, "seed": seed, "error": repr(error)})
        print(f"{profile} cell {cell_id + 1}/{len(cells)} complete", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(destination / "selected_results.csv", index=False)
    summary = paired_summary(frame)
    summary.to_csv(destination / "paired_summary.csv", index=False)
    save_json(destination / "failures.json", failures)
    if profile == "pilot":
        plot_pilot(summary, destination)
    report = {
        "stage": "E2", "profile": profile, "status": "base_grid_complete",
        "expected_datasets": len(cells) * (design["seeds"][1] - design["seeds"][0] + 1),
        "expected_rows": len(cells) * (design["seeds"][1] - design["seeds"][0] + 1) * 8,
        "actual_rows": len(frame), "methods": sorted(frame.method.unique()),
        "failures": len(failures), "seconds": time.perf_counter() - started,
        "seed_range": design["seeds"],
        "information_sets": {
            "M3_fixed": "Fixed target [1,.3], correct only on one DGP family",
            "M3_noisy": "Independent noisy measurement of the true coefficients; extra information",
            "M3_oracle": "Exact truth; diagnostic only",
        },
        "pending": "Misspecification, null-response, lagged-target and nonstationary stress cells",
        "theory": "Not developed",
    }
    finish(destination, report)
    return report


def verify(root, destination, stage, config):
    """Validate receipt hashes and scientific completeness rather than exit codes."""
    check_snapshot(root)
    manifest = read_json(destination / "manifest.json")
    if not manifest["artifacts"]:
        raise AssertionError("Empty stage")
    for name, expected in manifest["artifacts"].items():
        if digest(destination / name) != expected:
            raise AssertionError(f"Artifact drift: {name}")
    report = read_json(destination / "summary.json")
    validate_protocol(config)
    if stage == "E0":
        assert report["snapshot_frozen"] and report["solvers"]
        assert report["git_revision"] and report["data_status"]
        assert report["complete_subset_n"] > 84
        assert len(read_json(destination / "checks.json")["protocol_negative_controls"]) == 5
    elif stage == "E1":
        assert report["test_exit_code"] == 0 and report["selected_rows"] == 36
        assert report["ablation_rows"] == 648
        paths = pd.read_csv(destination / "empirical_paths.csv")
        assert len(paths) == report["paths_rows"] == 6 * 6 * len(lambdas(config))
        assert np.isfinite(paths.filter(like="beta_").to_numpy()).all()
        selected = pd.read_csv(destination / "empirical_selected.csv")
        assert selected.test_n.min() > 0
    elif stage == "E2":
        assert report["failures"] == 0
        frame = pd.read_csv(destination / "selected_results.csv")
        assert len(frame) == report["expected_rows"]
        keys = ["rho", "secondary", "n", "snr", "seed"]
        assert (frame.groupby(keys).method.nunique() == 8).all()
        assert not frame.duplicated(keys + ["method"]).any()
        assert np.isfinite(frame[["prediction_mse", "scenario_mse"]]).all().all()
        np.testing.assert_allclose(
            frame.secondary_mse, (frame.beta_secondary - frame.secondary) ** 2,
            atol=1e-12, rtol=1e-10,
        )
        assert frame.loc[frame.method == "M3_oracle", "oracle_target"].all()
        assert not frame.loc[frame.method != "M3_oracle", "oracle_target"].any()
    return {"verified": stage, "profile": report.get("profile"),
            "artifacts": len(manifest["artifacts"])}


def main():
    """Execute one bounded immutable stage or verify a completed stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "verify"])
    parser.add_argument("--stage", choices=["E0", "E1", "E2"], required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile", choices=["pilot", "confirmation"], default="pilot")
    args = parser.parse_args()
    root = validate_root(args.output_root)
    if args.command == "run" and args.stage == "E0":
        if root.exists():
            raise FileExistsError("Use a new immutable run root")
        root.mkdir(parents=True)
        snapshot(root)
    check_snapshot(root)
    sys.path.insert(0, str(root / "source_snapshot/src"))
    config = read_json(root / "source_snapshot/protocol.json")
    validate_protocol(config)
    name = args.stage if args.stage != "E2" else "E2_" + args.profile
    destination = root / name
    if args.command == "verify":
        report = verify(root, destination, args.stage, config)
    else:
        if args.stage != "E0":
            verify(root, root / "E0", "E0", config)
        if args.stage == "E2":
            verify(root, root / "E1", "E1", config)
        destination.mkdir()
        started = time.perf_counter()
        if args.stage == "E0":
            report = run_e0(root, destination, config)
        elif args.stage == "E1":
            report = run_e1(root, destination, config)
        else:
            report = run_e2(root, destination, config, args.profile)
        print(f"Stage runtime {time.perf_counter() - started:.2f}s", flush=True)
        verify(root, destination, args.stage, config)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()


