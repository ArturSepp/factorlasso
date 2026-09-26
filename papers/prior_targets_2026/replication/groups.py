"""E3 group-geometry extension using the frozen factorlasso path implementation."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib import metadata
import json
from pathlib import Path
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd

from papers.prior_targets_2026.replication.run import (
    HERE, check_snapshot, digest, lambdas, read_json, save_json, validate_root,
)
from papers.prior_targets_2026.replication.multiasset import (
    assemble, calibration, check_manifest, samples_for,
)

METRICS = ["credit_mse", "credit_bias", "beta_mse", "scenario_mse",
           "prediction_nmse", "credit_prediction_nmse", "systematic_covar_error",
           "full_covar_error"]


def model_for(setting, target, prior, groups, solver="CLARABEL"):
    """Build the unchanged package specification with declared prior provenance."""
    from factorlasso import LassoModel, LassoModelType
    modes = {"HCGL": LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
             "FCGL": LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
             "GROUP": LassoModelType.GROUP_LASSO}
    return LassoModel(
        model_type=modes[setting["geometry"]], group_data=groups if setting["known"] else None,
        cutoff_fraction=.4, span=None, warmup_period=None, solver=solver,
        solver_fallbacks=["ECOS"] if solver == "CLARABEL" else None,
        auto_sign_constraints=setting["signs"], auto_sign_threshold_t=1.,
        auto_sign_adaptive_weights=setting["adaptive"], auto_sign_adaptive_gamma=1.,
        auto_sign_adaptive_floor=.5, apply_ols_prior=target == "auto",
        factors_beta_prior=prior if target == "economic" else None,
    )


@contextmanager
def observe_solver_calls():
    """Record completed dispatches without modifying the solver or objective."""
    import factorlasso.lasso_estimator as module
    original, events = module._solve_with_fallback, []

    def observed(problem, solver, solver_fallbacks=None, **kwargs):
        """Delegate exactly once and retain the final backend and status."""
        original(problem, solver, solver_fallbacks, **kwargs)
        actual = problem.solver_stats.solver_name
        events.append(dict(lambda_index=len(events), requested_solver=solver,
                           final_solver=actual, fallback=actual != solver,
                           status=problem.status))

    module._solve_with_fallback = observed
    try:
        yield events
    finally:
        module._solve_with_fallback = original


def fit_paths(x, y, data, setting, target, grid):
    """Scale training data, then call the package path API without custom penalties."""
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    xs, ys = x.std(axis=0), y.std(axis=0)
    z = pd.DataFrame((x - xm) / xs, columns=data["factors"])
    v = pd.DataFrame((y - ym) / ys, columns=data["tickers"])
    prior = pd.DataFrame(data["prior"] * xs[None] / ys[:, None],
                         index=data["tickers"], columns=data["factors"])
    groups = pd.Series(data["groups"], index=data["tickers"])
    specification = model_for(setting, target, prior, groups)
    with warnings.catch_warnings(record=True) as caught, observe_solver_calls() as events:
        warnings.simplefilter("always")
        models = specification.fit_reg_lambda_path(z, v, grid)
    assert len(events) == len(grid)
    beta, intercept, raw_centres, effective = [], [], [], []
    for model in models:
        standard = model.coef_.to_numpy()
        assert np.isfinite(standard).all()
        if model.derived_signs_ is not None:
            signs = model.derived_signs_.to_numpy()
            assert np.max(abs(standard[signs == 0]), initial=0.) < 3e-5
            nz = np.isfinite(signs) & (signs != 0)
            assert np.min(standard[nz] * signs[nz], initial=0.) > -3e-5
        b = standard * ys[:, None] / xs[None]
        beta.append(b)
        intercept.append(ym + model.alpha_const_.to_numpy() * ys - b @ xm)
        if target == "auto":
            raw = model.ols_beta_prior_.to_numpy()
            centre = model.effective_beta_prior_.to_numpy()
        elif target == "economic":
            raw, centre = prior.to_numpy(), prior.to_numpy()
        else:
            raw = centre = np.zeros_like(b)
        raw_centres.append(raw * ys[:, None] / xs[None])
        effective.append(centre * ys[:, None] / xs[None])
    return (models, np.array(beta), np.array(intercept), np.array(raw_centres),
            np.array(effective), [str(w.message) for w in caught], events)


def panel_run(data, profile, seed, config):
    """Fit every setting/target on the same independent train/validation/test panel."""
    from papers.prior_targets_2026.replication.estimators import select_index
    from sklearn.metrics import adjusted_rand_score
    n, grid = config["n"], lambdas(config)
    samples, _, truth, sigma, noise, variance = samples_for(data, profile, n, seed, config)
    (xt, yt), (xv, yv), (xe, ye) = samples
    credit = data["credit_indices"]
    ci = list(data["factors"]).index("Credit")
    factor_covar = np.cov(xt, rowvar=False, ddof=1)
    true_systematic = assemble(truth, sigma, np.zeros(len(truth)), data)
    true_full = assemble(truth, sigma, noise, data)
    assets, panels, paths, messages, solvers = [], [], [], [], []
    for setting in config["settings"]:
        for target in config["targets"]:
            models, betas, intercepts, raw, effective, caught, events = fit_paths(
                xt, yt, data, setting, target, grid)
            val_predictions = np.einsum("tf,kaf->kta", xv, betas) + intercepts[:, None, :]
            per_asset_loss = np.mean((yv[None] - val_predictions)**2, axis=1)
            scores = np.mean(per_asset_loss / yt.var(axis=0)[None], axis=1)
            k = select_index(scores)
            b, alpha, model = betas[k], intercepts[k], models[k]
            cross_error = None
            if seed == config["pilot_seeds"][0]:
                xs, ys = xt.std(0), yt.std(0)
                z = pd.DataFrame((xt - xt.mean(0)) / xs, columns=data["factors"])
                v = pd.DataFrame((yt - yt.mean(0)) / ys, columns=data["tickers"])
                p = pd.DataFrame(data["prior"] * xs[None] / ys[:, None],
                                 index=data["tickers"], columns=data["factors"])
                g = pd.Series(data["groups"], index=data["tickers"])
                fresh = model_for(setting, target, p, g, solver="ECOS")
                fresh.reg_lambda = grid[k]
                fresh.fit(z, v)
                cross_error = float(np.max(abs(fresh.coef_.to_numpy() - model.coef_.to_numpy())))
                assert cross_error < 5e-4, cross_error
            solvers.extend(dict(profile=profile, seed=seed, setting=setting["name"],
                                target=target, reg_lambda=grid[e["lambda_index"]], **e)
                           for e in events)
            cluster = model.clusters_.reindex(data["tickers"])
            signs = (np.full_like(b, np.nan) if model.derived_signs_ is None
                     else model.derived_signs_.to_numpy())
            error = b - truth
            test_mse = np.mean((ye - xe @ b.T - alpha)**2, axis=0)
            residual_var = np.mean((yt - xt @ b.T - alpha)**2, axis=0)
            sys_est = assemble(b, factor_covar, np.zeros(len(b)), data)
            full_est = assemble(b, factor_covar, residual_var, data)
            row = dict(
                profile=profile, n=n, seed=seed, setting=setting["name"], target=target,
                reg_lambda=grid[k], lambda_index=k, validation_nmse=float(scores[k]),
                credit_mse=float(np.mean(error[credit, ci]**2)),
                credit_bias=float(np.mean(error[credit, ci])),
                beta_mse=float(np.mean(error**2)),
                scenario_mse=float(np.mean(np.mean(error**2 * np.diag(sigma), axis=1) / variance)),
                prediction_nmse=float(np.mean(test_mse / variance)),
                credit_prediction_nmse=float(np.mean(test_mse[credit] / variance[credit])),
                systematic_covar_error=float(np.linalg.norm(sys_est - true_systematic)
                                             / np.linalg.norm(true_systematic)),
                full_covar_error=float(np.linalg.norm(full_est - true_full)
                                       / np.linalg.norm(true_full)),
                cluster_count=int(cluster.nunique()),
                cluster_ari=float(adjusted_rand_score(data["groups"], cluster)),
                true_groups=setting["known"], warning_count=len(caught),
                selected_solver=events[k]["final_solver"],
                fallback_count=sum(e["fallback"] for e in events),
                cross_solver_selected_error=cross_error,
            )
            panels.append(row)
            for j, ticker in enumerate(data["tickers"]):
                record = dict(profile=profile, n=n, seed=seed, setting=setting["name"],
                              target=target, asset=j, ticker=ticker, is_credit=j in credit,
                              cluster=str(cluster.iloc[j]), reg_lambda=grid[k],
                              intercept=alpha[j], credit_mse=error[j, ci]**2,
                              prediction_nmse=test_mse[j] / variance[j],
                              residual_var=residual_var[j])
                for f in range(b.shape[1]):
                    record.update({f"beta{f}": b[j, f], f"truth{f}": truth[j, f],
                                   f"raw_target{f}": raw[k, j, f],
                                   f"target{f}": effective[k, j, f],
                                   f"sign{f}": signs[j, f]})
                assets.append(record)
            paths.extend(dict(profile=profile, n=n, seed=seed, setting=setting["name"],
                              target=target, lambda_index=j, reg_lambda=lv,
                              validation_nmse=float(scores[j])) for j, lv in enumerate(grid))
            messages.extend(dict(profile=profile, seed=seed, setting=setting["name"],
                                 target=target, warning=message) for message in caught)
    return dict(assets=assets, panels=panels, paths=paths, warnings=messages, solvers=solvers)


def summary(frame):
    """Compare targets within each geometry with paired seed uncertainty."""
    result = []
    for (profile, setting), group in frame.groupby(["profile", "setting"]):
        base = group[group.target == "zero"].set_index("seed").sort_index()
        for target, part in group.groupby("target"):
            part = part.set_index("seed").sort_index()
            assert part.index.equals(base.index)
            row = dict(profile=profile, setting=setting, target=target, seeds=len(part))
            for metric in METRICS:
                delta = (part[metric] - base[metric]).to_numpy()
                se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
                row.update({metric: float(part[metric].mean()), metric + "_delta": delta.mean(),
                            metric + "_se": se, metric + "_low": delta.mean() - 1.96 * se,
                            metric + "_high": delta.mean() + 1.96 * se})
            result.append(row)
    return pd.DataFrame(result)


def validate_frames(assets, panels, config, phase):
    """Require all paired cells and independently recompute Credit errors."""
    seeds = set(range(config[phase + "_seeds"][0], config[phase + "_seeds"][1] + 1))
    count = len(seeds) * len(config["profiles"]) * len(config["settings"]) * 3
    assert len(panels) == count and len(assets) == count * 102
    keys = ["profile", "seed", "setting", "target"]
    assert not panels.duplicated(keys).any()
    assert not assets.duplicated(keys + ["asset"]).any()
    assert set(panels.seed) == seeds
    assert set(panels.setting) == {s["name"] for s in config["settings"]}
    assert set(panels.profile) == set(config["profiles"])
    assert set(panels.target) == set(config["targets"])
    assert (assets.groupby(keys).asset.nunique() == 102).all()
    assert np.isfinite(panels[METRICS]).all().all()
    np.testing.assert_allclose(assets.credit_mse, (assets.beta2 - assets.truth2)**2, atol=1e-11)
    aggregated = assets[assets.is_credit].groupby(keys).credit_mse.mean()
    saved = panels.set_index(keys).loc[aggregated.index]
    np.testing.assert_allclose(aggregated, saved.credit_mse, atol=1e-11)
    assert panels.loc[panels.setting.str.startswith("Known"), "true_groups"].all()


def verify(out, config, phase):
    """Check outputs, pairing, reference intervals and deliberate corruptions."""
    from papers.prior_targets_2026.replication.robustness import reject_assertion
    assets = pd.read_csv(out / "asset_results.csv")
    panels = pd.read_csv(out / "panel_results.csv")
    validate_frames(assets, panels, config, phase)
    table = pd.read_csv(out / "paired_summary.csv")
    for metric in METRICS:
        wide = panels.pivot(index=["profile", "setting", "seed"], columns="target", values=metric)
        for _, row in table.iterrows():
            block = wide.loc[(row.profile, row.setting)]
            d = (block[row.target] - block.zero).to_numpy()
            mean = sum(d) / len(d)
            se = np.sqrt((d - mean) @ (d - mean) / (len(d) * (len(d) - 1)))
            np.testing.assert_allclose([row[metric + "_delta"], row[metric + "_se"]],
                                       [mean, se], atol=1e-12)
    bad = assets.copy()
    bad.loc[0, "credit_mse"] += 1.
    reject_assertion(lambda: validate_frames(bad, panels, config, phase))
    reject_assertion(lambda: validate_frames(assets.iloc[1:], panels, config, phase))
    path = pd.read_csv(out / "validation_paths.csv")
    assert len(path) == len(panels) * len(lambdas(config))
    assert read_json(out / "failures.json") == []
    solvers = pd.read_csv(out / "solver_events.csv")
    assert len(solvers) == len(path)
    assert not solvers.duplicated(["profile", "seed", "setting", "target", "lambda_index"]).any()
    assert solvers.status.isin(["optimal", "optimal_inaccurate"]).all()
    np.testing.assert_array_equal(solvers.fallback, solvers.final_solver != "CLARABEL")
    if phase == "pilot":
        checked = panels.cross_solver_selected_error.dropna()
        assert len(checked) == 30 and checked.max() < 5e-4
    return dict(asset_rows=len(assets), panel_method_rows=len(panels),
                panels=len(panels) // (3 * len(config["settings"])),
                path_rows=len(path), corrupt_metric_rejected=True, missing_row_rejected=True,
                independent_paired_intervals_verified=True)


def numerical_checks(data, config):
    """Compare the path API with fresh standalone fits for all ablations."""
    fixture = {key: value.copy() for key, value in data.items()}
    subset = np.array([0, 1, 4, 7, 10, 15, 23, 35, 50, 70, 85, 100])
    for key in ["beta", "prior", "signs", "residual_var_annual", "tickers", "groups"]:
        fixture[key] = fixture[key][subset]
    fixture["credit_indices"] = np.flatnonzero(np.isin(subset, data["credit_indices"]))
    samples, _, _, _, _, _ = samples_for(
        fixture, "baseline", config["n"], 959999, config)
    xt, yt = samples[0]
    xm, ym, xs, ys = xt.mean(0), yt.mean(0), xt.std(0), yt.std(0)
    z = pd.DataFrame((xt - xm) / xs, columns=fixture["factors"])
    v = pd.DataFrame((yt - ym) / ys, columns=fixture["tickers"])
    prior = pd.DataFrame(fixture["prior"] * xs[None] / ys[:, None],
                         index=fixture["tickers"], columns=fixture["factors"])
    groups = pd.Series(fixture["groups"], index=fixture["tickers"])
    maximum, cross_maximum, comparisons = 0., 0., 0
    grid = np.array([.2, .01, 0.])
    for setting in config["settings"]:
        for target in config["targets"]:
            models, _, _, _, _, _, _ = fit_paths(xt, yt, fixture, setting, target, grid)
            for k, lam in enumerate(grid):
                fresh = model_for(setting, target, prior, groups)
                fresh.reg_lambda = lam
                fresh.fit(z, v)
                difference = np.max(abs(fresh.coef_.to_numpy() - models[k].coef_.to_numpy()))
                maximum = max(maximum, float(difference))
                comparisons += 1
                independent = model_for(setting, target, prior, groups, solver="ECOS")
                independent.reg_lambda = lam
                independent.fit(z, v)
                cross_maximum = max(cross_maximum, float(np.max(abs(
                    independent.coef_.to_numpy() - models[k].coef_.to_numpy()))))
    assert maximum < 5e-4 and cross_maximum < 5e-4
    # Inputs to fitting are train-only. Future changes cannot affect their digest.
    train_before = (xt.copy(), yt.copy())
    samples[1][0][:] += 999
    samples[2][1][:] -= 999
    np.testing.assert_array_equal(xt, train_before[0])
    np.testing.assert_array_equal(yt, train_before[1])
    repeated = fit_paths(xt, yt, fixture, config["settings"][-1], "economic", grid)[0]
    for first, second in zip(models, repeated):
        np.testing.assert_allclose(first.coef_, second.coef_, atol=1e-10)
    # A shifted fitted path must fail the same reference tolerance.
    try:
        np.testing.assert_allclose(models[0].coef_, models[0].coef_ + .01, atol=5e-4)
    except AssertionError:
        rejected = True
    else:
        raise AssertionError("Deliberate coefficient defect was accepted")
    return dict(standalone_comparisons=comparisons, max_standardised_beta_error=maximum,
                cross_solver_max_error=cross_maximum,
                tolerance=.0005, shifted_coefficients_rejected=rejected,
                train_only_preprocessing_verified=True)


def main():
    """Run resumable immutable pilot/confirmation batches with C-local checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "resume", "verify"])
    parser.add_argument("--phase", choices=["pilot", "confirmation"], default="pilot")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=[1, 2, 3, 4], default=4)
    args = parser.parse_args()
    root = validate_root(args.output_root)
    check_snapshot(root)
    sys.path.insert(0, str(root / "source_snapshot/src"))
    check_manifest(root / "E3_L1_confirmation_v1")
    data = calibration(root)
    uni = pd.read_csv(root / "source_snapshot/data/etf_universe.csv").set_index("ticker")
    data["groups"] = uni.loc[data["tickers"], "sub_asset_class"].to_numpy()
    config = read_json(HERE / "groups_protocol_v2.json")
    out = root / ("E3_groups_" + args.phase + "_v2")
    if args.command == "verify":
        check_manifest(out)
        print(json.dumps(verify(out, read_json(out / "archive/groups_protocol_v2.json"),
                                args.phase), indent=2))
        return
    if args.phase == "confirmation":
        pilot = root / "E3_groups_pilot_v2"
        check_manifest(pilot)
        assert read_json(pilot / "summary.json")["failures"] == 0
        assert digest(pilot / "archive/groups_protocol_v2.json") == digest(
            HERE / "groups_protocol_v2.json")
    if args.command == "run":
        out.mkdir()
        (out / "archive").mkdir()
        (out / "chunks").mkdir()
        for name in ["groups.py", "groups_protocol_v2.json", "multiasset.py", "run.py",
                     "estimators.py", "robustness.py"]:
            shutil.copy2(HERE / name, out / "archive" / name)
        save_json(out / "numerical_checks.json", numerical_checks(data, config))
    else:
        assert not (out / "manifest.json").exists(), "Completed runs are immutable"
        for file in (out / "archive").iterdir():
            assert digest(file) == digest(HERE / file.name), "Source drift during resume"
    assets, panels, paths, messages, failures, solvers = [], [], [], [], [], []
    start = time.perf_counter()
    seeds = range(config[args.phase + "_seeds"][0], config[args.phase + "_seeds"][1] + 1)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {}
        for profile in config["profiles"]:
            for seed in seeds:
                chunk = out / "chunks" / f"{profile}_{seed}.json"
                if not chunk.exists():
                    jobs[pool.submit(panel_run, data, profile, seed, config)] = (
                        profile, seed, chunk)
        for job in as_completed(jobs):
            profile, seed, chunk = jobs[job]
            try:
                result = job.result()
                # JSON cannot encode NaN signs; missing constraints are stored as null.
                for row in result["assets"]:
                    for j in range(9):
                        if not np.isfinite(row[f"sign{j}"]):
                            row[f"sign{j}"] = None
                save_json(chunk, result)
            except Exception as error:
                failures.append(dict(profile=profile, seed=seed, error=repr(error)))
            print(f"{args.phase}: {profile} seed {seed} complete", flush=True)
    for profile in config["profiles"]:
        for seed in seeds:
            chunk = out / "chunks" / f"{profile}_{seed}.json"
            if chunk.exists():
                result = read_json(chunk)
                assets.extend(result["assets"])
                panels.extend(result["panels"])
                paths.extend(result["paths"])
                messages.extend(result["warnings"])
                solvers.extend(result["solvers"])
    pd.DataFrame(assets).to_csv(out / "asset_results.csv", index=False)
    frame = pd.DataFrame(panels)
    frame.to_csv(out / "panel_results.csv", index=False)
    summary(frame).to_csv(out / "paired_summary.csv", index=False)
    pd.DataFrame(paths).to_csv(out / "validation_paths.csv", index=False)
    save_json(out / "solver_warnings.json", messages)
    pd.DataFrame(solvers).to_csv(out / "solver_events.csv", index=False)
    save_json(out / "failures.json", failures)
    receipt = verify(out, config, args.phase)
    save_json(out / "verification.json", receipt)
    original = read_json(root / "E0/summary.json")["versions"]
    current = {name: metadata.version(name) for name in original}
    assert current == original
    save_json(out / "summary.json", {
        **receipt, "phase": args.phase, "seconds_this_invocation": time.perf_counter() - start,
        "failures": len(failures), "warnings": len(messages), "workers": args.workers,
        "fallback_solves": sum(e["fallback"] for e in solvers),
        "seeds": config[args.phase + "_seeds"], "versions": current,
        "protocol_sha256": digest(HERE / "groups_protocol_v2.json"),
        "prerequisite_manifest_sha256": digest(root / "E3_L1_confirmation_v1/manifest.json"),
    })
    save_json(out / "manifest.json", {
        str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()
    })
    print(json.dumps(read_json(out / "summary.json"), indent=2))


if __name__ == "__main__":
    main()

