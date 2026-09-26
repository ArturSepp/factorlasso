"""E3 fixed-geometry transfer: independently tuned per-asset prior-centred L1.

This is a new protocol, not a rerun of the legacy oracle/BIC/group comparison.
The audited calibration is reused without altering any legacy paper output.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
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

METRICS = ["credit_mse", "credit_bias", "zero_credit_mse", "beta_mse",
           "scenario_mse", "prediction_nmse", "credit_prediction_nmse",
           "systematic_covar_error", "full_covar_error", "oracle_components_covar_error"]


def check_manifest(path):
    """Require every frozen artifact to retain its recorded content hash."""
    for name, expected in read_json(path / "manifest.json").items():
        assert digest(path / name) == expected, name


def calibration(root):
    """Load only the independently audited, immutable calibration arrays."""
    source = root / "E3_calibration_audit_v1"
    check_manifest(source)
    with np.load(source / "calibration.npz") as data:
        return {name: data[name].copy() for name in data.files}


def generating_parameters(data, profile):
    """Apply the prespecified loading perturbation with noise held fixed."""
    beta = data["beta"].copy()
    ci = list(data["factors"]).index("Credit")
    ri = list(data["factors"]).index("Rates")
    if profile == "credit_removed":
        beta[data["credit_indices"], ci] = 0.
    elif profile == "rates_dominant":
        beta[data["credit_indices"], ri] *= 4.
    elif profile != "baseline":
        raise ValueError(profile)
    sigma, noise = data["sigma_annual"] / 12, data["residual_var_annual"] / 12
    response_var = np.einsum("ij,jk,ik->i", beta, sigma, beta) + noise
    return beta, sigma, noise, response_var


def samples_for(data, profile, n, seed, config):
    """Use independent train/validation/test streams, paired across profiles."""
    beta, sigma, noise, response_var = generating_parameters(data, profile)
    chol = np.linalg.cholesky(sigma)
    samples = []
    for stream, size in enumerate([n, n, config["test_size"]]):
        rng = np.random.default_rng(np.random.SeedSequence([seed, n, stream]))
        x = rng.normal(size=(size, beta.shape[1])) @ chol.T
        eps = rng.normal(size=(size, beta.shape[0])) * np.sqrt(noise)
        samples.append((x, x @ beta.T + eps))
    rng = np.random.default_rng(np.random.SeedSequence([seed, n, 3]))
    prior_noise = rng.normal(scale=.10, size=beta.shape[0])
    return samples, prior_noise, beta, sigma, noise, response_var


def fit_asset(train_x, train_y, val_x, val_y, economic, truth, prior_noise,
              is_credit, factors, grid):
    """Construct fold-local targets and choose each path from validation only."""
    from papers.prior_targets_2026.replication.estimators import fit_paths, prediction_losses, select_index
    ci, ri = factors.index("Credit"), factors.index("Rates")
    noisy, small = economic.copy(), np.zeros(len(factors))
    if is_credit:
        noisy[ci] += prior_noise
        narrow = np.column_stack([np.ones(len(train_y)), train_x[:, [ri, ci]]])
        small[[ri, ci]] = np.linalg.lstsq(narrow, train_y, rcond=None)[0][1:]
    paths = fit_paths(train_x, train_y, grid, external_targets={
        "M3_economic": economic, "M3_economic_noisy": noisy,
        "M3_small_model": small, "M3_oracle": truth,
    })
    selected = {}
    for method, path in paths.items():
        loss = prediction_losses(val_x, val_y, path["beta"], path["intercept"])
        k = len(grid) - 1 if method == "M0_ols" else select_index(loss)
        selected[method] = dict(beta=path["beta"][k], intercept=path["intercept"][k],
                                target=path["target"], winner=path["winner"],
                                lambda_index=k, reg_lambda=grid[k],
                                validation_mse=loss[k], gap=path["gap"])
    return selected


def assemble(beta, sigma, residual, data):
    """Use the exported factorlasso container for covariance construction."""
    from factorlasso import CurrentFactorCovarData, VarianceColumns
    container = CurrentFactorCovarData(
        x_covar=pd.DataFrame(sigma, index=data["factors"], columns=data["factors"]),
        y_betas=pd.DataFrame(beta, index=data["tickers"], columns=data["factors"]),
        y_variances=pd.DataFrame({VarianceColumns.RESIDUAL_VARS: residual},
                                index=data["tickers"]),
    )
    result = container.get_y_covar().to_numpy()
    reference = np.einsum("ik,kl,jl->ij", beta, sigma, beta) + np.diag(residual)
    np.testing.assert_allclose(result, reference, atol=1e-12, rtol=1e-10)
    return result


def panel_rows(data, profile, n, seed, config):
    """Return all asset errors and full-panel errors for one paired replication."""
    samples, prior_noise, beta, sigma, noise, response_var = samples_for(
        data, profile, n, seed, config)
    (xt, yt), (xv, yv), (xe, ye) = samples
    factors, tickers = list(data["factors"]), list(data["tickers"])
    ci = factors.index("Credit")
    credit_mask = np.isin(np.arange(len(tickers)), data["credit_indices"])
    methods, grid = config["methods"], lambdas(config)
    fitted = {m: np.zeros_like(beta) for m in methods}
    intercepts = {m: np.zeros(len(tickers)) for m in methods}
    assets = []
    for asset, ticker in enumerate(tickers):
        results = fit_asset(xt, yt[:, asset], xv, yv[:, asset], data["prior"][asset],
                            beta[asset], prior_noise[asset], credit_mask[asset], factors, grid)
        assert set(results) == set(methods)
        for method, result in results.items():
            bh, intercept = result["beta"], result["intercept"]
            fitted[method][asset], intercepts[method][asset] = bh, intercept
            error = bh - beta[asset]
            prediction = np.mean((ye[:, asset] - xe @ bh - intercept)**2)
            row = dict(profile=profile, n=n, seed=seed, asset=asset, ticker=ticker,
                       method=method, is_credit=bool(credit_mask[asset]),
                       beta_mse=float(np.mean(error**2)), credit_mse=error[ci]**2,
                       credit_bias=error[ci], prediction_nmse=prediction / response_var[asset],
                       scenario_mse=float(np.mean(error**2 * np.diag(sigma))
                                          / response_var[asset]),
                       true_credit=beta[asset, ci], estimated_credit=bh[ci],
                       response_var=response_var[asset], intercept=intercept,
                       winner=result["winner"], lambda_index=result["lambda_index"],
                       reg_lambda=result["reg_lambda"], validation_mse=result["validation_mse"],
                       residual_var=float(np.mean((yt[:, asset] - xt @ bh - intercept)**2)),
                       dual_gap=result["gap"], oracle_target=method == "M3_oracle")
            for j in range(len(factors)):
                row.update({f"beta{j}": bh[j], f"target{j}": result["target"][j],
                            f"truth{j}": beta[asset, j]})
            assets.append(row)
    frame = pd.DataFrame(assets)
    factor_estimate = np.cov(xt, rowvar=False, ddof=1)
    centred = xt - xt.mean(axis=0)
    np.testing.assert_allclose(factor_estimate, centred.T @ centred / (n - 1), atol=1e-14)
    systematic_truth = assemble(beta, sigma, np.zeros(len(tickers)), data)
    full_truth = assemble(beta, sigma, noise, data)
    panels = []
    for method in methods:
        part = frame[frame.method == method].set_index("asset").sort_index()
        bh = fitted[method]
        estimate_systematic = assemble(bh, factor_estimate, np.zeros(len(tickers)), data)
        estimate_full = assemble(bh, factor_estimate, part.residual_var.to_numpy(), data)
        oracle_components = assemble(bh, sigma, noise, data)
        def norm(a, b):
            """Return relative Frobenius error against the known covariance."""
            return float(np.linalg.norm(a - b) / np.linalg.norm(b))
        record = dict(profile=profile, n=n, seed=seed, method=method,
                      credit_mse=float(part.loc[credit_mask, "credit_mse"].mean()),
                      credit_bias=float(part.loc[credit_mask, "credit_bias"].mean()),
                      zero_credit_mse=float(part.loc[part.true_credit.abs() < 1e-14,
                                                     "credit_mse"].mean()),
                      beta_mse=float(part.beta_mse.mean()),
                      scenario_mse=float(part.scenario_mse.mean()),
                      prediction_nmse=float(part.prediction_nmse.mean()),
                      credit_prediction_nmse=float(
                          part.loc[credit_mask, "prediction_nmse"].mean()),
                      systematic_covar_error=norm(estimate_systematic, systematic_truth),
                      full_covar_error=norm(estimate_full, full_truth),
                      oracle_components_covar_error=norm(oracle_components, full_truth),
                      ols_endpoint_rate=float((part.reg_lambda == 0).mean()),
                      upper_endpoint_rate=float((part.lambda_index == 0).mean()))
        panels.append(record)
    return assets, panels


def paired_summary(frame):
    """Aggregate across independent replication seeds, never across ETFs."""
    result = []
    for (profile, n), group in frame.groupby(["profile", "n"]):
        base = group[group.method == "M1_zero"].set_index("seed").sort_index()
        for method, part in group.groupby("method"):
            part = part.set_index("seed").sort_index()
            assert part.index.equals(base.index)
            row = dict(profile=profile, n=n, method=method, seeds=len(part))
            for metric in METRICS:
                delta = (part[metric] - base[metric]).to_numpy()
                se = float(np.std(delta, ddof=1) / np.sqrt(len(delta)))
                row.update({metric: float(part[metric].mean()),
                            metric + "_delta": float(delta.mean()),
                            metric + "_se": se, metric + "_low": float(delta.mean() - 1.96 * se),
                            metric + "_high": float(delta.mean() + 1.96 * se)})
            result.append(row)
    return pd.DataFrame(result)


def check_assets(frame, data, config, phase):
    """Require complete cohorts and verify errors against saved native coefficients."""
    seed_range = config[phase + "_seeds"]
    seeds = set(range(seed_range[0], seed_range[1] + 1))
    expected = (len(seeds) * len(config["profiles"]) * len(config["sizes"])
                * len(config["methods"]) * len(data["tickers"]))
    assert len(frame) == expected
    keys = ["profile", "n", "seed", "asset", "method"]
    assert not frame.duplicated(keys).any()
    assert set(frame.seed) == seeds and set(frame.profile) == set(config["profiles"])
    assert set(frame.n) == set(config["sizes"])
    counts = frame.groupby(["profile", "n", "seed", "method"]).asset.nunique()
    assert (counts == len(data["tickers"])).all()
    assert set(frame.method) == set(config["methods"])
    errors = (frame[[f"beta{j}" for j in range(9)]].to_numpy()
              - frame[[f"truth{j}" for j in range(9)]].to_numpy())
    np.testing.assert_allclose(frame.beta_mse, np.mean(errors**2, axis=1), atol=1e-12)
    ci = list(data["factors"]).index("Credit")
    np.testing.assert_allclose(frame.credit_mse, errors[:, ci]**2, atol=1e-12)
    np.testing.assert_allclose(frame.scenario_mse,
                               np.mean(errors**2 * np.diag(data["sigma_annual"] / 12), axis=1)
                               / frame.response_var, atol=1e-12)
    assert np.isfinite(frame[["credit_mse", "prediction_nmse", "residual_var"]]).all().all()
    assert frame.loc[frame.method == "M3_oracle", "oracle_target"].all()
    assert not frame.loc[frame.method != "M3_oracle", "oracle_target"].any()


def verification(out, data, config, phase):
    """Audit pairing, recalculated errors and intentional corruptions."""
    from papers.prior_targets_2026.replication.robustness import reject_assertion
    assets = pd.read_csv(out / "asset_results.csv")
    panels = pd.read_csv(out / "panel_results.csv")
    check_assets(assets, data, config, phase)
    expected_panels = len(assets) // len(data["tickers"])
    assert len(panels) == expected_panels
    assert not panels.duplicated(["profile", "n", "seed", "method"]).any()
    expected = paired_summary(panels).sort_values(["profile", "n", "method"])
    saved = pd.read_csv(out / "paired_summary.csv").sort_values(["profile", "n", "method"])
    numeric = expected.select_dtypes(include="number").columns
    np.testing.assert_allclose(expected[numeric], saved[numeric], atol=1e-12)
    aggregated = assets[assets.is_credit].groupby(
        ["profile", "n", "seed", "method"]).credit_mse.mean()
    panel_values = panels.set_index(["profile", "n", "seed", "method"]).loc[aggregated.index]
    np.testing.assert_allclose(aggregated, panel_values.credit_mse, atol=1e-12)
    corrupted = assets.copy()
    corrupted.loc[0, "credit_mse"] += 1.
    reject_assertion(lambda: check_assets(corrupted, data, config, phase))
    reject_assertion(lambda: check_assets(assets.iloc[1:], data, config, phase))
    max_reconstruction = 0.
    asset = int(data["credit_indices"][0])
    for profile in config["profiles"]:
        for n in config["sizes"]:
            seed = config[phase + "_seeds"][0]
            samples, disturbance, truth, _, _, _ = samples_for(data, profile, n, seed, config)
            (xt, yt), (xv, yv), _ = samples
            results = fit_asset(xt, yt[:, asset], xv, yv[:, asset], data["prior"][asset],
                                truth[asset], disturbance[asset], True,
                                list(data["factors"]), lambdas(config))
            subset = assets[(assets.profile == profile) & (assets.n == n)
                            & (assets.seed == seed) & (assets.asset == asset)].set_index("method")
            for method, fitted in results.items():
                assert subset.loc[method, "lambda_index"] == fitted["lambda_index"]
                recorded = subset.loc[method, [f"beta{j}" for j in range(9)]].to_numpy(float)
                max_reconstruction = max(max_reconstruction, np.max(abs(recorded - fitted["beta"])))
                np.testing.assert_allclose(recorded, fitted["beta"], atol=1e-10)
    assert read_json(out / "failures.json") == []
    return dict(asset_rows=len(assets), panel_method_rows=len(panels),
                panels=expected_panels // len(config["methods"]),
                max_reconstruction_error=float(max_reconstruction),
                corrupt_metric_rejected=True, missing_row_rejected=True,
                paired_seed_aggregation_verified=True)


def numerical_checks(data, config):
    """Compare independently selected multivariate fits with the package solver."""
    from factorlasso import solve_lasso_cvx_problem
    from papers.prior_targets_2026.replication.estimators import prepare, prediction_losses
    errors, direct_scoring_error = [], 0.
    asset = int(data["credit_indices"][0])
    for profile in config["profiles"]:
        for n in config["sizes"]:
            samples, disturbance, truth, _, _, _ = samples_for(
                data, profile, n, config["pilot_seeds"][0], config)
            (xt, yt), (xv, yv), (xe, ye) = samples
            args = (xt, yt[:, asset], xv, yv[:, asset], data["prior"][asset], truth[asset],
                    disturbance[asset], True, list(data["factors"]), lambdas(config))
            selected = fit_asset(*args)
            z, v, _, _, xs, ys = prepare(xt, yt[:, asset])
            for method in ["M1_zero", "M2_auto", "M3_economic", "M4_free_winner"]:
                result = selected[method]
                penalty = np.ones((1, 9))
                if method == "M4_free_winner":
                    penalty[0, result["winner"]] = 0.
                ref = solve_lasso_cvx_problem(
                    z, v[:, None], reg_lambda=float(result["reg_lambda"]),
                    factors_beta_prior=(result["target"] * xs / ys)[None],
                    penalty_weights=penalty, solver="CLARABEL",
                ).estimated_beta[0]
                errors.append(float(np.max(abs(ref - result["beta"] * xs / ys))))
                direct = np.mean((ye[:, asset] - xe @ result["beta"] - result["intercept"])**2)
                reduced = prediction_losses(xe, ye[:, asset], result["beta"][None],
                                            np.array([result["intercept"]]))[0]
                direct_scoring_error = max(direct_scoring_error, abs(direct - reduced))
            # Test is absent from fit_asset; mutating it leaves all fitting inputs untouched.
            samples[2] = (xe + 999, ye - 999)
            after = fit_asset(*args)
            for method in selected:
                np.testing.assert_array_equal(selected[method]["beta"], after[method]["beta"])
                assert selected[method]["lambda_index"] == after[method]["lambda_index"]
    assert max(errors) < 3e-4
    return dict(standardised_package_max_beta_error=max(errors),
                package_comparisons=len(errors), package_tolerance=3e-4,
                direct_scoring_max_error=float(direct_scoring_error),
                future_perturbation_invariant=True)


def plot(out):
    """Render the complete set of loading-profile comparisons at n=112."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = pd.read_csv(out / "paired_summary.csv")
    methods = ["M2_auto", "M3_economic", "M3_economic_noisy", "M3_small_model",
               "M4_free_winner", "M5_post_lasso"]
    labels = ["Automatic", "Economic", "Economic + noise", "Small model",
              "Free winner", "Post-LASSO"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5), constrained_layout=True)
    for ax, profile in zip(axes, ["baseline", "credit_removed", "rates_dominant"]):
        part = summary[(summary.profile == profile) & (summary.n == 112)]
        part = part.set_index("method").loc[methods]
        ax.errorbar(part.credit_mse_delta, range(len(methods)), xerr=1.96 * part.credit_mse_se,
                    fmt="o", color="#0072B2", capsize=3)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set(yticks=range(len(methods)), yticklabels=labels, title=profile.replace("_", " "),
               xlabel="Credit-loading MSE difference vs tuned zero")
        ax.invert_yaxis()
    fig.suptitle("102-asset calibrated transfer: 17-fund credit cohort, n=112\n"
                 "Pointwise paired Monte Carlo intervals across independent panel seeds")
    fig.savefig(out / "multiasset_credit.png", dpi=160)
    plt.close(fig)


def parallel_check(data, config, workers):
    """Prove process execution preserves the serial estimator and metric values."""
    arguments = (data, config["profiles"][0], config["sizes"][0],
                 config["pilot_seeds"][0], config)
    serial = panel_rows(*arguments)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parallel = pool.submit(panel_rows, *arguments).result()
    for first, second in zip(serial, parallel):
        pd.testing.assert_frame_equal(pd.DataFrame(first), pd.DataFrame(second),
                                      check_exact=True)
    return {"parallel_serial_exact_identity": True, "workers": workers}



def main():
    """Execute or verify one immutable pilot/confirmation batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "verify", "check"])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", choices=["pilot", "confirmation"], default="pilot")
    parser.add_argument("--workers", type=int, choices=[1, 2, 3, 4], default=1)
    args = parser.parse_args()
    root = validate_root(args.output_root)
    check_snapshot(root)
    sys.path.insert(0, str(root / "source_snapshot/src"))
    check_manifest(root / "E2_robustness_v1")
    data = calibration(root)
    config = read_json(HERE / "multiasset_protocol.json")
    out = root / ("E3_L1_" + args.phase + "_v1")
    if args.command == "check":
        print(json.dumps(numerical_checks(data, config), indent=2))
        return
    if args.command == "verify":
        check_manifest(out)
        config = read_json(out / "archive/multiasset_protocol.json")
        print(json.dumps(verification(out, data, config, args.phase), indent=2))
        return
    if args.phase == "confirmation":
        pilot = root / "E3_L1_pilot_v1"
        check_manifest(pilot)
        assert read_json(pilot / "summary.json")["failures"] == 0
        assert digest(pilot / "archive/multiasset_protocol.json") == digest(
            HERE / "multiasset_protocol.json")
    out.mkdir()
    archive = out / "archive"
    archive.mkdir()
    for name in ["multiasset.py", "multiasset_protocol.json", "estimators.py",
                 "run.py", "robustness.py"]:
        shutil.copy2(HERE / name, archive / name)
    checks = numerical_checks(data, config)
    checks.update(parallel_check(data, config, args.workers))
    save_json(out / "numerical_checks.json", checks)
    assets, panels, failures = [], [], []
    start = time.perf_counter()
    seeds = config[args.phase + "_seeds"]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for profile in config["profiles"]:
            for n in config["sizes"]:
                jobs = {
                    seed: pool.submit(panel_rows, data, profile, n, seed, config)
                    for seed in range(seeds[0], seeds[1] + 1)
                }
                for seed, job in jobs.items():
                    try:
                        a, p = job.result()
                        assets.extend(a)
                        panels.extend(p)
                    except Exception as error:
                        failures.append(dict(profile=profile, n=n, seed=seed, error=repr(error)))
                print(f"{args.phase}: {profile}, n={n} complete", flush=True)
    pd.DataFrame(assets).to_csv(out / "asset_results.csv", index=False)
    panel_frame = pd.DataFrame(panels)
    panel_frame.to_csv(out / "panel_results.csv", index=False)
    paired_summary(panel_frame).to_csv(out / "paired_summary.csv", index=False)
    save_json(out / "failures.json", failures)
    receipt = verification(out, data, config, args.phase)
    save_json(out / "verification.json", receipt)
    plot(out)
    save_json(out / "summary.json", {
        **receipt, "phase": args.phase, "seeds": seeds, "failures": len(failures),
        "seconds": time.perf_counter() - start, "workers": args.workers,
        "protocol_sha256": digest(HERE / "multiasset_protocol.json"),
        "prerequisite_manifest_sha256": digest(root / "E2_robustness_v1/manifest.json"),
        "calibration_manifest_sha256": digest(root / "E3_calibration_audit_v1/manifest.json"),
        "pending": config["completion_boundary"],
    })
    save_json(out / "manifest.json", {
        str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()
    })
    print(json.dumps(read_json(out / "summary.json"), indent=2))


if __name__ == "__main__":
    main()

