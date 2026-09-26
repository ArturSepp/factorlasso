"""Frozen E2 stress experiments; no production defaults or historical results change.

The weighted adapter uses a global weighted intercept, not LassoModel's rolling
mean preprocessing. Its objective is checked against the exported low-level
factorlasso solver with the corresponding loss/penalty normalisation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
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

METRICS = ["secondary_mse", "beta_mse", "scenario_mse", "prediction_mse",
           "false_exposure_mse"]
BASE_METHODS = [
    "M0_ols", "M1_zero", "M2_auto", "M4_free_winner", "M5_post_lasso",
    "M3_fixed", "M3_same_ols", "M3_previous_ols", "M6_pooled_zero",
    "M3_noisy_std025", "M3_oracle",
]


def weights(n, span):
    """Return normalised squared-loss weights, newest observation last."""
    raw = np.ones(n) if span is None else (1 - 2 / (span + 1)) ** np.arange(n)[::-1]
    return raw / raw.sum()


def prepare_weighted(x, y, span):
    """Centre and standardise with training weights and return row-scaled data."""
    w = weights(len(y), span)
    xm, ym = w @ x, float(w @ y)
    xc, yc = x - xm, y - ym
    xs, ys = np.sqrt(w @ (xc * xc)), np.sqrt(w @ (yc * yc))
    if np.any(xs <= 0) or ys <= 0:
        raise ValueError("Degenerate weighted training sample")
    z, v = xc / xs, yc / ys
    row_scale = np.sqrt(len(y) * w)
    return z * row_scale[:, None], v * row_scale, xm, ym, xs, ys, w


def fit_weighted(x, y, grid, span, targets):
    """Solve matched L1 paths with a weighted intercept and raw-unit targets."""
    from factorlasso.beta_priors import _compute_ols_prior
    from papers.prior_targets_2026.replication.estimators import path_standardised
    z, v, xm, ym, xs, ys, w = prepare_weighted(x, y, span)
    _, r2, prior = _compute_ols_prior(x, y[:, None], span=span)
    winner = int(np.nanargmax(r2[0]))
    centres = {"M1_zero": np.zeros(x.shape[1]), "M2_auto": prior[0], **targets}
    paths = {}
    for method, target in centres.items():
        b, gap = path_standardised(z, v, grid, prior=target * xs / ys)
        paths[method] = dict(beta=b * ys / xs, target=target, gap=gap)
    b, gap = path_standardised(z, v, grid, unpenalised=winner)
    paths["M4_free_winner"] = dict(beta=b * ys / xs, target=np.zeros(x.shape[1]), gap=gap)
    ols = np.linalg.lstsq(z, v, rcond=None)[0]
    paths["M0_ols"] = dict(beta=np.tile(ols * ys / xs, (len(grid), 1)),
                           target=np.zeros(x.shape[1]), gap=0.)
    b = np.zeros_like(paths["M1_zero"]["beta"])
    for k, row in enumerate(paths["M1_zero"]["beta"] * xs / ys):
        support = np.flatnonzero(np.abs(row) > 1e-7)
        if len(support):
            b[k, support] = np.linalg.lstsq(z[:, support], v, rcond=None)[0]
    paths["M5_post_lasso"] = dict(beta=b * ys / xs, target=np.zeros(x.shape[1]), gap=0.)
    for path in paths.values():
        path.update(intercept=ym - path["beta"] @ xm, winner=winner,
                    winner_r2=float(r2[0, winner]), kish_ess=float(1 / (w @ w)))
    return paths


def ols_target(x, y, span=None):
    """Estimate a target only from the explicitly supplied information block."""
    z, v, _, _, xs, ys, _ = prepare_weighted(x, y, span)
    return np.linalg.lstsq(z, v, rcond=None)[0] * ys / xs


def cell_parameters(cell):
    """Resolve known DGP matrices and future/pre-break response variances."""
    beta = np.array(cell["beta"], float)
    before = np.array(cell.get("beta_before", beta), float)
    sigma = np.array(cell.get("sigma", [[1., cell.get("rho", 0.)],
                                       [cell.get("rho", 0.), 1.]]), float)
    signal_before = float(before @ sigma @ before)
    noise = signal_before if signal_before > 0 else 1.
    variance = float(beta @ sigma @ beta + noise)
    return beta, before, sigma, noise, variance


def draw_samples(cell, cell_id, seed, config):
    """Use separate streams; span variants share draws within each regime."""
    beta, before, sigma, noise, variance = cell_parameters(cell)
    stream_id = cell_id if cell["family"] != "temporal" else (
        1000 + int(cell.get("break_index") is not None))
    chol = np.linalg.cholesky(sigma)
    np.testing.assert_allclose(chol @ chol.T, sigma, atol=1e-14)
    n = cell["n"]
    samples = []
    for stream, size in enumerate([n, cell.get("validation_n", n), config["test_size"], n]):
        rng = np.random.default_rng(np.random.SeedSequence([seed, stream_id, stream]))
        x = rng.normal(size=(size, len(beta))) @ chol.T
        error = (rng.standard_t(3, size) / np.sqrt(3)
                 if cell.get("residual") == "t3" else rng.normal(size=size))
        local_beta = before if stream == 3 else beta
        signal = x @ local_beta
        if stream == 0 and cell.get("break_index") is not None:
            split = cell["break_index"]
            signal[:split] = x[:split] @ before
        samples.append((x, signal + np.sqrt(noise) * error))
    target_rng = np.random.default_rng(np.random.SeedSequence([seed, stream_id, 4]))
    disturbance = target_rng.normal(size=len(beta)) * np.sqrt(variance)
    return samples, disturbance, stream_id


def targets_for(cell, samples, disturbance, config):
    """Disclose fixed, same-sample, previous-sample and truth-derived centres."""
    beta = np.array(cell["beta"], float)
    fixed = np.zeros(len(beta))
    fixed[:2] = [1., .3]
    targets = {
        "M3_fixed": fixed,
        "M3_same_ols": ols_target(*samples[0], span=cell["span"]),
        "M3_previous_ols": ols_target(*samples[3]),
        "M3_noisy_std025": beta + .25 * disturbance,
        "M3_oracle": beta.copy(),
    }
    if cell["family"] == "quality":
        for scale in config["quality_scales"]:
            for sd in config["quality_noise_sd"]:
                centre = beta.copy()
                centre[1] *= scale
                targets[f"Q_scale{scale:g}_sd{sd:g}"] = centre + sd * disturbance
        centre = beta.copy()
        centre[1] *= -1
        targets["Q_wrong_sign"] = centre
        targets["Q_wrong_factor"] = beta[::-1].copy()
    return targets


def method_names(cell, config):
    """Return the exact expected roster before estimation."""
    names = BASE_METHODS.copy()
    if cell["family"] == "quality":
        names += [f"Q_scale{s:g}_sd{d:g}" for s in config["quality_scales"]
                  for d in config["quality_noise_sd"]]
        names += ["Q_wrong_sign", "Q_wrong_factor"]
    return sorted(names)


def estimate(cell, samples, disturbance, config):
    """Fit only training/history data; never accept a test-based tuning target."""
    grid = lambdas(config)
    targets = targets_for(cell, samples, disturbance, config)
    paths = fit_weighted(*samples[0], grid, cell["span"], targets)
    pooled_x = np.vstack([samples[3][0], samples[0][0]])
    pooled_y = np.r_[samples[3][1], samples[0][1]]
    pooled = fit_weighted(pooled_x, pooled_y, grid, None, {})
    paths["M6_pooled_zero"] = pooled["M1_zero"]
    return paths


def selected_index(path, validation, grid, method):
    """Choose a path index from validation only, with OLS explicitly at zero."""
    from papers.prior_targets_2026.replication.estimators import prediction_losses, select_index
    losses = prediction_losses(*validation, path["beta"], path["intercept"])
    index = len(grid) - 1 if method == "M0_ols" else select_index(losses)
    return index, losses


def rows_for(cell, cell_id, seed, config):
    """Score all methods on identical independent test observations."""
    from papers.prior_targets_2026.replication.estimators import prediction_losses
    beta, _, sigma, _, variance = cell_parameters(cell)
    samples, disturbance, stream_id = draw_samples(cell, cell_id, seed, config)
    paths = estimate(cell, samples, disturbance, config)
    grid = lambdas(config)
    p = len(beta)
    positive_scenarios = np.vstack([np.eye(p), np.ones(p)])
    scenarios = np.vstack([positive_scenarios, -positive_scenarios])
    result = []
    for method, path in paths.items():
        k, losses = selected_index(path, samples[1], grid, method)
        fitted, target = path["beta"][k], path["target"]
        error = fitted - beta
        null_indices = np.flatnonzero(np.abs(beta) <= 1e-14)
        predicted = samples[2][0] @ fitted + path["intercept"][k]
        direct_loss = float(np.mean((samples[2][1] - predicted) ** 2))
        reduced = prediction_losses(*samples[2], fitted[None],
                                    np.array([path["intercept"][k]]))[0]
        np.testing.assert_allclose(direct_loss, reduced, rtol=1e-11, atol=1e-12)
        row = dict(cell=cell["name"], cell_id=cell_id, family=cell["family"],
                   seed=seed, stream_id=stream_id, method=method, n=cell["n"], p=p,
                   span=cell["span"] or 0, rho=sigma[0, 1], response_var=variance,
                   lambda_index=k, reg_lambda=grid[k], validation_mse=losses[k],
                   prediction_mse=direct_loss, beta_mse=float(np.mean(error**2)),
                   secondary_mse=error[1]**2,
                   scenario_mse=float(np.mean((scenarios @ error)**2) / variance),
                   false_exposure_mse=float(np.mean(fitted[null_indices]**2))
                   if len(null_indices) else 0.,
                   null_factor_count=len(null_indices), secondary_zero=abs(fitted[1]) <= 1e-6,
                   winner=path["winner"], winner_r2=path["winner_r2"],
                   winner_zero_truth=abs(beta[path["winner"]]) <= 1e-14,
                   kish_ess=path["kish_ess"], dual_gap=path["gap"],
                   intercept=path["intercept"][k],
                   signal_prediction_mse_population=float(error @ sigma @ error),
                   truth_derived_target=method.startswith("Q_")
                   or method in ["M3_noisy_std025", "M3_oracle"],
                   exact_oracle_spec=method in ["M3_oracle", "Q_scale1_sd0"])
        for j in range(3):
            row.update({f"beta{j}": fitted[j] if j < p else 0.,
                        f"target{j}": target[j] if j < p else 0.,
                        f"truth{j}": beta[j] if j < p else 0.})
        result.append(row)
    return result


def summarise(frame):
    """Compute paired seed effects with pointwise Monte Carlo intervals."""
    result = []
    for cell, group in frame.groupby("cell"):
        base = group[group.method == "M1_zero"].set_index("seed").sort_index()
        for method, part in group.groupby("method"):
            part = part.set_index("seed").sort_index()
            assert part.index.equals(base.index)
            row = dict(cell=cell, method=method, seeds=len(part),
                       secondary_zero_rate=float(part.secondary_zero.mean()),
                       wrong_winner_rate=float(part.winner_zero_truth.mean()),
                       mean_kish_ess=float(part.kish_ess.mean()),
                       ols_endpoint_rate=float((part.reg_lambda == 0).mean()),
                       upper_endpoint_rate=float((part.lambda_index == 0).mean()))
            for metric in METRICS:
                diff = part[metric] - base[metric]
                se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
                row.update({metric: float(part[metric].mean()), metric + "_delta": diff.mean(),
                            metric + "_se": se, metric + "_low": diff.mean() - 1.96 * se,
                            metric + "_high": diff.mean() + 1.96 * se})
            result.append(row)
    return pd.DataFrame(result)


def validate_frame(frame, config):
    """Check complete paired rosters, known truth and independent seed identities."""
    assert not frame.empty and not frame.duplicated(["cell", "seed", "method"]).any()
    assert set(frame.cell) == {c["name"] for c in config["cells"]}
    expected_seeds = set(range(config["seeds"][0], config["seeds"][1] + 1))
    for cell in config["cells"]:
        part = frame[frame.cell == cell["name"]]
        assert set(part.seed) == expected_seeds
        roster = method_names(cell, config)
        assert len(part) == len(roster) * len(expected_seeds)
        assert all(sorted(g.method) == roster for _, g in part.groupby("seed"))
        truth = np.array(cell["beta"])
        fitted = part[[f"beta{j}" for j in range(len(truth))]].to_numpy()
        errors = fitted - truth
        np.testing.assert_allclose(part.secondary_mse, errors[:, 1]**2, atol=1e-12)
        np.testing.assert_allclose(part.beta_mse, np.mean(errors**2, axis=1), atol=1e-12)
        # Independent closed-form sum over isolated and joint signed shocks.
        reference_scenario = (np.sum(errors**2, axis=1) + errors.sum(axis=1)**2)
        reference_scenario /= (len(truth) + 1) * part.response_var.to_numpy()
        np.testing.assert_allclose(part.scenario_mse, reference_scenario, atol=1e-12)
        for j, true_value in enumerate(truth):
            np.testing.assert_allclose(part[f"truth{j}"], true_value)
    assert np.isfinite(frame[METRICS + ["beta0", "beta1", "beta2"]]).all().all()
    assert frame.loc[frame.method == "M3_oracle", "exact_oracle_spec"].all()


def reject_assertion(action):
    """Return success only when an intentionally defective result is rejected."""
    try:
        action()
    except AssertionError:
        return True
    raise AssertionError("The verification accepted an intentional defect")


def numerical_checks(config):
    """Use independent weighted solves and deliberate defects before simulation."""
    from factorlasso import solve_lasso_cvx_problem
    from factorlasso.beta_priors import _compute_ols_prior
    from factorlasso.lasso_estimator import _compute_solver_weights
    from papers.prior_targets_2026.replication.estimators import fit_paths
    rng = np.random.default_rng(config["verification"]["reference_seed"])
    x = rng.normal(size=(240, 3))
    x[:, 1] = .95 * x[:, 0] + np.sqrt(1 - .95**2) * x[:, 1]
    y = x @ [1., .3, -.1] + rng.normal(size=240)
    grid = np.array([.4, .03, .003, 0.])
    max_error, weighted_ols_error = 0., 0.
    weight_defect_error = 0.
    for span in [None, 36, 60]:
        paths = fit_weighted(x, y, grid, span, {"external": np.array([1., .1, 0.])})
        z, v, xm, ym, xs, ys, w = prepare_weighted(x, y, span)
        raw_z, raw_v = (x - xm) / xs, (y - ym) / ys
        package_weights = _compute_solver_weights(len(y), 1, span, np.ones(len(y)))
        np.testing.assert_allclose(w, package_weights**2 / np.sum(package_weights**2),
                                   atol=1e-14)
        slopes, r2, _ = _compute_ols_prior(x, y[:, None], span=span)
        for j in range(x.shape[1]):
            design = np.column_stack([np.ones(len(y)), x[:, j]])
            coef = np.linalg.lstsq(design * np.sqrt(w[:, None]), y * np.sqrt(w),
                                  rcond=None)[0]
            weighted_ols_error = max(weighted_ols_error, abs(slopes[0, j] - coef[1]))
            sse = np.sum(w * (y - design @ coef)**2)
            np.testing.assert_allclose(r2[0, j], 1 - sse / np.sum(w * (y - w @ y)**2),
                                       atol=1e-12)
        for method in ["M1_zero", "M2_auto", "M4_free_winner", "external"]:
            path = paths[method]
            penalty = np.ones((1, 3))
            if method == "M4_free_winner":
                penalty[0, path["winner"]] = 0.
            for k, lam in enumerate(grid):
                ref = solve_lasso_cvx_problem(
                    raw_z, raw_v[:, None], span=span,
                    reg_lambda=float(lam * np.mean(package_weights**2)),
                    factors_beta_prior=(path["target"] * xs / ys)[None],
                    penalty_weights=penalty, solver="CLARABEL",
                ).estimated_beta[0] * ys / xs
                max_error = max(max_error, float(np.max(np.abs(ref - path["beta"][k]))))
        if span is None:
            old = fit_paths(x, y, grid)
            for method in old:
                np.testing.assert_allclose(old[method]["beta"], paths[method]["beta"],
                                           atol=1e-9)
        else:
            bad_w = w[::-1]
            bad_design = np.column_stack([np.ones(len(y)), x])
            bad = np.linalg.lstsq(bad_design * np.sqrt(bad_w[:, None]),
                                 y * np.sqrt(bad_w), rcond=None)[0][1:]
            good = paths["M0_ols"]["beta"][0]
            weight_defect_error = max(weight_defect_error, np.max(np.abs(good - bad)))
            reject_assertion(lambda: np.testing.assert_allclose(good, bad, atol=1e-9))
        same = fit_weighted(x, y, grid, span, {"same": ols_target(x, y, span)})
        np.testing.assert_allclose(same["same"]["beta"],
                                   same["M0_ols"]["beta"], atol=1e-9)
    assert max_error < config["verification"]["package_beta_tolerance"]
    bad = fit_weighted(x, y, grid * 2, None, {})["M1_zero"]["beta"]
    good = fit_weighted(x, y, grid, None, {})["M1_zero"]["beta"]
    reject_assertion(lambda: np.testing.assert_allclose(good, bad, atol=1e-9))
    # Exact duplicated-factor tie: the label assignment changes under column exchange.
    twins = np.column_stack([x[:, 0], x[:, 0]])
    tied = fit_weighted(twins, y, np.array([.5]), None, {})
    assert tied["M2_auto"]["winner"] == 0
    reversed_tie = fit_weighted(twins[:, ::-1], y, np.array([.5]), None, {})
    b1, b2 = tied["M2_auto"]["beta"][0], reversed_tie["M2_auto"]["beta"][0][::-1]
    np.testing.assert_allclose(twins @ b1, twins @ b2, atol=1e-9)
    assert np.max(np.abs(b1 - b2)) > .1
    # Future perturbations cannot change training paths or validation selection.
    cell = config["cells"][0]
    samples, disturbance, _ = draw_samples(cell, 0, config["seeds"][0], config)
    first = estimate(cell, samples, disturbance, config)
    altered = deepcopy(samples)
    altered[2] = (samples[2][0] * 100, samples[2][1] - 1000)
    second = estimate(cell, altered, disturbance, config)
    for method in first:
        np.testing.assert_array_equal(first[method]["beta"], second[method]["beta"])
        k1, _ = selected_index(first[method], samples[1], lambdas(config), method)
        k2, _ = selected_index(second[method], altered[1], lambdas(config), method)
        assert k1 == k2
    fabricated_validation, fabricated_test = np.array([0., 10.]), np.array([10., 0.])
    reject_assertion(lambda: np.testing.assert_equal(
        np.argmin(fabricated_validation), np.argmin(fabricated_test)))
    return dict(package_max_abs_beta_error=max_error,
                weighted_ols_max_error=float(weighted_ols_error),
                reversed_weight_error=float(weight_defect_error),
                reversed_weights_rejected=True, double_lambda_rejected=True,
                future_perturbation_invariance=True, leaked_selection_rejected=True,
                same_sample_ols_target_equals_ols=True,
                tie_prediction_invariant=True, tie_beta_label_gap=float(np.max(abs(b1 - b2))))


def verify_results(destination, config, reconstruct=False):
    """Audit numerical outputs and independently recompute seed intervals."""
    frame = pd.read_csv(destination / "selected_results.csv")
    validate_frame(frame, config)
    expected = summarise(frame).sort_values(["cell", "method"]).reset_index(drop=True)
    actual = pd.read_csv(destination / "paired_summary.csv")
    actual = actual.sort_values(["cell", "method"]).reset_index(drop=True)
    numeric = expected.select_dtypes(include="number").columns
    np.testing.assert_allclose(expected[numeric], actual[numeric], atol=1e-12, rtol=1e-10)
    # Different aggregation route: pivot paired effects then use dot products for SE.
    for metric in METRICS:
        wide = frame.pivot(index=["cell", "seed"], columns="method", values=metric)
        for (cell, method), record in actual.set_index(["cell", "method"]).iterrows():
            differences = (wide.loc[cell, method] - wide.loc[cell, "M1_zero"]).dropna()
            d = differences.to_numpy()
            mean = np.sum(d) / len(d)
            se = np.sqrt(np.dot(d - mean, d - mean) / (len(d) * (len(d) - 1)))
            np.testing.assert_allclose([record[metric + "_delta"], record[metric + "_se"]],
                                       [mean, se], atol=1e-12)
    corrupted = frame.copy()
    corrupted.loc[0, "secondary_mse"] += 1.
    reject_assertion(lambda: validate_frame(corrupted, config))
    reject_assertion(lambda: validate_frame(frame.iloc[1:], config))
    selected_error = 0.
    if reconstruct:
        for i, cell in enumerate(config["cells"]):
            seed = config["seeds"][0]
            again = pd.DataFrame(rows_for(cell, i, seed, config)).set_index("method")
            saved = frame[(frame.cell == cell["name"]) & (frame.seed == seed)]
            saved = saved.set_index("method").loc[again.index]
            np.testing.assert_array_equal(saved.lambda_index, again.lambda_index)
            cols = ["beta0", "beta1", "beta2", "prediction_mse", "target0", "target1"]
            selected_error = max(selected_error, np.max(np.abs(saved[cols] - again[cols])))
            np.testing.assert_allclose(saved[cols], again[cols], atol=1e-10)
    assert read_json(destination / "failures.json") == []
    return dict(rows=len(frame), cells=frame.cell.nunique(), paired_intervals_verified=True,
                corrupt_secondary_error_rejected=True, missing_method_row_rejected=True,
                reconstructed_first_seed_every_cell=reconstruct,
                reconstructed_max_error=float(selected_error))


def plots(destination):
    """Render prespecified target-quality and stale-target exhibits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.read_csv(destination / "paired_summary.csv")
    colours = ["#0072B2", "#E69F00", "#009E73"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, metric, title in zip(
        axes, ["secondary_mse", "prediction_mse"],
        ["Secondary-loading error", "Prediction error"]):
        part = frame[frame.cell == "quality_r0.95_b0.3_n60"].set_index("method")
        for sd, colour in zip([0, .25, .5], colours):
            methods = [f"Q_scale{s:g}_sd{sd:g}" for s in [0, .5, 1, 1.5, 2]]
            y = part.loc[methods, metric + "_delta"].to_numpy()
            se = part.loc[methods, metric + "_se"].to_numpy()
            ax.errorbar([0, .5, 1, 1.5, 2], y, yerr=1.96 * se,
                        label=f"Target noise SD {sd:g}", color=colour, marker="o", capsize=3)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set(xlabel="Multiplier on true secondary target",
               ylabel="Difference vs tuned zero target",
               title=title)
        ax.legend(fontsize=8)
    fig.suptitle("Target sensitivity: correlation 0.95, secondary 0.3, n=60\n"
                 "200 paired seeds; pointwise 95% Monte Carlo intervals; truth-derived targets")
    fig.savefig(destination / "target_sensitivity.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    methods = ["M1_zero", "M2_auto", "M3_previous_ols", "M3_fixed", "M6_pooled_zero"]
    for ax, regime in zip(axes, ["stationary", "break"]):
        for method in methods:
            cells = [f"{regime}_span{s}" for s in ["uniform", "36", "60"]]
            part = frame[frame.method == method].set_index("cell").loc[cells]
            ax.plot(range(3), part.secondary_mse, marker="o", label=method)
        ax.set(xticks=range(3), xticklabels=["Uniform", "EWMA 36", "EWMA 60"],
               ylabel="Secondary-loading MSE", title=regime.capitalize())
        ax.legend(fontsize=8)
    fig.suptitle("Previous-sample targets: stationarity versus one loading break\n"
                 "240 training observations; 60 post-training validation; 200 paired seeds")
    fig.savefig(destination / "stale_targets.png", dpi=160)
    plt.close(fig)


def main():
    """Freeze code, execute a bounded batch, or verify an immutable completed run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "verify", "check"])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--name", default="E2_robustness_v1")
    args = parser.parse_args()
    root = validate_root(args.output_root)
    check_snapshot(root)
    sys.path.insert(0, str(root / "source_snapshot/src"))
    destination = validate_root(root / args.name)
    config_path = HERE / "robustness_protocol.json"
    config = read_json(config_path)
    if args.command == "check":
        print(json.dumps(numerical_checks(config), indent=2))
        return
    if args.command == "verify":
        manifest = read_json(destination / "manifest.json")
        for name, expected in manifest.items():
            assert digest(destination / name) == expected, name
        config = read_json(destination / "archive/robustness_protocol.json")
        print(json.dumps(verify_results(destination, config, reconstruct=True), indent=2))
        return
    destination.mkdir()
    archive = destination / "archive"
    archive.mkdir()
    for name in ["robustness.py", "robustness_protocol.json", "estimators.py", "run.py"]:
        shutil.copy2(HERE / name, archive / name)
    checks = numerical_checks(config)
    save_json(destination / "numerical_checks.json", checks)
    start = time.perf_counter()
    records, failures = [], []
    for i, cell in enumerate(config["cells"]):
        for seed in range(config["seeds"][0], config["seeds"][1] + 1):
            try:
                records.extend(rows_for(cell, i, seed, config))
            except Exception as error:
                failures.append(dict(cell=cell["name"], seed=seed, error=repr(error)))
        print(f"{i + 1}/{len(config['cells'])}: {cell['name']} complete", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(destination / "selected_results.csv", index=False)
    summarise(frame).to_csv(destination / "paired_summary.csv", index=False)
    save_json(destination / "failures.json", failures)
    receipt = verify_results(destination, config, reconstruct=True)
    save_json(destination / "verification.json", receipt)
    plots(destination)
    save_json(destination / "summary.json", {
        **receipt, "seconds": time.perf_counter() - start, "seeds": config["seeds"],
        "datasets": len(config["cells"]) * (config["seeds"][1] - config["seeds"][0] + 1),
        "failures": len(failures), "protocol_sha256": digest(config_path),
        "base_snapshot_hashes_sha256": digest(root / "snapshot_hashes.json"),
        "information_sets": config["target_noise_units"], "weighting": config["weights"],
    })
    manifest = {str(p.relative_to(destination)): digest(p) for p in destination.rglob("*")
                if p.is_file()}
    save_json(destination / "manifest.json", manifest)
    print(json.dumps(read_json(destination / "summary.json"), indent=2))


if __name__ == "__main__":
    main()

