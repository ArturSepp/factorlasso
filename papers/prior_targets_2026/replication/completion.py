"""Independent artifact review and verifiable evidence decision; no estimator changes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

from papers.prior_targets_2026.replication.run import (
    HERE, check_snapshot, digest, read_json, save_json, validate_root,
)
from papers.prior_targets_2026.replication.multiasset import (
    calibration, check_manifest, generating_parameters,
)


def rolling_review(root, out):
    """Reconstruct every prediction, scenario transition and paired empirical effect."""
    from papers.prior_targets_2026.replication.rolling import lane_data, stationary_indices
    source = root / "E4_rolling_v1"
    check_manifest(source)
    config = read_json(source / "archive/rolling_protocol_v2.json")
    frame = pd.read_csv(source / "rolling_results.csv")
    maximum_prediction, maximum_scenario = 0., 0.
    for lane, part in frame.groupby("lane"):
        x, y, _ = lane_data(root, config, lane)
        factors = list(x.columns)
        beta = part[["beta_"+f for f in factors]].to_numpy()
        values = x.loc[pd.to_datetime(part.date)].to_numpy()
        prediction = np.einsum("ij,ij->i", values, beta) + part.intercept.to_numpy()
        maximum_prediction = max(maximum_prediction,
                                 float(np.max(abs(prediction-part.prediction))))
        np.testing.assert_allclose(prediction, part.prediction, atol=1e-12)
        observed = np.array([y.loc[date, asset] for date, asset in
                             zip(pd.to_datetime(part.date), part.asset)])
        np.testing.assert_allclose(observed, part.actual, atol=1e-14)
        sd = x.iloc[:60].to_numpy().std(0)
        for _, line in part.groupby(["span", "asset", "method"]):
            line = line.sort_values("date")
            b = line[["beta_"+f for f in factors]].to_numpy()
            changes = np.diff(b, axis=0)*sd
            # +/- duplicates cancel; p isolated shocks plus one joint shock.
            direct = (np.sum(changes**2, axis=1)+np.sum(changes, axis=1)**2)/(len(sd)+1)
            direct /= line.response_var.iloc[0]
            saved = line.scenario_change_nmse.iloc[1:].to_numpy()
            maximum_scenario = max(maximum_scenario, float(np.max(abs(direct-saved))))
            np.testing.assert_allclose(direct, saved, atol=1e-12)
            expected_switch = line.winner.iloc[1:].to_numpy() != line.winner.iloc[:-1].to_numpy()
            np.testing.assert_array_equal(expected_switch, line.winner_switch.iloc[1:])
    bootstrap = pd.read_csv(source / "paired_bootstrap.csv")
    review = []
    for (lane, span, cohort), part in frame.groupby(["lane", "span", "cohort"]):
        # Sum across assets then divide explicitly; distinct from groupby means.
        counts = part.groupby(["date", "method"]).size().unstack()
        totals = part.groupby(["date", "method"]).nmse.sum().unstack()
        monthly = totals/counts
        zero = monthly.M1_zero.mean()
        for block in config["uncertainty"]["mean_block_lengths"]:
            indices = stationary_indices(29, 5000, block, config["uncertainty"]["seed"]+block)
            frequencies = np.array([np.bincount(index, minlength=29) for index in indices])/29
            for method in monthly:
                d = (monthly[method]-monthly.M1_zero).to_numpy()
                low, high = np.quantile(frequencies @ d, [.025, .975])
                saved = bootstrap[(bootstrap.lane==lane) & (bootstrap.span==span)
                                  & (bootstrap.cohort==cohort) & (bootstrap.method==method)
                                  & (bootstrap.block==block)]
                assert len(saved)==1
                np.testing.assert_allclose(
                    saved[["paired_delta", "bootstrap_low", "bootstrap_high"]].to_numpy()[0],
                    [d.mean(), low, high], atol=1e-12)
                review.append(dict(lane=lane, span=span, cohort=cohort, method=method,
                                   block=block, mean_nmse=monthly[method].mean(),
                                   relative_change_pct=100*d.mean()/zero,
                                   paired_delta=d.mean(), low=low, high=high))
    pd.DataFrame(review).to_csv(out / "paired_effects.csv", index=False)
    per_asset = frame.groupby(["lane", "span", "asset", "method"]).agg(
        mean_nmse=("nmse", "mean"), credit_mean=("beta_Credit", "mean"),
        credit_min=("beta_Credit", "min"), credit_max=("beta_Credit", "max"),
        scenario_change=("scenario_change_nmse", "mean")).reset_index()
    per_asset.to_csv(out / "asset_descriptions.csv", index=False)
    endpoints = frame.assign(ols_endpoint=frame.reg_lambda.eq(0),
                             upper_endpoint=frame.lambda_index.eq(0)).groupby(
        ["lane", "span", "cohort", "method"])[["ols_endpoint", "upper_endpoint"]].mean()
    endpoints.to_csv(out / "lambda_endpoints.csv")
    return dict(rows_reconstructed=len(frame), max_prediction_error=maximum_prediction,
                max_scenario_error=maximum_scenario, bootstrap_frequency_reference=True,
                literal_response_proxy_separation=True, independent_exposure_truth=False,
                fresh_public_rebuild=False, visual_review_required=True)


def group_review(root, out):
    """Reconstruct cohort scenario errors and geometry/sign/adaptive comparisons."""
    data = calibration(root)
    source = root / "E3_groups_confirmation_v2"
    check_manifest(source)
    panels = pd.read_csv(source / "panel_results.csv")
    assets = pd.read_csv(source / "asset_results.csv")
    solvers = pd.read_csv(source / "solver_events.csv")
    assert len(assets)==153000 and len(panels)==1500 and len(solvers)==24000
    assert read_json(source / "failures.json")==[]
    beta = assets[[f"beta{j}" for j in range(9)]].to_numpy()
    truth = assets[[f"truth{j}" for j in range(9)]].to_numpy()
    scenario = np.zeros(len(assets))
    for profile, part in assets.groupby("profile"):
        _, sigma, _, variance = generating_parameters(data, profile)
        scenario[part.index] = np.mean(
            (beta[part.index]-truth[part.index])**2*np.diag(sigma), axis=1
        )/variance[part.asset.to_numpy()]
    assets["scenario_mse"] = scenario
    keys = ["profile", "seed", "setting", "target"]
    direct = assets.groupby(keys).scenario_mse.mean()
    np.testing.assert_allclose(direct, panels.set_index(keys).loc[direct.index].scenario_mse,
                               atol=1e-12)
    cohort = assets[assets.is_credit].groupby(keys).agg(
        scenario_mse=("scenario_mse", "mean"), credit_mse=("credit_mse", "mean"),
        prediction_nmse=("prediction_nmse", "mean")).reset_index()
    practical, ablations = [], []
    for (profile, setting), group in cohort.groupby(["profile", "setting"]):
        base = group[group.target=="zero"].set_index("seed").sort_index()
        for target, part in group.groupby("target"):
            part = part.set_index("seed").sort_index()
            gain = np.sqrt(base.scenario_mse)-np.sqrt(part.scenario_mse)
            prediction_change = part.prediction_nmse.mean()/base.prediction_nmse.mean()-1
            practical.append(dict(profile=profile, setting=setting, target=target,
                                  scenario_rmse_gain=gain.mean(),
                                  prediction_relative_change=prediction_change,
                                  passes_scale=bool(gain.mean()>=.05 and prediction_change<=.05)))
    comparisons = [
        ("HCGL_sign", "HCGL", "add_signs"),
        ("HCGL_sign_adaptive", "HCGL_sign", "add_adaptive"),
        ("FCGL_sign_adaptive", "HCGL_sign_adaptive", "factor_vs_row_geometry"),
        ("Known_HCGL_sign_adaptive", "HCGL_sign_adaptive", "supplied_vs_learned_groups")]
    for profile, block in panels.groupby("profile"):
        for target in ["zero", "auto", "economic"]:
            for setting, baseline, label in comparisons:
                a = block[(block.target==target)&(block.setting==setting)].set_index("seed")
                b = block[(block.target==target)&(block.setting==baseline)].set_index("seed")
                for metric in ["credit_mse", "prediction_nmse", "full_covar_error"]:
                    delta = a[metric]-b[metric]
                    se = delta.std(ddof=1)/np.sqrt(len(delta))
                    ablations.append(dict(profile=profile, target=target, comparison=label,
                                          metric=metric, delta=delta.mean(),
                                          low=delta.mean()-1.96*se, high=delta.mean()+1.96*se))
    pd.DataFrame(practical).to_csv(out / "practical_scales.csv", index=False)
    pd.DataFrame(ablations).to_csv(out / "ablations.csv", index=False)
    per_asset = assets[assets.is_credit].groupby(
        ["profile", "setting", "target", "ticker"]).credit_mse.mean().unstack("target")
    per_asset["economic_minus_zero"] = per_asset.economic-per_asset.zero
    per_asset["auto_minus_zero"] = per_asset.auto-per_asset.zero
    per_asset.to_csv(out / "per_asset_credit.csv")
    diagnostics = assets[assets.is_credit].copy()
    diagnostics["credit_target_conflict"] = (
        (diagnostics.sign2.eq(0) & diagnostics.target2.abs().gt(1e-8))
        | ((diagnostics.sign2*diagnostics.target2).lt(-1e-8)))
    diagnostics["prior_filtered"] = (
        diagnostics.raw_target2-diagnostics.target2).abs().gt(1e-8)
    diagnostics["credit_estimated_zero"] = diagnostics.beta2.abs().lt(1e-6)
    diagnostics.groupby(["profile", "setting", "target"])[
        ["credit_target_conflict", "prior_filtered", "credit_estimated_zero"]
    ].mean().to_csv(out / "target_constraint_diagnostics.csv")
    panels.assign(ols_endpoint=panels.reg_lambda.eq(0), upper_endpoint=panels.lambda_index.eq(0)
                  ).groupby(["profile", "setting", "target"])[
                      ["ols_endpoint", "upper_endpoint"]].mean().to_csv(out / "endpoints.csv")
    solvers[solvers.fallback].to_csv(out / "fallbacks.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    table = pd.read_csv(source / "paired_summary.csv")
    settings = ["HCGL", "HCGL_sign", "HCGL_sign_adaptive",
                "FCGL_sign_adaptive", "Known_HCGL_sign_adaptive"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, profile in zip(axes, ["baseline", "credit_removed"]):
        for target, offset, color in [("auto", -.12, "#0072B2"),
                                      ("economic", .12, "#D55E00")]:
            part = table[(table.profile==profile)&(table.target==target)].set_index(
                "setting").loc[settings]
            ax.errorbar(part.credit_mse_delta, np.arange(5)+offset,
                        xerr=1.96*part.credit_mse_se, fmt="o", label=target, color=color)
        ax.axvline(0, color="black", lw=.8)
        ax.set(yticks=range(5), yticklabels=settings, title=profile.replace("_", " "),
               xlabel="Paired Credit-loading MSE difference vs zero target")
        ax.invert_yaxis()
    axes[0].legend()
    fig.suptitle("Target effects within each group specification\n"
                 "50 independent panel seeds per profile; pointwise 95% Monte Carlo intervals")
    fig.savefig(out / "group_target_effects.png", dpi=160)
    plt.close(fig)
    pilot = pd.read_csv(root / "E3_groups_pilot_v2/panel_results.csv")
    return dict(asset_rows=len(assets), panel_method_rows=len(panels), path_solves=len(solvers),
                fallbacks=int(solvers.fallback.sum()),
                inaccurate=int(solvers.status.eq("optimal_inaccurate").sum()),
                selected_fallbacks=int(panels.selected_solver.ne("CLARABEL").sum()),
                pilot_cross_solver_max=float(pilot.cross_solver_selected_error.max()),
                no_failed_confirmation_fits=True,
                practical_scale_passes=sum(r["passes_scale"] for r in practical),
                visual_review_required=True)


def main():
    """Write new immutable reviews, preserving source and prerequisite hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=["rolling", "groups"])
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = validate_root(args.output_root)
    check_snapshot(root)
    sys.path.insert(0, str(root / "source_snapshot/src"))
    out = root / ("E4_independent_review_v1" if args.study=="rolling"
                  else "E3_groups_review_v1")
    out.mkdir()
    shutil.copy2(HERE / "completion.py", out / "completion.py")
    result = rolling_review(root, out) if args.study=="rolling" else group_review(root, out)
    save_json(out / "summary.json", result)
    save_json(out / "manifest.json",
              {str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

