"""Independent post-run evidence checks; no fitting, tuning or result replacement."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from papers.prior_targets_2026.replication.run import HERE, digest, read_json, save_json, validate_root


def assert_hashes(folder):
    """Reject a changed completed experiment before summarising it."""
    for name, expected in read_json(folder / "manifest.json").items():
        assert digest(folder / name) == expected, name


def review_e2(source, out):
    """Report standardised zero thresholds and bias without changing old metrics."""
    frame = pd.read_csv(source / "selected_results.csv")
    rows = []
    for (cell, method), part in frame.groupby(["cell", "method"]):
        standardised = part.beta1 / np.sqrt(part.response_var)
        row = dict(cell=cell, method=method,
                   secondary_bias=float((part.beta1 - part.truth1).mean()),
                   secondary_native_rmse=float(np.sqrt(part.secondary_mse.mean())),
                   scenario_normalised_rmse=float(np.sqrt(part.scenario_mse.mean())),
                   native_zero_rate_original=float(part.secondary_zero.mean()))
        for threshold in [1e-8, 1e-6, 1e-4]:
            row[f"population_standardised_zero_rate_{threshold:g}"] = float(
                (standardised.abs() <= threshold).mean())
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(out / "bias_and_zero_thresholds.csv", index=False)
    change = (result["population_standardised_zero_rate_0.0001"]
              - result["population_standardised_zero_rate_1e-08"]).abs()
    convention = (result["population_standardised_zero_rate_1e-06"]
                  - result.native_zero_rate_original).abs()
    return dict(method_cells=len(result), max_zero_rate_threshold_change=float(change.max()),
                max_zero_rate_native_vs_population_standardised_change=float(convention.max()),
                units="Population factor variance1; coefficient divided by future response SD. "
                "Native-zero indicators in the original raw table are preserved.",
                caveat="MSE conclusions do not use a zero classification. "
                "Threshold diagnostics are descriptive post-run checks.")


def verify_intervals(panels, summary):
    """Recompute paired effects by a matrix pivot and explicit sum-of-squares SE."""
    from papers.prior_targets_2026.replication.multiasset import METRICS
    for metric in METRICS:
        wide = panels.pivot(index=["profile", "n", "seed"], columns="method", values=metric)
        for _, row in summary.iterrows():
            block = wide.loc[(row.profile, row.n)]
            difference = (block[row.method] - block.M1_zero).to_numpy()
            mean = sum(difference) / len(difference)
            centred = difference - mean
            se = np.sqrt(centred @ centred / (len(centred) * (len(centred) - 1)))
            np.testing.assert_allclose(
                [row[metric + "_delta"], row[metric + "_se"],
                 row[metric + "_low"], row[metric + "_high"]],
                [mean, se, mean - 1.96 * se, mean + 1.96 * se], atol=1e-12,
            )


def review_e3(source, out):
    """Validate intervals independently and retain per-ETF attribution evidence."""
    panels = pd.read_csv(source / "panel_results.csv")
    summary = pd.read_csv(source / "paired_summary.csv")
    verify_intervals(panels, summary)
    broken = summary.copy()
    broken.loc[0, "credit_mse_delta"] += 1
    try:
        verify_intervals(panels, broken)
    except AssertionError:
        rejected = True
    else:
        raise AssertionError("Corrupted interval was accepted")
    columns = ["profile", "n", "seed", "asset", "ticker", "method", "is_credit",
               "credit_mse", "credit_bias", "true_credit", "estimated_credit",
               "reg_lambda", "lambda_index", "scenario_mse", "prediction_nmse"]
    assets = pd.read_csv(source / "asset_results.csv", usecols=columns)
    per_asset = assets.groupby(["profile", "n", "ticker", "is_credit", "method"])[
        ["credit_mse", "credit_bias", "true_credit", "estimated_credit",
         "scenario_mse", "prediction_nmse"]].mean().reset_index()
    per_asset.to_csv(out / "per_asset_means.csv", index=False)
    practical = []
    for (profile, n), group in per_asset[per_asset.is_credit].groupby(["profile", "n"]):
        by_method = group.groupby("method")[["scenario_mse", "prediction_nmse"]].mean()
        base_rmse = float(np.sqrt(by_method.loc["M1_zero", "scenario_mse"]))
        base_prediction = float(by_method.loc["M1_zero", "prediction_nmse"])
        for method, record in by_method.iterrows():
            gain = base_rmse - float(np.sqrt(record.scenario_mse))
            cost = float(record.prediction_nmse / base_prediction - 1)
            practical.append(dict(
                profile=profile, n=n, method=method, credit_scenario_rmse_gain=gain,
                credit_prediction_relative_change=cost,
                exceeds_prespecified_rmse_gain_005=bool(gain >= .05),
                within_prediction_cost_005=bool(cost <= .05),
            ))
    pd.DataFrame(practical).to_csv(out / "practical_effect_scales.csv", index=False)

    zero_controls = assets[~assets.is_credit].pivot(
        index=["profile", "n", "seed", "ticker"], columns="method", values="estimated_credit")
    for method in ["M3_economic", "M3_economic_noisy", "M3_small_model"]:
        np.testing.assert_array_equal(zero_controls[method], zero_controls.M1_zero)
    endpoints = assets.groupby(["profile", "n", "method"]).agg(
        ols_endpoint_rate=("reg_lambda", lambda x: float((x == 0).mean())),
        target_endpoint_rate=("lambda_index", lambda x: float((x == 0).mean())),
    ).reset_index()
    endpoints.to_csv(out / "grid_endpoints.csv", index=False)
    summary[summary.n == 112].to_csv(out / "all_n112_comparisons.csv", index=False)
    result = []
    for (profile, n), group in per_asset[per_asset.is_credit].groupby(["profile", "n"]):
        matrix = group.pivot(index="ticker", columns="method", values="credit_mse")
        for method in matrix:
            difference = matrix[method] - matrix.M1_zero
            result.append(dict(profile=profile, n=n, method=method,
                               assets_improved=int((difference < -1e-10).sum()),
                               assets_worse=int((difference > 1e-10).sum()),
                               assets_equal=int((difference.abs() <= 1e-10).sum())))
    pd.DataFrame(result).to_csv(out / "cohort_asset_counts.csv", index=False)
    return dict(independent_paired_intervals_verified=True,
                corrupted_interval_rejected=rejected,
                zero_target_noncredit_controls_exactly_equal=True,
                asset_rows=len(assets), panel_method_rows=len(panels),
                per_asset_count_absolute_tolerance=1e-10,
                note="Per-ETF and endpoint tables are descriptive diagnostics; "
                "uncertainty is across panel seeds, not across ETFs.")


def main():
    """Create a separate immutable review directory for completed scientific outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["E2", "E3"], required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--review-name", default=None)
    args = parser.parse_args()
    root = validate_root(args.output_root)
    source = root / ("E2_robustness_v1" if args.study == "E2" else "E3_L1_confirmation_v1")
    assert_hashes(source)
    out = validate_root(root / (args.review_name or (args.study + "_independent_review_v1")))
    out.mkdir()
    shutil.copy2(HERE / "review.py", out / "review.py")
    report = review_e2(source, out) if args.study == "E2" else review_e3(source, out)
    original_versions = read_json(root / "E0/summary.json")["versions"]
    current_versions = {name: metadata.version(name) for name in original_versions}
    report["versions_observed_at_review"] = current_versions
    report["versions_match_E0"] = current_versions == original_versions
    assert current_versions == original_versions, "Runtime versions changed since E0"
    report["source_manifest_sha256"] = digest(source / "manifest.json")
    save_json(out / "summary.json", report)
    save_json(out / "manifest.json", {
        p.name: digest(p) for p in out.iterdir() if p.is_file()
    })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

