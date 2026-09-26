"""Retrospective rolling ETF evidence, with fold-local targets and no beta truth.

Uncertainty uses the stationary bootstrap of Politis and Romano (1994),
JASA 89, 1303-1313, https://doi.org/10.1080/01621459.1994.10476870.
Bootstrap restarts occur with probability 1/L; continuation wraps circularly.
Whole date rows are sampled jointly to preserve cross-asset/method dependence.
The sample is only 29 months: intervals are exploratory, not a coverage guarantee.
"""
from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

from papers.prior_targets_2026.replication.run import (
    HERE, REPO, check_snapshot, digest, lambdas, load_panels, read_json, save_json, validate_root,
)
from papers.prior_targets_2026.replication.multiasset import check_manifest

CREDIT_PRIORS = {"Global IG Bonds": .20, "Global HY Bonds": .40, "EM Bonds": .30}


def lane_data(root, config, lane):
    """Load frozen common dates and enforce literal proxy/response separation."""
    base = read_json(root / "source_snapshot/protocol.json")
    x, y = load_panels(root, base)
    names = config["credit_assets"] + list(config["controls"])
    if lane == "public_etf_proxies":
        proxy = config["proxy_factors"]
        x = y[list(proxy.values())].rename(columns={v: k for k, v in proxy.items()})
        names = [a for a in names if a not in proxy.values()]
        assert not set(names) & set(proxy.values())
    y = y[names]
    assert x.index.equals(y.index) and len(x) == 113
    assert np.isfinite(x).all().all() and np.isfinite(y).all().all()
    uni = pd.read_csv(root / "source_snapshot/data/etf_universe.csv").set_index("ticker")
    return x, y, uni


def restricted_target(x, y, indices, span):
    """Estimate a weighted intercept and chosen slopes using only supplied rows."""
    from papers.prior_targets_2026.replication.robustness import weights
    w = weights(len(y), span)
    design = np.column_stack([np.ones(len(y)), x[:, indices]])
    beta = np.linalg.lstsq(design * np.sqrt(w[:, None]), y * np.sqrt(w), rcond=None)[0]
    result = np.zeros(x.shape[1])
    result[indices] = beta[1:]
    return result


def target_paths(x, y, asset, factors, subclass, lane, span, grid, config):
    """Rebuild fixed/class anchors and restricted regression targets per fold."""
    from papers.prior_targets_2026.replication.robustness import fit_weighted
    fixed = np.zeros(len(factors))
    is_credit = asset in config["credit_assets"]
    if lane == "production_snapshot":
        if is_credit:
            fixed[factors.index("Credit")] = CREDIT_PRIORS[subclass]
    else:
        fixed[factors.index("Credit" if is_credit else config["controls"][asset])] = 1.
    small2_names = ["Rates", "Credit"] if is_credit else [config["controls"][asset]]
    small3_names = ["Equity", "Rates", "Credit"] if is_credit else small2_names
    small2 = restricted_target(x, y, [factors.index(f) for f in small2_names], span)
    small3 = restricted_target(x, y, [factors.index(f) for f in small3_names], span)
    paths = fit_weighted(x, y, grid, span, {
        "M3_fixed": fixed, "M3_small2": small2, "M3_small3": small3,
    })
    if lane == "public_etf_proxies" and is_credit:
        # The three-factor target is full same-sample OLS in this lane.
        np.testing.assert_allclose(paths["M3_small3"]["beta"],
                                   paths["M0_ols"]["beta"], atol=1e-8)
    return paths


def fit_at(x, y, asset, subclass, lane, span, end, config):
    """Select with two inner chronological folds and refit through end-1 only."""
    from papers.prior_targets_2026.replication.estimators import prediction_losses, select_index
    window, grid = config["window"], lambdas(config)
    start = end - window
    assert start >= 0
    # No supplied observations at end or later may enter fitting or selection.
    xt = x.iloc[start:end].to_numpy()
    yt = y.iloc[start:end].to_numpy()
    assert len(xt) == window
    scores = {m: np.zeros(len(grid)) for m in config["methods"]}
    folds = []
    for train_end in config["inner_train_ends"]:
        val_end = train_end + config["inner_validation_size"]
        fitted = target_paths(xt[:train_end], yt[:train_end], asset, list(x.columns),
                              subclass, lane, span, grid, config)
        for method, path in fitted.items():
            scores[method] += prediction_losses(
                xt[train_end:val_end], yt[train_end:val_end], path["beta"], path["intercept"])
        folds.append(dict(lane=lane, span=span or 0, outer_position=end,
                          outer_fit_start=str(x.index[start].date()),
                          inner_train_end=str(x.index[start + train_end - 1].date()),
                          inner_validation_start=str(x.index[start + train_end].date()),
                          inner_validation_end=str(x.index[start + val_end - 1].date()),
                          outer_fit_end=str(x.index[end - 1].date())))
    final = target_paths(xt, yt, asset, list(x.columns), subclass, lane, span, grid, config)
    selected = {}
    for method, path in final.items():
        k = len(grid) - 1 if method == "M0_ols" else select_index(scores[method])
        selected[method] = dict(beta=path["beta"][k], intercept=path["intercept"][k],
                                target=path["target"], winner=path["winner"],
                                lambda_index=k, reg_lambda=grid[k],
                                validation_mse=scores[method][k] / len(folds))
    return selected, folds


def indices_from_random(starts, restart):
    """Construct stationary-bootstrap indices, wrapping within the observed dates."""
    draws, n = starts.shape
    indices = np.zeros_like(starts)
    indices[:, 0] = starts[:, 0]
    for t in range(1, n):
        indices[:, t] = np.where(restart[:, t], starts[:, t], (indices[:, t - 1] + 1) % n)
    return indices


def stationary_indices(n, repetitions, block, seed):
    """Generate reproducible shared date indices without importing a sibling stack."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(repetitions, n))
    restart = rng.random((repetitions, n)) < 1 / block
    restart[:, 0] = True
    indices = indices_from_random(starts, restart)
    # Independent vectorised construction from the most recent restart location.
    t = np.arange(n)[None, :]
    last_restart = np.maximum.accumulate(np.where(restart, t, 0), axis=1)
    reference = (np.take_along_axis(starts, last_restart, axis=1) + t - last_restart) % n
    np.testing.assert_array_equal(indices, reference)
    return indices


def bootstrap_summary(frame, config):
    """Preserve date dependence and same-date cross-asset dependence in paired losses."""
    result = []
    spec = config["uncertainty"]
    for (lane, span, cohort), group in frame.groupby(["lane", "span", "cohort"]):
        monthly = group.groupby(["date", "method"]).nmse.mean().unstack("method").sort_index()
        assert len(monthly) == 29 and not monthly.isna().any().any()
        for block in spec["mean_block_lengths"]:
            indices = stationary_indices(len(monthly), spec["replications"], block,
                                         spec["seed"] + block)
            for method in monthly:
                difference = (monthly[method] - monthly.M1_zero).to_numpy()
                means = difference[indices].mean(axis=1)
                low, high = np.quantile(means, [.025, .975])
                result.append(dict(lane=lane, span=span, cohort=cohort, method=method,
                                   block=block, months=len(monthly),
                                   mean_nmse=float(monthly[method].mean()),
                                   paired_delta=float(difference.mean()),
                                   bootstrap_low=float(low), bootstrap_high=float(high)))
    return pd.DataFrame(result)


def numerical_checks(root, config):
    """Verify weighted targets, circular resampling, truncation and future perturbations."""
    from papers.prior_targets_2026.replication.robustness import prepare_weighted, reject_assertion
    max_target_error = 0.
    tests = 0
    for lane in config["lanes"]:
        x, y, uni = lane_data(root, config, lane)
        for span in config["spans"]:
            asset = "LQD"
            values = x.iloc[:60].to_numpy(), y[asset].iloc[:60].to_numpy()
            chosen = [list(x.columns).index(f) for f in ["Rates", "Credit"]]
            direct = restricted_target(*values, chosen, span)
            z, v, _, _, xs, ys, _ = prepare_weighted(values[0][:, chosen], values[1], span)
            reference = np.linalg.lstsq(z, v, rcond=None)[0] * ys / xs
            max_target_error = max(max_target_error, np.max(abs(reference - direct[chosen])))
            for end in [84, 98, 112]:
                args = (asset, uni.loc[asset, "sub_asset_class"], lane, span, end, config)
                first, _ = fit_at(x, y[asset], *args)
                future_x, future_y = x.copy(), y[asset].copy()
                future_x.iloc[end:] += 1000.
                future_y.iloc[end:] -= 500.
                changed, _ = fit_at(future_x, future_y, *args)
                truncated, _ = fit_at(x.iloc[:end], y[asset].iloc[:end], *args)
                for method in first:
                    np.testing.assert_allclose(first[method]["beta"], changed[method]["beta"],
                                               atol=1e-12)
                    np.testing.assert_allclose(first[method]["beta"], truncated[method]["beta"],
                                               atol=1e-12)
                    assert (first[method]["lambda_index"] == changed[method]["lambda_index"]
                            == truncated[method]["lambda_index"])
                tests += 1
    assert max_target_error < 1e-10
    starts = np.full((2, 5), 4)
    restart = np.zeros_like(starts, dtype=bool)
    restart[:, 0] = True
    wrapped = indices_from_random(starts, restart)
    np.testing.assert_array_equal(wrapped, [[4, 0, 1, 2, 3]] * 2)
    broken = np.minimum(4 + np.arange(5), 4)[None].repeat(2, axis=0)
    reject_assertion(lambda: np.testing.assert_array_equal(wrapped, broken))
    for block in config["uncertainty"]["mean_block_lengths"]:
        stationary_indices(29, 50, block, 989999)
    return dict(weighted_target_max_error=float(max_target_error),
                future_perturbation_and_truncation_cases=tests,
                independent_bootstrap_indices_verified=True,
                deliberate_no_wrap_defect_rejected=True)


def run(root, out, config):
    """Collect all prespecified rolling predictions and coefficient/scenario paths."""
    records, folds, failures, coverage = [], [], [], []
    for lane in config["lanes"]:
        x, y, uni = lane_data(root, config, lane)
        factor_sd = x.iloc[:60].std(ddof=0).to_numpy()
        response_var = y.iloc[:60].var(ddof=0)
        shocks = np.vstack([np.diag(factor_sd), factor_sd[None]])
        shocks = np.vstack([shocks, -shocks])
        for asset in y:
            coverage.append(dict(lane=lane, asset=asset, observations=len(y),
                                 first=str(y.index[0].date()), last=str(y.index[-1].date()),
                                 cohort="credit" if asset in config["credit_assets"] else "control",
                                 response_scale_variance=float(response_var[asset]),
                                 literal_proxy_overlap=False))
        for span in config["spans"]:
            previous = {}
            for end in range(config["window"], len(x)):
                for asset in y:
                    try:
                        fitted, fold_rows = fit_at(
                            x, y[asset], asset, uni.loc[asset, "sub_asset_class"],
                            lane, span, end, config)
                        if asset == y.columns[0]:
                            folds.extend(fold_rows)
                        for method, result in fitted.items():
                            beta, target = result["beta"], result["target"]
                            estimate = float(x.iloc[end].to_numpy() @ beta + result["intercept"])
                            observed = float(y[asset].iloc[end])
                            scenario = shocks @ beta
                            key = (asset, method)
                            old = previous.get(key)
                            change = (float(np.mean((scenario - old["scenario"])**2)
                                             / response_var[asset]) if old else np.nan)
                            row = dict(
                                lane=lane, span=span or 0, date=str(y.index[end].date()),
                                outer_position=end, asset=asset, method=method,
                                cohort="credit" if asset in config["credit_assets"] else "control",
                                train_start=str(x.index[end - config["window"]].date()),
                                train_end=str(x.index[end - 1].date()),
                                actual=observed, prediction=estimate,
                                squared_error=(observed-estimate)**2,
                                response_var=float(response_var[asset]),
                                nmse=(observed-estimate)**2 / response_var[asset],
                                scenario_change_nmse=change,
                                target_distance_norm=float(np.sqrt(np.mean(
                                    ((beta - target) * factor_sd)**2) / response_var[asset])),
                                winner=str(x.columns[result["winner"]]),
                                winner_switch=bool(old and old["winner"] != result["winner"]),
                                has_previous=old is not None, intercept=result["intercept"],
                                lambda_index=result["lambda_index"],
                                reg_lambda=result["reg_lambda"],
                                validation_mse=result["validation_mse"],
                            )
                            for f, factor in enumerate(x.columns):
                                row["beta_" + factor] = beta[f]
                                row["target_" + factor] = target[f]
                            records.append(row)
                            previous[key] = dict(scenario=scenario, winner=result["winner"])
                    except Exception as error:
                        failures.append(dict(lane=lane, span=span, end=end,
                                             asset=asset, error=repr(error)))
                print(f"{lane}, span={span}, {x.index[end].date()} complete", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(out / "rolling_results.csv", index=False)
    pd.DataFrame(folds).to_csv(out / "fold_cutoffs.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)
    bootstrap_summary(frame, config).to_csv(out / "paired_bootstrap.csv", index=False)
    frame.groupby(["lane", "span", "cohort", "method"]).agg(
        mean_nmse=("nmse", "mean"), scenario_change_nmse=("scenario_change_nmse", "mean"),
        target_distance_norm=("target_distance_norm", "mean"),
    ).to_csv(out / "method_summary.csv")
    transitions = frame[frame.has_previous]
    transitions.groupby(["lane", "span", "cohort", "method"]).winner_switch.mean().to_csv(
        out / "winner_switch_rates.csv")
    save_json(out / "failures.json", failures)


def validate(frame, folds, config):
    """Check expected counts, every temporal boundary and direct saved prediction losses."""
    counts = {"production_snapshot": 21, "public_etf_proxies": 20}
    assert len(frame) == sum(counts.values()) * 2 * 29 * 8
    assert not frame.duplicated(["lane", "span", "date", "asset", "method"]).any()
    for lane, count in counts.items():
        part = frame[frame.lane == lane]
        assert part.asset.nunique() == count
        assert part.date.nunique() == 29
        assert set(part.method) == set(config["methods"])
        assert set(part.span) == {0, 36}
        assert (part.groupby(["span", "date", "asset"]).method.nunique() == 8).all()
    proxy = frame[frame.lane == "public_etf_proxies"]
    assert not set(proxy.asset) & set(config["proxy_factors"].values())
    assert (pd.to_datetime(frame.train_end) < pd.to_datetime(frame.date)).all()
    assert (pd.to_datetime(folds.inner_train_end)
            < pd.to_datetime(folds.inner_validation_start)).all()
    assert (pd.to_datetime(folds.inner_validation_end)
            <= pd.to_datetime(folds.outer_fit_end)).all()
    assert (folds.outer_position >= config["window"]).all()
    assert len(folds) == 2 * 2 * 29 * 2
    np.testing.assert_allclose(frame.squared_error, (frame.actual-frame.prediction)**2, atol=1e-14)
    np.testing.assert_allclose(frame.nmse, frame.squared_error / frame.response_var, atol=1e-11)
    assert np.isfinite(frame[["prediction", "nmse", "target_distance_norm"]]).all().all()
    assert frame.groupby(["lane", "span", "asset", "method"]).has_previous.sum().eq(28).all()


def verify(out, config):
    """Validate results and repeat dependence-aware summaries from saved date rows."""
    from papers.prior_targets_2026.replication.robustness import reject_assertion
    frame = pd.read_csv(out / "rolling_results.csv")
    folds = pd.read_csv(out / "fold_cutoffs.csv")
    validate(frame, folds, config)
    bad = frame.copy()
    bad.loc[0, "train_end"] = bad.loc[0, "date"]
    reject_assertion(lambda: validate(bad, folds, config))
    bad = frame.copy()
    bad.loc[0, "nmse"] += 1.
    reject_assertion(lambda: validate(bad, folds, config))
    reject_assertion(lambda: validate(frame.iloc[1:], folds, config))
    expected = bootstrap_summary(frame, config)
    saved = pd.read_csv(out / "paired_bootstrap.csv")
    numeric = expected.select_dtypes(include="number").columns
    np.testing.assert_allclose(expected[numeric], saved[numeric], atol=1e-11)
    assert read_json(out / "failures.json") == []
    return dict(rows=len(frame), outer_months=29, lane_weight_settings=4,
                fit_asset_windows=len(frame) // 8, fold_records=len(folds),
                overlap_defect_rejected=True, corrupted_loss_rejected=True,
                missing_row_rejected=True, bootstrap_reconstructed=True)


def plots(out):
    """Show frozen representative assets' trajectories and paired empirical losses."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.read_csv(out / "rolling_results.csv", parse_dates=["date"])
    methods = ["M1_zero", "M2_auto", "M3_fixed", "M3_small2", "M3_small3"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for row, lane in enumerate(["production_snapshot", "public_etf_proxies"]):
        for ax, asset in zip(axes[row], ["LQD", "VCSH", "VCIT"]):
            part = frame[(frame.lane == lane) & (frame.span == 0) & (frame.asset == asset)]
            for method in methods:
                line = part[part.method == method]
                ax.plot(line.date, line.beta_Credit, label=method, linewidth=1.3)
            ax.set(title=asset,
                   ylabel="Native Credit beta" if row == 0 else "Native HYG-proxy beta")
            ax.tick_params(axis="x", labelrotation=25, labelsize=8)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Rolling exposure estimates, uniform weights — stability is not accuracy\n"
                 "Production factor snapshot (top); disjoint ETF proxies (bottom)")
    fig.savefig(out / "rolling_credit_paths.png", dpi=160)
    plt.close(fig)
    table = pd.read_csv(out / "paired_bootstrap.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    order = ["M2_auto", "M3_fixed", "M3_small2", "M3_small3", "M4_free_winner", "M5_post_lasso"]
    for ax, lane in zip(axes, ["production_snapshot", "public_etf_proxies"]):
        part = table[(table.lane == lane) & (table.cohort == "credit")
                     & (table.span == 0) & (table.block == 6)].set_index("method").loc[order]
        for i, (_, item) in enumerate(part.iterrows()):
            ax.plot([item.bootstrap_low, item.bootstrap_high], [i, i], color="#0072B2")
            ax.plot(item.paired_delta, i, "o", color="#0072B2")
        ax.axvline(0, color="black", linewidth=.8)
        ax.set(yticks=range(len(order)), yticklabels=order, title=lane.replace("_", " "),
               xlabel="Paired normalised reconstruction MSE difference vs zero")
        ax.invert_yaxis()
    fig.suptitle("Exploratory held-out conditional reconstruction: credit cohort\n"
                 "29 months; 95% stationary-bootstrap intervals, mean block length 6")
    fig.savefig(out / "rolling_prediction_comparison.png", dpi=160)
    plt.close(fig)


def main():
    """Freeze and run the prespecified retrospective lanes or verify their artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "verify"])
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = validate_root(args.output_root)
    check_snapshot(root)
    sys.path.insert(0, str(root / "source_snapshot/src"))
    check_manifest(root / "E2_robustness_v1")
    config = read_json(HERE / "rolling_protocol_v2.json")
    out = root / "E4_rolling_v1"
    if args.command == "verify":
        check_manifest(out)
        print(json.dumps(verify(out, read_json(out / "archive/rolling_protocol_v2.json")),
                         indent=2))
        return
    out.mkdir()
    (out / "archive").mkdir()
    for name in ["rolling.py", "rolling_protocol_v2.json", "robustness.py", "estimators.py",
                 "multiasset.py", "run.py"]:
        shutil.copy2(HERE / name, out / "archive" / name)
    shutil.copy2(REPO / "papers/jss_2026/applications/fetch_etf_panel.py",
                 out / "archive/fetch_etf_panel.py")
    save_json(out / "numerical_checks.json", numerical_checks(root, config))
    started = time.perf_counter()
    run(root, out, config)
    checks = verify(out, config)
    save_json(out / "verification.json", checks)
    plots(out)
    original = read_json(root / "E0/summary.json")["versions"]
    versions = {name: metadata.version(name) for name in original}
    assert versions == original
    save_json(out / "summary.json", {
        **checks, "seconds": time.perf_counter() - started, "failures": 0,
        "versions": versions, "protocol_sha256": digest(HERE / "rolling_protocol_v2.json"),
        "limitations": config["provenance"], "interpretation": config["scores"],
        "availability_amendment": config["availability_amendment"],
    })
    save_json(out / "manifest.json", {
        str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()
    })
    print(json.dumps(read_json(out / "summary.json"), indent=2))


if __name__ == "__main__":
    main()

