"""Producer family for estimation exhibits.

Exhibits
--------
``quickstart_workflow.png`` for ``docs/quickstart.md``: true and estimated loadings of twelve
funds on four factors, and the held-out R-squared that selects the penalty.

``sparse_factor_model_path.png`` for ``docs/sparse_factor_model.md``: the loadings of one response
and the loading error of the whole panel along the penalty grid.

``sign_constraints_and_priors_error.png`` for ``docs/sign_constraints_and_priors.md``: loading
error of the zero-centred and the prior-centred penalty along the grid, and the sampling
distribution of the error with and without the constraints.

``group_penalties_selection.png`` for ``docs/group_penalties_hcgl_fcgl.md``: the loadings kept by
the LASSO, HCGL and FCGL penalties at one common penalty, against the true loadings.

``prior_targets_paths.png`` for ``docs/prior_targets.md``: the Rates and Inflation loadings of one
inflation-linked response along the penalty grid under a zero, an automatic and a joint centre.

``ewma_weighting_shrinkage.png`` for ``docs/ewma_weighting_and_ragged_histories.md``: EWMA weight
profiles of three spans, and the shrinkage of loadings with ragged histories under the two loss
normalisations.

``pooled_sign_recovery.png`` for ``docs/gated_cluster_pooled_signs.md``: sign recovery on active
cells and false signs on null cells against the gate threshold, per response and pooled within
known clusters.

``adaptive_penalty_weights.png`` for ``docs/adaptive_penalty_weights.md``: the adaptive weight as a
function of the univariate slope, and the fitted loadings of a plain and an adaptive LASSO by true
loading.

``cooperative_lasso_geometry.png`` for ``docs/cooperative_lasso.md``: unit level sets of the group
and cooperative penalties for two loadings, and the loading of a cluster's rogue member along the
penalty grid under three penalties.

``unilasso_two_stage.png`` for ``docs/unilasso.md``: final UniLasso loadings against the stage-one
univariate slopes, with and without non-negative stage-two coefficients, and the noise loadings
kept along the penalty grid under three settings of the two UniLasso options.

``penalty_selection_paths.png`` for ``docs/penalty_selection.md``: held-out R-squared and held-out
residual sphericity along one penalty grid, on a complete panel and on a panel with an omitted
factor, with the penalty each selector takes.

Each calculation is the worked example of its article, loaded from ``examples/docs/``. Nothing is
re-implemented here: this module calls the example's functions, checks that the registry
configuration describes what actually ran, and draws.
"""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullFormatter

from tools.docs_analytics.fixtures import (
    GRID,
    INK,
    INK_MUTED,
    SERIES,
    SURFACE,
    load_example,
    loading_colormap,
    style_axis,
    verify_parameters,
)

MARKERS = ("o", "s", "D")
LINE = {"linewidth": 2, "markersize": 7, "markeredgecolor": SURFACE, "markeredgewidth": 1.5}


def _heatmap(axis, values: np.ndarray, rows: list, columns: list, limit: float, title: str,
             show_rows: bool, annotate: bool = True, tolerance: float = 5e-3) -> None:
    """Draw a loading matrix; cells under ``tolerance`` stay on the surface colour, unlabelled."""
    shown = np.where(np.abs(values) > tolerance, values, 0.0)
    axis.imshow(shown, cmap=loading_colormap(), vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(columns)), labels=columns, fontsize=10)
    axis.set_yticks(range(len(rows)), labels=rows if show_rows else [""] * len(rows), fontsize=10)
    axis.tick_params(length=0, colors=INK_MUTED)
    axis.set_xticks(np.arange(-0.5, len(columns)), minor=True)
    axis.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    axis.grid(which="minor", color=GRID, linewidth=0.8)
    axis.tick_params(which="minor", length=0)
    for spine in axis.spines.values():
        spine.set_color(GRID)
    axis.set_title(title, fontsize=12, loc="left", color=INK)
    if not annotate:
        # Without numbers a faint cell could pass for an empty one: mark every kept cell.
        kept_rows, kept_columns = np.nonzero(np.abs(values) > tolerance)
        axis.plot(kept_columns, kept_rows, linestyle="none", marker="o", markersize=3.5,
                  color=INK)
        return
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if abs(values[i, j]) > tolerance:
                colour = SURFACE if abs(values[i, j]) > 0.6 * limit else INK
                axis.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center",
                          fontsize=9, color=colour)


def _quickstart(spec: dict) -> tuple[Figure, dict, dict]:
    """True against estimated loadings, and the cross-validation curve that chose the penalty."""
    example = load_example(spec["example"])
    verify_parameters(example, spec["parameters"])
    x, y = example["make_panel"]()
    selector = example["select_and_fit"](x, y)
    model = selector.best_model_
    truth = example["TRUE_BETA"]
    scores = selector.cv_scores_
    mean_scores = scores.mean(axis=1)

    figure = Figure(figsize=(12.0, 4.8), facecolor=SURFACE, layout="constrained")
    left, middle, right = figure.subplots(1, 3, width_ratios=(1.15, 1.0, 1.35))
    limit = float(np.abs(truth).max())
    _heatmap(left, truth, list(y.columns), list(x.columns), limit, "True loadings", True)
    _heatmap(middle, model.coef_.to_numpy(), list(y.columns), list(x.columns), limit,
             "Estimated loadings", False)

    right.vlines(scores.index, scores.min(axis=1), scores.max(axis=1), color=SERIES[0],
                 linewidth=1.2, alpha=0.6)
    right.plot(mean_scores.index, mean_scores, color=SERIES[0], marker=MARKERS[0], **LINE)
    right.axvline(selector.best_lambda_, color=INK_MUTED, linestyle="--", linewidth=1.2)
    selected = f"selected {selector.best_lambda_:.0e}\nmean R-squared {selector.best_score_:.2f}"
    right.annotate(selected,
                   xy=(selector.best_lambda_, 0.15), xytext=(-8, 0), textcoords="offset points",
                   fontsize=11, color=INK, ha="right", va="center")
    right.set_xscale("log")
    right.invert_xaxis()
    right.set_ylim(-0.1, 1.0)
    right.set_xlabel("reg_lambda (the penalty falls to the right)")
    right.set_ylabel("Held-out R-squared: mean and range of 4 folds")
    right.set_title("Penalty selection", fontsize=12, loc="left")
    style_axis(right)

    tables = {
        "quickstart_true_loadings": pd.DataFrame(truth, index=y.columns, columns=x.columns),
        "quickstart_estimated_loadings": model.coef_,
        "quickstart_cv_scores": scores.rename(columns=lambda fold: f"fold_{fold}"),
    }
    checks = {
        "quickstart_selected_penalty_maximises_mean_score": bool(
            np.isclose(selector.best_lambda_, mean_scores.idxmax())),
        "quickstart_plotted_loadings_are_the_fitted_model": bool(
            np.allclose(tables["quickstart_estimated_loadings"], selector.best_model_.coef_)),
        "quickstart_three_clusters_recovered": bool(model.clusters_.nunique() == 3),
    }
    return figure, tables, checks


def _sparse_factor_model(spec: dict) -> tuple[Figure, dict, dict]:
    """Loadings of one response and the panel's loading error along the penalty grid."""
    example = load_example(spec["example"])
    verify_parameters(example, spec["parameters"])
    x, y = example["make_panel"]()
    summary, loadings = example["penalty_path"](x, y)
    truth = pd.Series(example["TRUE_BETA"][1], index=x.columns)
    n_train = example["N_TRAIN"]
    design = np.column_stack([np.ones(n_train), x.iloc[:n_train].to_numpy()])
    ols = np.linalg.lstsq(design, y.iloc[:n_train].to_numpy(), rcond=None)[0][1:].T
    rmse_ols = float(np.sqrt(np.mean((ols - example["TRUE_BETA"]) ** 2)))

    figure = Figure(figsize=(11.0, 4.6), facecolor=SURFACE, layout="constrained")
    left, right = figure.subplots(1, 2)
    for name in x.columns:
        relevant = truth[name] != 0.0
        left.plot(loadings.index, loadings[name], color=SERIES[0 if relevant else 1],
                  marker=MARKERS[0 if relevant else 1], linewidth=2 if relevant else 1.2,
                  markersize=6 if relevant else 4, markeredgecolor=SURFACE, markeredgewidth=1.0)
        if relevant:
            left.axhline(truth[name], color=INK_MUTED, linestyle="--", linewidth=1.0)
            left.annotate(f"{name}: true {truth[name]:.1f}", xy=(loadings.index[-1], truth[name]),
                          xytext=(0, -15), textcoords="offset points", ha="right", fontsize=11,
                          color=INK)
    irrelevant = [name for name in x.columns if truth[name] == 0.0]
    left.annotate("six factors with a true loading of zero",
                  xy=(loadings.index[-1], float(loadings[irrelevant].iloc[-1].max())),
                  xytext=(0, 9), textcoords="offset points", ha="right", fontsize=11, color=INK)
    left.set_xscale("log")
    left.invert_xaxis()
    left.set_xlabel("reg_lambda (the penalty falls to the right)")
    left.set_ylabel("Estimated loading of asset_2")
    left.set_title("Loadings of one response along the grid", fontsize=12.5, loc="left")

    best = summary["loading_rmse"].idxmin()
    right.plot(summary.index, summary["loading_rmse"], color=SERIES[0], marker=MARKERS[0], **LINE)
    right.axhline(rmse_ols, color=INK_MUTED, linestyle="--", linewidth=1.2)
    right.annotate(f"ordinary least squares {rmse_ols:.3f}", xy=(summary.index[-1], rmse_ols),
                   xytext=(0, 10), textcoords="offset points", ha="right", fontsize=11,
                   color=INK_MUTED)
    right.annotate(f"{summary.loc[best, 'loading_rmse']:.3f} with"
                   f" {int(summary.loc[best, 'n_loadings'])} loadings kept",
                   xy=(best, summary.loc[best, "loading_rmse"]), xytext=(0, -20),
                   textcoords="offset points", ha="center", fontsize=11, color=INK)
    right.set_xscale("log")
    right.invert_xaxis()
    right.set_ylim(0.0, None)
    right.set_xlabel("reg_lambda (the penalty falls to the right)")
    right.set_ylabel("Loading RMSE against the true loadings")
    right.set_title("Estimation error of all 48 loadings", fontsize=12.5, loc="left")
    for axis in (left, right):
        style_axis(axis)

    tables = {"sparse_factor_model_path": summary, "sparse_factor_model_asset_2_loadings": loadings}
    checks = {
        "sparse_factor_model_interior_minimum_beats_ols": bool(
            summary["loading_rmse"].min() < rmse_ols
            and summary.index[0] > best > summary.index[-1]),
        "sparse_factor_model_unpenalised_end_is_ols": bool(
            np.isclose(summary["loading_rmse"].iloc[-1], rmse_ols, atol=2e-3)),
        "sparse_factor_model_support_grows_as_penalty_falls": bool(
            summary["n_loadings"].is_monotonic_increasing),
    }
    return figure, tables, checks


def _sign_constraints_and_priors(spec: dict) -> tuple[Figure, dict, dict]:
    """Where each penalty shrinks to, and the sampling error of the three estimators."""
    example = load_example(spec["example"])
    verify_parameters(example, spec["parameters"])
    x, y = example["make_panel"]()
    path = example["penalty_path"](x, y)
    sample = example["sampling_comparison"]()
    truth, prior = example["TRUE_BETA"], example["PRIOR"]
    prior_rmse = float(np.sqrt(np.mean((prior - truth) ** 2)))
    zero_rmse = float(np.sqrt(np.mean(truth ** 2)))

    figure = Figure(figsize=(11.0, 4.6), facecolor=SURFACE, layout="constrained")
    left, right = figure.subplots(1, 2, width_ratios=(1.25, 1.0))
    labels = {"signs and prior": "penalty centred on the prior", "signs": "penalty centred on zero"}
    for (name, label), colour, marker, offset in zip(labels.items(), SERIES, MARKERS, (8, -18)):
        left.plot(path.index, path[name], color=colour, marker=marker, **LINE)
        # Direct labels at the large-penalty end, where the two penalties have parted.
        left.annotate(label, xy=(path.index[-1], path[name].iloc[-1]), xytext=(0, offset),
                      textcoords="offset points", ha="left", fontsize=11, color=INK)
    references = (
        (prior_rmse, f"error of the prior itself {prior_rmse:.3f}", path.index[-1], "left", -17),
        (zero_rmse, f"error of all-zero loadings {zero_rmse:.3f}", path.index[0], "right", 6),
    )
    for level, text, anchor, align, offset in references:
        left.axhline(level, color=INK_MUTED, linestyle="--", linewidth=1.2)
        left.annotate(text, xy=(anchor, level), xytext=(0, offset), ha=align,
                      textcoords="offset points", fontsize=11, color=INK_MUTED)
    left.set_xscale("log")
    left.invert_xaxis()
    left.set_ylim(0.0, 0.65)
    left.set_xlabel("reg_lambda (the penalty falls to the right)")
    left.set_ylabel("Loading RMSE against the true loadings")
    left.set_title("One panel of 36 months, sign matrix applied", fontsize=12.5, loc="left")

    order = list(example["ESTIMATORS"])
    grouped = sample.groupby("estimator")["loading_rmse"]
    summary = pd.DataFrame({
        "mean": grouped.mean(), "p10": grouped.quantile(0.10), "p90": grouped.quantile(0.90),
        "violation_share": sample.groupby("estimator")["violation_share"].mean(),
    }).loc[order]
    positions = np.arange(len(order))[::-1]
    right.hlines(positions, summary["p10"], summary["p90"], color=SERIES[0], linewidth=3,
                 alpha=0.45)
    right.plot(summary["mean"], positions, linestyle="none", color=SERIES[0], marker="o",
               markersize=9, markeredgecolor=SURFACE, markeredgewidth=1.5)
    for position, (name, row) in zip(positions, summary.iterrows()):
        right.annotate(f"{name}: {row['mean']:.3f}", xy=(row["mean"], position), xytext=(0, 12),
                       textcoords="offset points", ha="center", fontsize=11, color=INK)
    right.set_yticks([])
    right.set_ylim(-0.6, len(order) - 0.3)
    right.set_xlim(0.0, 1.12 * float(summary["p90"].max()))
    right.set_xlabel("Loading RMSE: mean and 10th to 90th percentile")
    right.set_title(f"{example['N_PANELS']} redrawn panels at reg_lambda = 3e-5",
                    fontsize=12.5, loc="left")
    for axis in (left, right):
        style_axis(axis)
    right.grid(axis="y", visible=False)

    tables = {
        "sign_constraints_and_priors_path": path,
        "sign_constraints_and_priors_sampling": summary.rename_axis("estimator"),
    }
    checks = {
        "signs_and_priors_large_penalty_returns_the_prior": bool(
            np.isclose(path["signs and prior"].iloc[-1], prior_rmse, atol=1e-3)),
        "signs_and_priors_large_penalty_returns_zero": bool(
            np.isclose(path["signs"].iloc[-1], zero_rmse, atol=1e-3)),
        "signs_and_priors_each_step_lowers_mean_error": bool(
            summary["mean"].is_monotonic_decreasing),
        "signs_and_priors_constrained_fits_have_no_violation": bool(
            (summary.loc[["signs", "signs and prior"], "violation_share"] == 0.0).all()),
    }
    return figure, tables, checks


def _group_penalties(spec: dict) -> tuple[Figure, dict, dict]:
    """True loadings and the loadings kept by three penalties at one common penalty."""
    example = load_example(spec["example"])
    verify_parameters(example, spec["parameters"])
    x, y = example["make_panel"]()
    truth = example["TRUE_BETA"]
    shown = ("LASSO", "HCGL", "FCGL")
    fits = {name: example["fit"](x, y, name, example["COMMON_LAMBDA"]) for name in shown}
    counts = pd.DataFrame(
        {name: example["support_counts"](model.coef_.to_numpy()) for name, model in fits.items()}).T

    figure = Figure(figsize=(12.5, 4.8), facecolor=SURFACE, layout="constrained")
    axes = figure.subplots(1, 4, width_ratios=(1.35, 1.0, 1.0, 1.0))
    limit = float(np.abs(truth).max())
    tolerance = example["ZERO_TOLERANCE"]
    _heatmap(axes[0], truth, list(y.columns), list(x.columns), limit,
             f"True loadings\n{int(np.count_nonzero(truth))} non-zero", True, annotate=False)
    for axis, name in zip(axes[1:], shown):
        row = counts.loc[name]
        title = (f"{name}: {int(row['n_loadings'])} kept\n"
                 f"{int(row['n_false'])} false, {int(row['n_missed'])} missed")
        _heatmap(axis, fits[name].coef_.to_numpy(), list(y.columns), list(x.columns), limit, title,
                 False, annotate=False, tolerance=tolerance)
    for axis in axes:
        axis.set_xticks(range(len(x.columns)), labels=list(x.columns), fontsize=9.5, rotation=90)
        axis.title.set_size(11)
        for boundary in range(1, len(example["GROUP_NAMES"])):
            axis.axhline(boundary * example["GROUP_SIZE"] - 0.5, color=INK_MUTED, linewidth=1.2)

    tables = {
        "group_penalties_support_counts": counts.rename_axis("penalty"),
        **{f"group_penalties_loadings_{name.lower()}": model.coef_ for name, model in fits.items()},
    }
    checks = {
        "group_penalties_fcgl_recovers_the_support": bool(
            counts.loc["FCGL", ["n_false", "n_missed"]].sum() == 0),
        "group_penalties_lasso_misses_small_loadings": bool(
            counts.loc["LASSO", "n_missed"] > 0 and counts.loc["LASSO", "n_false"] == 0),
        "group_penalties_row_penalty_keeps_dense_rows": bool(
            counts.loc["HCGL", "n_loadings"] >= truth.size - truth.shape[0]),
    }
    return figure, tables, checks


def _prior_targets(spec: dict) -> tuple[Figure, dict, dict]:
    """Where each prior-centre policy sends the two loadings as the penalty grows."""
    example = load_example(spec["example"])
    verify_parameters(example, spec["parameters"])
    x, y = example["make_panel"]()
    paths = example["loading_paths"](x, y)
    truth = example["TRUE_BETA"]
    policies = list(example["POLICIES"])
    grid = example["REG_LAMBDAS"]

    figure = Figure(figsize=(11.0, 4.6), facecolor=SURFACE, layout="constrained")
    axes = figure.subplots(1, 2)
    for axis, factor, true_value in zip(axes, example["FACTORS"], truth):
        axis.axhline(true_value, color=INK_MUTED, linestyle="--", linewidth=1.2)
        axis.annotate(f"true loading {true_value:.2f}", xy=(grid[0], true_value),
                      xytext=(-4, -16), textcoords="offset points", ha="right", fontsize=11,
                      color=INK_MUTED)
        ends = {}
        for name, colour, marker in zip(policies, SERIES, MARKERS):
            path = paths[paths["policy"] == name]
            axis.plot(path["reg_lambda"], path[factor], color=colour, marker=marker, **LINE)
            ends.setdefault(round(float(path[factor].iloc[-1]), 2), []).append(name)
        # Direct labels at the large-penalty end, where each fit has reached its centre.
        for level, names in ends.items():
            axis.annotate(" and ".join(names), xy=(grid[-1], level), xytext=(4, 7),
                          textcoords="offset points", ha="left", fontsize=11, color=INK)
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_ylim(-0.12, 1.05)
        axis.set_xlabel("reg_lambda (the penalty falls to the right)")
        axis.set_title(f"{factor.capitalize()} loading", fontsize=12.5, loc="left")
        style_axis(axis)
    axes[0].set_ylabel("Fitted loading of the inflation-linked response")

    signs = {name: example["fit"](x, y, 1e-4, **policy).derived_signs_.loc["linker"]
             for name, policy in example["POLICIES"].items()}
    tables = {
        "prior_targets_paths": paths.set_index(["policy", "reg_lambda"]),
        "prior_targets_signs": pd.DataFrame(signs).T.rename_axis("policy"),
    }
    large = paths[paths["reg_lambda"] == grid[-1]].set_index("policy")
    small = paths[paths["reg_lambda"] == grid[0]].set_index("policy")
    checks = {
        "prior_targets_zero_centre_returns_zero": bool(
            np.allclose(large.loc["zero centre", ["rates", "inflation"]], 0.0, atol=1e-3)),
        "prior_targets_detected_signs_hold_inflation_at_zero": bool(
            np.allclose(small.loc[["zero centre", "automatic centre"], "inflation"], 0.0,
                        atol=1e-4)),
        "prior_targets_joint_centre_restores_positive_inflation": bool(
            (paths.loc[paths["policy"] == "joint centre", "inflation"] > 0.3).all()),
        "prior_targets_joint_centre_flips_the_detected_sign": bool(
            signs["joint centre"]["inflation"] == 1.0
            and signs["automatic centre"]["inflation"] == -1.0),
    }
    return figure, tables, checks


def _ewma_weighting(spec: dict) -> tuple[Figure, dict, dict]:
    """Weight profiles of three spans, and shrinkage by history length under two normalisations."""
    example = load_example(spec["example"])
    verify_parameters(example, spec["parameters"])
    x, y = example["make_ragged_panel"]()
    profiles = pd.concat([example["weight_profile"](span) for span in example["SPANS"]], axis=1)
    tables = {
        "equal weights": example["shrinkage_by_history"](x, y, span=None),
        f"span {example['FIT_SPAN']}": example["shrinkage_by_history"](x, y,
                                                                      span=example["FIT_SPAN"]),
    }

    figure = Figure(figsize=(11.0, 4.6), facecolor=SURFACE, layout="constrained")
    left, right = figure.subplots(1, 2)
    for (name, profile), colour, marker, span in zip(profiles.items(), SERIES, MARKERS,
                                                   example["SPANS"]):
        half_life = np.log(0.5) / np.log(example["decay"](span))
        left.plot(profile.index, profile.to_numpy(), color=colour, linewidth=2,
                  label=f"{name}: half-life {half_life:.1f} months")
        left.plot([half_life], [0.5], linestyle="none", color=colour, marker=marker,
                  markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
    left.axhline(0.5, color=INK_MUTED, linestyle="--", linewidth=1.0)
    left.set_xlim(0, 120)
    left.set_ylim(0.0, 1.05)
    left.set_xlabel("Months before the last observation")
    left.set_ylabel("Observation weight")
    left.set_title("EWMA weights: the span is not a window", fontsize=12.5, loc="left")
    left.legend(frameon=False, fontsize=11, labelcolor=INK, loc="upper right")

    histories = example["HISTORIES"]
    styles = {"equal weights": "-", f"span {example['FIT_SPAN']}": "--"}
    for table_name, table in tables.items():
        for convention, colour, marker in zip(("sample", "weight_sum"), SERIES, MARKERS):
            values = table.loc[histories, convention]
            right.plot(histories, values.to_numpy(), color=colour, marker=marker,
                       linestyle=styles[table_name], label=f"{convention}, {table_name}",
                       **LINE)
    right.set_xscale("log")
    right.set_xticks(histories, labels=[str(h) for h in histories])
    right.minorticks_off()
    right.invert_xaxis()
    right.set_xlim(300, 30)
    right.set_ylim(0.0, 0.42)
    right.set_xlabel("Months of history of the response (fewer to the right)")
    right.set_ylabel("Shrinkage of the loading against least squares")
    right.set_title("Same loading, ragged histories, one panel", fontsize=12.5, loc="left")
    right.legend(frameon=False, fontsize=11, labelcolor=INK, loc="upper left")
    for axis in (left, right):
        style_axis(axis)

    flat, ewma = tables["equal weights"], tables[f"span {example['FIT_SPAN']}"]
    out_tables = {
        "ewma_weight_profiles": profiles,
        "ewma_shrinkage_equal_weights": flat,
        "ewma_shrinkage_span": ewma,
    }
    checks = {
        "ewma_closed_form_shrinkage_sample": bool(np.allclose(
            pd.concat([flat, ewma])["sample"], pd.concat([flat, ewma])["sample_closed_form"],
            atol=1e-4)),
        "ewma_closed_form_shrinkage_weight_sum": bool(np.allclose(
            pd.concat([flat, ewma])["weight_sum"],
            pd.concat([flat, ewma])["weight_sum_closed_form"], atol=1e-4)),
        "ewma_sample_shrinks_short_histories_harder": bool(
            flat.loc[36, "sample"] > 5 * flat.loc[240, "sample"]),
        "ewma_weight_sum_is_indifferent_to_history": bool(
            flat.loc[36, "weight_sum"] < 1.25 * flat.loc[240, "weight_sum"]),
    }
    return figure, out_tables, checks


def _pooled_signs(spec: dict) -> tuple[Figure, dict, dict]:
    """Recovery and false-sign rates of per-response and pooled signs across the gate."""
    from scipy import stats

    example = load_example(spec["example"])
    verify_parameters(example, spec["parameters"])
    sweep = example["gate_sweep"]()
    taus = np.array(example["TAUS"])
    labels = {"per response": "per response", "pooled": "pooled in known clusters"}

    figure = Figure(figsize=(11.0, 4.6), facecolor=SURFACE, layout="constrained")
    left, right = figure.subplots(1, 2)
    for (key, label), colour, marker, offset in zip(labels.items(), SERIES, MARKERS, (-20, 10)):
        rates = sweep.loc[key]
        left.plot(taus, rates["recovery"].to_numpy(), color=colour, marker=marker, **LINE)
        left.annotate(label, xy=(taus[-1], rates["recovery"].iloc[-1]), xytext=(-4, offset),
                      textcoords="offset points", ha="right", fontsize=11, color=INK)
        right.plot(taus, rates["false_sign"].to_numpy(), color=colour, marker=marker,
                   label=label, **LINE)
    right.plot(taus, 2.0 * stats.norm.sf(taus), color=INK_MUTED, linestyle="--", linewidth=1.2,
               label="2 Phi(-tau), one response under the null")
    right.legend(frameon=False, fontsize=11, labelcolor=INK, loc="upper right")
    for axis in (left, right):
        axis.axvline(0.75, color=GRID, linewidth=2.0, zorder=0)
        axis.set_xlabel("Gate threshold tau (package default 0.75)")
        axis.set_ylim(0.0, 1.02)
        style_axis(axis)
    left.set_ylabel("Share of active cells with the true sign")
    left.set_title("Signals recovered", fontsize=12.5, loc="left")
    right.set_ylabel("Share of null cells given a sign")
    right.set_title("Null cells signed", fontsize=12.5, loc="left")

    tables = {"pooled_sign_rates": sweep}
    per, pooled = sweep.loc["per response"], sweep.loc["pooled"]
    checks = {
        "pooled_signs_recover_more_at_default": bool(
            pooled.loc[0.75, "recovery"] > per.loc[0.75, "recovery"] + 0.2),
        "pooled_signs_never_flip": bool(pooled["flip"].max() < 0.01),
        "per_response_null_retention_matches_normal_reference": bool(np.allclose(
            per["false_sign"].to_numpy(), 2.0 * stats.norm.sf(taus), atol=0.03)),
    }
    return figure, tables, checks


def _adaptive_weights(spec: dict) -> tuple[Figure, dict, dict]:
    """Weight curves, and fitted loadings by true loading for plain and adaptive fits."""
    example = load_example(spec["example"])
    verify_parameters(example, spec["parameters"])
    x, y = example["make_panel"]()
    fits = {"plain LASSO": example["fit"](x, y, adaptive=False),
            "adaptive weights": example["fit"](x, y, adaptive=True)}
    grid = np.linspace(0.0, 2.0, 401)[1:]
    curves = {
        f"gamma {gamma}, floor {floor}": example["weight_curve"](grid, gamma, floor)
        for gamma, floor in ((0.5, example["FLOOR"]), (1.0, example["FLOOR"]),
                             (2.0, example["FLOOR"]), (1.0, 1e-3))
    }

    figure = Figure(figsize=(11.0, 4.6), facecolor=SURFACE, layout="constrained")
    left, right = figure.subplots(1, 2)
    styles = (("-", SERIES[0]), ("-", SERIES[1]), ("-", SERIES[2]), ("--", INK_MUTED))
    for (label, curve), (style, colour) in zip(curves.items(), styles):
        left.plot(grid, curve, linestyle=style, color=colour, linewidth=2, label=label)
    left.axhline(1.0, color=GRID, linewidth=1.5, zorder=0)
    left.set_ylim(0.0, 4.2)
    left.set_xlim(0.0, 2.0)
    left.set_xlabel("Absolute pooled univariate slope |b|")
    left.set_ylabel("Adaptive weight on the cell's penalty")
    left.set_title("W = 1 / max(|b|, floor) ** gamma", fontsize=12.5, loc="left")
    left.legend(frameon=False, fontsize=11, labelcolor=INK, loc="upper right")

    truth = example["TRUE_BETA"]
    classes = [0.0, 0.3, 1.0]
    for offset, ((label, model), colour, marker) in zip((-0.12, 0.12), zip(fits.items(), SERIES,
                                                                            MARKERS)):
        beta = model.coef_.to_numpy()
        for position, value in enumerate(classes):
            cells = beta[truth == value]
            jitter = np.linspace(-0.06, 0.06, cells.size)
            right.plot(position + offset + jitter, cells, linestyle="none", color=colour,
                       marker=marker, markersize=7, markeredgecolor=SURFACE,
                       markeredgewidth=1.2, label=label if position == 0 else None)
    for position, value in enumerate(classes):
        right.hlines(value, position - 0.3, position + 0.3, color=INK_MUTED, linestyle="--",
                     linewidth=1.2)
    right.set_xticks(range(len(classes)), labels=[f"true {value}" for value in classes])
    right.set_xlim(-0.5, len(classes) - 0.5)
    right.set_ylabel("Fitted loading")
    right.set_title(f"36 loadings at reg_lambda = {example['REG_LAMBDA']:g}", fontsize=12.5,
                    loc="left")
    right.legend(frameon=False, fontsize=11, labelcolor=INK, loc="upper left")
    for axis in (left, right):
        style_axis(axis)
    right.grid(axis="x", visible=False)

    summary = pd.DataFrame({label: example["error_summary"](model)
                            for label, model in fits.items()})
    tables = {"adaptive_weights_summary": summary.rename_axis("measure"),
              "adaptive_weights_cells": fits["adaptive weights"].sign_penalty_weights_}
    checks = {
        "adaptive_weights_remove_loadings_on_zeros": bool(
            summary.loc["kept_on_zeros", "adaptive weights"]
            < summary.loc["kept_on_zeros", "plain LASSO"]),
        "adaptive_weights_bounded_by_floor": bool(
            fits["adaptive weights"].sign_penalty_weights_.to_numpy().max()
            <= 1.0 / example["FLOOR"] + 1e-12),
    }
    return figure, tables, checks


def _cooperative(spec: dict) -> tuple[Figure, dict, dict]:
    """Penalty level sets, and the rogue member's loading along the penalty grid."""
    example = load_example(spec["example"])
    verify_parameters(example, spec["parameters"])
    x, y = example["make_panel"]()
    path = example["rogue_path"](x, y)

    figure = Figure(figsize=(11.0, 4.6), facecolor=SURFACE, layout="constrained")
    left, right = figure.subplots(1, 2, width_ratios=(1.0, 1.35))
    angles = np.linspace(0.0, 2.0 * np.pi, 721)
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    group_radius = 1.0 / np.linalg.norm(directions, axis=1)
    coop_radius = 1.0 / np.array([example["cooperative_penalty"](d) for d in directions])
    left.plot(group_radius * directions[:, 0], group_radius * directions[:, 1], color=SERIES[0],
              linewidth=2, label="group LASSO: norm of the block")
    left.plot(coop_radius * directions[:, 0], coop_radius * directions[:, 1], color=SERIES[1],
              linewidth=2, label="cooperative LASSO: norm of each sign part")
    left.axhline(0.0, color=GRID, linewidth=1.0, zorder=0)
    left.axvline(0.0, color=GRID, linewidth=1.0, zorder=0)
    left.set_aspect("equal")
    left.set_xlim(-1.35, 1.35)
    left.set_ylim(-1.35, 1.75)
    left.set_xlabel("Loading of member 1")
    left.set_ylabel("Loading of member 2")
    left.set_title("Loadings with penalty 1", fontsize=12.5, loc="left")
    left.legend(frameon=False, fontsize=10.5, labelcolor=INK, loc="upper center")

    styles = {"group LASSO": (SERIES[0], MARKERS[0]), "cooperative LASSO": (SERIES[1], MARKERS[1]),
              "hard pooled sign": (SERIES[2], MARKERS[2])}
    for name, (colour, marker) in styles.items():
        rows = path[path["penalty"] == name]
        right.plot(rows["reg_lambda"], rows["rogue"], color=colour, marker=marker, label=name,
                   **LINE)
    least_squares = float((x.T @ y / len(x)).loc["f1", "asset_4"])
    right.axhline(least_squares, color=INK_MUTED, linestyle="--", linewidth=1.2)
    right.annotate(f"least squares {least_squares:.2f}", xy=(0.8, least_squares), xytext=(0, 6),
                   textcoords="offset points", ha="left", fontsize=11, color=INK_MUTED)
    right.axhline(0.0, color=GRID, linewidth=1.0, zorder=0)
    right.set_ylim(least_squares - 0.03, 0.03)
    right.invert_xaxis()
    right.set_xlabel("reg_lambda, linear scale (the penalty falls to the right)")
    right.set_ylabel("Loading of the rogue member on factor 1")
    right.set_title("A member against its cluster's sign", fontsize=12.5, loc="left")
    right.legend(frameon=False, fontsize=11, labelcolor=INK, loc="upper left",
                 bbox_to_anchor=(0.02, 0.82))
    for axis in (left, right):
        style_axis(axis)

    tables = {"cooperative_rogue_path": path.set_index(["penalty", "reg_lambda"])}
    coop = path[path["penalty"] == "cooperative LASSO"].set_index("reg_lambda")["rogue"]
    group = path[path["penalty"] == "group LASSO"].set_index("reg_lambda")["rogue"]
    checks = {
        "cooperative_shrinks_the_rogue_faster": bool((coop >= group - 1e-6).all()),
        "cooperative_rogue_reaches_zero": bool(abs(coop.iloc[-1]) < 1e-4),
        "hard_sign_holds_the_rogue_at_zero": bool(
            path.loc[path["penalty"] == "hard pooled sign", "rogue"].abs().max() < 1e-4),
    }
    return figure, tables, checks


def _unilasso(spec: dict) -> tuple[Figure, dict, dict]:
    """Final loadings against univariate slopes, and noise loadings kept along the grid."""
    example = load_example(spec["example"])
    verify_parameters(example, spec["parameters"])
    x, y = example["make_panel"]()
    slopes = example["univariate_slopes"](x, y)
    fits = {"UniLasso": example["fit_unilasso"](x, y),
            "signs free": example["fit_unilasso"](x, y, non_negative=False)}
    path = example["settings_path"](x, y)
    styles = {"UniLasso": (SERIES[0], MARKERS[0]), "signs free": (SERIES[1], MARKERS[1]),
              "in-sample fits": (SERIES[2], MARKERS[2])}

    figure = Figure(figsize=(11.0, 4.6), facecolor=SURFACE, layout="constrained")
    left, right = figure.subplots(1, 2)
    limit = 1.25
    for corner in ((0.0, -limit), (-limit, 0.0)):
        left.add_patch(Rectangle(corner, limit, limit, facecolor=GRID, alpha=0.45, linewidth=0,
                                 zorder=0))
    left.text(0.62, -1.12, "sign reversed", ha="center", fontsize=10.5, color=INK_MUTED)
    left.text(-0.62, 1.05, "sign reversed", ha="center", fontsize=10.5, color=INK_MUTED)
    suppressor = np.zeros(slopes.shape, dtype=bool)
    suppressor[:, list(slopes.columns).index("f2")] = True
    for label, coef in fits.items():
        colour, marker = styles[label]
        values = coef.to_numpy()
        left.plot(slopes.to_numpy()[~suppressor], values[~suppressor], linestyle="none",
                  color=colour, marker=marker, markersize=7, markeredgecolor=SURFACE,
                  markeredgewidth=1.2, label=label)
        left.plot(slopes.to_numpy()[suppressor], values[suppressor], linestyle="none",
                  color=colour, marker=marker, markersize=9, markeredgecolor=INK,
                  markeredgewidth=1.4)
    left.plot([], [], linestyle="none", marker="o", markersize=9, markerfacecolor="none",
              markeredgecolor=INK, markeredgewidth=1.4, label="suppressor f2 (true -0.4)")
    left.axhline(0.0, color=GRID, linewidth=1.0, zorder=0)
    left.axvline(0.0, color=GRID, linewidth=1.0, zorder=0)
    left.set_xlim(-limit, limit)
    left.set_ylim(-limit, limit)
    left.set_xlabel("Stage-one univariate slope")
    left.set_ylabel("Final loading")
    left.set_title(f"72 loadings at reg_lambda = {example['REG_LAMBDA']:g}", fontsize=12.5,
                   loc="left")
    left.legend(frameon=False, fontsize=10.5, labelcolor=INK, loc="lower left")

    for name, (colour, marker) in styles.items():
        rows = path[path["setting"] == name]
        right.plot(rows["reg_lambda"], rows["noise_kept"], color=colour, marker=marker,
                   label=name, **LINE)
    right.axvline(example["REG_LAMBDA"], color=INK_MUTED, linestyle=":", linewidth=1.2)
    right.set_xscale("log")
    right.invert_xaxis()
    right.set_ylim(-1, 37)
    right.set_xlabel("reg_lambda, log scale (the penalty falls to the right)")
    right.set_ylabel("Loadings kept on the 36 noise cells")
    right.set_title("Noise loadings along the penalty grid", fontsize=12.5, loc="left")
    right.legend(frameon=False, fontsize=11, labelcolor=INK, loc="upper left")
    for axis in (left, right):
        style_axis(axis)

    tables = {"unilasso_settings_path": path.set_index(["setting", "reg_lambda"]),
              "unilasso_final_loadings": pd.concat(fits, names=["setting", "response"])}
    unilasso = fits["UniLasso"].to_numpy()
    checks = {
        "unilasso_keeps_univariate_signs": bool((unilasso * slopes.to_numpy() >= -1e-9).all()),
        "unilasso_drops_the_suppressor": bool(np.abs(unilasso[suppressor]).max() < 1e-6),
        "free_signs_reverse_the_suppressor": bool(
            (fits["signs free"].to_numpy()[suppressor] < 0).all()),
    }
    return figure, tables, checks


def _penalty_selection(spec: dict) -> tuple[Figure, dict, dict]:
    """Held-out R-squared and residual sphericity along the grid, with both selections."""
    example = load_example(spec["example"])
    verify_parameters(example, spec["parameters"])
    curves = {name: example["selection_curves"](*example["make_panel"](omitted))
              for name, omitted in (("complete panel", False), ("omitted factor", True))}

    figure = Figure(figsize=(11.0, 4.6), facecolor=SURFACE, layout="constrained")
    left, right = figure.subplots(1, 2)
    for (name, table), colour, marker in zip(curves.items(), SERIES, MARKERS):
        left.plot(table.index, table["held_out_r2"], color=colour, marker=marker, label=name,
                  **LINE)
        chosen = table.attrs["r2_lambda"]
        left.plot([chosen], [table.loc[chosen, "held_out_r2"]], linestyle="none", marker=marker,
                  markersize=15, markerfacecolor="none", markeredgecolor=INK, markeredgewidth=1.6)
        right.plot(table.index, table["sphericity"], color=colour, marker=marker, label=name,
                   **LINE)
        chosen = table.attrs["diagonality_lambda"]
        right.plot([chosen], [table.loc[chosen, "sphericity"]], linestyle="none", marker=marker,
                   markersize=15, markerfacecolor="none", markeredgecolor=INK,
                   markeredgewidth=1.6)
    threshold = float(curves["complete panel"]["threshold"].iloc[0])
    right.axhline(threshold, color=INK_MUTED, linestyle="--", linewidth=1.2)
    right.annotate(f"chi-square threshold {threshold:.0f}", xy=(1e-6, threshold), xytext=(0, -16),
                   textcoords="offset points", ha="right", fontsize=11, color=INK_MUTED)
    left.plot([], [], linestyle="none", marker="o", markersize=12, markerfacecolor="none",
              markeredgecolor=INK, markeredgewidth=1.6, label="selected penalty")
    left.set_ylim(-0.1, 0.9)
    left.set_ylabel("Held-out R-squared, mean over folds")
    left.set_title("LassoModelCV: best held-out fit", fontsize=12.5, loc="left")
    left.legend(frameon=False, fontsize=11, labelcolor=INK, loc="lower right")
    right.set_yscale("log")
    right.set_ylim(95.0, 1000.0)
    right.set_yticks([100, 150, 200, 300, 500, 800], labels=["100", "150", "200", "300", "500",
                                                             "800"])
    right.yaxis.set_minor_formatter(NullFormatter())
    right.set_ylabel("Held-out residual sphericity, mean over folds")
    right.set_title("LassoModelDiagonalityCV: sparsest passing", fontsize=12.5, loc="left")
    right.legend(frameon=False, fontsize=11, labelcolor=INK, loc="center right")
    for axis in (left, right):
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_xlabel("reg_lambda, log scale (the penalty falls to the right)")
        style_axis(axis)

    tables = {
        "penalty_selection_curves": pd.concat(
            {name: table.drop(columns="passes") for name, table in curves.items()},
            names=["panel", "reg_lambda"]),
        "penalty_selection_choices": pd.DataFrame({
            name: {"held_out_r2": table.attrs["r2_lambda"],
                   "diagonality": table.attrs["diagonality_lambda"],
                   "diagonality_passed": float(table.attrs["passed"])}
            for name, table in curves.items()}).T.rename_axis("panel"),
    }
    complete, omitted = curves["complete panel"], curves["omitted factor"]
    checks = {
        "diagonality_is_sparser_on_the_complete_panel": bool(
            complete.attrs["diagonality_lambda"] > complete.attrs["r2_lambda"]),
        "no_penalty_passes_with_an_omitted_factor": bool(not omitted["passes"].any()),
        "missing_component_is_the_hidden_block": bool(
            set(omitted.attrs["missing"]["series"]) == {f"y0{k}" for k in range(1, 7)}),
    }
    return figure, tables, checks


EXHIBITS = {
    "quickstart_workflow.png": _quickstart,
    "sparse_factor_model_path.png": _sparse_factor_model,
    "sign_constraints_and_priors_error.png": _sign_constraints_and_priors,
    "group_penalties_selection.png": _group_penalties,
    "prior_targets_paths.png": _prior_targets,
    "ewma_weighting_shrinkage.png": _ewma_weighting,
    "pooled_sign_recovery.png": _pooled_signs,
    "adaptive_penalty_weights.png": _adaptive_weights,
    "cooperative_lasso_geometry.png": _cooperative,
    "unilasso_two_stage.png": _unilasso,
    "penalty_selection_paths.png": _penalty_selection,
}


def produce(configuration: dict) -> dict:
    """Return figures, supporting tables, the applied configuration and numerical checks.

    Parameters
    ----------
    configuration : dict
        The producer's ``configuration`` record from ``registry.json``; its ``exhibits`` mapping
        names every PNG this module draws and the example and parameters behind it.

    Returns
    -------
    dict
        Keys ``figures`` (PNG basename to Matplotlib ``Figure``), ``tables`` (name to nonempty
        ``DataFrame``), ``configuration`` (echoed once verified against what ran) and ``checks``
        (name to ``True``).
    """
    if set(configuration["exhibits"]) != set(EXHIBITS):
        declared = sorted(configuration["exhibits"])
        raise ValueError(f"Registry exhibits {declared} != {sorted(EXHIBITS)}")
    figures, tables, checks = {}, {}, {}
    for basename, draw in EXHIBITS.items():
        figure, exhibit_tables, exhibit_checks = draw(configuration["exhibits"][basename])
        figures[basename] = figure
        tables.update(exhibit_tables)
        checks.update(exhibit_checks)
    return {"figures": figures, "tables": tables, "configuration": configuration, "checks": checks}
