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

Each calculation is the worked example of its article, loaded from ``examples/docs/``. Nothing is
re-implemented here: this module calls the example's functions, checks that the registry
configuration describes what actually ran, and draws.
"""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

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

MARKERS = ("o", "s")
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


EXHIBITS = {
    "quickstart_workflow.png": _quickstart,
    "sparse_factor_model_path.png": _sparse_factor_model,
    "sign_constraints_and_priors_error.png": _sign_constraints_and_priors,
    "group_penalties_selection.png": _group_penalties,
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
