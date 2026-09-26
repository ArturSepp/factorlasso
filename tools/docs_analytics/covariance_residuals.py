"""Producer family for covariance and residual exhibits.

Exhibits
--------
``residual_diagnostics_spectrum.png`` for ``docs/residual_diagnostics.md``: the eigenvalues of the
residual correlation matrix against the Marchenko-Pastur edge, and the sphericity statistic
against its chi-square threshold along a penalty grid, for a complete factor set and for the same
panel with one factor withheld.

``factor_covariance_assembly_history.png`` for ``docs/factor_covariance_assembly.md``: the error of
the assembled factor covariance and of the sample covariance against the population covariance,
and the realised volatility of the minimum-variance portfolio each implies, by history length.

``nowcast_alpha_tracking.png`` for ``docs/residual_alpha_nowcasting.md``: the nowcast alpha and
the economic intercept of a response whose alpha shifts, against its true alpha, over expanding
fits, and the error of each alpha estimate by response group against the EWMA noise floor.

``residual_correlation_blocks.png`` for ``docs/empirical_residual_correlation.md``: the residual
covariance of eight responses in correlation units at three retentions of the estimated residual
correlation, with the residual volatility of an equal-weight block portfolio.

``cma_decomposition.png`` for ``docs/app_portfolio_risk_models.md``: the capital market assumption
of each sleeve split into the risk-free rate, factor premia and declared adjustment, and its
variance split into factor contributions and residual variance, from one loading matrix.

Each calculation is the worked example of its article, loaded from ``examples/docs/``. Nothing is
re-implemented here: this module calls the example's functions, checks that the registry
configuration describes what actually ran, and draws.
"""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter, ScalarFormatter
from scipy import stats

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


def _residual_diagnostics(configuration: dict) -> tuple[Figure, dict, dict]:
    """Compute the worked example once and draw its two-panel exhibit."""
    example = load_example(configuration["example"])
    verify_parameters(example, configuration["parameters"])

    x, y = example["make_panel"]()
    x_short = x.drop(columns=example["WITHHELD_FACTOR"])
    labels = ("All four factors", f"'{example['WITHHELD_FACTOR']}' withheld")
    fits = {labels[0]: example["fit_and_diagnose"](x, y),
            labels[1]: example["fit_and_diagnose"](x_short, y)}
    paths = {labels[0]: example["penalty_path"](x, y),
             labels[1]: example["penalty_path"](x_short, y)}
    diagnostics = {label: fit[3] for label, fit in fits.items()}
    eigenvalues = pd.DataFrame({
        label: np.sort(np.linalg.eigvalsh(diag.correlation.to_numpy()))[::-1]
        for label, diag in diagnostics.items()
    })
    eigenvalues.index = pd.Index(range(1, len(eigenvalues) + 1), name="rank")
    complete, withheld = diagnostics[labels[0]], diagnostics[labels[1]]
    residuals = fits[labels[0]][2].to_numpy()
    corr = np.corrcoef(residuals, rowvar=False)
    upper = np.triu_indices(corr.shape[0], 1)

    figure = Figure(figsize=(11.0, 4.6), facecolor=SURFACE, layout="constrained")
    left, right = figure.subplots(1, 2)

    for label, colour, marker in zip(labels, SERIES, MARKERS):
        left.plot(eigenvalues.index, eigenvalues[label], color=colour, marker=marker,
                  markersize=7, linewidth=2, markeredgecolor=SURFACE, markeredgewidth=1.5,
                  label=label)
    left.axhline(complete.mp_edge, color=INK_MUTED, linestyle="--", linewidth=1.2)
    left.annotate(f"Marchenko-Pastur edge {complete.mp_edge:.2f}",
                  xy=(eigenvalues.index[-1], complete.mp_edge), xytext=(0, 5),
                  textcoords="offset points", ha="right", fontsize=11, color=INK_MUTED)
    left.annotate(f"{withheld.top_eigenvalue:.2f}", xy=(1, withheld.top_eigenvalue),
                  xytext=(9, -3), textcoords="offset points", fontsize=11, color=INK)
    left.set_xlabel("Eigenvalue rank")
    left.set_ylabel("Eigenvalue of the residual correlation matrix")
    left.set_title("Residual spectrum at reg_lambda = 1e-4", fontsize=12.5, loc="left")
    left.set_xticks(list(eigenvalues.index))
    left.legend(frameon=False, fontsize=11, labelcolor=INK)

    for label, colour, marker, offset in zip(labels, SERIES, MARKERS, (-18, 10)):
        path = paths[label]
        right.plot(path.index, path["sphericity"], color=colour, marker=marker, markersize=7,
                   linewidth=2, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        # Direct labels at the line ends replace a legend that would sit on the threshold.
        right.annotate(f"{label}: {path['sphericity'].iloc[-1]:.0f}",
                       xy=(path.index[-1], path["sphericity"].iloc[-1]), xytext=(0, offset),
                       textcoords="offset points", ha="right", fontsize=11, color=INK)
    right.axhline(complete.threshold, color=INK_MUTED, linestyle="--", linewidth=1.2)
    right.annotate(f"5% chi-square threshold {complete.threshold:.1f}",
                   xy=(paths[labels[0]].index[-1], complete.threshold), xytext=(0, 5),
                   textcoords="offset points", ha="right", fontsize=11, color=INK_MUTED)
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_ylim(20.0, 1500.0)
    right.set_yticks([30, 100, 300, 1000])
    right.yaxis.set_major_formatter(ScalarFormatter())
    right.yaxis.set_minor_formatter(NullFormatter())
    right.invert_xaxis()
    right.set_xlabel("reg_lambda (the penalty falls to the right)")
    right.set_ylabel("Sphericity statistic S (log scale)")
    right.set_title("In-sample sphericity along the penalty grid", fontsize=12.5, loc="left")
    for axis in (left, right):
        style_axis(axis)

    rates = example["null_rejection_rates"](nu=complete.nu, n_series=y.shape[1])
    tables = {
        "residual_eigenvalues": eigenvalues,
        "residual_diagnostics": pd.DataFrame(
            {label: diag.to_dict() for label, diag in diagnostics.items()}
        ).T.rename_axis("panel"),
        "penalty_path_complete": paths[labels[0]],
        "penalty_path_withheld": paths[labels[1]],
        "null_rejection_rates": (
            rates.rename("rejection_frequency").rename_axis("criterion").to_frame()
        ),
    }
    checks = {
        "complete_factor_set_passes": bool(complete.passes),
        "withheld_factor_fails": bool(not withheld.passes and withheld.n_above_edge >= 1),
        "no_penalty_passes_without_the_factor": bool(not paths[labels[1]]["passes"].any()),
        "threshold_matches_scipy_chi2": bool(np.isclose(
            complete.threshold,
            stats.chi2.ppf(1.0 - example["SIGNIFICANCE"], len(upper[0])))),
        "edge_matches_closed_form": bool(np.isclose(
            complete.mp_edge, (1.0 + np.sqrt(corr.shape[0] / complete.nu)) ** 2)),
        "sphericity_matches_direct_sum": bool(np.isclose(
            complete.sphericity, complete.nu * float(np.sum(corr[upper] ** 2)))),
        "eigenvalues_sum_to_series_count": bool(np.allclose(
            eigenvalues.sum().to_numpy(), corr.shape[0])),
        "plotted_top_eigenvalue_is_reported_value": bool(np.isclose(
            eigenvalues[labels[1]].iloc[0], withheld.top_eigenvalue)),
    }
    return figure, tables, checks


def _factor_covariance_assembly(configuration: dict) -> tuple[Figure, dict, dict]:
    """Assembled against sample covariance by history length: error and minimum-variance risk."""
    example = load_example(configuration["example"])
    verify_parameters(example, configuration["parameters"])
    study = example["history_study"]()
    truth = example["true_covariance"]()
    true_min_vol = example["minimum_variance_vol"](truth, truth)
    labels = {"factor model": "assembled factor covariance", "sample": "sample covariance"}

    figure = Figure(figsize=(11.0, 4.6), facecolor=SURFACE, layout="constrained")
    left, right = figure.subplots(1, 2)
    for (name, label), colour, marker in zip(labels.items(), SERIES, MARKERS):
        rows = study.loc[name]
        left.plot(rows.index, rows["relative_error"], color=colour, marker=marker, markersize=7,
                  linewidth=2, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        right.plot(rows.index, 100.0 * rows["min_variance_vol"], color=colour, marker=marker,
                   markersize=7, linewidth=2, markeredgecolor=SURFACE, markeredgewidth=1.5,
                   label=label)
        right.annotate(f"{100.0 * rows['min_variance_vol'].iloc[0]:.1f}",
                       xy=(rows.index[0], 100.0 * rows["min_variance_vol"].iloc[0]),
                       xytext=(8, 9), textcoords="offset points", va="center", fontsize=11,
                       color=INK)
    right.axhline(100.0 * true_min_vol, color=INK_MUTED, linestyle="--", linewidth=1.2)
    right.annotate(f"population optimum {100.0 * true_min_vol:.1f}",
                   xy=(study.loc["sample"].index[-1], 100.0 * true_min_vol), xytext=(0, -14),
                   textcoords="offset points", ha="right", fontsize=11, color=INK_MUTED)
    for axis in (left, right):
        axis.set_xticks(list(study.loc["sample"].index))
        axis.set_xlabel("Months of history (40 assets, 4 factors)")
        style_axis(axis)
    left.set_ylim(0.0, None)
    left.set_ylabel("Relative Frobenius error of the covariance")
    left.set_title("Error against the population covariance", fontsize=12.5, loc="left")
    left.legend(frameon=False, fontsize=11, labelcolor=INK)
    right.set_ylim(0.0, 1.12 * 100.0 * float(study["min_variance_vol"].max()))
    right.set_ylabel("Annualised volatility, % (under the population)")
    right.set_title("Realised risk of the minimum-variance portfolio", fontsize=12.5, loc="left")

    model_rows, sample_rows = study.loc["factor model"], study.loc["sample"]
    tables = {"factor_covariance_assembly_history": study.reset_index().set_index("n_obs")}
    checks = {
        "assembly_error_below_sample_at_every_history": bool(
            (model_rows["relative_error"] < sample_rows["relative_error"]).all()),
        "assembly_better_conditioned_at_every_history": bool(
            (model_rows["condition_number"] < sample_rows["condition_number"]).all()),
        "assembly_realised_risk_between_optimum_and_sample": bool(
            ((model_rows["min_variance_vol"] > true_min_vol)
             & (model_rows["min_variance_vol"] < sample_rows["min_variance_vol"])).all()),
        "assembly_error_falls_with_history": bool(
            model_rows["relative_error"].is_monotonic_decreasing),
    }
    return figure, tables, checks


def _residual_partition_share(configuration: dict) -> tuple[Figure, dict, dict]:
    """Adjusted partition share of the residuals: the carrier partition against a shuffled one."""
    example = load_example(configuration["example"])
    verify_parameters(example, configuration["parameters"])

    x, y = example["make_panel"]()
    residuals = example["fit_and_diagnose"](x, y)[2]
    residuals_short = example["fit_and_diagnose"](
        x.drop(columns=example["WITHHELD_FACTOR"]), y)[2]
    carriers = [f"asset_{i + 1}" for i in np.flatnonzero(example["TRUE_BETA"][:, 3])]
    labels = pd.Series(np.where(y.columns.isin(carriers), "carriers", "others"), index=y.columns)
    shares = example["partition_shares"](
        {"complete": residuals, "withheld": residuals_short}, labels)
    summary = pd.DataFrame({
        "mean": shares.mean(), "p10": shares.quantile(0.10), "p90": shares.quantile(0.90),
    })
    names = {
        "withheld, carriers": f"'{example['WITHHELD_FACTOR']}' withheld, carrier partition",
        "withheld, shuffled": f"'{example['WITHHELD_FACTOR']}' withheld, shuffled partition",
        "complete, carriers": "All four factors, carrier partition",
        "complete, shuffled": "All four factors, shuffled partition",
    }
    order = list(names)

    figure = Figure(figsize=(11.0, 3.9), facecolor=SURFACE, layout="constrained")
    axis = figure.subplots(1, 1)
    positions = np.arange(len(order))[::-1]
    for position, key in zip(positions, order):
        row = summary.loc[key]
        colour = SERIES[0] if key.endswith("carriers") else SERIES[1]
        marker = MARKERS[0] if key.endswith("carriers") else MARKERS[1]
        axis.hlines(position, row["p10"], row["p90"], color=colour, linewidth=3, alpha=0.45)
        axis.plot(row["mean"], position, linestyle="none", color=colour, marker=marker,
                  markersize=9, markeredgecolor=SURFACE, markeredgewidth=1.5)
        axis.annotate(f"{names[key]}: mean {row['mean']:.2f}", xy=(row["p90"], position),
                      xytext=(8, -4), textcoords="offset points", ha="left", fontsize=11,
                      color=INK)
    axis.axvline(0.0, color=INK_MUTED, linestyle="--", linewidth=1.2)
    axis.annotate("permutation floor", xy=(0.0, positions[0]), xytext=(4, 14),
                  textcoords="offset points", ha="left", fontsize=11, color=INK_MUTED)
    axis.set_yticks([])
    axis.set_ylim(-0.7, len(order) - 0.2)
    axis.set_xlim(-0.35, 1.45)
    axis.set_xticks(np.linspace(-0.2, 1.0, 7))  # the space beyond 1 carries the labels
    axis.set_xlabel("Adjusted partition share per month: mean and 10th to 90th percentile")
    axis.set_title(f"{len(shares)} monthly cross-sections of {y.shape[1]} residual series",
                   fontsize=12.5, loc="left")
    style_axis(axis)
    axis.grid(axis="y", visible=False)

    tables = {"residual_partition_share": summary.loc[order].rename_axis("panel, partition")}
    checks = {
        "partition_share_carriers_form_a_block_when_withheld": bool(
            summary.loc["withheld, carriers", "mean"] > 0.15),
        "partition_share_vanishes_with_the_complete_factor_set": bool(
            abs(summary.loc["complete, carriers", "mean"]) < 0.03),
        "partition_share_shuffled_partition_near_floor": bool(
            summary.loc[["complete, shuffled", "withheld, shuffled"], "mean"].abs().max() < 0.06),
    }
    return figure, tables, checks


def _nowcast(configuration: dict) -> tuple[Figure, dict, dict]:
    """Alpha paths for a shifting response, and alpha errors by group against the noise floor."""
    example = load_example(configuration["example"])
    verify_parameters(example, configuration["parameters"])
    x, y = example["make_panel"]()
    errors = example["alpha_errors"](x, y)
    table = example["rmse_by_group"](errors)
    path = errors[errors["response"] == "shift1"].set_index("date")
    floor = 10000 * 0.02 / np.sqrt(example["ALPHA_SPAN"])

    figure = Figure(figsize=(11.0, 4.6), facecolor=SURFACE, layout="constrained")
    left, right = figure.subplots(1, 2, width_ratios=(1.4, 1.0))
    left.plot(path.index, 10000 * path["true"], color=INK, linewidth=1.6, linestyle="--",
              label="true alpha")
    left.plot(path.index, 10000 * path["nowcast"], color=SERIES[0], linewidth=2,
              label=f"nowcast: EWMA span {example['ALPHA_SPAN']} of residuals")
    left.plot(path.index, 10000 * path["economic"], color=SERIES[1], linewidth=2,
              label="economic intercept of the fit")
    left.axhline(0.0, color=INK_MUTED, linewidth=0.8, zorder=0)
    left.set_ylabel("Alpha for the next month, bp")
    left.set_xlabel("Nowcast month")
    left.set_title("Response shift1: alpha shifts from 0 to 100 bp", fontsize=12.5, loc="left")
    left.legend(frameon=False, fontsize=10.5, labelcolor=INK, loc="upper left")

    groups = list(table.index)
    positions = np.arange(len(groups))
    for offset, (name, colour) in zip((-0.18, 0.18), (("nowcast", SERIES[0]),
                                                      ("economic", SERIES[1]))):
        right.bar(positions + offset, 10000 * table[name], width=0.34, color=colour,
                  label="nowcast alpha" if name == "nowcast" else "economic intercept")
    right.axhline(floor, color=INK_MUTED, linestyle=":", linewidth=1.4)
    right.annotate(f"EWMA noise, {floor:.0f} bp", xy=(-0.45, floor), xytext=(0, 5),
                   textcoords="offset points", ha="left", fontsize=10.5, color=INK_MUTED)
    right.set_xticks(positions, labels=[{"const": "constant 50 bp", "shift": "shift 0 to 100 bp",
                                         "zero": "zero"}[g] for g in groups])
    right.set_xlim(-0.5, len(groups) - 0.5)
    right.set_ylabel("Error of the alpha estimate, RMSE in bp")
    right.set_title("Adaptivity against noise", fontsize=12.5, loc="left")
    right.legend(frameon=False, fontsize=10.5, labelcolor=INK, loc="upper left")
    for axis in (left, right):
        style_axis(axis)
    right.grid(axis="x", visible=False)

    tables = {"nowcast_alpha_path": path[["true", "nowcast", "economic"]].rename_axis("date"),
              "nowcast_alpha_rmse": table.rename_axis("group")}
    checks = {
        "nowcast_wins_under_the_shift": bool(table.loc["shift", "nowcast"]
                                             < table.loc["shift", "economic"]),
        "economic_intercept_wins_when_alpha_is_constant": bool(
            table.loc["const", "economic"] < table.loc["const", "nowcast"]),
    }
    return figure, tables, checks


def _residual_blocks(configuration: dict) -> tuple[Figure, dict, dict]:
    """The residual block in correlation units at each retention rho."""
    example = load_example(configuration["example"])
    verify_parameters(example, configuration["parameters"])
    residuals, metadata = example["make_residuals"]()
    prepared = example["estimate"](residuals, metadata)
    blocks = example["residual_covariances"](prepared)
    portfolio = example["block_portfolio_vol"](blocks)
    names = example["NAMES"]

    figure = Figure(figsize=(11.0, 4.2), facecolor=SURFACE, layout="constrained")
    axes = figure.subplots(1, len(blocks))
    for axis, (rho, block) in zip(axes, blocks.items()):
        values = block.to_numpy()
        vol = np.sqrt(np.diag(values))
        image = axis.imshow(values / np.outer(vol, vol), cmap=loading_colormap(), vmin=-1.0,
                            vmax=1.0, interpolation="nearest")
        axis.set_xticks(range(len(names)), labels=names, fontsize=10)
        axis.set_yticks(range(len(names)), labels=names, fontsize=10)
        axis.tick_params(length=0, colors=INK_MUTED)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_title(f"rho = {rho:g}: block vol {100 * portfolio[rho]:.2f}%", fontsize=12.5,
                       loc="left")
    colorbar = figure.colorbar(image, ax=list(axes), fraction=0.025, pad=0.02)
    colorbar.set_label("Residual correlation implied by D")
    colorbar.outline.set_visible(False)

    tables = {f"residual_block_rho_{str(rho).replace('.', '_')}": block.rename_axis("response")
              for rho, block in blocks.items()}
    tables["residual_block_portfolio_vol"] = pd.Series(portfolio, name="vol").rename_axis(
        "rho").to_frame()
    checks = {
        "diagonal_is_unchanged": bool(all(np.allclose(np.diag(b.to_numpy()),
                                                      np.diag(blocks[0.0].to_numpy()))
                                          for b in blocks.values())),
        "portfolio_vol_rises_with_rho": bool(portfolio[0.0] < portfolio[0.5] < portfolio[1.0]),
    }
    return figure, tables, checks


def _cma_decomposition(configuration: dict) -> tuple[Figure, dict, dict]:
    """Expected return and variance of each sleeve, split by the same loadings."""
    example = load_example(configuration["example"])
    verify_parameters(example, configuration["parameters"])
    x, y = example["make_returns"]()
    snapshot = example["risk_model"](x, y)
    betas = snapshot.y_betas
    premia = example["PREMIA"]
    factors = list(premia.index)
    returns = pd.DataFrame({"risk-free": example["RISK_FREE"],
                            **{f: betas[f] * premia[f] for f in factors},
                            "declared adjustment": example["ADJUSTMENTS"].reindex(
                                betas.index).fillna(0.0)}, index=betas.index)
    sigma_f = snapshot.x_covar.to_numpy()
    residual = snapshot.y_variances[example["fl"].VarianceColumns.RESIDUAL_VARS.value]
    euler = betas.to_numpy() * (betas.to_numpy() @ sigma_f)
    total = euler.sum(axis=1) + residual.to_numpy()
    shares = pd.DataFrame(euler / total[:, None], index=betas.index, columns=factors)
    shares["residual"] = residual.to_numpy() / total
    colours = {"risk-free": GRID, factors[0]: SERIES[0], factors[1]: SERIES[1],
               factors[2]: SERIES[2], "declared adjustment": INK_MUTED, "residual": INK_MUTED}

    figure = Figure(figsize=(11.0, 4.8), facecolor=SURFACE, layout="constrained")
    left, right = figure.subplots(1, 2, sharey=True)
    rows = np.arange(len(betas.index))[::-1]
    for axis, table, scale in ((left, returns, 100.0), (right, shares, 100.0)):
        start = np.zeros(len(table))
        for name in table.columns:
            values = scale * table[name].to_numpy()
            axis.barh(rows, values, left=start, height=0.62, color=colours[name],
                      edgecolor=SURFACE, linewidth=1.0, label=name)
            start = start + values
        style_axis(axis)
        axis.grid(axis="y", visible=False)
    left.set_yticks(rows, labels=list(betas.index))
    left.tick_params(axis="y", length=0)
    left.set_xlabel("Capital market assumption, % per year")
    left.set_title("Expected return: r_f + beta' lambda + adjustment", fontsize=12.5, loc="left")
    right.set_xlim(0.0, 100.0)
    right.set_xlabel("Share of variance, %")
    right.set_title("Risk: beta Sigma_F beta' + D", fontsize=12.5, loc="left")
    handles, labels = left.get_legend_handles_labels()
    residual_handle = right.get_legend_handles_labels()[0][-1]
    figure.legend(handles=handles + [residual_handle], labels=labels + ["residual variance"],
                  loc="outside lower center", ncol=6, frameon=False, fontsize=10.5,
                  labelcolor=INK)

    tables = {"cma_components": returns.rename_axis("sleeve"),
              "variance_shares": shares.rename_axis("sleeve")}
    checks = {
        "components_sum_to_the_cma": bool(np.allclose(
            returns.sum(axis=1), example["capital_market_assumptions"](betas))),
        "variance_shares_sum_to_one": bool(np.allclose(shares.sum(axis=1), 1.0)),
    }
    return figure, tables, checks


EXHIBITS = {
    "residual_diagnostics_spectrum.png": _residual_diagnostics,
    "residual_partition_share.png": _residual_partition_share,
    "factor_covariance_assembly_history.png": _factor_covariance_assembly,
    "nowcast_alpha_tracking.png": _nowcast,
    "residual_correlation_blocks.png": _residual_blocks,
    "cma_decomposition.png": _cma_decomposition,
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
