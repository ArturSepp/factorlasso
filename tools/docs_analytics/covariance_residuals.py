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
    INK,
    INK_MUTED,
    SERIES,
    SURFACE,
    load_example,
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


EXHIBITS = {
    "residual_diagnostics_spectrum.png": _residual_diagnostics,
    "factor_covariance_assembly_history.png": _factor_covariance_assembly,
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
