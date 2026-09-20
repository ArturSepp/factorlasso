"""Producer family for covariance and residual exhibits.

Exhibits
--------
``residual_diagnostics_spectrum.png`` for ``docs/residual_diagnostics.md``: the eigenvalues of the
residual correlation matrix against the Marchenko-Pastur edge, and the sphericity statistic
against its chi-square threshold along a penalty grid, for a complete factor set and for the same
panel with one factor withheld.

The calculation is the article's worked example, loaded from
``examples/docs/residual_diagnostics.py``. Nothing is re-implemented here: this module reads the
example's functions, checks that the registry configuration describes what actually ran, and draws.
"""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter, ScalarFormatter
from scipy import stats

from tools.docs_analytics.fixtures import GRID, INK, INK_MUTED, SERIES, SURFACE, load_example

MARKERS = ("o", "s")


def _style(axis) -> None:
    """Recessive frame and grid; text in ink, never in a series colour."""
    axis.set_facecolor(SURFACE)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=INK_MUTED, labelsize=10.5)
    axis.xaxis.label.set_color(INK)
    axis.xaxis.label.set_size(11.5)
    axis.yaxis.label.set_size(11.5)
    axis.yaxis.label.set_color(INK)
    axis.title.set_color(INK)


def _residual_diagnostics(configuration: dict) -> tuple[Figure, dict, dict]:
    """Compute the worked example once and draw its two-panel exhibit."""
    example = load_example(configuration["example"])
    applied = {
        "seed": example["SEED"],
        "n_obs": example["N_OBS"],
        "reg_lambda": example["REG_LAMBDA"],
        "significance": example["SIGNIFICANCE"],
        "withheld_factor": example["WITHHELD_FACTOR"],
        "n_null_panels": example["N_NULL_PANELS"],
    }
    declared = {key: configuration["parameters"][key] for key in applied}
    if applied != declared:
        raise ValueError(f"Registry parameters {declared} differ from the example's {applied}")

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
        _style(axis)

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


def produce(configuration: dict) -> dict:
    """Return figures, supporting tables, the applied configuration and numerical checks.

    Parameters
    ----------
    configuration : dict
        The producer's ``configuration`` record from ``registry.json``.

    Returns
    -------
    dict
        Keys ``figures`` (PNG basename to Matplotlib ``Figure``), ``tables`` (name to nonempty
        ``DataFrame``), ``configuration`` (echoed once verified against what ran) and ``checks``
        (name to ``True``).
    """
    figure, tables, checks = _residual_diagnostics(configuration)
    return {
        "figures": {"residual_diagnostics_spectrum.png": figure},
        "tables": tables,
        "configuration": configuration,
        "checks": checks,
    }
