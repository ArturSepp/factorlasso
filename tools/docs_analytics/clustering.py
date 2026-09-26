"""Producer family for cluster exhibits.

Exhibits
--------
``cluster_discovery_dendrograms.png`` for ``docs/cluster_discovery.md``: Ward dendrograms of a
shocked three-block panel under the Pearson and Spearman dependence measures, cut at three
clusters, and the realised cluster count against ``cutoff_fraction`` under the three distance
transforms.

``common_mode_spectrum.png`` for ``docs/common_mode_removal.md``: the eigenvalues of the response
correlation matrix before and after the dominant common mode is removed, and the share of panels
whose three clusters are the sectors against the market volatility, with and without the removal.

``rolling_smoothing_churn.png`` for ``docs/rolling_cluster_smoothing.md``: the cluster of a
migrating response at each monthly estimation date under the four causal smoothers, and the mean
number of its cluster switches over ten panels, with the lag of its final switch.

``stability_weights_heatmap.png`` for ``docs/cluster_stability_and_pooled_scoring.md``: the
co-cluster stability weight of every response at every monthly partition date, and the mean change
of a within-cluster score under stability pooling against the mean weight of each response.

Each calculation is the worked example of its article, loaded from ``examples/docs/``. Nothing is
re-implemented here: this module calls the example's functions, checks that the registry
configuration describes what actually ran, and draws.
"""

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as spc
from matplotlib.figure import Figure

from tools.docs_analytics.fixtures import (
    INK,
    INK_MUTED,
    SERIES,
    SURFACE,
    load_example,
    style_axis,
    verify_parameters,
)

MARKERS = ("o", "s", "D")
LINE = {"linewidth": 2, "markersize": 7, "markeredgecolor": SURFACE, "markeredgewidth": 1.5}


def _dendrogram(axis, linkage: np.ndarray, labels: list, n_clusters: int, title: str) -> None:
    """Draw a neutral dendrogram and its cut at ``n_clusters``, between the two merges."""
    n_merges = len(labels) - n_clusters
    cut = 0.5 * (linkage[n_merges - 1, 2] + linkage[n_merges, 2])
    spc.dendrogram(linkage, labels=labels, ax=axis, color_threshold=0.0,
                   above_threshold_color=INK_MUTED, leaf_font_size=10.5)
    axis.axhline(cut, color=INK, linestyle="--", linewidth=1.2)
    axis.annotate(f"cut at {n_clusters} clusters", xy=(0.0, cut),
                  xycoords=("axes fraction", "data"),
                  xytext=(4, 5), textcoords="offset points", ha="left", fontsize=10.5, color=INK)
    axis.set_title(title, fontsize=12.5, loc="left")
    axis.set_ylabel("Ward merge height, 1 - rho")
    axis.tick_params(axis="x", colors=INK, length=0)


def _cluster_discovery(spec: dict) -> tuple[Figure, dict, dict]:
    """Dendrograms under two dependence measures, and cluster counts by cut fraction."""
    example = load_example(spec["example"])
    verify_parameters(example, spec["parameters"])
    _, y = example["make_panel"]()
    n_clusters = example["N_CLUSTERS"]
    trees = {measure: example["discover"](y, measure, n_clusters=n_clusters)
             for measure in ("pearson", "spearman")}
    counts = example["cluster_counts"](y, "spearman")

    figure = Figure(figsize=(12.0, 4.6), facecolor=SURFACE, layout="constrained")
    left, middle, right = figure.subplots(1, 3, width_ratios=(1.0, 1.0, 1.15))
    _dendrogram(left, trees["pearson"][1], list(y.columns), n_clusters,
                "Pearson: blocks b and c merge")
    _dendrogram(middle, trees["spearman"][1], list(y.columns), n_clusters,
                "Spearman: three blocks")
    labels = {"one_minus_rho": "1 - rho", "chord": "chord", "arccos": "arccos"}
    for (transform, label), colour, marker in zip(labels.items(), SERIES, MARKERS):
        right.plot(counts.index, counts[transform], color=colour, marker=marker, label=label,
                   **LINE)
    right.axhline(n_clusters, color=INK_MUTED, linestyle=":", linewidth=1.2)
    right.axvline(0.5, color=INK_MUTED, linestyle="--", linewidth=1.2)
    right.annotate("default 0.5", xy=(0.5, 7.0), xytext=(-5, 0), textcoords="offset points",
                   ha="right", fontsize=10.5, color=INK_MUTED, va="center")
    right.set_xlim(0.0, 1.03)
    right.set_ylim(0.0, 13.0)
    right.set_xlabel("cutoff_fraction")
    right.set_ylabel("Clusters, Spearman dependence")
    right.set_title("Cluster count by fractional cut", fontsize=12.5, loc="left")
    right.legend(frameon=False, fontsize=11, labelcolor=INK, loc="lower left")
    for axis in (left, middle, right):
        style_axis(axis)
    for axis in (left, middle):
        axis.grid(axis="x", visible=False)

    partitions = pd.DataFrame({measure: tree[0] for measure, tree in trees.items()})
    tables = {"cluster_discovery_partitions": partitions.rename_axis("response"),
              "cluster_discovery_counts": counts,
              "cluster_discovery_block_correlations":
                  example["block_correlations"](y).rename_axis("measure")}
    checks = {
        "pearson_misses_the_blocks": not example["recovers_blocks"](trees["pearson"][0]),
        "spearman_recovers_the_blocks": example["recovers_blocks"](trees["spearman"][0]),
        "default_fraction_isolates_every_response": bool(counts.loc[0.5].eq(len(y.columns)).all()),
    }
    return figure, tables, checks


def _common_mode(spec: dict) -> tuple[Figure, dict, dict]:
    """Spectrum before and after removal, and sector recovery against market strength."""
    example = load_example(spec["example"])
    verify_parameters(example, spec["parameters"])
    _, y = example["make_panel"]()
    _, _, removal = example["partitions"](y)
    spectra = pd.DataFrame({
        "correlation": np.linalg.eigvalsh(y.corr().to_numpy())[::-1],
        "common mode removed": np.linalg.eigvalsh(removal.correlation.to_numpy())[::-1],
    }, index=pd.RangeIndex(1, len(y.columns) + 1, name="rank"))
    sweep = example["recovery_by_market_vol"]()

    figure = Figure(figsize=(11.0, 4.6), facecolor=SURFACE, layout="constrained")
    left, right = figure.subplots(1, 2)
    for (name, values), colour, marker in zip(spectra.items(), SERIES, MARKERS):
        left.plot(spectra.index, values, color=colour, marker=marker, label=name, **LINE)
    left.set_xticks(spectra.index)
    left.set_xlabel("Eigenvalue rank")
    left.set_ylabel("Eigenvalue of the correlation matrix")
    left.set_title(f"Market volatility {example['MARKET_VOL']:.0%} per month", fontsize=12.5,
                   loc="left")
    left.legend(frameon=False, fontsize=11, labelcolor=INK, loc="upper right")

    market = 100.0 * sweep.index.to_numpy()
    for name, colour, marker in zip(("correlation", "common mode removed"), SERIES, MARKERS):
        right.plot(market, sweep[name], color=colour, marker=marker, label=name, **LINE)
    right.axvline(100.0 * example["MARKET_VOL"], color=INK_MUTED, linestyle=":", linewidth=1.2)
    right.set_ylim(-0.05, 1.05)
    right.set_xlabel("Market volatility, % per month")
    right.set_ylabel(f"Share of {example['N_PANELS']} panels clustered by sector")
    right.set_title("Three clusters equal the three sectors", fontsize=12.5, loc="left")
    right.legend(frameon=False, fontsize=11, labelcolor=INK, loc="center right")
    for axis in (left, right):
        style_axis(axis)

    tables = {"common_mode_spectra": spectra, "common_mode_recovery": sweep}
    checks = {
        "removal_recovers_sectors_under_a_dominant_mode": bool(
            (sweep.loc[0.05:, "common mode removed"] == 1.0).all()),
        "raw_correlation_fails_under_a_dominant_mode": bool(
            (sweep.loc[0.06:, "correlation"] <= 0.05).all()),
        "removal_hurts_without_a_common_mode": bool(
            sweep.loc[0.0, "common mode removed"] < sweep.loc[0.0, "correlation"]),
    }
    return figure, tables, checks


def _rolling_smoothing(spec: dict) -> tuple[Figure, dict, dict]:
    """The migrating response's cluster over time per smoother, and churn over ten panels."""
    example = load_example(spec["example"])
    verify_parameters(example, spec["parameters"])
    y = example["make_panel"]()
    names = list(example["SMOOTHERS"])
    assert set(names) == {"none", "hold", "partition bonus", "bonus at noise floor",
                          "similarity EWMA"}
    tracks = {name: example["migrant_track"](example["rolling_partitions"](y, name))
              for name in names}
    panels = example["churn_over_panels"]()
    colours = {"none": INK_MUTED, "hold": SERIES[0], "partition bonus": SERIES[1],
               "bonus at noise floor": SERIES[1], "similarity EWMA": SERIES[2]}
    dashed = {"bonus at noise floor"}                 # the same smoother at a calibrated delta

    figure = Figure(figsize=(11.0, 4.6), facecolor=SURFACE, layout="constrained")
    left, right = figure.subplots(1, 2, width_ratios=(1.45, 1.0))
    dates = example["DATES"]
    start = dates[example["DRIFT_START"]]
    end = dates[example["DRIFT_START"] + example["DRIFT_MONTHS"] - 1]
    left.axvspan(start, end, color=INK_MUTED, alpha=0.08, linewidth=0)
    left.axvline(example["CROSSING"], color=INK, linestyle="--", linewidth=1.2)
    for row, name in enumerate(reversed(names)):
        track = tracks[name]
        left.step(track.index, row + 0.55 * track.to_numpy(), where="post", color=colours[name],
                  linewidth=2, linestyle="--" if name in dashed else "-")
    left.set_yticks([row + 0.275 for row in range(len(names))], labels=list(reversed(names)))
    left.tick_params(axis="y", length=0)
    left.set_ylim(-0.4, len(names) - 0.1)
    left.annotate("loadings equal", xy=(example["CROSSING"], len(names) - 0.25),
                  xytext=(-5, 0), textcoords="offset points", ha="right", fontsize=10.5,
                  color=INK)
    left.set_xlabel("Estimation date; shaded: a4 migrates from block a to block b")
    left.set_title("Cluster of a4: low with block a, high with block b", fontsize=12.5,
                   loc="left")

    switches = panels["a4 switches"]
    positions = np.arange(len(names))[::-1]
    right.barh(positions, switches.reindex(names), color=[colours[n] for n in names],
               hatch=["//" if n in dashed else "" for n in names], edgecolor=SURFACE,
               height=0.6)
    for position, name in zip(positions, names):
        lag = panels.loc[name, "final switch, months after crossing"]
        right.annotate(f"{switches[name]:.1f}, final switch +{lag:.1f} months",
                       xy=(switches[name], position), xytext=(5, 0), textcoords="offset points",
                       va="center", fontsize=10.5, color=INK)
    right.set_yticks(positions, labels=names)
    right.tick_params(axis="y", length=0)
    right.set_xlim(0.0, 5.2)
    right.set_xlabel(f"Cluster switches of a4, mean over {example['N_PANELS']} panels")
    right.set_title("One switch is the truth", fontsize=12.5, loc="left")
    for axis in (left, right):
        style_axis(axis)
    left.grid(axis="y", visible=False)
    right.grid(axis="y", visible=False)

    tables = {"rolling_smoothing_tracks": pd.DataFrame(tracks).rename_axis("date"),
              "rolling_smoothing_churn": panels.rename_axis("smoother")}
    checks = {
        "smoothers_reduce_switches": bool(
            (panels.loc[names[1:], "a4 switches"] <= panels.loc["none", "a4 switches"]).all()),
        "every_panel_switches_at_least_once": bool((panels["a4 switches"] >= 1.0).all()),
    }
    return figure, tables, checks


def _stability(spec: dict) -> tuple[Figure, dict, dict]:
    """Stability weights over dates and responses, and the pooling effect by mean weight."""
    from matplotlib.colors import LinearSegmentedColormap

    example = load_example(spec["example"])
    verify_parameters(example, spec["parameters"])
    y = example["make_panel"]()
    partitions = example["rolling_partitions"](y)
    stats = example["stability"](partitions)
    signal = example["momentum"](y)
    unpooled = example["score"](signal, partitions, stats.w_i,
                                example["fl"].StabilityPoolingType.NONE)
    pooled = example["score"](signal, partitions, stats.w_i,
                              example["fl"].StabilityPoolingType.ASSET_VARIANCE)
    change = (pooled - unpooled).abs().mean()
    names = example["NAMES"]
    weights = stats.w_i[names]

    figure = Figure(figsize=(11.0, 4.6), facecolor=SURFACE, layout="constrained")
    left, right = figure.subplots(1, 2, width_ratios=(1.6, 1.0))
    colormap = LinearSegmentedColormap.from_list("stability", [SERIES[0], SURFACE])
    dates = weights.index
    image = left.imshow(weights.T.to_numpy(), aspect="auto", cmap=colormap, vmin=0.0, vmax=1.0,
                        interpolation="nearest",
                        extent=(0, len(dates), len(names) - 0.5, -0.5))
    years = [k for k, date in enumerate(dates) if date.month == 12 and date.year % 3 == 0]
    left.set_xticks(years, labels=[str(dates[k].year) for k in years])
    left.set_yticks(range(len(names)), labels=names)
    left.tick_params(length=0)
    left.set_xlabel("Partition date")
    left.set_title("Stability weight w of each response", fontsize=12.5, loc="left")
    colorbar = figure.colorbar(image, ax=left, fraction=0.04, pad=0.02)
    colorbar.set_label("w; darker is less stable")
    colorbar.outline.set_visible(False)

    bridges = [name for name in names if name.startswith("x")]
    members = [name for name in names if name not in bridges]
    mean_w = weights.mean()
    right.plot(mean_w[members], change[members], linestyle="none", marker=MARKERS[0],
               color=SERIES[0], markersize=8, markeredgecolor=SURFACE, label="block members")
    right.plot(mean_w[bridges], change[bridges], linestyle="none", marker=MARKERS[1],
               color=SERIES[1], markersize=9, markeredgecolor=SURFACE, label="bridge responses")
    for name in bridges:
        right.annotate(name, xy=(mean_w[name], change[name]), xytext=(7, 0),
                       textcoords="offset points", va="center", fontsize=11, color=INK)
    right.set_xlim(0.6, 1.02)
    right.set_ylim(0.0, 0.14)
    right.set_xlabel("Mean stability weight w")
    right.set_ylabel("Mean |score change| from pooling")
    right.set_title("Pooling acts where w is low", fontsize=12.5, loc="left")
    right.legend(frameon=False, fontsize=11, labelcolor=INK, loc="upper right")
    style_axis(right)
    left.grid(False)
    for spine in left.spines.values():
        spine.set_visible(False)

    tables = {"stability_weights": weights.rename_axis("date"),
              "stability_pooling_change": pd.DataFrame(
                  {"mean_w": mean_w, "mean_abs_score_change": change}).rename_axis("response")}
    checks = {
        "bridges_are_least_stable": bool(mean_w[bridges].max() < mean_w[members].min()),
        "pooling_changes_bridges_most": bool(change[bridges].min() > change[members].max()),
    }
    return figure, tables, checks


EXHIBITS = {
    "cluster_discovery_dendrograms.png": _cluster_discovery,
    "common_mode_spectrum.png": _common_mode,
    "rolling_smoothing_churn.png": _rolling_smoothing,
    "stability_weights_heatmap.png": _stability,
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
