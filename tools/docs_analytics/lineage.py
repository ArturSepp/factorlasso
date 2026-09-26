"""Producer family for cluster lineage exhibits.

Exhibits
--------
``cluster_lineage_tracks.png`` for ``docs/cluster_lineage.md``: the raw cluster label of each
group of responses at each date, shuffled from date to date, and the persistent track that the
lineage analysis assigns to each group, with the per-transition matcher for comparison.

Each calculation is the worked example of its article, loaded from ``examples/docs/``. Nothing is
re-implemented here: this module calls the example's functions, checks that the registry
configuration describes what actually ran, and draws.
"""

from matplotlib.dates import DateFormatter, MonthLocator
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

MARKERS = ("o", "s", "D", "^")


def _lineage(spec: dict) -> tuple[Figure, dict, dict]:
    """Raw labels against persistent tracks, per group of responses."""
    example = load_example(spec["example"])
    verify_parameters(example, spec["parameters"])
    covar_data, raw = example["make_rolling"]()
    joint = example["derived_by_group"](example["lineage"](covar_data))
    per_transition = example["derived_by_group"](example["lineage"](covar_data, "hungarian"))
    order = sorted(track for track in set(joint.to_numpy().ravel()) if isinstance(track, str))
    number = {track: position + 1 for position, track in enumerate(order)}
    new_row = len(order) + 1
    # the per-transition matcher numbers its own tracks: map them onto the joint rows by group
    alias = {per_transition[group].dropna().iloc[0]: number[joint[group].dropna().iloc[0]]
             for group in joint.columns}
    colours = dict(zip(raw.columns, (*SERIES, INK_MUTED)))
    dates = example["DATES"]
    events = {"merge": dates[example["MERGE_DATE"]], "birth": dates[example["BIRTH_DATE"]]}

    figure = Figure(figsize=(11.0, 5.0), facecolor=SURFACE, layout="constrained")
    left, right = figure.subplots(1, 2)
    handles = []
    for (group, labels), marker in zip(raw.items(), MARKERS):
        left.plot(labels.index, labels, color=colours[group], marker=marker, linewidth=1.2,
                  markersize=7, markeredgecolor=SURFACE)
        tracks = joint[group].map(number)
        handles += right.plot(tracks.index, tracks, color=colours[group], marker=marker,
                              linewidth=2, markersize=7, markeredgecolor=SURFACE, label=group)
    fragmented = per_transition["credit"].map(lambda track: alias.get(track, new_row))
    handles += right.plot(fragmented.index, fragmented, color=colours["credit"], linestyle="--",
                          linewidth=1.6, label="credit, per-transition matcher")
    for axis, top in ((left, 4.75), (right, new_row + 0.75)):
        for name, date in events.items():
            axis.axvline(date, color=INK_MUTED, linestyle=":", linewidth=1.2)
            axis.annotate(name, xy=(date, top), xytext=(3, 0), textcoords="offset points",
                          va="center", fontsize=10.5, color=INK_MUTED)
        style_axis(axis)
        axis.xaxis.set_major_locator(MonthLocator(bymonth=(1, 7)))
        axis.xaxis.set_major_formatter(DateFormatter("%b %Y"))
    left.set_yticks(range(1, int(raw.max().max()) + 1))
    left.set_ylim(0.6, 5.0)
    left.set_ylabel("Raw cluster label")
    left.set_title("Raw labels are reassigned every month", fontsize=12.5, loc="left")
    right.set_yticks([*number.values(), new_row], labels=[*number, "new id"])
    right.set_ylim(0.6, new_row + 1.0)
    right.set_ylabel("Derived track")
    right.set_title("Lineage keeps one track per group", fontsize=12.5, loc="left")
    figure.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False,
                  fontsize=10.5, labelcolor=INK)

    tables = {"lineage_raw_labels": raw.rename_axis("date"),
              "lineage_tracks": joint.rename_axis("date"),
              "lineage_tracks_per_transition": per_transition.rename_axis("date")}
    checks = {
        "each_group_keeps_one_track_outside_the_merge": bool(
            joint.drop(events["merge"]).nunique().eq(1).all()),
        "per_transition_matcher_fragments_credit": bool(
            per_transition["credit"].nunique() > joint["credit"].nunique()),
    }
    return figure, tables, checks


EXHIBITS = {
    "cluster_lineage_tracks.png": _lineage,
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
