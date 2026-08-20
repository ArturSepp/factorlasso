"""Reference tests for causal EWMA cluster-stability statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _monthly_partitions(periods: int = 14) -> dict[pd.Timestamp, pd.Series]:
    """Return a monthly partition history with one moving boundary asset."""
    dates = pd.date_range("2025-01-31", periods=periods, freq="ME")
    partitions = {}
    for position, date in enumerate(dates):
        labels = ["left", "left", "right", "right"]
        if position >= 12 and position % 2:
            labels[1] = "right"
        partitions[date] = pd.Series(labels, index=list("abcd"))
    return partitions


def _direct_ewma_reference(
    partitions: dict[pd.Timestamp, pd.Series],
    span: int,
) -> pd.DataFrame:
    """Compute the definition directly, independently of the vectorised path."""
    dates = sorted(partitions)
    rows = {}
    for position, date in enumerate(dates):
        current = partitions[date].dropna()
        row = {}
        for asset, cluster_id in current.items():
            peers = current.index[(current == cluster_id) & (current.index != asset)]
            if len(peers) == 0:
                row[asset] = 1.0
                continue
            observations = []
            for prior_date in dates[:position + 1]:
                prior = partitions[prior_date]
                available_peers = peers.intersection(prior.dropna().index)
                if asset not in prior.dropna().index or len(available_peers) == 0:
                    observations.append(np.nan)
                else:
                    observations.append(
                        float((prior.loc[available_peers] == prior.loc[asset]).mean())
                    )
            row[asset] = pd.Series(observations).ewm(span=span, adjust=True).mean().iloc[-1]
        rows[date] = row
    return pd.DataFrame.from_dict(rows, orient="index").sort_index(axis=1)


def test_ewma_co_association_matches_hand_computed_reference() -> None:
    """EWMA co-association must use pandas adjust-True span arithmetic."""
    from factorlasso import compute_cluster_stability_statistics

    dates = pd.date_range("2026-01-31", periods=3, freq="ME")
    partitions = {
        dates[0]: pd.Series(["x", "x", "y", "y"], index=list("abcd")),
        dates[1]: pd.Series(["x", "z", "y", "y"], index=list("abcd")),
        dates[2]: pd.Series(["x", "x", "y", "y"], index=list("abcd")),
    }
    statistics = compute_cluster_stability_statistics(
        partitions,
        span_by_freq={"ME": 3, "QE": 2},
        min_history=1,
    )

    # span=3 gives alpha=1/2 and adjust=True weights 1/4, 1/2, 1.
    expected_boundary_weight = (0.25 * 1.0 + 0.5 * 0.0 + 1.0 * 1.0) / 1.75
    assert statistics.w_i.loc[dates[-1], "a"] == expected_boundary_weight
    assert statistics.w_i.loc[dates[-1], "b"] == expected_boundary_weight
    assert statistics.w_i.loc[dates[-1], "c"] == 1.0
    assert statistics.w_g.loc[dates[-1], "x"] == expected_boundary_weight
    assert statistics.w_i.stack().between(0.0, 1.0).all()


def test_vectorised_ewma_matches_an_independent_direct_reference() -> None:
    """Vectorisation must preserve peer changes and missing-asset arithmetic."""
    from factorlasso import compute_co_association_panel

    dates = pd.date_range("2025-01-31", periods=5, freq="ME")
    partitions = {
        dates[0]: pd.Series(["x", "x", "y", "y", "y"], index=list("abcde")),
        dates[1]: pd.Series(["x", "z", "y", "y"], index=list("abcd")),
        dates[2]: pd.Series(["x", "x", "x", "y", "y"], index=list("abcde")),
        dates[3]: pd.Series(["x", "x", "y", "y", "y"], index=list("abcde")),
        dates[4]: pd.Series(["x", "x", "y", "y", "x"], index=list("abcde")),
    }

    actual = compute_co_association_panel(partitions, span=4, min_history=1, adjust=True)
    expected = _direct_ewma_reference(partitions, span=4)

    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-15, atol=1e-15)


def test_span_map_resolves_monthly_and_quarterly_frequencies() -> None:
    """The pinned span map must resolve ME and QE panels independently."""
    from factorlasso import compute_cluster_stability_statistics

    monthly = _monthly_partitions()
    quarter_dates = pd.date_range("2023-03-31", periods=14, freq="QE")
    quarterly = {
        date: pd.Series(["left", "left", "right", "right"], index=list("abcd"))
        for date in quarter_dates
    }
    span_map = {"ME": 36, "QE": 18}

    monthly_statistics = compute_cluster_stability_statistics(monthly, span_map)
    quarterly_statistics = compute_cluster_stability_statistics(quarterly, span_map)

    assert monthly_statistics.frequency == "ME"
    assert monthly_statistics.span == 36
    assert quarterly_statistics.frequency == "QE"
    assert quarterly_statistics.span == 18


def test_ewma_statistics_are_causal() -> None:
    """Changing future partitions must not alter EWMA weights through the cutoff."""
    from factorlasso import compute_cluster_stability_statistics

    history = _monthly_partitions(periods=16)
    dates = sorted(history)
    cutoff = dates[13]
    short = compute_cluster_stability_statistics(
        {date: history[date] for date in dates if date <= cutoff},
        {"ME": 36, "QE": 18},
    )
    changed = dict(history)
    for date in dates[14:]:
        changed[date] = pd.Series(["future"] * 4, index=list("abcd"))
    full = compute_cluster_stability_statistics(changed, {"ME": 36, "QE": 18})

    pd.testing.assert_frame_equal(short.w_i, full.w_i.loc[:cutoff], check_exact=True)


def test_short_history_uses_exact_unit_weights() -> None:
    """The first eleven observed partitions must retain the exact unpooled fallback."""
    from factorlasso import compute_cluster_stability_statistics

    history = _monthly_partitions()
    statistics = compute_cluster_stability_statistics(
        history,
        {"ME": 36, "QE": 18},
        min_history=12,
    )
    dates = sorted(history)

    assert (statistics.w_i.loc[dates[:11]].stack() == 1.0).all()
    assert (statistics.w_i.loc[dates[11]:].stack() < 1.0).any()
    assert statistics.coverage.loc[dates[0], "short_history_fallback"]


def test_statistics_expose_boundary_and_confound_diagnostics() -> None:
    """The statistics object must own the diagnostics formerly held by the harness."""
    from factorlasso import compute_cluster_stability_statistics

    history = _monthly_partitions()
    statistics = compute_cluster_stability_statistics(history, {"ME": 36, "QE": 18})
    membership = pd.DataFrame.from_dict(history, orient="index")
    boundary = statistics.boundary_statistics(membership)
    size = statistics.size_vs_w_correlation()
    dispersion = statistics.within_cluster_asset_w_dispersion()

    assert {
        "mean_w_reassigned",
        "mean_w_stable",
        "reassignment_rate_bottom_w_quartile",
        "reassignment_rate_top_w_quartile",
    }.issubset(boundary.columns)
    assert {"date", "size_w_correlation"}.issubset(size.columns)
    assert {"date", "cluster_id", "w_asset_std"}.issubset(dispersion.columns)
    assert np.isfinite(statistics.coverage["coverage"]).all()


def test_v3_is_absent_from_the_public_api() -> None:
    """The unreleased V3 comparison arm must leave no public enum member."""
    import factorlasso
    from factorlasso import StabilityPoolingType

    assert "CLUSTER_MEAN_VARIANCE" not in StabilityPoolingType.__members__
    assert "CLUSTER_MEAN_VARIANCE" not in factorlasso.__all__
