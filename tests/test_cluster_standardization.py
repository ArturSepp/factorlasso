"""Reference and causality tests for stability-pooled cluster scoring."""

import numpy as np
import pandas as pd
import pytest

DATE = pd.Timestamp("2026-01-31")
ASSETS = list("abcdef")
CLUSTERS = {DATE: pd.Series(["g1"] * 3 + ["g2"] * 3, index=ASSETS)}
RAW = pd.DataFrame([[1.0, 2.0, 4.0, 10.0, 20.0, 40.0]], index=[DATE], columns=ASSETS)
WEIGHTS = pd.DataFrame([[0.2, 0.4, 0.6, 0.8, 0.6, 0.4]], index=[DATE], columns=ASSETS)


def _explicit_scores(mode: str) -> pd.Series:
    """Return the hand-computed pooled score for one reference row."""
    row = RAW.loc[DATE]
    global_var = row.var()
    expected = pd.Series(0.0, index=ASSETS)
    for label, members in CLUSTERS[DATE].groupby(CLUSTERS[DATE]).groups.items():
        del label
        values = row.loc[members]
        cluster_mean = values.mean()
        cluster_var = values.var()
        cluster_weight = WEIGHTS.loc[DATE, members].mean()
        if mode == "asset":
            weight = WEIGHTS.loc[DATE, members]
        else:
            weight = pd.Series(cluster_weight, index=members)
        variance = weight * cluster_var + (1.0 - weight) * global_var
        mean = pd.Series(cluster_mean, index=members)
        expected.loc[members] = (values - mean) / np.sqrt(variance)
    return expected


@pytest.mark.parametrize(
    ("pooling_name", "reference_mode"),
    [
        ("CLUSTER_VARIANCE", "cluster"),
        ("ASSET_VARIANCE", "asset"),
    ],
)
def test_stability_pooling_matches_hand_computed_reference(pooling_name, reference_mode):
    """V1 and V2 must match explicit pooled-variance arithmetic."""
    from factorlasso import StabilityPoolingType, score_with_stability_pooled_clusters

    actual = score_with_stability_pooled_clusters(
        raw_signal=RAW,
        rolling_clusters=CLUSTERS,
        stability_weights=WEIGHTS,
        min_cluster_size=1,
        pooling_type=StabilityPoolingType[pooling_name],
    )

    np.testing.assert_allclose(
        actual.loc[DATE, ASSETS].to_numpy(),
        _explicit_scores(reference_mode).to_numpy(),
        rtol=0.0,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "pooling_name",
    ["CLUSTER_VARIANCE", "ASSET_VARIANCE"],
)
def test_zero_stability_uses_cluster_demeaning_and_global_variance(pooling_name):
    """At w=0 every denominator is global while the cluster mean stays local."""
    from factorlasso import StabilityPoolingType, score_with_stability_pooled_clusters

    zero = pd.DataFrame(0.0, index=[DATE], columns=ASSETS)
    actual = score_with_stability_pooled_clusters(
        RAW,
        CLUSTERS,
        zero,
        min_cluster_size=1,
        pooling_type=StabilityPoolingType[pooling_name],
    )
    row = RAW.loc[DATE]
    expected = pd.Series(0.0, index=ASSETS)
    for _, members in CLUSTERS[DATE].groupby(CLUSTERS[DATE]).groups.items():
        expected.loc[members] = (row.loc[members] - row.loc[members].mean()) / np.sqrt(
            row.var()
        )

    np.testing.assert_allclose(actual.loc[DATE], expected, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize(
    "pooling_name",
    ["CLUSTER_VARIANCE", "ASSET_VARIANCE"],
)
def test_small_cluster_global_fallback_precedes_every_pooling_variant(pooling_name):
    """The existing minimum-size gate must ignore w and use full-cross-section statistics."""
    from factorlasso import StabilityPoolingType, score_with_stability_pooled_clusters

    clusters = {DATE: pd.Series(["large"] * 4 + ["small"] * 2, index=ASSETS)}
    actual = score_with_stability_pooled_clusters(
        RAW,
        clusters,
        WEIGHTS,
        min_cluster_size=2,
        pooling_type=StabilityPoolingType[pooling_name],
    )
    row = RAW.loc[DATE]
    expected = (row.loc[["e", "f"]] - row.mean()) / row.std()

    np.testing.assert_allclose(actual.loc[DATE, ["e", "f"]], expected, rtol=0.0, atol=0.0)


def _partition_history(periods: int = 15) -> dict[pd.Timestamp, pd.Series]:
    """Return dated partitions with one boundary asset that moves after warmup."""
    dates = pd.date_range("2024-01-31", periods=periods, freq="ME")
    history = {}
    for i, date in enumerate(dates):
        labels = ["left", "left", "right", "right"]
        if i >= 12 and i % 2:
            labels[1] = "right"
        history[date] = pd.Series(labels, index=list("abcd"))
    return history


def test_public_co_association_accessor_is_causal():
    """Changing partitions after t must leave every stability weight through t unchanged."""
    from factorlasso import compute_co_association_panel

    history = _partition_history()
    dates = sorted(history)
    cutoff = dates[12]
    short = compute_co_association_panel(
        {date: history[date] for date in dates if date <= cutoff}, window=12, min_history=12
    )
    changed = dict(history)
    for date in dates[13:]:
        changed[date] = pd.Series(["future"] * 4, index=list("abcd"))
    full = compute_co_association_panel(changed, window=12, min_history=12)

    pd.testing.assert_frame_equal(short, full.loc[:cutoff], check_exact=True)


def test_public_co_association_accessor_uses_one_during_short_history():
    """Fewer than twelve partition dates must reproduce current unpooled scoring."""
    from factorlasso import compute_co_association_panel

    history = _partition_history()
    panel = compute_co_association_panel(history, window=12, min_history=12)
    dates = sorted(history)

    assert (panel.loc[dates[:11]].stack() == 1.0).all()
    assert (panel.loc[dates[11]:].stack() < 1.0).any()


def test_public_accessor_default_preserves_the_private_panel():
    """Promotion to a public name must not alter the panel attached to rolling clusters."""
    from factorlasso import compute_co_association_panel
    from factorlasso.cluster_smoothing import _co_association_panel

    history = _partition_history()
    expected = _co_association_panel(history, window=6)
    actual = compute_co_association_panel(history, window=6)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
