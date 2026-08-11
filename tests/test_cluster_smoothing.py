"""Temporal cluster-smoothing and external-partition invariants."""

import numpy as np
import pandas as pd
import pytest

from factorlasso import LassoModel, LassoModelType


def _panel(seed=41):
    """Return a deterministic factor/asset panel with two response blocks."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2022-01-03", periods=150)
    x = pd.DataFrame(rng.standard_normal((150, 3)), index=index, columns=list("xyz"))
    common = rng.standard_normal((150, 2))
    y = np.column_stack([
        common[:, 0] + 0.12 * rng.standard_normal(150) for _ in range(4)
    ] + [
        common[:, 1] + 0.12 * rng.standard_normal(150) for _ in range(4)
    ])
    y = pd.DataFrame(y + 0.15 * x.to_numpy() @ rng.standard_normal((3, 8)),
                     index=index, columns=[f"a{i}" for i in range(8)])
    return x, y


def _same_partition(left, right):
    """Return whether two label vectors induce identical equivalence classes."""
    left = left.sort_index().to_numpy()
    right = right.sort_index().to_numpy()
    return np.array_equal(left[:, None] == left[None, :],
                          right[:, None] == right[None, :])


def test_none_smoother_matches_current_fit_partitions():
    """NONE must reproduce the pre-0.13 in-fit partition at every date."""
    from factorlasso import ClusterSmootherType, compute_rolling_smoothed_clusters

    x, y = _panel()
    dates = list(y.index[[79, 109, 139]])
    model = LassoModel(
        model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
        cluster_smoother_type=ClusterSmootherType.NONE,
        span=36,
    )
    rolling = compute_rolling_smoothed_clusters(y, dates, model)
    for date in dates:
        fitted = model.copy().fit(x.loc[:date], y.loc[:date])
        assert _same_partition(rolling.clusters[date], fitted.clusters_)
        np.testing.assert_array_equal(rolling.linkages[date], fitted.linkage_)
        assert rolling.cutoffs[date] == fitted.cutoff_


def test_zero_strength_smoothers_match_none():
    """Zero delta and zero lambda must equal NONE on every date."""
    from factorlasso import ClusterSmootherType, compute_rolling_smoothed_clusters

    _, y = _panel()
    dates = list(y.index[[79, 109, 139]])
    base = dict(model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO, span=36)
    none = compute_rolling_smoothed_clusters(
        y, dates, LassoModel(**base, cluster_smoother_type=ClusterSmootherType.NONE)
    )
    bonus = compute_rolling_smoothed_clusters(
        y, dates, LassoModel(**base,
                            cluster_smoother_type=ClusterSmootherType.PARTITION_BONUS,
                            smoother_delta=0.0)
    )
    similarity = compute_rolling_smoothed_clusters(
        y, dates, LassoModel(**base,
                            cluster_smoother_type=ClusterSmootherType.SIMILARITY_EWMA,
                            smoother_lambda=0.0)
    )
    for date in dates:
        assert _same_partition(none.clusters[date], bonus.clusters[date])
        assert _same_partition(none.clusters[date], similarity.clusters[date])


def test_smoothing_primitives_validate_and_preserve_matrix_invariants():
    """Public smoothing primitives validate inputs and preserve matrix structure."""
    from factorlasso import apply_partition_distance_bonus, smooth_similarity_ewma

    corr = pd.DataFrame(
        [[1.0, 0.4, 0.2], [0.4, 1.0, 0.1], [0.2, 0.1, 1.0]],
        index=list("abc"),
        columns=list("abc"),
    )
    previous = pd.DataFrame(
        [[1.0, 0.8], [0.8, 1.0]], index=list("ab"), columns=list("ab")
    )
    smoothed = smooth_similarity_ewma(corr, previous, 0.5)
    np.testing.assert_allclose(np.diag(smoothed), 1.0)
    assert smoothed.loc["a", "b"] == pytest.approx(0.6)
    assert smoothed.loc["a", "c"] == pytest.approx(corr.loc["a", "c"])
    with pytest.raises(ValueError, match="1.0"):
        smooth_similarity_ewma(corr, previous, 1.0)

    distance = np.array([[0.0, 0.2, 0.4], [0.2, 0.0, 0.5], [0.4, 0.5, 0.0]])
    labels = pd.Series([1.0, 1.0, np.nan], index=list("abc"))
    discounted = apply_partition_distance_bonus(distance, labels, 0.3)
    np.testing.assert_allclose(discounted, [[0.0, 0.0, 0.4], [0.0, 0.0, 0.5], [0.4, 0.5, 0.0]])
    with pytest.raises(ValueError, match="non-negative"):
        apply_partition_distance_bonus(distance, labels, -0.1)
    with pytest.raises(ValueError, match="does not match"):
        apply_partition_distance_bonus(distance[:2, :2], labels, 0.1)


def test_hold_smoother_and_empty_schedule():
    """HOLD reuses a partition between anchors and an empty schedule is empty."""
    from factorlasso import ClusterSmootherType, compute_rolling_smoothed_clusters

    _, y = _panel()
    model = LassoModel(
        model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
        cluster_smoother_type=ClusterSmootherType.HOLD,
        recluster_freq="YE",
        span=None,
    )
    empty = compute_rolling_smoothed_clusters(y, [], model)
    assert empty.clusters == {}
    assert empty.co_association.empty

    dates = list(y.index[[79, 109, 139]])
    held = compute_rolling_smoothed_clusters(y, dates, model)
    assert _same_partition(held.clusters[dates[0]], held.clusters[dates[1]])
    assert _same_partition(held.clusters[dates[1]], held.clusters[dates[2]])
    np.testing.assert_array_equal(held.linkages[dates[0]], held.linkages[dates[2]])


def test_fcgl_external_partition_is_coefficient_identical():
    """External internal clusters must preserve FCGL coefficients and metadata."""
    x, y = _panel()
    plain = LassoModel(
        model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
        reg_lambda=1e-5,
        span=36,
        auto_sign_constraints=True,
    ).fit(x, y)
    external = plain.copy().fit(
        x,
        y,
        external_clusters=plain.clusters_,
        external_linkage=plain.linkage_,
        external_cutoff=plain.cutoff_,
    )
    assert external.model_type == LassoModelType.FACTOR_CLUSTER_GROUP_LASSO
    np.testing.assert_allclose(external.coef_, plain.coef_, rtol=0.0, atol=1e-12)
    assert _same_partition(external.clusters_, plain.clusters_)
    np.testing.assert_array_equal(external.linkage_, plain.linkage_)
    assert external.cutoff_ == plain.cutoff_


def test_rolling_smoothing_is_causal():
    """Appending observations after t must leave the partition at t unchanged."""
    from factorlasso import ClusterSmootherType, compute_rolling_smoothed_clusters

    _, y = _panel()
    date = y.index[109]
    model = LassoModel(
        model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
        cluster_smoother_type=ClusterSmootherType.PARTITION_BONUS,
        smoother_delta=0.05,
        span=36,
    )
    short = compute_rolling_smoothed_clusters(y.loc[:date], [date], model)
    full = compute_rolling_smoothed_clusters(y, [date], model)
    assert _same_partition(short.clusters[date], full.clusters[date])
    np.testing.assert_array_equal(short.linkages[date], full.linkages[date])
    assert short.cutoffs[date] == full.cutoffs[date]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"smoother_delta": -0.1}, "-0.1"),
        ({"smoother_lambda": 1.0}, "1.0"),
        ({"cluster_smoother_type": "HOLD"}, "None"),
        ({"recluster_freq": "QE"}, "QE"),
    ],
)
def test_smoother_configuration_validation(kwargs, match):
    """Every invalid smoother field combination must name its bad value."""
    from factorlasso import ClusterSmootherType

    values = dict(kwargs)
    if values.get("cluster_smoother_type") == "HOLD":
        values["cluster_smoother_type"] = ClusterSmootherType.HOLD
    with pytest.raises(ValueError, match=match):
        LassoModel(**values)
