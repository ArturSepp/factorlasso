"""Canonical example for docs/cluster_stability_and_pooled_scoring.md.

Twelve responses form three blocks of four over 240 months; two more, x1 and x2, load half on
one block and half on the next, so they sit on cluster boundaries. Monthly causal partitions
(36-month EWMA correlation, three clusters) give each response a co-cluster stability weight w:
the EWMA share of its current peers that shared its cluster before. The script recomputes w by
direct counting, relates it to reassignment, and scores a momentum signal within the clusters
with and without stability pooling, checking the pooled score against its formula. Synthetic
data.
"""

import numpy as np
import pandas as pd

import factorlasso as fl

SEED = 20261007
N_OBS = 240
SPAN = 36
STABILITY_SPAN = 12
MIN_HISTORY = 12
FIRST_ESTIMATION = 36
BLOCK_VOL, RESIDUAL_VOL = 0.03, 0.02
NAMES = [f"{block}{k}" for block in "abc" for k in range(1, 5)] + ["x1", "x2"]
HOME = pd.Series(list("aaaabbbbcccc") + ["ab", "bc"], index=NAMES, name="home")


def make_panel(seed: int = SEED) -> pd.DataFrame:
    """Three blocks of four and two bridge responses loading half on two blocks."""
    rng = np.random.default_rng(seed)
    loadings = np.zeros((len(NAMES), 3))
    loadings[np.arange(12), np.repeat([0, 1, 2], 4)] = 1.0
    loadings[12] = [0.5, 0.5, 0.0]
    loadings[13] = [0.0, 0.5, 0.5]
    blocks = BLOCK_VOL * rng.standard_normal((N_OBS, 3))
    y = blocks @ loadings.T + RESIDUAL_VOL * rng.standard_normal((N_OBS, len(NAMES)))
    dates = pd.date_range("2006-01-31", periods=N_OBS, freq="ME")
    return pd.DataFrame(y, index=dates, columns=NAMES)


def rolling_partitions(y: pd.DataFrame) -> dict:
    """Monthly causal partitions without smoothing."""
    model = fl.LassoModel(
        model_type=fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
        span=SPAN,
        n_clusters=3,
    )
    return fl.compute_rolling_smoothed_clusters(y, list(y.index[FIRST_ESTIMATION:]), model).clusters


def stability(partitions: dict) -> fl.ClusterStabilityStatistics:
    """EWMA co-cluster stability weights with a 12-month span and a 12-date warm-up."""
    return fl.compute_cluster_stability_statistics(
        partitions, span_by_freq={"ME": STABILITY_SPAN}, min_history=MIN_HISTORY,
    )


def co_association_by_counting(partitions: dict, asset: str, date: pd.Timestamp) -> float:
    """w for one asset and date: the EWMA share of current peers that shared its cluster."""
    dates = [d for d in sorted(partitions) if d <= date]
    current = partitions[date]
    peers = current.index[(current == current[asset]) & (current.index != asset)]
    if len(peers) == 0:
        return 1.0
    decay = 1.0 - 2.0 / (STABILITY_SPAN + 1.0)
    shares = np.array([np.mean([partitions[d][p] == partitions[d][asset] for p in peers])
                       for d in dates])
    weights = decay ** np.arange(len(dates) - 1, -1, -1)
    return float(shares @ weights / weights.sum())


def block_membership(partitions: dict) -> pd.DataFrame:
    """Persistent labels for this synthetic panel: each cluster named by its members' home block."""
    rows = {}
    for date, clusters in partitions.items():
        names = {}
        for label, members in clusters.groupby(clusters).groups.items():
            homes = HOME[members]
            names[label] = homes[homes.str.len() == 1].mode().iloc[0]
        rows[date] = clusters.map(names)
    return pd.DataFrame(rows).T


def momentum(y: pd.DataFrame) -> pd.DataFrame:
    """Trailing 12-month return, the signal to be scored within clusters."""
    return y.rolling(12).sum().iloc[FIRST_ESTIMATION:]


def score(signal: pd.DataFrame, partitions: dict, weights: pd.DataFrame,
          pooling: fl.StabilityPoolingType) -> pd.DataFrame:
    """Within-cluster z-scores, unpooled or with variance pooled by stability."""
    return fl.score_with_stability_pooled_clusters(
        signal, partitions, stability_weights=weights, min_cluster_size=3, pooling_type=pooling,
    )


def main() -> None:
    y = make_panel()
    partitions = rolling_partitions(y)
    stats = stability(partitions)

    # --- w by direct counting ------------------------------------------------------------------
    dates = sorted(partitions)
    for date in (dates[20], dates[100], dates[-1]):
        for asset in ("a1", "x1", "x2"):
            assert np.isclose(stats.w_i.loc[date, asset],
                              co_association_by_counting(partitions, asset, date))
    assert (stats.w_i.loc[dates[:MIN_HISTORY - 1]] == 1.0).all().all()   # warm-up: unit weights
    for date in (dates[50], dates[150]):                     # w_g: the cluster mean of w_i
        clusters = partitions[date]
        for label, members in clusters.groupby(clusters).groups.items():
            assert np.isclose(stats.w_g.loc[date, label], stats.w_i.loc[date, members].mean())

    # --- w against reassignment ----------------------------------------------------------------
    membership = block_membership(partitions)
    boundary = stats.boundary_statistics(membership).iloc[0]
    mean_w = stats.w_i.mean()
    print(mean_w.round(2).to_dict())
    print(boundary[["mean_w_reassigned", "mean_w_stable", "reassignment_rate_bottom_w_quartile",
                    "reassignment_rate_top_w_quartile"]].round(2).to_dict())
    switches = (membership != membership.shift()).iloc[1:].sum()
    moved = switches[NAMES[:12]]
    assert moved.sum() == 2 and moved["c3"] == 2             # c3 strays for two months only

    # --- pooled scoring ------------------------------------------------------------------------
    signal = momentum(y)
    unpooled = score(signal, partitions, stats.w_i, fl.StabilityPoolingType.NONE)
    pooled = score(signal, partitions, stats.w_i, fl.StabilityPoolingType.ASSET_VARIANCE)
    date = dates[150]
    clusters, values, w = partitions[date], signal.loc[date], stats.w_i.loc[date]
    for _, members in clusters.groupby(clusters).groups.items():
        cluster = values[members]
        z = (cluster - cluster.mean()) / cluster.std()
        assert np.allclose(unpooled.loc[date, members], z)
        pooled_var = w[members] * cluster.var() + (1.0 - w[members]) * values.var()
        assert np.allclose(pooled.loc[date, members],
                           (cluster - cluster.mean()) / np.sqrt(pooled_var))
    global_z = (values - values.mean()) / values.std()        # clusters at or below the size cut
    fallback = fl.score_with_stability_pooled_clusters(signal.loc[[date]], partitions,
                                                        min_cluster_size=6)
    small = clusters.index[clusters.map(clusters.value_counts()) <= 6]
    assert np.allclose(fallback.loc[date, small], global_z[small])
    change = (pooled - unpooled).abs().mean()
    print(change.round(3).to_dict())

    # quoted values
    assert mean_w.round(2)[["x1", "x2"]].tolist() == [0.75, 0.76]
    assert mean_w.round(2)[NAMES[:12]].between(0.95, 0.97).all()
    assert round(boundary["mean_w_reassigned"], 2) == 0.4
    assert round(boundary["mean_w_stable"], 2) == 0.91
    assert switches[["x1", "x2"]].tolist() == [27, 16]
    assert round(boundary["reassignment_rate_bottom_w_quartile"], 2) == 0.09
    assert boundary["reassignment_rate_top_w_quartile"] == 0.0
    assert change.round(2)[["x1", "x2"]].tolist() == [0.1, 0.11]
    assert change[NAMES[:12]].max() < 0.025


if __name__ == "__main__":
    main()
