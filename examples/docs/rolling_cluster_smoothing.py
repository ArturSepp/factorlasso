"""Canonical example for docs/rolling_cluster_smoothing.md.

Twelve responses form three blocks of four over 240 months. One response, a4, migrates: its
loading moves linearly from block a to block b between months 60 and 180, so its two loadings are
equal at month 120. Partitions are re-estimated every month from month 36 with a 36-month EWMA
correlation, cut at three clusters, under the four causal smoothers. The script counts how often
a4 changes cluster and how late its final switch comes, on one panel and over ten, and checks the
rolling partitions against a direct fit and the smoother updates against their formulas.
Synthetic data.
"""

import numpy as np
import pandas as pd

import factorlasso as fl

SEED = 20261006
N_OBS = 240
SPAN = 36
BLOCK_VOL = 0.03
RESIDUAL_VOL = 0.02
DRIFT_START, DRIFT_MONTHS = 60, 120
FIRST_ESTIMATION = 36
N_CLUSTERS = 3
N_PANELS = 10
NAMES = [f"{block}{k}" for block in "abc" for k in range(1, 5)]
DATES = pd.date_range("2006-01-31", periods=N_OBS, freq="ME")
CROSSING = DATES[DRIFT_START + DRIFT_MONTHS // 2]         # a4 loads equally on a and b


def noise_floor_delta(rho: float, span: float, z: float = 1.6449, kappa: float = 0.0) -> float:
    """Partition bonus at the noise floor of an EWMA correlation (Sepp, 2026, working paper)."""
    return z * np.sqrt(1.0 + kappa) * (1.0 - rho**2) / np.sqrt(span)


WITHIN_BLOCK_RHO = BLOCK_VOL**2 / (BLOCK_VOL**2 + RESIDUAL_VOL**2)
CALIBRATED_DELTA = round(noise_floor_delta(WITHIN_BLOCK_RHO, SPAN), 2)
SMOOTHERS = {
    "none": {"cluster_smoother_type": fl.ClusterSmootherType.NONE},
    "hold": {"cluster_smoother_type": fl.ClusterSmootherType.HOLD, "recluster_freq": "QE"},
    "partition bonus": {"cluster_smoother_type": fl.ClusterSmootherType.PARTITION_BONUS,
                        "smoother_delta": 0.05},
    "bonus at noise floor": {"cluster_smoother_type": fl.ClusterSmootherType.PARTITION_BONUS,
                             "smoother_delta": CALIBRATED_DELTA},
    "similarity EWMA": {"cluster_smoother_type": fl.ClusterSmootherType.SIMILARITY_EWMA,
                        "smoother_lambda": 0.7},
}


def make_panel(seed: int = SEED) -> pd.DataFrame:
    """Three blocks of four; a4 moves from block a to block b between months 60 and 180."""
    rng = np.random.default_rng(seed)
    blocks = BLOCK_VOL * rng.standard_normal((N_OBS, 3))
    loadings = np.zeros((N_OBS, len(NAMES), 3))
    loadings[:, np.arange(len(NAMES)), np.repeat([0, 1, 2], 4)] = 1.0
    weight = np.clip((np.arange(N_OBS) - DRIFT_START) / DRIFT_MONTHS, 0.0, 1.0)
    loadings[:, 3, 0], loadings[:, 3, 1] = 1.0 - weight, weight
    y = (np.einsum("tk,tik->ti", blocks, loadings)
         + RESIDUAL_VOL * rng.standard_normal((N_OBS, len(NAMES))))
    return pd.DataFrame(y, index=DATES, columns=NAMES)


def rolling_partitions(y: pd.DataFrame, smoother: str) -> fl.RollingClusterData:
    """Monthly causal partitions under one smoother, with the model's discovery settings."""
    model = fl.LassoModel(
        model_type=fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
        span=SPAN,
        n_clusters=N_CLUSTERS,
        **SMOOTHERS[smoother],
    )
    return fl.compute_rolling_smoothed_clusters(y, list(y.index[FIRST_ESTIMATION:]), model)


def co_membership(clusters: pd.Series) -> np.ndarray:
    """Pairs in the same cluster, which does not depend on how clusters are numbered."""
    labels = clusters.to_numpy()
    return labels[:, None] == labels[None, :]


def migrant_track(result: fl.RollingClusterData) -> pd.Series:
    """1 on the dates a4 shares b1's cluster, 0 when it shares a1's, NaN otherwise."""
    rows = {}
    for date, clusters in sorted(result.clusters.items()):
        with_b, with_a = clusters["a4"] == clusters["b1"], clusters["a4"] == clusters["a1"]
        rows[date] = 1.0 if with_b else (0.0 if with_a else np.nan)
    return pd.Series(rows, name="a4 with block b")


def churn(result: fl.RollingClusterData) -> dict:
    """Membership changes, label changes, and the lag of a4's final switch after the crossing."""
    dates = sorted(result.clusters)
    partitions = [result.clusters[date] for date in dates]
    membership = sum(not np.array_equal(co_membership(p), co_membership(q))
                     for p, q in zip(partitions, partitions[1:]))
    labels = sum(not p.equals(q) for p, q in zip(partitions, partitions[1:]))
    track = migrant_track(result)
    switches = track.index[1:][track.diff().iloc[1:].ne(0).to_numpy()]
    final = switches[-1]
    lag = (final.year - CROSSING.year) * 12 + final.month - CROSSING.month
    return {"membership changes": membership, "label changes": labels,
            "a4 switches": len(switches), "final switch, months after crossing": lag}


def churn_over_panels() -> pd.DataFrame:
    """Mean churn statistics of each smoother over N_PANELS panels."""
    rows = [{"panel": k, "smoother": name, **churn(rolling_partitions(make_panel(SEED + k), name))}
            for k in range(N_PANELS) for name in SMOOTHERS]
    return pd.DataFrame(rows).groupby("smoother", sort=False).mean().drop(columns="panel")


def main() -> None:
    y = make_panel()
    results = {name: rolling_partitions(y, name) for name in SMOOTHERS}

    # --- causality: the unsmoothed partition at a date is the partition a fit on y[:date] finds
    for date in (DATES[100], DATES[150], DATES[200]):
        history = y.loc[:date]
        factor = pd.DataFrame({"f": np.linspace(-0.01, 0.01, len(history))}, index=history.index)
        model = fl.LassoModel(
            model_type=fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
            span=SPAN, n_clusters=N_CLUSTERS,
        ).fit(x=factor, y=history)
        assert np.array_equal(co_membership(model.clusters_),
                              co_membership(results["none"].clusters[date]))
    early = fl.compute_rolling_smoothed_clusters(
        y.loc[:DATES[150]], list(y.index[FIRST_ESTIMATION:151]),
        fl.LassoModel(model_type=fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO, span=SPAN,
                      n_clusters=N_CLUSTERS, **SMOOTHERS["similarity EWMA"]))
    assert all(early.clusters[d].equals(results["similarity EWMA"].clusters[d])
               for d in early.clusters)                      # later data never changes the past

    # --- the smoother updates are the documented formulas -------------------------------------
    corr_now, corr_before = y.iloc[:120].corr(), y.iloc[:100].corr()
    smoothed = fl.smooth_similarity_ewma(corr_now, corr_before, smoother_lambda=0.7)
    expected = 0.3 * corr_now.to_numpy() + 0.7 * corr_before.to_numpy()
    np.fill_diagonal(expected, 1.0)
    assert np.allclose(smoothed.to_numpy(), expected)
    distance = 1.0 - corr_now.to_numpy()
    previous = results["none"].clusters[DATES[119]]
    bonus = fl.apply_partition_distance_bonus(distance, previous, delta=0.05)
    same = co_membership(previous) & ~np.eye(len(NAMES), dtype=bool)
    assert np.allclose(bonus[same], np.maximum(distance[same] - 0.05, 0.0))
    assert np.allclose(bonus[~same & ~np.eye(len(NAMES), dtype=bool)],
                       distance[~same & ~np.eye(len(NAMES), dtype=bool)])

    # --- churn on this panel and over ten -----------------------------------------------------
    single = pd.DataFrame({name: churn(result) for name, result in results.items()}).T
    print(single)
    for result in results.values():                            # only a4 ever changes cluster
        for clusters in result.clusters.values():
            rest = clusters.drop("a4")
            assert pd.crosstab(rest, rest.index.str[0]).gt(0).sum(axis=1).eq(1).all()
    panels = churn_over_panels()
    print(panels.round(1))

    # quoted values
    assert single["a4 switches"].tolist() == [3, 3, 1, 1, 1]
    assert single["final switch, months after crossing"].tolist() == [21, 23, 20, 22, 21]
    assert single["label changes"].tolist() == [40, 28, 40, 37, 18]
    assert single.loc["none", "label changes"] > single.loc["none", "membership changes"]
    assert panels["a4 switches"].round(1).tolist() == [2.8, 2.0, 1.4, 1.0, 1.4]
    assert panels["final switch, months after crossing"].round(1).tolist() == [
        21.2, 22.4, 21.5, 24.8, 22.6]
    assert CALIBRATED_DELTA == 0.14 and round(WITHIN_BLOCK_RHO, 2) == 0.69
    assert np.isclose(1.0 - (1.0 - 2.0 / (SPAN + 1.0)), 2.0 / 37.0)


if __name__ == "__main__":
    main()
