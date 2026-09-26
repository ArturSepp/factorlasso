"""Canonical example for docs/cluster_discovery.md.

Twelve responses form three blocks of four over 240 months, each block driven by its own
factor, all sharing a weak common factor. On six dates a joint shock of 25% hits the eight
responses of blocks b and c together, as a crisis would. The script clusters the responses
under the three dependence measures and the three distance transforms, and checks the Gerber
statistic against a direct count, the transforms against their formulas, and ``LassoModel``
against the stand-alone clustering function. Synthetic data.
"""

import numpy as np
import pandas as pd

import factorlasso as fl

SEED = 20261004
N_OBS = 240
BLOCK_VOL = 0.015
COMMON_VOL = 0.02
RESIDUAL_VOL = 0.02
SHOCK = 0.25
N_SHOCKS = 6
N_CLUSTERS = 3
NAMES = [f"{block}{k}" for block in "abc" for k in range(1, 5)]
TRUE_BLOCKS = pd.Series(np.repeat(["a", "b", "c"], 4), index=NAMES, name="block")


def make_panel(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Block responses with six joint shocks on blocks b and c; the common factor as x."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2006-01-31", periods=N_OBS, freq="ME")
    blocks = BLOCK_VOL * rng.standard_normal((N_OBS, 3))
    common = COMMON_VOL * rng.standard_normal(N_OBS)
    y = (blocks[:, np.repeat([0, 1, 2], 4)] + 0.5 * common[:, None]
         + RESIDUAL_VOL * rng.standard_normal((N_OBS, len(NAMES))))
    shock_dates = rng.choice(N_OBS, N_SHOCKS, replace=False)
    shock_signs = rng.choice([-1.0, 1.0], N_SHOCKS)
    y[shock_dates[:, None], np.arange(4, 12)[None, :]] += (shock_signs * SHOCK)[:, None]
    x = pd.DataFrame({"common": common}, index=dates)
    return x, pd.DataFrame(y, index=dates, columns=NAMES)


def dependence(y: pd.DataFrame, measure: str) -> pd.DataFrame:
    """The clustering dependence matrix of the de-meaned responses, as LassoModel builds it."""
    values = fl.compute_dependence_matrix((y - y.mean()).to_numpy(), dependence_measure=measure)
    return pd.DataFrame(values, index=y.columns, columns=y.columns)


def discover(y: pd.DataFrame, measure: str, transform: str = "one_minus_rho",
             cutoff_fraction: float = 0.5, n_clusters: int | None = None) -> tuple:
    """Partition, linkage and cut height for one dependence measure and distance transform."""
    return fl.compute_clusters_from_corr_matrix(
        dependence(y, measure),
        cutoff_fraction=cutoff_fraction,
        linkage_method="ward",
        distance_transform=transform,
        n_clusters=n_clusters,
    )


def recovers_blocks(clusters: pd.Series) -> bool:
    """True when the partition equals the three true blocks, up to relabelling."""
    table = pd.crosstab(clusters, TRUE_BLOCKS)
    return bool(table.shape == (3, 3) and ((table > 0).sum(axis=1) == 1).all()
                and ((table > 0).sum(axis=0) == 1).all())


def gerber_by_counting(y: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    """The equal-weight Gerber statistic, counted pair by pair."""
    a = (y - y.mean()).to_numpy()
    limit = threshold * a.std(axis=0)
    up, down = a >= limit, a <= -limit
    n_assets = a.shape[1]
    out = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(n_assets):
            if i == j:
                continue
            concordant = np.sum(up[:, i] & up[:, j]) + np.sum(down[:, i] & down[:, j])
            discordant = np.sum(up[:, i] & down[:, j]) + np.sum(down[:, i] & up[:, j])
            neither = np.sum(~up[:, i] & ~down[:, i] & ~up[:, j] & ~down[:, j])
            out[i, j] = (concordant - discordant) / (len(a) - neither)
    return out


FRACTIONS = np.round(np.linspace(0.05, 1.0, 20), 2)


def cluster_counts(y: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Realised cluster count against cutoff_fraction under the three distance transforms."""
    return pd.DataFrame({
        transform: [discover(y, measure, transform, float(f))[0].nunique() for f in FRACTIONS]
        for transform in ("one_minus_rho", "chord", "arccos")
    }, index=pd.Index(FRACTIONS, name="cutoff_fraction"))


def block_correlations(y: pd.DataFrame) -> pd.DataFrame:
    """Mean dependence within block b and between blocks b and c, per measure."""
    b, c = ["b1", "b2", "b3", "b4"], ["c1", "c2", "c3", "c4"]
    rows = {}
    for measure in ("pearson", "spearman", "gerber"):
        matrix = dependence(y, measure)
        within = matrix.loc[b, b].to_numpy()[~np.eye(4, dtype=bool)].mean()
        rows[measure] = {"within b": within, "b with c": matrix.loc[b, c].to_numpy().mean()}
    return pd.DataFrame(rows).T


def main() -> None:
    x, y = make_panel()

    # --- the dependence measures: the shocks dominate Pearson, not ranks or co-movement counts --
    table = block_correlations(y)
    print(table.round(2))
    assert np.allclose(dependence(y, "gerber").to_numpy(), gerber_by_counting(y), atol=1e-12)

    partitions = {m: discover(y, m, n_clusters=N_CLUSTERS)[0]
                  for m in ("pearson", "spearman", "gerber")}
    for measure, clusters in partitions.items():
        print(measure, clusters.tolist())
    pearson = partitions["pearson"]
    assert not recovers_blocks(pearson)
    assert pearson[4:].nunique() == 1 and pearson[:4].nunique() == 2   # b and c merge, a splits
    assert recovers_blocks(partitions["spearman"]) and recovers_blocks(partitions["gerber"])

    # --- the fractional cut does not port across measures or transforms ------------------------
    defaults = {m: discover(y, m)[0].nunique() for m in ("pearson", "spearman", "gerber")}
    assert defaults == {"pearson": 5, "spearman": 12, "gerber": 12}
    counts = cluster_counts(y, "spearman")
    print(counts.T)
    assert counts.loc[0.75].tolist() == [3, 10, 6]
    assert recovers_blocks(discover(y, "spearman", cutoff_fraction=0.75)[0])
    assert counts.loc[1.0].eq(3).all()                  # the top Ward merges exceed every distance
    _, linkage, _ = discover(y, "spearman", cutoff_fraction=1.0)
    rho_s = dependence(y, "spearman").to_numpy()
    assert (linkage[-2:, 2] > (1.0 - rho_s).max()).all()
    for f in FRACTIONS:                                  # sqrt(f) under CHORD matches 1 - rho at f
        assert discover(y, "spearman", "chord", float(np.sqrt(f)))[0].equals(
            discover(y, "spearman", "one_minus_rho", float(f))[0])

    # the transforms are the documented functions of the correlation
    rho = dependence(y, "spearman").to_numpy()
    off = ~np.eye(len(rho), dtype=bool)
    for transform, formula in (("one_minus_rho", 1.0 - rho),
                               ("chord", np.sqrt(2.0 * (1.0 - rho))),
                               ("arccos", np.arccos(np.clip(rho, -1.0, 1.0)))):
        _, _, cut = discover(y, "spearman", transform)
        assert np.isclose(cut, 0.5 * formula[off].max())
    # single linkage reads only the order of the distances: the same tree under every transform
    trees = [fl.compute_clusters_from_corr_matrix(dependence(y, "spearman"),
                                                  linkage_method="single",
                                                  distance_transform=t)[1][:, :2]
             for t in ("one_minus_rho", "chord", "arccos")]
    assert all(np.array_equal(trees[0], tree) for tree in trees[1:])

    # --- LassoModel discovers the same partition inside fit -----------------------------------
    model = fl.LassoModel(
        model_type=fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
        dependence_measure="spearman",
        n_clusters=N_CLUSTERS,
    ).fit(x=x, y=y)
    assert model.clusters_.equals(partitions["spearman"])

    # quoted values
    assert table.round(2).to_dict("list") == {"within b": [0.81, 0.45, 0.24],
                                               "b with c": [0.74, 0.2, 0.09]}


if __name__ == "__main__":
    main()
