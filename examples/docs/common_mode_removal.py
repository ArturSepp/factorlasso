"""Canonical example for docs/common_mode_removal.md.

Twelve responses form three sectors of four over 240 months. Every response also loads on a
market factor, with betas of 0.4, 0.8, 1.2 and 1.6 inside each sector. When the market is strong,
the correlation between two responses is set mostly by their betas, and Ward clustering groups
responses by beta instead of by sector. Removing the dominant eigencomponent of the correlation
matrix before clustering restores the sectors; without a market factor, the same removal takes
away a sector instead. The script checks the transform against its formula and sweeps the market
volatility. Synthetic data.
"""

import numpy as np
import pandas as pd

import factorlasso as fl

SEED = 20261005
N_OBS = 240
MARKET_VOL = 0.06
SECTOR_VOL = 0.02
RESIDUAL_VOL = 0.02
N_PANELS = 50
MARKET_VOLS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1]
NAMES = [f"{sector}{k}" for sector in "abc" for k in range(1, 5)]
SECTORS = pd.Series(np.repeat(["a", "b", "c"], 4), index=NAMES, name="sector")
BETAS = np.tile([0.4, 0.8, 1.2, 1.6], 3)


def make_panel(market_vol: float = MARKET_VOL, seed: int = SEED
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sector responses with uneven market betas; the market factor as x."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2006-01-31", periods=N_OBS, freq="ME")
    market = market_vol * rng.standard_normal(N_OBS)
    sectors = SECTOR_VOL * rng.standard_normal((N_OBS, 3))
    y = (np.outer(market, BETAS) + sectors[:, np.repeat([0, 1, 2], 4)]
         + RESIDUAL_VOL * rng.standard_normal((N_OBS, len(NAMES))))
    return (pd.DataFrame({"market": market}, index=dates),
            pd.DataFrame(y, index=dates, columns=NAMES))


def recovers_sectors(clusters: pd.Series) -> bool:
    """True when the partition equals the three sectors, up to relabelling."""
    table = pd.crosstab(clusters, SECTORS)
    return bool(table.shape == (3, 3) and ((table > 0).sum(axis=1) == 1).all()
                and ((table > 0).sum(axis=0) == 1).all())


def partitions(y: pd.DataFrame) -> tuple:
    """Three-cluster Ward partitions of the correlation and of its common-mode residual."""
    corr = y.corr()
    removal = fl.remove_first_principal_component(corr)
    raw, _, _ = fl.compute_clusters_from_corr_matrix(corr, n_clusters=3)
    residual, _, _ = fl.compute_clusters_from_corr_matrix(removal.correlation, n_clusters=3)
    return raw, residual, removal


def recovery_by_market_vol() -> pd.DataFrame:
    """Share of panels whose three clusters are the sectors, with and without the removal."""
    rows = {}
    for market_vol in MARKET_VOLS:
        hits = np.zeros(2)
        shares = []
        for k in range(N_PANELS):
            raw, residual, removal = partitions(make_panel(market_vol, SEED + k)[1])
            hits += [recovers_sectors(raw), recovers_sectors(residual)]
            shares.append(removal.removed_variance_share)
        rows[market_vol] = {"correlation": hits[0] / N_PANELS,
                            "common mode removed": hits[1] / N_PANELS,
                            "removed share": float(np.mean(shares))}
    return pd.DataFrame(rows).T.rename_axis("market_vol")


def main() -> None:
    x, y = make_panel()

    # --- the transform is the documented rank-one deflation, restandardised --------------------
    corr = y.corr()
    removal = fl.remove_first_principal_component(corr)
    values, vectors = np.linalg.eigh(corr.to_numpy())
    residual = corr.to_numpy() - values[-1] * np.outer(vectors[:, -1], vectors[:, -1])
    scale = np.sqrt(np.diag(residual))
    assert np.allclose(removal.correlation.to_numpy(), residual / np.outer(scale, scale))
    assert np.isclose(removal.removed_variance_share, values[-1] / len(NAMES))
    assert np.isclose(removal.eigengap, values[-1] - values[-2])
    assert np.isclose(removal.minimum_residual_variance, np.diag(residual).min())
    assert fl.apply_cluster_correlation_transform(corr, "none") is corr

    # --- a strong market sorts by beta; the residual sorts by sector ---------------------------
    raw, residual, _ = partitions(y)
    print("correlation:", raw.tolist())
    print("common mode removed:", residual.tolist())
    assert not recovers_sectors(raw) and recovers_sectors(residual)
    sizes = raw.value_counts()
    singletons = sizes.index[sizes == 1]
    assert sorted(sizes) == [1, 1, 10]                      # two responses split off, ten merge
    assert (BETAS[raw.isin(singletons).to_numpy()] == 0.4).all()   # the least exposed ones

    # LassoModel applies the same removal to the dependence matrix inside fit
    model = fl.LassoModel(
        model_type=fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
        cluster_correlation_transform="remove_pc1",
        n_clusters=3,
    ).fit(x=x, y=y)
    assert model.clusters_.equals(residual)

    # --- the sweep: removal helps under a dominant mode and hurts without one ------------------
    sweep = recovery_by_market_vol()
    print(sweep.round(2))
    assert sweep.loc[0.06].round(2).tolist() == [0.02, 1.0, 0.78]
    assert sweep.loc[0.0].round(2).tolist() == [1.0, 0.4, 0.23]
    assert (sweep.loc[0.05:, "common mode removed"] == 1.0).all()

    # quoted values
    assert round(removal.removed_eigenvalue, 2) == 9.03
    assert round(removal.removed_variance_share, 2) == 0.75
    assert np.round(np.linalg.eigvalsh(removal.correlation.to_numpy())[::-1][:3], 2).tolist() == [
        3.1, 2.9, 1.32]


if __name__ == "__main__":
    main()
