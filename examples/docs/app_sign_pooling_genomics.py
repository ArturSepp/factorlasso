"""Canonical example for docs/app_sign_pooling_genomics.md.

The sign-pooling paper (Section 5) applies HCGL with gated cluster-pooled signs to the yeast
eQTL cross of Brem and Kruglyak (2005): 64 MAPK genes, 202 screened markers, 112 segregants. The
fit is archived: its results are committed under ``papers/sign_pooling_2026/replication/results``
and were produced by ``eqtl_pipeline.py`` with the versions pinned in its ``requirements.txt``.
This script does not refit. It reads the committed results and recomputes the statistics the
case study quotes: cluster sizes and purity, the adjusted Rand index (by its formula, not a
library), the hotspot counts, the gate's abstention share and the prediction parity.
"""

from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[2] / "papers/sign_pooling_2026/replication/results"
CONFIGURATION = {                        # eqtl_pipeline.fit_model, as archived in the paper
    "model_type": "HIERARCHICAL_CLUSTER_GROUP_LASSO",
    "reg_lambda": 0.3,
    "cutoff_fraction": 0.7,
    "auto_sign_constraints": True,
    "auto_sign_threshold_t": 2.0,
    "auto_sign_variance": "independent",
    "demean": True,
}


def load_results() -> dict:
    """The committed result tables of the eQTL application."""
    return {
        "clusters": pd.read_csv(RESULTS / "clusters.csv"),
        "hotspots": pd.read_csv(RESULTS / "hotspots.csv"),
        "signs": pd.read_csv(RESULTS / "derived_signs.csv", index_col=0),
        "parity": pd.read_csv(RESULTS / "parity.csv", index_col=0),
        "summary": pd.read_csv(RESULTS / "summary.csv", index_col=0).iloc[:, 0],
    }


def adjusted_rand_index(labels: pd.Series, reference: pd.Series) -> float:
    """Hubert and Arabie's adjusted Rand index from the contingency table."""
    table = pd.crosstab(labels, reference).to_numpy()
    pairs = sum(comb(int(n), 2) for n in table.ravel())
    rows = sum(comb(int(n), 2) for n in table.sum(axis=1))
    cols = sum(comb(int(n), 2) for n in table.sum(axis=0))
    expected = rows * cols / comb(int(table.sum()), 2)
    return (pairs - expected) / (0.5 * (rows + cols) - expected)


def cluster_table(clusters: pd.DataFrame) -> pd.DataFrame:
    """Size, dominant sub-pathway and purity of each discovered cluster."""
    rows = {}
    for cluster, members in clusters.groupby("cluster"):
        counts = members["module"].value_counts()
        rows[cluster] = {"genes": len(members), "dominant": counts.index[0],
                         "purity": counts.iloc[0] / len(members)}
    return pd.DataFrame(rows).T


def main() -> None:
    results = load_results()
    clusters, hotspots, signs = results["clusters"], results["hotspots"], results["signs"]

    # --- clusters against the MAPK sub-pathways ------------------------------------------------
    table = cluster_table(clusters)
    print(table)
    assert len(clusters) == 64 and len(table) == 9
    assert table["genes"].tolist() == [5, 8, 4, 7, 7, 6, 13, 4, 10]
    ari = adjusted_rand_index(clusters["cluster"], clusters["module"])
    purity = float(table["purity"].mean())
    assert np.isclose(ari, results["summary"]["ari"])
    assert np.isclose(purity, results["summary"]["purity"])
    assert [round(ari, 3), round(purity, 3)] == [0.114, 0.614]
    pheromone = table[table["dominant"].str.startswith("pheromone")]["purity"]
    assert sorted(pheromone.round(2).tolist()) == [0.9, 1.0]

    # --- gated signs and hotspots --------------------------------------------------------------
    assert signs.shape == (64, 202) and len(hotspots) == 202
    zero_share = float((signs == 0).to_numpy().mean())
    top = hotspots.sort_values("hub_count", ascending=False)
    print(top.head(6).to_string(index=False))
    assert top.iloc[0][["chr", "hub_count"]].tolist() == [14.0, 42.0]
    above_30 = sorted(top.loc[top["hub_count"] > 30, "chr"].astype(int).unique().tolist())
    assert above_30 == [3, 5, 7, 12, 14]
    assert round(zero_share, 2) == 0.59
    assert round(float(results["summary"]["sign_coherence"]), 2) == 0.93

    # --- prediction parity (three-fold cross-validation) ---------------------------------------
    medians = results["parity"]["oos_r2_median"].round(3)
    assert medians.to_dict() == {"HCGL": 0.038, "FCGL": 0.088, "LASSO": 0.118}


if __name__ == "__main__":
    main()
