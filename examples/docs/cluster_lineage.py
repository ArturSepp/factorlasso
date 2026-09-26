"""Canonical example for docs/cluster_lineage.md.

Twenty-four monthly snapshots of a factor model hold three groups of four responses, an
equity-like, a rates-like and a credit-like group, and from month 13 a fourth group of three
high-yield responses. The raw cluster labels are shuffled on every date, the equity and credit
groups share one raw cluster in month 9, and one credit response joins the equity group from
month 17. The lineage analysis should give each group one persistent identity. The script checks
the identities and the lineage events, and compares the joint matcher with the per-transition
one. Synthetic snapshots; no covariance is estimated.
"""

import numpy as np
import pandas as pd

import factorlasso as fl

SEED = 20261008
N_DATES = 24
MERGE_DATE, BIRTH_DATE, MOVE_DATE = 8, 12, 16             # zero-based positions of the events
FACTORS = ["Equity", "Rates", "Credit"]
SIGMA = np.array([[0.0400, 0.0010, 0.0060],               # annualised factor covariance
                  [0.0010, 0.0025, 0.0005],
                  [0.0060, 0.0005, 0.0090]])
PROTOTYPES = {"equity": [0.95, 0.05, 0.10], "rates": [0.05, 0.85, 0.10],
              "credit": [0.25, 0.10, 0.90], "high yield": [0.60, -0.05, 0.60]}
GROUPS = {"equity": ["e1", "e2", "e3", "e4"], "rates": ["r1", "r2", "r3", "r4"],
          "credit": ["c1", "c2", "c3", "c4"], "high yield": ["h1", "h2", "h3"]}
DATES = pd.date_range("2024-01-31", periods=N_DATES, freq="ME")


def asset_betas(seed: int = SEED) -> pd.DataFrame:
    """Each response's factor betas: its group's prototype plus a small perturbation."""
    rng = np.random.default_rng(seed)
    rows = {asset: np.asarray(PROTOTYPES[group]) + 0.03 * rng.standard_normal(3)
            for group, assets in GROUPS.items() for asset in assets}
    return pd.DataFrame(rows, index=FACTORS).T


def true_groups(position: int) -> dict:
    """Group membership at one date, before raw labels are assigned."""
    groups = {name: list(assets) for name, assets in GROUPS.items()}
    if position < BIRTH_DATE:
        del groups["high yield"]
    if position >= MOVE_DATE:                              # c4 joins the equity group for good
        groups["credit"].remove("c4")
        groups["equity"].append("c4")
    if position == MERGE_DATE:                             # one raw cluster for equity and credit
        groups["equity"] = groups["equity"] + groups.pop("credit")
    return groups


def make_rolling(seed: int = SEED) -> tuple[fl.RollingFactorCovarData, pd.DataFrame]:
    """Snapshots with shuffled raw labels, and the raw label of each group's anchor per date."""
    rng = np.random.default_rng(seed + 1)
    betas = asset_betas(seed)
    snapshots, raw = {}, {}
    for position, date in enumerate(DATES):
        groups = true_groups(position)
        labels = rng.permutation(len(groups)) + 1
        clusters = pd.Series({asset: int(label) for label, assets in zip(labels, groups.values())
                              for asset in assets}).sort_index()
        members = clusters.index
        variances = pd.DataFrame({fl.VarianceColumns.RESIDUAL_VARS.value: 0.01,
                                  fl.VarianceColumns.CLUSTER.value: clusters}, index=members)
        snapshots[date] = fl.CurrentFactorCovarData(
            x_covar=pd.DataFrame(SIGMA, index=FACTORS, columns=FACTORS),
            y_betas=betas.loc[members], y_variances=variances, clusters=clusters)
        raw[date] = {name: clusters[assets[0]] for name, assets in GROUPS.items()
                     if assets[0] in clusters.index}
    return fl.RollingFactorCovarData(data=snapshots), pd.DataFrame(raw).T


def lineage(covar_data: fl.RollingFactorCovarData, method: str = "mcf") -> fl.RiskClusterReport:
    """Persistent tracks with the default link gate and bridge settings."""
    return fl.analyze_cluster_lineage(covar_data, method=method)


def derived_by_group(report: fl.RiskClusterReport) -> pd.DataFrame:
    """The derived track of each group's anchor response at each date."""
    panel = report.to_membership_panel()
    anchors = {name: assets[0] for name, assets in GROUPS.items()}
    return pd.DataFrame({name: panel[anchor] for name, anchor in anchors.items()})


def main() -> None:
    covar_data, raw = make_rolling()
    report = lineage(covar_data)
    tracks = derived_by_group(report)
    print(tracks.nunique())

    # --- the raw labels move; the derived tracks do not -----------------------------------------
    assert raw["equity"].nunique() > 1 and raw["rates"].nunique() > 1
    assert tracks.nunique().tolist() == [1, 1, 2, 1]             # credit is absorbed in month 9
    credit = tracks["credit"]
    assert credit.drop(DATES[MERGE_DATE]).nunique() == 1          # ... and keeps its identity
    assert credit[DATES[MERGE_DATE]] == tracks["equity"][DATES[MERGE_DATE]]
    assert tracks["high yield"].isna().sum() == BIRTH_DATE
    moved = report.to_membership_panel()["c4"]
    assert (moved[DATES[MOVE_DATE]:] == tracks["equity"].iloc[0]).all()

    # --- events: a bridge around the merge, a birth for the new group ---------------------------
    events = report.lineage
    credit_id = credit.iloc[0]
    bridge = events[(events["event"] == "bridge") & (events["child_id"] == credit_id)]
    assert bridge["date"].tolist() == [DATES[MERGE_DATE + 1]]
    births = events[events["event"] == "birth"]
    assert DATES[BIRTH_DATE] in births["date"].tolist()

    # --- the per-transition matcher fragments the absorbed track --------------------------------
    hungarian = derived_by_group(lineage(covar_data, method="hungarian"))
    assert hungarian["credit"].nunique() > credit.nunique()
    print("tracks:", len(report.tracks), "joint;", len(lineage(covar_data, "hungarian").tracks),
          "per transition")

    # --- classification and labels -------------------------------------------------------------
    classes = report.classification
    print(classes[["track_type", "persistence", "coverage", "stability_label"]])
    labels = report.factor_labels()
    print(labels.to_string().encode("ascii", "replace").decode())
    assert labels[tracks["equity"].iloc[0]].startswith("Equity high-")
    assert classes.loc[tracks["equity"].iloc[0], "track_type"] == "Equity-HighBeta"
    assert classes.loc[tracks["rates"].iloc[0], "track_type"] == "Rates"
    assert classes.loc[credit_id, "track_type"] == "Credit"
    assert classes.loc[tracks["high yield"].dropna().iloc[0], "persistence"] == "Episodic"

    # quoted values
    ids = [tracks[group].dropna().iloc[0] for group in GROUPS]
    assert classes.loc[ids, "coverage"].round(2).tolist() == [1.0, 1.0, 0.96, 0.5]
    assert classes.loc[ids, "persistence"].tolist() == ["Core", "Core", "Core", "Episodic"]
    assert classes.loc[ids, "track_type"].tolist()[3] == "Equity-Core"
    assert labels[ids].tolist() == ["Equity high-β · high-vol",
                                    "Rates long-duration · low-vol",
                                    "Credit · mid-vol",
                                    "Equity core · high-vol"]
    assert len(report.tracks) == 4 and len(lineage(covar_data, "hungarian").tracks) == 5


if __name__ == "__main__":
    main()
