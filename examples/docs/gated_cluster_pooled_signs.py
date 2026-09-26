"""Canonical example for docs/gated_cluster_pooled_signs.md.

A small version of the simulation design of the sign-pooling paper: responses in clusters that
share their active predictors and coefficient signs, independent standard-normal predictors, and
a low population R-squared. Signs are derived per response and by pooling within the known
clusters, and passed through the noise-floor gate for a grid of thresholds.

Independent references: the pooled slope, its standard error and the t-statistic of the
paper's closed form computed with NumPy; the per-response false-sign rate against its
normal-reference value 2 * Phi(-tau); and the signs derived inside ``LassoModel.fit`` against the
standalone function on the same demeaned arrays.
"""

import numpy as np
import pandas as pd
from scipy import stats

import factorlasso as fl

SEED = 20260929
N_CLUSTERS = 4
CLUSTER_SIZE = 6
N_PREDICTORS = 8
N_ACTIVE = 3
N_OBS = 60
TARGET_R2 = 0.10
N_PANELS = 100
TAUS = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]


def make_panel(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray,
                                                  pd.Series]:
    """Cluster-coherent sparse loadings, standard-normal predictors, demeaned panels."""
    n_responses = N_CLUSTERS * CLUSTER_SIZE
    beta = np.zeros((n_responses, N_PREDICTORS))
    clusters = np.repeat(np.arange(N_CLUSTERS), CLUSTER_SIZE)
    for g in range(N_CLUSTERS):
        active = rng.choice(N_PREDICTORS, N_ACTIVE, replace=False)
        signs = rng.choice([-1.0, 1.0], N_ACTIVE)
        members = clusters == g
        beta[np.ix_(members, active)] = signs * rng.uniform(0.5, 1.5, (members.sum(), N_ACTIVE))
    x = rng.standard_normal((N_OBS, N_PREDICTORS))
    signal = x @ beta.T
    noise_sd = np.sqrt(np.sum(beta ** 2, axis=1) * (1.0 - TARGET_R2) / TARGET_R2)
    y = signal + noise_sd * rng.standard_normal((N_OBS, n_responses))
    names = [f"asset_{k + 1}" for k in range(n_responses)]
    x = pd.DataFrame(x - x.mean(axis=0), columns=[f"f{j + 1}" for j in range(N_PREDICTORS)])
    y = pd.DataFrame(y - y.mean(axis=0), columns=names)
    return x, y, beta, pd.Series(clusters, index=names, name="cluster")


def derive_signs(x: pd.DataFrame, y: pd.DataFrame, clusters: pd.Series, tau: float,
                 pooled: bool) -> np.ndarray:
    """Gated sign matrix, per response or pooled within each known cluster."""
    x_np, y_np, labels = x.to_numpy(), y.to_numpy(), clusters.to_numpy()
    groups = ([[k] for k in range(y_np.shape[1])] if not pooled
              else [list(np.flatnonzero(labels == g)) for g in np.unique(labels)])
    signs = np.empty((y_np.shape[1], x_np.shape[1]))
    for members in groups:
        signs[members] = fl.derive_sign_constraints(x_np, y_np[:, members],
                                                    auto_sign_threshold_t=tau)
    return signs


def recovery_rates(signs: np.ndarray, beta: np.ndarray) -> dict:
    """Recovery, flip and abstention on active cells; false signs on null cells."""
    active = beta != 0.0
    truth = np.sign(beta)
    return {
        "recovery": float(np.mean(signs[active] == truth[active])),
        "flip": float(np.mean(signs[active] == -truth[active])),
        "abstention": float(np.mean(signs[active] == 0.0)),
        "false_sign": float(np.mean(signs[~active] != 0.0)),
    }


def gate_sweep(seed: int = SEED, n_panels: int = N_PANELS) -> pd.DataFrame:
    """Mean rates over redrawn panels for each threshold and derivation."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_panels):
        x, y, beta, clusters = make_panel(rng)
        for tau in TAUS:
            for pooled in (False, True):
                rates = recovery_rates(derive_signs(x, y, clusters, tau, pooled), beta)
                rows.append({"tau": tau, "derivation": "pooled" if pooled else "per response",
                             **rates})
    return pd.DataFrame(rows).groupby(["derivation", "tau"]).mean()


def main() -> None:
    rng = np.random.default_rng(SEED)
    x, y, beta, clusters = make_panel(rng)

    # --- the pooled slope and t-statistic of one cluster, by hand ------------------------------
    members = list(clusters.index[clusters == 0])
    y_cluster = y[members].to_numpy()
    xx = np.sum(x.to_numpy() ** 2, axis=0)
    size = len(members)
    slope = x.to_numpy().T @ y_cluster.sum(axis=1) / (size * xx)
    ssr = np.sum(y_cluster ** 2) - size * slope ** 2 * xx
    se = np.sqrt(ssr / (size * N_OBS - size) / (size * xx))
    t_stat = slope / se
    signs, slopes = fl.derive_sign_constraints(x, y[members], auto_sign_threshold_t=0.75,
                                               return_slopes=True)
    assert np.allclose(slopes.iloc[0].to_numpy(), slope)
    assert np.array_equal(signs.iloc[0].to_numpy(), np.where(np.abs(t_stat) >= 0.75,
                                                             np.sign(slope), 0.0))

    # --- the same derivation inside LassoModel, pooling within the supplied groups -------------
    model = fl.LassoModel(
        model_type=fl.LassoModelType.GROUP_LASSO,
        group_data=clusters,
        reg_lambda=1e-3,
        auto_sign_constraints=True,
        auto_sign_threshold_t=0.75,
    ).fit(x=x, y=y)
    assert np.array_equal(model.derived_signs_.to_numpy(),
                          derive_signs(x, y, clusters, 0.75, pooled=True))
    assert np.allclose(model.sign_t_stats_.loc[members[0]].to_numpy(), t_stat)

    # --- the gate across thresholds, per response and pooled -----------------------------------
    sweep = gate_sweep()
    print(sweep.round(3))
    per, pooled = sweep.loc["per response"], sweep.loc["pooled"]
    # per-response null cells are retained at about 2 * Phi(-tau)
    reference = 2.0 * stats.norm.sf(np.array(TAUS))
    assert np.allclose(per["false_sign"].to_numpy(), reference, atol=0.03)
    # pooling clears far more true signals at the default threshold, through lower abstention
    assert pooled.loc[0.75, "recovery"] > per.loc[0.75, "recovery"] + 0.2
    assert pooled.loc[0.75, "abstention"] < per.loc[0.75, "abstention"] - 0.2
    # and pays for it on null cells, which a higher threshold controls
    assert pooled.loc[0.75, "false_sign"] > per.loc[0.75, "false_sign"]
    assert pooled.loc[2.5, "false_sign"] < 0.1 and pooled.loc[2.5, "recovery"] > 0.6

    # quoted values
    assert round(float(per.loc[0.75, "recovery"]), 2) == 0.72
    assert round(float(pooled.loc[0.75, "recovery"]), 2) == 0.98
    assert round(float(per.loc[0.75, "abstention"]), 2) == 0.26
    assert round(float(pooled.loc[0.75, "abstention"]), 2) == 0.02
    assert round(float(per.loc[0.75, "false_sign"]), 2) == 0.47
    assert round(float(pooled.loc[0.75, "false_sign"]), 2) == 0.55
    assert round(float(pooled.loc[2.5, "recovery"]), 2) == 0.74
    assert round(float(pooled.loc[2.5, "false_sign"]), 2) == 0.04

    # --- validate_cluster_signs flags a predictor that disagrees with its predictor cluster ----
    predictor_clusters = np.array([0, 0, 0, 1, 1, 2, 2, 3])
    flagged = fl.validate_cluster_signs(x, y[members], predictor_clusters, warn=False)
    column_signs = np.sign(slope)
    for index in flagged:
        group = predictor_clusters == predictor_clusters[index]
        aggregate = x.to_numpy()[:, group].mean(axis=1)
        cluster_sign = np.sign(aggregate @ y_cluster.sum(axis=1))
        assert column_signs[index] != cluster_sign


if __name__ == "__main__":
    main()
