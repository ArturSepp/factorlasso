"""Canonical example for docs/cooperative_lasso.md.

Two clusters of four responses load on two factors. In the first cluster, three members have a
positive loading on the first factor and one member, the "rogue", a negative loading. Three
penalties are compared at one common penalty strength on the cluster-by-factor blocks:

* the group LASSO block penalty of FCGL, which ignores signs;
* the cooperative LASSO, which penalises the positive and negative parts of a block separately,
  so that a mixed-sign block pays more;
* FCGL with a hard sign pooled over the cluster, which forces the rogue to the cluster's sign.

The factors are made exactly orthonormal (X'X / T = I), which gives every penalty a closed form:
a block's loadings are the least-squares loadings shrunk by a group soft threshold, applied to
the whole block (group LASSO) or separately to its positive and negative parts (cooperative
LASSO). The script checks the solver against these closed forms. Synthetic data.
"""

import numpy as np
import pandas as pd

import factorlasso as fl

SEED = 20261001
N_OBS = 120
REG_LAMBDA = 0.3
TRUE_BETA = np.array([
    [0.8, 0.2],
    [0.7, 0.3],
    [0.9, 0.1],
    [-0.3, 0.2],      # the rogue member of the first cluster
    [0.1, 0.8],
    [0.2, 0.7],
    [0.1, 0.9],
    [0.2, 0.6],
])
CLUSTERS = pd.Series([1, 1, 1, 1, 2, 2, 2, 2],
                     index=[f"asset_{k + 1}" for k in range(8)], name="cluster")


def make_panel(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two exactly orthonormal factors, X'X / T = I, and eight responses."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((N_OBS, 2))
    raw -= raw.mean(axis=0)
    q, _ = np.linalg.qr(raw)
    x = pd.DataFrame(q * np.sqrt(N_OBS), columns=["f1", "f2"])
    noise = 0.3 * rng.standard_normal((N_OBS, len(CLUSTERS)))
    y = pd.DataFrame(x.to_numpy() @ TRUE_BETA.T + noise, columns=CLUSTERS.index)
    return x, y - y.mean()


def cooperative_penalty(block: np.ndarray) -> float:
    """Norm of the positive part plus norm of the negative part of a block."""
    return float(np.linalg.norm(np.maximum(block, 0.0)) + np.linalg.norm(np.minimum(block, 0.0)))


def fit_three(x: pd.DataFrame, y: pd.DataFrame, reg_lambda: float = REG_LAMBDA) -> dict:
    """Group LASSO blocks, cooperative LASSO blocks, and blocks with a hard pooled sign."""
    x_np, y_np = x.to_numpy(), y.to_numpy()
    groups = fl.set_group_loadings(group_data=CLUSTERS).to_numpy()
    group = fl.solve_group_lasso_cvx_problem(
        x=x_np, y=y_np, group_loadings=groups, reg_lambda=reg_lambda,
        block_mode="cluster_factor",
    )
    cooperative = fl.solve_cooperative_group_lasso_cvx_problem(
        x=x_np, y=y_np, group_loadings=groups, reg_lambda=reg_lambda,
    )
    hard = fl.LassoModel(
        model_type=fl.LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
        reg_lambda=reg_lambda,
        auto_sign_constraints=True,
        auto_sign_threshold_t=None,
    ).fit(x=x, y=y, external_clusters=CLUSTERS)
    return {
        "group LASSO": pd.DataFrame(group.estimated_beta, index=y.columns, columns=x.columns),
        "cooperative LASSO": pd.DataFrame(cooperative.estimated_beta, index=y.columns,
                                          columns=x.columns),
        "hard pooled sign": hard.coef_,
    }


REG_LAMBDAS = np.linspace(0.0, 0.8, 17)


def rogue_path(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Loading of the rogue and of the first coherent member along the penalty grid."""
    rows = []
    for reg_lambda in REG_LAMBDAS[1:]:
        fits = fit_three(x, y, reg_lambda=float(reg_lambda))
        for name, fit in fits.items():
            rows.append({"reg_lambda": reg_lambda, "penalty": name,
                         "rogue": fit.loc["asset_4", "f1"], "coherent": fit.loc["asset_1", "f1"]})
    return pd.DataFrame(rows)


def closed_form(z: np.ndarray, threshold: float, split_signs: bool) -> np.ndarray:
    """Group soft threshold of a block, whole or separately on its positive and negative parts."""
    out = np.zeros_like(z)
    parts = [z > 0.0, z < 0.0] if split_signs else [np.ones_like(z, dtype=bool)]
    for part in parts:
        norm = np.linalg.norm(z[part])
        if norm > 0.0:
            out[part] = z[part] * max(0.0, 1.0 - threshold / (2.0 * norm))
    return out


def main() -> None:
    # --- the penalty: coherent blocks pay their norm, mixed blocks pay more --------------------
    coherent, mixed = np.array([0.6, 0.8]), np.array([0.6, -0.8])
    assert np.isclose(cooperative_penalty(coherent), np.linalg.norm(coherent))
    assert cooperative_penalty(mixed) > np.linalg.norm(mixed)
    assert np.isclose(cooperative_penalty(mixed), 1.4)             # |0.6| + |0.8| = 1.4 > 1.0
    balanced = np.array([1.0, -1.0]) / np.sqrt(2.0)
    assert np.isclose(cooperative_penalty(balanced) / np.linalg.norm(balanced), np.sqrt(2.0))

    # --- three penalties on one panel ----------------------------------------------------------
    x, y = make_panel()
    assert np.allclose(x.T @ x / N_OBS, np.eye(2))
    fits = fit_three(x, y)
    z = (x.T @ y / N_OBS).T.to_numpy()                            # least-squares loadings
    table = pd.DataFrame({"least squares": z[:4, 0]} | {
        name: fit.iloc[:4, 0].to_numpy() for name, fit in fits.items()},
        index=y.columns[:4])
    print(table.round(3))

    # closed forms of the two block penalties, with the cluster weight sqrt(|g| / G) = sqrt(2)
    threshold = REG_LAMBDA * np.sqrt(4 / 2)
    for cluster in (1, 2):
        members = (CLUSTERS == cluster).to_numpy()
        for j in range(2):
            block = z[members, j]
            group_expected = closed_form(block, threshold, split_signs=False)
            coop_expected = closed_form(block, threshold, split_signs=True)
            assert np.allclose(fits["group LASSO"].to_numpy()[members, j], group_expected,
                               atol=1e-4)
            assert np.allclose(fits["cooperative LASSO"].to_numpy()[members, j], coop_expected,
                               atol=1e-4)

    # the rogue: least squares keeps its negative loading; the group LASSO shrinks it with its
    # block; the cooperative LASSO shrinks it harder, as the lone member of its sign group; the
    # hard pooled sign forces it to zero
    rogue = table.loc["asset_4"]
    assert rogue["least squares"] < rogue["group LASSO"] < rogue["cooperative LASSO"] < 0.0
    assert abs(rogue["hard pooled sign"]) < 1e-4
    # the coherent members are shrunk about equally; the rogue no longer props up their norm
    coherent_members = table.iloc[:3]
    assert np.allclose(coherent_members["cooperative LASSO"], coherent_members["group LASSO"],
                       atol=0.02)

    # quoted values
    assert [round(float(v), 2) for v in rogue] == [-0.35, -0.30, -0.14, 0.0]
    assert [round(float(v), 3) for v in table.loc["asset_1", ["group LASSO",
                                                               "cooperative LASSO"]]] == [0.648,
                                                                                          0.644]


if __name__ == "__main__":
    main()
