"""Canonical example for docs/adaptive_penalty_weights.md.

Six responses load on six factors with large (1.0), small (0.3) and zero loadings, over 60
months. A LASSO with derived signs is fitted with and without adaptive penalty weights at one
common penalty. The adaptive weights lighten the penalty on cells with strong univariate
evidence, so large loadings are shrunk less, and make it heavier on weak cells.

Independent references:

* the weights against ``1 / max(|slope|, floor) ** gamma`` recomputed from the fitted slopes;
* the adaptive fit against the plain LASSO on a rescaled design, response by response: dividing
  factor ``j`` by the weight ``W_kj`` turns the weighted penalty into an ordinary one, and the
  loadings are recovered by dividing back (Zou, 2006);
* the FCGL block weights against the root-mean-square of the cell weights within each cluster.

The data are synthetic decimal monthly returns with equal weights.
"""

import numpy as np
import pandas as pd

import factorlasso as fl

SEED = 20260930
N_OBS = 60
FACTOR_VOL = 0.04
RESIDUAL_VOL = 0.02
REG_LAMBDA = 2e-4
FLOOR = 0.5            # the production floor of the JSS study
GAMMA = 1.0
TAU = 1.0
TRUE_BETA = np.array([
    [1.0, 0.3, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.3, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.3, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.3, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.3],
    [0.3, 0.0, 1.0, 0.0, 0.0, 0.0],
])


def make_panel(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Six independent factors and six responses with large, small and zero loadings."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-31", periods=N_OBS, freq="ME")
    x = pd.DataFrame(FACTOR_VOL * rng.standard_normal((N_OBS, 6)), index=dates,
                     columns=[f"f{j + 1}" for j in range(6)])
    noise = RESIDUAL_VOL * rng.standard_normal((N_OBS, 6))
    y = pd.DataFrame(x.to_numpy() @ TRUE_BETA.T + noise, index=dates,
                     columns=[f"asset_{k + 1}" for k in range(6)])
    return x, y


def fit(x: pd.DataFrame, y: pd.DataFrame, adaptive: bool,
        model_type: fl.LassoModelType = fl.LassoModelType.LASSO) -> fl.LassoModel:
    """LASSO with gated derived signs, with or without adaptive penalty weights."""
    model = fl.LassoModel(
        model_type=model_type,
        reg_lambda=REG_LAMBDA,
        auto_sign_constraints=True,
        auto_sign_threshold_t=TAU,
        auto_sign_adaptive_weights=adaptive,
        auto_sign_adaptive_gamma=GAMMA,
        auto_sign_adaptive_floor=FLOOR,
    )
    return model.fit(x=x, y=y)


def weight_curve(slopes: np.ndarray, gamma: float, floor: float) -> np.ndarray:
    """The adaptive weight as a function of the absolute univariate slope."""
    return 1.0 / np.maximum(np.abs(slopes), floor) ** gamma


def error_summary(model: fl.LassoModel) -> dict:
    """Mean shrinkage of the large and small loadings, and loadings kept on true zeros."""
    beta = model.coef_.to_numpy()
    large, small, zero = TRUE_BETA == 1.0, TRUE_BETA == 0.3, TRUE_BETA == 0.0
    return {
        "large_shrinkage": float(np.mean(TRUE_BETA[large] - beta[large])),
        "small_shrinkage": float(np.mean(TRUE_BETA[small] - beta[small])),
        "kept_on_zeros": int(np.sum(np.abs(beta[zero]) > 1e-3)),
    }


def main() -> None:
    x, y = make_panel()
    plain, adaptive = fit(x, y, adaptive=False), fit(x, y, adaptive=True)

    # the cell weights recomputed from the fitted slopes and detected signs
    slopes, detected = adaptive.sign_slopes_.to_numpy(), adaptive.detected_signs_.to_numpy()
    expected = np.where(detected == 0.0, 1.0, weight_curve(slopes, GAMMA, FLOOR))
    weights = adaptive.sign_penalty_weights_
    assert np.allclose(weights.to_numpy(), expected)
    print(weights.round(2))

    # reference: the plain LASSO on a design rescaled by 1 / W_kj, one response at a time
    for k, asset in enumerate(y.columns):
        w = weights.loc[asset].to_numpy()
        rescaled = fl.LassoModel(
            model_type=fl.LassoModelType.LASSO,
            reg_lambda=REG_LAMBDA,
            factors_beta_loading_signs=adaptive.derived_signs_.loc[[asset]],
        ).fit(x=x / w, y=y[[asset]])
        theta = rescaled.coef_.loc[asset].to_numpy()
        # to interior-point accuracy: cells at the kink of the penalty agree to 1e-3
        assert np.allclose(theta / w, adaptive.coef_.loc[asset].to_numpy(), atol=1e-3)

    # the FCGL block weight is the root-mean-square of the ungated cell weights in the block
    fcgl = fit(x, y, adaptive=True, model_type=fl.LassoModelType.FACTOR_CLUSTER_GROUP_LASSO)
    cell = fcgl.sign_penalty_weights_.to_numpy()
    active = fcgl.detected_signs_.to_numpy() != 0.0
    labels = fcgl.clusters_.loc[y.columns].to_numpy()
    # the rows of the block weights follow the columns of the cluster-membership matrix
    order = fl.set_group_loadings(group_data=fcgl.clusters_.loc[y.columns]).columns
    for g, cluster in enumerate(order):
        members = labels == cluster
        for j in range(x.shape[1]):
            cells = cell[members, j][active[members, j]]
            expected_block = np.sqrt(np.mean(cells ** 2)) if cells.size else 1.0
            assert np.isclose(fcgl.sign_block_weights_[g, j], expected_block)

    summary = pd.DataFrame({"plain": error_summary(plain), "adaptive": error_summary(adaptive)})
    print(summary.round(3))

    # the heavier penalty on weak cells removes loadings on true zeros and shrinks small loadings
    # harder; the large loadings, whose weights are close to one, are shrunk about as much
    assert summary.loc["kept_on_zeros", "adaptive"] < summary.loc["kept_on_zeros", "plain"]
    assert summary.loc["small_shrinkage", "adaptive"] > summary.loc["small_shrinkage", "plain"]
    assert abs(summary.loc["large_shrinkage", "adaptive"]
               - summary.loc["large_shrinkage", "plain"]) < 0.02
    # quoted values
    assert summary.loc["kept_on_zeros"].tolist() == [7, 2]
    assert [round(float(v), 2) for v in summary.loc["small_shrinkage"]] == [0.09, 0.15]
    assert [round(float(v), 2) for v in summary.loc["large_shrinkage"]] == [0.07, 0.07]
    assert weights.to_numpy().min() > 0.9 and weights.to_numpy().max() == 1.0 / FLOOR
    assert round(float(weights.to_numpy().min()), 2) == 0.94


if __name__ == "__main__":
    main()
