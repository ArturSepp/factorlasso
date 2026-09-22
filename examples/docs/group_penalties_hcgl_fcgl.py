"""
Worked example for ``docs/group_penalties_hcgl_fcgl.md``: what a group penalty selects.
========================================================================================

Twelve synthetic monthly fund returns form three clusters of four. Every fund in a cluster loads
on the same two of six factors, with its own magnitudes. The script fits the same panel under four
penalties - cell-wise LASSO, the row-grouped HCGL penalty, HCGL with an added cell-wise L1 term,
and the cluster-by-factor FCGL penalty - and compares which loadings each one removes.

Every number quoted in the article is asserted here against a reference computed a different way:
the estimator against a direct call of the CVXPY solver on hand-built group loadings, the two
group-weight conventions against their closed-form rescaling, the selection geometry against a
cell count of each row and each cluster-by-factor block, and the discovered partition against the
one that generated the data.

Synthetic data, fixed seed, no network, no files written. Solver: CVXPY with CLARABEL.
"""

import numpy as np
import pandas as pd

import factorlasso as fl

SEED = 20260924
N_OBS = 48                         # monthly observations
FACTOR_NAMES = ["equity", "rates", "credit", "commodity", "value", "momentum"]
FACTOR_VOL = 0.04                  # per-month factor volatility, decimal returns
RESIDUAL_VOL = 0.025               # per-month idiosyncratic volatility, decimal returns
GROUP_NAMES = ["equity_fund", "bond_fund", "real_asset_fund"]
GROUP_SIZE = 4
PENALTY_GRID = np.logspace(-2.0, -5.0, 13)
COMMON_LAMBDA = 1e-3               # one penalty at which the four fits are displayed side by side
L1_WEIGHT = 0.3                    # share of the penalty given to the cell-wise L1 term
ZERO_TOLERANCE = 1e-3              # absolute cut below which a loading counts as zero

# True loadings, responses by factors: each cluster shares a two-factor support (24 of 72 cells).
TRUE_BETA = np.array([
    [1.0, 0.0, 0.0, 0.0, 0.5, 0.0],
    [0.9, 0.0, 0.0, 0.0, 0.3, 0.0],
    [1.1, 0.0, 0.0, 0.0, 0.4, 0.0],
    [0.8, 0.0, 0.0, 0.0, 0.2, 0.0],
    [0.0, 1.2, 0.3, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.7, 0.8, 0.0, 0.0, 0.0],
    [0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
    [0.3, 0.0, 0.0, 0.9, 0.0, 0.0],
    [0.4, 0.0, 0.0, 0.8, 0.0, 0.0],
    [0.5, 0.0, 0.0, 0.6, 0.0, 0.0],
    [0.2, 0.0, 0.0, 0.7, 0.0, 0.0],
])

MODEL_TYPES = {
    "LASSO": (fl.LassoModelType.LASSO, 0.0),
    "HCGL": (fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO, 0.0),
    "sparse HCGL": (fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO, L1_WEIGHT),
    "FCGL": (fl.LassoModelType.FACTOR_CLUSTER_GROUP_LASSO, 0.0),
}


def make_panel(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw the factor panel ``x`` (T x 6) and the response panel ``y`` (T x 12)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-31", periods=N_OBS, freq="ME")
    x = pd.DataFrame(
        FACTOR_VOL * rng.standard_normal((N_OBS, len(FACTOR_NAMES))),
        index=dates,
        columns=FACTOR_NAMES,
    )
    names = [f"{group}_{i + 1}" for group in GROUP_NAMES for i in range(GROUP_SIZE)]
    noise = RESIDUAL_VOL * rng.standard_normal((N_OBS, len(names)))
    y = pd.DataFrame(x.to_numpy() @ TRUE_BETA.T + noise, index=dates, columns=names)
    return x, y


def fit(
    x: pd.DataFrame,
    y: pd.DataFrame,
    name: str,
    reg_lambda: float,
    group_penalty: str = "normalized",
) -> fl.LassoModel:
    """Fit one of the four penalties; the cluster modes discover the partition from ``y``."""
    model_type, l1_weight = MODEL_TYPES[name]
    model = fl.LassoModel(
        model_type=model_type,
        reg_lambda=reg_lambda,
        l1_weight=l1_weight,               # 0 is the pure group penalty, 1 the pure LASSO
        group_penalty=group_penalty,       # w_g = sqrt(|g| / G); "yuan_lin" gives sqrt(|g|)
    )
    return model.fit(x=x, y=y)


def support_counts(beta: np.ndarray) -> dict:
    """Kept, false and missed loadings against the true support, and the loading error."""
    kept = np.abs(beta) > ZERO_TOLERANCE
    truth = TRUE_BETA != 0.0
    return {
        "n_loadings": int(kept.sum()),
        "n_false": int((kept & ~truth).sum()),
        "n_missed": int((~kept & truth).sum()),
        "loading_rmse": float(np.sqrt(np.mean((beta - TRUE_BETA) ** 2))),
    }


def penalty_paths(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Support counts and loading error of the four penalties along the grid."""
    rows = []
    for name in MODEL_TYPES:
        for reg_lambda in PENALTY_GRID:
            beta = fit(x, y, name, float(reg_lambda)).coef_.to_numpy()
            rows.append({"penalty": name, "reg_lambda": float(reg_lambda), **support_counts(beta)})
    return pd.DataFrame(rows).set_index(["penalty", "reg_lambda"])


def best_fits(x: pd.DataFrame, y: pd.DataFrame, paths: pd.DataFrame) -> dict:
    """Refit each penalty where its loading error against the truth is smallest on the grid."""
    fits = {}
    for name in MODEL_TYPES:
        reg_lambda = float(paths.loc[name, "loading_rmse"].idxmin())
        fits[name] = fit(x, y, name, reg_lambda)
    return fits


def refit_on_support(x: pd.DataFrame, y: pd.DataFrame, selected: fl.LassoModel) -> fl.LassoModel:
    """Re-estimate without shrinkage the cells ``selected`` kept; its zeros become constraints."""
    kept = np.abs(selected.coef_.to_numpy()) > ZERO_TOLERANCE
    support = pd.DataFrame(np.where(kept, np.nan, 0.0), index=y.columns, columns=x.columns)
    return fl.LassoModel(reg_lambda=1e-12, factors_beta_loading_signs=support).fit(x=x, y=y)


def block_pattern(beta: np.ndarray, by_row: bool) -> np.ndarray:
    """Number of kept cells in each asset row, or in each cluster-by-factor block."""
    kept = np.abs(beta) > ZERO_TOLERANCE
    if by_row:
        return kept.sum(axis=1)
    return kept.reshape(len(GROUP_NAMES), GROUP_SIZE, -1).sum(axis=1)


def main() -> None:
    """Run the example, verify each quoted number independently, and print the summary."""
    x, y = make_panel()
    paths = penalty_paths(x, y)
    fits = best_fits(x, y, paths)
    hcgl, fcgl = fits["HCGL"], fits["FCGL"]

    # --- 1. The cluster modes recover the partition that generated the data ---
    expected = np.repeat(np.arange(len(GROUP_NAMES)), GROUP_SIZE)
    for model in (hcgl, fcgl):
        labels = model.clusters_.to_numpy()
        together = labels[:, None] == labels[None, :]
        assert np.array_equal(together, expected[:, None] == expected[None, :])

    # --- 2. LassoModel is the documented solver applied to indicator group loadings ---
    x_np, y_np, valid_mask = fl.get_x_y_np(x=x, y=y, span=None)
    group_loadings = fl.set_group_loadings(group_data=hcgl.clusters_)
    assert group_loadings.shape == (len(y.columns), len(GROUP_NAMES))
    assert np.array_equal(group_loadings.sum(axis=1).to_numpy(), np.ones(len(y.columns)))
    direct = fl.solve_group_lasso_cvx_problem(
        x=x_np, y=y_np, group_loadings=group_loadings.to_numpy(), valid_mask=valid_mask,
        reg_lambda=hcgl.reg_lambda, group_penalty="normalized", block_mode="row",
    )
    assert np.allclose(direct.estimated_beta, hcgl.coef_.to_numpy(), atol=1e-7)
    direct_block = fl.solve_group_lasso_cvx_problem(
        x=x_np, y=y_np, group_loadings=group_loadings.to_numpy(), valid_mask=valid_mask,
        reg_lambda=fcgl.reg_lambda, group_penalty="normalized", block_mode="cluster_factor",
    )
    assert np.allclose(direct_block.estimated_beta, fcgl.coef_.to_numpy(), atol=1e-7)

    # --- 3. The two group-weight conventions differ by sqrt(G) in the penalty scale ---
    n_groups = len(GROUP_NAMES)
    yuan_lin = fit(x, y, "FCGL", fcgl.reg_lambda / np.sqrt(n_groups), group_penalty="yuan_lin")
    assert np.allclose(yuan_lin.coef_.to_numpy(), fcgl.coef_.to_numpy(), atol=1e-4)

    # --- 4. Selection geometry: rows for HCGL, cluster-by-factor blocks for FCGL ---
    n_factors = len(FACTOR_NAMES)
    common = {name: fit(x, y, name, COMMON_LAMBDA) for name in MODEL_TYPES}
    blocks = block_pattern(common["FCGL"].coef_.to_numpy(), by_row=False)
    assert set(blocks.ravel()) == {0, GROUP_SIZE}                       # whole blocks in or out
    assert np.array_equal(blocks > 0, TRUE_BETA.reshape(n_groups, GROUP_SIZE, -1).any(axis=1))
    # a row penalty removes whole rows: a kept row is dense, up to one loading under the tolerance
    rows = block_pattern(fit(x, y, "HCGL", 10.0 ** -2.5).coef_.to_numpy(), by_row=True)
    assert np.all((rows == 0) | (rows >= n_factors - 1)) and (rows == 0).sum() == 3
    assert block_pattern(common["HCGL"].coef_.to_numpy(), by_row=True).min() >= n_factors - 1

    # with equal cluster sizes the row penalty does not depend on who shares a cluster ...
    scrambled = pd.Series(np.tile(np.arange(1, n_groups + 1), GROUP_SIZE), index=y.columns)
    row_scrambled = fl.LassoModel(
        model_type=fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO, reg_lambda=hcgl.reg_lambda,
    ).fit(x=x, y=y, external_clusters=scrambled)
    assert np.allclose(row_scrambled.coef_.to_numpy(), hcgl.coef_.to_numpy(), atol=1e-6)
    # ... while the block penalty does
    block_scrambled = fl.LassoModel(
        model_type=fl.LassoModelType.FACTOR_CLUSTER_GROUP_LASSO, reg_lambda=fcgl.reg_lambda,
    ).fit(x=x, y=y, external_clusters=scrambled)
    assert np.max(np.abs(block_scrambled.coef_.to_numpy() - fcgl.coef_.to_numpy())) > 0.05

    # --- 5. l1_weight interpolates: at 1 the group term vanishes and the fit is the LASSO ---
    lasso = fits["LASSO"]
    all_l1 = fl.LassoModel(
        model_type=fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
        reg_lambda=lasso.reg_lambda, l1_weight=1.0,
    ).fit(x=x, y=y)
    assert np.allclose(all_l1.coef_.to_numpy(), lasso.coef_.to_numpy(), atol=1e-3)

    # --- 6. What each penalty keeps: at one common penalty, and at its own smallest error ---
    at_common = pd.DataFrame(
        {name: support_counts(model.coef_.to_numpy()) for name, model in common.items()}).T
    assert at_common.loc["FCGL", ["n_loadings", "n_false", "n_missed"]].tolist() == [24, 0, 0]
    assert at_common.loc["LASSO", ["n_loadings", "n_false", "n_missed"]].tolist() == [19, 0, 5]
    assert at_common.loc["HCGL", "n_false"] == 47
    summary = pd.DataFrame(
        {name: {"reg_lambda": model.reg_lambda, **support_counts(model.coef_.to_numpy())}
         for name, model in fits.items()}
    ).T
    design = np.column_stack([np.ones(N_OBS), x.to_numpy()])
    ols = np.linalg.lstsq(design, y.to_numpy(), rcond=None)[0][1:].T
    rmse_ols = float(np.sqrt(np.mean((ols - TRUE_BETA) ** 2)))
    assert summary.loc["FCGL", "loading_rmse"] < summary.loc["LASSO", "loading_rmse"] < rmse_ols
    assert summary.loc["LASSO", "loading_rmse"] < summary.loc["HCGL", "loading_rmse"] < rmse_ols

    # --- 7. Select with FCGL, then re-estimate the kept cells without shrinkage ---
    refit = refit_on_support(x, y, common["FCGL"])
    refit_counts = support_counts(refit.coef_.to_numpy())
    assert refit_counts["n_false"] == 0 and refit_counts["n_missed"] == 0
    assert refit_counts["loading_rmse"] < summary.loc["FCGL", "loading_rmse"]
    assert 0.058 < refit_counts["loading_rmse"] < 0.063                  # quoted: 0.060

    print(f"Ordinary least squares loading RMSE: {rmse_ols:.3f}")
    counts = {"n_loadings": int, "n_false": int, "n_missed": int}
    print(f"Each penalty at reg_lambda = {COMMON_LAMBDA:.0e}")
    print(at_common.astype(counts).round(3).to_string())
    print("Each penalty at its smallest loading RMSE on the grid")
    print(summary.astype(counts).round(4).to_string())
    print(f"FCGL support refitted without shrinkage: RMSE {refit_counts['loading_rmse']:.3f}")
    print(f"FCGL loadings at reg_lambda = {COMMON_LAMBDA:.0e}")
    print(common["FCGL"].coef_.round(2).to_string())


if __name__ == "__main__":
    main()
