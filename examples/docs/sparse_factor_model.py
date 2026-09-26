"""
Worked example for ``docs/sparse_factor_model.md``: the multi-output LASSO factor model.
=========================================================================================

Six synthetic monthly return series load on three of eight candidate factors. The script fits
:class:`factorlasso.LassoModel` along a penalty grid and shows what the penalty buys: loadings on
the five irrelevant factors are removed, and the estimation error against the known loadings falls
below the error of ordinary least squares before it rises again.

Every number quoted in the article is asserted here against a reference computed a different way:
the solution against a NumPy coordinate-descent solver and the Karush-Kuhn-Tucker conditions of the
objective, the zero-penalty limit against ``numpy.linalg.lstsq``, the intercept against sample
means, the multi-output fit against one fit per response, and the scaling rule of the penalty
against a refit in percent units.

Synthetic data, fixed seed, no network, no files written. Solver: CVXPY with CLARABEL.
"""

import numpy as np
import pandas as pd

import factorlasso as fl

SEED = 20260922
N_TRAIN = 60                       # monthly observations used for estimation
N_TEST = 60                        # later observations used only for held-out R-squared
FACTOR_NAMES = [
    "equity", "rates", "credit", "commodity", "value", "momentum", "carry", "volatility",
]
FACTOR_VOL = 0.04                  # per-month factor volatility, decimal returns
RESIDUAL_VOL = 0.02                # per-month idiosyncratic volatility, decimal returns
REG_LAMBDA = 10.0 ** -3.5          # penalty of the point fit examined in detail
PENALTY_GRID = np.logspace(-2.0, -6.0, 17)
ZERO_TOLERANCE = 1e-3              # absolute cut below which a loading counts as zero

# True loadings, responses by factors. Only the first three factors carry any response.
TRUE_BETA = np.array([
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.8, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.6, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.3, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
])
TRUE_ALPHA = np.array([0.002, 0.0, 0.001, 0.0, -0.001, 0.003])    # per-month intercepts


def make_panel(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw ``N_TRAIN + N_TEST`` months of the factor panel ``x`` and the response panel ``y``."""
    rng = np.random.default_rng(seed)
    n_obs = N_TRAIN + N_TEST
    dates = pd.date_range("2015-01-31", periods=n_obs, freq="ME")
    x = pd.DataFrame(
        FACTOR_VOL * rng.standard_normal((n_obs, len(FACTOR_NAMES))),
        index=dates,
        columns=FACTOR_NAMES,
    )
    noise = RESIDUAL_VOL * rng.standard_normal((n_obs, TRUE_BETA.shape[0]))
    y = pd.DataFrame(
        TRUE_ALPHA + x.to_numpy() @ TRUE_BETA.T + noise,
        index=dates,
        columns=[f"asset_{i + 1}" for i in range(TRUE_BETA.shape[0])],
    )
    return x, y


def fit_lasso(x: pd.DataFrame, y: pd.DataFrame, reg_lambda: float = REG_LAMBDA) -> fl.LassoModel:
    """Fit the cell-wise L1 model with uniform observation weights and sample-mean centring."""
    model = fl.LassoModel(
        model_type=fl.LassoModelType.LASSO,
        reg_lambda=reg_lambda,
        span=None,                         # uniform weights; an EWMA span would discount old rows
        demean=True,                       # centre x and y, so no intercept enters the programme
    )
    return model.fit(x=x, y=y)


def lasso_coordinate_descent(
    x: np.ndarray,
    y: np.ndarray,
    reg_lambda: float,
    n_sweeps: int = 2000,
) -> np.ndarray:
    """Minimise ``(1/T) ||x b - y||^2 + reg_lambda ||b||_1`` for one centred response."""
    n_obs, n_factors = x.shape
    b = np.zeros(n_factors)
    scale = np.sum(x ** 2, axis=0) / n_obs
    for _ in range(n_sweeps):
        previous = b.copy()
        for j in range(n_factors):
            partial_residual = y - x @ b + x[:, j] * b[j]
            rho = x[:, j] @ partial_residual / n_obs
            b[j] = np.sign(rho) * max(abs(rho) - 0.5 * reg_lambda, 0.0) / scale[j]
        if np.max(np.abs(b - previous)) < 1e-14:
            break
    return b


def objective(
    beta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    reg_lambda: float = REG_LAMBDA,
) -> float:
    """Value of ``(1/T) ||x beta' - y||_F^2 + reg_lambda ||beta||_1`` on centred arrays."""
    return float(np.sum((x @ beta.T - y) ** 2) / x.shape[0] + reg_lambda * np.abs(beta).sum())


def penalty_path(x: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the first ``N_TRAIN`` rows along the grid; return the summary and asset_2's loadings."""
    x_train, y_train = x.iloc[:N_TRAIN], y.iloc[:N_TRAIN]
    x_test, y_test = x.iloc[N_TRAIN:], y.iloc[N_TRAIN:]
    rows, loadings = [], {}
    for reg_lambda in PENALTY_GRID:
        model = fit_lasso(x_train, y_train, reg_lambda=float(reg_lambda))
        beta = model.coef_.to_numpy()
        kept = np.abs(beta) > ZERO_TOLERANCE
        rows.append({
            "reg_lambda": float(reg_lambda),
            "n_loadings": int(kept.sum()),
            "n_false_loadings": int((kept & (TRUE_BETA == 0.0)).sum()),
            "loading_rmse": float(np.sqrt(np.mean((beta - TRUE_BETA) ** 2))),
            "r2_in_sample": model.score(x_train, y_train),
            "r2_held_out": model.score(x_test, y_test),
        })
        loadings[float(reg_lambda)] = model.coef_.loc["asset_2"]
    summary = pd.DataFrame(rows).set_index("reg_lambda")
    return summary, pd.DataFrame(loadings).T.rename_axis("reg_lambda")


def main() -> None:
    """Run the example, verify each quoted number independently, and print the summary."""
    x, y = make_panel()
    x_train, y_train = x.iloc[:N_TRAIN], y.iloc[:N_TRAIN]
    model = fit_lasso(x_train, y_train)
    beta = model.coef_.to_numpy()

    # --- 1. The solver output is the minimiser of the documented objective ---
    x_centred = x_train.to_numpy() - x_train.to_numpy().mean(axis=0)
    y_centred = y_train.to_numpy() - y_train.to_numpy().mean(axis=0)
    reference = np.vstack([
        lasso_coordinate_descent(x_centred, y_centred[:, i], REG_LAMBDA)
        for i in range(y_centred.shape[1])
    ])
    assert np.max(np.abs(beta - reference)) < 1e-4                      # observed: 2e-5
    gap = objective(beta, x_centred, y_centred) - objective(reference, x_centred, y_centred)
    assert abs(gap) < 1e-9

    # Karush-Kuhn-Tucker: the loss gradient equals -lambda*sign(beta) on kept cells, is within
    # [-lambda, lambda] on zeroed cells
    gradient = 2.0 / N_TRAIN * (x_centred @ beta.T - y_centred).T @ x_centred      # (N, M)
    kept = np.abs(beta) > ZERO_TOLERANCE
    assert np.allclose(gradient[kept], -REG_LAMBDA * np.sign(beta[kept]), atol=1e-6)
    assert np.all(np.abs(gradient[~kept]) <= REG_LAMBDA * (1.0 + 1e-6))
    assert np.array_equal(kept, reference != 0.0)                        # the reference's support
    assert int(kept.sum()) == 13 and int((kept & (TRUE_BETA == 0.0)).sum()) == 3

    # the input arrays the solver receives are the centred panels
    x_solver, y_solver, valid_mask = fl.get_x_y_np(x=x_train, y=y_train, span=None)
    assert np.allclose(x_solver, x_centred) and np.allclose(y_solver, y_centred)
    assert valid_mask.all()
    direct = fl.solve_lasso_cvx_problem(x=x_solver, y=y_solver, reg_lambda=REG_LAMBDA)
    assert isinstance(direct, fl.LassoEstimationResult)
    assert np.allclose(direct.estimated_beta, beta, atol=1e-8)

    # --- 2. Intercepts: alpha_const_ is the regression intercept, intercept_ a solver residual ---
    alpha = y_train.mean().to_numpy() - x_train.mean().to_numpy() @ beta.T
    assert np.allclose(model.alpha_const_.to_numpy(), alpha)
    assert np.allclose(model.intercept_.to_numpy(), 0.0, atol=1e-12)
    assert np.allclose(model.predict(x_train).to_numpy(), alpha + x_train.to_numpy() @ beta.T)

    # --- 3. The programme separates by response: one joint fit equals six single fits ---
    single = pd.concat(
        [fit_lasso(x_train, y_train[[name]]).coef_ for name in y_train.columns]
    )
    assert np.allclose(single.to_numpy(), beta, atol=1e-3)               # observed: 3e-4

    # --- 4. Limits: no penalty gives least squares; a large penalty removes every loading ---
    design = np.column_stack([np.ones(N_TRAIN), x_train.to_numpy()])
    ols = np.linalg.lstsq(design, y_train.to_numpy(), rcond=None)[0][1:].T
    unpenalised = fit_lasso(x_train, y_train, reg_lambda=1e-12).coef_.to_numpy()
    assert np.allclose(unpenalised, ols, atol=1e-6)
    lambda_max = float(np.max(np.abs(2.0 / N_TRAIN * y_centred.T @ x_centred)))
    collapsed = fit_lasso(x_train, y_train, reg_lambda=1.01 * lambda_max).coef_.to_numpy()
    assert np.max(np.abs(collapsed)) < ZERO_TOLERANCE
    assert 3.8e-3 < lambda_max < 4.0e-3                                 # quoted: 3.9e-3

    # --- 5. reg_lambda carries squared return units: percent returns need 1e4 times the penalty ---
    percent = fl.LassoModel(reg_lambda=1e4 * REG_LAMBDA).fit(x=100.0 * x_train, y=100.0 * y_train)
    assert np.allclose(percent.coef_.to_numpy(), beta, atol=1e-4)

    # --- 6. The penalty path: fewer false loadings and a smaller error than least squares ---
    summary, _ = penalty_path(x, y)
    rmse_ols = float(np.sqrt(np.mean((ols - TRUE_BETA) ** 2)))
    best = summary["loading_rmse"].idxmin()
    assert np.isclose(best, 10.0 ** -3.75) and np.isclose(summary["r2_held_out"].idxmax(), best)
    assert 0.043 < summary.loc[best, "loading_rmse"] < 0.047 < 0.072 < rmse_ols < 0.076
    assert 0.050 < summary.loc[REG_LAMBDA, "loading_rmse"] < 0.055     # the sparser point fit
    assert summary["n_loadings"].iloc[0] == 0 and summary["n_loadings"].iloc[-1] == TRUE_BETA.size
    assert summary["n_loadings"].is_monotonic_increasing
    assert summary["r2_in_sample"].is_monotonic_increasing

    print(f"Point fit at reg_lambda = {REG_LAMBDA:.2e}")
    print(f"  kept loadings        : {int(kept.sum())} of {beta.size}"
          f" (true support {int(np.count_nonzero(TRUE_BETA))},"
          f" false {int((kept & (TRUE_BETA == 0)).sum())})")
    print(f"  max |beta - CD ref|  : {np.max(np.abs(beta - reference)):.1e}")
    print(f"  lambda_max           : {lambda_max:.2e}")
    print(f"  loading RMSE         : {summary.loc[REG_LAMBDA, 'loading_rmse']:.3f}"
          f" (ordinary least squares {rmse_ols:.3f}; smallest on the grid"
          f" {summary.loc[best, 'loading_rmse']:.3f} at reg_lambda = {best:.2e})")
    print(model.coef_.round(2).to_string())
    print("Penalty path, estimation on the first 60 months")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
