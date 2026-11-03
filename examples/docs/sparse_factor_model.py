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
The dense ridge baseline is independently checked against augmented least squares on the same
centred training panel, including its zero- and infinite-penalty endpoints.
The final recipe selects pooled shrinkage using blocked reconstruction loss,
refits all training-derived inputs at a decision date, and combines raw-unit
loadings with supplied one-month excess premiums and explicitly admitted alpha.
Conditional HCGL is illustrated at a fixed penalty, separately from selection.

Synthetic data, fixed seed, no network, no files written. Solver: CVXPY with CLARABEL.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.cluster.hierarchy import fcluster

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
FORECAST_PENALTIES = (0.0, 0.1, 1.0, np.inf)   # illustrative standardized-coordinate grid
FORECAST_SPAN = 24.0                          # EW span in calendar months
HCGL_ILLUSTRATIVE_PENALTY = 0.1                # fixed, not selected by the pooled-fit grid

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


def fit_ridge(x: pd.DataFrame, y: pd.DataFrame, reg_lambda: float = REG_LAMBDA) -> fl.LassoModel:
    """Fit the zero-centred ridge baseline on the same uniformly weighted training panel."""
    return fl.LassoModel(
        model_type=fl.LassoModelType.RIDGE,
        reg_lambda=reg_lambda,
        span=None,
        demean=True,
        loss_normalization="sample",
    ).fit(x=x, y=y)


def check_ridge_reference(x: pd.DataFrame, y: pd.DataFrame) -> None:
    """Compare ridge with augmented least squares without using package solver internals."""
    centred_x = x.to_numpy() - x.to_numpy().mean(axis=0)
    centred_y = y.to_numpy() - y.to_numpy().mean(axis=0)
    n_rows, n_factors = centred_x.shape
    augmented_x = np.vstack([
        centred_x, np.sqrt(n_rows * REG_LAMBDA) * np.eye(n_factors),
    ])
    augmented_y = np.vstack([centred_y, np.zeros((n_factors, y.shape[1]))])
    reference = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0].T
    ridge = fit_ridge(x, y)
    assert np.allclose(ridge.coef_.to_numpy(), reference, rtol=1e-6, atol=1e-7)
    expected_intercept = y.mean().to_numpy() - x.mean().to_numpy() @ reference.T
    assert np.allclose(ridge.alpha_const_.to_numpy(), expected_intercept, atol=1e-7)
    expected_prediction = expected_intercept + x.to_numpy() @ reference.T
    assert np.allclose(ridge.predict(x).to_numpy(), expected_prediction, atol=1e-7)
    ols = np.linalg.lstsq(centred_x, centred_y, rcond=None)[0].T
    assert np.allclose(fit_ridge(x, y, 0.0).coef_.to_numpy(), ols, atol=1e-7)
    assert np.all(fit_ridge(x, y, np.inf).coef_.to_numpy() == 0.0)


def check_pooled_shrinkage_reference(x: pd.DataFrame, y: pd.DataFrame) -> None:
    """Check pooled-target shrinkage by augmented least squares and assemble a forecast."""
    groups = pd.Series([0, 0, 1, 1, 2, 2], index=y.columns)
    weights = pd.Series(np.linspace(0.5, 1.5, len(x)), index=x.index)
    fitted = fl.fit_pooled_loading_shrinkage(
        x, y, groups, REG_LAMBDA, weights=weights,
    )
    assert isinstance(fitted, fl.PooledLoadingShrinkageResult)
    w = weights.to_numpy() / weights.mean()
    x_mean = np.average(x.to_numpy(), axis=0, weights=w)
    y_mean = np.average(y.to_numpy(), axis=0, weights=w)
    xw = np.sqrt(w[:, None]) * (x.to_numpy() - x_mean)
    yw = np.sqrt(w[:, None]) * (y.to_numpy() - y_mean)
    individual = np.linalg.lstsq(xw, yw, rcond=None)[0].T
    target = np.empty_like(individual)
    for group in groups.unique():
        members = groups.to_numpy() == group
        target[members] = individual[members].mean(axis=0)
    penalty_root = np.sqrt(len(x) * REG_LAMBDA)
    augmented_x = np.vstack([xw, penalty_root * np.eye(x.shape[1])])
    augmented_y = np.vstack([yw, penalty_root * target.T])
    reference = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0].T
    assert np.allclose(fitted.pooled_target, target, atol=1e-11)
    assert np.allclose(fitted.coefficients, reference, rtol=1e-6, atol=1e-7)
    assert np.allclose(fitted.intercept, y_mean - reference @ x_mean, atol=1e-8)
    assert fitted.stationarity_residual <= 1e-7
    endpoint = fl.fit_pooled_loading_shrinkage(x, y, groups, np.inf, weights=weights)
    assert np.array_equal(endpoint.coefficients, endpoint.pooled_target)
    assert endpoint.stationarity_residual is None and endpoint.objective is None

    # Supplied one-month excess-return premiums; admit no residual alpha to the forecast.
    premium = pd.Series([0.003, 0.001, 0.002, 0.0, 0.0, 0.0, 0.0, 0.0], index=x.columns)
    admitted_alpha = pd.Series(0.0, index=y.columns)
    assert premium.index.equals(fitted.coefficients.columns)
    assert admitted_alpha.index.equals(fitted.coefficients.index)
    forecast = fitted.coefficients.dot(premium) + admitted_alpha
    reference_forecast = reference @ premium.to_numpy() + admitted_alpha.to_numpy()
    assert np.allclose(forecast, reference_forecast, atol=1e-9)


@dataclass(frozen=True)
class ForecastWindow:
    """Example-local training transforms and evidence; not a package API."""

    x: pd.DataFrame
    y: pd.DataFrame
    weights: pd.Series
    x_mean: pd.Series
    y_mean: pd.Series
    x_scale: pd.Series
    y_scale: pd.Series
    groups: pd.Series
    gate: fl.PooledSignGateResult


@dataclass(frozen=True)
class ForecastRecipeResult:
    """Example-local outputs in monthly decimal excess-return units."""

    penalty: float
    validation_losses: pd.DataFrame
    coefficients: pd.DataFrame
    regression_intercept: pd.Series
    forecast: pd.Series
    hcgl_forecast: pd.Series
    window: ForecastWindow
    pooled_fit: fl.PooledLoadingShrinkageResult


def prepare_forecast_window(x: pd.DataFrame, y: pd.DataFrame) -> ForecastWindow:
    """Prepare only the supplied prefix, preserving calendar-month ages in weights."""
    if (not isinstance(x.index, pd.DatetimeIndex) or x.index.hasnans
            or not x.index.is_unique or not x.index.is_monotonic_increasing
            or not x.index.equals(y.index) or len(x) <= x.shape[1] + 1):
        raise ValueError(
            "use aligned ordered unique dates and positive residual degrees of freedom")
    if (not np.isfinite(x.to_numpy()).all() or not np.isfinite(y.to_numpy()).all()):
        raise ValueError("the training prefix must be complete and finite")
    months = x.index.to_period("M").asi8
    if np.any(np.diff(months) <= 0):
        raise ValueError("the example accepts at most one observation per calendar month")
    weights = pd.Series((1 - 2 / (FORECAST_SPAN + 1)) ** (months[-1] - months), index=x.index)
    weights /= weights.mean()
    xm = pd.Series(np.average(x, axis=0, weights=weights), index=x.columns)
    ym = pd.Series(np.average(y, axis=0, weights=weights), index=y.columns)
    xc, yc = x - xm, y - ym
    xs = pd.Series(np.sqrt(np.average(xc**2, axis=0, weights=weights)), index=x.columns)
    ys = pd.Series(np.sqrt(np.average(yc**2, axis=0, weights=weights)), index=y.columns)
    if not np.all(xs > 0) or not np.all(ys > 0):
        raise ValueError("training factors and responses must have positive weighted scale")
    xa, ya = xc / xs, yc / ys
    corr = (ya.mul(weights, axis=0).T @ ya) / len(ya)
    _, linkage, _ = fl.compute_clusters_from_corr_matrix(
        corr, linkage_method="ward", distance_transform=fl.DistanceTransform.CHORD,
    )
    # The paper cut scales the final merge height; the core default scales pairwise distance.
    labels = (fcluster(linkage, 0.6 * linkage[-1, 2], criterion="distance")
              if len(linkage) else np.ones(y.shape[1], dtype=int))
    groups = pd.Series(labels, index=y.columns)
    gate = fl.compute_pooled_sign_gate(xa, ya, groups, weights=weights, action="exclude")
    return ForecastWindow(xa, ya, weights, xm, ym, xs, ys, groups, gate)


def fit_forecast_window(
    window: ForecastWindow, penalty: float,
) -> tuple[pd.DataFrame, pd.Series, fl.PooledLoadingShrinkageResult]:
    """Fit pooled shrinkage in analysis units and recover slopes/intercept in raw units."""
    fitted = fl.fit_pooled_loading_shrinkage(
        window.x, window.y, window.groups, penalty, weights=window.weights,
    )
    beta = fitted.coefficients.mul(window.y_scale, axis=0).div(window.x_scale, axis=1)
    alpha = window.y_mean - beta.dot(window.x_mean)
    return beta, alpha, fitted


def select_forecast_penalty(losses: pd.DataFrame) -> float:
    """Require every blocked score; choose the largest penalty within 1e-12 of the minimum."""
    complete = np.isfinite(losses.to_numpy()).all(axis=1)
    if not np.any(complete):
        raise RuntimeError("no candidate has a finite score in every validation block")
    average = losses.loc[complete].mean(axis=1)
    return float(max(average.index[average <= average.min() + 1e-12]))


def assemble_expected_return(
    coefficients: pd.DataFrame, premium: pd.Series, admitted_alpha: pd.Series,
) -> pd.Series:
    """Combine supplied same-horizon inputs, with strict factor and response labels."""
    if (not isinstance(premium, pd.Series) or not isinstance(admitted_alpha, pd.Series)
            or not premium.index.is_unique or not admitted_alpha.index.is_unique
            or not premium.index.equals(coefficients.columns)
            or not admitted_alpha.index.equals(coefficients.index)
            or not np.isfinite(premium.to_numpy()).all()
            or not np.isfinite(admitted_alpha.to_numpy()).all()):
        raise ValueError("premiums and admitted alpha must be finite and match labels in order")
    return (coefficients.dot(premium) + admitted_alpha).rename("monthly_excess_forecast")


def forecast_recipe(
    x: pd.DataFrame,
    y: pd.DataFrame,
    decision_date: pd.Timestamp,
    premium: pd.Series,
    admitted_alpha: pd.Series,
    premium_asof: pd.Timestamp,
    alpha_asof: pd.Timestamp,
) -> ForecastRecipeResult:
    """Select/refit on 60 eligible months; later rows do not enter any calculation.

    Training prefixes 30/40/50 each precede ten validation observations. The
    score is raw reconstruction MSE divided by that prefix's response variance,
    not expected-return loss. The caller declares premiums and alpha available
    no later than decision_date, in one-month decimal excess-return units.
    """
    if (not isinstance(x.index, pd.DatetimeIndex) or x.index.hasnans
            or not x.index.is_unique or not x.index.is_monotonic_increasing
            or not x.index.equals(y.index)):
        raise ValueError("panels must have aligned, ordered, unique, nonmissing dates")
    dates = [pd.Timestamp(value) for value in (decision_date, premium_asof, alpha_asof)]
    if any(pd.isna(value) for value in dates) or any(value > dates[0] for value in dates[1:]):
        raise ValueError("premium/alpha dates must be known and no later than the decision")
    eligible = x.index <= dates[0]
    x_train, y_train = x.loc[eligible], y.loc[eligible]
    if len(x_train) != N_TRAIN:
        raise ValueError("this bounded teaching recipe requires 60 eligible training months")
    losses = pd.DataFrame(np.nan, index=FORECAST_PENALTIES, columns=[30, 40, 50])
    for stop in losses.columns:
        window = prepare_forecast_window(x_train.iloc[:stop], y_train.iloc[:stop])
        for penalty in losses.index:
            try:
                beta, alpha, _ = fit_forecast_window(window, float(penalty))
                predicted = x_train.iloc[stop:stop+10] @ beta.T + alpha
                errors = (y_train.iloc[stop:stop+10] - predicted) / window.y_scale
                losses.loc[penalty, stop] = float(np.mean(errors.to_numpy()**2))
            except RuntimeError:
                # A failed numerical fit invalidates this candidate, not just its bad block.
                continue
    selected = select_forecast_penalty(losses)
    window = prepare_forecast_window(x_train, y_train)
    beta, alpha, fitted = fit_forecast_window(window, selected)
    forecast = assemble_expected_return(beta, premium, admitted_alpha)
    root_weights = np.sqrt(window.weights)
    hcgl = fl.LassoModel(
        model_type=fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
        reg_lambda=HCGL_ILLUSTRATIVE_PENALTY,
        demean=False, span=None, warmup_period=None,
        loss_normalization="sample", group_penalty="normalized", l1_weight=0.5,
        auto_sign_constraints=False, apply_ols_prior=False, auto_sign_adaptive_weights=False,
        factors_beta_loading_signs=window.gate.restrictions,
    ).fit(window.x.mul(root_weights, axis=0), window.y.mul(root_weights, axis=0),
          external_clusters=window.groups)
    hcgl_beta = hcgl.coef_.mul(window.y_scale, axis=0).div(window.x_scale, axis=1)
    hcgl_forecast = assemble_expected_return(hcgl_beta, premium, admitted_alpha)
    return ForecastRecipeResult(selected, losses, beta, alpha, forecast, hcgl_forecast, window,
                                fitted)


def check_forecast_recipe(x: pd.DataFrame, y: pd.DataFrame) -> None:
    """Run the bounded forecast recipe and independently check its selected raw-unit fit."""
    decision_date = x.index[N_TRAIN - 1]
    premium = pd.Series([0.003, 0.001, 0.002, 0.0, 0.0, 0.0, 0.0, 0.0], index=x.columns)
    admitted_alpha = pd.Series(0.0, index=y.columns)  # distinct from the regression intercept
    result = forecast_recipe(x, y, decision_date, premium, admitted_alpha,
                             premium_asof=decision_date, alpha_asof=decision_date)
    assert np.isfinite(result.validation_losses.to_numpy()).all()
    window = result.window
    t, m = window.x.shape
    xw = window.x.to_numpy() * np.sqrt(window.weights.to_numpy()[:, None])
    yw = window.y.to_numpy() * np.sqrt(window.weights.to_numpy()[:, None])
    individual = np.linalg.lstsq(xw, yw, rcond=None)[0].T
    target = np.empty_like(individual)
    for label in window.groups.unique():
        members = window.groups.to_numpy() == label
        target[members] = individual[members].mean(axis=0)
    if np.isposinf(result.penalty):
        reference = target
    else:
        penalty_root = np.sqrt(t * result.penalty)
        reference = np.linalg.lstsq(np.vstack([xw, penalty_root * np.eye(m)]),
                                   np.vstack([yw, penalty_root * target.T]), rcond=None)[0].T
    raw = reference * window.y_scale.to_numpy()[:, None] / window.x_scale.to_numpy()[None, :]
    assert np.allclose(result.coefficients, raw, rtol=1e-7, atol=1e-8)
    assert np.allclose(result.regression_intercept,
                       window.y_mean.to_numpy() - raw @ window.x_mean.to_numpy(), atol=1e-10)
    assert np.allclose(result.forecast, raw @ premium.to_numpy(), atol=1e-10)
    print(f"Pooled forecast recipe: selected standardized penalty {result.penalty:g}")
    print(pd.concat([result.forecast, result.hcgl_forecast.rename("fixed_penalty_hcgl")],
                    axis=1).round(6).to_string())


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
    check_ridge_reference(x_train, y_train)
    check_pooled_shrinkage_reference(x_train, y_train)
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
    check_forecast_recipe(x, y)


if __name__ == "__main__":
    main()
