"""Canonical example for docs/unilasso.md.

Twelve responses load on six factors. The first two factors are correlated, and every response
loads positively on the first and negatively on the second, so the marginal (univariate) slope on
the second factor has the wrong sign: the suppressor case. The third factor is independent, with
loadings of both signs; the last three factors carry no signal.

UniLasso (Chatterjee, Hastie and Tibshirani, 2025) fits in two stages per response: univariate
slopes and their leave-one-out fits, then a LASSO of the response on those fits with
non-negative coefficients theta. The final loading is theta times the univariate slope, so it
keeps the univariate sign or is zero. The script checks the solver against an independent
coordinate-descent solution and compares three settings of the two UniLasso options. Synthetic
data.
"""

import numpy as np
import pandas as pd

import factorlasso as fl

SEED = 20261002
N_OBS = 120
N_RESPONSES = 12
REG_LAMBDA = 0.02
FACTOR_CORRELATION = 0.7
NOISE_SD = 0.6
FACTORS = ["f1", "f2", "f3", "f4", "f5", "f6"]


def true_loadings() -> pd.DataFrame:
    """f1 positive, f2 a negative suppressor, f3 of both signs, f4 to f6 zero."""
    beta = np.zeros((N_RESPONSES, len(FACTORS)))
    beta[:, 0] = np.linspace(0.6, 1.2, N_RESPONSES)
    beta[:, 1] = -0.4
    beta[:, 2] = np.linspace(-0.6, 0.6, N_RESPONSES)
    return pd.DataFrame(beta, index=[f"y{k + 1}" for k in range(N_RESPONSES)], columns=FACTORS)


def make_panel(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unit-variance factors, f1 and f2 correlated; de-meaned responses."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((N_OBS, len(FACTORS)))
    z[:, 1] = FACTOR_CORRELATION * z[:, 0] + np.sqrt(1.0 - FACTOR_CORRELATION**2) * z[:, 1]
    x = pd.DataFrame(z - z.mean(axis=0), columns=FACTORS)
    beta = true_loadings()
    noise = NOISE_SD * rng.standard_normal((N_OBS, N_RESPONSES))
    y = pd.DataFrame(x.to_numpy() @ beta.T.to_numpy() + noise, columns=beta.index)
    return x, y - y.mean()


def univariate_slopes(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Stage one: the slope of each response on each factor alone, through the origin."""
    x_np, y_np = x.to_numpy(), y.to_numpy()
    slopes = (y_np.T @ x_np) / np.sum(x_np**2, axis=0)
    return pd.DataFrame(slopes, index=y.columns, columns=x.columns)


def fit_unilasso(x: pd.DataFrame, y: pd.DataFrame, reg_lambda: float = REG_LAMBDA,
                 loo: bool = True, non_negative: bool = True) -> pd.DataFrame:
    """UniLasso through LassoModel; loo and non_negative are the two stage-two options."""
    model = fl.LassoModel(
        model_type=fl.LassoModelType.UNILASSO,
        reg_lambda=reg_lambda,
        unilasso_loo=loo,
        unilasso_non_negative=non_negative,
    ).fit(x=x, y=y)
    return model.coef_


def stage_two_by_coordinate_descent(x: np.ndarray, y: np.ndarray, reg_lambda: float,
                                    n_sweeps: int = 2000) -> np.ndarray:
    """An independent UniLasso for one response: stage two by non-negative coordinate descent."""
    t = x.shape[0]
    s_xx, s_xy = np.sum(x**2, axis=0), x.T @ y
    slope = s_xy / s_xx
    eta = x * (s_xy[None, :] - x * y[:, None]) / (s_xx[None, :] - x**2)   # leave-one-out fits
    theta = np.zeros(x.shape[1])
    for _ in range(n_sweeps):
        for j in range(x.shape[1]):
            partial = y - eta @ theta + eta[:, j] * theta[j]
            theta[j] = max(0.0, (2.0 / t * eta[:, j] @ partial - reg_lambda)
                           / (2.0 / t * eta[:, j] @ eta[:, j]))
    return theta * slope


REG_LAMBDAS = np.geomspace(1e-3, 0.3, 16)


def settings_path(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Mean suppressor loading and noise loadings kept along the penalty grid, per setting."""
    settings = {"UniLasso": (True, True), "in-sample fits": (False, True),
                "signs free": (True, False)}
    rows = []
    for reg_lambda in REG_LAMBDAS:
        for name, (loo, non_negative) in settings.items():
            coef = fit_unilasso(x, y, float(reg_lambda), loo=loo, non_negative=non_negative)
            rows.append({"reg_lambda": reg_lambda, "setting": name,
                         "f2": coef["f2"].mean(),
                         "noise_kept": int((coef[["f4", "f5", "f6"]].abs() > 1e-6).sum().sum())})
    return pd.DataFrame(rows)


def main() -> None:
    x, y = make_panel()
    truth = true_loadings()
    slopes = univariate_slopes(x, y)
    ols = pd.DataFrame(np.linalg.lstsq(x, y, rcond=None)[0].T, index=y.columns, columns=FACTORS)

    # --- stage one: the suppressor's univariate slope has the wrong sign ----------------------
    assert (truth["f2"] < 0).all() and (slopes["f2"] > 0).all()
    assert (ols["f2"] < 0).all()                                   # the joint fit recovers it
    loo_row = 7
    x_np, y_np = x.to_numpy(), y.to_numpy()
    keep = np.arange(N_OBS) != loo_row
    brute = (x_np[keep].T @ y_np[keep, 0]) / np.sum(x_np[keep]**2, axis=0)
    s_xx, s_xy = np.sum(x_np**2, axis=0), x_np.T @ y_np[:, 0]
    assert np.allclose(brute, (s_xy - x_np[loo_row] * y_np[loo_row, 0])
                       / (s_xx - x_np[loo_row]**2))                # leave-one-out slope identity

    # --- stage two: loadings keep the univariate sign or are zero ----------------------------
    fits = {"UniLasso": fit_unilasso(x, y),
            "in-sample fits": fit_unilasso(x, y, loo=False),
            "signs free": fit_unilasso(x, y, non_negative=False)}
    unilasso = fits["UniLasso"]
    assert (unilasso.to_numpy() * slopes.to_numpy() >= -1e-9).all()
    for k in range(N_RESPONSES):
        independent = stage_two_by_coordinate_descent(x_np, y_np[:, k], REG_LAMBDA)
        assert np.allclose(unilasso.to_numpy()[k], independent, atol=1e-5)
    direct = fl.solve_unilasso_cvx_problem(x=x_np, y=y_np, reg_lambda=REG_LAMBDA)
    assert np.allclose(direct.estimated_beta, unilasso.to_numpy(), atol=1e-6)

    table = pd.DataFrame({
        "true": truth.mean(), "OLS": ols.mean(), "univariate": slopes.mean(),
        **{name: fit.mean() for name, fit in fits.items()},
    }).loc[["f1", "f2"]]
    print(table.round(2))                                          # mean loading over responses
    kept = {name: int((fit[["f4", "f5", "f6"]].abs() > 1e-6).sum().sum())
            for name, fit in fits.items()}
    print("loadings kept on the 36 noise cells:", kept)

    # the suppressor: UniLasso cannot give it its negative sign, so it drops it and the f1
    # loading absorbs the omission; with the sign free, stage two reverses the univariate slope
    assert (unilasso["f2"].abs() < 1e-6).all()
    assert (fits["signs free"]["f2"] < 0).all()
    # f3 has loadings of both signs: each kept loading has the sign of its own univariate slope
    f3_kept = unilasso["f3"].abs() > 1e-6
    assert (np.sign(unilasso["f3"][f3_kept]) == np.sign(slopes["f3"][f3_kept])).all()
    assert (np.sign(unilasso["f3"][f3_kept]) == np.sign(truth["f3"][f3_kept])).all()

    # quoted values
    assert table.round(2).loc["f2"].tolist() == [-0.4, -0.4, 0.23, 0.0, 0.0, -0.29]
    assert table.round(2).loc["f1"].tolist() == [0.9, 0.91, 0.61, 0.59, 0.6, 0.8]
    assert kept == {"UniLasso": 1, "in-sample fits": 6, "signs free": 1}
    assert int(f3_kept.sum()) == 10

    # along the penalty grid: the leave-one-out fits screen the noise factors
    path = settings_path(x, y)
    smallest = path[path["reg_lambda"] == REG_LAMBDAS[0]].set_index("setting")["noise_kept"]
    assert smallest.to_dict() == {"UniLasso": 11, "in-sample fits": 25, "signs free": 32}
    assert (path.loc[path["setting"] != "signs free", "f2"].abs() < 1e-6).all()


if __name__ == "__main__":
    main()
