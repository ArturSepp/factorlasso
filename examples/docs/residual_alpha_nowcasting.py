"""Canonical example for docs/residual_alpha_nowcasting.md.

Six responses load on three factors over 240 months. Two carry a constant alpha of 0.5% per
month, two an alpha that shifts from zero to 1% per month at month 180, and two none; one
response starts at month 60 and one misses five months. Loadings are fitted with uniform weights
on an expanding window, and each month's nowcast adds the terminal 24-month EWMA of the fit's
residuals to the factor component of the next month. The script checks the nowcast against its
definition, exercises its fail-closed rules, and compares the nowcast alpha with the economic
intercept of the fit as estimates of the next month's true alpha. Synthetic data.
"""

import numpy as np
import pandas as pd

import factorlasso as fl

SEED = 20261009
N_OBS = 240
ALPHA_SPAN = 24
REG_LAMBDA = 1e-5
FIRST_CUTOFF = 119
FACTORS = ["f1", "f2", "f3"]
NAMES = ["const1", "const2", "shift1", "shift2", "zero1", "zero2"]
SHIFT = 180
BETAS = np.array([[1.0, 0.3, 0.0], [0.8, 0.0, 0.4], [0.0, 1.0, 0.3],
                  [0.5, 0.5, 0.0], [0.0, 0.2, 1.0], [0.6, 0.0, 0.6]])


def true_alphas() -> pd.DataFrame:
    """Monthly alpha of each response: constant, shifting from 0 to 1% at SHIFT, or zero."""
    dates = pd.date_range("2006-01-31", periods=N_OBS, freq="ME")
    shift = np.where(np.arange(N_OBS) >= SHIFT, 0.01, 0.0)
    values = np.column_stack([np.full(N_OBS, 0.005), np.full(N_OBS, 0.005), shift, shift,
                              np.zeros(N_OBS), np.zeros(N_OBS)])
    return pd.DataFrame(values, index=dates, columns=NAMES)


def make_panel(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Factor returns and responses; shift2 starts at month 60, zero2 misses five months."""
    rng = np.random.default_rng(seed)
    alphas = true_alphas()
    x = pd.DataFrame(0.04 * rng.standard_normal((N_OBS, 3)), index=alphas.index, columns=FACTORS)
    noise = 0.02 * rng.standard_normal((N_OBS, len(NAMES)))
    y = alphas + x.to_numpy() @ BETAS.T + noise
    y.iloc[:60, NAMES.index("shift2")] = np.nan
    y.iloc[[70, 71, 150, 151, 152], NAMES.index("zero2")] = np.nan
    return x, y


def fit(x: pd.DataFrame, y: pd.DataFrame) -> fl.LassoModel:
    """Loadings with uniform weights; the nowcast keeps them fixed."""
    return fl.LassoModel(reg_lambda=REG_LAMBDA, span=None).fit(x=x, y=y)


def nowcast_next(x: pd.DataFrame, y: pd.DataFrame, cutoff: int) -> fl.LassoNowcastResult:
    """Fit on months up to the cutoff and nowcast the next month from its realised factors."""
    model = fit(x.iloc[: cutoff + 1], y.iloc[: cutoff + 1])
    return model.nowcast(x.iloc[[cutoff + 1]], alpha_span=ALPHA_SPAN)


def alpha_errors(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Nowcast alpha and economic intercept against the next month's true alpha, per cutoff."""
    truth = true_alphas()
    rows = []
    for cutoff in range(FIRST_CUTOFF, N_OBS - 1):
        if y.iloc[cutoff].isna().any():                      # the nowcast refuses these cutoffs
            continue
        result = nowcast_next(x, y, cutoff)
        target = truth.iloc[cutoff + 1]
        economic = result.diagnostics["alpha_const"]
        for name in NAMES:
            rows.append({"date": truth.index[cutoff + 1], "response": name,
                         "true": target[name], "nowcast": result.stat_alpha[name],
                         "economic": economic[name]})
    return pd.DataFrame(rows)


def rmse_by_group(errors: pd.DataFrame) -> pd.DataFrame:
    """Root mean squared error of each alpha estimate by response group, in return units."""
    groups = errors["response"].str.rstrip("12")
    squared = errors[["nowcast", "economic"]].sub(errors["true"], axis=0) ** 2
    squared["no alpha"] = errors["true"] ** 2
    return np.sqrt(squared.groupby(groups, sort=False).mean())


def main() -> None:
    x, y = make_panel()

    # --- the nowcast is the factor component plus the terminal EWMA of the residuals ----------
    cutoff = N_OBS - 2
    model = fit(x.iloc[: cutoff + 1], y.iloc[: cutoff + 1])
    target = x.iloc[[cutoff + 1]]
    result = model.nowcast(target, alpha_span=ALPHA_SPAN)
    residuals = y.iloc[: cutoff + 1] - x.iloc[: cutoff + 1].to_numpy() @ model.coef_.T.to_numpy()
    terminal = residuals.apply(lambda series: series.ewm(span=ALPHA_SPAN, adjust=False,
                                                         ignore_na=True).mean().iloc[-1])
    assert np.allclose(result.stat_alpha, terminal)
    assert np.allclose(result.factor_component, target.to_numpy() @ model.coef_.T.to_numpy())
    assert np.allclose(result.prediction, result.factor_component + result.stat_alpha)
    # with uniform weights and no alpha span, the nowcast alpha is the economic intercept
    uniform = model.nowcast(target)
    assert np.allclose(uniform.stat_alpha, model.alpha_const_)

    # --- fail-closed rules ----------------------------------------------------------------------
    failures = {
        "demean=False": lambda: fl.LassoModel(demean=False).fit(
            x=x.iloc[:cutoff + 1], y=y.iloc[:cutoff + 1]).nowcast(target),
        "target not after cutoff": lambda: model.nowcast(x.iloc[[cutoff]]),
        "factor order": lambda: model.nowcast(target[FACTORS[::-1]]),
        "incomplete final row": lambda: fit(x.iloc[:151], y.iloc[:151]).nowcast(x.iloc[[151]]),
    }
    for name, call in failures.items():
        try:
            call()
        except ValueError:
            continue
        raise AssertionError(f"nowcast accepted {name}")

    # --- estimating the next month's alpha: adaptivity against noise ---------------------------
    errors = alpha_errors(x, y)
    table = rmse_by_group(errors)
    print((10000 * table).round(1))                          # basis points per month
    after = errors[errors["date"] >= true_alphas().index[SHIFT]]
    after_shift = rmse_by_group(after).loc["shift"]

    # quoted values, in basis points per month
    assert (10000 * table).round(0).to_dict("list") == {
        "nowcast": [32.0, 53.0, 50.0], "economic": [12.0, 62.0, 17.0],
        "no alpha": [50.0, 72.0, 0.0]}
    assert (10000 * after_shift[["nowcast", "economic"]]).round(0).tolist() == [67.0, 86.0]
    assert round(10000 * 0.02 / np.sqrt(ALPHA_SPAN)) == 41      # EWMA noise, sigma / sqrt(span)


if __name__ == "__main__":
    main()
