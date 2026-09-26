"""Canonical example for docs/ewma_weighting_and_ragged_histories.md.

Part 1 checks the EWMA conventions: the decay of a span, the half-life, the effective sample
size, the running mean against pandas, and the square-root row weights of the solver.

Part 2 fits four responses with the same true loading but histories of 240, 120, 60 and 36
months in one panel, on one factor. With one factor and no intercept the LASSO is a soft
threshold, so the shrinkage of each loading has a closed form:

* ``loss_normalization="sample"``: ``lambda * T / (2 * sum_valid w x^2)``, where ``T`` is the
  panel's row count, so a response with a short history is shrunk harder;
* ``loss_normalization="weight_sum"``: ``lambda * m_i / (2 * sum_valid w x^2)``, where ``m_i`` is
  the response's own weight mass, so missing pre-history adds no shrinkage.

Independent references: ``pandas.DataFrame.ewm(adjust=False)``, direct sums of the weights,
weighted least squares by hand, the closed-form shrinkage, and single-response fits on each
response's own window. The data are synthetic decimal monthly returns with zero mean, fitted
with ``demean=False``.
"""

import numpy as np
import pandas as pd

import factorlasso as fl

SEED = 20260928
SPANS = [12, 36, 60]
N_OBS = 240
HISTORIES = [240, 120, 60, 36]
TRUE_BETA = 0.8
FACTOR_VOL = 0.04
RESIDUAL_VOL = 0.02
FIT_SPAN = 60
# Chosen so that the full-history loading is shrunk by about 0.05 under each weighting.
REG_LAMBDA = {FIT_SPAN: 2e-5, None: 1.6e-4}


def decay(span: float) -> float:
    """EWMA decay of a span: lambda = 1 - 2 / (span + 1)."""
    return 1.0 - 2.0 / (span + 1.0)


def weight_profile(span: float, n: int = 121) -> pd.Series:
    """Weight of the observation k periods before the last date, for k = 0 .. n-1."""
    weights = fl.compute_expanding_power(n=n, power_lambda=decay(span))
    return pd.Series(weights, index=pd.RangeIndex(n, name="periods_back"), name=f"span {span}")


def make_ragged_panel(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One zero-mean factor and four responses with equal loadings and ragged histories."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2006-01-31", periods=N_OBS, freq="ME")
    x = pd.DataFrame({"equity": FACTOR_VOL * rng.standard_normal(N_OBS)}, index=dates)
    noise = RESIDUAL_VOL * rng.standard_normal((N_OBS, len(HISTORIES)))
    y = pd.DataFrame(TRUE_BETA * x.to_numpy() + noise, index=dates,
                     columns=[f"history_{h}" for h in HISTORIES])
    for column, history in zip(y.columns, HISTORIES):
        y.iloc[: N_OBS - history, y.columns.get_loc(column)] = np.nan
    return x, y


def fit(x: pd.DataFrame, y: pd.DataFrame, loss_normalization: str, reg_lambda: float,
        span: float | None) -> fl.LassoModel:
    """LASSO with EWMA row weights, no demeaning, and the given loss normalisation."""
    model = fl.LassoModel(
        model_type=fl.LassoModelType.LASSO,
        reg_lambda=reg_lambda,
        span=span,
        demean=False,
        loss_normalization=loss_normalization,
    )
    return model.fit(x=x, y=y)


def shrinkage_by_history(x: pd.DataFrame, y: pd.DataFrame, span: float | None) -> pd.DataFrame:
    """Shrinkage of each loading against its own weighted least squares, both normalisations."""
    lam = 1.0 if span is None else decay(span)
    full_mass = float(np.sum(lam ** np.arange(N_OBS)))
    reg_lambda = REG_LAMBDA[span]
    fits = {
        "sample": fit(x, y, "sample", reg_lambda, span),
        # converted so that the full-history response is fitted identically
        "weight_sum": fit(x, y, "weight_sum", reg_lambda * N_OBS / full_mass, span),
    }
    weights = lam ** np.arange(N_OBS)[::-1]
    rows = []
    for column, history in zip(y.columns, HISTORIES):
        valid = y[column].notna().to_numpy()
        w, xv, yv = weights[valid], x["equity"].to_numpy()[valid], y[column].to_numpy()[valid]
        ols = np.sum(w * xv * yv) / np.sum(w * xv ** 2)
        row = {"history_months": history, "weight_mass": w.sum(), "ols": ols}
        for name, model in fits.items():
            row[name] = ols - model.coef_.loc[column, "equity"]
        row["sample_closed_form"] = reg_lambda * N_OBS / (2.0 * np.sum(w * xv ** 2))
        row["weight_sum_closed_form"] = (reg_lambda * N_OBS / full_mass * w.sum()
                                         / (2.0 * np.sum(w * xv ** 2)))
        rows.append(row)
    return pd.DataFrame(rows).set_index("history_months")


def main() -> None:
    # --- 1. EWMA conventions --------------------------------------------------------------------
    for span in SPANS:
        lam = decay(span)
        half_life = np.log(0.5) / np.log(lam)
        weights = lam ** np.arange(10_000)
        ess = weights.sum() ** 2 / np.sum(weights ** 2)
        print(f"span {span}: decay {lam:.3f}, half-life {half_life:.2f}, effective size {ess:.1f}")
        assert np.isclose(ess, span) and np.isclose(ess, (1 + lam) / (1 - lam))
        assert np.allclose(weight_profile(span).to_numpy(), lam ** np.arange(121))
    assert [round(np.log(0.5) / np.log(decay(s)), 2) for s in SPANS] == [4.15, 12.47, 20.79]

    # the running EWMA mean is pandas' recursive mean
    x, y = make_ragged_panel()
    running = fl.compute_ewm(x, span=FIT_SPAN)
    assert np.allclose(running.to_numpy(), x.ewm(span=FIT_SPAN, adjust=False).mean().to_numpy())

    # the solver's row weights are square roots of the EWMA weights, zero on missing cells
    _, _, valid = fl.get_x_y_np(x, y, span=None, demean=False)
    assert valid.sum(axis=0).tolist() == HISTORIES
    root = fl.compute_expanding_power(N_OBS, np.sqrt(decay(FIT_SPAN)), reverse_columns=True)
    assert np.isclose(root[-1], 1.0) and np.isclose(root[-2] ** 2, decay(FIT_SPAN))

    # --- 2. Ragged histories under the two loss normalisations ---------------------------------
    model = fit(x, y, "weight_sum", REG_LAMBDA[FIT_SPAN], FIT_SPAN)
    print(model.loss_weight_mass_.round(1))
    ewma = shrinkage_by_history(x, y, span=FIT_SPAN)
    flat = shrinkage_by_history(x, y, span=None)
    columns = ["weight_mass", "sample", "weight_sum"]
    print(ewma[columns].round(3))
    print(flat[columns].round(3))

    # the fitted weight mass and the closed-form shrinkage, to solver tolerance
    assert np.allclose(model.loss_weight_mass_.to_numpy(), ewma["weight_mass"].to_numpy())
    for table in (ewma, flat):
        assert np.allclose(table["sample"], table["sample_closed_form"], atol=1e-4)
        assert np.allclose(table["weight_sum"], table["weight_sum_closed_form"], atol=1e-4)
        assert abs(table.loc[240, "sample"] - table.loc[240, "weight_sum"]) < 1e-4

    # the joint "sample" fit of a short history equals its own-window fit at lambda * T / T_i
    column, history = "history_36", 36
    own = fit(x.iloc[-history:], y[[column]].iloc[-history:], "sample",
              REG_LAMBDA[None] * N_OBS / history, None)
    assert abs(own.coef_.loc[column, "equity"] - (flat.loc[36, "ols"] - flat.loc[36, "sample"])
               ) < 1e-4

    # quoted values
    assert np.round(ewma["weight_mass"].to_numpy(), 1).tolist() == [30.5, 29.9, 26.4, 21.3]
    ratio = {name: table.loc[36, ["sample", "weight_sum"]] / table.loc[240, ["sample",
                                                                            "weight_sum"]]
             for name, table in (("ewma", ewma), ("flat", flat))}
    print({name: value.round(2).to_dict() for name, value in ratio.items()})
    assert 1.2 < ratio["ewma"]["sample"] < 1.8 and 0.8 < ratio["ewma"]["weight_sum"] < 1.25
    assert 5.0 < ratio["flat"]["sample"] < 8.5 and 0.8 < ratio["flat"]["weight_sum"] < 1.25
    assert [round(float(ratio[name][c]), 1) for name, c in
            (("flat", "sample"), ("flat", "weight_sum"), ("ewma", "sample"))] == [7.3, 1.1, 1.3]
    # the shrinkage table of the article, rounded from the closed form the fits match to 1e-4
    quoted = {
        "flat": ([0.052, 0.105, 0.284, 0.379], [0.052, 0.053, 0.071, 0.057]),
        "ewma": ([0.058, 0.060, 0.069, 0.076], [0.058, 0.059, 0.060, 0.053]),
    }
    for name, table in (("flat", flat), ("ewma", ewma)):
        assert np.round(table["sample_closed_form"].to_numpy(), 3).tolist() == quoted[name][0]
        assert np.round(table["weight_sum_closed_form"].to_numpy(), 3).tolist() == quoted[name][1]
    assert round(decay(FIT_SPAN) ** 120, 2) == 0.02


if __name__ == "__main__":
    main()
