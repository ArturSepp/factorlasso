"""Canonical example for docs/app_portfolio_risk_models.md.

Eight asset-class sleeves load on three tradable factors, Equity, Rates and Credit. Loadings are
fitted by factorlasso on ten years of synthetic monthly returns and assembled with the factor
covariance and residual variances into one risk model. The same loadings, applied to annual factor
premia, give the capital market assumptions; two sleeves carry declared adjustments. The script
checks the assembly against NumPy, then audits the CMA vector as the MATF-CMA paper does: the GLS
projection on the loadings, its residual alpha, and the split of the squared Sharpe ratio.
Synthetic data; the premia and adjustments are illustrative, not the paper's.
"""

import numpy as np
import pandas as pd

import factorlasso as fl

SEED = 20261011
N_OBS = 120
REG_LAMBDA = 1e-6
FACTORS = ["Equity", "Rates", "Credit"]
FACTOR_VOL = np.array([0.16, 0.05, 0.07])                # annual
FACTOR_CORR = np.array([[1.0, 0.0, 0.6], [0.0, 1.0, 0.2], [0.6, 0.2, 1.0]])
SLEEVES = pd.DataFrame(                                   # true loadings and annual residual vol
    [[1.00, 0.00, 0.00, 0.03], [1.05, 0.00, 0.00, 0.06], [1.15, 0.00, 0.00, 0.10],
     [0.00, 1.00, 0.00, 0.01], [0.00, 0.70, 0.60, 0.02], [0.20, 0.30, 1.00, 0.04],
     [1.10, 0.00, 0.20, 0.12], [0.35, 0.00, 0.25, 0.05]],
    index=["US equity", "Europe equity", "EM equity", "Government bonds", "IG credit",
           "HY credit", "Private equity", "Hedge funds"],
    columns=[*FACTORS, "residual_vol"])
RISK_FREE = 0.03
PREMIA = pd.Series([0.050, 0.010, 0.015], index=FACTORS)  # annual factor excess returns
ADJUSTMENTS = pd.Series({"Private equity": 0.020, "Hedge funds": 0.015})


def make_returns(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly factor and sleeve returns with the stated loadings."""
    rng = np.random.default_rng(seed)
    monthly_cov = np.outer(FACTOR_VOL, FACTOR_VOL) * FACTOR_CORR / 12.0
    dates = pd.date_range("2016-01-31", periods=N_OBS, freq="ME")
    x = pd.DataFrame(rng.multivariate_normal(np.zeros(3), monthly_cov, size=N_OBS),
                     index=dates, columns=FACTORS)
    noise = rng.standard_normal((N_OBS, len(SLEEVES))) * SLEEVES["residual_vol"].to_numpy()
    y = x.to_numpy() @ SLEEVES[FACTORS].T.to_numpy() + noise / np.sqrt(12.0)
    return x, pd.DataFrame(y, index=dates, columns=SLEEVES.index)


def risk_model(x: pd.DataFrame, y: pd.DataFrame) -> fl.CurrentFactorCovarData:
    """Fitted loadings with the annualised factor covariance and residual variances."""
    model = fl.LassoModel(reg_lambda=REG_LAMBDA).fit(x=x, y=y)
    residuals = y - x.to_numpy() @ model.coef_.T.to_numpy()
    return fl.CurrentFactorCovarData(
        x_covar=12.0 * x.cov(),
        y_betas=model.coef_,
        y_variances=pd.DataFrame({fl.VarianceColumns.RESIDUAL_VARS.value: 12.0 * residuals.var()}),
        estimation_date=x.index[-1],
    )


def capital_market_assumptions(betas: pd.DataFrame) -> pd.Series:
    """CMA_i = r_f + beta_i' lambda + declared adjustment_i."""
    return RISK_FREE + betas @ PREMIA + ADJUSTMENTS.reindex(betas.index).fillna(0.0)


def audit(m: pd.Series, betas: pd.DataFrame, residual_var: pd.Series) -> dict:
    """GLS projection of excess CMAs on the loadings: implied premia and residual alpha."""
    b, d_inv = betas.to_numpy(), 1.0 / residual_var.to_numpy()
    gram = b.T @ (d_inv[:, None] * b)
    premia = np.linalg.solve(gram, b.T @ (d_inv * m.to_numpy()))
    alpha = m.to_numpy() - b @ premia
    return {"premia": pd.Series(premia, index=betas.columns),
            "alpha": pd.Series(alpha, index=m.index),
            "sr2_alpha": float(alpha @ (d_inv * alpha))}


def main() -> None:
    x, y = make_returns()
    snapshot = risk_model(x, y)
    betas = snapshot.y_betas
    sigma_f = snapshot.x_covar.to_numpy()
    d = snapshot.y_variances[fl.VarianceColumns.RESIDUAL_VARS.value]

    # --- one risk model: Sigma = B Sigma_F B' + w D --------------------------------------------
    full = snapshot.get_y_covar(residual_var_weight=1.0)
    systematic = snapshot.get_y_covar(residual_var_weight=0.0)
    assert np.allclose(full, betas.to_numpy() @ sigma_f @ betas.T.to_numpy() + np.diag(d))
    assert np.allclose(systematic, betas.to_numpy() @ sigma_f @ betas.T.to_numpy())
    portfolio = pd.Series({"US equity": 0.35, "Europe equity": 0.15, "EM equity": 0.10,
                           "Government bonds": 0.20, "IG credit": 0.15, "HY credit": 0.05})
    w = portfolio.reindex(betas.index).fillna(0.0).to_numpy()
    vols = {"full": np.sqrt(w @ full.to_numpy() @ w),
            "factor only": np.sqrt(w @ systematic.to_numpy() @ w)}
    tilt = pd.Series({"EM equity": 0.10, "US equity": -0.10, "HY credit": 0.05,
                      "IG credit": -0.05})
    a = tilt.reindex(betas.index).fillna(0.0).to_numpy()
    tracking = {"full": np.sqrt(a @ full.to_numpy() @ a),
                "factor only": np.sqrt(a @ systematic.to_numpy() @ a)}
    print({k: round(100 * v, 2) for k, v in vols.items()},
          {k: round(100 * v, 2) for k, v in tracking.items()})

    # --- the same loadings give the capital market assumptions ---------------------------------
    cma = capital_market_assumptions(betas)
    m = cma - RISK_FREE
    result = audit(m, betas, d)
    assert np.allclose(betas.T.to_numpy() @ (result["alpha"] / d).to_numpy(), 0.0)
    sigma_inv = np.linalg.inv(full.to_numpy())
    total = float(m @ sigma_inv @ m)
    fitted = betas.to_numpy() @ result["premia"].to_numpy()
    systematic_sr2 = float(fitted @ sigma_inv @ fitted)
    assert np.isclose(total, systematic_sr2 + result["sr2_alpha"])  # MATF-CMA Appendix A
    ceiling = float(PREMIA @ np.linalg.inv(sigma_f) @ PREMIA)
    achievable = float((betas @ PREMIA) @ sigma_inv @ (betas @ PREMIA))
    location = (result["alpha"] ** 2 / d) / result["sr2_alpha"]
    print((100 * cma).round(2).to_dict())
    print(result["premia"].round(4).to_dict(), round(result["sr2_alpha"], 3))
    print(location.round(3).to_dict())
    print(round(total, 3), round(systematic_sr2, 3), round(ceiling, 3), round(achievable, 3))

    # quoted values
    assert [round(100 * vols[k], 2) for k in vols] == [11.26, 11.13]
    assert [round(100 * tracking[k], 2) for k in tracking] == [1.11, 0.44]
    assert [round(100 * cma[k], 2) for k in ("Private equity", "Hedge funds")] == [10.2, 6.85]
    assert cma.idxmin() == "Government bonds" and round(100 * cma.min(), 1) == 4.0
    assert (100 * result["premia"]).round(2).tolist() == [5.22, 1.0, 1.62]
    assert [round(v, 3) for v in (total, systematic_sr2, result["sr2_alpha"])] == [
        0.281, 0.145, 0.136]
    assert location.round(2)[["Hedge funds", "Private equity"]].tolist() == [0.72, 0.2]
    assert [round(ceiling, 3), round(achievable, 3)] == [0.142, 0.137]


if __name__ == "__main__":
    main()
