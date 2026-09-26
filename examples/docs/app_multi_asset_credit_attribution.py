"""Canonical example for docs/app_multi_asset_credit_attribution.md.

The JSS software paper (Sections 5.5 and 6) fits bond and equity funds on macro factors whose
Credit and Equity returns are 0.84 correlated. Shrink-to-zero penalties then move the credit
exposure of bond funds into Equity, while a penalty centred on economic credit priors keeps it.
This script builds the paper's production configuration on a small synthetic panel with the same
collinearity and asserts that mechanism. It does not reproduce the paper's numbers, which come
from the calibrated 102-fund design and the frozen market panel of ``papers/jss_2026``.

Independent references: ordinary least squares with ``numpy.linalg.lstsq`` at a vanishing
penalty, and the prior matrix itself at a strong penalty. Data are synthetic decimal monthly
returns with equal observation weights.
"""

import numpy as np
import pandas as pd

import factorlasso as fl

SEED = 20260927
N_OBS = 112
FACTORS = ["equity", "credit", "rates"]
FACTOR_VOL = np.array([0.045, 0.02, 0.015])
CREDIT_EQUITY_CORR = 0.84
TARGET_R2 = 0.7
# True loadings on (equity, credit, rates) per sleeve, before a 15% per-fund perturbation.
SLEEVES = {
    "EQ": ([1.0, 0.0, 0.0], 4),
    "IG": ([0.0, 0.2, 0.6], 3),
    "HY": ([0.1, 0.4, 0.2], 3),
    "EM": ([0.1, 0.3, 0.4], 3),
}
CREDIT_PRIORS = {"IG": 0.20, "HY": 0.40, "EM": 0.30}
REG_LAMBDAS = [1e-7, 1e-5, 1e-4, 1e-3]

# The production configuration of the JSS study (Table tab:lassomodel-params), without its
# EWMA span, as in the paper's stationary designs.
PRODUCTION = {
    "cutoff_fraction": 0.40,
    "auto_sign_constraints": True,
    "auto_sign_threshold_t": 1.0,
    "auto_sign_adaptive_weights": True,
    "auto_sign_adaptive_floor": 0.5,
}


def make_panel(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Factors, fund returns and the true loadings of one synthetic multi-asset panel."""
    rng = np.random.default_rng(seed)
    corr = np.eye(3)
    corr[0, 1] = corr[1, 0] = CREDIT_EQUITY_CORR
    cov = np.outer(FACTOR_VOL, FACTOR_VOL) * corr
    factors = rng.multivariate_normal(np.zeros(3), cov, size=N_OBS)
    names, rows = [], []
    for sleeve, (loadings, count) in SLEEVES.items():
        for k in range(count):
            names.append(f"{sleeve}_{k + 1}")
            rows.append(np.array(loadings) * (1.0 + 0.15 * rng.standard_normal(3)))
    beta = np.array(rows)
    systematic_var = np.einsum("ij,jk,ik->i", beta, cov, beta)
    residual_vol = np.sqrt(systematic_var * (1.0 - TARGET_R2) / TARGET_R2)
    returns = factors @ beta.T + residual_vol * rng.standard_normal((N_OBS, len(names)))
    dates = pd.date_range("2017-02-28", periods=N_OBS, freq="ME")
    x = pd.DataFrame(factors, index=dates, columns=FACTORS)
    y = pd.DataFrame(returns, index=dates, columns=names)
    return x, y, pd.DataFrame(beta, index=names, columns=FACTORS)


def credit_prior(funds: pd.Index) -> pd.DataFrame:
    """Economic credit centres for the bond sleeves; every other centre is zero."""
    prior = pd.DataFrame(0.0, index=funds, columns=FACTORS)
    for fund in funds:
        sleeve = fund.split("_")[0]
        if sleeve in CREDIT_PRIORS:
            prior.loc[fund, "credit"] = CREDIT_PRIORS[sleeve]
    return prior


def build_model(model_type: fl.LassoModelType, reg_lambda: float,
                prior: pd.DataFrame | None = None) -> fl.LassoModel:
    """The production configuration, with or without the credit prior."""
    return fl.LassoModel(
        model_type=model_type,
        reg_lambda=reg_lambda,
        factors_beta_prior=prior,
        **PRODUCTION,
    )


def credit_path(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Mean Credit loading of the bond funds for each configuration along the penalty grid."""
    prior = credit_prior(y.columns)
    bonds = [fund for fund in y.columns if not fund.startswith("EQ")]
    configurations = {
        "HCGL shrink-to-zero": (fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO, None),
        "HCGL + prior": (fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO, prior),
        "FCGL + prior": (fl.LassoModelType.FACTOR_CLUSTER_GROUP_LASSO, prior),
    }
    path = {}
    for name, (model_type, centre) in configurations.items():
        fits = [build_model(model_type, reg_lambda, centre).fit(x=x, y=y)
                for reg_lambda in REG_LAMBDAS]
        path[name] = [model.coef_.loc[bonds, "credit"].mean() for model in fits]
    return pd.DataFrame(path, index=pd.Index(REG_LAMBDAS, name="reg_lambda"))


def main() -> None:
    x, y, beta = make_panel()
    path = credit_path(x, y)
    print(path.round(3))

    bonds = [fund for fund in y.columns if not fund.startswith("EQ")]
    xd = x.to_numpy() - x.to_numpy().mean(axis=0)
    yd = y.to_numpy() - y.to_numpy().mean(axis=0)
    ols = pd.DataFrame(np.linalg.lstsq(xd, yd, rcond=None)[0].T, index=y.columns, columns=FACTORS)
    prior_mean = credit_prior(y.columns).loc[bonds, "credit"].mean()
    print(f"true mean credit loading {beta.loc[bonds, 'credit'].mean():.3f}, "
          f"OLS {ols.loc[bonds, 'credit'].mean():.3f}, prior {prior_mean:.3f}")

    # the collinearity of the study
    assert abs(np.corrcoef(x["credit"], x["equity"])[0, 1] - CREDIT_EQUITY_CORR) < 0.06

    # at a strong penalty the shrink-to-zero fit has lost the credit exposure of the bond funds,
    # while both prior-centred fits keep it at or above the prior
    strong = path.loc[REG_LAMBDAS[-1]]
    assert strong["HCGL shrink-to-zero"] < 0.25 * prior_mean
    assert strong["HCGL + prior"] > 0.8 * prior_mean
    assert strong["FCGL + prior"] > 0.8 * prior_mean

    # at a vanishing penalty all three return close to the unconstrained least-squares loading
    weak = path.loc[REG_LAMBDAS[0]]
    assert np.allclose(weak.to_numpy(), ols.loc[bonds, "credit"].mean(), atol=0.1)

    # the lost credit exposure is booked as equity by the shrink-to-zero fit, not by the prior fit
    hcgl = fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO
    equity = {
        name: build_model(hcgl, 1e-4, centre).fit(x=x, y=y).coef_.loc[bonds, "equity"].mean()
        for name, centre in (("shrink-to-zero", None), ("prior", credit_prior(y.columns)))
    }
    print({name: round(float(value), 3) for name, value in equity.items()},
          f"OLS {ols.loc[bonds, 'equity'].mean():.3f}")
    assert equity["shrink-to-zero"] > ols.loc[bonds, "equity"].mean() + 0.02
    assert equity["prior"] < equity["shrink-to-zero"]

    # the values the case study quotes
    assert round(ols.loc[bonds, "credit"].mean(), 2) == 0.28
    assert round(beta.loc[bonds, "credit"].mean(), 2) == 0.28
    assert strong["HCGL shrink-to-zero"] < 0.01
    assert abs(strong["HCGL + prior"] - 0.30) < 0.01 and abs(strong["FCGL + prior"] - 0.30) < 0.01
    assert round(ols.loc[bonds, "equity"].mean(), 2) == 0.06
    assert abs(equity["shrink-to-zero"] - 0.12) < 0.01


if __name__ == "__main__":
    main()
