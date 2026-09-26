"""Canonical example for docs/prior_targets.md: automatic and mapped OLS prior centres.

One inflation-linked-bond-like response loads on a Rates and an Inflation factor whose returns
are strongly negatively correlated. Its marginal Inflation slope is negative although its
conditional loading is positive, so the automatic highest-R-squared centre and a joint
Rates/Inflation centre lead to different sign sets and different fits. Every number the article
quotes is asserted here against a reference computed a different way:

* marginal and joint slopes against ``numpy.linalg.lstsq`` with an intercept column;
* the population marginal slopes against the omitted-variable identity;
* the small-penalty fits against ``scipy.optimize.lsq_linear`` with the derived sign bounds;
* the large-penalty fits against the centres themselves.

The data are synthetic decimal monthly returns with equal weights (``span=None``).
"""

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

import factorlasso as fl

SEED = 20260926
N_OBS = 360
RHO = -0.8
FACTOR_VOL = 0.02
TRUE_BETA = np.array([0.9, 0.45])
TARGET_R2 = 0.8
FACTORS = ["rates", "inflation"]
REG_LAMBDAS = np.logspace(-8, -2, 13)
POLICIES = {
    "zero centre": {},
    "automatic centre": {"apply_ols_prior": True},
    "joint centre": {
        "apply_ols_prior": True,
        "factor_for_prior": {"linker": ("rates", "inflation")},
    },
}


def make_panel(seed: int = SEED, n_obs: int = N_OBS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two correlated factors and one response with a population R-squared of TARGET_R2."""
    rng = np.random.default_rng(seed)
    cov = FACTOR_VOL**2 * np.array([[1.0, RHO], [RHO, 1.0]])
    factors = rng.multivariate_normal(np.zeros(2), cov, size=n_obs)
    systematic_var = TRUE_BETA @ cov @ TRUE_BETA
    residual_vol = np.sqrt(systematic_var * (1.0 - TARGET_R2) / TARGET_R2)
    response = factors @ TRUE_BETA + residual_vol * rng.standard_normal(n_obs)
    dates = pd.date_range("1996-01-31", periods=n_obs, freq="ME")
    x = pd.DataFrame(factors, index=dates, columns=FACTORS)
    y = pd.DataFrame({"linker": response}, index=dates)
    return x, y


def population_marginal_slopes() -> np.ndarray:
    """Omitted-variable identity for two factors with equal volatility."""
    beta_rates, beta_inflation = TRUE_BETA
    return np.array([beta_rates + beta_inflation * RHO, beta_inflation + beta_rates * RHO])


def fit(x: pd.DataFrame, y: pd.DataFrame, reg_lambda: float, **policy) -> fl.LassoModel:
    """LASSO with automatically derived signs and the given prior-centre policy."""
    model = fl.LassoModel(
        model_type=fl.LassoModelType.LASSO,
        reg_lambda=reg_lambda,
        auto_sign_constraints=True,
        **policy,
    )
    return model.fit(x=x, y=y)


def loading_paths(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Rates and Inflation loadings of each policy along the penalty grid."""
    rows = []
    for name, policy in POLICIES.items():
        for reg_lambda in REG_LAMBDAS:
            coef = fit(x, y, reg_lambda, **policy).coef_.loc["linker"]
            rows.append({"policy": name, "reg_lambda": reg_lambda, **coef.to_dict()})
    return pd.DataFrame(rows)


def main() -> None:
    x, y = make_panel()

    # --- centres -----------------------------------------------------------------------------
    automatic = fit(x, y, 1e-4, **POLICIES["automatic centre"])
    joint = fit(x, y, 1e-4, **POLICIES["joint centre"])
    print(automatic.ols_betas_.round(3))
    print(automatic.ols_r2_.round(3))
    print(automatic.effective_beta_prior_.round(3))
    print(joint.effective_beta_prior_.round(3))

    # --- signs -------------------------------------------------------------------------------
    zero = fit(x, y, 1e-4, **POLICIES["zero centre"])
    print(zero.derived_signs_)
    print(joint.derived_signs_)

    # --- where each policy leads -------------------------------------------------------------
    paths = loading_paths(x, y)
    print(paths.groupby("policy").agg(first=("inflation", "first"), last=("inflation", "last")))

    # --- independent references --------------------------------------------------------------
    xv, yv = x.to_numpy(), y["linker"].to_numpy()
    design = np.column_stack([np.ones(len(xv)), xv])
    marginal = np.array([
        np.linalg.lstsq(design[:, [0, j + 1]], yv, rcond=None)[0][1] for j in range(2)
    ])
    joint_ols = np.linalg.lstsq(design, yv, rcond=None)[0][1:]
    r2 = []
    for j in range(2):
        slope_fit = np.linalg.lstsq(design[:, [0, j + 1]], yv, rcond=None)[0]
        residual = yv - design[:, [0, j + 1]] @ slope_fit
        r2.append(1.0 - residual @ residual / np.sum((yv - yv.mean()) ** 2))

    # OLS diagnostics and centres
    assert np.allclose(automatic.ols_betas_.loc["linker"].to_numpy(), marginal, atol=1e-8)
    assert np.allclose(automatic.ols_r2_.loc["linker"].to_numpy(), r2, atol=1e-8)
    assert automatic.ols_r2_.loc["linker"].idxmax() == "rates"
    assert np.allclose(automatic.effective_beta_prior_.loc["linker"].to_numpy(),
                       [marginal[0], 0.0], atol=1e-8)
    assert np.allclose(joint.effective_beta_prior_.loc["linker"].to_numpy(), joint_ols, atol=1e-8)

    # the omitted-variable identity: population slopes 0.54 and -0.27, sample within sampling error
    population = population_marginal_slopes()
    assert np.allclose(population, [0.54, -0.27])
    assert np.all(np.abs(marginal - population) < 0.08)
    assert np.all(np.abs(joint_ols - TRUE_BETA) < 0.08)

    # the values the article quotes, rounded as printed
    assert np.round(marginal, 3).tolist() == [0.544, -0.261]
    assert np.round(r2, 3).tolist() == [0.634, 0.152]
    assert np.round(joint_ols, 3).tolist() == [0.929, 0.468]

    # signs: detection makes Inflation non-positive; the positive joint centre overrides it
    assert zero.derived_signs_.loc["linker"].tolist() == [1.0, -1.0]
    assert automatic.derived_signs_.loc["linker"].tolist() == [1.0, -1.0]
    assert joint.derived_signs_.loc["linker"].tolist() == [1.0, 1.0]

    # small penalty: the sign-constrained least-squares fit of each sign set
    xd, yd = xv - xv.mean(axis=0), yv - yv.mean()
    detected = lsq_linear(xd, yd, bounds=([0.0, -np.inf], [np.inf, 0.0])).x
    prior_informed = lsq_linear(xd, yd, bounds=([0.0, 0.0], [np.inf, np.inf])).x
    small = paths[paths["reg_lambda"] == REG_LAMBDAS[0]].set_index("policy")[FACTORS]
    for name in ("zero centre", "automatic centre"):
        assert np.allclose(small.loc[name].to_numpy(), detected, atol=1e-4)
    assert np.allclose(small.loc["joint centre"].to_numpy(), prior_informed, atol=1e-4)
    assert np.allclose(detected, [marginal[0], 0.0], atol=1e-6)
    assert np.allclose(prior_informed, joint_ols, atol=1e-6)

    # large penalty: every fit returns its centre
    large = paths[paths["reg_lambda"] == REG_LAMBDAS[-1]].set_index("policy")[FACTORS]
    assert np.allclose(large.loc["zero centre"].to_numpy(), [0.0, 0.0], atol=1e-3)
    assert np.allclose(large.loc["automatic centre"].to_numpy(), [marginal[0], 0.0], atol=1e-3)
    assert np.allclose(large.loc["joint centre"].to_numpy(), joint_ols, atol=1e-3)

    # an explicit hard sign still wins: its conflicting centre is set to zero, not moved
    hard = pd.DataFrame({"rates": [np.nan], "inflation": [-1.0]}, index=["linker"])
    blocked = fit(x, y, 1e-4, factors_beta_loading_signs=hard, **POLICIES["joint centre"])
    assert blocked.derived_signs_.loc["linker", "inflation"] == -1.0
    assert blocked.effective_beta_prior_.loc["linker", "inflation"] == 0.0
    assert np.isclose(blocked.effective_beta_prior_.loc["linker", "rates"], joint_ols[0])


if __name__ == "__main__":
    main()
