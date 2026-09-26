"""
Worked example for ``docs/sign_constraints_and_priors.md``: economic knowledge as constraints.
==============================================================================================

Six synthetic bond funds load on rates, credit and equity. Credit and equity are 85% correlated
and the history is 36 months, so an unconstrained fit moves exposure between the two factors and
contradicts what is known about the funds. The script imposes a cell-level sign matrix, then
centres the penalty on a prior loading matrix, and measures what each step buys against the known
loadings, on one panel and over 50 redrawn panels.

Every number quoted in the article is asserted here against a reference computed a different way:
the sign-constrained fit against ``scipy.optimize.lsq_linear``, the prior-centred fit against a
zero-centred fit of the prior's residual, the large-penalty limit against the prior itself, and the
reported sign matrix against the one supplied.

Synthetic data, fixed seed, no network, no files written. Solver: CVXPY with CLARABEL.
"""

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

import factorlasso as fl

SEED = 20260923
N_OBS = 36                         # monthly observations
FACTOR_NAMES = ["rates", "credit", "equity"]
FACTOR_VOL = np.array([0.015, 0.020, 0.045])      # per-month volatility, decimal returns
FACTOR_CORR = np.array([
    [1.00, -0.10, -0.20],
    [-0.10, 1.00, 0.85],
    [-0.20, 0.85, 1.00],
])
ASSET_NAMES = [
    "gov_bond_1", "gov_bond_2", "ig_credit_1", "ig_credit_2", "hy_credit_1", "hy_credit_2",
]
RESIDUAL_VOL = 0.010               # per-month idiosyncratic volatility, decimal returns
REG_LAMBDA = 3e-5
PENALTY_GRID = np.logspace(-6.0, -2.0, 13)
N_PANELS = 50                      # redrawn panels for the sampling comparison

# True loadings, responses by factors.
TRUE_BETA = np.array([
    [1.0, 0.0, 0.0],
    [0.7, 0.0, 0.0],
    [0.8, 0.4, 0.0],
    [0.6, 0.6, 0.0],
    [0.2, 1.0, 0.1],
    [0.1, 1.2, 0.2],
])

# Sign matrix: 1 non-negative, -1 non-positive, 0 forced to zero, NaN free.
SIGNS = np.array([
    [1.0, 0.0, 0.0],               # government bonds: duration only
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],               # investment grade: duration and credit, no equity
    [1.0, 1.0, 0.0],
    [1.0, 1.0, np.nan],            # high yield: equity loading left free
    [1.0, 1.0, np.nan],
])

# Prior loadings: a plausible house view, deliberately not equal to the truth.
PRIOR = np.array([
    [0.8, 0.0, 0.0],
    [0.8, 0.0, 0.0],
    [0.8, 0.5, 0.0],
    [0.8, 0.5, 0.0],
    [0.3, 1.0, 0.0],
    [0.3, 1.0, 0.0],
])


def make_panel(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw the factor panel ``x`` (T x 3) and the response panel ``y`` (T x 6)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-31", periods=N_OBS, freq="ME")
    factor_covar = FACTOR_CORR * np.outer(FACTOR_VOL, FACTOR_VOL)
    x = pd.DataFrame(
        rng.multivariate_normal(np.zeros(len(FACTOR_NAMES)), factor_covar, size=N_OBS),
        index=dates,
        columns=FACTOR_NAMES,
    )
    noise = RESIDUAL_VOL * rng.standard_normal((N_OBS, len(ASSET_NAMES)))
    y = pd.DataFrame(x.to_numpy() @ TRUE_BETA.T + noise, index=dates, columns=ASSET_NAMES)
    return x, y


def as_frame(values: np.ndarray) -> pd.DataFrame:
    """Label an (N x M) array by asset and factor, the layout ``LassoModel`` expects."""
    return pd.DataFrame(values, index=ASSET_NAMES, columns=FACTOR_NAMES)


def fit_free(x: pd.DataFrame, y: pd.DataFrame, reg_lambda: float = REG_LAMBDA) -> fl.LassoModel:
    """LASSO with no sign constraint and the penalty centred on zero."""
    return fl.LassoModel(reg_lambda=reg_lambda).fit(x=x, y=y)


def fit_signed(x: pd.DataFrame, y: pd.DataFrame, reg_lambda: float = REG_LAMBDA) -> fl.LassoModel:
    """LASSO under the cell-level sign matrix, penalty centred on zero."""
    model = fl.LassoModel(
        reg_lambda=reg_lambda,
        factors_beta_loading_signs=as_frame(SIGNS),
    )
    return model.fit(x=x, y=y)


def fit_signed_with_prior(
    x: pd.DataFrame,
    y: pd.DataFrame,
    reg_lambda: float = REG_LAMBDA,
) -> fl.LassoModel:
    """LASSO under the sign matrix with the penalty ``reg_lambda * ||beta - prior||_1``."""
    model = fl.LassoModel(
        reg_lambda=reg_lambda,
        factors_beta_loading_signs=as_frame(SIGNS),
        factors_beta_prior=as_frame(PRIOR),
    )
    return model.fit(x=x, y=y)


ESTIMATORS = {
    "free": fit_free,
    "signs": fit_signed,
    "signs and prior": fit_signed_with_prior,
}


def loading_rmse(model: fl.LassoModel) -> float:
    """Root mean squared error of the fitted loadings against ``TRUE_BETA``."""
    return float(np.sqrt(np.mean((model.coef_.to_numpy() - TRUE_BETA) ** 2)))


def violation_share(model: fl.LassoModel, tolerance: float = 1e-3) -> float:
    """Share of cells whose fitted loading contradicts the sign matrix (zero cells included)."""
    beta = model.coef_.to_numpy()
    wrong = np.where(SIGNS == 0.0, np.abs(beta) > tolerance, SIGNS * beta < -tolerance)
    return float(wrong[np.isfinite(SIGNS)].mean())


def penalty_path(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Loading error along the grid for the zero-centred and the prior-centred penalty."""
    rows = []
    for reg_lambda in PENALTY_GRID:
        rows.append({
            "reg_lambda": float(reg_lambda),
            "signs": loading_rmse(fit_signed(x, y, reg_lambda=float(reg_lambda))),
            "signs and prior": loading_rmse(
                fit_signed_with_prior(x, y, reg_lambda=float(reg_lambda))),
        })
    return pd.DataFrame(rows).set_index("reg_lambda")


def sampling_comparison(n_panels: int = N_PANELS) -> pd.DataFrame:
    """Loading error and violation share of each estimator over redrawn panels."""
    rows = []
    for draw in range(n_panels):
        x, y = make_panel(seed=SEED + 1 + draw)
        for name, fit in ESTIMATORS.items():
            model = fit(x, y)
            rows.append({"draw": draw, "estimator": name, "loading_rmse": loading_rmse(model),
                         "violation_share": violation_share(model)})
    return pd.DataFrame(rows)


def main() -> None:
    """Run the example, verify each quoted number independently, and print the summary."""
    x, y = make_panel()
    models = {name: fit(x, y) for name, fit in ESTIMATORS.items()}
    free, signed, centred = (models[name].coef_.to_numpy() for name in ESTIMATORS)

    # --- 1. The sign matrix: reported as supplied, obeyed by the fit ---
    reported = models["signs"].derived_signs_
    assert reported.equals(as_frame(SIGNS))
    assert np.all(np.abs(signed[SIGNS == 0.0]) < 1e-7)
    assert np.all(signed[SIGNS == 1.0] > -1e-7)
    assert violation_share(models["signs"]) == 0.0 and violation_share(models["free"]) > 0.0

    # bounded least squares from SciPy as the zero-penalty reference, one response at a time
    x_centred = x.to_numpy() - x.to_numpy().mean(axis=0)
    y_centred = y.to_numpy() - y.to_numpy().mean(axis=0)
    lower = np.where(SIGNS == 1.0, 0.0, -np.inf)
    lower = np.where(SIGNS == 0.0, -1e-12, lower)
    upper = np.where(SIGNS == 0.0, 1e-12, np.inf)
    bounded = np.vstack([
        lsq_linear(x_centred, y_centred[:, i], bounds=(lower[i], upper[i]), tol=1e-14).x
        for i in range(len(ASSET_NAMES))
    ])
    assert np.allclose(fit_signed(x, y, reg_lambda=1e-12).coef_.to_numpy(), bounded, atol=1e-4)

    # --- 2. The prior: a shifted penalty is a zero-centred fit of what the prior leaves over ---
    unconstrained = fl.LassoModel(
        reg_lambda=REG_LAMBDA, factors_beta_prior=as_frame(PRIOR)).fit(x=x, y=y)
    leftover = y - x.to_numpy() @ PRIOR.T
    shifted = PRIOR + fit_free(x, leftover).coef_.to_numpy()
    assert np.allclose(unconstrained.coef_.to_numpy(), shifted, atol=1e-4)

    # a large penalty returns the prior wherever the prior is feasible ...
    at_large_penalty = fit_signed_with_prior(x, y, reg_lambda=1.0).coef_.to_numpy()
    assert np.allclose(at_large_penalty, PRIOR, atol=1e-5)
    # ... and the constraint wins where the prior contradicts it
    contradicting = PRIOR.copy()
    contradicting[4, 0] = -0.5                                         # a negative rates prior
    clipped = fl.LassoModel(
        reg_lambda=1.0,
        factors_beta_loading_signs=as_frame(SIGNS),
        factors_beta_prior=as_frame(contradicting),
    ).fit(x=x, y=y)
    assert abs(clipped.coef_.iloc[4, 0]) < 1e-6

    # --- 3. What each step buys on this panel ---
    errors = {name: loading_rmse(model) for name, model in models.items()}
    prior_rmse = float(np.sqrt(np.mean((PRIOR - TRUE_BETA) ** 2)))
    assert errors["signs and prior"] < errors["signs"] < errors["free"]
    assert errors["signs and prior"] < prior_rmse                      # the data improve the view

    path = penalty_path(x, y)
    zero_rmse = float(np.sqrt(np.mean(TRUE_BETA ** 2)))
    assert np.isclose(path["signs"].iloc[-1], zero_rmse, atol=1e-3)     # shrinks to zero
    assert np.isclose(path["signs and prior"].iloc[-1], prior_rmse, atol=1e-3)   # to the prior

    # --- 4. The same comparison over redrawn panels ---
    sample = sampling_comparison()
    mean_rmse = sample.groupby("estimator")["loading_rmse"].mean()
    mean_violation = sample.groupby("estimator")["violation_share"].mean()
    assert mean_rmse["signs and prior"] < mean_rmse["signs"] < mean_rmse["free"]
    assert mean_violation["signs"] == 0.0 and mean_violation["free"] > 0.1

    print(f"One panel, reg_lambda = {REG_LAMBDA:.0e}")
    for name, model in models.items():
        print(f"  {name:<16}: loading RMSE {errors[name]:.3f},"
              f" violation share {violation_share(model):.2f}")
    print(f"  prior itself    : loading RMSE {prior_rmse:.3f}")
    print("Free fit")
    print(models["free"].coef_.round(2).to_string())
    print("Signs and prior")
    print(models["signs and prior"].coef_.round(2).to_string())
    print("Loading RMSE along the penalty grid")
    print(path.round(3).to_string())
    print(f"Means over {N_PANELS} redrawn panels")
    print(pd.DataFrame({"loading_rmse": mean_rmse, "violation_share": mean_violation})
          .loc[list(ESTIMATORS)].round(3).to_string())


if __name__ == "__main__":
    main()
