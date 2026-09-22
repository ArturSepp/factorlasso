"""
Worked example for ``docs/quickstart.md``: from two return panels to a factor covariance matrix.
=================================================================================================

Twelve synthetic monthly fund returns load sparsely on four correlated factors. The script walks
the whole package once: select the penalty by expanding-window cross-validation, fit a
hierarchical-cluster group LASSO with data-derived sign constraints, read the fitted loadings,
clusters and signs, test the residuals for diagonality, and assemble ``Sigma_y = beta Sigma_x
beta' + D``.

Every number quoted in the quickstart is asserted here against a reference computed a different
way: predictions and R-squared from NumPy, the assembled covariance from a direct matrix product,
the selected penalty from the score table, and the estimation error from the known true loadings.

Synthetic data, fixed seed, no network, no files written. Solver: CVXPY with CLARABEL.
"""

import numpy as np
import pandas as pd

import factorlasso as fl

SEED = 20260921
N_OBS = 120                        # monthly observations
PERIODS_PER_YEAR = 12
FACTOR_NAMES = ["equity", "rates", "credit", "commodity"]
FACTOR_VOL = np.array([0.045, 0.015, 0.020, 0.050])    # per-month volatility, decimal returns
FACTOR_CORR = np.array([
    [1.0, -0.2, 0.6, 0.3],
    [-0.2, 1.0, 0.0, 0.0],
    [0.6, 0.0, 1.0, 0.2],
    [0.3, 0.0, 0.2, 1.0],
])
GROUP_NAMES = ["equity_fund", "bond_fund", "real_asset_fund"]
GROUP_SIZE = 4
RESIDUAL_VOL = np.repeat([0.015, 0.005, 0.025], GROUP_SIZE)   # per-month idiosyncratic volatility
PENALTY_GRID = np.logspace(-6.0, -2.0, 9)
N_SPLITS = 4

# True loadings, responses by factors: 20 of 48 cells are non-zero and each group shares a support.
TRUE_BETA = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.9, 0.0, 0.0, 0.0],
    [1.1, 0.0, 0.0, 0.0],
    [0.8, 0.0, 0.0, 0.0],
    [0.0, 1.2, 0.3, 0.0],
    [0.0, 1.0, 0.5, 0.0],
    [0.0, 0.7, 0.8, 0.0],
    [0.0, 0.5, 1.0, 0.0],
    [0.3, 0.0, 0.0, 0.9],
    [0.4, 0.0, 0.0, 0.8],
    [0.5, 0.0, 0.0, 0.6],
    [0.2, 0.0, 0.0, 0.7],
])


def make_panel(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw the factor panel ``x`` (T x 4) and the response panel ``y`` (T x 12)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-31", periods=N_OBS, freq="ME")
    factor_covar = FACTOR_CORR * np.outer(FACTOR_VOL, FACTOR_VOL)
    x = pd.DataFrame(
        rng.multivariate_normal(np.zeros(len(FACTOR_NAMES)), factor_covar, size=N_OBS),
        index=dates,
        columns=FACTOR_NAMES,
    )
    names = [f"{group}_{i + 1}" for group in GROUP_NAMES for i in range(GROUP_SIZE)]
    noise = RESIDUAL_VOL * rng.standard_normal((N_OBS, len(names)))
    y = pd.DataFrame(x.to_numpy() @ TRUE_BETA.T + noise, index=dates, columns=names)
    return x, y


def select_and_fit(x: pd.DataFrame, y: pd.DataFrame) -> fl.LassoModelCV:
    """Select ``reg_lambda`` by out-of-sample R-squared and refit on the full sample."""
    template = fl.LassoModel(
        model_type=fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
        auto_sign_constraints=True,        # signs from cluster-pooled univariate slopes
    )
    selector = fl.LassoModelCV(
        lambdas=PENALTY_GRID,
        n_splits=N_SPLITS,
        base_model=template,
        use_lambda_path=True,              # one canonical form per fold, swept over the grid
    )
    return selector.fit(x=x, y=y)


def check_residuals(
    model: fl.LassoModel,
    x: pd.DataFrame,
    y: pd.DataFrame,
) -> fl.ResidualDiagnostics:
    """Test the in-sample residuals for a diagonal covariance, charging the loadings kept."""
    residuals = y - model.predict(x)
    sparsity = fl.effective_sparsity(model.coef_)          # the solver returns no exact zeros
    return fl.diagnose_residuals(residuals, n_fitted_per_asset=sparsity.per_asset)


def assemble_covariance(model: fl.LassoModel, x: pd.DataFrame) -> fl.CurrentFactorCovarData:
    """Collect loadings, factor covariance and residual variances, all annualised."""
    result = model.estimation_result_
    y_variances = pd.DataFrame(
        {
            fl.VarianceColumns.EWMA_VARIANCE.value: PERIODS_PER_YEAR * result.ss_total,
            fl.VarianceColumns.RESIDUAL_VARS.value: PERIODS_PER_YEAR * result.ss_res,
            fl.VarianceColumns.INSAMPLE_ALPHA.value: PERIODS_PER_YEAR * model.alpha_const_.values,
            fl.VarianceColumns.R2.value: result.r2,
        },
        index=model.coef_.index,
    )
    return fl.CurrentFactorCovarData(
        x_covar=PERIODS_PER_YEAR * x.cov(ddof=0),
        y_betas=model.coef_,
        y_variances=y_variances,
        estimation_date=x.index[-1],
        clusters=model.clusters_,
        derived_signs=model.derived_signs_,
    )


def true_covariance() -> np.ndarray:
    """Annualised population covariance of the twelve responses."""
    factor_covar = FACTOR_CORR * np.outer(FACTOR_VOL, FACTOR_VOL)
    return PERIODS_PER_YEAR * (TRUE_BETA @ factor_covar @ TRUE_BETA.T + np.diag(RESIDUAL_VOL ** 2))


def relative_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Frobenius norm of the error relative to the Frobenius norm of the truth."""
    return float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth))


def main() -> None:
    """Run the quickstart, verify each quoted number independently, and print the summary."""
    x, y = make_panel()

    # --- 1. Penalty selection: the chosen penalty maximises the mean held-out R-squared ---
    selector = select_and_fit(x, y)
    mean_scores = selector.cv_scores_.mean(axis=1)
    assert selector.cv_scores_.shape == (len(PENALTY_GRID), N_SPLITS)
    assert np.isclose(selector.best_lambda_, mean_scores.idxmax())
    assert np.isclose(selector.best_lambda_, 1e-5)
    assert 0.84 < selector.best_score_ < 0.86                           # quoted: 0.85
    model = selector.best_model_

    # --- 2. Fitted loadings, clusters and signs ---
    beta = model.coef_.to_numpy()
    assert model.coef_.shape == TRUE_BETA.shape
    expected_partition = np.repeat(np.arange(len(GROUP_NAMES)), GROUP_SIZE)
    same_cluster = model.clusters_.to_numpy()[:, None] == model.clusters_.to_numpy()[None, :]
    assert np.array_equal(same_cluster, expected_partition[:, None] == expected_partition[None, :])
    signs = model.derived_signs_
    assert signs.groupby(model.clusters_).nunique(dropna=False).eq(1).all().all()   # pooled
    allowed = np.where(signs.to_numpy() > 0, beta >= -1e-8, beta <= 1e-8)
    assert allowed[np.isfinite(signs.to_numpy())].all()                 # the fit obeys its signs

    # predictions and score against NumPy
    fitted = model.alpha_const_.to_numpy() + x.to_numpy() @ beta.T
    assert np.allclose(model.predict(x).to_numpy(), fitted)
    residuals = y - model.predict(x)
    r2_by_hand = 1.0 - residuals.var(ddof=0) / y.var(ddof=0)
    assert np.isclose(model.score(x, y), r2_by_hand.mean())

    # estimation error against the known loadings, with ordinary least squares as the reference
    design = np.column_stack([np.ones(N_OBS), x.to_numpy()])
    ols = np.linalg.lstsq(design, y.to_numpy(), rcond=None)[0][1:].T
    rmse_model = float(np.sqrt(np.mean((beta - TRUE_BETA) ** 2)))
    rmse_ols = float(np.sqrt(np.mean((ols - TRUE_BETA) ** 2)))
    assert 0.050 < rmse_model < 0.056 and 0.088 < rmse_ols < 0.094       # quoted: 0.053 and 0.091
    sparsity = fl.effective_sparsity(model.coef_)
    assert sparsity.n_nonzero == 34 and int(np.count_nonzero(TRUE_BETA)) == 20

    # --- 3. Residual diagnostics: nothing systematic is left in the residuals ---
    diagnostics = check_residuals(model, x, y)
    assert np.isclose(diagnostics.nu, N_OBS - sparsity.per_asset - 1.0)
    assert diagnostics.passes and diagnostics.n_above_edge == 0
    assert 74.0 < diagnostics.sphericity < 75.5 < 85.5 < diagnostics.threshold < 86.5

    # --- 4. Covariance assembly against a direct matrix product ---
    covar_data = assemble_covariance(model, x)
    sigma_y = covar_data.get_y_covar()
    direct = beta @ covar_data.x_covar.to_numpy() @ beta.T + np.diag(
        PERIODS_PER_YEAR * model.estimation_result_.ss_res)
    assert np.allclose(sigma_y.to_numpy(), direct)
    assert np.linalg.eigvalsh(sigma_y.to_numpy()).min() > 0.0
    vols = covar_data.get_model_vols()
    assert np.allclose(vols["total_vol"] ** 2, vols["sys_vol"] ** 2 + vols["resid_vol"] ** 2)
    assert np.allclose(vols["total_vol"] ** 2, np.diag(sigma_y.to_numpy()))
    snapshot = covar_data.get_snapshot()                                # loadings, R2, alpha, vols
    assert snapshot.shape == (TRUE_BETA.shape[0], len(FACTOR_NAMES) + 6)
    error_model = relative_error(sigma_y.to_numpy(), true_covariance())
    error_sample = relative_error(PERIODS_PER_YEAR * y.cov(ddof=0).to_numpy(), true_covariance())

    print(f"Selected reg_lambda       : {selector.best_lambda_:.0e}"
          f" (mean held-out R-squared {selector.best_score_:.3f})")
    print(f"In-sample R-squared       : {model.score(x, y):.3f}")
    print(f"Cluster sizes             : {model.clusters_.value_counts().sort_index().to_dict()}")
    print(f"Kept loadings             : {sparsity.n_nonzero} of {sparsity.n_total}"
          f" (true support: {int(np.count_nonzero(TRUE_BETA))})")
    print(f"Loading RMSE vs truth     : {rmse_model:.3f} (ordinary least squares: {rmse_ols:.3f})")
    print(f"Residual sphericity       : {diagnostics.sphericity:.1f} against"
          f" {diagnostics.threshold:.1f}; passes: {diagnostics.passes}")
    print(f"Covariance relative error : factor model {error_model:.3f}, sample {error_sample:.3f}")
    print("Estimated loadings")
    print(model.coef_.round(2).to_string())
    print("Annualised volatility decomposition")
    print(vols.round(3).to_string())


if __name__ == "__main__":
    main()
