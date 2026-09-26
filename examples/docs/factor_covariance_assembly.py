"""
Worked example for ``docs/factor_covariance_assembly.md``: a covariance matrix from a factor model.
===================================================================================================

Forty synthetic monthly return series load on four correlated factors. The script fits the
loadings, stores them with the factor covariance and the residual variances in
:class:`factorlasso.CurrentFactorCovarData`, and assembles ``Sigma_y = beta Sigma_x beta' + D``.
It then repeats the estimation for histories of 48 to 240 months and compares the assembled matrix
with the sample covariance, by the error against the known population covariance and by the
realised volatility of the minimum-variance portfolio each matrix implies.

Every number quoted in the article is asserted here against a reference computed a different way:
the assembled matrix against a direct NumPy product, the volatility split against the matrix
diagonal, sub-universe assembly against the matching block, the rolling container against its
snapshots, and the stored cluster metadata against the SciPy linkage it came from.

Synthetic data, fixed seed, no network, no files written. Solver: CVXPY with CLARABEL.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster

import factorlasso as fl

SEED = 20260925
PERIODS_PER_YEAR = 12
FACTOR_NAMES = ["equity", "rates", "credit", "commodity"]
FACTOR_VOL = 0.04                  # per-month factor volatility, decimal returns
FACTOR_CORRELATION = 0.3           # equicorrelated factors
N_GROUPS = 4
GROUP_SIZE = 10
REG_LAMBDA = 1e-5
HISTORY_GRID = [48, 72, 120, 240]  # months of history used for estimation
N_PANELS = 16                      # redrawn panels per history length

# Asset i of group k loads on factor k (0.6 to 1.2) and on factor k+1 (0.0 to 0.5): 76 of 160 cells.
TRUE_BETA = np.zeros((N_GROUPS * GROUP_SIZE, len(FACTOR_NAMES)))
for group in range(N_GROUPS):
    members = slice(group * GROUP_SIZE, (group + 1) * GROUP_SIZE)
    TRUE_BETA[members, group] = np.linspace(0.6, 1.2, GROUP_SIZE)
    TRUE_BETA[members, (group + 1) % N_GROUPS] = np.linspace(0.0, 0.5, GROUP_SIZE)
RESIDUAL_VOL = np.tile(np.linspace(0.02, 0.04, GROUP_SIZE), N_GROUPS)    # per-month, per asset
ASSET_NAMES = [f"asset_{i + 1:02d}" for i in range(TRUE_BETA.shape[0])]


def factor_covariance() -> np.ndarray:
    """Per-month population covariance of the four factors."""
    corr = np.full((len(FACTOR_NAMES), len(FACTOR_NAMES)), FACTOR_CORRELATION)
    np.fill_diagonal(corr, 1.0)
    return FACTOR_VOL ** 2 * corr


def true_covariance() -> np.ndarray:
    """Annualised population covariance of the forty responses."""
    per_month = TRUE_BETA @ factor_covariance() @ TRUE_BETA.T + np.diag(RESIDUAL_VOL ** 2)
    return PERIODS_PER_YEAR * per_month


def make_panel(n_obs: int, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw ``n_obs`` months of the factor panel ``x`` (T x 4) and the responses ``y`` (T x 40)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-31", periods=n_obs, freq="ME")
    x = pd.DataFrame(
        rng.multivariate_normal(np.zeros(len(FACTOR_NAMES)), factor_covariance(), size=n_obs),
        index=dates,
        columns=FACTOR_NAMES,
    )
    noise = RESIDUAL_VOL * rng.standard_normal((n_obs, len(ASSET_NAMES)))
    y = pd.DataFrame(x.to_numpy() @ TRUE_BETA.T + noise, index=dates, columns=ASSET_NAMES)
    return x, y


def estimate_covar_data(
    x: pd.DataFrame,
    y: pd.DataFrame,
    model_type: fl.LassoModelType = fl.LassoModelType.LASSO,
) -> tuple[fl.LassoModel, fl.CurrentFactorCovarData]:
    """Fit the loadings and collect every input of ``Sigma_y`` in one snapshot, annualised."""
    model = fl.LassoModel(model_type=model_type, reg_lambda=REG_LAMBDA).fit(x=x, y=y)
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
    covar_data = fl.CurrentFactorCovarData(
        x_covar=PERIODS_PER_YEAR * x.cov(ddof=0),
        y_betas=model.coef_,
        y_variances=y_variances,
        estimation_date=x.index[-1],
        residuals=PERIODS_PER_YEAR * (y - model.predict(x)),       # annualised, like the alphas
    )
    return model, covar_data


def relative_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Frobenius norm of the error relative to the Frobenius norm of the truth."""
    return float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth))


def minimum_variance_vol(covariance: np.ndarray, truth: np.ndarray) -> float:
    """Annualised volatility, under ``truth``, of the fully invested minimum-variance portfolio."""
    ones = np.ones(covariance.shape[0])
    weights = np.linalg.solve(covariance, ones)
    weights = weights / weights.sum()
    return float(np.sqrt(weights @ truth @ weights))


def history_study() -> pd.DataFrame:
    """Mean error and minimum-variance volatility of both estimators for each history length."""
    truth = true_covariance()
    rows = []
    for n_obs in HISTORY_GRID:
        for draw in range(N_PANELS):
            x, y = make_panel(n_obs, seed=SEED + 1000 * n_obs + draw)
            _, covar_data = estimate_covar_data(x, y)
            estimates = {
                "factor model": covar_data.get_y_covar().to_numpy(),
                "sample": PERIODS_PER_YEAR * y.cov(ddof=0).to_numpy(),
            }
            for name, estimate in estimates.items():
                rows.append({
                    "n_obs": n_obs,
                    "estimator": name,
                    "relative_error": relative_error(estimate, truth),
                    "condition_number": float(np.linalg.cond(estimate)),
                    "min_variance_vol": minimum_variance_vol(estimate, truth),
                })
    return pd.DataFrame(rows).groupby(["estimator", "n_obs"]).mean()


def main() -> None:
    """Run the example, verify each quoted number independently, and print the summary."""
    x, y = make_panel(n_obs=120)
    model, covar_data = estimate_covar_data(x, y)
    beta = model.coef_.to_numpy()

    # --- 1. Assembly against a direct matrix product ---
    sigma_y = covar_data.get_y_covar()
    residual_vars = covar_data.y_variances[fl.VarianceColumns.RESIDUAL_VARS.value].to_numpy()
    direct = beta @ covar_data.x_covar.to_numpy() @ beta.T + np.diag(residual_vars)
    assert np.allclose(sigma_y.to_numpy(), direct)
    assert sigma_y.equals(covar_data.y_covar)                           # the property is shorthand
    assert np.allclose(residual_vars, PERIODS_PER_YEAR * (y - model.predict(x)).var(ddof=0))
    assert np.linalg.eigvalsh(sigma_y.to_numpy()).min() > 0.0

    # the residual block scales with residual_var_weight; zero leaves the systematic part
    systematic = covar_data.get_y_covar(residual_var_weight=0.0)
    assert np.allclose(systematic.to_numpy(), beta @ covar_data.x_covar.to_numpy() @ beta.T)
    assert np.linalg.matrix_rank(systematic.to_numpy()) == len(FACTOR_NAMES)
    assert np.allclose(covar_data.get_residual_covar().to_numpy(), np.diag(residual_vars))
    orthogonal = covar_data.get_y_covar(residual_type=fl.ResidualType.ORTHOGONAL)
    assert orthogonal.equals(sigma_y)                                   # the default residual type

    # volatility split and the snapshot table
    vols = covar_data.get_model_vols()
    assert np.allclose(vols["total_vol"] ** 2, np.diag(sigma_y.to_numpy()))
    assert np.allclose(vols["total_vol"] ** 2, vols["sys_vol"] ** 2 + vols["resid_vol"] ** 2)
    snapshot = covar_data.get_snapshot()
    assert list(snapshot.columns[:4]) == FACTOR_NAMES and snapshot.shape[0] == len(ASSET_NAMES)
    # stat_alpha is the last value of an EWMA of the stored residuals, in their units
    ewma_alpha = fl.compute_ewm(covar_data.residuals, span=120).iloc[-1]
    assert np.allclose(covar_data.estimate_alpha(alpha_span=120), ewma_alpha)
    assert np.allclose(snapshot["stat_alpha"], ewma_alpha)

    # a sub-universe is the matching block, by argument or by filtering the snapshot
    subset = ASSET_NAMES[:3] + ASSET_NAMES[-2:]
    block = sigma_y.loc[subset, subset]
    assert np.allclose(covar_data.get_y_covar(assets=subset), block)
    assert np.allclose(covar_data.filter_on_tickers(subset).get_y_covar(), block)

    # --- 2. A rolling container returns the snapshot that was available at each date ---
    rolling = fl.RollingFactorCovarData()
    for n_obs in (96, 120):
        _, snapshot_data = estimate_covar_data(x.iloc[:n_obs], y.iloc[:n_obs])
        rolling.add(snapshot_data.estimation_date, snapshot_data)
    assert len(rolling) == 2 and rolling.get_latest().estimation_date == x.index[-1]
    assert np.allclose(rolling.get_y_covars()[x.index[-1]], sigma_y)
    between = pd.DatetimeIndex([x.index[100]])
    held = rolling.get_y_covars(dates=between)[between[0]]
    assert np.allclose(held, rolling[x.index[95]].get_y_covar())        # no look-ahead
    assert rolling.get_residual_vars().shape == (2, len(ASSET_NAMES))

    # --- 3. Cluster metadata travels with the snapshot and returns as SciPy objects ---
    hcgl_type = fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO
    hcgl, _ = estimate_covar_data(x, y, model_type=hcgl_type)
    linkages = pd.DataFrame(
        hcgl.linkage_,
        columns=["left", "right", "distance", "n_samples"],
        index=[f"ME:step_{i}" for i in range(hcgl.linkage_.shape[0])],
    )
    with_clusters = fl.CurrentFactorCovarData(
        x_covar=covar_data.x_covar,
        y_betas=hcgl.coef_,
        y_variances=covar_data.y_variances,
        clusters="ME:" + hcgl.clusters_.astype(str),
        linkages=linkages,
        cutoffs=pd.Series({"ME": hcgl.cutoff_}),
    )
    assert "cluster" in with_clusters.y_variances.columns                # mirrored for persistence
    clusters_by_freq = fl.get_clusters_by_freq(with_clusters.clusters)
    assert clusters_by_freq["ME"].astype(int).equals(hcgl.clusters_.astype(int))
    linkage = fl.get_linkage_array(with_clusters.linkages, "ME")
    assert np.array_equal(linkage, fl.get_linkages_by_freq(with_clusters.linkages)["ME"])
    cutoff = fl.get_cutoffs_by_freq(with_clusters.cutoffs)["ME"]
    recut = fcluster(linkage, t=cutoff, criterion="distance")
    labels = hcgl.clusters_.to_numpy()
    assert np.array_equal(recut[:, None] == recut[None, :], labels[:, None] == labels[None, :])

    # --- 4. Why assemble: error and conditioning against the sample covariance ---
    truth = true_covariance()
    study = history_study()
    model_rows, sample_rows = study.loc["factor model"], study.loc["sample"]
    assert (model_rows["relative_error"] < sample_rows["relative_error"]).all()
    assert (model_rows["condition_number"] < sample_rows["condition_number"]).all()
    assert (model_rows["min_variance_vol"] < sample_rows["min_variance_vol"]).all()
    true_min_vol = minimum_variance_vol(truth, truth)
    assert (model_rows["min_variance_vol"] > true_min_vol).all()
    assert sample_rows.loc[48, "min_variance_vol"] > 2.0 * true_min_vol

    print("One panel of 120 months")
    print(f"  relative error      : factor model {relative_error(sigma_y.to_numpy(), truth):.3f},"
          f" sample {relative_error(PERIODS_PER_YEAR * y.cov(ddof=0).to_numpy(), truth):.3f}")
    print(f"  smallest eigenvalue : {np.linalg.eigvalsh(sigma_y.to_numpy()).min():.4f}")
    print(vols.head(4).round(3).to_string())
    print(f"Means over {N_PANELS} panels per history length;"
          f" true minimum-variance volatility {true_min_vol:.4f}")
    print(study.round(4).to_string())


if __name__ == "__main__":
    main()
