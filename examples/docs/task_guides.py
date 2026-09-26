"""Canonical example for docs/task-guides.md: short recipes, each checked.

The recipes use only the public API and small synthetic panels. They check structural and
algebraic contracts rather than solver-sensitive decimals: sign and prior inputs reach the
solver, automatic prior centres are recorded, a discovered partition has the requested size,
rolling partitions ignore later data, and the covariance container assembles
B Sigma_F B' + D. The articles linked from the page explain each step in depth.
"""

import numpy as np
import pandas as pd

import factorlasso as fl


def constrained_fit() -> tuple[fl.LassoModel, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hard signs and a prior centre on three responses, one with a late start."""
    rng = np.random.default_rng(12)
    index = pd.date_range("2020-01-31", periods=80, freq="ME")
    x = pd.DataFrame(rng.normal(size=(80, 2)), index=index, columns=["growth", "rates"])
    beta = np.array([[0.7, -0.2], [-0.4, 0.0], [0.1, 0.6]])
    y = pd.DataFrame(x.to_numpy() @ beta.T + 0.04 * rng.normal(size=(80, 3)),
                     index=index, columns=["asset_a", "asset_b", "asset_c"])
    y.loc[index[:8], "asset_c"] = np.nan
    signs = pd.DataFrame([[1.0, np.nan], [-1.0, 0.0], [np.nan, 1.0]],
                         index=y.columns, columns=x.columns)
    prior = pd.DataFrame(0.0, index=y.columns, columns=x.columns)
    prior.loc["asset_a", "growth"] = 0.5
    model = fl.LassoModel(
        reg_lambda=1e-3,
        span=20,
        warmup_period=12,
        factors_beta_loading_signs=signs,
        factors_beta_prior=prior,
    ).fit(x=x, y=y)
    return model, x, y, signs


def automatic_prior_fit(x: pd.DataFrame, y: pd.DataFrame, signs: pd.DataFrame) -> fl.LassoModel:
    """OLS prior centres from the highest-R-squared factor, at the loss span of the fit."""
    return fl.LassoModel(
        span=20, apply_ols_prior=True, prior_selection_type="highest_r2",
        factors_beta_loading_signs=signs,
    ).fit(x=x, y=y, span=16)


def two_cluster_fit() -> fl.LassoModel:
    """HCGL asked to discover at most two groups of four responses."""
    rng = np.random.default_rng(23)
    x = pd.DataFrame(rng.normal(size=(90, 2)), columns=["market", "style"])
    common_1, common_2 = rng.normal(size=90), rng.normal(size=90)
    y = pd.DataFrame({"a": common_1 + 0.15 * rng.normal(size=90),
                      "b": common_1 + 0.15 * rng.normal(size=90),
                      "c": common_2 + 0.15 * rng.normal(size=90),
                      "d": common_2 + 0.15 * rng.normal(size=90)})
    return fl.LassoModel(
        model_type=fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
        n_clusters=2,
        reg_lambda=1e-3,
        warmup_period=12,
    ).fit(x=x, y=y)


def rolling_partitions_ignore_the_future() -> bool:
    """Rolling partitions at two dates, before and after every later observation changes."""
    rng = np.random.default_rng(31)
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    base = rng.normal(size=(80, 2))
    y = pd.DataFrame({"a": base[:, 0] + 0.1 * rng.normal(size=80),
                      "b": base[:, 0] + 0.1 * rng.normal(size=80),
                      "c": base[:, 1] + 0.1 * rng.normal(size=80),
                      "d": base[:, 1] + 0.1 * rng.normal(size=80)}, index=dates)
    estimation_dates = [dates[39], dates[59]]
    spec = fl.LassoModel(
        model_type=fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
        n_clusters=2,
        warmup_period=12,
        cluster_smoother_type=fl.ClusterSmootherType.SIMILARITY_EWMA,
    )
    before = fl.compute_rolling_smoothed_clusters(y, estimation_dates, spec)
    changed = y.copy()
    changed.loc[dates[60]:, :] = 100.0
    after = fl.compute_rolling_smoothed_clusters(changed, estimation_dates, spec)
    return all(before.clusters[date].equals(after.clusters[date]) for date in estimation_dates)


def assembled_covariance() -> tuple[pd.DataFrame, np.ndarray]:
    """B Sigma_F B' + D from a snapshot, and the same matrix by NumPy."""
    factor_covariance = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]],
                                     index=["growth", "rates"], columns=["growth", "rates"])
    loadings = pd.DataFrame([[1.0, 0.2], [0.4, -0.5]],
                            index=["asset_a", "asset_b"], columns=factor_covariance.columns)
    residual_variance = pd.Series([0.02, 0.03], index=loadings.index)
    snapshot = fl.CurrentFactorCovarData(
        x_covar=factor_covariance,
        y_betas=loadings,
        y_variances=pd.DataFrame({fl.VarianceColumns.RESIDUAL_VARS.value: residual_variance}),
    )
    independent = (loadings.to_numpy() @ factor_covariance.to_numpy() @ loadings.to_numpy().T
                   + np.diag(residual_variance.to_numpy()))
    return snapshot.get_y_covar(), independent


def main() -> None:
    model, x, y, signs = constrained_fit()
    coef = model.coef_
    assert coef.shape == (3, 2) and model.valid_mask_.shape == (79, 3)
    assert coef.loc["asset_a", "growth"] >= -1e-6 and coef.loc["asset_b", "growth"] <= 1e-6
    assert abs(coef.loc["asset_b", "rates"]) <= 1e-6 and coef.loc["asset_c", "rates"] >= -1e-6
    assert model.derived_signs_.equals(signs)

    automatic = automatic_prior_fit(x, y, signs)
    assert automatic.ols_prior_span_ == automatic.effective_span_ == 16
    assert automatic.ols_betas_.shape == automatic.coef_.shape
    assert automatic.effective_beta_prior_.loc["asset_b", "rates"] == 0.0

    hcgl = two_cluster_fit()
    assert len(hcgl.clusters_) == 4 and hcgl.clusters_.nunique() <= 2
    assert hcgl.linkage_.shape == (3, 4)

    assert rolling_partitions_ignore_the_future()

    assembled, independent = assembled_covariance()
    assert assembled.shape == (2, 2) and np.allclose(assembled, independent)
    assert np.allclose(assembled, assembled.T)


if __name__ == "__main__":
    main()
