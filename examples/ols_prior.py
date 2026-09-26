"""Automatic OLS priors with the loss span and forced-zero PE exposures.

Synthetic monthly log returns; span 36 observations, warmup 12, no annualisation.
The prior is per response, with an intercept, on original return observations.
An independent weighted-design least-squares fit checks the marginal slopes.
"""
import numpy as np
import pandas as pd

from factorlasso import LassoModel, LassoModelType


def main():
    """Fit a clustered model and inspect selected versus admissible priors."""
    rng = np.random.default_rng(290)
    index = pd.date_range("2016-01-31", periods=100, freq="ME")
    x = pd.DataFrame(.02 * rng.normal(size=(100, 3)), index=index,
                     columns=["Rates", "Credit", "Private Equity"])
    y = pd.DataFrame({"Bond fund": .2 * x.Rates + 1.2 * x.Credit,
                      "PE-like public fund": 1.8 * x["Private Equity"]}, index=index)
    y.iloc[:10, 0] = np.nan
    signs = pd.DataFrame(1., index=y.columns, columns=x.columns)
    signs["Private Equity"] = 0.
    model = LassoModel(
        model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
        apply_ols_prior=True, prior_selection_type="highest_r2",
        span=36, cluster_correlation_span=60,
        warmup_period=12, reg_lambda=1e-5, solver="CLARABEL",
        factors_beta_loading_signs=signs,
    ).fit(x, y)

    weights = (1. - 2. / 37.) ** np.arange(len(x) - 1, -1, -1)
    for asset in y:
        for factor in x:
            valid = x[factor].notna() & y[asset].notna()
            design = np.column_stack([np.ones(valid.sum()), x.loc[valid, factor]])
            root_weight = np.sqrt(weights[valid])
            reference = np.linalg.lstsq(
                design * root_weight[:, None], y.loc[valid, asset] * root_weight,
                rcond=None,
            )[0][1]
            np.testing.assert_allclose(model.ols_betas_.loc[asset, factor],
                                       reference, atol=1e-12, rtol=0)
    for asset in y:
        winner = model.ols_r2_.loc[asset].idxmax()
        expected = pd.Series(0., index=x.columns)
        expected[winner] = model.ols_betas_.loc[asset, winner]
        np.testing.assert_allclose(model.ols_beta_prior_.loc[asset], expected, atol=1e-12)
    assert model.ols_prior_span_ == model.effective_span_ == 36
    assert model.effective_cluster_correlation_span_ == 60
    np.testing.assert_allclose(
        model.ols_beta_prior_.loc["PE-like public fund", "Private Equity"], 1.8,
        atol=1e-12, rtol=0,
    )
    assert (model.effective_beta_prior_["Private Equity"] == 0.).all()
    assert (model.effective_beta_prior_.loc["PE-like public fund"] == 0.).all()
    print("Marginal OLS checked against weighted least squares")
    print("Prior uses loss span 36; clustering uses span 60")
    print("PE prior removed for ineligible assets without reallocating votes")
    print(model.effective_beta_prior_.round(3).to_string())


if __name__ == "__main__":
    main()
