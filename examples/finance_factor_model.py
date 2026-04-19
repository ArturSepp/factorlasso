"""
Example: Multi-asset factor model with sign-constrained LASSO
=============================================================

Demonstrates the full factorlasso pipeline on a synthetic multi-asset
portfolio problem:

1. Estimate sparse factor loadings with sign constraints
2. Assemble the factor covariance decomposition Σ_y = β Σ_x β' + D
3. Inspect diagnostics (R², volatilities, betas)

This mirrors the methodology from:
    Sepp, Ossa, Kastenholz (2026), "Robust Optimization of Strategic and
    Tactical Asset Allocation for Multi-Asset Portfolios", JPM 52(4).
"""

import numpy as np
import pandas as pd

from factorlasso import (
    CurrentFactorCovarData,
    LassoModel,
    LassoModelType,
    VarianceColumns,
)
from factorlasso.ewm_utils import compute_ewm_covar
from factorlasso.lasso_estimator import get_x_y_np


def main():
    # --- 1. Generate synthetic factor returns and asset returns ---
    np.random.seed(2026)
    T = 260  # ~5 years of weekly data
    factor_names = ['Equity', 'Rates', 'Credit', 'Commodity']
    asset_names = ['US_Eq', 'EU_Eq', 'EM_Eq', 'US_Govt', 'EU_Govt',
                   'IG_Credit', 'HY_Credit', 'Gold', 'Oil']
    M = len(factor_names)
    N = len(asset_names)

    # True (sparse) factor loadings
    beta_true = np.array([
        # Equity  Rates  Credit  Commodity
        [1.0,    0.0,   0.0,    0.0],    # US_Eq
        [0.9,    0.0,   0.0,    0.0],    # EU_Eq
        [1.2,    0.0,   0.1,    0.0],    # EM_Eq
        [0.0,    1.0,   0.0,    0.0],    # US_Govt
        [0.0,    0.8,   0.0,    0.0],    # EU_Govt
        [0.0,    0.3,   0.8,    0.0],    # IG_Credit
        [0.2,    0.0,   1.0,    0.0],    # HY_Credit
        [0.0,    0.0,   0.0,    0.7],    # Gold
        [0.0,    0.0,   0.0,    1.0],    # Oil
    ])

    X = pd.DataFrame(0.02 * np.random.randn(T, M), columns=factor_names)
    noise = 0.005 * np.random.randn(T, N)
    Y = pd.DataFrame(X.values @ beta_true.T + noise, columns=asset_names)

    # --- 2. Sign constraints ---
    # Equity assets must have non-negative equity beta;
    # Government bonds must have non-negative rates beta;
    # Gold/Oil must have non-negative commodity beta.
    signs = pd.DataFrame(np.nan, index=asset_names, columns=factor_names)
    signs.loc[['US_Eq', 'EU_Eq', 'EM_Eq'], 'Equity'] = 1       # non-negative
    signs.loc[['US_Govt', 'EU_Govt'], 'Rates'] = 1              # non-negative
    signs.loc[['US_Govt', 'EU_Govt'], 'Equity'] = 0             # zero
    signs.loc[['Gold', 'Oil'], 'Commodity'] = 1                 # non-negative

    # --- 3. Fit HCGL model ---
    model = LassoModel(
        model_type=LassoModelType.GROUP_LASSO_CLUSTERS,
        reg_lambda=1e-5,
        span=52,
        factors_beta_loading_signs=signs,
    )
    model.fit(x=X, y=Y)

    print("=== Estimated betas (N × M) ===")
    print(model.coef_.round(3))
    print()

    print("=== Clusters ===")
    print(model.clusters_)
    print()

    # --- 4. Assemble covariance decomposition ---
    # Compute factor covariance from raw demeaned returns (no model fit needed)
    x_dm, _, _ = get_x_y_np(x=X, y=Y, span=52)
    x_covar_np = compute_ewm_covar(a=x_dm, span=52)
    # Annualise (52 weekly observations per year)
    x_covar = pd.DataFrame(52.0 * x_covar_np, index=factor_names, columns=factor_names)

    result = model.estimation_result_
    y_variances = pd.DataFrame({
        VarianceColumns.EWMA_VARIANCE: result.ss_total * 52,
        VarianceColumns.RESIDUAL_VARS: result.ss_res * 52,
        VarianceColumns.INSAMPLE_ALPHA: result.alpha * 52,
        VarianceColumns.R2: result.r2,
    }, index=asset_names)

    covar_data = CurrentFactorCovarData(
        x_covar=x_covar,
        y_betas=model.coef_,
        y_variances=y_variances,
    )

    sigma_y = covar_data.get_y_covar()
    print("=== Asset covariance matrix (annualised) ===")
    print(sigma_y.round(4))
    print()

    vols = covar_data.get_model_vols()
    print("=== Volatility decomposition ===")
    print(vols.round(4))
    print()

    print("=== R² per asset ===")
    print(y_variances[VarianceColumns.R2].round(3))


if __name__ == '__main__':
    main()
