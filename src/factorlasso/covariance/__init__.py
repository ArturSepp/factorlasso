"""Factor covariance assembly.

Current and rolling containers that assemble ``Sigma_y = B Sigma_x B' + D`` from fitted
loadings, factor covariance and residual variances, with orthogonal, empirical and
cluster-structured residual blocks, and the prepared common-period residual correlation.
The names are also exported from :mod:`factorlasso`.
"""

from factorlasso.covariance._factor_covar import (
    CurrentFactorCovarData, ResidualType, RollingFactorCovarData, VarianceColumns,
)
from factorlasso.covariance._residual_correlation import (
    ResidualCorrelationData, estimate_residual_correlation,
)
from factorlasso.covariance._alpha_uncertainty import (
    AlphaUncertainty, estimate_alpha_uncertainty, gaussian_quadratic_summary,
    calibrate_alpha_uncertainty,
    sample_gaussian_estimates,
)

__all__ = [
    "calibrate_alpha_uncertainty",
    "AlphaUncertainty", "estimate_alpha_uncertainty", "gaussian_quadratic_summary",
    "sample_gaussian_estimates",
    "CurrentFactorCovarData",
    "ResidualCorrelationData",
    "ResidualType",
    "RollingFactorCovarData",
    "VarianceColumns",
    "estimate_residual_correlation",
]
