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

__all__ = [
    "CurrentFactorCovarData",
    "ResidualCorrelationData",
    "ResidualType",
    "RollingFactorCovarData",
    "VarianceColumns",
    "estimate_residual_correlation",
]
