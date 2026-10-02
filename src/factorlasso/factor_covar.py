"""Historical location of the factor covariance containers.

Compatibility module of the 0.23.0 layout. Every name is re-exported from its canonical
owner; new code imports from :mod:`factorlasso` or :mod:`factorlasso.covariance`. Assigning a
name here raises :class:`AttributeError` (see :mod:`factorlasso._compat`).
"""

# ruff: noqa: F401

from factorlasso._compat import guard_legacy_module
from factorlasso.cluster._hierarchical import compute_clusters_from_corr_matrix
from factorlasso.covariance._factor_covar import (
    CurrentFactorCovarData, ResidualType, RollingFactorCovarData, _validate_residual_options,
    VarianceColumns,
)
from factorlasso.covariance._residual_correlation import ResidualCorrelationData
from factorlasso.utils._ewm import compute_ewm

guard_legacy_module(__name__)
