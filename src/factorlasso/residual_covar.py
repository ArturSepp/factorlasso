"""Historical location of the prepared residual correlation.

Compatibility module of the 0.23.0 layout. Every name is re-exported from its canonical
owner; new code imports from :mod:`factorlasso` or :mod:`factorlasso.covariance`. Assigning a
name here raises :class:`AttributeError` (see :mod:`factorlasso._compat`).
"""

# ruff: noqa: F401

from factorlasso._compat import guard_legacy_module
from factorlasso.covariance._residual_correlation import (
    _aggregate_residuals, _compatible_boundary, estimate_residual_correlation,
    _moment_to_correlation, _prepare_common_residuals, ResidualCorrelationData,
)
from factorlasso.utils._ewm import compute_ewm, compute_ewm_covar, _validate_span

guard_legacy_module(__name__)
