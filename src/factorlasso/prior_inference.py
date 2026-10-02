"""Historical location of the prior-inference calculations.

Compatibility module of the 0.23.0 layout. Every name is re-exported from its canonical
owner; new code imports from :mod:`factorlasso` or :mod:`factorlasso.priors`. Assigning a
name here raises :class:`AttributeError` (see :mod:`factorlasso._compat`).
"""

# ruff: noqa: F401

from factorlasso._compat import guard_legacy_module
from factorlasso.priors._inference import (
    _angular_log_density, Ar1PriorInterval, _ar_covariance, compute_ar1_prior_interval,
    compute_prior_hac_geometry, _gaussian_critical, gaussian_prior_critical_value, _pivot_tail,
    PriorHacGeometry, _probability, _responses, _symmetric,
)

guard_legacy_module(__name__)
