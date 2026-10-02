"""Historical location of the expert prior bounds.

Compatibility module of the 0.23.0 layout. Every name is re-exported from its canonical
owner; new code imports from :mod:`factorlasso` or :mod:`factorlasso.priors`. Assigning a
name here raises :class:`AttributeError` (see :mod:`factorlasso._compat`).
"""

# ruff: noqa: F401

from factorlasso._compat import guard_legacy_module
from factorlasso.priors._bounds import (
    _compute_expert_prior_bounds, compute_expert_prior_statistics, _validate_expert_bound_settings,
    _validate_hac_lags,
)
from factorlasso.utils._ewm import _validate_span

guard_legacy_module(__name__)
