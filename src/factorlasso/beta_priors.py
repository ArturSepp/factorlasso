"""Historical location of the OLS prior centres.

Compatibility module of the 0.23.0 layout. Every name is re-exported from its canonical
owner; new code imports from :mod:`factorlasso` or :mod:`factorlasso.priors`. Assigning a
name here raises :class:`AttributeError` (see :mod:`factorlasso._compat`).
"""

# ruff: noqa: F401

from factorlasso._compat import guard_legacy_module
from factorlasso.priors._ols import (
    _centred_finite_values, _compute_joint_ols_prior, _compute_ols_prior,
    _validate_prior_selection_type, _zero_incompatible_priors,
)
from factorlasso.utils._ewm import compute_expanding_power, _validate_span

guard_legacy_module(__name__)
