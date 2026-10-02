"""Historical location of the prior-risk calculations.

Compatibility module of the 0.23.0 layout. Every name is re-exported from its canonical
owner; new code imports from :mod:`factorlasso` or :mod:`factorlasso.priors`. Assigning a
name here raises :class:`AttributeError` (see :mod:`factorlasso._compat`).
"""

# ruff: noqa: F401

from factorlasso._compat import guard_legacy_module
from factorlasso.priors._inference import _symmetric
from factorlasso.priors._risk import (
    gaussian_dominance_information, GaussianDominanceInformation, _nonnegative,
    two_factor_limit_risk, two_factor_minimax_radius,
)

guard_legacy_module(__name__)
