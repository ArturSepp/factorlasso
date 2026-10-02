"""Historical location of the EWMA and group-loading utilities.

Compatibility module of the 0.23.0 layout. Every name is re-exported from its canonical
owner; new code imports from :mod:`factorlasso` or :mod:`factorlasso.utils`. Assigning a
name here raises :class:`AttributeError` (see :mod:`factorlasso._compat`).
"""

# ruff: noqa: F401

from factorlasso._compat import guard_legacy_module
from factorlasso.utils._ewm import (
    compute_ewm, compute_ewm_covar, compute_expanding_power, ewm_recursion, InitType, NanBackfill,
    set_group_loadings, set_init_dim1, _to_np, _validate_ewm_lambda, _validate_span,
)

guard_legacy_module(__name__)
