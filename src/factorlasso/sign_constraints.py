"""Historical location of sign constraints and adaptive penalty weights.

Compatibility module of the 0.23.0 layout. Every name is re-exported from its canonical
owner; new code imports from :mod:`factorlasso` or :mod:`factorlasso.priors`. Assigning a
name here raises :class:`AttributeError` (see :mod:`factorlasso._compat`).
"""

# ruff: noqa: F401

from factorlasso._compat import guard_legacy_module
from factorlasso.priors._signs import (
    _adaptive_penalty_weights, _aggregate_to_block_weights, _aggregate_to_row_weights,
    _compute_sign_matrix_per_response, _compute_sign_vector, derive_sign_constraints,
    _pooled_sign_statistics, _sign_observation_weights, validate_cluster_signs,
)

guard_legacy_module(__name__)
