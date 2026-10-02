"""Historical location of the dependence measures for clustering.

Compatibility module of the 0.23.0 layout. Every name is re-exported from its canonical
owner; new code imports from :mod:`factorlasso` or :mod:`factorlasso.cluster`. Assigning a
name here raises :class:`AttributeError` (see :mod:`factorlasso._compat`).
"""

# ruff: noqa: F401

from factorlasso._compat import guard_legacy_module
from factorlasso.cluster._dependence import (
    compute_dependence_matrix, compute_gerber_matrix, DEFAULT_DEPENDENCE_MEASURE,
    DEFAULT_GERBER_THRESHOLD, DependenceMeasure, _normalised_ewm_weights,
)
from factorlasso.utils._ewm import compute_ewm_covar, NanBackfill

guard_legacy_module(__name__)
