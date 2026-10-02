"""Historical location of stability-pooled scoring.

Compatibility module of the 0.23.0 layout. Every name is re-exported from its canonical
owner; new code imports from :mod:`factorlasso` or :mod:`factorlasso.cluster`. Assigning a
name here raises :class:`AttributeError` (see :mod:`factorlasso._compat`).
"""

# ruff: noqa: F401

from factorlasso._compat import guard_legacy_module
from factorlasso.cluster._standardization import (
    _global_panel_score, _global_zscore, _last_on_or_before, _pooled_cluster_score,
    score_with_stability_pooled_clusters, _stability_row, StabilityPoolingType,
)

guard_legacy_module(__name__)
