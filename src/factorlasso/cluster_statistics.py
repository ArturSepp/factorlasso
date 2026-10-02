"""Historical location of the cluster-stability statistics.

Compatibility module of the 0.23.0 layout. Every name is re-exported from its canonical
owner; new code imports from :mod:`factorlasso` or :mod:`factorlasso.cluster`. Assigning a
name here raises :class:`AttributeError` (see :mod:`factorlasso._compat`).
"""

# ruff: noqa: F401

from factorlasso._compat import guard_legacy_module
from factorlasso.cluster._smoothing import compute_co_association_panel
from factorlasso.cluster._stability import (
    _cluster_weight_panel, ClusterStabilityStatistics, compute_cluster_stability_statistics,
    _coverage_frame, _infer_partition_frequency, _normalise_frequency, _validate_span_map,
)

guard_legacy_module(__name__)
