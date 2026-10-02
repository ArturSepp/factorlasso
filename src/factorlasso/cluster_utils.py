"""Historical location of the correlation-to-distance, linkage and cut utilities.

Compatibility module of the 0.23.0 layout. Every name is re-exported from its canonical
owner; new code imports from :mod:`factorlasso` or :mod:`factorlasso.cluster`. Assigning a
name here raises :class:`AttributeError` (see :mod:`factorlasso._compat`).
"""

# ruff: noqa: F401

from factorlasso._compat import guard_legacy_module
from factorlasso.cluster._hierarchical import (
    apply_cluster_correlation_transform, ClusterCorrelationTransform,
    ClusterCorrelationTransformResult, compute_clusters_from_corr_matrix, _corr_to_distance,
    _CORRELATION_TOLERANCE, DEFAULT_CLUSTER_CORRELATION_TRANSFORM, DEFAULT_CUTOFF_FRACTION,
    DEFAULT_DISTANCE_TRANSFORM, DEFAULT_LINKAGE_METHOD, DistanceTransform, get_clusters_by_freq,
    get_cutoffs_by_freq, get_linkage_array, get_linkages_by_freq, remove_first_principal_component,
    _RESIDUAL_VARIANCE_FLOOR, VALID_LINKAGE_METHODS, _validated_clustering_correlation,
)

guard_legacy_module(__name__)
