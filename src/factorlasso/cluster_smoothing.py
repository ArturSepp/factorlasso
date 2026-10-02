"""Historical location of causal rolling-cluster smoothing.

Compatibility module of the 0.23.0 layout. Every name is re-exported from its canonical
owner; new code imports from :mod:`factorlasso` or :mod:`factorlasso.cluster`. Assigning a
name here raises :class:`AttributeError` (see :mod:`factorlasso._compat`).
"""

# ruff: noqa: F401

from factorlasso._compat import guard_legacy_module
from factorlasso.cluster._dependence import compute_dependence_matrix, DependenceMeasure
from factorlasso.cluster._hierarchical import (
    apply_cluster_correlation_transform, ClusterCorrelationTransform,
    compute_clusters_from_corr_matrix, _corr_to_distance,
)
from factorlasso.cluster._smoothing import (
    apply_partition_distance_bonus, _cluster_distance_matrix, ClusterSmootherType,
    _co_association_panel, compute_co_association_panel, compute_rolling_smoothed_clusters,
    _correlation_input, _effective_cluster_correlation_span, _ewma_co_association_panel,
    _is_recluster_date, _iter_correlation_inputs, _join_entrants, RollingClusterData,
    smooth_similarity_ewma,
)

guard_legacy_module(__name__)
