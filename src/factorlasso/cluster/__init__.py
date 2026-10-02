"""Response clustering: dependence measures, hierarchical partitions and their stability.

Correlation-to-distance transforms and dendrogram cuts, Pearson/Spearman/Gerber dependence,
common-mode removal, causal rolling partitions with temporal smoothing, co-association
stability statistics and stability-pooled scoring. The names are also exported from
:mod:`factorlasso`.
"""

from factorlasso.cluster._dependence import (
    DependenceMeasure, compute_dependence_matrix, compute_gerber_matrix,
)
from factorlasso.cluster._hierarchical import (
    ClusterCorrelationTransform, ClusterCorrelationTransformResult, DistanceTransform,
    apply_cluster_correlation_transform, compute_clusters_from_corr_matrix,
    get_clusters_by_freq, get_cutoffs_by_freq, get_linkage_array, get_linkages_by_freq,
    remove_first_principal_component,
)
from factorlasso.cluster._smoothing import (
    ClusterSmootherType, RollingClusterData, apply_partition_distance_bonus,
    compute_co_association_panel, compute_rolling_smoothed_clusters, smooth_similarity_ewma,
)
from factorlasso.cluster._stability import (
    ClusterStabilityStatistics, compute_cluster_stability_statistics,
)
from factorlasso.cluster._standardization import (
    StabilityPoolingType, score_with_stability_pooled_clusters,
)

__all__ = [
    "ClusterCorrelationTransform",
    "ClusterCorrelationTransformResult",
    "ClusterSmootherType",
    "ClusterStabilityStatistics",
    "DependenceMeasure",
    "DistanceTransform",
    "RollingClusterData",
    "StabilityPoolingType",
    "apply_cluster_correlation_transform",
    "apply_partition_distance_bonus",
    "compute_cluster_stability_statistics",
    "compute_clusters_from_corr_matrix",
    "compute_co_association_panel",
    "compute_dependence_matrix",
    "compute_gerber_matrix",
    "compute_rolling_smoothed_clusters",
    "get_clusters_by_freq",
    "get_cutoffs_by_freq",
    "get_linkage_array",
    "get_linkages_by_freq",
    "remove_first_principal_component",
    "score_with_stability_pooled_clusters",
    "smooth_similarity_ewma",
]
