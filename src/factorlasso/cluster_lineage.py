"""Historical location of offline cluster lineage.

Compatibility module of the 0.23.0 layout. Every name is re-exported from its canonical
owner; new code imports from :mod:`factorlasso` or :mod:`factorlasso.diagnostics`. Assigning a
name here raises :class:`AttributeError` (see :mod:`factorlasso._compat`).
"""

# ruff: noqa: F401

from factorlasso._compat import guard_legacy_module
from factorlasso.covariance._factor_covar import (
    CurrentFactorCovarData, RollingFactorCovarData, VarianceColumns,
)
from factorlasso.diagnostics._lineage import (
    analyze_cluster_lineage, _build_tracks, _classify, _cluster_series, _Fingerprint, _match_panel,
    _match_panel_mcf, _overlap, _psd_clip, _qualifies, RiskClusterReport,
    run_cluster_lineage_report, _snapshot_fingerprints, solve_max_weight_matching, TaxonomyConfig,
    TrackPanel, _validate_edges,
)

guard_legacy_module(__name__)
