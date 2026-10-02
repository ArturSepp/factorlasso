"""Loading restrictions and prior analytics.

Univariate-derived sign constraints and adaptive penalty weights, OLS prior centres and
expert bounds, the metadata-to-factor mapping recipe for expert priors, and the optional
prior-inference and limiting-risk diagnostics. Hard signs and adaptive weights are loading
restrictions rather than statistical priors; they live here because they are derived and
combined together with the prior centres. The names are also exported from
:mod:`factorlasso`.
"""

from factorlasso.priors._bounds import compute_expert_prior_statistics
from factorlasso.priors._expert_map import ExpertPriorResolution, map_expert_factor_priors
from factorlasso.priors._inference import (
    Ar1PriorInterval, PriorHacGeometry, compute_ar1_prior_interval, compute_prior_hac_geometry,
    gaussian_prior_critical_value,
)
from factorlasso.priors._risk import (
    GaussianDominanceInformation, gaussian_dominance_information, two_factor_limit_risk,
    two_factor_minimax_radius,
)
from factorlasso.priors._signs import derive_sign_constraints, validate_cluster_signs

__all__ = [
    "Ar1PriorInterval",
    "ExpertPriorResolution",
    "GaussianDominanceInformation",
    "PriorHacGeometry",
    "compute_ar1_prior_interval",
    "compute_expert_prior_statistics",
    "compute_prior_hac_geometry",
    "derive_sign_constraints",
    "gaussian_dominance_information",
    "gaussian_prior_critical_value",
    "map_expert_factor_priors",
    "two_factor_limit_risk",
    "two_factor_minimax_radius",
    "validate_cluster_signs",
]
