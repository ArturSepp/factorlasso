"""Weighted regression uncertainty and fixed-model Gaussian interval calibration."""

from factorlasso.inference._pooling import HierarchicalMeanPosterior, pool_gaussian_means
from factorlasso.inference._pooled_signs import PooledSignGateResult, compute_pooled_sign_gate
from factorlasso.inference._ragged_signs import (
    RaggedPooledSignGateResult, compute_ragged_pooled_sign_gate,
)
from factorlasso.inference._conditional_signs import (
    ConditionalGaussianGeometry, GaussianSignGateRisk,
    compute_conditional_gaussian_geometry, gaussian_sign_gate_risk,
)

from factorlasso.inference._ar1 import Ar1Interval, compute_ar1_interval
from factorlasso.inference._gaussian import gaussian_critical_value
from factorlasso.inference._geometry import LinearHacGeometry, compute_wls_hac_geometry
from factorlasso.inference._wls import WlsHacStatistics, compute_wls_hac_statistics
from factorlasso.inference._joint_hac import (
    aggregate_influence_calendar, compute_hac_covariance, delta_method_hac_intervals,
)
from factorlasso.inference._mean import (
    compute_weighted_mean_hac_geometry, weighted_mean_hac_expectation,
)
from factorlasso.inference._quadratic import (
    gaussian_quadratic_quantile, quadratic_confidence_summary,
)
from factorlasso.inference._intervals import linear_confidence_intervals
from factorlasso.inference._resampling import bootstrap_weighted_means
from factorlasso.inference._covariance_moments import calibrate_covariance_moments
from factorlasso.inference._joint_regression import joint_wls_gaussian_region
from factorlasso.inference._scalar_capacity import (
    quadratic_scalar_confidence_summary, positive_part_confidence_summary,
)

__all__ = [
    "PooledSignGateResult", "compute_pooled_sign_gate",
    "RaggedPooledSignGateResult", "compute_ragged_pooled_sign_gate",
    "ConditionalGaussianGeometry", "GaussianSignGateRisk",
    "compute_conditional_gaussian_geometry", "gaussian_sign_gate_risk",
    "aggregate_influence_calendar", "compute_hac_covariance", "delta_method_hac_intervals",
    "calibrate_covariance_moments",
    "HierarchicalMeanPosterior", "pool_gaussian_means",
    "quadratic_scalar_confidence_summary", "positive_part_confidence_summary",
    "joint_wls_gaussian_region",
    "weighted_mean_hac_expectation",
    "compute_weighted_mean_hac_geometry", "gaussian_quadratic_quantile",
    "quadratic_confidence_summary", "linear_confidence_intervals", "bootstrap_weighted_means",
    "WlsHacStatistics", "compute_wls_hac_statistics", "LinearHacGeometry",
    "compute_wls_hac_geometry", "gaussian_critical_value", "Ar1Interval",
    "compute_ar1_interval",
]
