"""Weighted regression uncertainty and fixed-model Gaussian interval calibration."""

from factorlasso.inference._ar1 import Ar1Interval, compute_ar1_interval
from factorlasso.inference._gaussian import gaussian_critical_value
from factorlasso.inference._geometry import LinearHacGeometry, compute_wls_hac_geometry
from factorlasso.inference._wls import WlsHacStatistics, compute_wls_hac_statistics
from factorlasso.inference._mean import (
    compute_weighted_mean_hac_geometry, weighted_mean_hac_expectation,
)
from factorlasso.inference._quadratic import (
    gaussian_quadratic_quantile, quadratic_confidence_summary,
)
from factorlasso.inference._intervals import linear_confidence_intervals
from factorlasso.inference._resampling import bootstrap_weighted_means

__all__ = [
    "weighted_mean_hac_expectation",
    "compute_weighted_mean_hac_geometry", "gaussian_quadratic_quantile",
    "quadratic_confidence_summary", "linear_confidence_intervals", "bootstrap_weighted_means",
    "WlsHacStatistics", "compute_wls_hac_statistics", "LinearHacGeometry",
    "compute_wls_hac_geometry", "gaussian_critical_value", "Ar1Interval",
    "compute_ar1_interval",
]
