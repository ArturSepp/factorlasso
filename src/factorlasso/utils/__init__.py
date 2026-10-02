"""Numerical utilities shared by estimation and clustering.

EWMA moments, group-loading matrices and the panel preparation that aligns factor and
response returns, masks missing observations and demeans them for the solvers. The names are
also exported from :mod:`factorlasso`.
"""

from factorlasso.utils._ewm import (
    compute_ewm, compute_ewm_covar, compute_expanding_power, set_group_loadings,
)
from factorlasso.utils._panel import get_x_y_np

__all__ = [
    "compute_ewm",
    "compute_ewm_covar",
    "compute_expanding_power",
    "get_x_y_np",
    "set_group_loadings",
]
