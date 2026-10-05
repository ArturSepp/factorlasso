"""Bartlett aggregation of parameter influences on the original row grid.

Newey and West (1987), Econometrica 55, 703-708, doi:10.2307/1913610.
Weights, estimator derivatives and finite-sample corrections belong to callers.
This aggregation alone supplies no finite-sample or post-selection coverage.
"""
import numpy as np


def lag_covariance(influence, hac_lags, *, diagonal=False):
    """Aggregate finite parameter influences without allocating a grid-sized matrix."""
    if diagonal:
        value = np.sum(influence**2, axis=0)
    else:
        value = influence.T @ influence
    for lag in range(1, min(hac_lags, len(influence)-1) + 1):
        weight = 1 - lag/(hac_lags + 1)
        if diagonal:
            value += 2*weight*np.sum(influence[lag:]*influence[:-lag], axis=0)
        else:
            cross = influence[lag:].T @ influence[:-lag]
            value += weight*(cross + cross.T)
    return value
