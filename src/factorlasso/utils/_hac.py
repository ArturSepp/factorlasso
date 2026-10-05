"""Shared Bartlett calendar kernel for weighted-score HAC calculations.

Newey and West (1987), Econometrica 55, 703-708, doi:10.2307/1913610.
Explicit irregular-calendar distances extend the kernel construction; they do
not confer finite-sample or post-selection coverage on weighted estimates.
"""
import numpy as np


def bartlett_kernel(calendar, bandwidth):
    """Return the triangular PSD kernel with zero weight at the stated distance."""
    times = np.asarray(calendar, dtype=float)
    if (times.ndim != 1 or len(times) == 0 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError('calendar must be a finite strictly increasing vector')
    if isinstance(bandwidth, (bool, np.bool_)) or not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('bandwidth must be finite and positive in calendar units')
    return np.maximum(0., 1-np.abs(times[:, None]-times[None, :])/bandwidth)


def score_covariance(scores, kernel):
    """Aggregate weighted scores jointly without pairwise normalization or PSD repair."""
    value = scores.T @ kernel @ scores
    return (value+value.T)/2
