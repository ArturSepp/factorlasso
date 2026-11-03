"""Map the observed-cell conditional inference gate to prepared estimator arrays.

Statistical estimation and date-clustered HC1 belong to inference._ragged_signs.
This adapter restores missing cells and preserves the solver's original EWMA
calendar. It neither changes adaptive penalties nor falls back to marginal signs.
"""

from typing import Optional

import numpy as np
import pandas as pd

from factorlasso.inference._ragged_signs import compute_ragged_pooled_sign_gate
from factorlasso.priors._signs import _sign_observation_weights


def conditional_automatic_signs(
    x: pd.DataFrame,
    x_np: np.ndarray,
    y_np: np.ndarray,
    valid_mask: np.ndarray,
    response_columns: pd.Index,
    asset_clusters: Optional[pd.Series],
    sign_span: Optional[float],
    threshold_t: Optional[float],
    on_unidentified: str,
):
    """Return conditional restrictions/evidence and owned group-support diagnostics."""
    offset = len(x) - len(x_np)
    xx = pd.DataFrame(
        np.where(x.iloc[offset:].notna().to_numpy(), x_np, np.nan),
        index=x.index[offset:],
        columns=x.columns,
    )
    yy = pd.DataFrame(
        np.where(valid_mask > 0, y_np, np.nan), index=xx.index, columns=response_columns
    )
    groups = (
        pd.Series(np.arange(len(response_columns)), index=response_columns)
        if asset_clusters is None
        else asset_clusters.reindex(response_columns)
    )
    weights = pd.Series(_sign_observation_weights(len(xx), sign_span), index=xx.index)
    # Preserve LassoModel's historical threshold endpoints. The public inference
    # API intentionally admits only finite thresholds; evidence stays unchanged.
    exclude_all = threshold_t is not None and np.isposinf(threshold_t)
    threshold = 0.0 if threshold_t is None or exclude_all else threshold_t
    gate = compute_ragged_pooled_sign_gate(
        xx,
        yy,
        groups,
        weights=weights,
        threshold=threshold,
        action="exclude",
        on_unidentified=on_unidentified,
    )
    restrictions = gate.restrictions.to_numpy(copy=True)
    if exclude_all:
        restrictions[np.isfinite(restrictions)] = 0.0
    support = gate.diagnostics.loc[groups.to_numpy()]
    count = len(x.columns)
    diagnostics = dict(
        t_stats=gate.t_stats.to_numpy(),
        effective_n=np.repeat(support.effective_n.to_numpy()[:, None], count, axis=1),
        n_obs=np.repeat(support.response_observations.to_numpy()[:, None], count, axis=1),
    )
    return restrictions, gate.slopes.to_numpy(), diagnostics, gate.diagnostics.copy(deep=True)
