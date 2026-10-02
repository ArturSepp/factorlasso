"""Public result types and the estimator mode enum, plus private mode metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd



class LassoModelType(Enum):
    """Supported LASSO estimation methods."""
    # univariate estimators (no grouping, no clustering)
    LASSO = 1                   #: Standard L1 LASSO
    UNILASSO = 2                #: UniLasso — per-response univariate-guided
    # group estimator (user-defined partition)
    GROUP_LASSO = 3             #: Group LASSO with user-defined groups
    # cluster-based, discovered partition (HCGL / FCGL)
    HIERARCHICAL_CLUSTER_GROUP_LASSO = 4  #: HCGL — row-grouped penalty on discovered clusters
    FACTOR_CLUSTER_GROUP_LASSO = 5  #: FCGL — cluster-by-factor block penalty
    # cooperative (soft within-block sign coherence)
    COOPERATIVE_GROUP_LASSO = 6  #: coop-LASSO on user groups
    COOPERATIVE_CLUSTER_GROUP_LASSO = 7  #: coop-LASSO on discovered clusters


# Solvers that take no hard sign constraint: UniLasso is a two-stage univariate-guided fit and
# the cooperative penalty handles signs softly through the positive and negative parts of beta.
# A sign matrix or ``nonneg`` is rejected for these modes instead of being dropped silently.
_MODES_WITHOUT_SIGN_CONSTRAINTS = (
    LassoModelType.UNILASSO,
    LassoModelType.COOPERATIVE_GROUP_LASSO,
    LassoModelType.COOPERATIVE_CLUSTER_GROUP_LASSO,
)


@dataclass
class LassoEstimationResult:
    """
    Output container for LASSO / Group LASSO solver functions.

    Attributes
    ----------
    estimated_beta : np.ndarray, shape (N, M)
        Factor loadings.  NaN if solver failed.
    alpha : np.ndarray, shape (N,)
        EWMA-weighted mean of the *demeaned* residuals, per response.

        Important: this is **not** the regression intercept in the original
        ``y = α + Xβ + ε`` representation.  Because :func:`get_x_y_np` removes
        the conditional mean of both ``y`` and ``X`` before the solver runs,
        the model that is actually fitted is

            ``y_demeaned ≈ X_demeaned · β``       (no intercept term)

        and ``alpha`` here is computed *post-hoc* as the weighted mean of the
        residuals on the demeaned data, in the nominal-span EWMA norm
        (per-observation weight ``lambda^k``, the same norm as the solver
        loss; before v0.5.0 this diagnostic was inadvertently computed at an
        effective span of ``≈ 2·span``).  In particular:

        * If ``span is None`` (sample-mean demeaning), this quantity is
          identically zero by the OLS first-order condition.
        * If ``span`` is set (one-sided EWMA demeaning), this quantity is the
          leftover when the EWMA mean does not match the sample mean — i.e.
          a *finite-sample EWMA-demean residual*, not an intercept.

        ``LassoModel`` exposes this value as ``model.intercept_`` for
        backward compatibility; the **economic intercept** of the regression
        in original units is available separately as ``model.alpha_const_``.
    ss_total : np.ndarray, shape (N,)
        EWMA-weighted total variance per response variable
        (nominal-span norm).
    ss_res : np.ndarray, shape (N,)
        EWMA-weighted residual variance per response variable
        (nominal-span norm).
    r2 : np.ndarray, shape (N,)
        R-squared per response variable.
    """
    estimated_beta: np.ndarray
    alpha: np.ndarray
    ss_total: np.ndarray
    ss_res: np.ndarray
    r2: np.ndarray


@dataclass(frozen=True)
class LassoNowcastResult:
    """Immutable output of :meth:`LassoModel.nowcast`.

    Attributes
    ----------
    prediction : pd.DataFrame, shape (K, N)
        Target-period response nowcasts in the fitted return units.
    factor_component : pd.DataFrame, shape (K, N)
        Target factor returns multiplied by the fitted betas, before alpha.
    target_factors : pd.DataFrame, shape (K, M)
        Deep copy of the target factor frame in exact fitted factor order.
    stat_alpha : pd.Series, shape (N,)
        Terminal causal mean of original-unit residuals ``y - X @ beta``.
    betas : pd.DataFrame, shape (N, M)
        Deep copy of the fitted coefficients used by this nowcast.
    residuals : pd.DataFrame, shape (T, N)
        Deep copy of the fit-time original-unit residual snapshot.
    diagnostics : pd.DataFrame, shape (N, D)
        Per-response sample metadata and fitted solver diagnostics.
    """

    prediction: pd.DataFrame
    factor_component: pd.DataFrame
    target_factors: pd.DataFrame
    stat_alpha: pd.Series
    betas: pd.DataFrame
    residuals: pd.DataFrame
    diagnostics: pd.DataFrame
