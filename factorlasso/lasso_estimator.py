"""
LASSO and Group LASSO factor model estimation using CVXPY.

Implements sparse multi-output regression with support for:

- **Standard L1 LASSO** — element-wise sparsity
- **Group LASSO** with predefined groups — structured sparsity
- **Hierarchical Clustering Group LASSO (HCGL)** — data-driven group
  discovery via Ward's method, then Group LASSO with adaptive penalties
- **Sign constraints** on regression coefficients
  (non-negative, non-positive, zero, free)
- **Prior-centered regularisation** — penalise ‖β − β₀‖ instead of ‖β‖
- **EWMA-weighted objectives** — exponential decay for non-stationary data
- **NaN-aware estimation** — validity masking preserves all usable data

Convention
----------
The factor model follows the paper convention (column vectors)::

    Y_t = α + β X_t + ε_t

where Y_t is ``(N × 1)``, X_t is ``(M × 1)``, β is ``(N × M)``,
and α is ``(N × 1)``.  *N* is the number of response variables
and *M* is the number of regressors (factors).

In Python, pandas DataFrames store observations as rows (T × N).
The code computes the equivalent row-major form ``Y = X β' + α``
internally, but stores β as ``coef_`` in the paper shape ``(N × M)``.

When ``demean=True`` (default), the intercept α is absorbed by subtracting
the (EWMA) rolling mean from both Y and X before estimation.  The fitted
intercept ``intercept_`` is recovered as the EWMA-weighted mean of residuals.

The API follows scikit-learn conventions: ``fit(X, y)`` estimates parameters,
``predict(X)`` returns fitted values, ``score(X, y)`` returns R².  Fitted
attributes carry a trailing underscore (``coef_``, ``intercept_``, etc.).

References
----------
Sepp A., Ossa I., Kastenholz M. (2026), "Robust Optimization of
Strategic and Tactical Asset Allocation for Multi-Asset Portfolios",
*Journal of Portfolio Management*, 52(4), 86–120.

Yuan, M., Lin, Y. (2006), "Model selection and estimation in regression
with grouped variables", *J. R. Statist. Soc. B*, 68(1), 49–67.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import cvxpy as cvx
import numpy as np
import pandas as pd

from factorlasso.cluster_utils import (
    DEFAULT_CUTOFF_FRACTION,
    compute_clusters_from_corr_matrix,
)
from factorlasso.ewm_utils import (
    _validate_span,
    compute_ewm,
    compute_ewm_covar,
    compute_expanding_power,
    set_group_loadings,
)

# ═══════════════════════════════════════════════════════════════════════
# Public enums & result types
# ═══════════════════════════════════════════════════════════════════════

class LassoModelType(Enum):
    """Supported LASSO estimation methods."""
    LASSO = 1                   #: Standard L1 LASSO
    GROUP_LASSO = 2             #: Group LASSO with user-defined groups
    GROUP_LASSO_CLUSTERS = 3    #: HCGL — Group LASSO with hierarchical clustering


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
        residuals on the demeaned data.  In particular:

        * If ``span is None`` (sample-mean demeaning), this quantity is
          identically zero by the OLS first-order condition.
        * If ``span`` is set (one-sided EWMA demeaning), this quantity is the
          leftover when the EWMA mean does not match the sample mean — i.e.
          a *finite-sample EWMA-demean residual*, not an intercept.

        ``LassoModel`` exposes this value as ``model.intercept_`` for
        backward compatibility; the **economic intercept** of the regression
        in original units is available separately as ``model.alpha_const_``.
    ss_total : np.ndarray, shape (N,)
        EWMA-weighted total variance per response variable.
    ss_res : np.ndarray, shape (N,)
        EWMA-weighted residual variance per response variable.
    r2 : np.ndarray, shape (N,)
        R-squared per response variable.
    """
    estimated_beta: np.ndarray
    alpha: np.ndarray
    ss_total: np.ndarray
    ss_res: np.ndarray
    r2: np.ndarray


# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════

def _compute_solver_diagnostics(
    x: np.ndarray,
    y: np.ndarray,
    estimated_beta: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """In-sample fit diagnostics from solver weights.

    The returned ``alpha`` is the EWMA-weighted mean of residuals on the
    demeaned data the solver received. It is **not** the regression
    intercept in original units; see the docstring of
    :class:`LassoEstimationResult` for the distinction. The economic
    intercept of ``y = α + Xβ + ε`` is computed in
    :meth:`LassoModel.fit` from the sample means of the original (pre-
    demean) ``y`` and ``X``, and is exposed as ``model.alpha_const_``.
    """
    col_sums = np.sum(weights, axis=0)
    norm_w = np.divide(weights, col_sums, out=np.zeros_like(weights),
                       where=col_sums != 0)

    residuals = y - x @ estimated_beta.T
    alpha = np.sum(norm_w * residuals, axis=0)
    ss_res = np.sum(norm_w * np.square(residuals), axis=0)
    y_wmean = np.sum(norm_w * y, axis=0)
    ss_total = np.sum(norm_w * np.square(y - y_wmean), axis=0)
    r2 = np.zeros_like(ss_res)
    np.divide(ss_res, ss_total, out=r2, where=ss_total > 0.0)
    r2 = 1.0 - r2
    return alpha, ss_total, ss_res, r2


def _compute_solver_weights(
    t: int, n_y: int, span: Optional[float], valid_mask: np.ndarray
) -> np.ndarray:
    """Observation weights: EWMA decay × validity mask."""
    if span is not None:
        w = compute_expanding_power(
            n=t,
            power_lambda=np.sqrt(1.0 - 2.0 / (span + 1.0)),
            reverse_columns=True,
        )
    else:
        w = np.ones(t)

    if valid_mask.ndim == 2:
        w = np.tile(w, (n_y, 1)).T

    return w * valid_mask


def _clean_beta_prior(
    factors_beta_prior: Optional[np.ndarray], n_y: int, n_x: int
) -> np.ndarray:
    """Return clean prior (N × M): NaN → 0, None → zeros."""
    if factors_beta_prior is not None:
        return np.where(np.isnan(factors_beta_prior), 0.0, factors_beta_prior)
    return np.zeros((n_y, n_x))


def _derive_valid_mask_from_y(
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Derive validity mask from NaN positions and zero-fill NaNs."""
    nan_mask = np.isnan(y)
    return np.where(nan_mask, 0.0, y), (~nan_mask).astype(float)


def _build_sign_constraints(
    beta: cvx.Variable,
    signs: np.ndarray,
) -> list:
    """Build CVXPY constraints from sign matrix."""
    constraints = []
    zero_mask = np.isclose(signs, 0.0).astype(float)
    nonneg_mask = np.greater(signs, 0.0).astype(float)
    nonpos_mask = np.less(signs, 0.0).astype(float)

    if np.any(zero_mask > 0):
        constraints.append(cvx.multiply(zero_mask, beta) == 0)
    if np.any(nonneg_mask > 0):
        constraints.append(cvx.multiply(nonneg_mask, beta) >= 0)
    if np.any(nonpos_mask > 0):
        constraints.append(cvx.multiply(nonpos_mask, beta) <= 0)
    return constraints


def _nan_result(n_y: int, n_x: int) -> LassoEstimationResult:
    """Return NaN-filled result for failed solves."""
    return LassoEstimationResult(
        estimated_beta=np.full((n_y, n_x), np.nan),
        alpha=np.full(n_y, np.nan),
        ss_total=np.full(n_y, np.nan),
        ss_res=np.full(n_y, np.nan),
        r2=np.full(n_y, np.nan),
    )


# ═══════════════════════════════════════════════════════════════════════
# Data preparation
# ═══════════════════════════════════════════════════════════════════════

def get_x_y_np(
    x: Union[pd.DataFrame, pd.Series],
    y: Union[pd.DataFrame, pd.Series],
    span: Optional[float] = None,
    demean: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare numpy arrays from regressor/response DataFrames with NaN masking.

    Parameters
    ----------
    x : pd.DataFrame or pd.Series, shape (T, N) or (T,)
        Regressor data.  May have all-NaN rows.
    y : pd.DataFrame or pd.Series, shape (T, N) or (T,)
        Response data.  May contain NaNs (different history lengths).
        Series is converted to single-column DataFrame.
    span : float, optional
        EWMA span for demeaning.  ``None`` uses simple mean.  Must be ≥ 1
        when provided.  Float accepted — the recursion math does not
        require an integer span.
    demean : bool, default True
        If True, subtract (rolling) mean before estimation.

    Returns
    -------
    x_np : np.ndarray, shape (T', M)
    y_np : np.ndarray, shape (T', N)
    valid_mask : np.ndarray, shape (T', N)
        ``T' = T − 1`` when EWMA demeaning is used.
    """
    _validate_span(span)
    if isinstance(x, pd.Series):
        x = x.to_frame()
    if isinstance(y, pd.Series):
        y = y.to_frame()
    if not x.index.equals(y.index):
        raise ValueError(
            f"x and y must share the same index: "
            f"x has {len(x.index)} rows, y has {len(y.index)} rows"
        )

    nan_mask_y = y.isna().to_numpy().copy()
    x_all_nan = x.isna().all(axis=1).to_numpy()
    if np.any(x_all_nan):
        nan_mask_y[x_all_nan, :] = True

    x_np = x.fillna(0.0).to_numpy()
    y_np = y.fillna(0.0).to_numpy()

    if demean:
        if span is None:
            x_np = x_np - np.nanmean(x_np, axis=0)
            y_np = y_np - np.nanmean(y_np, axis=0)
        else:
            x_np = x_np - compute_ewm(x_np, span=span)
            y_np = y_np - compute_ewm(y_np, span=span)
            x_np = x_np[1:, :]
            y_np = y_np[1:, :]
            nan_mask_y = nan_mask_y[1:, :]

    return x_np, y_np, (~nan_mask_y).astype(float)


# ═══════════════════════════════════════════════════════════════════════
# Solvers
# ═══════════════════════════════════════════════════════════════════════

def solve_lasso_cvx_problem(
    x: np.ndarray,
    y: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    reg_lambda: float = 1e-8,
    span: Optional[float] = None,
    verbose: bool = False,
    solver: str = 'CLARABEL',
    nonneg: bool = False,
    factors_beta_loading_signs: Optional[np.ndarray] = None,
    factors_beta_prior: Optional[np.ndarray] = None,
) -> LassoEstimationResult:
    r"""
    L1-regularised (LASSO) multi-output regression via CVXPY.

    Minimises

    .. math::

        \frac{1}{T}\|W \odot (X\beta^\top - Y)\|_F^2
        + \lambda\|\beta - \beta_0\|_1

    where β is ``(N × M)``, X is ``(T × M)``, Y is ``(T × N)``.

    Parameters
    ----------
    x : np.ndarray, shape (T, M)
        Regressor matrix.
    y : np.ndarray, shape (T, N)
        Response matrix.
    valid_mask : np.ndarray, shape (T, N), optional
        Binary validity mask.  Derived from ``y`` if ``None``.
    reg_lambda : float, default 1e-8
        L1 regularisation strength.
    span : float, optional
        EWMA span for observation weighting.  Must be ≥ 1 when provided.
        Float accepted.
    verbose : bool, default False
        Print CVXPY solver output.
    solver : str, default 'CLARABEL'
        CVXPY solver name.
    nonneg : bool, default False
        Constrain all β ≥ 0.
    factors_beta_loading_signs : np.ndarray, shape (N, M), optional
        Element-wise sign constraints.
    factors_beta_prior : np.ndarray, shape (N, M), optional
        Prior β₀.  NaN entries → zero prior.

    Returns
    -------
    LassoEstimationResult
    """
    _validate_span(span)
    assert y.ndim == 2 and x.ndim == 2 and x.shape[0] == y.shape[0]
    t, n_x = x.shape
    n_y = y.shape[1]

    if valid_mask is None:
        y, valid_mask = _derive_valid_mask_from_y(y)
    if t < 5:
        warnings.warn(f"insufficient observations for lasso: t={t}")
        return _nan_result(n_y, n_x)

    # Variable and constraints
    if factors_beta_loading_signs is not None:
        beta = cvx.Variable((n_y, n_x))
        constraints = _build_sign_constraints(beta, factors_beta_loading_signs)
    else:
        beta = cvx.Variable((n_y, n_x), nonneg=nonneg)
        constraints = []

    weights = _compute_solver_weights(t, n_y, span, valid_mask)
    prior = _clean_beta_prior(factors_beta_prior, n_y, n_x)

    objective = cvx.Minimize(
        (1.0 / t) * cvx.sum_squares(cvx.multiply(weights, x @ beta.T - y))
        + reg_lambda * cvx.norm1(beta - prior)
    )
    problem = cvx.Problem(objective, constraints) if constraints else cvx.Problem(objective)
    problem.solve(verbose=verbose, solver=solver)

    if beta.value is None:
        warnings.warn("lasso problem not solved")
        return _nan_result(n_y, n_x)

    alpha, ss_total, ss_res, r2 = _compute_solver_diagnostics(
        x, y, beta.value, weights
    )
    return LassoEstimationResult(
        estimated_beta=beta.value, alpha=alpha,
        ss_total=ss_total, ss_res=ss_res, r2=r2,
    )


def solve_group_lasso_cvx_problem(
    x: np.ndarray,
    y: np.ndarray,
    group_loadings: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    reg_lambda: float = 1e-8,
    span: Optional[float] = None,
    nonneg: bool = False,
    verbose: bool = False,
    solver: str = 'CLARABEL',
    factors_beta_loading_signs: Optional[np.ndarray] = None,
    factors_beta_prior: Optional[np.ndarray] = None,
    group_penalty: str = "normalized",
    l1_weight: float = 0.0,
) -> LassoEstimationResult:
    r"""
    Group LASSO multi-output regression via CVXPY.

    Minimises

    .. math::

        \frac{1}{T}\|W \odot (X\beta^\top - Y)\|_F^2
        + (1 - \alpha)\,\lambda \sum_g w_g \sum_{i \in g}
          \|\beta_{i,:} - \beta_{0,i,:}\|_2
        + \alpha\,\lambda \,\|\beta - \beta_0\|_1

    where *g* indexes groups of response variables (rows of β) and the
    per-group weight ``w_g`` is set by ``group_penalty`` (see below).
    The inner sum of the group term is the ``L_{2,1}`` norm of the group
    submatrix — each response's loading vector is shrunk by an L2 norm,
    and block sparsity is driven across responses within a group. The
    optional L1 term drives elementwise sparsity on top, zeroing
    individual assets whose loadings are noisy even within an "active"
    group — the Simon–Friedman–Hastie–Tibshirani (2013) Sparse Group
    LASSO formulation.

    At ``l1_weight=0.0`` (default) the problem reduces to the previous
    pure group LASSO and is numerically identical to v0.3.1.

    Parameters
    ----------
    x : np.ndarray, shape (T, M)
    y : np.ndarray, shape (T, N)
    group_loadings : np.ndarray, shape (N, G)
        Binary group membership matrix.
    valid_mask, reg_lambda, span, nonneg, verbose, solver,
    factors_beta_loading_signs, factors_beta_prior
        See :func:`solve_lasso_cvx_problem`.
    group_penalty : {"normalized", "yuan_lin"}, default "normalized"
        Per-group weighting convention:

        - ``"normalized"``: ``w_g = √(|g|/G)``. Group-count-invariant —
          keeps the effective regularisation scale stable across
          problems with different numbers of groups G. This is the
          package default and the appropriate choice for HCGL, where
          G is data-driven and can vary across estimation dates or
          rolling windows.
        - ``"yuan_lin"``: ``w_g = √|g|``. Classical Yuan–Lin (2006)
          weighting. Opt in when the number of groups is fixed by the
          problem specification (not data-driven) and you want the
          textbook convention.

        The two conventions are related by a constant factor √G, so
        results under ``"yuan_lin"`` at regularisation ``λ`` match
        results under ``"normalized"`` at regularisation ``λ·√G``.
    l1_weight : float, default 0.0
        Sparse Group LASSO mixing parameter ``α ∈ [0, 1]``. Weight on
        the elementwise L1 penalty term; ``(1 - α)`` weights the group
        L2 term. Set ``α = 0`` (default) for pure group LASSO —
        backward compatible with v0.3.1. Set ``α = 1`` for pure LASSO
        (no group structure). Typical research values are ``α ∈
        [0.05, 0.20]``: preserve group structure as the primary
        selection mechanism while allowing additional within-group
        elementwise zeroing for assets whose loadings are noisy. The
        L1 term shrinks ``β`` toward the prior ``β_0`` elementwise,
        consistent with the group term which also shrinks toward the
        prior.

    Returns
    -------
    LassoEstimationResult
    """
    _validate_span(span)
    if group_penalty not in ("normalized", "yuan_lin"):
        raise ValueError(
            f"group_penalty must be 'normalized' or 'yuan_lin', "
            f"got {group_penalty!r}"
        )
    if not (0.0 <= l1_weight <= 1.0):
        raise ValueError(
            f"l1_weight must lie in [0, 1], got {l1_weight!r}"
        )
    assert y.ndim == 2 and x.ndim == 2 and group_loadings.ndim == 2
    assert x.shape[0] == y.shape[0] and y.shape[1] == group_loadings.shape[0]

    t, n_x = x.shape
    n_y = y.shape[1]
    n_groups = group_loadings.shape[1]

    if valid_mask is None:
        y, valid_mask = _derive_valid_mask_from_y(y)
    if t < 5:
        warnings.warn(f"insufficient observations for group lasso: t={t}")
        return _nan_result(n_y, n_x)

    # Variable and constraints
    if factors_beta_loading_signs is not None:
        beta = cvx.Variable((n_y, n_x))
        constraints = _build_sign_constraints(beta, factors_beta_loading_signs)
    else:
        beta = cvx.Variable((n_y, n_x), nonneg=nonneg)
        constraints = []

    weights = _compute_solver_weights(t, n_y, span, valid_mask)
    prior = _clean_beta_prior(factors_beta_prior, n_y, n_x)

    # Fit term
    fit = (1.0 / t) * cvx.sum_squares(cvx.multiply(weights, x @ beta.T - y))

    # Per-group weight. "normalized" (default) preserves the v0.2.2
    # behaviour √(|g|/G); "yuan_lin" uses the classical √|g|.
    def _weight(m: np.ndarray) -> float:
        g = np.sum(m)
        if group_penalty == "yuan_lin":
            return float(np.sqrt(g))
        return float(np.sqrt(g / n_groups))

    # Group penalty (L_{2,1} norm within each group, scaled by (1 - α))
    masks = [
        np.isclose(group_loadings[:, g], 1.0) for g in range(n_groups)
    ]
    group_pen = cvx.sum([
        reg_lambda * _weight(m)
        * cvx.sum(cvx.norm2(beta[m, :] - prior[m, :], axis=1))
        for m in masks
    ])

    # Elementwise L1 penalty (scaled by α). Shrinks toward the same
    # prior used by the group term so at α=1 the problem is consistent
    # with plain LASSO centred on β₀. At α=0 this term vanishes and
    # the problem reduces exactly to the v0.3.1 pure group LASSO.
    if l1_weight > 0.0:
        l1_pen = reg_lambda * cvx.sum(cvx.abs(beta - prior))
        penalty = (1.0 - l1_weight) * group_pen + l1_weight * l1_pen
    else:
        penalty = group_pen

    problem = cvx.Problem(cvx.Minimize(fit + penalty), constraints) \
        if constraints else cvx.Problem(cvx.Minimize(fit + penalty))
    problem.solve(verbose=verbose, solver=solver)

    if beta.value is None:
        warnings.warn("group lasso problem not solved")
        return _nan_result(n_y, n_x)

    alpha, ss_total, ss_res, r2 = _compute_solver_diagnostics(
        x, y, beta.value, weights
    )
    return LassoEstimationResult(
        estimated_beta=beta.value, alpha=alpha,
        ss_total=ss_total, ss_res=ss_res, r2=r2,
    )


# ═══════════════════════════════════════════════════════════════════════
# High-level model class
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LassoModel:
    """
    Configurable LASSO / Group LASSO / HCGL factor model estimator.

    Estimates the model ``Y_t = α + β X_t + ε_t`` with sparse β using
    L1 (LASSO) or Group L2/L1 (Group LASSO) regularisation via CVXPY.

    The API follows scikit-learn conventions:

    - ``fit(x, y)`` estimates parameters, returns ``self``
    - ``predict(x)`` returns Ŷ_t = α + β X_t (computed as ``X @ β' + α``)
    - ``score(x, y)`` returns mean R² across response variables
    - Fitted attributes use trailing underscore: ``coef_``, ``intercept_``

    Convention
    ----------
    β is ``(N × M)`` following the paper.  After ``fit()``:

    - ``coef_`` (also ``estimated_betas``): DataFrame (N × M)
    - ``intercept_``: Series (N,) — the α vector

    Sign constraints
    ~~~~~~~~~~~~~~~~
    ``factors_beta_loading_signs`` is ``(N × M)``::

        0  → constrained to zero
        1  → constrained non-negative
       -1  → constrained non-positive
       NaN → unconstrained (free)

    Prior-centered regularisation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``factors_beta_prior`` is ``(N × M)``.  The penalty becomes
    ``‖β − β₀‖`` instead of ``‖β‖``.

    Parameters
    ----------
    model_type : LassoModelType, default LASSO
    reg_lambda : float, default 1e-5
    span : float, optional
        EWMA span for observation weighting.  Must be ≥ 1 when provided.
        Float accepted — integer is the common case, but the recursion
        math does not require it.
    span_freq_dict : dict, optional
        Per-frequency override of ``span`` used by multi-frequency
        pipelines downstream (``optimalportfolios`` / ``rosaa``).  Keys
        are pandas freq codes (``'ME'``, ``'QE'``), values are spans at
        that frequency (float).  Carried through the model specification
        but not consumed by :meth:`fit`; the caller selects the right
        span when it slices per frequency.
    group_data : pd.Series, optional
        Group labels (required for ``GROUP_LASSO``).
    cutoff_fraction : float, default 0.5
        Fraction of ``max(pdist)`` at which to cut the dendrogram when
        ``model_type == GROUP_LASSO_CLUSTERS``.  Ignored by other modes.
        See :func:`factorlasso.compute_clusters_from_corr_matrix`.
    group_penalty : {"normalized", "yuan_lin"}, default "normalized"
        Per-group weighting for the group-LASSO penalty.  ``"normalized"``
        uses ``√(|g|/G)`` (group-count-invariant) and is the default —
        appropriate for HCGL where the number of groups is data-driven.
        ``"yuan_lin"`` uses the classical Yuan–Lin (2006) ``√|g|``.
        Ignored for ``model_type == LASSO``.  See
        :func:`solve_group_lasso_cvx_problem` for the full formula.
    l1_weight : float, default 0.0
        Sparse Group LASSO mixing parameter ``α ∈ [0, 1]``. Adds an
        elementwise L1 penalty ``α·λ·|β - β₀|`` on top of the standard
        group L2 penalty (which is scaled by ``(1 - α)``). Set ``α = 0``
        (default) for pure group LASSO — backward compatible with
        v0.3.1. Typical research values: ``α ∈ [0.05, 0.20]`` — preserve
        group structure as the primary mechanism while allowing
        additional within-group elementwise zeroing for assets whose
        loadings are noisy. Only consumed when ``model_type`` is
        ``GROUP_LASSO`` or ``GROUP_LASSO_CLUSTERS``; ignored for pure
        ``LASSO`` since L1 is the only penalty already.
    factors_beta_loading_signs : pd.DataFrame, optional
    factors_beta_prior : pd.DataFrame, optional
    auto_sign_constraints : bool, default False
        If True, signs are derived inside ``fit()`` from the EWMA-demeaned,
        NaN-masked arrays returned by ``get_x_y_np`` (i.e. the same data the
        CVXPY solver consumes). Pooling strategy is dispatched by
        ``model_type``:

        * ``LASSO`` (or single-column y): per-y-column independent
          univariate sign derivation; rows of ``derived_signs_`` may differ.
        * ``GROUP_LASSO``: signs pooled within each ``group_data`` group;
          members of a group share their ``derived_signs_`` row.
        * ``GROUP_LASSO_CLUSTERS``: signs pooled within each HCGL asset
          cluster (the same clustering the group solver uses).

        When ``factors_beta_loading_signs`` is *also* supplied, the explicit
        matrix is overlaid on the auto-derived signs per-cell: non-NaN
        explicit values win, NaN cells inherit the auto value.
    demean : bool, default True
    solver : str, default 'CLARABEL'
    warmup_period : int, default 12

    Attributes (fitted, set by ``fit()``)
    --------------------------------------
    coef_ : pd.DataFrame, shape (N, M)
        Estimated factor loadings β.
    alpha_const_ : pd.Series, shape (N,)
        **Economic intercept α** — the constant term in the regression
        ``y = α + Xβ + ε`` paired consistently with the fitted β.
        Reconstructed from weighted means of ``y`` and ``X`` using the
        same weighting that produced β:

        * for ``span=None`` (uniform weights), this is the sample-mean
          reconstruction ``α = ȳ_sample − x̄_sample · β``, identical to
          the OLS intercept;
        * for ``span=integer`` (EWMA weights), this uses EWMA-weighted
          means with the same weights factorlasso applies in the loss
          function, so the ``(α, β)`` pair represents one coherent
          weighted-least-squares solution rather than two estimators
          under different weightings.

        This is the field to read when reporting "alpha after factor
        exposure".
    intercept_ : pd.Series, shape (N,)
        Raw solver output: the EWMA-weighted mean of residuals on the
        *demeaned* data, equal to ``estimation_result_.alpha``. Because
        the underlying solver fits a no-intercept model on centered data,
        this is a mechanical artefact of the fit, **not** the regression
        intercept in original units:

        * for ``span=None`` this is identically zero by the OLS
          first-order condition;
        * for ``span=integer`` it is a finite-sample EWMA-demean leftover.

        Preserved under this name for back-compatibility with code that
        read ``model.intercept_`` in pre-0.3.4 versions. New code should
        use ``alpha_const_`` for the economic intercept.
    estimation_result_ : LassoEstimationResult
        Full diagnostics (alpha, ss_total, ss_res, r2).
    clusters_ : pd.Series or None
        Cluster labels (HCGL only).
    linkage_ : np.ndarray or None
        Scipy linkage matrix (HCGL only).
    cutoff_ : float or None
        Dendrogram cut distance (HCGL only).
    derived_signs_ : pd.DataFrame or None
        The final ``(N × M)`` sign matrix that was passed to the solver,
        in ``LassoModel.factors_beta_loading_signs`` convention
        (``+1`` non-negative, ``-1`` non-positive, ``0`` forced zero,
        ``NaN`` unconstrained). Populated whenever sign constraints were
        actually applied during the fit:

        * ``auto_sign_constraints=True`` only — pooled univariate signs
          from the EWMA-demeaned, NaN-masked arrays the solver consumes,
          identical across response rows.
        * ``factors_beta_loading_signs`` only — the user's matrix reindexed
          to the fit universe.
        * Both — auto-derived signs as the base layer, overlaid with the
          explicit per-cell values wherever ``factors_beta_loading_signs``
          is non-NaN (per-asset overrides for the asset-specific master
          constraints).

        Read this attribute to inspect, log, or render the constraints
        that actually shaped the fitted ``coef_``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from factorlasso import LassoModel, LassoModelType
    >>> np.random.seed(42)
    >>> T, M, N = 200, 3, 5
    >>> X = pd.DataFrame(np.random.randn(T, M), columns=[f'f{i}' for i in range(M)])
    >>> beta_true = np.array([[1, 0, .5], [0, 1, 0], [.3, 0, 0],
    ...                       [0, .8, .2], [1, .5, 0]])
    >>> Y = pd.DataFrame(X.values @ beta_true.T + .1*np.random.randn(T, N),
    ...                   columns=[f'y{i}' for i in range(N)])
    >>> model = LassoModel(model_type=LassoModelType.LASSO, reg_lambda=1e-4)
    >>> _ = model.fit(x=X, y=Y)
    >>> model.coef_.shape
    (5, 3)
    >>> y_hat = model.predict(X)
    >>> r2 = model.score(X, Y)
    """
    # ── Hyperparameters (constructor args) ────────────────────────────
    model_type: LassoModelType = LassoModelType.LASSO
    group_data: Optional[pd.Series] = None
    reg_lambda: float = 1e-5
    span: Optional[float] = None
    span_freq_dict: Optional[Dict[str, float]] = None
    cutoff_fraction: float = DEFAULT_CUTOFF_FRACTION
    group_penalty: str = "normalized"
    l1_weight: float = 0.0
    demean: bool = True
    solver: str = 'CLARABEL'
    warmup_period: Optional[int] = 12
    nonneg: bool = False
    factors_beta_loading_signs: Optional[pd.DataFrame] = None
    factors_beta_prior: Optional[pd.DataFrame] = None
    # Auto sign-constraint derivation (signs computed inside fit on the
    # solver-ready EWMA-demeaned, NaN-masked arrays). Pooling strategy is
    # determined by ``model_type``:
    #   * GROUP_LASSO_CLUSTERS → pool within each asset cluster from HCGL
    #   * GROUP_LASSO          → pool within each ``group_data`` group
    #   * LASSO / single-col y → per-y-column independent derivation
    auto_sign_constraints: bool = False

    # Significance gate for auto-derived signs.  When set (>0), only
    # columns whose univariate ``|t|`` meets the threshold get a hard
    # sign constraint; columns failing the threshold are pinned to 0
    # (β forced to zero), excluding them from the regression.  This
    # enforces parsimony directly and is robust to the choice of
    # ``reg_lambda``.  Default 0.75 acts as a noise floor — it is
    # well below conventional significance levels but high enough
    # to filter columns whose univariate slope sign is dominated by
    # sampling noise (|t| < 0.75 ⇒ two-sided p > 0.45).  Pass
    # ``None`` to disable the gate and reproduce v0.3.6 behaviour.
    #
    # Typical alternative values: 0.5 (looser) to 1.0 (stricter).
    # Only effective when ``auto_sign_constraints=True``.
    auto_sign_threshold_t: Optional[float] = 0.75

    # ── Fitted state (set by fit(), trailing underscore) ──────────────
    x_: Optional[pd.DataFrame] = None
    y_: Optional[pd.DataFrame] = None
    coef_: Optional[pd.DataFrame] = None
    intercept_: Optional[pd.Series] = None
    alpha_const_: Optional[pd.Series] = None
    estimation_result_: Optional[LassoEstimationResult] = None
    clusters_: Optional[pd.Series] = None
    linkage_: Optional[np.ndarray] = None
    cutoff_: Optional[float] = None
    valid_mask_: Optional[np.ndarray] = None
    effective_span_: Optional[float] = None
    derived_signs_: Optional[pd.DataFrame] = None

    def __post_init__(self):
        if self.model_type == LassoModelType.GROUP_LASSO and self.group_data is None:
            raise ValueError(
                "group_data must be provided for model_type=GROUP_LASSO"
            )
        _validate_span(self.span)
        if not (0.0 < self.cutoff_fraction <= 1.0):
            raise ValueError(
                f"cutoff_fraction must lie in (0, 1], "
                f"got {self.cutoff_fraction!r}"
            )
        if self.group_penalty not in ("normalized", "yuan_lin"):
            raise ValueError(
                f"group_penalty must be 'normalized' or 'yuan_lin', "
                f"got {self.group_penalty!r}"
            )
        if not (0.0 <= self.l1_weight <= 1.0):
            raise ValueError(
                f"l1_weight must lie in [0, 1], got {self.l1_weight!r}"
            )

    # ── Backward-compatible property aliases ─────────────────────────

    @property
    def estimated_betas(self) -> Optional[pd.DataFrame]:
        """Alias for ``coef_`` (backward compatibility)."""
        return self.coef_

    @estimated_betas.setter
    def estimated_betas(self, value):
        self.coef_ = value

    @property
    def clusters(self) -> Optional[pd.Series]:
        """Alias for ``clusters_`` (backward compatibility)."""
        return self.clusters_

    @clusters.setter
    def clusters(self, value):
        self.clusters_ = value

    @property
    def linkage(self) -> Optional[np.ndarray]:
        """Alias for ``linkage_`` (backward compatibility)."""
        return self.linkage_

    @linkage.setter
    def linkage(self, value):
        self.linkage_ = value

    @property
    def cutoff(self) -> Optional[float]:
        """Alias for ``cutoff_`` (backward compatibility)."""
        return self.cutoff_

    @cutoff.setter
    def cutoff(self, value):
        self.cutoff_ = value

    @property
    def x(self) -> Optional[pd.DataFrame]:
        """Alias for ``x_`` (backward compatibility)."""
        return self.x_

    @x.setter
    def x(self, value):
        self.x_ = value

    @property
    def y(self) -> Optional[pd.DataFrame]:
        """Alias for ``y_`` (backward compatibility)."""
        return self.y_

    @y.setter
    def y(self, value):
        self.y_ = value

    # ── scikit-learn compatibility ───────────────────────────────────

    @classmethod
    def _constructor_param_names(cls) -> List[str]:
        """Names of constructor (non-fitted) dataclass fields."""
        return [f.name for f in fields(cls) if not f.name.endswith("_")]

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Return constructor hyperparameters as a dict (sklearn-compatible).

        Parameters
        ----------
        deep : bool, default True
            Present for sklearn API parity; LassoModel has no nested
            estimators so this argument has no effect.

        Returns
        -------
        dict
            Mapping ``{param_name: value}`` for every constructor argument.
            Fitted attributes (trailing underscore) are excluded.
        """
        del deep  # unused, kept for API parity
        return {name: getattr(self, name) for name in self._constructor_param_names()}

    def set_params(self, **params: Any) -> "LassoModel":
        """
        Set constructor hyperparameters in place (sklearn-compatible).

        Returns ``self`` for method chaining.

        Raises
        ------
        ValueError
            If any key is not a valid constructor parameter.
        """
        valid = set(self._constructor_param_names())
        invalid = sorted(set(params) - valid)
        if invalid:
            raise ValueError(
                f"Invalid parameter(s) for LassoModel: {invalid}. "
                f"Valid parameters: {sorted(valid)}"
            )
        for name, value in params.items():
            setattr(self, name, value)
        return self

    # ── Core API ─────────────────────────────────────────────────────

    @staticmethod
    def _validate_fit_inputs(
        x: Union[pd.DataFrame, pd.Series],
        y: Union[pd.DataFrame, pd.Series],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Coerce Series → DataFrame and validate shapes / index alignment."""
        if isinstance(x, pd.Series):
            x = x.to_frame()
        if isinstance(y, pd.Series):
            y = y.to_frame()
        if not isinstance(x, pd.DataFrame):
            raise TypeError(
                f"x must be pd.DataFrame or pd.Series, got {type(x).__name__}"
            )
        if not isinstance(y, pd.DataFrame):
            raise TypeError(
                f"y must be pd.DataFrame or pd.Series, got {type(y).__name__}"
            )
        if len(x) == 0:
            raise ValueError("Empty input: x and y must have at least one row")
        if not x.index.equals(y.index):
            raise ValueError(
                f"x and y must share the same index: "
                f"x has {len(x)} rows, y has {len(y)} rows"
            )
        return x, y

    def copy(self, kwargs: Optional[Dict] = None) -> LassoModel:
        """Create a copy, optionally overriding parameters."""
        this = asdict(self).copy()
        if kwargs is not None:
            this.update(kwargs)
        return LassoModel(**this)

    def fit(
        self,
        x: Union[pd.DataFrame, pd.Series],
        y: Union[pd.DataFrame, pd.Series],
        verbose: bool = False,
        span: Optional[float] = None,
    ) -> LassoModel:
        """
        Estimate model: Y_t = α + β X_t + ε_t.

        Parameters
        ----------
        x : pd.DataFrame or pd.Series, shape (T, M) or (T,)
            Regressor (factor) returns.  Series is converted to single-column DataFrame.
        y : pd.DataFrame or pd.Series, shape (T, N) or (T,)
            Response (asset) returns.  May contain NaNs.
            Series is converted to single-column DataFrame.
        verbose : bool, default False
            Print solver diagnostics.
        span : float, optional
            Per-call override of the model's ``span`` hyperparameter.
            ``None`` (the default) falls back to ``self.span`` without
            modification — previous versions used ``span or self.span``
            which would treat ``span=0`` as "unset".

        Returns
        -------
        self
            Updated with ``coef_`` (N × M) and ``intercept_`` (N,).
        """
        x, y = self._validate_fit_inputs(x, y)

        # Explicit None-check for span precedence: ``span or self.span``
        # would mistakenly treat span=0 as falsy and fall back to
        # self.span. Zero is not a valid span anyway (validated below),
        # but the correct idiom is an explicit None check.
        eff_span = self.span if span is None else span
        _validate_span(eff_span)
        x_np, y_np, valid_mask = get_x_y_np(
            x=x, y=y, span=eff_span, demean=self.demean
        )

        # ── Asset-side clustering (length N), computed once and shared by
        #    both the auto-sign derivation block and the solver dispatch.
        #    None for plain LASSO / single-column y; pd.Series indexed by
        #    y.columns for the GROUP modes.
        # ----------------------------------------------------------------
        asset_clusters: Optional[pd.Series] = None
        linkage = None
        cutoff = None
        is_lasso_mode = (
            self.model_type == LassoModelType.LASSO or y_np.shape[1] == 1
        )

        if is_lasso_mode:
            # asset_clusters stays None → per-y-column sign derivation below
            pass
        elif self.model_type == LassoModelType.GROUP_LASSO:
            asset_clusters = self.group_data[y.columns]
        elif self.model_type == LassoModelType.GROUP_LASSO_CLUSTERS:
            # Restore NaN before EWMA correlation (see block comment in the
            # solver-dispatch section below for the rationale).
            y_for_corr = np.where(valid_mask > 0, y_np, np.nan)
            corr = compute_ewm_covar(
                a=y_for_corr, span=eff_span, is_corr=True,
            )
            corr_df = pd.DataFrame(corr, columns=y.columns, index=y.columns)
            asset_clusters, linkage, cutoff = compute_clusters_from_corr_matrix(
                corr_df, cutoff_fraction=self.cutoff_fraction,
            )

        # ── Sign-constraint assembly ─────────────────────────────────
        # Two layers can contribute:
        #
        #   1. Auto-derived signs (auto_sign_constraints=True): univariate
        #      slopes computed on the EWMA-demeaned, NaN-masked arrays the
        #      solver actually consumes. Pooling strategy mirrors the
        #      solver's structural assumption:
        #        * GROUP modes  → pool y within each asset cluster; signs
        #          shared by every cluster member (rows identical within
        #          a cluster, can differ across clusters).
        #        * LASSO / single-col → derive signs per y-column
        #          independently; each row of the (N × M) matrix comes
        #          from a univariate fit against a single response.
        #
        #   2. Explicit factors_beta_loading_signs (N × M, NaN-permissive):
        #      asset-specific per-cell overrides. NaN means "use the auto
        #      layer here"; any non-NaN value wins.
        #
        # When both are supplied, (2) is overlaid on (1). When only (1) or
        # only (2) is supplied, the other layer is treated as all-NaN.
        # ----------------------------------------------------------------
        signs_np = None
        auto_signs_np = None
        explicit_signs_np = None

        if self.auto_sign_constraints:
            from factorlasso.sign_constraints import (
                _compute_sign_vector,
                _compute_sign_matrix_per_response,
            )
            N = y_np.shape[1]
            M = x_np.shape[1]

            if asset_clusters is not None:
                # Pool y columns within each asset cluster — one call per
                # cluster, broadcast result to all members.
                auto_signs_np = np.empty((N, M), dtype=float)
                cluster_vals = (
                    asset_clusters.values
                    if isinstance(asset_clusters, pd.Series)
                    else np.asarray(asset_clusters)
                )
                for c in np.unique(cluster_vals):
                    members_idx = np.where(cluster_vals == c)[0]
                    y_sub = y_np[:, members_idx]
                    sign_vec, _ = _compute_sign_vector(
                        x_arr=x_np, y_arr=y_sub,
                        clusters=None, master_constraints=None,
                        auto_sign_threshold_t=self.auto_sign_threshold_t,
                    )
                    auto_signs_np[members_idx, :] = sign_vec
            else:
                # LASSO or single-column y: per-y-column independent signs.
                # Bulk-vectorised closed-form path — eliminates the N-deep
                # Python loop of the prior implementation.
                auto_signs_np = _compute_sign_matrix_per_response(
                    x_arr=x_np, y_arr=y_np,
                    auto_sign_threshold_t=self.auto_sign_threshold_t,
                )

        if self.factors_beta_loading_signs is not None:
            explicit_signs_np = self.factors_beta_loading_signs.loc[
                y.columns, x.columns
            ].to_numpy()

        if auto_signs_np is not None and explicit_signs_np is not None:
            # Overlay: explicit per-cell value wins where non-NaN
            signs_np = np.where(
                np.isnan(explicit_signs_np), auto_signs_np, explicit_signs_np
            )
        elif auto_signs_np is not None:
            signs_np = auto_signs_np
        elif explicit_signs_np is not None:
            signs_np = explicit_signs_np

        # Persist the final solver-facing sign matrix for monitoring /
        # downstream inspection. Stored only when signs were actually used.
        if signs_np is not None:
            self.derived_signs_ = pd.DataFrame(
                signs_np, index=y.columns, columns=x.columns,
            )

        prior_np = None
        if self.factors_beta_prior is not None:
            prior_np = self.factors_beta_prior.loc[
                y.columns, x.columns
            ].to_numpy()

        # ── Solver dispatch (consumes the pre-computed asset_clusters) ──
        if is_lasso_mode:
            result = solve_lasso_cvx_problem(
                x=x_np, y=y_np, valid_mask=valid_mask,
                reg_lambda=self.reg_lambda, span=eff_span,
                verbose=verbose, solver=self.solver,
                nonneg=self.nonneg,
                factors_beta_loading_signs=signs_np,
                factors_beta_prior=prior_np,
            )

        elif self.model_type == LassoModelType.GROUP_LASSO:
            # Restore NaN positions before computing the EWMA correlation for
            # clustering. ``get_x_y_np`` zero-fills NaN in y_np so the CVXPY
            # quadratic-loss solvers can process a finite array (the weights
            # matrix ``valid_mask`` separately zeros out those observations in
            # the loss). But ``compute_ewm_covar`` interprets every finite row
            # as a legitimate observation — a zero-filled row enters the
            # correlation recursion as a "zero return" observation rather than
            # as "no observation", which systematically shrinks the estimated
            # correlations of assets with longer leading-NaN prefixes toward
            # zero. That shrinkage then propagates into Ward linkage distances,
            # dendrogram cuts, and group-lasso β estimates, producing results
            # that depend on how much pre-history each asset carries rather
            # than on economic correlation structure.
            #
            # Passing NaN through lets compute_ewm_covar's NaN-aware recursion
            # (NanBackfill.FFILL on non-finite outer products) handle leading
            # and mid-stream missing observations correctly: the correlation
            # for each asset is estimated only over its valid window, and the
            # zero-variance guard on the diagonal still protects against
            # degenerate columns. See CHANGELOG entry for 2026-04-16 commit
            # 937ba7c "align ewma with qis" for compute_ewm_covar's current
            # NaN-handling contract.
            gl = set_group_loadings(group_data=asset_clusters)
            result = solve_group_lasso_cvx_problem(
                x=x_np, y=y_np, group_loadings=gl.to_numpy(),
                valid_mask=valid_mask,
                reg_lambda=self.reg_lambda, span=eff_span,
                verbose=verbose, solver=self.solver,
                nonneg=self.nonneg,
                factors_beta_loading_signs=signs_np,
                factors_beta_prior=prior_np,
                group_penalty=self.group_penalty,
                l1_weight=self.l1_weight,
            )

        elif self.model_type == LassoModelType.GROUP_LASSO_CLUSTERS:
            gl = set_group_loadings(group_data=asset_clusters)
            result = solve_group_lasso_cvx_problem(
                x=x_np, y=y_np, group_loadings=gl.to_numpy(),
                valid_mask=valid_mask,
                reg_lambda=self.reg_lambda, span=eff_span,
                verbose=verbose, solver=self.solver,
                nonneg=self.nonneg,
                factors_beta_loading_signs=signs_np,
                factors_beta_prior=prior_np,
                group_penalty=self.group_penalty,
                l1_weight=self.l1_weight,
            )
        else:
            raise NotImplementedError(f"Unsupported model_type: {self.model_type}")

        # Zero out betas for variables with insufficient history
        est_beta = result.estimated_beta
        short_assets: Optional[pd.Index] = None
        if self.warmup_period is not None:
            n_valid = np.count_nonzero(~np.isnan(y.to_numpy()), axis=0)
            short = n_valid < self.warmup_period
            if np.any(short):
                est_beta[short, :] = 0.0
                for attr in ('alpha', 'ss_total', 'ss_res', 'r2'):
                    getattr(result, attr)[short] = np.nan
                # Capture for use below — the cluster-assignment step
                # at the end of fit() needs to drop these same assets
                # so clusters_, coef_, and per-asset diagnostics stay
                # mutually consistent. Without this the zeroed-beta
                # assets would still receive spurious singleton cluster
                # labels that inflate downstream n_clusters and pollute
                # cluster-based risk attribution / regime diagnostics.
                short_assets = y.columns[short]

        # Store fitted state (trailing underscore convention)
        self.x_ = x
        self.y_ = y
        self.valid_mask_ = valid_mask
        self.effective_span_ = eff_span
        self.coef_ = pd.DataFrame(
            est_beta, index=y.columns, columns=x.columns,
        )
        # intercept_ : preserved from v0.3.3 — the raw solver output, namely
        # the EWMA-weighted residual mean on the demeaned data. This is the
        # mechanical artefact of fitting a no-intercept model on centered
        # inputs (see :class:`LassoEstimationResult` docstring on ``alpha``).
        # It is NOT the regression intercept in the original
        # ``y = α + Xβ + ε`` representation; for span=None it is identically
        # zero by construction. Kept under this name for back-compat with
        # any analytics that read ``model.intercept_``.
        self.intercept_ = pd.Series(
            result.alpha, index=y.columns, name='intercept',
        )
        # alpha_const_ : the economic intercept α — what users typically
        # mean by "alpha" when decomposing returns into ``α + Xβ + ε``.
        #
        # Reconstructed from the same weighting that produced β: for
        # ``span=None`` (uniform weights) this is the sample-mean
        # reconstruction ``α = ȳ - x̄·β``; for ``span=integer`` it uses
        # EWMA-weighted means with the same weights factorlasso applies in
        # the loss function. This guarantees the (α, β) pair is
        # internally consistent — both are estimators on the same weighted
        # objective. Using sample means with EWMA-weighted β would mix two
        # different estimators and the resulting α would not be the
        # weighted-residual-mean that pairs with β.
        #
        # For ``span=None`` and unconstrained coefficients this equals
        # the OLS intercept exactly.
        x_arr = x.to_numpy(dtype=float)
        y_arr = y.to_numpy(dtype=float)
        T_full, n_x_full = x_arr.shape
        n_y_full = y_arr.shape[1]
        # Per-response valid mask of full (pre-demean) length T_full.
        # NaN in y[:, j] or all-NaN in x[t] makes row invalid for response j.
        y_valid = (~np.isnan(y_arr)).astype(float)
        x_row_valid = (~np.isnan(x_arr).all(axis=1)).astype(float)
        valid_full = y_valid * x_row_valid[:, None]
        # Weights aligned with the original T_full rows. Match factorlasso's
        # loss function: w_t² = (1 - 2/(span+1))^(T-1-t) for EWMA, uniform
        # for span=None.
        if eff_span is None:
            w_sq_full = np.ones(T_full)
        else:
            lam = 1.0 - 2.0 / (float(eff_span) + 1.0)
            w_sq_full = lam ** np.arange(T_full - 1, -1, -1)
        x_safe = np.nan_to_num(x_arr)
        y_safe = np.nan_to_num(y_arr)
        x_means = np.zeros((n_y_full, n_x_full))
        y_means = np.zeros(n_y_full)
        for j in range(n_y_full):
            w_j = w_sq_full * valid_full[:, j]
            tot = w_j.sum()
            if tot > 0.0:
                w_j_norm = w_j / tot
                x_means[j] = w_j_norm @ x_safe
                y_means[j] = w_j_norm @ y_safe[:, j]
            else:
                x_means[j] = np.nan
                y_means[j] = np.nan
        beta_arr = np.where(np.isnan(est_beta), 0.0, est_beta)
        alpha_const_arr = y_means - np.einsum('ij,ij->i', beta_arr, x_means)
        alpha_const_ser = pd.Series(
            alpha_const_arr, index=y.columns, name='alpha_const',
        )
        if short_assets is not None:
            alpha_const_ser.loc[short_assets] = np.nan
        self.alpha_const_ = alpha_const_ser
        self.estimation_result_ = result
        # asset_clusters is already populated by the upstream dispatch (HCGL
        # output for GROUP_LASSO_CLUSTERS, group_data for GROUP_LASSO, None
        # for plain LASSO). Filter out cluster labels for ghost assets —
        # assets whose betas were zeroed above because they had fewer than
        # ``warmup_period`` valid observations. Dropping them here keeps
        # ``clusters_`` consistent with ``coef_`` (zeroed) and per-asset
        # diagnostics (NaN), so downstream consumers that count or analyse
        # clusters see only assets that actually contributed to the fit.
        # Without this, pre-launch / short-history assets receive
        # placeholder singleton labels that inflate ``n_clusters`` in early
        # history (observed: 83 raw vs 31 real on a 160-asset multi-asset
        # universe at 2002-12-31).
        if asset_clusters is not None and short_assets is not None and len(short_assets) > 0:
            asset_clusters = asset_clusters.drop(short_assets, errors='ignore')
        self.clusters_ = asset_clusters
        self.linkage_ = linkage
        self.cutoff_ = cutoff
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Predict response values.

        Paper convention: Ŷ_t = α + β X_t.
        Code (row-major): Ŷ = X @ β' + α.

        Parameters
        ----------
        x : pd.DataFrame, shape (T, M)
            Regressor data with columns matching ``fit()``.

        Returns
        -------
        pd.DataFrame, shape (T, N)
        """
        if self.coef_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        # Paper: Y_t = α + β X_t.  Row-major equivalent: Y = X @ β' + α
        y_hat = x[self.coef_.columns] @ self.coef_.T
        if self.intercept_ is not None:
            y_hat = y_hat + self.intercept_.values
        return y_hat

    def score(self, x: pd.DataFrame, y: pd.DataFrame) -> float:
        """
        Mean R² across response variables.

        Parameters
        ----------
        x : pd.DataFrame, shape (T, M)
        y : pd.DataFrame, shape (T, N)

        Returns
        -------
        float
            Mean R² (higher is better).
        """
        y_hat = self.predict(x)
        ss_res = ((y - y_hat) ** 2).sum(axis=0)
        ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
        r2 = 1.0 - ss_res / ss_tot.replace(0, np.nan)
        return float(r2.mean())
