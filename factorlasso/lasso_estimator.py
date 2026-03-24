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
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import cvxpy as cvx
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as spc

from factorlasso.ewm_utils import (
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
        EWMA-weighted mean residual per response variable.
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
    """In-sample fit diagnostics from solver weights."""
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
    t: int, n_y: int, span: Optional[int], valid_mask: np.ndarray
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

    if n_y > 1 and valid_mask.ndim == 2:
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
    x: pd.DataFrame,
    y: pd.DataFrame,
    span: Optional[int] = None,
    demean: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare numpy arrays from regressor/response DataFrames with NaN masking.

    Parameters
    ----------
    x : pd.DataFrame, shape (T, M)
        Regressor data.  May have all-NaN rows.
    y : pd.DataFrame, shape (T, N)
        Response data.  May contain NaNs (different history lengths).
    span : int, optional
        EWMA span for demeaning.  ``None`` uses simple mean.
    demean : bool, default True
        If True, subtract (rolling) mean before estimation.

    Returns
    -------
    x_np : np.ndarray, shape (T', M)
    y_np : np.ndarray, shape (T', N)
    valid_mask : np.ndarray, shape (T', N)
        ``T' = T − 1`` when EWMA demeaning is used.
    """
    assert x.index.equals(y.index), (
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
    span: Optional[int] = None,
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
    span : int, optional
        EWMA span for observation weighting.
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
    span: Optional[int] = None,
    nonneg: bool = False,
    verbose: bool = False,
    solver: str = 'CLARABEL',
    factors_beta_loading_signs: Optional[np.ndarray] = None,
    factors_beta_prior: Optional[np.ndarray] = None,
) -> LassoEstimationResult:
    r"""
    Group LASSO multi-output regression via CVXPY.

    Minimises

    .. math::

        \frac{1}{T}\|W \odot (X\beta^\top - Y)\|_F^2
        + \sum_g \lambda\sqrt{|g|/G}\,\|\beta_{g,:} - \beta_{0,g,:}\|_2

    where *g* indexes groups of response variables (rows of β).

    Parameters
    ----------
    x : np.ndarray, shape (T, M)
    y : np.ndarray, shape (T, N)
    group_loadings : np.ndarray, shape (N, G)
        Binary group membership matrix.
    valid_mask, reg_lambda, span, nonneg, verbose, solver,
    factors_beta_loading_signs, factors_beta_prior
        See :func:`solve_lasso_cvx_problem`.

    Returns
    -------
    LassoEstimationResult
    """
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

    # Group penalty
    masks = [
        np.isclose(group_loadings[:, g], 1.0) for g in range(n_groups)
    ]
    penalty = cvx.sum([
        reg_lambda * np.sqrt(np.sum(m) / n_groups)
        * cvx.sum(cvx.norm2(beta[m, :] - prior[m, :], axis=1))
        for m in masks
    ])

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
# Clustering
# ═══════════════════════════════════════════════════════════════════════

def compute_clusters_from_corr_matrix(
    corr_matrix: pd.DataFrame,
) -> Tuple[pd.Series, np.ndarray, float]:
    """
    Hierarchical clustering from a correlation matrix (Ward's method).

    Converts correlation to distance ``(1 − corr)``, applies Ward's
    agglomerative clustering, and cuts the dendrogram at 50 % of the
    maximum pairwise distance.

    Parameters
    ----------
    corr_matrix : pd.DataFrame, shape (N, N)
        Square correlation matrix.

    Returns
    -------
    clusters : pd.Series
        Cluster labels (1-indexed) for each column.
    linkage : np.ndarray
        Scipy linkage matrix.
    cutoff : float
        Distance threshold used for cutting.
    """
    corr_matrix = corr_matrix.fillna(0.0)
    pdist = spc.distance.pdist(1.0 - corr_matrix.to_numpy())
    linkage = spc.linkage(pdist, method='ward')
    cutoff = 0.5 * np.max(pdist)
    idx = spc.fcluster(linkage, cutoff, 'distance')
    clusters = pd.Series(idx, index=corr_matrix.columns)
    return clusters, linkage, cutoff


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
    span : int, optional
        EWMA span for observation weighting.
    group_data : pd.Series, optional
        Group labels (required for ``GROUP_LASSO``).
    factors_beta_loading_signs : pd.DataFrame, optional
    factors_beta_prior : pd.DataFrame, optional
    demean : bool, default True
    solver : str, default 'CLARABEL'
    warmup_period : int, default 12

    Attributes (fitted, set by ``fit()``)
    --------------------------------------
    coef_ : pd.DataFrame, shape (N, M)
        Estimated factor loadings β.
    intercept_ : pd.Series, shape (N,)
        Estimated intercept α.
    estimation_result_ : LassoEstimationResult
        Full diagnostics (alpha, ss_total, ss_res, r2).
    clusters_ : pd.Series or None
        Cluster labels (HCGL only).
    linkage_ : np.ndarray or None
        Scipy linkage matrix (HCGL only).
    cutoff_ : float or None
        Dendrogram cut distance (HCGL only).

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
    span: Optional[int] = None
    span_freq_dict: Optional[Dict[str, int]] = None
    demean: bool = True
    solver: str = 'CLARABEL'
    warmup_period: Optional[int] = 12
    nonneg: bool = False
    factors_beta_loading_signs: Optional[pd.DataFrame] = None
    factors_beta_prior: Optional[pd.DataFrame] = None

    # ── Fitted state (set by fit(), trailing underscore) ──────────────
    x_: Optional[pd.DataFrame] = None
    y_: Optional[pd.DataFrame] = None
    coef_: Optional[pd.DataFrame] = None
    intercept_: Optional[pd.Series] = None
    estimation_result_: Optional[LassoEstimationResult] = None
    clusters_: Optional[pd.Series] = None
    linkage_: Optional[np.ndarray] = None
    cutoff_: Optional[float] = None
    valid_mask_: Optional[np.ndarray] = None
    effective_span_: Optional[int] = None

    def __post_init__(self):
        if self.model_type == LassoModelType.GROUP_LASSO and self.group_data is None:
            raise ValueError(
                "group_data must be provided for model_type=GROUP_LASSO"
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

    # ── Core API ─────────────────────────────────────────────────────

    def copy(self, kwargs: Optional[Dict] = None) -> LassoModel:
        """Create a copy, optionally overriding parameters."""
        this = asdict(self).copy()
        if kwargs is not None:
            this.update(kwargs)
        return LassoModel(**this)

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        verbose: bool = False,
        span: Optional[float] = None,
    ) -> LassoModel:
        """
        Estimate model: Y_t = α + β X_t + ε_t.

        Parameters
        ----------
        x : pd.DataFrame, shape (T, M)
            Regressor (factor) returns.
        y : pd.DataFrame, shape (T, N)
            Response (asset) returns.  May contain NaNs.
        verbose : bool, default False
            Print solver diagnostics.
        span : float, optional
            Override EWMA span for this call.

        Returns
        -------
        self
            Updated with ``coef_`` (N × M) and ``intercept_`` (N,).
        """
        eff_span = span or self.span
        x_np, y_np, valid_mask = get_x_y_np(
            x=x, y=y, span=eff_span, demean=self.demean
        )

        # Extract sign constraints and prior as numpy
        signs_np = None
        if self.factors_beta_loading_signs is not None:
            signs_np = self.factors_beta_loading_signs.loc[
                y.columns, x.columns
            ].to_numpy()

        prior_np = None
        if self.factors_beta_prior is not None:
            prior_np = self.factors_beta_prior.loc[
                y.columns, x.columns
            ].to_numpy()

        clusters = linkage = cutoff = None

        if self.model_type == LassoModelType.LASSO or y_np.shape[1] == 1:
            result = solve_lasso_cvx_problem(
                x=x_np, y=y_np, valid_mask=valid_mask,
                reg_lambda=self.reg_lambda, span=eff_span,
                verbose=verbose, solver=self.solver,
                nonneg=self.nonneg,
                factors_beta_loading_signs=signs_np,
                factors_beta_prior=prior_np,
            )

        elif self.model_type == LassoModelType.GROUP_LASSO:
            gl = set_group_loadings(group_data=self.group_data[y.columns])
            result = solve_group_lasso_cvx_problem(
                x=x_np, y=y_np, group_loadings=gl.to_numpy(),
                valid_mask=valid_mask,
                reg_lambda=self.reg_lambda, span=eff_span,
                verbose=verbose, solver=self.solver,
                nonneg=self.nonneg,
                factors_beta_loading_signs=signs_np,
                factors_beta_prior=prior_np,
            )

        elif self.model_type == LassoModelType.GROUP_LASSO_CLUSTERS:
            corr = compute_ewm_covar(
                a=y_np, span=eff_span, is_corr=True,
            )
            corr_df = pd.DataFrame(corr, columns=y.columns, index=y.columns)
            clusters, linkage, cutoff = compute_clusters_from_corr_matrix(corr_df)
            gl = set_group_loadings(group_data=clusters)
            result = solve_group_lasso_cvx_problem(
                x=x_np, y=y_np, group_loadings=gl.to_numpy(),
                valid_mask=valid_mask,
                reg_lambda=self.reg_lambda, span=eff_span,
                verbose=verbose, solver=self.solver,
                nonneg=self.nonneg,
                factors_beta_loading_signs=signs_np,
                factors_beta_prior=prior_np,
            )
        else:
            raise NotImplementedError(f"Unsupported model_type: {self.model_type}")

        # Zero out betas for variables with insufficient history
        est_beta = result.estimated_beta
        if self.warmup_period is not None:
            n_valid = np.count_nonzero(~np.isnan(y.to_numpy()), axis=0)
            short = n_valid < self.warmup_period
            if np.any(short):
                est_beta[short, :] = 0.0
                for attr in ('alpha', 'ss_total', 'ss_res', 'r2'):
                    getattr(result, attr)[short] = np.nan

        # Store fitted state (trailing underscore convention)
        self.x_ = x
        self.y_ = y
        self.valid_mask_ = valid_mask
        self.effective_span_ = eff_span
        self.coef_ = pd.DataFrame(
            est_beta, index=y.columns, columns=x.columns,
        )
        self.intercept_ = pd.Series(
            result.alpha, index=y.columns, name='intercept',
        )
        self.estimation_result_ = result
        self.clusters_ = clusters
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
