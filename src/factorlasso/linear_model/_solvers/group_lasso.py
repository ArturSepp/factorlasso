"""Group LASSO solvers: one penalty, and the DPP-parametrised regularisation path.

References
----------
Yuan, M., Lin, Y. (2006), "Model selection and estimation in regression
with grouped variables", *J. R. Statist. Soc. B*, 68(1), 49–67.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple, Union

import cvxpy as cvx
import numpy as np

from factorlasso.utils._ewm import _validate_span
from factorlasso.linear_model._types import LassoEstimationResult
from factorlasso.linear_model._solvers.common import (
    _compute_solver_diagnostics, _compute_solver_weights, _weighted_squared_loss, _clean_beta_prior,
    _derive_valid_mask_from_y, _build_sign_constraints, _nan_result, _solve_with_fallback,
    _build_loading_bound_constraints,
)


def _build_group_lasso_problem(
    x: np.ndarray,
    y: np.ndarray,
    group_loadings: np.ndarray,
    reg_lambda: Union[float, cvx.Parameter],
    *,
    valid_mask: Optional[np.ndarray],
    span: Optional[float],
    nonneg: bool,
    factors_beta_loading_signs: Optional[np.ndarray],
    factors_beta_prior: Optional[np.ndarray],
    group_penalty: str,
    l1_weight: float,
    penalty_weights: Optional[np.ndarray],
    row_weights: Optional[np.ndarray],
    block_mode: str,
    col_weights: Optional[np.ndarray],
    loss_normalization: str = "sample",
    beta_lower_bounds: Optional[np.ndarray] = None,
    beta_upper_bounds: Optional[np.ndarray] = None,
) -> Optional[Tuple[cvx.Problem, cvx.Variable, np.ndarray, np.ndarray]]:
    """assemble the sparse group-LASSO CVXPY problem.

    Shared by :func:`solve_group_lasso_cvx_problem` (a single
    ``reg_lambda``) and :func:`solve_group_lasso_path` (``reg_lambda`` a
    ``cvxpy.Parameter`` swept over a grid). ``reg_lambda`` enters only the
    penalty terms, so a non-negative parameter keeps the programme
    DPP-compliant and CVXPY reuses the canonical form across solves. The
    construction is otherwise identical for the float and parameter cases.

    Returns
    -------
    (problem, beta, weights, y) : Tuple or None
        The assembled problem, the loading variable, the observation
        weights, and the NaN-filled response matrix used for fit
        diagnostics. ``None`` signals ``t < 5``; the caller then returns a
        NaN result.
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
        return None

    # Variable and constraints
    if factors_beta_loading_signs is not None:
        beta = cvx.Variable((n_y, n_x))
        constraints = _build_sign_constraints(beta, factors_beta_loading_signs)
    else:
        beta = cvx.Variable((n_y, n_x), nonneg=nonneg)
        constraints = []

    constraints.extend(_build_loading_bound_constraints(
        beta, beta_lower_bounds, beta_upper_bounds))
    weights = _compute_solver_weights(t, n_y, span, valid_mask)
    prior = _clean_beta_prior(factors_beta_prior, n_y, n_x)

    # Fit term
    fit = _weighted_squared_loss(x @ beta.T - y, weights, t, loss_normalization)

    # Per-group weight. "normalized" (default) preserves the v0.2.2
    # behaviour √(|g|/G); "yuan_lin" uses the classical √|g|.
    def _weight(m: np.ndarray) -> float:
        g = np.sum(m)
        if group_penalty == "yuan_lin":
            return float(np.sqrt(g))
        return float(np.sqrt(g / n_groups))

    # Group penalty (L_{2,1} norm within each group, scaled by (1 - α)).
    # Per-asset row weights (Wang & Leng 2008 adaptive group lasso) are
    # applied as scalar multipliers on each row's L2 norm when supplied.
    # When row_weights=None, each row's multiplier is 1.0 — backward
    # compatible with the v0.3.8 pure group LASSO.
    if row_weights is not None:
        if row_weights.shape != (n_y,):
            raise ValueError(
                f"row_weights shape {row_weights.shape} != expected ({n_y},)"
            )
    masks = [
        np.isclose(group_loadings[:, g], 1.0) for g in range(n_groups)
    ]
    if block_mode not in ("row", "cluster_factor"):
        raise ValueError(
            f"block_mode must be 'row' or 'cluster_factor', got {block_mode!r}"
        )
    group_terms = []
    if block_mode == "row":
        for m in masks:
            # cvx.norm2(beta[m, :] - prior[m, :], axis=1) is a vector of L2
            # norms, one per asset in the group. With row_weights, each
            # element is scaled by its asset-specific weight before summing.
            norms = cvx.norm2(beta[m, :] - prior[m, :], axis=1)
            if row_weights is not None:
                norms = cvx.multiply(row_weights[m], norms)
            group_terms.append(reg_lambda * _weight(m) * cvx.sum(norms))
    else:
        # cluster_factor: the group of the norm is the cluster x factor
        # block. For cluster g, cvx.norm2(beta[m, :] - prior[m, :], axis=0)
        # is a length-M vector of L2 norms, one per factor, taken over the
        # cluster's member rows. Summing these M norms and scaling by the
        # cluster weight gives a penalty that selects whole cluster x factor
        # blocks in or out together. Unlike the row mode this couples assets
        # within a cluster, so the problem is NOT block-separable across
        # assets — do not refactor it into per-asset fits.
        if col_weights is not None:
            if col_weights.shape != (n_groups, n_x):
                raise ValueError(
                    f"col_weights shape {col_weights.shape} != expected "
                    f"({n_groups}, {n_x})"
                )
        for gi, m in enumerate(masks):
            col_norms = cvx.norm2(beta[m, :] - prior[m, :], axis=0)
            if col_weights is not None:
                col_norms = cvx.multiply(col_weights[gi], col_norms)
            group_terms.append(reg_lambda * _weight(m) * cvx.sum(col_norms))
    group_pen = cvx.sum(group_terms)

    # Elementwise L1 penalty (scaled by α). Shrinks toward the same
    # prior used by the group term so at α=1 the problem is consistent
    # with plain LASSO centred on β₀. At α=0 this term vanishes and
    # the problem reduces exactly to the v0.3.1 pure group LASSO.
    if l1_weight > 0.0:
        if penalty_weights is not None:
            if penalty_weights.shape != (n_y, n_x):
                raise ValueError(
                    f"penalty_weights shape {penalty_weights.shape} "
                    f"!= expected ({n_y}, {n_x})"
                )
            l1_pen = reg_lambda * cvx.sum(
                cvx.multiply(penalty_weights, cvx.abs(beta - prior))
            )
        else:
            l1_pen = reg_lambda * cvx.sum(cvx.abs(beta - prior))
        penalty = (1.0 - l1_weight) * group_pen + l1_weight * l1_pen
    else:
        penalty = group_pen

    problem = cvx.Problem(cvx.Minimize(fit + penalty), constraints) \
        if constraints else cvx.Problem(cvx.Minimize(fit + penalty))
    return problem, beta, weights, y


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
    solver_fallbacks: Optional[Sequence[str]] = None,
    factors_beta_loading_signs: Optional[np.ndarray] = None,
    factors_beta_prior: Optional[np.ndarray] = None,
    group_penalty: str = "normalized",
    l1_weight: float = 0.0,
    penalty_weights: Optional[np.ndarray] = None,
    row_weights: Optional[np.ndarray] = None,
    block_mode: str = "row",
    col_weights: Optional[np.ndarray] = None,
    loss_normalization: str = "sample",
    beta_lower_bounds: Optional[np.ndarray] = None,
    beta_upper_bounds: Optional[np.ndarray] = None,
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
    submatrix — each response's loading vector is shrunk by an L2 norm, so
    a response is removed as a whole or kept with a dense row; the group
    enters through the weight ``w_g`` only, and rows within a group are
    not coupled (``block_mode="cluster_factor"`` below is the geometry
    that couples them). The optional L1 term drives elementwise sparsity
    on top, zeroing
    individual assets whose loadings are noisy even within an "active"
    group — the Simon–Friedman–Hastie–Tibshirani (2013) Sparse Group
    LASSO formulation.

    At ``l1_weight=0.0`` (default) the problem reduces to the previous
    pure group LASSO and is numerically identical to v0.3.1.

    For a grid of ``reg_lambda`` values with every other argument held
    fixed, :func:`solve_group_lasso_path` assembles the problem once with
    ``reg_lambda`` as a ``cvxpy.Parameter`` and reuses the canonical form
    across the grid, which is faster than calling this function per grid
    point.

    Parameters
    ----------
    x : np.ndarray, shape (T, M)
    y : np.ndarray, shape (T, N)
    group_loadings : np.ndarray, shape (N, G)
        Binary group membership matrix.
    valid_mask, reg_lambda, span, nonneg, verbose, solver,
    factors_beta_loading_signs, factors_beta_prior
        See :func:`solve_lasso_cvx_problem`.
    loss_normalization : {"sample", "weight_sum"}, default "sample"
        Observation-loss convention; see :func:`solve_lasso_cvx_problem`.
    group_penalty : {"normalized", "yuan_lin"}, default "normalized"
        Per-group weighting convention:

        - ``"normalized"``: ``w_g = √(|g|/G)``. A heuristic cluster-size
          scaling that moderates the relative influence of large
          clusters and adjusts the effective regularisation scale for
          the data-driven group count G. It is not invariant to
          arbitrary refinements of the partition. This is the
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
    block_mode : {"row", "cluster_factor"}, default "row"
        Geometry of the group L2 norm. ``"row"`` (HCGL) takes the norm
        over each asset's M factor loadings, summed across assets; the
        cluster enters only through the weight ``w_g``, and the programme
        is block-separable across assets. ``"cluster_factor"`` (FCGL)
        takes the norm over each cluster-by-factor block,

        .. math::

            (1 - \alpha)\,\lambda \sum_g w_g \sum_{j=1}^{M}
              \|\beta_{g, j} - \beta_{0, g, j}\|_2,

        where the block collects the loadings of cluster *g*'s assets on
        factor *j*. In ``"cluster_factor"`` mode the cluster is the group
        of the norm itself, so a whole cluster-by-factor block enters or
        leaves the model together; the programme is not block-separable
        across assets and is solved as one coupled cone programme.
    penalty_weights : np.ndarray, shape (N, M), optional
        Per-cell adaptive weights multiplying the L1 term, from the
        adaptive-reweighting layer. ``None`` applies unit weights.
    row_weights : np.ndarray, shape (N,), optional
        Per-asset adaptive weights multiplying each row L2 norm in
        ``"row"`` mode. ``None`` applies unit weights.
    col_weights : np.ndarray, shape (G, M), optional
        Per-(cluster, factor) adaptive weights multiplying each block L2
        norm in ``"cluster_factor"`` mode. ``None`` applies unit weights.

    beta_lower_bounds, beta_upper_bounds : ndarray, shape (N, M), optional
        Individual coefficient bounds; NaN cells add no constraint. These arrays
        are already resolved by the caller, including any sign precedence.

    Returns
    -------
    LassoEstimationResult
    """
    built = _build_group_lasso_problem(
        x, y, group_loadings, reg_lambda,
        valid_mask=valid_mask, span=span, nonneg=nonneg,
        factors_beta_loading_signs=factors_beta_loading_signs,
        factors_beta_prior=factors_beta_prior, group_penalty=group_penalty,
        l1_weight=l1_weight, penalty_weights=penalty_weights,
        row_weights=row_weights, block_mode=block_mode, col_weights=col_weights,
        loss_normalization=loss_normalization,
        beta_lower_bounds=beta_lower_bounds, beta_upper_bounds=beta_upper_bounds,
    )
    if built is None:
        return _nan_result(y.shape[1], x.shape[1])
    problem, beta, weights, y_used = built
    _solve_with_fallback(problem, solver, solver_fallbacks, verbose=verbose)

    if beta.value is None:
        warnings.warn("group lasso problem not solved")
        return _nan_result(y.shape[1], x.shape[1])

    alpha, ss_total, ss_res, r2 = _compute_solver_diagnostics(
        x, y_used, beta.value, weights
    )
    return LassoEstimationResult(
        estimated_beta=beta.value, alpha=alpha,
        ss_total=ss_total, ss_res=ss_res, r2=r2,
    )


def solve_group_lasso_path(
    x: np.ndarray,
    y: np.ndarray,
    group_loadings: np.ndarray,
    reg_lambdas: Sequence[float],
    valid_mask: Optional[np.ndarray] = None,
    span: Optional[float] = None,
    nonneg: bool = False,
    verbose: bool = False,
    solver: str = 'CLARABEL',
    solver_fallbacks: Optional[Sequence[str]] = None,
    factors_beta_loading_signs: Optional[np.ndarray] = None,
    factors_beta_prior: Optional[np.ndarray] = None,
    group_penalty: str = "normalized",
    l1_weight: float = 0.0,
    penalty_weights: Optional[np.ndarray] = None,
    row_weights: Optional[np.ndarray] = None,
    block_mode: str = "row",
    col_weights: Optional[np.ndarray] = None,
    loss_normalization: str = "sample",
    beta_lower_bounds: Optional[np.ndarray] = None,
    beta_upper_bounds: Optional[np.ndarray] = None,
) -> List[LassoEstimationResult]:
    r"""Group-LASSO over a regularisation path, reusing one canonical form.

    Solves :func:`solve_group_lasso_cvx_problem` at each value in
    ``reg_lambdas`` while every other argument is held fixed. The penalty
    weight ``reg_lambda`` is supplied to CVXPY as a
    ``cvxpy.Parameter(nonneg=True)``, so the disciplined-parametrised
    programme is canonicalised once and the compiled conic form is reused
    on every solve (a warm start of the problem structure, not of the
    solver iterate). Recompilation, which is otherwise repeated per grid
    point, is paid a single time.

    Use this for any workflow that solves the same panel over a grid of
    ``reg_lambda`` with the sign matrix and the adaptive weights fixed:
    cross-validated or BIC-based ``reg_lambda`` selection, rolling
    backtests that re-select ``reg_lambda`` per date, regularisation-path
    figures, and threshold or cutoff sensitivity sweeps whose inner loop
    is over ``reg_lambda``. For a single ``reg_lambda`` there is no benefit
    over one :func:`solve_group_lasso_cvx_problem` call. The speed-up is
    bounded by the share of per-solve time spent in canonicalisation
    rather than in the solver itself.

    The sign matrix, the prior, and the adaptive weights must not depend on
    ``reg_lambda`` (they do not in this package: signs and weights are
    derived from univariate slopes, which are ``reg_lambda``-independent).
    The path covers the group-LASSO family (group LASSO, HCGL via
    ``block_mode="row"``, FCGL via ``block_mode="cluster_factor"``, the
    sparse-group L1 term, sign constraints, the prior, and adaptive
    reweighting). The cooperative and UniLasso estimators have their own
    solvers and are not handled here.

    Parameters
    ----------
    reg_lambdas : sequence of float
        Non-negative regularisation weights to solve, in any order. The
        returned list is aligned with this sequence.
    x, y, group_loadings, valid_mask, span, nonneg, verbose, solver,
    factors_beta_loading_signs, factors_beta_prior, group_penalty,
    l1_weight, penalty_weights, row_weights, block_mode, col_weights, loss_normalization
        As in :func:`solve_group_lasso_cvx_problem`.

    beta_lower_bounds, beta_upper_bounds : ndarray, shape (N, M), optional
        Individual coefficient bounds; NaN cells add no constraint. These arrays
        are already resolved by the caller, including any sign precedence.

    Returns
    -------
    list of LassoEstimationResult
        One result per entry in ``reg_lambdas``, in the same order. Each is
        identical, to solver tolerance, to the result of calling
        :func:`solve_group_lasso_cvx_problem` with that ``reg_lambda``.

    Raises
    ------
    ValueError
        If ``reg_lambdas`` is empty or contains a negative value.
    """
    lambdas = [float(lv) for lv in reg_lambdas]
    if len(lambdas) == 0:
        raise ValueError("reg_lambdas must be non-empty")
    if min(lambdas) < 0.0:
        raise ValueError(
            f"reg_lambdas must be non-negative, got min {min(lambdas)!r}"
        )

    n_y, n_x = y.shape[1], x.shape[1]
    reg_lambda = cvx.Parameter(nonneg=True)
    built = _build_group_lasso_problem(
        x, y, group_loadings, reg_lambda,
        valid_mask=valid_mask, span=span, nonneg=nonneg,
        factors_beta_loading_signs=factors_beta_loading_signs,
        factors_beta_prior=factors_beta_prior, group_penalty=group_penalty,
        l1_weight=l1_weight, penalty_weights=penalty_weights,
        row_weights=row_weights, block_mode=block_mode, col_weights=col_weights,
        loss_normalization=loss_normalization,
        beta_lower_bounds=beta_lower_bounds, beta_upper_bounds=beta_upper_bounds,
    )
    if built is None:
        return [_nan_result(n_y, n_x) for _ in lambdas]
    problem, beta, weights, y_used = built

    if not problem.is_dcp(dpp=True):
        warnings.warn(
            "group lasso path problem is not DPP; the canonical form will be "
            "rebuilt on each solve and the warm-start speed-up is lost"
        )

    results: List[LassoEstimationResult] = []
    for lv in lambdas:
        reg_lambda.value = lv
        _solve_with_fallback(problem, solver, solver_fallbacks, verbose=verbose, warm_start=True)
        if beta.value is None:
            warnings.warn(
                f"group lasso path problem not solved at reg_lambda={lv!r}"
            )
            results.append(_nan_result(n_y, n_x))
            continue
        alpha, ss_total, ss_res, r2 = _compute_solver_diagnostics(
            x, y_used, beta.value, weights
        )
        results.append(LassoEstimationResult(
            estimated_beta=beta.value.copy(), alpha=alpha,
            ss_total=ss_total, ss_res=ss_res, r2=r2,
        ))
    return results
