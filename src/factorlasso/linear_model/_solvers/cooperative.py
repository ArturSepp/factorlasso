"""Cooperative group LASSO solver: soft within-block sign coherence."""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import cvxpy as cvx
import numpy as np

from factorlasso.utils._ewm import _validate_span
from factorlasso.linear_model._types import LassoEstimationResult
from factorlasso.linear_model._solvers.common import (
    _compute_solver_diagnostics, _compute_solver_weights, _weighted_squared_loss, _clean_beta_prior,
    _derive_valid_mask_from_y, _nan_result, _solve_with_fallback,
)


def solve_cooperative_group_lasso_cvx_problem(
    x: np.ndarray,
    y: np.ndarray,
    group_loadings: np.ndarray,
    *,
    valid_mask: Optional[np.ndarray] = None,
    reg_lambda: float = 1e-8,
    span: Optional[float] = None,
    verbose: bool = False,
    solver: str = 'CLARABEL',
    solver_fallbacks: Optional[Sequence[str]] = None,
    factors_beta_prior: Optional[np.ndarray] = None,
    group_penalty: str = "normalized",
    l1_weight: float = 0.0,
    col_weights: Optional[np.ndarray] = None,
    loss_normalization: str = "sample",
) -> LassoEstimationResult:
    r"""cooperative-LASSO multi-output regression via CVXPY (soft sign coherence).

    Minimises

    .. math::

        \frac{1}{T}\|W \odot (X\beta^\top - Y)\|_F^2
        + (1 - \alpha)\,\lambda \sum_g w_g \sum_{j=1}^{M}
          \big(\|(\beta_{g,j} - \beta_{0,g,j})_+\|_2
             + \|(\beta_{g,j} - \beta_{0,g,j})_-\|_2\big)
        + \alpha\,\lambda \,\|\beta - \beta_0\|_1

    where :math:`(\cdot)_+` and :math:`(\cdot)_-` are the elementwise positive
    and negative parts and the block :math:`\beta_{g,j}` collects the loadings
    of cluster *g*'s assets on factor *j*. A sign-coherent block (all-positive
    or all-negative) pays only :math:`\|\beta_{g,j}\|_2`; a mixed-sign block
    pays strictly more, so within-block sign coherence is encouraged softly and
    the data may overrule it. This is the cooperative-LASSO penalty of Chiquet,
    Grandvalet & Charbonnier (2012), here in the cluster-by-factor block
    geometry. Signs are not imposed and there is no gate; contrast the hard,
    gated, pooled sign of FACTOR_CLUSTER_GROUP_LASSO. Solved via the positive /
    negative split :math:`\beta - \beta_0 = P - N`, :math:`P, N \ge 0`, which is
    DCP-clean.

    Parameters
    ----------
    x : np.ndarray, shape (T, M)
    y : np.ndarray, shape (T, N)
    group_loadings : np.ndarray, shape (N, G)
        Binary group membership matrix (one column per cluster).
    valid_mask, reg_lambda, span, verbose, solver, factors_beta_prior
        See :func:`solve_group_lasso_cvx_problem`.
    loss_normalization : {"sample", "weight_sum"}, default "sample"
        Observation-loss convention; see :func:`solve_lasso_cvx_problem`.
    group_penalty : {"normalized", "yuan_lin"}, default "normalized"
        Per-group weight convention; see :func:`solve_group_lasso_cvx_problem`.
    l1_weight : float, default 0.0
        Mixing weight alpha on an elementwise L1 term, giving a sparse
        cooperative-LASSO. 0.0 is the pure cooperative-LASSO.
    col_weights : np.ndarray, shape (G, M), optional
        Per-(cluster, factor) adaptive weights on each block penalty.

    Returns
    -------
    LassoEstimationResult
    """
    _validate_span(span)
    if group_penalty not in ("normalized", "yuan_lin"):
        raise ValueError(
            f"group_penalty must be 'normalized' or 'yuan_lin', got {group_penalty!r}"
        )
    if not (0.0 <= l1_weight <= 1.0):
        raise ValueError(f"l1_weight must lie in [0, 1], got {l1_weight!r}")
    assert y.ndim == 2 and x.ndim == 2 and group_loadings.ndim == 2
    assert x.shape[0] == y.shape[0] and y.shape[1] == group_loadings.shape[0]

    t, n_x = x.shape
    n_y = y.shape[1]
    n_groups = group_loadings.shape[1]
    if valid_mask is None:
        y, valid_mask = _derive_valid_mask_from_y(y)
    if t < 5:
        warnings.warn(f"insufficient observations for cooperative lasso: t={t}")
        return _nan_result(n_y, n_x)

    weights = _compute_solver_weights(t, n_y, span, valid_mask)
    prior = _clean_beta_prior(factors_beta_prior, n_y, n_x)

    # beta - prior = P - N, P, N >= 0  (positive / negative split)
    pos = cvx.Variable((n_y, n_x), nonneg=True)
    neg = cvx.Variable((n_y, n_x), nonneg=True)
    beta = prior + pos - neg

    fit = _weighted_squared_loss(x @ beta.T - y, weights, t, loss_normalization)

    def _weight(m: np.ndarray) -> float:
        g = np.sum(m)
        if group_penalty == "yuan_lin":
            return float(np.sqrt(g))
        return float(np.sqrt(g / n_groups))

    masks = [np.isclose(group_loadings[:, g], 1.0) for g in range(n_groups)]
    if col_weights is not None and col_weights.shape != (n_groups, n_x):
        raise ValueError(
            f"col_weights shape {col_weights.shape} != expected ({n_groups}, {n_x})"
        )
    coop_terms = []
    for gi, m in enumerate(masks):
        # length-M vectors of block L2 norms over the cluster's member rows,
        # one per factor, on the positive and the negative part separately.
        block = cvx.norm2(pos[m, :], axis=0) + cvx.norm2(neg[m, :], axis=0)
        if col_weights is not None:
            block = cvx.multiply(col_weights[gi], block)
        coop_terms.append(reg_lambda * _weight(m) * cvx.sum(block))
    coop_pen = cvx.sum(coop_terms)

    if l1_weight > 0.0:
        # P + N equals |beta - prior| elementwise at the optimum.
        l1_pen = reg_lambda * cvx.sum(pos + neg)
        penalty = (1.0 - l1_weight) * coop_pen + l1_weight * l1_pen
    else:
        penalty = coop_pen

    problem = cvx.Problem(cvx.Minimize(fit + penalty))
    _solve_with_fallback(problem, solver, solver_fallbacks, verbose=verbose)

    if pos.value is None or neg.value is None:
        warnings.warn("cooperative lasso problem not solved")
        return _nan_result(n_y, n_x)

    beta_val = prior + pos.value - neg.value
    alpha, ss_total, ss_res, r2 = _compute_solver_diagnostics(
        x, y, beta_val, weights
    )
    return LassoEstimationResult(
        estimated_beta=beta_val, alpha=alpha,
        ss_total=ss_total, ss_res=ss_res, r2=r2,
    )
