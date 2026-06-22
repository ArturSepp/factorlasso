"""
Tests for ``solve_group_lasso_path``: the DPP regularisation-path solver.

Coverage:
- Parity: a path solve is identical, to solver tolerance, to calling
  ``solve_group_lasso_cvx_problem`` once per grid point. Checked across the
  group-LASSO family (plain group, sign constraints, prior, adaptive row
  weights, the sparse-group L1 term, all combined, the ``yuan_lin`` group
  weighting, and the ``cluster_factor`` block mode).
- DPP: the parametrised problem assembled by the shared builder is
  disciplined-parametrised-convex, which is what makes the canonical form
  reusable across the grid.
- Result independence: each grid point yields its own loading array, not a
  shared reference to the last solve.
- Order preservation: the returned list aligns with ``reg_lambdas`` for an
  unsorted grid.
- Guards: empty grid and negative penalties raise; ``t < 5`` yields a list
  of NaN results aligned with the grid.
"""
from __future__ import annotations

import numpy as np
import pytest

from factorlasso.lasso_estimator import (
    _build_group_lasso_problem,
    solve_group_lasso_cvx_problem,
    solve_group_lasso_path,
)

# Parity tolerance. CLARABEL solves the parametrised and the rebuilt
# programmes to interior-point accuracy, so the two paths can differ by the
# solver tolerance; observed gaps are O(1e-6).
PARITY_ATOL = 1e-4


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

def _panel(seed: int = 0, T: int = 90, N: int = 12, M: int = 4):
    """Deterministic block-structured panel: each asset loads one factor."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, M))
    group_loadings = np.zeros((N, M))
    for k in range(N):
        group_loadings[k, k % M] = 1.0
    beta_true = rng.standard_normal((N, M)) * group_loadings
    y = x @ beta_true.T + 0.3 * rng.standard_normal((T, N))
    signs = np.sign(beta_true)
    signs[np.abs(beta_true) < 0.2] = np.nan          # NaN-permissive signs
    prior = 0.05 * group_loadings
    row_weights = 1.0 + 0.5 * rng.random(N)
    return dict(
        x=x, y=y, group_loadings=group_loadings,
        signs=signs, prior=prior, row_weights=row_weights, N=N, M=M,
    )


GRID = [float(v) for v in np.logspace(-4, -1, 6)]

# (label, extra kwargs) for the parity sweep
CONFIGS = [
    ("plain_group", {}),
    ("sign", lambda p: {"factors_beta_loading_signs": p["signs"]}),
    ("prior", lambda p: {"factors_beta_prior": p["prior"]}),
    ("adaptive_rows", lambda p: {"row_weights": p["row_weights"]}),
    ("sgl", {"l1_weight": 0.4}),
    ("yuan_lin", {"group_penalty": "yuan_lin"}),
    ("cluster_factor", {"block_mode": "cluster_factor"}),
    ("combined", lambda p: {
        "factors_beta_loading_signs": p["signs"],
        "factors_beta_prior": p["prior"],
        "row_weights": p["row_weights"],
        "l1_weight": 0.3,
    }),
]


# ═══════════════════════════════════════════════════════════════════════
# Parity
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("label,extra", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_path_matches_per_lambda_single_solve(label, extra):
    p = _panel()
    kw = extra(p) if callable(extra) else dict(extra)
    base = dict(x=p["x"], y=p["y"], group_loadings=p["group_loadings"], span=60.0)

    path = solve_group_lasso_path(reg_lambdas=GRID, **base, **kw)
    assert len(path) == len(GRID)

    for lam, pr in zip(GRID, path):
        sr = solve_group_lasso_cvx_problem(reg_lambda=lam, **base, **kw)
        assert np.allclose(
            pr.estimated_beta, sr.estimated_beta, atol=PARITY_ATOL, rtol=0.0
        ), f"{label}: loadings diverge at reg_lambda={lam:g}"
        # diagnostics travel with the result (r2 is per-response)
        assert np.array_equal(
            np.isfinite(np.atleast_1d(pr.r2)),
            np.isfinite(np.atleast_1d(sr.r2)),
        )


# ═══════════════════════════════════════════════════════════════════════
# DPP property
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("block_mode", ["row", "cluster_factor"])
@pytest.mark.parametrize("l1_weight", [0.0, 0.3])
def test_parametrised_problem_is_dpp(block_mode, l1_weight):
    import cvxpy as cvx

    p = _panel()
    reg_lambda = cvx.Parameter(nonneg=True)
    problem, _, _, _ = _build_group_lasso_problem(
        p["x"], p["y"], p["group_loadings"], reg_lambda,
        valid_mask=None, span=60.0, nonneg=False,
        factors_beta_loading_signs=p["signs"], factors_beta_prior=p["prior"],
        group_penalty="normalized", l1_weight=l1_weight,
        penalty_weights=None, row_weights=p["row_weights"],
        block_mode=block_mode, col_weights=None,
    )
    assert problem.is_dcp(dpp=True)


# ═══════════════════════════════════════════════════════════════════════
# Result independence and ordering
# ═══════════════════════════════════════════════════════════════════════

def test_results_are_independent_arrays():
    """Each grid point must own its loading array (no aliasing to the last
    solve via the shared cvxpy variable)."""
    p = _panel()
    grid = [1e-4, 1e-1]                                  # widely separated
    path = solve_group_lasso_path(
        x=p["x"], y=p["y"], group_loadings=p["group_loadings"],
        reg_lambdas=grid, span=60.0,
    )
    assert path[0].estimated_beta is not path[1].estimated_beta
    # heavier penalty must shrink the loadings overall
    assert np.nansum(np.abs(path[1].estimated_beta)) < \
        np.nansum(np.abs(path[0].estimated_beta))


def test_order_is_preserved_for_unsorted_grid():
    p = _panel()
    grid = [1e-2, 1e-4, 5e-2, 1e-3]
    path = solve_group_lasso_path(
        x=p["x"], y=p["y"], group_loadings=p["group_loadings"],
        reg_lambdas=grid, span=60.0,
    )
    for lam, pr in zip(grid, path):
        sr = solve_group_lasso_cvx_problem(
            x=p["x"], y=p["y"], group_loadings=p["group_loadings"],
            reg_lambda=lam, span=60.0,
        )
        assert np.allclose(
            pr.estimated_beta, sr.estimated_beta, atol=PARITY_ATOL, rtol=0.0
        )


# ═══════════════════════════════════════════════════════════════════════
# Guards
# ═══════════════════════════════════════════════════════════════════════

def test_empty_grid_raises():
    p = _panel()
    with pytest.raises(ValueError, match="non-empty"):
        solve_group_lasso_path(
            x=p["x"], y=p["y"], group_loadings=p["group_loadings"],
            reg_lambdas=[],
        )


def test_negative_lambda_raises():
    p = _panel()
    with pytest.raises(ValueError, match="non-negative"):
        solve_group_lasso_path(
            x=p["x"], y=p["y"], group_loadings=p["group_loadings"],
            reg_lambdas=[1e-3, -1.0],
        )


def test_insufficient_obs_returns_nan_list():
    p = _panel()
    short = solve_group_lasso_path(
        x=p["x"][:3], y=p["y"][:3], group_loadings=p["group_loadings"],
        reg_lambdas=GRID,
    )
    assert len(short) == len(GRID)
    assert all(np.all(np.isnan(r.estimated_beta)) for r in short)
