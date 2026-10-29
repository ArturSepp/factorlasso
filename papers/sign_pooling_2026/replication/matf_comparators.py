"""Opt-in research comparators for the ETF/MATF known-truth pilot.

The squared wrong-direction hinge leaves permitted magnitudes free. A separate
zero-cell quadratic can soften exclusion; it is never implicit. Conditional
gates apply the package's full weighted regression to a response-cluster mean,
with date-level HC1 scores (zero HAC lags). This is experimental regularization,
not a calibrated test after learning clusters or under serial dependence.

The existing HCGL/FCGL objective builder and public ridge/WLS implementations
are reused. Nothing changes LassoModel configuration or production defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
import warnings

import cvxpy as cp
import numpy as np

from factorlasso import LassoModel, LassoModelType, compute_wls_hac_statistics
from factorlasso.priors._signs import _pooled_sign_statistics

if __package__:
    from . import french_estimators as F
    from .cvxpy_shape_compat import safe_cvxpy_sum_shape
else:
    import french_estimators as F
    from cvxpy_shape_compat import safe_cvxpy_sum_shape


@dataclass
class Training:
    """Training-only transforms and explicit mean-one observation weights."""

    state: F.State
    weighted: F.State
    weights: np.ndarray
    span: float | None


def prepare(x, y, span=None):
    """Standardize within training using the same weights as the fitted loss."""
    state = F.prepare(x, y)  # Also validates chronology, finiteness and support.
    n = len(x)
    if span is not None and (not np.isfinite(span) or span <= 1):
        raise ValueError("Research EW span must exceed one or be None")
    w = np.ones(n) if span is None else (1 - 2 / (span + 1)) ** np.arange(n - 1, -1, -1)
    w = n * w / w.sum()
    xx, yy = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
    xm, ym = np.average(xx, axis=0, weights=w), np.average(yy, axis=0, weights=w)
    xs = np.sqrt(np.average((xx - xm) ** 2, axis=0, weights=w))
    ys = np.sqrt(np.average((yy - ym) ** 2, axis=0, weights=w))
    if min(xs.min(), ys.min()) <= 1e-12:
        raise ValueError("Degenerate weighted training scale")
    state = replace(
        state, x=(xx - xm) / xs, y=(yy - ym) / ys, x_mean=xm, x_sd=xs, y_mean=ym, y_sd=ys
    )
    weighted = replace(state, x=state.x * np.sqrt(w[:, None]), y=state.y * np.sqrt(w[:, None]))
    return Training(state, weighted, w, span)


def specification(training, family="hcgl", cut=0.6, labels=None):
    """One common penalty partition/scale for every gate intervention."""
    if family not in ("hcgl", "fcgl"):
        raise ValueError("Sparse comparator family must be hcgl or fcgl")
    labels = (
        training.weighted.labels(cut) if labels is None else F.canonical_labels(np.asarray(labels))
    )
    if labels.shape != (training.state.y.shape[1],):
        raise ValueError("Penalty labels must cover all responses")
    weights = np.array([np.sqrt(np.sum(labels == g) / len(np.unique(labels))) for g in labels])
    spec = F.Spec(
        "row" if family == "hcgl" else "block",
        labels.copy(),
        weights,
        np.full((len(labels), training.state.x.shape[1]), np.nan),
        0.0,
    )
    spec.scale = F.kkt_scale(training.weighted, spec)
    return spec


def gate(training, labels, kind="marginal", threshold=2.0):
    """Return pooled signs and evidence; zero means the threshold did not pass.

    Conditional HC1 uses T/(T-K-1), including the fitted intercept, with positive
    date weights. Marginal date variance retains package Kish/(Kish-1). Both
    retain within-date response covariance by pooling scores before squaring.
    No HAC, selection correction, missing-data imputation or rank fallback.
    """
    if kind not in ("marginal", "conditional") or threshold < 0 or not np.isfinite(threshold):
        raise ValueError("Invalid gate kind or finite nonnegative threshold")
    s = training.state
    labels = np.asarray(labels)
    if labels.shape != (s.y.shape[1],):
        raise ValueError("Gate labels must cover every response")
    signs, slopes, errors, stats = [np.zeros((s.y.shape[1], s.x.shape[1])) for _ in range(4)]
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        if kind == "marginal":
            result = _pooled_sign_statistics(s.x, s.y[:, members], training.span, "date")
            b, se, t = result["slopes"], result["standard_errors"], result["t_stats"]
        else:
            # Mean response exactly represents a common slope with date-cluster scores.
            result = compute_wls_hac_statistics(
                s.x,
                s.y[:, members].mean(1),
                weights=training.weights.copy(),
                hac_lags=0,
                fit_intercept=True,
                return_covariance=True,
            )
            if result.status != "ok":
                raise ValueError("Conditional gate is unidentified: " + result.status)
            b, se = result.coefficients[1:], result.standard_error[1:]
            t = np.divide(b, se, out=np.zeros_like(b), where=se > 0)
            t[(se == 0) & (b != 0)] = np.sign(b[(se == 0) & (b != 0)]) * np.inf
        signs[members] = np.where(abs(t) >= threshold, np.sign(b), 0.0)
        slopes[members], errors[members], stats[members] = b, se, t
    return dict(
        signs=signs,
        slopes=slopes,
        standard_errors=errors,
        t_stats=stats,
        kind=kind,
        threshold=float(threshold),
        labels=labels.copy(),
    )


def hard_signs(signs, policy):
    """Convert explicit gate semantics without changing the penalty groups."""
    signs = np.asarray(signs, dtype=float).copy()
    if not np.isin(signs, [-1, 0, 1]).all():
        raise ValueError("Gate directions must be -1, 0, +1")
    if policy == "release":
        signs[signs == 0] = np.nan
    elif policy == "zero_only":
        signs[signs != 0] = np.nan
    elif policy != "exclude":
        raise ValueError("Unknown hard-gate policy")
    return signs


def soft_loss_gradient(beta, signs, strength, zero_strength):
    """Return the squared directional/zero penalty and its exact smooth gradient."""
    signed = np.where(np.isfinite(signs), signs, 0.0)
    wrong = np.maximum(-signed * beta, 0.0)
    zero = np.isfinite(signs) & (signs == 0)
    loss = strength * np.sum(wrong**2) + zero_strength * np.sum(beta[zero] ** 2)
    gradient = -2 * strength * signed * wrong + 2 * zero_strength * zero * beta
    return float(loss), gradient


def audit(training, spec, beta, lam, soft_signs, strength, zero_strength):
    """Evaluate objective, feasibility and a proximal-gradient stationarity residual."""
    x, y = training.weighted.x, training.weighted.y
    residual = x @ beta.T - y
    added, gsoft = soft_loss_gradient(beta, soft_signs, strength, zero_strength)
    gradient = 2 * residual.T @ x / len(x) + gsoft
    lipschitz = max(1.0, 2 * np.linalg.norm(x, 2) ** 2 / len(x) + 2 * max(strength, zero_strength))
    mapping = lipschitz * (beta - F.proximal(beta - gradient / lipschitz, spec, lam / lipschitz))
    return dict(
        objective=float(np.sum(residual**2) / len(x) + lam * F.penalty(beta, spec) + added),
        kkt_residual=float(abs(mapping).max() / (1 + abs(beta).max() + abs(gradient).max())),
        constraint_violation=float(abs(beta - F.cone_project(beta, spec.signs)).max()),
    )


def refine(training, spec, beta, lam, soft, strength, zero_strength):
    """Refine the same objective to a tighter KKT target; never change a penalty."""
    x, y = training.weighted.x, training.weighted.y
    gram, cross = 2 * x.T @ x / len(x), 2 * y.T @ x / len(x)
    lipschitz = max(1.0, np.linalg.eigvalsh(gram)[-1] + 2 * max(strength, zero_strength))
    current = F.cone_project(beta, spec.signs)
    accelerated, momentum = current.copy(), 1.0
    initial = audit(training, spec, current, lam, soft, strength, zero_strength)
    for iteration in range(1, 10001):
        gradient = (
            accelerated @ gram
            - cross
            + soft_loss_gradient(accelerated, soft, strength, zero_strength)[1]
        )
        updated = F.proximal(accelerated - gradient / lipschitz, spec, lam / lipschitz)
        after = audit(training, spec, updated, lam, soft, strength, zero_strength)
        if after["kkt_residual"] <= 1e-7:
            current = updated
            break
        next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        accelerated = updated + (momentum - 1) / next_momentum * (updated - current)
        current, momentum = updated, next_momentum
    accepted = (
        after["objective"] <= initial["objective"] + 1e-12 * (1 + initial["objective"])
        and after["kkt_residual"] < initial["kkt_residual"]
    )
    return (current if accepted else beta), dict(
        algorithm="FISTA on unchanged objective",
        steps=iteration,
        accepted=bool(accepted),
        before=initial,
        after=after,
    )


@safe_cvxpy_sum_shape()
def fit(training, spec, lam, *, hard=None, soft=None, strength=0.0, zero_strength=0.0):
    """Fit one unchanged HCGL/FCGL objective plus explicit opt-in restrictions.

    Loss is sum_t w_t ||y_t-X_t B'||^2 / T, with sum(w)=T. Soft weights
    multiply sums over cells (not averages), just as the existing sparse penalty.
    There are no adaptive weights, magnitude priors or implicit sign restrictions.
    """
    if not all(np.isfinite(v) and v >= 0 for v in (lam, strength, zero_strength)):
        raise ValueError("Penalty strengths must be finite and nonnegative")
    if spec.family not in ("row", "block"):
        raise ValueError("Use ridge_fit for ridge")
    shape = spec.signs.shape
    hard = np.full(shape, np.nan) if hard is None else np.asarray(hard, dtype=float)
    soft = np.full(shape, np.nan) if soft is None else np.asarray(soft, dtype=float)
    for signs in (hard, soft):
        if signs.shape != shape or not np.isin(signs[np.isfinite(signs)], [-1, 0, 1]).all():
            raise ValueError("Invalid restriction shape/directions")
        if np.isinf(signs).any():
            raise ValueError("Infinite sign is not an unrestricted cell; use NaN")
    if (strength or zero_strength) and np.isnan(soft).all():
        raise ValueError("A positive soft penalty requires explicit directions/zero cells")
    restricted = replace(spec, signs=hard.copy())
    started = time.perf_counter()
    if np.all(hard == 0):
        beta = np.zeros(shape)
        return dict(
            beta=beta,
            alpha=np.zeros(shape[0]),
            status="exact_zero",
            attempts=[],
            seconds=time.perf_counter() - started,
            **audit(training, restricted, beta, lam, soft, strength, zero_strength),
        )
    # Orthogonal compression preserves SSE/T and removes a large constant residual
    # block from the conic programme. Every arm uses the same exact reduction.
    x, y = training.weighted.x, training.weighted.y
    q, r = np.linalg.qr(x, mode="reduced")
    projected = q.T @ y
    rows = max(5, len(r))  # The package builder requires at least five rows.
    compressed_x = np.zeros((rows, x.shape[1]))
    compressed_y = np.zeros((rows, y.shape[1]))
    compressed_x[: len(r)] = r * np.sqrt(rows / len(x))
    compressed_y[: len(r)] = projected * np.sqrt(rows / len(x))
    constant_loss = float(np.sum((y - q @ projected) ** 2) / len(x))
    compressed = replace(training.weighted, x=compressed_x, y=compressed_y)
    problem, variable, parameter = F.conic_program(compressed, restricted)
    parameter.value = lam
    # A zero-strength arm must retain the identical baseline cone programme.
    if strength or zero_strength:
        added = 0
        # Omit identically zero terms rather than introducing redundant cone rows.
        flat = cp.vec(variable, order="C")
        active = np.flatnonzero(np.isfinite(soft.ravel()) & (soft.ravel() != 0))
        zero = np.flatnonzero(np.isfinite(soft.ravel()) & (soft.ravel() == 0))
        if strength and len(active):
            added += strength * cp.sum_squares(
                cp.pos(-cp.multiply(soft.ravel()[active], flat[active]))
            )
        if zero_strength and len(zero):
            added += zero_strength * cp.sum_squares(flat[zero])
        problem = cp.Problem(cp.Minimize(problem.objective.expr + added), problem.constraints)
    problem = cp.Problem(cp.Minimize(problem.objective.expr + constant_loss), problem.constraints)
    attempts = []
    for retry, tolerance in enumerate((1e-7, 1e-8)):
        solver_error = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                problem.solve(
                    solver="CLARABEL",
                    max_iter=500 if retry == 0 else 2000,
                    tol_gap_abs=tolerance,
                    tol_gap_rel=tolerance,
                    tol_feas=tolerance,
                    warm_start=False,
                )
            except cp.error.SolverError as error:
                solver_error = repr(error)
        if solver_error is not None:
            attempts.append(
                dict(
                    status="solver_error",
                    tolerance=tolerance,
                    error=solver_error,
                    warnings=[str(w.message) for w in caught],
                )
            )
            continue
        if variable.value is None or not np.isfinite(variable.value).all():
            raise RuntimeError("Nonfinite comparator solution")
        beta = np.asarray(variable.value)
        checks = audit(training, restricted, beta, lam, soft, strength, zero_strength)
        refinement = None
        if checks["kkt_residual"] > 1e-7:
            beta, refinement = refine(
                training, restricted, beta, lam, soft, strength, zero_strength
            )
            checks = audit(training, restricted, beta, lam, soft, strength, zero_strength)
        discrepancy = abs(checks["objective"] - problem.value) / (1 + abs(problem.value))
        solver = problem._solver_cache.get("CLARABEL")
        info = solver.get_info() if solver is not None else None
        # CVXPY restores the objective constant after the solver's canonical solve.
        offset = problem.value - float(info.cost_primal) if info is not None else 0.0
        dual_gap = (
            (
                abs(checks["objective"] - float(info.cost_dual) - offset)
                / (1 + abs(checks["objective"]))
            )
            if info is not None
            else np.inf
        )
        resolved = (
            problem.status == cp.OPTIMAL_INACCURATE
            and retry == 1
            and info is not None
            and dual_gap <= 1e-8
            and float(info.res_primal) <= 1e-7
            and float(info.res_dual) <= 1e-7
            and checks["kkt_residual"] <= 1e-7
        )
        attempts.append(
            dict(
                status=problem.status,
                tolerance=tolerance,
                **checks,
                objective_discrepancy=float(discrepancy),
                orthogonal_residual_constant=constant_loss,
                numerical_refinement=refinement,
                independent_relative_dual_gap=float(dual_gap),
                solver_primal_residual=float(info.res_primal) if info is not None else None,
                solver_dual_residual=float(info.res_dual) if info is not None else None,
                inaccurate_resolved=bool(resolved),
                warnings=[str(w.message) for w in caught],
            )
        )
        if (
            (problem.status == cp.OPTIMAL or resolved)
            and checks["kkt_residual"] <= 1e-5
            and checks["constraint_violation"] <= 1e-7
            and discrepancy <= 1e-7
            and not any(issubclass(w.category, RuntimeWarning) for w in caught)
        ):
            return dict(
                beta=beta.copy(),
                alpha=np.zeros(shape[0]),
                status="resolved_inaccurate" if resolved else "optimal",
                attempts=attempts,
                seconds=time.perf_counter() - started,
                **checks,
            )
    raise RuntimeError(f"Unresolved comparator fit: {attempts}")


def ridge_fit(training, lam):
    """Public package ridge under precisely the same weighted, standardized loss."""
    if np.isnan(lam) or lam < 0:
        raise ValueError("Ridge penalty must be nonnegative")
    s = training.weighted
    model = LassoModel(
        model_type=LassoModelType.RIDGE,
        reg_lambda=float(lam),
        span=None,
        demean=False,
        warmup_period=None,
        loss_normalization="sample",
    ).fit(x=s.x, y=s.y)
    beta = model.coef_.to_numpy(copy=True)
    return dict(beta=beta, alpha=np.zeros(len(beta)), status="package_ridge", lambda_=float(lam))


def chronological_select(x, y, family, grid, span=None, cut=0.6):
    """Three prefix folds; select by standardized held-out reconstruction, never truth.

    Returns a grid value, not a full-sample numerical sparse penalty. For HCGL/
    FCGL that value multiplies the independently recomputed full-training KKT
    scale. Ties within 1e-12 select the larger grid value. The caller owns any
    predeclared boundary extension. No validation outcomes leave this ledger.
    """
    grid = sorted(set(float(v) for v in grid))
    if not grid or any(not np.isfinite(v) or v < 0 for v in grid):
        raise ValueError("Finite nonnegative candidate grid required")
    if family not in ("hcgl", "fcgl", "ridge"):
        raise ValueError("Unknown selector family")
    n = len(x)
    step = n // 6
    if n % 6 or n // 2 <= x.shape[1] + 2:
        raise ValueError("Pilot selector needs six equal blocks and identified training prefixes")
    ledger = []
    for stop in (n // 2, n // 2 + step, n // 2 + 2 * step):
        data = prepare(x.iloc[:stop], y.iloc[:stop], span)
        spec = None if family == "ridge" else specification(data, family, cut)
        xx = (x.iloc[stop : stop + step].to_numpy() - data.state.x_mean) / data.state.x_sd
        yy = (y.iloc[stop : stop + step].to_numpy() - data.state.y_mean) / data.state.y_sd
        for value in grid:
            result = (
                ridge_fit(data, value) if family == "ridge" else fit(data, spec, value * spec.scale)
            )
            ledger.append(
                dict(
                    train_rows=stop,
                    validation_rows=step,
                    candidate=value,
                    mse=float(np.mean((yy - xx @ result["beta"].T) ** 2)),
                    training_hash=F.array_hash(data.state.x),
                    labels=None if spec is None else spec.labels.tolist(),
                )
            )
    scores = {
        value: np.mean([r["mse"] for r in ledger if r["candidate"] == value]) for value in grid
    }
    minimum = min(scores.values())
    selected = max(v for v, score in scores.items() if score <= minimum + 1e-12)
    return dict(selected=selected, scores=scores, folds=ledger)
