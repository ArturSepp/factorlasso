"""Study adapters and independent optimality checks for the frozen French design.

Uses factorlasso's public ridge estimator, date gate and sparse-group CVXPY construction. The LASSO,
cooperative and UniLasso research programs implement the same published
objectives as estimator_benchmark.py, with intercepted LOO features for UniLasso.
No package defaults or installed modules are modified. All penalties use SSE/n.

Numerical refinement uses FISTA (Beck and Teboulle, 2009, SIAM Journal on
Imaging Sciences 2:183-202, doi:10.1137/080716542) on the identical objective.
It is recorded separately from the frozen CLARABEL attempts and does not
alter candidates, gate settings, penalties, or the acceptance tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
import warnings

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import lsq_linear, nnls
from scipy.sparse import csc_matrix
from scipy.spatial.distance import pdist

from factorlasso import LassoModel, LassoModelType, derive_sign_constraints
from factorlasso.linear_model._solvers.group_lasso import _build_group_lasso_problem

if __package__:
    from .cvxpy_shape_compat import safe_cvxpy_sum_shape
    from .estimator_benchmark import unilasso_features
else:
    from cvxpy_shape_compat import safe_cvxpy_sum_shape
    from estimator_benchmark import unilasso_features

KKT_TOL = 1e-5
OBJECTIVE_TOL = 1e-7
POLISH_TOL = 1e-7
POLISH_MAX_STEPS = 10000


def array_hash(value: np.ndarray) -> str:
    """Hash shape, dtype and exact finite array bytes, normalizing NaN restrictions."""
    a = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def canonical_labels(labels: np.ndarray) -> np.ndarray:
    """Make label IDs independent of linkage's arbitrary cluster numbering."""
    lookup = {}
    return np.asarray([lookup.setdefault(int(v), len(lookup)) for v in labels], dtype=int)


@dataclass
class State:
    """Transformations fitted exclusively on the supplied training rows."""

    x: np.ndarray
    y: np.ndarray
    x_mean: np.ndarray
    x_sd: np.ndarray
    y_mean: np.ndarray
    y_sd: np.ndarray
    dates: list[str]
    factors: list[str]
    portfolios: list[str]

    def __post_init__(self):
        self._labels = {}
        self._gates = {}
        self._tree = None
        self._unilasso = None

    def labels(self, partition) -> np.ndarray:
        if partition not in self._labels:
            q = self.y.shape[1]
            if partition in (None, "singleton"):
                labels = np.arange(q)
            elif partition == "one_cluster" or q == 1:
                labels = np.zeros(q, dtype=int)
            else:
                if self._tree is None:
                    self._tree = linkage(pdist(self.y.T) / np.sqrt(len(self.y)), method="ward")
                labels = fcluster(self._tree, float(partition) * self._tree[-1, 2],
                                  criterion="distance")
            self._labels[partition] = canonical_labels(labels)
        return self._labels[partition]

    def gates(self, labels: np.ndarray, tau) -> np.ndarray:
        key = (tuple(labels), tau)
        if key not in self._gates:
            signs = np.zeros((self.y.shape[1], self.x.shape[1]))
            xx = pd.DataFrame(self.x, columns=self.factors)
            for label in np.unique(labels):
                members = np.flatnonzero(labels == label)
                yy = pd.DataFrame(self.y[:, members], columns=[self.portfolios[k] for k in members])
                # derive_sign_constraints.clusters means predictor clusters;
                # response pooling is requested by slicing the response block.
                signs[members] = derive_sign_constraints(
                    xx, yy, auto_sign_threshold_t=np.inf if tau == "inf" else float(tau),
                    variance_estimator="date", ewma_span=None).to_numpy()
            self._gates[key] = signs
        return self._gates[key]

    def unilasso(self):
        if self._unilasso is None:
            pieces = [unilasso_features(self.x, self.y[:, k]) for k in range(self.y.shape[1])]
            eta = np.stack([v[0] for v in pieces], axis=1)
            mean = eta.mean(axis=0)
            self._unilasso = (eta-mean, mean, np.stack([v[1] for v in pieces]),
                              np.stack([v[2] for v in pieces]))
            if not all(np.isfinite(v).all() for v in self._unilasso):
                raise ValueError("Nonfinite UniLasso features")
        return self._unilasso

    def manifest(self) -> dict:
        return {"train_dates": self.dates, "x_mean": self.x_mean.tolist(),
                "x_sd": self.x_sd.tolist(), "y_mean": self.y_mean.tolist(),
                "y_sd": self.y_sd.tolist(), "x_hash": array_hash(self.x),
                "y_hash": array_hash(self.y), "factor_columns": self.factors,
                "portfolio_columns": self.portfolios}


def prepare(x: pd.DataFrame, y: pd.DataFrame) -> State:
    """No validation/test data may enter this function."""
    if not x.index.equals(y.index) or not x.index.is_unique or not x.index.is_monotonic_increasing:
        raise ValueError("Training inputs require identical ordered unique dates")
    xx, yy = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
    if len(x) < 5 or not np.isfinite(xx).all() or not np.isfinite(yy).all():
        raise ValueError("At least five complete finite training rows are required")
    xm, xs, ym, ys = xx.mean(axis=0), xx.std(axis=0), yy.mean(axis=0), yy.std(axis=0)
    if np.any(xs <= 1e-12) or np.any(ys <= 1e-12):
        raise ValueError("Zero-variance training column; panel cannot be silently changed")
    return State((xx-xm)/xs, (yy-ym)/ys, xm, xs, ym, ys,
                 x.index.strftime("%Y-%m-%d").tolist(), list(x.columns), list(y.columns))


@dataclass
class Spec:
    """The actual objective and restrictions, independently of a candidate's label."""

    family: str
    labels: np.ndarray
    weights: np.ndarray
    signs: np.ndarray
    scale: float

    def key(self) -> str:
        if self.family in ("lasso", "unilasso", "ridge", "ols"):
            geometry = ""
        elif self.family == "row":
            geometry = array_hash(self.weights)
        else:
            geometry = array_hash(self.labels) + array_hash(self.weights)
        return self.family + geometry + array_hash(np.nan_to_num(self.signs, nan=2.0))


def specification(state: State, candidate: dict) -> Spec:
    method = candidate["method"]
    family = ("row" if method.startswith("hcgl") else "block" if method.startswith("fcgl")
              else "lasso" if method in ("lasso", "singleton_exclude") else method)
    labels = state.labels(candidate.get("partition"))
    groups = np.unique(labels)
    weights = np.asarray([np.sqrt(np.sum(labels == label)/len(groups)) for label in labels])
    signs = np.full((state.y.shape[1], state.x.shape[1]), np.nan)
    if method in ("singleton_exclude", "hcgl_singleton_exclude"):
        signs = state.gates(state.labels("singleton"), candidate["tau"]).copy()
    elif method in ("hcgl_exclude", "hcgl_unrestricted", "fcgl_exclude"):
        signs = state.gates(labels, candidate["tau"]).copy()
        if method == "hcgl_unrestricted":
            signs[signs == 0] = np.nan
    spec = Spec(family, labels, weights, signs, 0.0)
    spec.scale = kkt_scale(state, spec)
    return spec


def kkt_scale(state: State, spec: Spec) -> float:
    """Exact zero-solution dual threshold for the frozen objective/sign cone."""
    if spec.family in ("ridge", "ols"):
        return 0.0
    if spec.family == "unilasso":
        eta = state.unilasso()[0]
        return float(np.maximum(2*np.einsum("nqp,nq->qp", eta, state.y)/len(state.x), 0).max())
    h = 2*state.y.T @ state.x/len(state.x)
    magnitude = np.abs(h)
    magnitude = np.where(spec.signs == 0, 0, magnitude)
    magnitude = np.where(spec.signs == 1, np.maximum(h, 0), magnitude)
    magnitude = np.where(spec.signs == -1, np.maximum(-h, 0), magnitude)
    if spec.family == "lasso":
        return float(magnitude.max(initial=0))
    if spec.family == "cooperative":
        return max(float(max(np.linalg.norm(np.maximum(h[g], 0), axis=0).max(),
                             np.linalg.norm(np.minimum(h[g], 0), axis=0).max())) /
                   spec.weights[g[0]]
                   for g in group_indices(spec.labels))
    lower, upper = 0.0, float(magnitude.max(initial=0)/0.5)
    for _ in range(55):
        middle = (lower+upper)/2
        valid = True
        if spec.family == "row":
            norms = np.linalg.norm(np.maximum(magnitude-0.5*middle, 0), axis=1)
            valid = bool(np.all(norms <= 0.5*middle*spec.weights))
        else:
            for g in group_indices(spec.labels):
                norms = np.linalg.norm(np.maximum(magnitude[g]-0.5*middle, 0), axis=0)
                valid &= bool(np.all(norms <= 0.5*middle*spec.weights[g[0]]))
        if valid:
            upper = middle
        else:
            lower = middle
    return upper


def group_indices(labels):
    return [np.flatnonzero(labels == value) for value in np.unique(labels)]


def cone_project(beta, signs):
    result = beta.copy()
    result[signs == 0] = 0
    result[signs == 1] = np.maximum(result[signs == 1], 0)
    result[signs == -1] = np.minimum(result[signs == -1], 0)
    return result


def proximal(value, spec: Spec, lam):
    """Independent disjoint-group proximal map, including the sign cone."""
    if spec.family == "unilasso":
        return np.maximum(value-lam, 0)
    out = cone_project(value, spec.signs)
    if spec.family in ("block", "cooperative") and len(np.unique(spec.labels)) == len(out):
        threshold = (spec.weights if spec.family == "cooperative" else .5+.5*spec.weights)*lam
        return np.sign(out)*np.maximum(np.abs(out)-threshold[:, None], 0)
    if spec.family == "cooperative":
        for g in group_indices(spec.labels):
            pos, neg = np.maximum(out[g], 0), np.minimum(out[g], 0)
            for part in (pos, neg):
                norms = np.linalg.norm(part, axis=0)
                part *= np.maximum(1-lam*spec.weights[g[0]]/np.maximum(norms, 1e-300), 0)
            out[g] = pos+neg
        return out
    alpha = 1.0 if spec.family == "lasso" else 0.5
    out = np.sign(out)*np.maximum(np.abs(out)-alpha*lam, 0)
    if spec.family == "row":
        norms = np.linalg.norm(out, axis=1)
        out *= np.maximum(1-0.5*lam*spec.weights/np.maximum(norms, 1e-300), 0)[:, None]
    elif spec.family == "block":
        for g in group_indices(spec.labels):
            norms = np.linalg.norm(out[g], axis=0)
            out[g] *= np.maximum(1-0.5*lam*spec.weights[g[0]] /
                                 np.maximum(norms, 1e-300), 0)
    return out


def penalty(beta, spec: Spec) -> float:
    """Evaluate the objective with explicit row/block norms, outside CVXPY."""
    if spec.family in ("lasso", "unilasso"):
        return float(np.abs(beta).sum())
    if spec.family == "row":
        return float(0.5*np.abs(beta).sum()+0.5*np.sum(spec.weights*np.linalg.norm(beta, axis=1)))
    total = 0.0
    for g in group_indices(spec.labels):
        if spec.family == "cooperative":
            norms = (np.linalg.norm(np.maximum(beta[g], 0), axis=0) +
                     np.linalg.norm(np.minimum(beta[g], 0), axis=0))
            total += spec.weights[g[0]]*float(norms.sum())
        else:
            total += spec.weights[g[0]]*float(np.linalg.norm(beta[g], axis=0).sum())
    return total if spec.family == "cooperative" else 0.5*np.abs(beta).sum()+0.5*total


def audit(state: State, spec: Spec, value, lam, *, ridge=False) -> dict:
    """Independent objective, feasibility and proximal-gradient KKT residual."""
    n = len(state.x)
    if spec.family == "unilasso":
        eta = state.unilasso()[0]
        residual = np.einsum("nqp,qp->nq", eta, value)-state.y
        gradient = 2*np.einsum("nqp,nq->qp", eta, residual)/n
        gram = 2*np.einsum("nqi,nqj->qij", eta, eta)/n
        lipschitz = max(1., float(np.linalg.eigvalsh(gram).max()))
        objective = float(np.square(residual).sum()/n+lam*np.abs(value).sum())
        violation = float(np.maximum(-value, 0).max(initial=0))
    else:
        residual = state.x@value.T-state.y
        gradient = 2*residual.T@state.x/n
        lipschitz = max(1., float(2*np.linalg.eigvalsh(state.x.T@state.x/n).max()))
        violation = float(np.abs(value-cone_project(value, spec.signs)).max(initial=0))
        objective = float(np.square(residual).sum()/n)
        if ridge:
            objective += float(lam*np.square(value).sum())
            gradient += 2*lam*value
        else:
            objective += lam*penalty(value, spec)
    if ridge or spec.family == "ols":
        difference = gradient
    else:
        step = 1/lipschitz
        difference = (value-proximal(value-step*gradient, spec, lam*step))/step
    scaled = float(np.abs(difference).max(initial=0) /
                   (1+np.abs(value).max(initial=0)+np.abs(gradient).max(initial=0)))
    return {"objective": objective, "constraint_violation": violation, "kkt_residual": scaled}


def zero_penalty(state: State, spec: Spec):
    """Exact unconstrained/bounded least squares; no positive-lambda surrogate."""
    if spec.family == "unilasso":
        eta, mean, slopes, constants = state.unilasso()
        theta = np.stack([nnls(eta[:, k], state.y[:, k], maxiter=1000)[0]
                          for k in range(state.y.shape[1])])
        return theta*slopes, np.sum((constants-mean)*theta, axis=1), theta
    beta = np.zeros_like(spec.signs)
    for k in range(len(beta)):
        keep = spec.signs[k] != 0
        if not keep.any():
            continue
        x, signs = state.x[:, keep], spec.signs[k, keep]
        lower, upper = np.where(signs == 1, 0., -np.inf), np.where(signs == -1, 0., np.inf)
        if not np.isfinite(lower).any() and not np.isfinite(upper).any():
            value = np.linalg.lstsq(x, state.y[:, k], rcond=None)[0]
        else:
            result = lsq_linear(x, state.y[:, k], bounds=(lower, upper), method="bvls",
                                tol=1e-12, max_iter=1000)
            if not result.success:
                raise RuntimeError(f"Bounded least squares failed: {result.message}")
            value = result.x
        beta[k, keep] = value
    return beta, np.zeros(len(beta)), beta


def polish(state: State, spec: Spec, value: np.ndarray, lam: float):
    """Refine an approximate conic point without changing its convex objective."""
    before = audit(state, spec, value, lam)
    # A conic point can sit just outside a binding sign constraint and have an
    # objective below the feasible optimum. Compare descent with its feasible
    # projection, while retaining the raw objective discrepancy checks below.
    feasible = (np.maximum(value, 0) if spec.family == "unilasso" else
                cone_project(value, spec.signs))
    feasible_before = audit(state, spec, feasible, lam)
    if spec.family == "unilasso":
        eta = state.unilasso()[0]
        h = 2*np.einsum("nqi,nqj->qij", eta, eta)/len(state.x)
        cross = 2*np.einsum("nqp,nq->qp", eta, state.y)/len(state.x)
        lipschitz = max(1., float(np.linalg.eigvalsh(h).max()))
        def gradient(v):
            return np.einsum("qij,qj->qi", h, v)-cross
    else:
        h = 2*state.x.T@state.x/len(state.x)
        cross = 2*state.y.T@state.x/len(state.x)
        lipschitz = max(1., float(np.linalg.eigvalsh(h).max()))
        def gradient(v):
            return v@h-cross
    step = 1/lipschitz
    current, extrapolated, momentum = value.copy(), value.copy(), 1.
    for iteration in range(1, POLISH_MAX_STEPS+1):
        updated = proximal(extrapolated-step*gradient(extrapolated), spec, lam*step)
        g = gradient(updated)
        mapping = (updated-proximal(updated-step*g, spec, lam*step))/step
        residual = float(np.abs(mapping).max(initial=0) /
                         (1+np.abs(updated).max(initial=0)+np.abs(g).max(initial=0)))
        if residual <= POLISH_TOL:
            current = updated
            break
        next_momentum = (1+np.sqrt(1+4*momentum**2))/2
        extrapolated = updated+(momentum-1)/next_momentum*(updated-current)
        current, momentum = updated, next_momentum
    after = audit(state, spec, current, lam)
    accepted = (after["objective"] <= feasible_before["objective"]
                + 1e-12*(1+abs(feasible_before["objective"])) and
                after["kkt_residual"] < before["kkt_residual"])
    details = {"algorithm": "FISTA on unchanged objective", "steps": iteration,
               "accepted": bool(accepted), "before_kkt_residual": before["kkt_residual"],
               "after_kkt_residual": after["kkt_residual"],
               "objective_change": after["objective"]-before["objective"],
               "feasible_start_objective": feasible_before["objective"],
               "initial_constraint_violation": before["constraint_violation"],
               "objective_change_from_feasible_start":
                   after["objective"]-feasible_before["objective"]}
    return (current if accepted else value), details


@safe_cvxpy_sum_shape()
def conic_program(state: State, spec: Spec):
    """Reuse the package group builder; keep lambda as a DPP parameter."""
    n, p, q = len(state.x), state.x.shape[1], state.y.shape[1]
    lam = cp.Parameter(nonneg=True)
    if spec.family in ("row", "block") and np.isnan(spec.signs).all():
        groups = np.eye(len(np.unique(spec.labels)))[spec.labels]
        built = _build_group_lasso_problem(
            state.x, state.y, groups, lam, valid_mask=None, span=None, nonneg=False,
            factors_beta_loading_signs=spec.signs, factors_beta_prior=None,
            group_penalty="normalized", l1_weight=.5, penalty_weights=None,
            row_weights=None, block_mode="row" if spec.family == "row" else "cluster_factor",
            col_weights=None, loss_normalization="sample")
        problem, variable, _, _ = built
    else:
        if spec.family == "unilasso":
            variable = cp.Variable((q, p))
            eta = state.unilasso()[0]
            predicted = cp.sum(cp.multiply(eta, variable[None, :, :]), axis=2)
            fit = cp.sum_squares(predicted-state.y)/n
            pen = cp.sum(variable)
            constraints = [variable >= 0]
        else:
            # Remove pinned coefficients and redundant zero rows of the cone
            # representation. This is exactly the same restricted objective.
            keep = np.flatnonzero(spec.signs.ravel() != 0)
            free = cp.Variable(len(keep))
            embedding = csc_matrix((np.ones(len(keep)), (keep, np.arange(len(keep)))),
                                   shape=(q*p, len(keep)))
            variable = cp.reshape(embedding@free, (q, p), order="C")
            signs = spec.signs.ravel()[keep]
            constraints = []
            if np.any(signs == 1):
                constraints.append(free[signs == 1] >= 0)
            if np.any(signs == -1):
                constraints.append(free[signs == -1] <= 0)
            l1 = cp.norm1(free)
            fit = cp.sum_squares(state.x@variable.T-state.y)/n
            if spec.family == "lasso":
                pen = l1
            elif spec.family == "row":
                rows = np.flatnonzero(np.any(spec.signs != 0, axis=1))
                group = cp.sum(cp.multiply(spec.weights[rows], cp.norm(variable[rows], 2, axis=1)))
                pen = .5*l1+.5*group
            elif spec.family == "block":
                terms = []
                for g in group_indices(spec.labels):
                    columns = np.flatnonzero(np.any(spec.signs[g] != 0, axis=0))
                    if len(columns):
                        terms.append(spec.weights[g[0]]*cp.sum(cp.norm(
                            variable[np.ix_(g, columns)], 2, axis=0)))
                pen = .5*l1+.5*sum(terms)
            else:
                pen = sum(spec.weights[g[0]]*cp.sum(cp.norm(cp.pos(variable[g]), 2, axis=0) +
                                                  cp.norm(cp.neg(variable[g]), 2, axis=0))
                          for g in group_indices(spec.labels))
        problem = cp.Problem(cp.Minimize(fit+lam*pen), constraints)
    if not problem.is_dcp(dpp=True):
        raise RuntimeError("Study path must be DPP-compliant")
    return problem, variable, lam


def fit_path(state: State, spec: Spec, penalties: list[float], protocol: dict, *, on_result=None):
    """Fit requested distinct penalties with exact endpoints and a strict retry ledger."""
    program = None
    results = []
    for lam in penalties:
        started = time.perf_counter()
        attempts = []
        selected_attempt = None
        zero = lam == 0 or spec.family == "ols"
        upper_endpoint = lam >= spec.scale and spec.scale > 0 and spec.family != "ridge"
        if spec.family == "ridge":
            # The study has already centred and standardised each training fold.
            # Disable package demeaning and warmup zeroing to retain SSE/n and
            # the frozen intercept reconstruction in original_units().
            model = LassoModel(
                model_type=LassoModelType.RIDGE, reg_lambda=float(lam),
                span=None, demean=False, warmup_period=None,
                loss_normalization="sample", solver=protocol["solver"]["conic"],
            ).fit(x=state.x, y=state.y)
            beta = model.coef_.to_numpy(copy=True)
            alpha = np.zeros(state.y.shape[1])
            if np.isinf(lam):
                if np.any(beta != 0):
                    raise RuntimeError("Package ridge infinite endpoint must be exactly zero")
                check = {"objective": float(np.square(state.y).sum()/len(state.x)),
                         "kkt_residual": 0., "constraint_violation": 0.}
            else:
                check = audit(state, spec, beta, lam, ridge=True)
            kind = "package_ridge"
        elif zero:
            beta, alpha, value = zero_penalty(state, spec)
            check = audit(state, spec, value, 0.)
            kind = "direct_nnls" if spec.family == "unilasso" else "direct_least_squares"
        elif upper_endpoint or spec.scale == 0:
            beta, alpha = np.zeros_like(spec.signs), np.zeros(state.y.shape[1])
            check = audit(state, spec, beta, lam)
            kind = "analytic_zero_kkt"
        else:
            if program is None:
                program = conic_program(state, spec)
            problem, variable, parameter = program
            parameter.value = lam
            policy = protocol["solver"]
            qualified = []
            for retry, options in enumerate((policy["options"], policy["retry"]["options"])):
                attempt_start = time.perf_counter()
                error = None
                with safe_cvxpy_sum_shape(), warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    try:
                        # DPP still reuses canonicalization; a cold numerical
                        # start makes an isolated resumed point reproducible.
                        problem.solve(solver=policy["conic"], warm_start=False, **options)
                    except Exception as exc:
                        error = repr(exc)
                value = None if variable.value is None else np.asarray(variable.value).copy()
                check = (audit(state, spec, value, lam) if value is not None and
                         np.isfinite(value).all() else {"objective": None,
                         "kkt_residual": 1e100, "constraint_violation": 1e100})
                raw_discrepancy = (abs(check["objective"]-problem.value) /
                                   (1+abs(check["objective"])) if check["objective"] is not None and
                                   problem.value is not None else 1e100)
                refinement = None
                target = POLISH_TOL if problem.status == cp.OPTIMAL_INACCURATE else KKT_TOL
                if (error is None and problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and
                        raw_discrepancy <= OBJECTIVE_TOL and check["kkt_residual"] > target):
                    value, refinement = polish(state, spec, value, lam)
                    check = audit(state, spec, value, lam)
                discrepancy = (abs(check["objective"]-problem.value) /
                               (1+abs(check["objective"])) if check["objective"] is not None and
                               problem.value is not None else 1e100)
                info = problem._solver_cache.get("CLARABEL")
                info = info.get_info() if info is not None else None
                independent_gap = (abs(check["objective"]-float(info.cost_dual)) /
                                   (1+abs(check["objective"])) if info is not None and
                                   check["objective"] is not None else 1e100)
                eligible_flag = (problem.status == cp.OPTIMAL_INACCURATE and
                                 check["kkt_residual"] <= POLISH_TOL and
                                 check["constraint_violation"] <= policy["max_constraint_violation"]
                                 and independent_gap <= 1e-8 and info is not None and
                                 float(info.res_dual) <= policy["max_constraint_violation"])
                resolved_flag = retry == 1 and eligible_flag
                attempt = {"retry": retry, "solver": policy["conic"], "options": options,
                           "status": problem.status, "error": error,
                           "warnings": [str(w.message) for w in caught],
                           "seconds": time.perf_counter()-attempt_start,
                           "objective_discrepancy": discrepancy, **check,
                           "raw_objective_discrepancy": raw_discrepancy,
                           "numerical_refinement": refinement,
                           "independent_relative_dual_gap": independent_gap,
                           "inaccurate_flag_meets_independent_checks": bool(eligible_flag),
                           "inaccurate_flag_resolved_by_independent_checks": bool(resolved_flag),
                           "primal_residual": float(info.res_primal) if info else None,
                           "dual_residual": float(info.res_dual) if info else None,
                           "gap_abs": float(info.gap_abs) if info else None}
                attempts.append(attempt)
                if (eligible_flag and error is None and raw_discrepancy <= OBJECTIVE_TOL and
                        discrepancy <= OBJECTIVE_TOL and
                        not any(issubclass(w.category, RuntimeWarning) for w in caught)):
                    qualified.append((independent_gap, check["kkt_residual"], retry,
                                      value.copy(), check.copy()))
                okay = (error is None and (problem.status == cp.OPTIMAL or resolved_flag) and
                        check["constraint_violation"] <= policy["max_constraint_violation"] and
                        check["kkt_residual"] <= KKT_TOL and discrepancy <= OBJECTIVE_TOL and
                        raw_discrepancy <= OBJECTIVE_TOL and
                        not any(issubclass(w.category, RuntimeWarning) for w in caught))
                if okay:
                    selected_attempt = retry
                    break
            else:
                if qualified:
                    _, _, selected_attempt, value, check = min(qualified, key=lambda v: v[:3])
                    chosen_attempt = attempts[selected_attempt]
                    chosen_attempt["inaccurate_flag_resolved_by_independent_checks"] = True
                else:
                    exc = RuntimeError(f"Unresolved numerical fit: {attempts}")
                    exc.attempts = attempts
                    raise exc
            if spec.family == "unilasso":
                _, mean, slopes, constants = state.unilasso()
                beta, alpha = value*slopes, np.sum((constants-mean)*value, axis=1)
            else:
                beta, alpha = value, np.zeros(state.y.shape[1])
            kind = "conic"
        if not np.isfinite(beta).all() or check["kkt_residual"] > KKT_TOL:
            raise RuntimeError(f"Endpoint optimality failure: {check}")
        status = ("resolved_inaccurate" if attempts and
                  attempts[selected_attempt]["inaccurate_flag_resolved_by_independent_checks"]
                  else "optimal")
        results.append({"beta": beta, "alpha": alpha, "lambda": float(lam),
                        "kind": kind, "status": status, "attempts": attempts,
                        "selected_attempt_index": selected_attempt,
                        "seconds": time.perf_counter()-started, **check})
        if on_result is not None:
            on_result(len(results)-1, results[-1])
    return results


def original_units(state: State, result: dict):
    beta = result["beta"]*state.y_sd[:, None]/state.x_sd[None, :]
    intercept = state.y_mean+state.y_sd*result["alpha"]-beta@state.x_mean
    return beta, intercept
