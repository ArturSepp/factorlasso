"""Feasible-tuned, paired full-estimator benchmark for the CSDA revision.

Every inner fold repeats centering, scaling, chord-distance Ward clustering
and date-score sign screening.
Only held-out prediction loss selects lambda and tau. Independent test outcomes
and coefficient truth never enter fitting or tuning. Known groups are supplied
only to the explicitly named known-partition diagnostic.

The matched HCGL methods minimize ||Y-X B'||_F^2/n + lambda times
  (1-alpha) sum_g sqrt(|g|/G) sum_{k in g} ||B_k||_2
  + alpha sum_kj W_kj |B_kj|,
with alpha=.5, zero prior and W=1. This benchmark is restricted to gamma=0;
it does not validate an adaptive cell/row weighting variant.
The FCGL comparator replaces response-row norms by cluster-by-predictor norms.
The cooperative comparator uses positive/negative cluster-by-predictor norms.
All loss functions use 1/n (sklearn's alpha is therefore lambda/2).

The UniLasso comparator implements the published intercept-containing LOO
univariate regressions and nonnegative second-stage Lasso with an intercept and
NO feature standardization. This is not the package's centered through-origin
LOO approximation. Reference: Chatterjee, Hastie and Tibshirani (2025),
doi:10.1162/99608f92.c79ff6db, algorithm 1;
https://pmc.ncbi.nlm.nih.gov/articles/PMC13505989/ .

Outputs are new external run files; archived reference inputs are never read or
overwritten. A small run is a protocol check, not publication-level evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
import tomllib

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.optimize import nnls
from sklearn.linear_model import Lasso

import factorlasso
from factorlasso.priors import derive_sign_constraints

if __package__:
    from .paths import REPOSITORY_DIR, output_directory
    from .cvxpy_shape_compat import safe_cvxpy_sum_shape
else:
    from paths import REPOSITORY_DIR, output_directory
    from cvxpy_shape_compat import safe_cvxpy_sum_shape


METHODS = (
    "lasso", "hcgl_unconstrained", "hcgl_per_response",
    "hcgl_pooled_zero", "hcgl_pooled_free", "hcgl_known_zero",
    "fcgl_unconstrained", "cooperative", "unilasso",
)
SIGNED_METHODS = frozenset(("hcgl_per_response", "hcgl_pooled_zero",
                            "hcgl_pooled_free", "hcgl_known_zero"))
ZERO_TOLERANCES = (1e-8, 1e-6, 1e-4)
PRIMARY_ZERO_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Scenario:
    name: str
    train_size: int = 80
    test_size: int = 1000
    predictors: int = 12
    cluster_sizes: tuple[int, ...] = (4, 4, 4)
    active_per_cluster: int = 3
    r2: float = .30
    partial_sharing: bool = False
    mixed_signs: bool = False
    predictor_correlation: float = 0.
    residual_correlation: float = 0.
    unequal_variances: bool = False


SCENARIOS = (
    Scenario("baseline"),
    Scenario("weak_signal", r2=.10),
    Scenario("partial_support", partial_sharing=True),
    Scenario("mixed_signs", mixed_signs=True),
    Scenario("correlated", predictor_correlation=.6, residual_correlation=.6),
    Scenario("unequal", cluster_sizes=(2, 4, 6), unequal_variances=True),
)


@dataclass(frozen=True)
class Protocol:
    folds: int = 3
    lambda_fractions: tuple[float, ...] = (1., .3, .1, .03, .01, .003, .001)
    tau_grid: tuple[float, ...] = (.75, 1.5, 2.5)
    cutoff_fraction: float = .5
    alpha: float = .5
    adaptive_gamma: float = 0.
    pilot_floor: float = .05
    solver: str = "MOSEK"

    def __post_init__(self):
        if not 0 < self.alpha <= 1 or self.adaptive_gamma < 0 or self.pilot_floor <= 0:
            raise ValueError("Require 0<alpha<=1, gamma>=0 and positive pilot floor")
        if not self.lambda_fractions or any(f <= 0 for f in self.lambda_fractions):
            raise ValueError("Lambda fractions must be positive")
        if not self.tau_grid or any(t < 0 for t in self.tau_grid):
            raise ValueError("Tau grid must be nonnegative")
        if self.adaptive_gamma != 0:
            raise ValueError("This matched benchmark validates gamma=0 only")


@dataclass
class Prepared:
    x: np.ndarray
    y: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    labels: np.ndarray

    def transform_x(self, x):
        return (np.asarray(x) - self.x_mean) / self.x_scale

    def transform_y(self, y):
        return (np.asarray(y) - self.y_mean) / self.y_scale


@dataclass
class Fitted:
    beta: np.ndarray
    intercept: np.ndarray
    prepared: Prepared
    lambda_value: float
    lambda_fraction: float
    tau: float
    cv_mse: float
    seconds: float
    signs: np.ndarray

    def predict(self, x):
        z = self.prepared.transform_x(x) @ self.beta.T + self.intercept
        return z * self.prepared.y_scale + self.prepared.y_mean

    @property
    def beta_raw(self):
        return self.beta * self.prepared.y_scale[:, None] / self.prepared.x_scale


def simulate(scenario: Scenario, seed: int):
    """Return independent train/test draws under common population parameters."""
    p = scenario.predictors
    labels = np.repeat(np.arange(len(scenario.cluster_sizes)), scenario.cluster_sizes)
    n_y = len(labels)
    rng_parameters, rng_train, rng_test = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
    )
    beta = np.zeros((n_y, p))
    for group in np.unique(labels):
        active = rng_parameters.choice(p, scenario.active_per_cluster, replace=False)
        signs = rng_parameters.choice([-1., 1.], len(active))
        members = np.flatnonzero(labels == group)
        for local, k in enumerate(members):
            selected = active.copy()
            if scenario.partial_sharing and local % 2:
                inactive = np.setdiff1d(np.arange(p), active)
                selected[-1] = rng_parameters.choice(inactive)
            beta[k, selected] = signs * rng_parameters.uniform(.5, 1.5, len(active))
            if scenario.mixed_signs and local % 2:
                beta[k, selected[0]] *= -1
    rho = scenario.predictor_correlation
    sigma = (1-rho)*np.eye(p) + rho*np.ones((p, p))
    chol = np.linalg.cholesky(sigma)
    signal_var = np.einsum("ij,jk,ik->i", beta, sigma, beta)
    noise_sd = np.sqrt(signal_var * (1-scenario.r2)/scenario.r2)
    if scenario.unequal_variances:
        noise_sd *= np.linspace(.5, 2., n_y)

    def draw(rng, n):
        x = rng.standard_normal((n, p)) @ chol.T
        eps = rng.standard_normal((n, n_y))
        if scenario.residual_correlation:
            rho_e = scenario.residual_correlation
            common = rng.standard_normal((n, len(scenario.cluster_sizes)))
            eps = np.sqrt(1-rho_e)*eps + np.sqrt(rho_e)*common[:, labels]
        return x, x @ beta.T + eps*noise_sd

    x, y = draw(rng_train, scenario.train_size)
    xt, yt = draw(rng_test, scenario.test_size)
    return x, y, xt, yt, beta, labels


def ward_labels(y: np.ndarray, cutoff_fraction: float) -> np.ndarray:
    """Ward on Euclidean chord distances; cut at a fraction of final merge height."""
    if not 0 <= cutoff_fraction <= 1:
        raise ValueError("cutoff_fraction must lie in [0, 1]")
    n = y.shape[1]
    if n == 1 or cutoff_fraction == 1:
        return np.zeros(n, dtype=int)
    if cutoff_fraction == 0:
        return np.arange(n)
    centered = y - np.mean(y, axis=0)
    norms = np.linalg.norm(centered, axis=0)
    if np.any(norms == 0):
        raise ValueError("Ward clustering requires nonconstant responses")
    # Distances are exactly sqrt(2*(1-rho)); do not feed squared distances to Ward.
    tree = linkage(pdist((centered/norms).T), method="ward")
    return fcluster(tree, cutoff_fraction*tree[-1, 2], criterion="distance") - 1


def prepare(x: np.ndarray, y: np.ndarray, cutoff_fraction=.5) -> Prepared:
    """Use only the supplied training rows for all preprocessing."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("This complete-panel benchmark does not impute missing values")
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    xs, ys = x.std(axis=0), y.std(axis=0)
    if np.any(xs == 0) or np.any(ys == 0):
        raise ValueError("Constant training columns are unsupported")
    xx, yy = (x-xm)/xs, (y-ym)/ys
    return Prepared(xx, yy, xm, xs, ym, ys, ward_labels(yy, cutoff_fraction))


def derive_gate(x, y, labels, tau):
    """Pool responses, never predictors, using the current public date-score gate."""
    signs = np.empty((y.shape[1], x.shape[1]))
    slopes = np.empty_like(signs)
    for group in np.unique(labels):
        members = np.flatnonzero(labels == group)
        s, b = derive_sign_constraints(
            x, y[:, members], auto_sign_threshold_t=tau,
            return_slopes=True, variance_estimator="date",
        )
        signs[members], slopes[members] = s, b
    if np.isposinf(tau):
        # The study's +infinity endpoint fails every gate, even for a noiseless
        # nonzero slope whose date-score statistic is itself infinite.
        signs.fill(0.)
    return signs, slopes


def method_inputs(prepared, method, tau, protocol, known_labels=None):
    labels = prepared.labels
    if method == "hcgl_known_zero":
        if known_labels is None:
            raise ValueError("Known-partition diagnostic requires its oracle partition")
        labels = np.asarray(known_labels)
    gate_labels = np.arange(prepared.y.shape[1]) if method == "hcgl_per_response" else labels
    signs = np.full((prepared.y.shape[1], prepared.x.shape[1]), np.nan)
    if method in SIGNED_METHODS:
        signs, _ = derive_gate(prepared.x, prepared.y, gate_labels, tau)
        if method == "hcgl_pooled_free":
            signs[signs == 0] = np.nan
    weights = np.ones_like(signs)
    return labels, signs, weights


def unilasso_features(x, y):
    """Exact intercept-containing LOO univariate fits and full-data slopes/intercepts."""
    n = len(x)
    sx, sy = x.sum(axis=0), y.sum()
    sxx, sxy = np.sum(x*x, axis=0), x.T@y
    dx, dy = sx[None, :]-x, sy-y[:, None]
    denom = sxx[None, :] - x*x - dx*dx/(n-1)
    slopes_loo = (sxy[None, :] - x*y[:, None] - dx*dy/(n-1))/denom
    intercepts_loo = dy/(n-1) - slopes_loo*dx/(n-1)
    slopes = (sxy-sx*sy/n)/(sxx-sx*sx/n)
    intercepts = sy/n - slopes*sx/n
    return intercepts_loo + slopes_loo*x, slopes, intercepts


def _solve(problem, solver):
    kwargs = {"mosek_params": {"MSK_IPAR_NUM_THREADS": 1}} if solver == "MOSEK" else {}
    problem.solve(solver=solver, warm_start=True, **kwargs)
    if problem.status != cp.OPTIMAL:
        raise RuntimeError(f"{solver} returned {problem.status}; no silent fallback")


def lambda_fractions_for(method, protocol):
    """UniLasso includes its NNLS endpoint after pilot CV exposed grid truncation."""
    fractions = protocol.lambda_fractions
    if method == "unilasso":
        fractions = tuple(sorted(set(fractions + (1e-4, 1e-5, 1e-6, 0.)), reverse=True))
    return fractions


@safe_cvxpy_sum_shape()
def fit_path(prepared, method, tau, protocol, known_labels=None, fractions=None):
    """Fit a training-only relative-lambda path; return standardized coefficients."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}")
    x, y = prepared.x, prepared.y
    n, p, q = len(x), x.shape[1], y.shape[1]
    labels, signs, weights = method_inputs(prepared, method, tau, protocol, known_labels)
    fractions = np.asarray(lambda_fractions_for(method, protocol)
                           if fractions is None else fractions)
    corr = 2*x.T@y/n
    if method == "unilasso":
        features = [unilasso_features(x, y[:, k]) for k in range(q)]
        lam_max = max(max(float(np.max(2*(eta-eta.mean(axis=0)).T@y[:, k]/n)), 0.)
                      for k, (eta, _, _) in enumerate(features))
        lambdas = fractions*max(lam_max, 1e-12)
        betas, intercepts = np.zeros((len(lambdas), q, p)), np.zeros((len(lambdas), q))
        for k, (eta, slopes, constants) in enumerate(features):
            estimator = Lasso(positive=True, fit_intercept=True, max_iter=20000,
                              tol=1e-9, warm_start=True)
            for i, lam in enumerate(lambdas):
                if lam == 0:
                    theta, _ = nnls(eta-eta.mean(axis=0), y[:, k]-y[:, k].mean(),
                                    maxiter=20*p)
                    constant = y[:, k].mean() - eta.mean(axis=0)@theta
                else:
                    estimator.set_params(alpha=lam/2).fit(eta, y[:, k])
                    theta, constant = estimator.coef_, estimator.intercept_
                betas[i, k] = slopes*theta
                intercepts[i, k] = constant + constants@theta
        return betas, intercepts, lambdas, signs
    if method == "lasso":
        lambdas = fractions*max(np.max(np.abs(corr)), 1e-12)
        betas = np.zeros((len(lambdas), q, p))
        for k in range(q):
            estimator = Lasso(fit_intercept=False, max_iter=20000, tol=1e-9, warm_start=True)
            for i, lam in enumerate(lambdas):
                estimator.set_params(alpha=lam/2).fit(x, y[:, k])
                betas[i, k] = estimator.coef_
        return betas, np.zeros((len(lambdas), q)), lambdas, signs
    groups = [np.flatnonzero(labels == g) for g in np.unique(labels)]
    group_weights = [np.sqrt(len(g)/len(groups)) for g in groups]
    lam_parameter = cp.Parameter(nonneg=True)
    if method == "cooperative":
        pos, neg = cp.Variable((q, p), nonneg=True), cp.Variable((q, p), nonneg=True)
        beta = pos-neg
        penalty = sum(w*cp.sum(cp.norm(pos[g], 2, axis=0) + cp.norm(neg[g], 2, axis=0))
                      for g, w in zip(groups, group_weights))
        lam_max = max(max(np.max(np.linalg.norm(np.maximum(corr[:, g], 0), axis=1)),
                          np.max(np.linalg.norm(np.minimum(corr[:, g], 0), axis=1)))/w
                      for g, w in zip(groups, group_weights))
        constraints = []
    else:
        beta = cp.Variable((q, p))
        axis = 0 if method == "fcgl_unconstrained" else 1
        group_penalty = sum(w*cp.sum(cp.norm(beta[g], 2, axis=axis))
                            for g, w in zip(groups, group_weights))
        penalty = ((1-protocol.alpha)*group_penalty
                   + protocol.alpha*cp.sum(cp.multiply(weights, cp.abs(beta))))
        lam_max = np.max(np.abs(corr.T)/(protocol.alpha*weights))
        constraints = []
        for mask, action in ((signs == 0, "zero"), (signs == 1, "positive"),
                             (signs == -1, "negative")):
            if np.any(mask):
                constraints.append(beta[mask] == 0 if action == "zero" else
                                   beta[mask] >= 0 if action == "positive" else beta[mask] <= 0)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(y-x@beta.T)/n
                                    + lam_parameter*penalty), constraints)
    lambdas = fractions*max(float(lam_max), 1e-12)
    betas = []
    for lam in lambdas:
        lam_parameter.value = lam
        _solve(problem, protocol.solver)
        if beta.value is None or not np.isfinite(beta.value).all():
            raise RuntimeError("Solver produced nonfinite coefficients")
        betas.append(beta.value.copy())
    return np.asarray(betas), np.zeros((len(lambdas), q)), lambdas, signs


def cv_splits(n, folds, seed):
    """Identical row splits for all methods; no test data or truth arguments."""
    if not 2 <= folds <= n//5:
        raise ValueError("Need at least five validation rows per fold")
    chunks = np.array_split(np.random.default_rng(seed+29403).permutation(n), folds)
    return [(np.setdiff1d(np.arange(n), v), v) for v in chunks]


def tune_and_fit(x, y, method, protocol=Protocol(), seed=0, known_labels=None):
    """Select using validation outcomes only, then refit on all training rows."""
    start = time.perf_counter()
    taus = protocol.tau_grid if method in SIGNED_METHODS else (np.nan,)
    fractions = lambda_fractions_for(method, protocol)
    errors = np.zeros((len(taus), len(fractions)))
    for train, valid in cv_splits(len(x), protocol.folds, seed):
        prep = prepare(x[train], y[train], protocol.cutoff_fraction)
        xv, yv = prep.transform_x(x[valid]), prep.transform_y(y[valid])
        for t, tau in enumerate(taus):
            betas, intercepts, _, _ = fit_path(prep, method, tau, protocol, known_labels)
            predictions = np.einsum("ij,lkj->lik", xv, betas) + intercepts[:, None, :]
            # Equal response weighting in training-standardized units; unequal fold sizes weighted.
            errors[t] += np.sum((predictions-yv[None, :, :])**2, axis=(1, 2))/y.shape[1]
    errors /= len(x)
    tau_index, lambda_index = np.unravel_index(np.argmin(errors), errors.shape)
    tau, fraction = taus[tau_index], fractions[lambda_index]
    prep = prepare(x, y, protocol.cutoff_fraction)
    betas, intercepts, lambdas, signs = fit_path(
        prep, method, tau, protocol, known_labels, fractions=[fraction])
    return Fitted(betas[0], intercepts[0], prep, float(lambdas[0]), fraction, float(tau),
                  float(errors[tau_index, lambda_index]), time.perf_counter()-start, signs)


def coefficient_metrics(beta, truth, zero_tolerance=PRIMARY_ZERO_TOLERANCE):
    """Coefficient signs at a common standardized tolerance, with distinct denominators."""
    beta, truth = np.asarray(beta), np.asarray(truth)
    if zero_tolerance < 0 or not np.isfinite(beta).all():
        raise ValueError("Require nonnegative tolerance and finite fitted coefficients")
    active, selected = truth != 0, np.abs(beta) > zero_tolerance
    true_positive, false_positive = np.sum(active & selected), np.sum(~active & selected)
    fitted_signs = np.where(selected, np.sign(beta), 0)
    return dict(
        support_precision=float(true_positive/max(np.sum(selected), 1)),
        support_recall=float(true_positive/max(np.sum(active), 1)),
        support_fdp=float(false_positive/max(np.sum(selected), 1)),
        null_selection=float(false_positive/max(np.sum(~active), 1)),
        active_sign_recovery=float(np.mean(fitted_signs[active] == np.sign(truth[active]))),
        active_sign_flip=float(np.mean(
            selected[active] & (fitted_signs[active] != np.sign(truth[active])))),
        active_abstention=float(np.mean(~selected[active])),
        selected_cells=int(np.sum(selected)),
        beta_mse_standardized=float(np.mean((beta-truth)**2)),
    )


def evaluate(fitted, x_test, y_test, beta_true, tolerance):
    truth_standardized = (beta_true*fitted.prepared.x_scale
                          / fitted.prepared.y_scale[:, None])
    result = coefficient_metrics(fitted.beta, truth_standardized, tolerance)
    prediction = fitted.predict(x_test)
    error = (prediction-y_test)/fitted.prepared.y_scale
    denominator = (y_test-fitted.prepared.y_mean)/fitted.prepared.y_scale
    result.update(beta_mse_raw=float(np.mean((fitted.beta_raw-beta_true)**2)),
                  prediction_mse=float(np.mean((prediction-y_test)**2)),
                  prediction_mse_standardized=float(np.mean(error**2)),
                  prediction_r2=float(1-np.sum(error**2)/np.sum(denominator**2)))
    active = beta_true != 0
    signs = fitted.signs
    signed = np.isfinite(signs) & (signs != 0)
    result.update(
        gate_active_signed=float(np.mean(signed[active])),
        gate_active_wrong_direction=float(np.mean(
            signed[active] & (signs[active] != np.sign(beta_true[active])))),
        gate_active_excluded=float(np.mean(signs[active] == 0)),
        gate_active_unrestricted=float(np.mean(np.isnan(signs[active]))),
        gate_null_signed=float(np.mean(signed[~active])),
    )
    return result


def aggregate(rows):
    """Monte Carlo means, sample SD and SE; seeds (not response cells) are replicates."""
    excluded = {"seed", "scenario", "method", "zero_tolerance"}
    numeric = [c for c in rows.select_dtypes(include=np.number) if c not in excluded]
    group = rows.groupby(["scenario", "method", "zero_tolerance"], sort=False)
    summary = group[numeric].mean().add_suffix("_mean")
    summary = summary.join(group[numeric].std(ddof=1).add_suffix("_sd"))
    summary = summary.join(group[numeric].sem(ddof=1).add_suffix("_se"))
    summary["replications"] = group.size()
    return summary.reset_index()


def safe_output(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        raise ValueError("Output path must be absolute")
    path = path.resolve()
    if REPOSITORY_DIR in path.parents or path == REPOSITORY_DIR or any(
            part.casefold().startswith("onedrive") for part in path.parts):
        raise ValueError("Outputs must be external to the checkout and OneDrive")
    path.mkdir(parents=True, exist_ok=True)
    return path


def package_version(name):
    """Record optional solver absence without breaking an explicit CLARABEL run."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run(out, reps=20, scenarios=SCENARIOS, methods=METHODS, protocol=Protocol(), first_seed=0):
    """Write every completed fit immediately and a source/version receipt."""
    out = safe_output(out)
    if (out/"estimator_draws.csv").exists():
        raise FileExistsError("Choose a new run directory; existing draws are never overwritten")
    receipt = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                   protocol=asdict(protocol), scenarios=[asdict(s) for s in scenarios],
                   methods=list(methods), reps=reps, first_seed=first_seed,
                   zero_tolerances=ZERO_TOLERANCES, primary_zero_tolerance=PRIMARY_ZERO_TOLERANCE,
                   tolerance_units="coefficients after training X/Y standardization",
                   runtime="wall time for each method's preprocessing, CV paths and final refit",
                   tuning="common response-averaged validation MSE; training-only lambda scales",
                   lambda_scale=("HCGL/FCGL use a conservative sufficient zero-solution bound "
                                 "from their L1 term; other paths use their zero threshold. "
                                 "Finite grids are not claimed exhaustive."),
                   unilasso_lambda_fractions=lambda_fractions_for("unilasso", protocol),
                   unilasso_grid_note=("Extended after initial pilot selected fraction .001 often; "
                                       "the zero endpoint is exact NNLS with an intercept."),
                   solver_threads=1, python=platform.python_version(),
                   factorlasso_module=str(Path(factorlasso.__file__).resolve()),
                   versions={name: package_version(name) for name in (
                       "factorlasso", "numpy", "scipy", "pandas", "cvxpy",
                       "scikit-learn", "Mosek")},
                   repository_head=subprocess.check_output(
                       ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_DIR, text=True).strip(),
                   source_hashes={str(p.relative_to(REPOSITORY_DIR)):
                                  hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                  [Path(__file__),
                                   REPOSITORY_DIR/"src/factorlasso/priors/_signs.py",
                                   REPOSITORY_DIR/"pyproject.toml"]},
                   source_pyproject_version=tomllib.loads(
                       (REPOSITORY_DIR/"pyproject.toml").read_text(encoding="utf-8"))["project"]["version"])
    (out/"receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    (out/"source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    rows = []
    for scenario in scenarios:
        for seed in range(first_seed, first_seed+reps):
            x, y, xt, yt, truth, known_labels = simulate(scenario, seed)
            for method in methods:
                fit = tune_and_fit(x, y, method, protocol, seed,
                                   known_labels if method == "hcgl_known_zero" else None)
                method_fractions = lambda_fractions_for(method, protocol)
                for tolerance in ZERO_TOLERANCES:
                    row = dict(scenario=scenario.name, seed=seed, method=method,
                               zero_tolerance=tolerance, lambda_value=fit.lambda_value,
                               lambda_fraction=fit.lambda_fraction, tau=fit.tau,
                               lambda_upper_boundary=int(
                                   fit.lambda_fraction == max(method_fractions)),
                               lambda_lower_boundary=int(
                                   fit.lambda_fraction == min(method_fractions)),
                               unilasso_nnls_endpoint=int(method == "unilasso"
                                                          and fit.lambda_value == 0),
                               tau_boundary=int(method in SIGNED_METHODS and fit.tau in
                                                (min(protocol.tau_grid), max(protocol.tau_grid))),
                               cv_mse=fit.cv_mse, runtime_seconds=fit.seconds,
                               discovered_clusters=len(np.unique(fit.prepared.labels)),
                               **evaluate(fit, xt, yt, truth, tolerance))
                    rows.append(row)
                # Immediate checkpoint: a failure never silently removes completed replications.
                pd.DataFrame(rows).to_csv(out/"estimator_draws.csv", index=False)
            print(f"{scenario.name} seed {seed}: completed {len(methods)} methods", flush=True)
    draws = pd.DataFrame(rows)
    aggregate(draws).to_csv(out/"estimator_summary.csv", index=False)
    primary = draws[draws.zero_tolerance == PRIMARY_ZERO_TOLERANCE]
    baseline = primary[primary.method == "lasso"].set_index(["scenario", "seed"])
    paired = primary.copy()
    for metric in ("prediction_mse_standardized", "prediction_r2", "beta_mse_standardized",
                   "active_sign_recovery", "support_fdp"):
        paired[metric+"_minus_lasso"] = [
            r[metric]-baseline.loc[(r["scenario"], r["seed"]), metric]
            if (r["scenario"], r["seed"]) in baseline.index else np.nan
            for _, r in paired.iterrows()]
    aggregate(paired).to_csv(out/"estimator_paired_summary.csv", index=False)
    receipt["completed_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["completed_fits"] = len(primary)
    (out/"receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return draws


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=output_directory()/"estimator_benchmark")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--scenarios", nargs="+", choices=[s.name for s in SCENARIOS])
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--solver", choices=("MOSEK", "CLARABEL"), default="MOSEK")
    parser.add_argument("--adaptive-gamma", type=float, default=0.)
    args = parser.parse_args()
    if args.reps < 1:
        parser.error("--reps must be positive")
    run(args.out, args.reps,
        tuple(s for s in SCENARIOS if not args.scenarios or s.name in args.scenarios),
        tuple(args.methods or METHODS),
        Protocol(solver=args.solver, adaptive_gamma=args.adaptive_gamma), args.first_seed)


if __name__ == "__main__":
    main()
