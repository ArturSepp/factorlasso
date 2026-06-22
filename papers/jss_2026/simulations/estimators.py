"""
Unified estimator interface for the simulation harness.

Each estimator is a function with signature::

    fit_fn(X_train, y_train, reg_lambda, *, true_clusters=None) -> EstimatorResult

returning a :class:`EstimatorResult` with fitted ``beta_hat`` and runtime.
Estimators are registered in the ``ESTIMATORS`` dict by string name; the
run orchestrator looks them up from the study YAML.

Each estimator is stateless and takes a single ``reg_lambda``. Oracle-λ
selection is performed in the orchestration layer by minimising
``beta_mse_norm`` across the ``lambda_grid`` for each (regime, seed,
estimator) tuple.

factorlasso configurations
--------------------------
- ``factorlasso_lasso``                  — pure L1 (no group structure)
- ``factorlasso_grp_oracle``             — Group LASSO with true clusters
- ``factorlasso_grp_hcgl``               — Group LASSO with HCGL-discovered clusters
- ``factorlasso_grp_hcgl_sign``          — HCGL + cluster-pooled sign derivation
- ``factorlasso_grp_hcgl_sign_adapt``    — HCGL + signs + adaptive reweighting
- ``factorlasso_sgl_hcgl_sign_adapt``    — Sparse Group LASSO (l1_weight=0.1) + signs + adaptive
- ``factorlasso_unilasso``               — UniLasso, per-response univariate-guided (no grouping)
- ``factorlasso_coop_oracle``            — Cooperative LASSO on the true clusters
- ``factorlasso_coop_hcgl``              — Cooperative LASSO on HCGL-discovered clusters

These six rows give the §4-§5 ablation:

    1 (LASSO baseline)
        ↓ add group structure (oracle)
    2 ───────────────────────────────────────────────
        ↓ discover clusters via HCGL
    3 ───────────────────────────────────────────────
        ↓ add cluster-pooled sign derivation
    4 ───────────────────────────────────────────────
        ↓ add adaptive penalty reweighting
    5 ───────────────────────────────────────────────
        ↓ add Sparse Group LASSO mixing
    6

External competitor configurations
----------------------------------
Off-the-shelf sparse-regression solvers, each applied **per asset**.
This is the honest comparator geometry: factorlasso's group penalty is
``lam * sum_g sqrt(|g|/G) * sum_{k in g} ||beta_k||_2`` (see the group
objective in :mod:`factorlasso.lasso_estimator`), which is separable
across assets — each asset's loadings form their own L2 group and the
cluster only sets a per-asset weight. A per-asset external solver
therefore targets exactly the same coefficient geometry; the
distinctive factorlasso layers the competitors lack are the
cluster-pooled *sign derivation* and the adaptive reweighting.

Wired (each lazy-imports its package; a missing package raises
``NotImplementedError`` with an install hint so the study YAML can
still declare the estimator)::

- ``sklearn_lasso``     — sklearn Lasso, per-asset L1 baseline
- ``skglm_grouplasso``  — skglm GroupLasso, per-asset single-group (row sparsity)
- ``asgl_sgl``          — asgl sparse-group LASSO, per-asset (l1-fraction 0.1)
- ``adelie_grp``        — adelie grpnet group LASSO, per-asset single-group

Not available in a pip-only environment::

- ``sparsegl_sgl``      — sparsegl is an R/CRAN package; wire via rpy2 + R
                          or load offline R fits. Left as a documented stub.

Each external is tuned on its own oracle-lambda path by the comparison
driver, so the package-specific penalty scale is immaterial; the metric
is best-achievable recovery, not behaviour at a shared lambda.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Dict, Optional

import cvxpy as cvx
import numpy as np
import pandas as pd

from factorlasso import LassoModel, LassoModelType

# Solver fallback chain. CLARABEL is the modern default — fast on
# well-conditioned QPs — but occasionally fails on degenerate problem
# instances (e.g. orthogonal factor cov + cluster-pooled sign constraints
# at certain β-geometries). ECOS is a more conservative interior-point
# method that handles many such cases; SCS is a first-order splitting
# method that's slower but very robust. The chain tries them in order;
# whichever solves first wins, and the choice is recorded in
# ``EstimatorResult.extra['solver_used']`` for downstream auditing.
SOLVER_FALLBACK_CHAIN: tuple[str, ...] = ("CLARABEL", "ECOS", "SCS")


@dataclass
class EstimatorResult:
    """One fit's output. ``extra`` carries method-specific diagnostics."""

    name: str
    reg_lambda: float
    beta_hat: pd.DataFrame  # (N, M)
    intercept_hat: pd.Series  # (N,)
    runtime: float  # seconds, wall-clock for the fit call only
    extra: dict = field(default_factory=dict)


# Registry populated by ``@register("name")`` decorator
ESTIMATORS: Dict[str, Callable[..., EstimatorResult]] = {}


def register(name: str):
    """Decorator to add a function to ``ESTIMATORS`` under the given name."""

    def decorator(func):
        if name in ESTIMATORS:
            raise ValueError(f"Estimator {name!r} already registered")
        ESTIMATORS[name] = func
        func.__estimator_name__ = name
        return func

    return decorator


# ── factorlasso wrappers ──────────────────────────────────────────────


def _factorlasso_fit(
    model_kwargs: dict,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    name: str,
    reg_lambda: float,
) -> EstimatorResult:
    """
    Common factorlasso fit-and-package helper with solver fallback chain.

    Tries solvers in order :data:`SOLVER_FALLBACK_CHAIN` (CLARABEL →
    ECOS → SCS). Whichever succeeds first wins, and the choice is
    recorded in ``EstimatorResult.extra['solver_used']``. Only
    ``cvxpy.error.SolverError`` triggers fallback — DCP errors,
    ValueErrors, and so on are programmer/data bugs that should
    propagate unmodified.

    If the model_kwargs explicitly sets ``solver=...``, that solver is
    tried first; the remaining chain follows in order.

    If all solvers in the chain fail, the *original* (first) exception
    is re-raised so the diagnostic message is unchanged from a
    no-fallback run.
    """
    primary_solver = model_kwargs.get("solver", "CLARABEL")
    candidates = [primary_solver] + [
        s for s in SOLVER_FALLBACK_CHAIN if s != primary_solver
    ]

    first_exc: Optional[Exception] = None
    for solver_name in candidates:
        kwargs = {**model_kwargs, "solver": solver_name}
        t0 = perf_counter()
        try:
            model = LassoModel(**kwargs).fit(x=X_train, y=y_train)
        except cvx.error.SolverError as exc:
            if first_exc is None:
                first_exc = exc
            continue
        runtime = perf_counter() - t0

        intercept = (
            model.alpha_const_ if model.alpha_const_ is not None else model.intercept_
        )

        extra: dict[str, Any] = {"solver_used": solver_name}
        if model.clusters_ is not None:
            extra["clusters_hat"] = model.clusters_
        if model.derived_signs_ is not None:
            extra["derived_signs"] = model.derived_signs_

        return EstimatorResult(
            name=name,
            reg_lambda=reg_lambda,
            beta_hat=model.coef_,
            intercept_hat=intercept,
            runtime=runtime,
            extra=extra,
        )

    # All solvers exhausted — re-raise the original failure
    assert first_exc is not None
    raise first_exc


@register("factorlasso_lasso")
def fit_factorlasso_lasso(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    reg_lambda: float,
    *,
    true_clusters: Optional[pd.Series] = None,
) -> EstimatorResult:
    """Pure L1 baseline. No group structure, no sign constraints."""
    kwargs = dict(
        model_type=LassoModelType.LASSO,
        reg_lambda=reg_lambda,
    )
    return _factorlasso_fit(kwargs, X_train, y_train, "factorlasso_lasso", reg_lambda)


@register("factorlasso_grp_oracle")
def fit_factorlasso_grp_oracle(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    reg_lambda: float,
    *,
    true_clusters: Optional[pd.Series] = None,
) -> EstimatorResult:
    """
    Group LASSO with the TRUE cluster assignment passed in. Oracle
    baseline — represents the unattainable "best case" where the
    practitioner knows the cluster structure exactly. HCGL methods
    should approach this baseline as cluster recovery improves.
    """
    if true_clusters is None:
        raise ValueError("factorlasso_grp_oracle requires true_clusters")
    kwargs = dict(
        model_type=LassoModelType.GROUP_LASSO,
        reg_lambda=reg_lambda,
        group_data=true_clusters,
    )
    return _factorlasso_fit(
        kwargs, X_train, y_train, "factorlasso_grp_oracle", reg_lambda,
    )


@register("factorlasso_grp_hcgl")
def fit_factorlasso_grp_hcgl(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    reg_lambda: float,
    *,
    true_clusters: Optional[pd.Series] = None,
) -> EstimatorResult:
    """
    Group LASSO with HCGL-discovered clusters. The headline factorlasso
    contribution at the clustering layer — same group structure benefit
    as oracle, but without requiring the cluster labels as input.
    """
    kwargs = dict(
        model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
        reg_lambda=reg_lambda,
        cutoff_fraction=0.5,
    )
    return _factorlasso_fit(
        kwargs, X_train, y_train, "factorlasso_grp_hcgl", reg_lambda,
    )


@register("factorlasso_grp_hcgl_sign")
def fit_factorlasso_grp_hcgl_sign(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    reg_lambda: float,
    *,
    true_clusters: Optional[pd.Series] = None,
) -> EstimatorResult:
    """HCGL + cluster-pooled sign derivation. Tests the sign-mechanism."""
    kwargs = dict(
        model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
        reg_lambda=reg_lambda,
        cutoff_fraction=0.5,
        auto_sign_constraints=True,
        auto_sign_threshold_t=0.75,
    )
    return _factorlasso_fit(
        kwargs, X_train, y_train, "factorlasso_grp_hcgl_sign", reg_lambda,
    )


@register("factorlasso_grp_hcgl_sign_adapt")
def fit_factorlasso_grp_hcgl_sign_adapt(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    reg_lambda: float,
    *,
    true_clusters: Optional[pd.Series] = None,
) -> EstimatorResult:
    """HCGL + signs + Zou/Wang-Leng adaptive reweighting. Production config."""
    kwargs = dict(
        model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
        reg_lambda=reg_lambda,
        cutoff_fraction=0.5,
        auto_sign_constraints=True,
        auto_sign_threshold_t=0.75,
        auto_sign_adaptive_weights=True,
        auto_sign_adaptive_gamma=1.0,
    )
    return _factorlasso_fit(
        kwargs, X_train, y_train, "factorlasso_grp_hcgl_sign_adapt", reg_lambda,
    )


@register("factorlasso_sgl_hcgl_sign_adapt")
def fit_factorlasso_sgl_hcgl_sign_adapt(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    reg_lambda: float,
    *,
    true_clusters: Optional[pd.Series] = None,
) -> EstimatorResult:
    """Sparse Group LASSO (l1_weight=0.1) + signs + adaptive."""
    kwargs = dict(
        model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
        reg_lambda=reg_lambda,
        cutoff_fraction=0.5,
        l1_weight=0.1,
        auto_sign_constraints=True,
        auto_sign_threshold_t=0.75,
        auto_sign_adaptive_weights=True,
        auto_sign_adaptive_gamma=1.0,
    )
    return _factorlasso_fit(
        kwargs, X_train, y_train, "factorlasso_sgl_hcgl_sign_adapt", reg_lambda,
    )


# ── factorlasso family members outside the ablation chain ────────


@register("factorlasso_unilasso")
def fit_factorlasso_unilasso(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    reg_lambda: float,
    *,
    true_clusters: Optional[pd.Series] = None,
) -> EstimatorResult:
    """
    UniLasso (Chatterjee, Hastie & Tibshirani 2025): per-response
    two-stage univariate-guided fit with no grouping. Stage one fits the
    univariate slope of each response on each factor; stage two combines
    them with a non-negative coefficient, so each loading keeps the sign
    of its univariate slope. ``group_data`` and ``cutoff_fraction`` are
    ignored.
    """
    kwargs = dict(
        model_type=LassoModelType.UNILASSO,
        reg_lambda=reg_lambda,
        unilasso_loo=True,
        unilasso_non_negative=True,
    )
    return _factorlasso_fit(
        kwargs, X_train, y_train, "factorlasso_unilasso", reg_lambda,
    )


@register("factorlasso_coop_oracle")
def fit_factorlasso_coop_oracle(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    reg_lambda: float,
    *,
    true_clusters: Optional[pd.Series] = None,
) -> EstimatorResult:
    """
    Cooperative LASSO (Chiquet, Grandvalet & Charbonnier 2012) on the
    TRUE cluster assignment. The cooperative penalty splits each
    coefficient into a non-negative and a non-positive part penalised as
    separate groups, so members of a group tend to share a sign while the
    data can overrule it. Oracle group analogue of the cooperative mode.
    """
    if true_clusters is None:
        raise ValueError("factorlasso_coop_oracle requires true_clusters")
    kwargs = dict(
        model_type=LassoModelType.COOPERATIVE_GROUP_LASSO,
        reg_lambda=reg_lambda,
        group_data=true_clusters,
    )
    return _factorlasso_fit(
        kwargs, X_train, y_train, "factorlasso_coop_oracle", reg_lambda,
    )


@register("factorlasso_coop_hcgl")
def fit_factorlasso_coop_hcgl(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    reg_lambda: float,
    *,
    true_clusters: Optional[pd.Series] = None,
) -> EstimatorResult:
    """
    Cooperative LASSO on HCGL-discovered clusters. Same soft
    sign-coherence penalty as ``factorlasso_coop_oracle`` but on the
    partition discovered from ``corr(Y)`` rather than supplied.
    """
    kwargs = dict(
        model_type=LassoModelType.COOPERATIVE_CLUSTER_GROUP_LASSO,
        reg_lambda=reg_lambda,
        cutoff_fraction=0.5,
    )
    return _factorlasso_fit(
        kwargs, X_train, y_train, "factorlasso_coop_hcgl", reg_lambda,
    )


# ── External competitor stubs ─────────────────────────────────────────


@register("sklearn_lasso")
def fit_sklearn_lasso(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    reg_lambda: float,
    *,
    true_clusters: Optional[pd.Series] = None,
) -> EstimatorResult:
    """
    sklearn.linear_model.Lasso baseline, applied per-asset.

    sklearn parametrises by alpha = reg_lambda; we pass through directly
    so the orchestrator's lambda_grid maps 1:1. The per-asset loop is
    the only available adaptation since sklearn's Lasso is single-
    response.

    Requires ``scikit-learn``.
    """
    try:
        from sklearn.linear_model import Lasso
    except ImportError as exc:
        raise NotImplementedError(
            "sklearn_lasso estimator requires scikit-learn. "
            "Install with: pip install scikit-learn"
        ) from exc

    t0 = perf_counter()
    beta_rows = []
    intercepts = []
    for asset in y_train.columns:
        m = Lasso(alpha=reg_lambda, fit_intercept=True, max_iter=10_000)
        m.fit(X_train.values, y_train[asset].values)
        beta_rows.append(m.coef_)
        intercepts.append(m.intercept_)
    runtime = perf_counter() - t0

    beta_hat = pd.DataFrame(
        np.vstack(beta_rows), index=y_train.columns, columns=X_train.columns,
    )
    intercept_hat = pd.Series(intercepts, index=y_train.columns)

    return EstimatorResult(
        name="sklearn_lasso",
        reg_lambda=reg_lambda,
        beta_hat=beta_hat,
        intercept_hat=intercept_hat,
        runtime=runtime,
    )


@register("sparsegl_sgl")
def fit_sparsegl_sgl(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    reg_lambda: float,
    *,
    true_clusters: Optional[pd.Series] = None,
) -> EstimatorResult:
    """
    sparsegl (Sparse Group LASSO) external comparison.

    Wire-up notes:

    - The native package is R. A Python port exists but is not the
      reference implementation; for a definitive comparison call R via
      rpy2 or run the R fits offline and load the results.
    - Asset-by-asset application is required (sparsegl is single-
      response).
    - Group structure: pass the TRUE clusters as the group_id argument
      (this is the only fair comparison: sparsegl does not perform
      HCGL discovery).

    Not wired: sparsegl is an R/CRAN package with no reference Python
    implementation. In a pip-only environment it is unavailable. To
    include it, install R + the sparsegl package and call it through
    rpy2, or run the R fits offline and load the coefficients; apply
    per-asset with the TRUE clusters as ``group_id`` (sparsegl does not
    do HCGL discovery). Left as a documented stub so the study YAML can
    declare it and the runner reports it as not-implemented rather than
    silently substituting a different solver.
    """
    raise NotImplementedError(
        "sparsegl_sgl is an R/CRAN package, not available in a pip-only "
        "environment. See estimators.fit_sparsegl_sgl docstring for the "
        "rpy2 / offline-fit wiring path."
    )


@register("adelie_grp")
def fit_adelie_grp(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    reg_lambda: float,
    *,
    true_clusters: Optional[pd.Series] = None,
) -> EstimatorResult:
    """
    adelie (Yang & Hastie 2024) group LASSO, per-asset single group.

    Each asset's M loadings are one group, so the group penalty is
    ``reg_lambda * ||beta_k||_2`` — the per-asset analogue of
    factorlasso's GROUP_LASSO geometry, solved by adelie's block
    coordinate descent rather than CVXPY. adelie standardises and fits
    an unpenalised intercept by default; ``reg_lambda`` is taken on
    adelie's native ``lmda`` scale (the comparison driver tunes it on an
    oracle-lambda path, so the scale convention is immaterial).

    Requires ``adelie`` (``pip install adelie``).
    """
    try:
        import adelie as ad
    except ImportError as exc:
        raise NotImplementedError(
            "adelie_grp estimator requires adelie. "
            "Install with: pip install adelie"
        ) from exc

    Xv = np.asarray(X_train.values, dtype=float)
    groups = np.array([0])  # single group spanning all M features

    t0 = perf_counter()
    beta_rows = []
    intercepts = []
    for asset in y_train.columns:
        yv = np.asarray(y_train[asset].values, dtype=float)
        glm = ad.glm.gaussian(y=yv.copy())
        fit = ad.grpnet(
            X=Xv.copy(), glm=glm, groups=groups,
            lmda_path=np.array([float(reg_lambda)]),
            intercept=True, progress_bar=False,
        )
        betas = fit.betas
        b_arr = np.asarray(betas.todense()) if hasattr(betas, "todense") \
            else np.asarray(betas)
        beta_rows.append(b_arr[-1])
        intercepts.append(float(np.asarray(fit.intercepts).ravel()[-1]))
    runtime = perf_counter() - t0

    beta_hat = pd.DataFrame(
        np.vstack(beta_rows), index=y_train.columns, columns=X_train.columns,
    )
    intercept_hat = pd.Series(intercepts, index=y_train.columns)
    return EstimatorResult(
        name="adelie_grp",
        reg_lambda=reg_lambda,
        beta_hat=beta_hat,
        intercept_hat=intercept_hat,
        runtime=runtime,
    )


@register("skglm_grouplasso")
def fit_skglm_grouplasso(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    reg_lambda: float,
    *,
    true_clusters: Optional[pd.Series] = None,
) -> EstimatorResult:
    """
    skglm GroupLasso, per-asset single group (all M factors one group).

    Solves ``(1/2T)||y_k - X beta_k||^2 + reg_lambda * sqrt(M) ||beta_k||_2``
    per asset by coordinate descent — the row-sparse (whole-asset
    in/out) analogue of factorlasso's GROUP_LASSO, with skglm's default
    sqrt(group-size) weighting. Tuned on an oracle-lambda path by the
    comparison driver.

    Requires ``skglm`` (``pip install skglm``).
    """
    try:
        from skglm import GroupLasso
    except ImportError as exc:
        raise NotImplementedError(
            "skglm_grouplasso estimator requires skglm. "
            "Install with: pip install skglm"
        ) from exc

    M = X_train.shape[1]
    Xv = X_train.values

    t0 = perf_counter()
    beta_rows = []
    intercepts = []
    for asset in y_train.columns:
        m = GroupLasso(
            groups=M, alpha=reg_lambda, fit_intercept=True,
            tol=1e-8, max_iter=100, max_epochs=100_000,
        )
        m.fit(Xv, y_train[asset].values)
        beta_rows.append(np.asarray(m.coef_).ravel())
        intercepts.append(float(np.ravel(m.intercept_)[0]))
    runtime = perf_counter() - t0

    beta_hat = pd.DataFrame(
        np.vstack(beta_rows), index=y_train.columns, columns=X_train.columns,
    )
    intercept_hat = pd.Series(intercepts, index=y_train.columns)
    return EstimatorResult(
        name="skglm_grouplasso",
        reg_lambda=reg_lambda,
        beta_hat=beta_hat,
        intercept_hat=intercept_hat,
        runtime=runtime,
    )


@register("asgl_sgl")
def fit_asgl_sgl(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    reg_lambda: float,
    *,
    true_clusters: Optional[pd.Series] = None,
) -> EstimatorResult:
    """
    asgl sparse-group LASSO, per-asset single group.

    The single-group SGL penalty is
    ``reg_lambda * (0.1 ||beta_k||_1 + 0.9 ||beta_k||_2)`` per asset —
    the per-asset analogue of factorlasso's SGL configuration
    (``l1_weight = 0.1``) minus the sign-derivation gate. The L1
    fraction ``alpha = 0.1`` mirrors factorlasso's ``l1_weight``. Tuned
    on an oracle-lambda path by the comparison driver.

    Requires ``asgl`` (``pip install asgl``).
    """
    try:
        from asgl import Regressor
    except ImportError as exc:
        raise NotImplementedError(
            "asgl_sgl estimator requires asgl. "
            "Install with: pip install asgl"
        ) from exc

    M = X_train.shape[1]
    Xv = X_train.values
    group_index = np.ones(M, dtype=int)

    t0 = perf_counter()
    beta_rows = []
    intercepts = []
    for asset in y_train.columns:
        r = Regressor(
            model="lm", penalization="sgl",
            lambda1=reg_lambda, alpha=0.1, fit_intercept=True,
        )
        r.fit(Xv, y_train[asset].values, group_index=group_index)
        beta_rows.append(np.asarray(r.coef_).ravel())
        ic = getattr(r, "intercept_", 0.0)
        intercepts.append(float(np.ravel(ic)[0]) if ic is not None else 0.0)
    runtime = perf_counter() - t0

    beta_hat = pd.DataFrame(
        np.vstack(beta_rows), index=y_train.columns, columns=X_train.columns,
    )
    intercept_hat = pd.Series(intercepts, index=y_train.columns)
    return EstimatorResult(
        name="asgl_sgl",
        reg_lambda=reg_lambda,
        beta_hat=beta_hat,
        intercept_hat=intercept_hat,
        runtime=runtime,
    )


# ── Convenience helpers ──────────────────────────────────────────────


def list_estimators() -> list[str]:
    """Names of all registered estimators."""
    return sorted(ESTIMATORS.keys())


def is_wired(name: str) -> bool:
    """
    Probe whether an estimator is implemented (vs. a NotImplementedError stub).

    Runs the fit function on a tiny synthetic instance and catches
    NotImplementedError. Imports etc. are also exercised; returns False
    if any dependency is missing.
    """
    try:
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((30, 3)), columns=["F0", "F1", "F2"])
        Y = pd.DataFrame(rng.standard_normal((30, 5)), columns=[f"A{i}" for i in range(5)])
        clusters = pd.Series([0, 0, 1, 1, 2], index=Y.columns)
        ESTIMATORS[name](X, Y, reg_lambda=1e-3, true_clusters=clusters)
        return True
    except NotImplementedError:
        return False
    except Exception:
        # Unexpected error — re-raise so the user sees real problems
        raise
