"""
Univariate-derived sign constraints for :class:`~factorlasso.LassoModel`.
============================================================================

This module supplies the sign matrix consumed by
:attr:`LassoModel.factors_beta_loading_signs`. The user picks the regressor
columns and (optionally) their cluster grouping; :func:`derive_sign_constraints`
returns a DataFrame ready to drop in:

    >>> signs = derive_sign_constraints(x=X, y=Y, clusters=clusters)
    >>> model = LassoModel(factors_beta_loading_signs=signs).fit(x=X, y=Y)

Convention is identical to ``LassoModel.factors_beta_loading_signs``:

* ``+1``  → non-negative
* ``-1``  → non-positive
* ``0``   → forced to zero
* ``NaN`` → unconstrained (only producible via ``master_constraints``)

Two derivation modes share the same entry point:

* **column-level** (``clusters=None``) — one sign per regressor from its own
  pooled univariate slope. Allows within-cluster sign disagreement if two
  highly-correlated regressors have opposite marginal correlation with y.

* **cluster-level** (``clusters=<array>``) — one sign per cluster from the
  slope of the cluster-mean regressor vs y, broadcast to every cluster
  member. Guarantees within-cluster sign coherence — the property that
  eliminates artificial long/short alternations within tightly-collinear
  regressor groups.

No preprocessing is done on ``x`` or ``y`` — any centering, standardization,
residualization etc. is the caller's responsibility. Optional ``ewma_span``
weights observations on their original row grid; it does not demean the data.

References
----------
The univariate-slope-as-sign-constraint mechanism is adapted from the
uniLasso framework:

* Chatterjee, S., Hastie, T., & Tibshirani, R. (2025). Univariate-guided
  sparse regression. *Harvard Data Science Review* 7(3).
* Richland, J., Kiiskinen, T., Wang, W., Lu, S., Narasimhan, B., Hastie,
  T., Rivas, M., & Tibshirani, R. (2025). Univariate-guided sparse
  regression for biobank-scale high-dimensional -omics data.
  arXiv:2511.22049.

Specifically, Richland et al. (2025) eq. (3.3) imposes
``sign(γ_j) = sign(β̃_j)`` as a hard constraint on the original variables
— structurally identical to what ``factors_beta_loading_signs`` encodes
in factorlasso. The broader use of univariate evidence to guide a
multivariate fit goes back to Zou (2006)'s adaptive Lasso. The
``auto_sign_threshold_t`` noise-floor gate is conceptually closer to
Fan & Lv (2008)'s Sure Independence Screening; it is not part of
uniLasso, which achieves smoother noise downweighting via its
leave-one-out stage-2 reparameterization.

The date-score sandwich follows the one-way cluster-score construction of
Cameron, A. C., and Miller, D. L. (2015), A Practitioner's Guide to
Cluster-Robust Inference, Journal of Human Resources 50(2), 317-372,
doi:10.3368/jhr.50.2.317. Kish effective-date scaling for EWMA is an
implementation convention, not a calibrated significance theorem; temporal
serial dependence is not covered.
"""
from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd




def _sign_observation_weights(size, ewma_span):
    """Build decay on the original grid; missing rows never compress time."""
    if ewma_span is None:
        return np.ones(size, dtype=float)
    if not np.isscalar(ewma_span) or not np.isfinite(ewma_span) or ewma_span < 1:
        raise ValueError("ewma_span must be finite and >= 1, or None")
    return (1.0 - 2.0 / (ewma_span + 1.0)) ** np.arange(size - 1, -1, -1)


def _pooled_sign_statistics(x_arr, y_arr, ewma_span=None, variance_estimator="independent"):
    """Masked WLS slopes and score variances, without implicit centering.

    ``date`` sums response scores within a date before squaring (one-way
    sandwich), retaining contemporaneous covariance. Its HC1-style correction
    uses Kish effective *dates*, never the number of response cells. It assumes
    independence across dates; it is not HAC or a calibrated Student t test.

    ``independent`` is a reproduction/ablation option: unweighted data use the
    historical pooled homoskedastic formula. Weighted data use independent-cell
    score variance, not an inverse-variance interpretation of recency weights.
    """
    if variance_estimator not in ("date", "independent"):
        raise ValueError("variance_estimator must be 'date' or 'independent'")
    x_arr = np.asarray(x_arr, dtype=float)
    y_arr = np.asarray(y_arr, dtype=float)
    if x_arr.ndim != 2 or y_arr.ndim != 2 or len(x_arr) != len(y_arr):
        raise ValueError("x and y must be 2-D arrays with the same number of rows")
    if np.isinf(x_arr).any() or np.isinf(y_arr).any():
        raise ValueError("sign analytics require finite observations or NaN")
    weights = _sign_observation_weights(len(x_arr), ewma_span)
    vx, vy = ~np.isnan(x_arr), ~np.isnan(y_arr)
    x = np.nan_to_num(x_arr, nan=0.)
    y = np.nan_to_num(y_arr, nan=0.)
    count = vy.sum(axis=1)
    y_sum = y.sum(axis=1)
    denominator = (x*x).T @ (weights * count)
    numerator = x.T @ (weights * y_sum)
    slopes = np.divide(numerator, denominator, out=np.zeros(x.shape[1]),
                       where=denominator > 0)
    present = vx & (count[:, None] > 0)
    weighted_dates = weights[:, None] * present
    weight_mass = weighted_dates.sum(axis=0)
    weight_sq_mass = (weighted_dates**2).sum(axis=0)
    effective_n = np.divide(weight_mass**2, weight_sq_mass,
                            out=np.zeros_like(weight_mass), where=weight_sq_mass > 0)
    n_obs = vx.astype(float).T @ count
    if variance_estimator == "date":
        # u_tj = w_t x_tj sum_k v_tjk (y_tk - beta_j x_tj).
        scores = weights[:, None]*x*(y_sum[:, None]-x*slopes*count[:, None])
        meat = np.sum(scores*scores, axis=0)
        correction = np.divide(effective_n, effective_n-1.,
                               out=np.ones_like(effective_n),
                               where=effective_n > 1.+1e-12)
        variance = np.divide(meat*correction, denominator**2,
                             out=np.full_like(slopes, np.inf), where=denominator > 0)
    else:
        q_eff = np.sum((vx.astype(float).T @ vy.astype(float)) > 0, axis=1)
        if ewma_span is None:
            y_ss = vx.astype(float).T @ (y*y).sum(axis=1)
            ssr = np.maximum(y_ss-slopes*slopes*denominator, 0.)
            variance = np.divide(ssr, np.maximum(n_obs-q_eff, 1.)*denominator,
                                 out=np.full_like(slopes, np.inf), where=denominator > 0)
        else:
            # Sum squared *cell* scores: deliberately omits within-date covariance.
            residual_ss = ((y*y).sum(axis=1)[:, None]
                           - 2.*x*slopes*y_sum[:, None]
                           + x*x*slopes*slopes*count[:, None])
            meat = np.sum(weights[:, None]**2*x*x*np.maximum(residual_ss, 0.), axis=0)
            sw = vx.astype(float).T @ (weights*count)
            sw2 = vx.astype(float).T @ (weights**2*count)
            cell_ess = np.divide(sw*sw, sw2, out=np.zeros_like(sw), where=sw2 > 0)
            correction = np.divide(cell_ess, cell_ess-q_eff,
                                   out=np.ones_like(sw), where=cell_ess > q_eff+1e-12)
            variance = np.divide(meat*correction, denominator**2,
                                 out=np.full_like(slopes, np.inf), where=denominator > 0)
    sufficient = (effective_n > 1.+1e-12) & (denominator > 0)
    variance[~sufficient] = np.inf
    se = np.sqrt(np.maximum(variance, 0.))
    t_stats = np.divide(slopes, se, out=np.zeros_like(slopes), where=se > 0)
    # A nonzero exact signal has no residual noise; a zero/absent factor has no evidence.
    perfect = sufficient & (se == 0) & (slopes != 0)
    t_stats[perfect] = np.sign(slopes[perfect])*np.inf
    return dict(slopes=slopes, t_stats=t_stats, standard_errors=se,
                effective_n=effective_n, n_obs=n_obs, weight_mass=weight_mass)


def _compute_sign_vector(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    clusters: Optional[np.ndarray] = None,
    master_constraints: Optional[dict] = None,
    col_names: Optional[list] = None,
    auto_sign_threshold_t: Optional[float] = None,
    ewma_span: Optional[float] = None,
    variance_estimator: str = "independent",
    return_diagnostics: bool = False,
) -> tuple:
    """Derive pooled marginal signs, retaining masks and optional recency weights.

    Factor clusters use their complete-row mean; response columns are pooled.
    ``date`` variance treats rows as independent and responses as dependent.
    ``independent`` reproduces the historical equal-weight gate for ablation.
    Master constraints override the automatic gate. With diagnostics requested,
    append a dictionary of pre-override slopes, score statistics and sample sizes.
    """
    x_arr = np.asarray(x_arr, dtype=float)
    y_arr = np.asarray(y_arr, dtype=float)
    if x_arr.ndim != 2:
        raise ValueError("x must be 2-D")
    m = x_arr.shape[1]
    if clusters is None:
        diag = _pooled_sign_statistics(x_arr, y_arr, ewma_span, variance_estimator)
    else:
        clusters_arr = np.asarray(clusters)
        if clusters_arr.ndim != 1 or len(clusters_arr) != m:
            raise ValueError(f"clusters length {len(clusters_arr)} != M={m}")
        ids, inverse = np.unique(clusters_arr, return_inverse=True)
        aggregated = np.column_stack([x_arr[:, clusters_arr == c].mean(axis=1) for c in ids])
        base = _pooled_sign_statistics(aggregated, y_arr, ewma_span, variance_estimator)
        diag = {key: value[inverse] for key, value in base.items()}
    slopes = diag["slopes"]
    signs = np.sign(slopes).astype(float)
    if auto_sign_threshold_t is not None and auto_sign_threshold_t > 0:
        signs[np.abs(diag["t_stats"]) < auto_sign_threshold_t] = 0.
    if master_constraints:
        for key, s in master_constraints.items():
            if s is None or (isinstance(s, float) and np.isnan(s)):
                s_val = np.nan
            elif s in (-1, 0, 1):
                s_val = float(s)
            else:
                raise ValueError(
                    f"master_constraints[{key!r}]={s}; must be in {{-1, 0, +1}}, NaN, or None")
            if isinstance(key, str):
                if col_names is None:
                    raise ValueError(
                        f"name {key!r} in master_constraints but no column names were available")
                if key not in col_names:
                    raise KeyError(f"master_constraints key {key!r} not found in col_names")
                idx = col_names.index(key)
            else:
                idx = int(key)
                if not 0 <= idx < m:
                    raise IndexError(f"master_constraints index {idx} out of range for M={m}")
            signs[idx] = s_val
    if return_diagnostics:
        return signs, slopes, diag
    return signs, slopes


def _compute_sign_matrix_per_response(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    auto_sign_threshold_t: Optional[float] = None,
    return_slopes: bool = False,
    ewma_span: Optional[float] = None,
    variance_estimator: str = "independent",
    return_diagnostics: bool = False,
) -> Union[np.ndarray, tuple]:
    """Vectorized single-response WLS/score statistics, returning (N, M) arrays.

    Response-independent fits have one score per date. Polynomial moment
    expansions avoid materialising a T-by-N-by-M residual tensor. Diagnostic
    mode returns signs, slopes and a dictionary, regardless of return_slopes.
    """
    if variance_estimator not in ('date', 'independent'):
        raise ValueError("variance_estimator must be 'date' or 'independent'")
    x_arr, y_arr = np.asarray(x_arr, dtype=float), np.asarray(y_arr, dtype=float)
    if x_arr.ndim != 2 or y_arr.ndim != 2 or len(x_arr) != len(y_arr):
        raise ValueError("x and y must be 2-D arrays with the same number of rows")
    if np.isinf(x_arr).any() or np.isinf(y_arr).any():
        raise ValueError("sign analytics require finite observations or NaN")
    w = _sign_observation_weights(len(x_arr), ewma_span)[:, None]
    vx, vy = (~np.isnan(x_arr)).astype(float), (~np.isnan(y_arr)).astype(float)
    x, y = np.nan_to_num(x_arr, nan=0.), np.nan_to_num(y_arr, nan=0.)
    d = vy.T @ (w*x*x)
    slopes = np.divide(y.T @ (w*x), d, out=np.zeros_like(d), where=d > 0)
    sw = vy.T @ (w*vx)
    sw2 = vy.T @ (w*w*vx)
    ess = np.divide(sw*sw, sw2, out=np.zeros_like(d), where=sw2 > 0)
    n_obs = vy.T @ vx
    if variance_estimator == 'independent' and ewma_span is None:
        ssr = np.maximum((y*y).T @ vx-slopes*slopes*d, 0.)
        variance = np.divide(ssr, np.maximum(n_obs-1., 1.)*d,
                             out=np.full_like(d, np.inf), where=d > 0)
    else:
        meat = ((y*y).T @ (w*w*x*x) - 2.*slopes*(y.T @ (w*w*x*x*x))
                + slopes*slopes*(vy.T @ (w*w*x*x*x*x)))
        correction = np.divide(ess, ess-1., out=np.ones_like(d), where=ess > 1.+1e-12)
        variance = np.divide(np.maximum(meat, 0.)*correction, d*d,
                             out=np.full_like(d, np.inf), where=d > 0)
    sufficient = (ess > 1.+1e-12) & (d > 0)
    variance[~sufficient] = np.inf
    se = np.sqrt(variance)
    t_stats = np.divide(slopes, se, out=np.zeros_like(slopes), where=se > 0)
    perfect = sufficient & (se == 0) & (slopes != 0)
    t_stats[perfect] = np.sign(slopes[perfect])*np.inf
    signs = np.sign(slopes)
    if auto_sign_threshold_t is not None and auto_sign_threshold_t > 0:
        signs[np.abs(t_stats) < auto_sign_threshold_t] = 0.
    if return_diagnostics:
        return signs, slopes, dict(slopes=slopes, t_stats=t_stats,
            standard_errors=se, effective_n=ess, n_obs=n_obs, weight_mass=sw)
    return (signs, slopes) if return_slopes else signs


def derive_sign_constraints(
    x: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.DataFrame, pd.Series, np.ndarray],
    clusters: Optional[Union[np.ndarray, pd.Series, list]] = None,
    master_constraints: Optional[dict] = None,
    auto_sign_threshold_t: Optional[float] = 0.75,
    return_slopes: bool = False,
    ewma_span: Optional[float] = None,
    variance_estimator: str = "independent",
) -> Union[pd.DataFrame, np.ndarray, tuple]:
    """
    Compute masked, optionally recency-weighted marginal sign constraints.

    ``ewma_span=None`` gives equal observation weights; finite spans >= 1
    apply EWMA decay before masking. ``variance_estimator="date"`` uses
    a date-score sandwich with a Kish-date HC1 correction; it accounts
    for contemporaneous response dependence, not serial correlation.
    The gate is a screening statistic, not a Student t significance test.
    ``"independent"`` retains the former equal-weight gate for replication.
    Fewer than two effective dates cannot pass a positive threshold.

    For each regressor column ``x_j``::

        β̂_j = (x_j · y_sum) / (q · ||x_j||²),    y_sum = Σ_k y[:, k]

    Sign(β̂_j) is the data-derived constraint. When ``clusters`` is provided,
    the slope is instead computed against the cluster-mean regressor and
    broadcast to every cluster member, guaranteeing within-cluster sign
    coherence.

    Parameters
    ----------
    x : pd.DataFrame or np.ndarray, shape (T, M)
        Regressor (factor) data.
    y : pd.DataFrame, pd.Series, or np.ndarray, shape (T, N) or (T,)
        Response (asset) data. Multi-response slopes are pooled (averaged)
        across response columns.
    clusters : array-like of length M, optional
        Per-regressor cluster IDs. If ``None`` (default), runs in column-level
        mode. If provided, runs in cluster-level mode. Length must equal M.
    master_constraints : dict, optional
        ``{regressor_name_or_index: sign}`` where sign is in ``{-1, 0, +1}``
        or ``NaN`` / ``None`` to release a column from any constraint.
        Applied as a strict per-column override after data-derived signs.
    auto_sign_threshold_t : float, optional
        When set (>0), data-derived signs are enforced only for columns whose
        univariate t-statistic satisfies ``|t_j| >= auto_sign_threshold_t``.
        Columns failing the threshold are pinned to ``0`` (β forced to zero
        by the solver), excluding them from the regression entirely.

        Default 0.75 acts as a noise floor — well below conventional
        significance levels (|t| = 0.75 corresponds to two-sided p ≈ 0.45)
        but high enough to filter columns whose univariate slope sign is
        dominated by sampling noise. Pass ``None`` to disable the gate
        entirely and reproduce v0.3.6 behaviour (always enforce the
        slope-sign for every column).

        Typical alternative values for financial panels: 0.5 (looser —
        only the most negligible signals removed) to 1.0 (stricter).
    return_slopes : bool, default False
        If True, also return raw pooled slopes (useful for diagnostics or
        adaptive-weighting downstream).

    Returns
    -------
    signs : pd.DataFrame (N × M) when inputs are pandas, else ndarray (M,)
        Per-regressor signs, broadcast across responses when shape is (N, M).
        Directly compatible with ``LassoModel(factors_beta_loading_signs=…)``.
    slopes : pd.DataFrame or ndarray, same shape, optional
        Pooled univariate slopes (cluster-mode: cluster slope broadcast).

    Notes
    -----
    For multi-response ``y`` (``N > 1``), all response rows of the output
    DataFrame are identical — this is the pooled estimator (Σ_k y[:, k]).
    To produce per-response signs, call once per response column.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from factorlasso import LassoModel, derive_sign_constraints
    >>> rng = np.random.default_rng(0)
    >>> T, M, N = 100, 4, 3
    >>> X = pd.DataFrame(rng.standard_normal((T, M)),
    ...                  columns=['f0', 'f1', 'f2', 'f3'])
    >>> Y = pd.DataFrame(X.values @ np.array([[1, -1, 0, .5]]*N).T
    ...                    + 0.1 * rng.standard_normal((T, N)),
    ...                  columns=['y0', 'y1', 'y2'])
    >>> signs = derive_sign_constraints(X, Y, master_constraints={'f3': 0})
    >>> signs.shape
    (3, 4)
    >>> # signs is now droppable into LassoModel
    """
    # ------------------------------------------------------------------ #
    # 1. coerce inputs                                                   #
    # ------------------------------------------------------------------ #
    x_is_df = isinstance(x, pd.DataFrame)
    y_is_pandas = isinstance(y, (pd.DataFrame, pd.Series))

    col_names: Optional[list] = list(x.columns) if x_is_df else None
    x_arr = x.values if x_is_df else np.asarray(x, dtype=float)
    if x_arr.ndim != 2:
        raise ValueError(f"x must be 2-D, got shape {x_arr.shape}")

    if isinstance(y, pd.Series):
        y_arr = y.values
        y_row_names = None  # Series-y produces (M,) only on ndarray-y path
    elif isinstance(y, pd.DataFrame):
        y_arr = y.values
        y_row_names = list(y.columns)
    else:
        y_arr = np.asarray(y, dtype=float)
        y_row_names = None

    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)

    T, M = x_arr.shape
    if y_arr.shape[0] != T:
        raise ValueError(
            f"x has {T} rows but y has {y_arr.shape[0]} rows"
        )
    q = y_arr.shape[1]

    # ------------------------------------------------------------------ #
    # 2. delegate to pure-numpy core                                     #
    # ------------------------------------------------------------------ #
    clusters_np = np.asarray(clusters) if clusters is not None else None
    sign_vec, slopes = _compute_sign_vector(
        x_arr=x_arr,
        y_arr=y_arr,
        clusters=clusters_np,
        master_constraints=master_constraints,
        col_names=col_names,
        auto_sign_threshold_t=auto_sign_threshold_t,
        ewma_span=ewma_span, variance_estimator=variance_estimator,
    )

    # ------------------------------------------------------------------ #
    # 3. shape output to caller's preference                             #
    # ------------------------------------------------------------------ #
    if x_is_df and y_is_pandas:
        # Broadcast (M,) → (N, M) DataFrame matching LassoModel convention
        n_rows = q
        row_index = y_row_names if y_row_names is not None else pd.RangeIndex(n_rows)
        signs_df = pd.DataFrame(
            np.tile(sign_vec, (n_rows, 1)),
            index=row_index,
            columns=col_names,
        )
        if return_slopes:
            slopes_df = pd.DataFrame(
                np.tile(slopes, (n_rows, 1)),
                index=row_index,
                columns=col_names,
            )
            return signs_df, slopes_df
        return signs_df

    if return_slopes:
        return sign_vec, slopes
    return sign_vec


def validate_cluster_signs(
    x: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.DataFrame, pd.Series, np.ndarray],
    clusters: Union[np.ndarray, pd.Series, list],
    warn: bool = True,
    ewma_span: Optional[float] = None,
) -> np.ndarray:
    """
    Detect cluster misspecification by comparing column- vs cluster-level signs.

    For each regressor, computes its own univariate slope sign (column-level)
    and the sign that its cluster's aggregate would impose (cluster-level).
    When these disagree, the regressor is a candidate for being in the wrong
    cluster — its marginal correlation with y has the opposite direction to
    its cluster's aggregate.

    Use this before calling :func:`derive_sign_constraints` in cluster mode
    to surface any HCGL groupings that mix economically-different regressors.

    Parameters
    ----------
    x, y, clusters : as in :func:`derive_sign_constraints`.
    warn : bool, default True
        Emit a ``UserWarning`` listing disagreeing regressors. Disable when
        you want to handle the result programmatically without noise.

    Returns
    -------
    disagreements : np.ndarray of int
        Indices of regressors whose column-level and cluster-level signs
        disagree. Empty array if the clustering is internally consistent.
    """
    _, slopes_col = derive_sign_constraints(x, y, return_slopes=True, ewma_span=ewma_span)
    _, slopes_clu = derive_sign_constraints(
        x, y, clusters=clusters, return_slopes=True, ewma_span=ewma_span
    )

    col_arr = slopes_col.values[0] if isinstance(slopes_col, pd.DataFrame) else slopes_col
    clu_arr = slopes_clu.values[0] if isinstance(slopes_clu, pd.DataFrame) else slopes_clu

    col_signs = np.sign(col_arr).astype(int)
    clu_signs = np.sign(clu_arr).astype(int)
    # Disagreements: both signs nonzero and different
    disagreements = np.where(
        (col_signs != clu_signs) & (col_signs != 0) & (clu_signs != 0)
    )[0]

    if warn and len(disagreements) > 0:
        col_names = list(x.columns) if isinstance(x, pd.DataFrame) else None
        if col_names is not None:
            names = [col_names[i] for i in disagreements]
            detail = f"regressors {names} (indices {disagreements.tolist()})"
        else:
            detail = f"indices {disagreements.tolist()}"
        warnings.warn(
            f"{len(disagreements)} regressor(s) have column-level univariate "
            f"sign disagreeing with their cluster's aggregate sign — possible "
            f"cluster misspecification at {detail}.",
            UserWarning,
            stacklevel=2,
        )
    return disagreements


def _adaptive_penalty_weights(
    slopes: np.ndarray,
    signs: np.ndarray,
    gamma: float = 1.0,
    floor: float = 1e-3,
) -> np.ndarray:
    """
    Derive adaptive L1 penalty weights from univariate slope magnitudes,
    following the Zou (2006) adaptive-LASSO construction. The weight is a
    function of the univariate slope *magnitude* and is therefore distinct
    from the univariate-guided *sign* constraint of Richland et al. (2025)
    eq. (3.3) — that constraint fixes ``sign(β)`` and is applied separately
    via the sign matrix; this function supplies only the magnitude-aware
    penalty multiplier.

    For each cell (k, j), the adaptive penalty weight is

        W_kj = 1 / max(|β̂_uni_kj|, floor)^gamma

    where ``β̂_uni_kj`` is the pooled univariate slope and ``floor`` is a
    stabiliser preventing weight explosion on near-zero slopes. Cells where
    the sign-gate has already pinned ``s_kj = 0`` receive weight ``1.0`` as
    a placeholder (the hard sign constraint forces ``β_kj = 0`` so the
    penalty term contributes zero regardless).

    Parameters
    ----------
    slopes : ndarray (N, M)
        Univariate slope matrix β̂_uni from
        ``_compute_sign_matrix_per_response(..., return_slopes=True)`` or
        from the per-cluster path in ``LassoModel.fit``.
    signs : ndarray (N, M) of {-1, 0, +1}
        Threshold-gated sign matrix from the same source. Cells with
        ``signs[k, j] == 0`` are pinned to zero by the solver and receive
        a placeholder weight of 1.0.
    gamma : float, default 1.0
        Zou (2006) exponent. ``gamma = 1`` is the standard adaptive-Lasso
        choice; larger values amplify the magnitude-aware reweighting.
    floor : float, default 1e-3
        Stabiliser on the absolute slope. The slope magnitude is clipped
        at this floor before inversion to prevent weight explosion when a
        cell's univariate evidence is borderline-significant.

    Returns
    -------
    weights : ndarray (N, M) of float, all values in [floor**(-gamma), 1.0]
              after subsequent normalisation; weights of zero-pinned cells
              are set to 1.0 placeholder.
    """
    abs_slopes = np.abs(slopes)
    clipped = np.maximum(abs_slopes, floor)
    weights = 1.0 / (clipped ** gamma)
    # Cells pinned to zero by the sign-gate get a placeholder weight of 1.0;
    # the hard equality β_kj = 0 makes the penalty term contribution zero
    # regardless.
    weights = np.where(signs == 0.0, 1.0, weights)
    return weights


def _aggregate_to_row_weights(
    cell_weights: np.ndarray,
    signs: np.ndarray,
) -> np.ndarray:
    """
    Aggregate the ``(N, M)`` cell-level adaptive weights into a per-asset
    row-weight vector of length ``N`` suitable for adaptive Group LASSO
    reweighting.

    For each asset row ``k``, the aggregation is the root-mean-square of
    the cell weights over the *non-gated* factors (cells where
    ``signs[k, j] != 0``)::

        W_k = sqrt( mean_{j: s_kj != 0} W_kj^2 )

    Rationale:

    - Root-mean-square is the L2-natural aggregation to pair with the
      L2 norm in the group-LASSO penalty term ``W_k * ||β_k - β⁰_k||_2``.
      Mean would be too soft; max would be overly aggressive.
    - For an asset with ``|β̂_uni_kj| = 1`` across all factors, the
      aggregation returns ``W_k = 1`` exactly, preserving the existing
      per-cluster scaling ``√(|g|/G)`` of ``solve_group_lasso_cvx_problem``
      without any multiplicative drift.
    - Gate-pinned cells (``s_kj = 0``) are *excluded* from the
      aggregation. Their placeholder weight of 1.0 is irrelevant — the
      hard sign constraint forces ``β_kj = 0`` independently — and
      including them would artificially pull ``W_k`` toward unity.

    If an asset has *every* cell pinned (degenerate row), the
    aggregation falls back to ``W_k = 1`` to avoid a divide-by-zero.

    Parameters
    ----------
    cell_weights : ndarray (N, M)
        Per-cell adaptive weights from ``_adaptive_penalty_weights``.
    signs : ndarray (N, M) of {-1, 0, +1}
        Gated sign matrix; cells with ``signs[k, j] == 0`` are excluded
        from the row aggregation.

    Returns
    -------
    row_weights : ndarray (N,) of float, with W_k = 1.0 for any
                  fully-pinned row.
    """
    active = (signs != 0.0).astype(float)              # (N, M)
    n_active = active.sum(axis=1)                      # (N,)
    sq_sum = (active * cell_weights * cell_weights).sum(axis=1)  # (N,)
    # Mean of squares, fallback to 1.0 for fully-pinned rows
    with np.errstate(divide="ignore", invalid="ignore"):
        ms = np.where(n_active > 0, sq_sum / np.where(n_active > 0, n_active, 1.0), 1.0)
    return np.sqrt(ms)


def _aggregate_to_block_weights(
    cell_weights: np.ndarray,
    signs: np.ndarray,
    group_loadings: np.ndarray,
) -> np.ndarray:
    """
    Aggregate the ``(N, M)`` cell-level adaptive weights into a per-block
    weight matrix of shape ``(G, M)`` for the cluster x factor group
    penalty of ``FACTOR_CLUSTER_GROUP_LASSO``.

    For each cluster ``g`` and factor ``j``, the aggregation is the
    root-mean-square of the cell weights over the *non-gated* members of
    that cluster (cells where ``signs[k, j] != 0`` and ``k`` is in
    cluster ``g``)::

        W_gj = sqrt( mean_{k in g: s_kj != 0} W_kj^2 )

    Rationale mirrors :func:`_aggregate_to_row_weights`, one dimension
    over: root-mean-square is the L2-natural aggregation to pair with the
    per-block L2 norm ``W_gj * ||β_{g,j} - β⁰_{g,j}||_2``. For a block
    whose members all have ``|β̂_uni_kj| = 1`` the aggregation returns
    ``W_gj = 1`` exactly, preserving the per-cluster ``√(|g|/G)`` scaling
    without multiplicative drift. Gate-pinned cells (``s_kj = 0``) are
    excluded; a fully-pinned block falls back to ``W_gj = 1``.

    Parameters
    ----------
    cell_weights : ndarray (N, M)
        Per-cell adaptive weights from ``_adaptive_penalty_weights``.
    signs : ndarray (N, M) of {-1, 0, +1}
        Gated sign matrix; cells with ``signs[k, j] == 0`` are excluded.
    group_loadings : ndarray (N, G)
        Binary cluster-membership matrix; column ``g`` is 1 for the
        members of cluster ``g``.

    Returns
    -------
    col_weights : ndarray (G, M) of float, with W_gj = 1.0 for any
                  fully-pinned block.
    """
    active = (signs != 0.0).astype(float)              # (N, M)
    wsq = active * cell_weights * cell_weights         # (N, M)
    membership = (np.isclose(group_loadings, 1.0)).astype(float)  # (N, G)
    n_groups = membership.shape[1]
    n_factors = cell_weights.shape[1]
    block_w = np.ones((n_groups, n_factors), dtype=float)
    for g in range(n_groups):
        m = membership[:, g] > 0.0                     # members of cluster g
        if not m.any():
            continue
        n_active = active[m, :].sum(axis=0)            # (M,)
        sq_sum = wsq[m, :].sum(axis=0)                 # (M,)
        with np.errstate(divide="ignore", invalid="ignore"):
            ms = np.where(
                n_active > 0,
                sq_sum / np.where(n_active > 0, n_active, 1.0),
                1.0,
            )
        block_w[g, :] = np.sqrt(ms)
    return block_w
