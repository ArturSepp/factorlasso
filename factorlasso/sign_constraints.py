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
EWMA-weighting, residualization etc. is the caller's responsibility.

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
"""
from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd


def _compute_sign_vector(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    clusters: Optional[np.ndarray] = None,
    master_constraints: Optional[dict] = None,
    col_names: Optional[list] = None,
    auto_sign_threshold_t: Optional[float] = None,
) -> tuple:
    """
    Pure-numpy core of sign derivation. Shared by the user-facing
    :func:`derive_sign_constraints` wrapper and by ``LassoModel.fit`` when
    ``auto_sign_constraints=True``.

    Parameters
    ----------
    x_arr : ndarray (T, M)
    y_arr : ndarray (T, N)   — already reshaped to 2-D if originally 1-D
    clusters : ndarray (M,), optional
    master_constraints : dict {name_or_index: sign}, optional
        sign ∈ {-1, 0, +1} | NaN | None
    col_names : list of length M, optional
        Required only if ``master_constraints`` uses string keys.
    auto_sign_threshold_t : float, optional
        When set (>0), the data-derived sign for a column is enforced only
        if the univariate t-statistic of the pooled slope satisfies
        ``|t_j| >= auto_sign_threshold_t``.  Columns failing the threshold
        are pinned to ``0`` (β forced to zero) so the downstream LASSO
        solver excludes them from the regression entirely.

        Rationale: when the univariate marginal signal is weak (small ``|t|``)
        the sign of the slope is dominated by sampling noise.  Propagating a
        noise-driven sign into the multivariate constraint set lets the
        solver fit residual variance using offsetting loadings (Credit ↔
        Inflation, etc.) that L1 regularization may not be strong enough to
        discipline.  Hard-zeroing weak-evidence columns enforces parsimony
        directly and is robust to the choice of ``reg_lambda``.

        Recommended values for typical financial panels: 0.5 (very loose —
        only the most negligible signals removed) to 1.0 (moderate).
        ``None`` (default) preserves the previous behaviour of always
        enforcing the slope-sign for every column.

    Returns
    -------
    sign_vec : ndarray (M,) float in {-1.0, 0.0, +1.0, NaN}
    slopes   : ndarray (M,) float
    """
    T, M = x_arr.shape
    if y_arr.shape[0] != T:
        raise ValueError(
            f"x has {T} rows but y has {y_arr.shape[0]} rows"
        )

    # NaN handling. We zero-fill (x, y) for the arithmetic but FIRST record
    # the per-cell validity mask so every reduction below ranges over genuine
    # observations only. Zero-filling alone is *not* equivalent to a valid-row
    # fit for a pooled estimator with heterogeneous inception dates: a missing
    # y_{tk} contributes 0 to the cross-asset sum (harmless), but it must not
    # be counted in the slope denominator Σ x_{tj}², the SSR, or the degrees
    # of freedom. Treating zero-filled cells as real observations deflates the
    # residual variance and inflates |t| (anticonservative). We therefore
    # carry the valid mask explicitly and weight every reduction by it.
    valid_y = ~np.isnan(y_arr)                 # (T, q) genuine response cells
    valid_x = ~np.isnan(x_arr)                 # (T, M) genuine factor cells
    if np.isnan(x_arr).any():
        x_arr = np.nan_to_num(x_arr, nan=0.0)
    if np.isnan(y_arr).any():
        y_arr = np.nan_to_num(y_arr, nan=0.0)

    y_sum = y_arr.sum(axis=1)
    # Per-response-cell count of valid (row) observations entering each
    # (factor, response) inner product. A row contributes to factor j and
    # response k only when both x_{tj} and y_{tk} are present.
    slopes = np.zeros(M, dtype=float)

    if clusters is None:
        # Pooled denominator Σ_k Σ_t v_{tk} x_{tj}² uses the per-response
        # valid mask, NOT q · Σ_t x_{tj}². When y is fully observed these
        # coincide; under leading-NaN they diverge.
        x2 = x_arr ** 2                         # (T, M)
        # (M,): Σ_j over rows where x valid, summed across responses where y valid
        numerator = x_arr.T @ y_sum             # (M,) — missing y already 0
        # denominator[j] = Σ_k Σ_t (x_valid_{tj} & y_valid_{tk}) x_{tj}²
        denominator = (x2 * valid_x).T @ valid_y.sum(axis=1)  # (M,)
        safe = denominator > 0
        slopes[safe] = numerator[safe] / denominator[safe]
    else:
        clusters_arr = np.asarray(clusters)
        if len(clusters_arr) != M:
            raise ValueError(
                f"clusters length {len(clusters_arr)} != M={M}"
            )
        for c in np.unique(clusters_arr):
            idx = np.where(clusters_arr == c)[0]
            x_agg = x_arr[:, idx].mean(axis=1)          # (T,)
            x_agg_valid = valid_x[:, idx].all(axis=1)   # (T,) agg defined
            # The aggregated regressor is defined only on rows where EVERY
            # cluster member is observed; on other rows the zero-filled mean
            # is a biased (shrunken) value. Numerator and denominator must
            # range over the SAME valid rows — masking only the denominator
            # (the pre-fix behaviour) inflates the slope whenever cluster
            # members carry heterogeneous NaN patterns.
            x_agg_m = x_agg * x_agg_valid                # (T,) masked agg
            xa2 = (x_agg ** 2) * x_agg_valid             # (T,)
            denom = float(xa2 @ valid_y.sum(axis=1))     # Σ_k Σ_t v x_agg²
            if denom > 0:
                slopes[idx] = float(x_agg_m @ y_sum) / denom

    sign_vec = np.sign(slopes).astype(float)

    # Optional t-stat gating: pin weak-evidence columns to sign=0 (β forced
    # to zero by the solver).
    if auto_sign_threshold_t is not None and auto_sign_threshold_t > 0.0:
        # No-intercept pooled fit on the demeaned inputs. The SSR has a
        # closed form that avoids materialising residuals; we evaluate every
        # reduction over valid (row, response) cells so the variance and dof
        # reflect the true sample size under heterogeneous inception dates:
        #   SSR_j = Σ_{k,t: x_j and y_k valid} (y_{tk} − β_j x_{tj})²
        #         = ‖Y‖²_{F,(j)} − β_j² · D_j,   (‖Y‖²_{F,(j)} over x_j-valid rows)
        #   D_j = Σ_k Σ_t v_{tk} x_{tj}²  (the same valid denominator above),
        # using the slope identity x_j' y_sum = β_j · D_j. The degrees of
        # freedom charge one parameter per response column actually present:
        #   df_j = n_valid_j − q_eff,  n_valid_j = Σ_k Σ_t (x_valid & y_valid),
        # with q_eff the number of responses contributing ≥1 valid row.
        # The per-factor SS must be masked by valid_x exactly as D_j and df_j are;
        # a single global Σ y² over-counts rows where factor j is missing. The two
        # coincide when every factor is fully observed (the fast path below).
        y2_rowsum = (y_arr * y_arr).sum(axis=1)          # (T,) Σ_k v_{tk} y_{tk}²
        t_stats = np.zeros(M, dtype=float)
        n_valid_per_resp = valid_y.sum(axis=0)           # (q,) rows valid per k
        q_eff = int((n_valid_per_resp > 0).sum())
        if clusters is None:
            x2 = x_arr ** 2
            vy_rowcount = valid_y.sum(axis=1)            # (T,) #valid responses per row
            D = (x2 * valid_x).T @ vy_rowcount           # (M,) valid denominator
            # n_valid_j = Σ_t valid_x_{tj} · (#valid responses in row t)
            n_valid = (valid_x.astype(float).T @ vy_rowcount)  # (M,) valid (t,k) cells
            df = np.maximum(n_valid - q_eff, 1.0)
            if valid_x.all():
                # complete factors: bit-identical to prior global
                Y_ss = np.full(M, float((y_arr * y_arr).sum()))
            else:
                Y_ss = valid_x.astype(float).T @ y2_rowsum  # (M,) Σ_t v_x·(Σ_k v_y y²)
            ssr = Y_ss - slopes * slopes * D             # (M,)
            sigma2 = np.maximum(ssr, 0.0) / df
            with np.errstate(divide="ignore", invalid="ignore"):
                se = np.where(
                    (sigma2 > 0) & (D > 0),
                    np.sqrt(sigma2 / np.where(D > 0, D, 1.0)),
                    np.inf,
                )
                t_stats = np.where(se > 0, slopes / se, 0.0)
        else:
            clusters_arr = np.asarray(clusters)
            for c in np.unique(clusters_arr):
                idx = np.where(clusters_arr == c)[0]
                x_agg = x_arr[:, idx].mean(axis=1)
                x_agg_valid = valid_x[:, idx].all(axis=1)
                xa2 = (x_agg ** 2) * x_agg_valid
                D_c = float(xa2 @ valid_y.sum(axis=1))
                if D_c <= 0.0:
                    continue
                slope_c = float(slopes[idx[0]])
                # valid (t,k) cells for this aggregated factor
                n_valid_c = float((x_agg_valid[:, None] & valid_y).sum())
                df_c = max(n_valid_c - q_eff, 1.0)
                Y_ss_c = float(x_agg_valid.astype(float) @ y2_rowsum)  # x_agg-valid rows
                ssr_c = Y_ss_c - slope_c * slope_c * D_c
                sigma2 = max(ssr_c, 0.0) / df_c
                se_c = np.sqrt(sigma2 / D_c) if (sigma2 > 0 and D_c > 0) else np.inf
                t_c = slope_c / se_c if se_c > 0 else 0.0
                t_stats[idx] = t_c

        # Pin columns failing the threshold to sign=0 (β forced to zero).
        weak = np.abs(t_stats) < auto_sign_threshold_t
        sign_vec[weak] = 0.0

    if master_constraints:
        for key, s in master_constraints.items():
            if s is None or (isinstance(s, float) and np.isnan(s)):
                s_val = np.nan
            elif s in (-1, 0, 1):
                s_val = float(s)
            else:
                raise ValueError(
                    f"master_constraints[{key!r}]={s}; "
                    f"must be in {{-1, 0, +1}}, NaN, or None"
                )
            if isinstance(key, str):
                if col_names is None:
                    raise ValueError(
                        f"name {key!r} in master_constraints but no column "
                        f"names were available"
                    )
                if key not in col_names:
                    raise KeyError(
                        f"master_constraints key {key!r} not found in col_names"
                    )
                idx = col_names.index(key)
            else:
                idx = int(key)
                if not 0 <= idx < M:
                    raise IndexError(
                        f"master_constraints index {idx} out of range for M={M}"
                    )
            sign_vec[idx] = s_val

    return sign_vec, slopes


def _compute_sign_matrix_per_response(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    auto_sign_threshold_t: Optional[float] = None,
    return_slopes: bool = False,
) -> Union[np.ndarray, tuple]:
    """
    Vectorised bulk equivalent of N calls to ``_compute_sign_vector`` with
    ``clusters=None`` and ``y_arr[:, k:k+1]`` — i.e. per-y-column independent
    univariate sign derivation. Used by ``LassoModel.fit`` in LASSO mode to
    avoid an N-deep Python loop.

    Computes the full ``(N, M)`` slope and t-stat matrix in a single
    matrix-product + closed-form SSR (q = 1 per row). Returns the
    threshold-gated ``(N, M)`` sign matrix, and optionally the underlying
    univariate slope matrix for downstream adaptive-weight derivation.

    Parameters
    ----------
    x_arr : ndarray (T, M)
    y_arr : ndarray (T, N)
    auto_sign_threshold_t : float, optional
        Threshold-gate parameter; see ``_compute_sign_vector``.
    return_slopes : bool, default False
        If True, returns ``(signs, slopes)``; otherwise returns ``signs`` only.
        Slopes are the raw univariate estimates β̂_kj before thresholding —
        i.e. the magnitudes consumed by the Zou (2006) adaptive-weight
        formula ``λ · |β_kj| / |β̂_kj|`` when paired with
        ``LassoModel.auto_sign_adaptive_weights=True``.

    Returns
    -------
    signs : ndarray (N, M) of float in {-1, 0, +1}
    slopes : ndarray (N, M), only when ``return_slopes=True``
    """
    # NaN handling (same contract as _compute_sign_vector): record validity
    # masks before zero-filling so the slope denominator, SSR, and dof range
    # over genuine observations only — never over zero-filled rows.
    valid_x = ~np.isnan(x_arr)                     # (T, M)
    valid_y = ~np.isnan(y_arr)                     # (T, N)
    if np.isnan(x_arr).any():
        x_arr = np.nan_to_num(x_arr, nan=0.0)
    if np.isnan(y_arr).any():
        y_arr = np.nan_to_num(y_arr, nan=0.0)

    x2 = x_arr * x_arr                             # (T, M)
    y2 = y_arr * y_arr                             # (T, N) zero-filled y adds 0
    xy = x_arr.T @ y_arr                           # (M, N)

    # Per-(response k, factor j) valid-row count and denominator. A row t
    # enters cell (k, j) only when both x_{tj} and y_{tk} are present.
    # n_valid[k, j] = Σ_t valid_x_{tj} & valid_y_{tk}
    n_valid = (valid_y.astype(float).T @ valid_x.astype(float))   # (N, M)
    # denom[k, j] = Σ_t (valid_x_{tj} & valid_y_{tk}) x_{tj}²
    denom_kj = (valid_y.astype(float).T @ (x2 * valid_x))         # (N, M)

    # β[k, j] = (x_j' y_k) / denom[k, j]  (valid-row denominator)
    safe = denom_kj > 0
    slopes = np.zeros((y_arr.shape[1], x_arr.shape[1]), dtype=float)  # (N, M)
    slopes[safe] = (xy.T)[safe] / denom_kj[safe]
    signs = np.sign(slopes).astype(float)          # (N, M)

    if auto_sign_threshold_t is not None and auto_sign_threshold_t > 0.0:
        # df = n_valid − 1 (one slope per (k, j) univariate fit), evaluated
        # on the valid-row count rather than nominal T.
        df = np.maximum(n_valid - 1.0, 1.0)
        # Y-SS per (response k, factor j) over rows where x_j is observed, matching
        # the valid_x masking on denom_kj and n_valid. A per-response Σ_t y²
        # (independent of j) over-counts when factor j carries NaN rows; the two
        # are equal only when every factor is fully observed (the fast path).
        if valid_x.all():
            yss = (y2.sum(axis=0))[:, None]                 # (N,1) complete-factor fast path
        else:
            yss = y2.T @ valid_x.astype(float)              # (N,M) Σ_t v_x v_y y²
        ssr = yss - slopes * slopes * denom_kj              # (N, M)
        sigma2 = np.maximum(ssr, 0.0) / df
        with np.errstate(divide="ignore", invalid="ignore"):
            se = np.where(
                (sigma2 > 0) & (denom_kj > 0),
                np.sqrt(sigma2 / np.where(denom_kj > 0, denom_kj, 1.0)),
                np.inf,
            )
            t_stats = np.where(se > 0, slopes / se, 0.0)
        signs[np.abs(t_stats) < auto_sign_threshold_t] = 0.0

    if return_slopes:
        return signs, slopes
    return signs


def derive_sign_constraints(
    x: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.DataFrame, pd.Series, np.ndarray],
    clusters: Optional[Union[np.ndarray, pd.Series, list]] = None,
    master_constraints: Optional[dict] = None,
    auto_sign_threshold_t: Optional[float] = 0.75,
    return_slopes: bool = False,
) -> Union[pd.DataFrame, np.ndarray, tuple]:
    """
    Compute pooled univariate sign constraints for downstream LASSO estimation.

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
    _, slopes_col = derive_sign_constraints(x, y, return_slopes=True)
    _, slopes_clu = derive_sign_constraints(
        x, y, clusters=clusters, return_slopes=True
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
    penalty of ``CLUSTER_FACTOR_GROUP_LASSO``.

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
