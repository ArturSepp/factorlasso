"""
Metric functions for the factorlasso simulation study.

Every function is a pure mapping from (β_true, β_hat, …) → scalar. No
state, no I/O, no logging. Returned floats are publication-ready.

Mandatory metrics (paper §5 main table):

- :func:`support_recovery_f1`     — selection accuracy on the non-zero support
- :func:`sign_agreement_rate`     — sign accuracy conditional on β_true ≠ 0
- :func:`beta_mse_normalised`     — ‖β̂ − β_true‖²_F / ‖β_true‖²_F

Secondary metrics (paper §5 supporting analysis):

- :func:`cluster_sign_coherence`  — within-cluster sign agreement (the
                                    quantity the cluster-pooled mechanism
                                    specifically targets)
- :func:`oos_r2`                  — out-of-sample R² on a held-out window
- :func:`factor_rp_rmse`          — RMSE of implied factor risk premia β·λ
- :func:`cluster_recovery_ari`    — Adjusted Rand Index of HCGL clusters
                                    vs. true clusters

All array inputs are numpy ndarrays with the conventional layout:

- β ∈ R^{N × M}: assets along axis 0, factors along axis 1
- y ∈ R^{T × N}
- λ ∈ R^M

A small ``tol`` parameter (default 1e-6) distinguishes "structurally
zero" from "numerically small" — relevant because CVXPY solvers return
β ~ 1e-10 rather than exact zero even when the optimal solution is zero.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

_DEFAULT_TOL = 1e-6


# ── Mandatory metrics ─────────────────────────────────────────────────


def support_recovery_f1(
    beta_true: np.ndarray, beta_hat: np.ndarray, tol: float = _DEFAULT_TOL,
) -> float:
    """
    F1 score on the non-zero support recovery problem.

    Treats ``|β| > tol`` as "non-zero" in both true and fitted matrices.
    Returns ``2 · precision · recall / (precision + recall)``.

    Returns 0.0 if either precision or recall is zero. Returns NaN if
    β_true has zero non-zero entries (the support-recovery problem is
    undefined).
    """
    true_support = np.abs(beta_true) > tol
    hat_support = np.abs(beta_hat) > tol

    if true_support.sum() == 0:
        return float("nan")

    tp = int((true_support & hat_support).sum())
    fp = int((~true_support & hat_support).sum())
    fn = int((true_support & ~hat_support).sum())

    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return float(2 * precision * recall / (precision + recall))


def sign_agreement_rate(
    beta_true: np.ndarray, beta_hat: np.ndarray, tol: float = _DEFAULT_TOL,
) -> float:
    """
    Fraction of (asset, factor) cells where ``sign(β_hat) == sign(β_true)``,
    conditional on ``|β_true| > tol``.

    Cells where β_true is structurally zero are excluded from both
    numerator and denominator — they are addressed by support recovery,
    not by sign agreement.

    Returns NaN if no β_true cells exceed ``tol`` (denominator undefined).
    """
    true_support = np.abs(beta_true) > tol
    if true_support.sum() == 0:
        return float("nan")

    # Threshold beta_hat symmetrically with beta_true before taking signs.
    # Interior-point solvers return ±O(1e-10) on cells they have effectively
    # zeroed, and the sign of that numerical noise is platform-dependent
    # (BLAS build, solver version). Without the threshold the metric drifts
    # by O(1e-3) between otherwise-identical runs on different platforms;
    # with it, a near-zero estimate on a true-support cell deterministically
    # counts as a miss (sign 0 against ±1).
    beta_hat_z = np.where(np.abs(beta_hat) > tol, beta_hat, 0.0)
    agree = (np.sign(beta_true) == np.sign(beta_hat_z)) & true_support
    return float(agree.sum() / true_support.sum())


def beta_mse_normalised(beta_true: np.ndarray, beta_hat: np.ndarray) -> float:
    """
    ‖β̂ − β_true‖²_F / ‖β_true‖²_F.

    Normalising by ‖β_true‖² makes the metric scale-invariant in the true
    coefficient magnitudes — comparable across regimes where ``β`` has
    different overall scale (e.g. dense vs sparse).

    Returns NaN if ‖β_true‖² is below numerical floor (degenerate).
    """
    diff = beta_hat - beta_true
    denom = float((beta_true ** 2).sum())
    if denom < 1e-12:
        return float("nan")
    return float((diff ** 2).sum() / denom)


# ── Secondary metrics ────────────────────────────────────────────────


def cluster_sign_coherence(
    beta_hat: np.ndarray,
    clusters_true: np.ndarray,
    tol: float = _DEFAULT_TOL,
) -> float:
    """
    Within-cluster sign coherence of the fitted β.

    For each true cluster k and each factor column j:

    - Restrict to cluster k's members
    - Drop cells where ``|β_hat_kj| ≤ tol`` (treated as inactive)
    - Mark the (cluster, factor) pair as coherent if all remaining
      non-zero entries share the same sign

    The metric is the mean coherence over (cluster, factor) pairs that
    have at least one non-zero entry. A pair where every member is zero
    on factor j is excluded from the average (the question is undefined).

    This is the quantity the cluster-pooled sign mechanism specifically
    targets: a sign-derived constraint should produce coherent within-
    cluster signs, while an unconstrained fit may not.
    """
    coherences = []
    for k in np.unique(clusters_true):
        member_mask = clusters_true == k
        cluster_betas = beta_hat[member_mask]  # (n_k, M)
        signs = np.sign(np.where(np.abs(cluster_betas) > tol, cluster_betas, 0.0))
        # For each factor column: count distinct non-zero signs
        for j in range(signs.shape[1]):
            col_signs = signs[:, j]
            non_zero = col_signs[col_signs != 0]
            if len(non_zero) == 0:
                continue  # All inactive — skip
            coherent = len(np.unique(non_zero)) == 1
            coherences.append(1.0 if coherent else 0.0)

    if not coherences:
        return float("nan")
    return float(np.mean(coherences))


def oos_r2(y_test: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean out-of-sample R² across asset columns.

    For each asset k::

        R²_k = 1 − Σ_t (y_test[t,k] − y_pred[t,k])² / Σ_t (y_test[t,k] − ȳ_test_k)²

    where ȳ_test_k is the held-out sample mean. Returns the average
    across assets, ignoring assets with degenerate test variance.

    Negative R² is allowed and reported as-is (a fit can be worse than
    the test-window mean).
    """
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    if y_test.shape != y_pred.shape:
        raise ValueError(
            f"y_test and y_pred must have identical shape, "
            f"got {y_test.shape} vs {y_pred.shape}"
        )

    ss_res = ((y_test - y_pred) ** 2).sum(axis=0)
    means = y_test.mean(axis=0, keepdims=True)
    ss_tot = ((y_test - means) ** 2).sum(axis=0)

    valid = ss_tot > 1e-12
    if not valid.any():
        return float("nan")

    r2 = np.full(y_test.shape[1], np.nan)
    r2[valid] = 1.0 - ss_res[valid] / ss_tot[valid]
    return float(np.nanmean(r2))


def factor_rp_rmse(
    beta_true: np.ndarray, beta_hat: np.ndarray, lambda_true: np.ndarray,
) -> float:
    """
    RMSE of implied per-asset factor risk premium ``β · λ``.

    For each asset k::

        RP_true_k = Σ_j β_true[k, j] · λ[j]
        RP_hat_k  = Σ_j β_hat[k, j]  · λ[j]
        RMSE      = √( mean_k (RP_hat_k − RP_true_k)² )

    This is the downstream quantity that matters for the MATF-CMA
    application: a factor model's value to CMA construction is the
    accuracy of its implied excess returns, not the accuracy of its
    individual β coefficients.
    """
    rp_true = beta_true @ lambda_true
    rp_hat = beta_hat @ lambda_true
    return float(np.sqrt(((rp_hat - rp_true) ** 2).mean()))


def cluster_recovery_ari(
    clusters_true: np.ndarray, clusters_hat: np.ndarray,
) -> float:
    """
    Adjusted Rand Index between true and HCGL-discovered cluster
    assignments. Hubert & Arabie (1985).

    ARI = 1 for identical clusterings (up to permutation),
    ARI ≈ 0 for random clusterings,
    ARI < 0 possible but unusual.

    Implemented directly to avoid a sklearn dependency.
    """
    clusters_true = np.asarray(clusters_true)
    clusters_hat = np.asarray(clusters_hat)
    if clusters_true.shape != clusters_hat.shape:
        raise ValueError(
            f"clusters_true and clusters_hat must have identical shape, "
            f"got {clusters_true.shape} vs {clusters_hat.shape}"
        )

    n = len(clusters_true)
    if n < 2:
        return float("nan")

    classes_true = np.unique(clusters_true)
    classes_hat = np.unique(clusters_hat)
    contingency = np.zeros((len(classes_true), len(classes_hat)), dtype=np.int64)
    for i, ct in enumerate(classes_true):
        for j, ch in enumerate(classes_hat):
            contingency[i, j] = int(((clusters_true == ct) & (clusters_hat == ch)).sum())

    def _comb2(x):
        # C(x, 2) without scipy
        return x * (x - 1) // 2

    sum_comb_c = sum(_comb2(int(n_ij)) for n_ij in contingency.flatten())
    sum_comb_k = sum(_comb2(int(n_i)) for n_i in contingency.sum(axis=1))
    sum_comb_m = sum(_comb2(int(n_j)) for n_j in contingency.sum(axis=0))
    total = _comb2(n)

    if total == 0:
        return float("nan")
    expected = sum_comb_k * sum_comb_m / total
    max_index = (sum_comb_k + sum_comb_m) / 2

    if max_index == expected:
        # Both clusterings collapse to one cluster — perfectly agreeing
        return 1.0
    return float((sum_comb_c - expected) / (max_index - expected))


# ── Economic attribution metrics (calibrated MATF universe, §5) ───────
#
# These replace abstract support/sign metrics with the quantities a CMA
# desk actually cares about: is a named factor's exposure attributed to the
# right factor, or absorbed into a collinear neighbour? They take explicit
# factor and asset index sets so the same functions serve any factor (the
# paper uses Credit over the credit-linked assets).


def factor_recovery(
    beta_hat: np.ndarray, factor_idx: int, asset_idx: np.ndarray,
) -> float:
    """
    Mean fitted loading ``β_hat[asset_idx, factor_idx]``.

    Compared against the true mean loading, this exposes shrink-to-zero: an
    estimator that zeros a collinear factor returns ~0 here even when the true
    mean loading is materially positive.
    """
    return float(beta_hat[np.asarray(asset_idx), factor_idx].mean())


def factor_abs_error(
    beta_true: np.ndarray, beta_hat: np.ndarray,
    factor_idx: int, asset_idx: np.ndarray,
) -> float:
    """Mean ``|β_hat − β_true|`` on one factor over the given assets."""
    ai = np.asarray(asset_idx)
    return float(np.abs(beta_hat[ai, factor_idx] - beta_true[ai, factor_idx]).mean())


def factor_leakage(
    beta_true: np.ndarray, beta_hat: np.ndarray,
    target_factor_idx: int, asset_idx: np.ndarray,
) -> float:
    """
    Mean signed error ``β_hat − β_true`` on ``target_factor_idx`` over the given
    assets — how much exposure has *leaked into* the target factor.

    With ``target_factor_idx = Equity`` over the credit-linked assets, a
    positive value is the credit exposure mis-attributed to the equity beta.
    """
    ai = np.asarray(asset_idx)
    return float((beta_hat[ai, target_factor_idx] - beta_true[ai, target_factor_idx]).mean())


def factor_risk_share_error(
    beta_true: np.ndarray, beta_hat: np.ndarray, factor_cov: np.ndarray,
    factor_idx: int, asset_idx: np.ndarray,
) -> float:
    """
    Mean absolute error in a factor's share of *systematic* variance, over the
    given assets.

    For asset i the marginal (Euler) contribution of factor j to systematic
    variance is ``c_ij = β_ij · (Σ_F β_i)_j`` and its share is
    ``c_ij / (β_i' Σ_F β_i)`` (shares over j sum to 1). The metric is

        mean_i | share_hat[i, factor_idx] − share_true[i, factor_idx] |.

    This is the risk-model analogue of :func:`factor_abs_error`: it asks not
    "is the beta right" but "does the covariance attribute the right share of
    risk to this factor". Assets with degenerate systematic variance are
    dropped.
    """
    ai = np.asarray(asset_idx)

    def _shares(beta):
        b = beta[ai]                              # (n, M)
        sb = b @ factor_cov                       # (n, M) = (Σ_F β_i)_j rows
        contrib = b * sb                          # c_ij
        total = contrib.sum(axis=1)               # β_i' Σ_F β_i
        valid = np.abs(total) > 1e-12
        out = np.full(len(ai), np.nan)
        out[valid] = contrib[valid, factor_idx] / total[valid]
        return out

    err = np.abs(_shares(beta_hat) - _shares(beta_true))
    if not np.isfinite(err).any():
        return float("nan")
    return float(np.nanmean(err))


def compute_attribution(
    beta_true: np.ndarray,
    beta_hat: np.ndarray,
    *,
    factor_cov: np.ndarray,
    target_factor_idx: int,
    leak_factor_idx: int,
    asset_idx: np.ndarray,
) -> dict:
    """
    Attribution metric bundle for one factor over one asset set. Returns a flat
    dict: ``recovery`` (mean β_hat on the target factor), ``abs_error``,
    ``leakage`` (signed error into ``leak_factor_idx``), and
    ``risk_share_error``.
    """
    return {
        "recovery": factor_recovery(beta_hat, target_factor_idx, asset_idx),
        "abs_error": factor_abs_error(beta_true, beta_hat, target_factor_idx, asset_idx),
        "leakage": factor_leakage(beta_true, beta_hat, leak_factor_idx, asset_idx),
        "risk_share_error": factor_risk_share_error(
            beta_true, beta_hat, factor_cov, target_factor_idx, asset_idx
        ),
    }


# ── Convenience: compute all metrics in one call ──────────────────────


def compute_all(
    beta_true: np.ndarray,
    beta_hat: np.ndarray,
    clusters_true: np.ndarray,
    factor_premia: np.ndarray,
    *,
    y_test: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    clusters_hat: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute the full publication metric set in one call. Returns a flat
    dict keyed by metric name, suitable for direct insertion into the
    long-form results DataFrame.

    Optional arguments unlock secondary metrics:

    - ``y_test`` and ``y_pred`` → adds ``oos_r2``
    - ``clusters_hat``          → adds ``cluster_ari``

    Metrics that cannot be computed (missing arguments, degenerate
    inputs) are reported as NaN rather than raising.
    """
    out = {
        "support_f1": support_recovery_f1(beta_true, beta_hat),
        "sign_rate": sign_agreement_rate(beta_true, beta_hat),
        "beta_mse_norm": beta_mse_normalised(beta_true, beta_hat),
        "cluster_coherence_hat": cluster_sign_coherence(beta_hat, clusters_true),
        "factor_rp_rmse": factor_rp_rmse(beta_true, beta_hat, factor_premia),
    }
    if y_test is not None and y_pred is not None:
        out["oos_r2"] = oos_r2(y_test, y_pred)
    else:
        out["oos_r2"] = float("nan")

    if clusters_hat is not None:
        out["cluster_ari"] = cluster_recovery_ari(clusters_true, clusters_hat)
    else:
        out["cluster_ari"] = float("nan")

    return out
