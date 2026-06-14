"""
Data-generating processes for the factorlasso simulation study.

The synthetic panel is

    Y = X β'  +  ε,        β ∈ R^{N×M}

where:

- X is (T × M) factor returns with controllable factor covariance;
- β is (N × M) loadings with controllable cluster structure, sparsity,
  and sign coherence;
- ε is (T × N) residuals with controllable structure;
- SNR is enforced by rescaling ε to hit a target ratio of explained to
  total variance.

The DGP exposes five regime axes that the JSS paper §5 ablates:

1. ``T``         — sample length (rows of X, Y)
2. ``N``         — asset count (columns of Y, rows of β)
3. ``K``         — number of true asset clusters
4. ``sign_mix``  — within-cluster sign coherence (clean / mixed /
                   idiosyncratic)
5. ``sparsity``  — sparsity of cluster-template β (sparse / moderate /
                   dense)
6. ``snr``       — signal-to-noise ratio (R² of true model)

Plus secondary axes (``factor_cov``, ``residual_cov``, ``rho_beta``).

All randomness is driven by a single integer seed via
``numpy.random.default_rng``. Two calls with identical configs return
identical data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Type-like literals, exposed for run.py to validate YAML
SIGN_MIX_VALUES = ("clean", "mixed", "idiosyncratic")
SPARSITY_VALUES = ("sparse", "moderate", "dense")
FACTOR_COV_VALUES = ("orthogonal", "block_diag", "matf_calibrated")
RESIDUAL_COV_VALUES = ("diagonal", "rank1_plus_diag")

# Sparsity → number of active factors out of M (assumes M=9; scales)
_SPARSITY_NUM_ACTIVE = {"sparse": 2, "moderate": 4, "dense": 7}

# Mixed-sign-mix probability of an asset having one cluster-flipped sign
_MIXED_FLIP_PROB = 0.25

# Block-diagonal factor cov: within-block correlation
_BLOCK_DIAG_RHO = 0.4
_BLOCK_DIAG_N_BLOCKS = 3


@dataclass(frozen=True)
class DGPConfig:
    """
    Configuration for a single DGP draw.

    All fields have defaults so a regime YAML can override only the axes
    it varies. ``seed`` is the only field a regime YAML never sets
    directly — the runner enumerates seeds across cells.
    """

    T: int = 120
    N: int = 50
    M: int = 9
    K: int = 6
    rho_beta: float = 0.8
    sign_mix: str = "clean"
    sparsity: str = "moderate"
    snr: float = 0.25
    factor_cov: str = "block_diag"
    residual_cov: str = "diagonal"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.sign_mix not in SIGN_MIX_VALUES:
            raise ValueError(
                f"sign_mix must be one of {SIGN_MIX_VALUES}, got {self.sign_mix!r}"
            )
        if self.sparsity not in SPARSITY_VALUES:
            raise ValueError(
                f"sparsity must be one of {SPARSITY_VALUES}, got {self.sparsity!r}"
            )
        if self.factor_cov not in FACTOR_COV_VALUES:
            raise ValueError(
                f"factor_cov must be one of {FACTOR_COV_VALUES}, "
                f"got {self.factor_cov!r}"
            )
        if self.residual_cov not in RESIDUAL_COV_VALUES:
            raise ValueError(
                f"residual_cov must be one of {RESIDUAL_COV_VALUES}, "
                f"got {self.residual_cov!r}"
            )
        if not (0.0 < self.snr < 1.0):
            raise ValueError(f"snr must lie in (0, 1), got {self.snr!r}")
        if not (0.0 <= self.rho_beta <= 1.0):
            raise ValueError(f"rho_beta must lie in [0, 1], got {self.rho_beta!r}")
        if self.K > self.N:
            raise ValueError(f"K ({self.K}) must be ≤ N ({self.N})")


@dataclass
class DGPOutput:
    """One realised synthetic panel."""

    X: pd.DataFrame  # (T, M) factor returns
    Y: pd.DataFrame  # (T, N) asset returns
    beta_true: pd.DataFrame  # (N, M) true loadings
    clusters_true: pd.Series  # (N,) integer cluster labels
    factor_premia: pd.Series  # (M,) λ vector for risk-premium recovery
    realised_snr: pd.Series = field(default=None)  # (N,) per-asset realised R²


def make_synthetic_panel(config: DGPConfig) -> DGPOutput:
    """
    Generate one synthetic panel matching ``config``.

    The pipeline is:

    1. ``X``: draw factor returns from N(0, Σ_X) — covariance chosen by
       ``config.factor_cov``.
    2. ``β``: build K cluster templates with ``_SPARSITY_NUM_ACTIVE[sparsity]``
       active factors each, then draw N assets by anchoring to a template
       (correlation ``rho_beta``) and applying the within-cluster sign-mix
       rule.
    3. ``ε``: draw residuals from N(0, Σ_ε) — covariance chosen by
       ``config.residual_cov``.
    4. Scale ε per-asset so that ``Var(Xβ_k') / (Var(Xβ_k') + Var(ε_k)) = snr``
       for every asset k.
    5. Form Y = X β' + ε.

    Returns a :class:`DGPOutput` with everything labelled and indexed
    consistently. Asset names ``A0..A{N-1}``, factor names ``F0..F{M-1}``.

    The ``matf_calibrated`` factor-cov regime is a special case: it produces
    the production MATF-CMA universe (real loadings, real factor covariance,
    R^2-calibrated noise) and is delegated to :func:`_make_calibrated_panel`.
    """
    if config.factor_cov == "matf_calibrated":
        return _make_calibrated_panel(config)

    rng = np.random.default_rng(config.seed)

    X = _generate_factor_returns(config.T, config.M, config.factor_cov, rng)
    beta, clusters = _generate_true_beta(
        config.N, config.M, config.K,
        config.rho_beta, config.sign_mix, config.sparsity, rng,
    )
    eps = _generate_errors(config.T, config.N, config.residual_cov, rng)

    # SNR enforcement: per-asset rescale of ε
    signal = X @ beta.T                          # (T, N)
    signal_var = signal.var(axis=0, ddof=1)      # (N,)
    eps_var = eps.var(axis=0, ddof=1)            # (N,)
    # Solve target_eps_var: snr = signal_var / (signal_var + target_eps_var)
    #                    => target_eps_var = signal_var * (1/snr - 1)
    target_eps_var = signal_var * (1.0 / config.snr - 1.0)
    # Guard against zero-signal assets (sparsity=0 edge case): set unit variance
    scale = np.where(
        (eps_var > 1e-12) & (signal_var > 1e-12),
        np.sqrt(target_eps_var / eps_var),
        1.0,
    )
    eps = eps * scale  # broadcast over T

    y = signal + eps

    # Factor premia λ: simple choice — mean absolute loading per factor + floor.
    # This gives the factor_rp_rmse metric something nontrivial to compare.
    factor_premia_vec = np.abs(beta).mean(axis=0) * 0.5 + 0.02

    # Realised SNR (sanity check; should be close to config.snr)
    realised_snr = signal_var / (signal_var + (eps ** 2).mean(axis=0))

    assets = [f"A{i}" for i in range(config.N)]
    factors = [f"F{j}" for j in range(config.M)]
    time_index = pd.RangeIndex(config.T, name="t")

    return DGPOutput(
        X=pd.DataFrame(X, index=time_index, columns=factors),
        Y=pd.DataFrame(y, index=time_index, columns=assets),
        beta_true=pd.DataFrame(beta, index=assets, columns=factors),
        clusters_true=pd.Series(clusters, index=assets, name="cluster"),
        factor_premia=pd.Series(factor_premia_vec, index=factors, name="lambda"),
        realised_snr=pd.Series(realised_snr, index=assets, name="realised_snr"),
    )


# ── Helpers ──────────────────────────────────────────────────────────


def _load_calibration():
    """Import the production calibration module (package or direct run)."""
    try:
        from . import matf_calibration as cal
    except ImportError:  # direct script execution puts simulations/ on path
        import matf_calibration as cal
    return cal


def _make_calibrated_panel(config: DGPConfig) -> DGPOutput:
    """
    Production-calibrated panel for ``factor_cov="matf_calibrated"``.

    Uses the MATF-CMA universe directly: factor returns drawn from the
    Exhibit 10 covariance, true loadings from Exhibit 15, and per-asset
    idiosyncratic noise calibrated to the published R^2. Only ``config.T`` and
    ``config.seed`` are honoured; the random-cluster axes (``N``, ``K``,
    ``sparsity``, ``sign_mix``, ``snr``, ``residual_cov``) are ignored — ``N``
    is fixed at the universe size and the loadings are not synthetic. Clusters
    are the three asset classes (bonds / equities / alternatives).
    """
    cal = _load_calibration()
    rng = np.random.default_rng(config.seed)

    X = cal.draw_factor_returns(config.T, rng)            # (T, M)
    beta = cal.BETA                                       # (N, M)
    signal = X @ beta.T                                   # (T, N)
    eps = cal.draw_idiosyncratic(config.T, rng)           # (T, N)
    y = signal + eps

    signal_var = signal.var(axis=0, ddof=1)
    realised_snr = signal_var / (signal_var + eps.var(axis=0, ddof=1))

    time_index = pd.RangeIndex(config.T, name="t")
    return DGPOutput(
        X=pd.DataFrame(X, index=time_index, columns=cal.FACTORS),
        Y=pd.DataFrame(y, index=time_index, columns=cal.ASSETS),
        beta_true=pd.DataFrame(beta, index=cal.ASSETS, columns=cal.FACTORS),
        clusters_true=pd.Series(cal.class_labels(), index=cal.ASSETS, name="cluster"),
        factor_premia=pd.Series(cal.FACTOR_PREMIA, index=cal.FACTORS, name="lambda"),
        realised_snr=pd.Series(realised_snr, index=cal.ASSETS, name="realised_snr"),
    )


def _generate_factor_returns(
    T: int, M: int, cov_type: str, rng: np.random.Generator,
) -> np.ndarray:
    """Draw X ~ N(0, Σ_X) of shape (T, M) per the requested cov_type."""
    if cov_type == "orthogonal":
        return rng.standard_normal((T, M))

    if cov_type == "block_diag":
        # Build block-diagonal Σ_X with within-block correlation _BLOCK_DIAG_RHO.
        cov = np.eye(M)
        block_size = max(1, M // _BLOCK_DIAG_N_BLOCKS)
        for b in range(_BLOCK_DIAG_N_BLOCKS):
            start = b * block_size
            end = M if b == _BLOCK_DIAG_N_BLOCKS - 1 else (b + 1) * block_size
            if end > start + 1:
                block = (
                    _BLOCK_DIAG_RHO * np.ones((end - start, end - start))
                    + (1 - _BLOCK_DIAG_RHO) * np.eye(end - start)
                )
                cov[start:end, start:end] = block
        L = np.linalg.cholesky(cov)
        Z = rng.standard_normal((T, M))
        return Z @ L.T

    if cov_type == "matf_calibrated":
        # Production-calibrated factor covariance (Exhibit 10), drawn at
        # monthly scale. M must match the calibrated factor count.
        cal = _load_calibration()
        if M != cal.N_FACTORS:
            raise ValueError(
                f"matf_calibrated factor cov is defined for {cal.N_FACTORS} "
                f"factors; got M={M}."
            )
        L = np.linalg.cholesky(cal.SIGMA_F_MONTHLY)
        return rng.standard_normal((T, M)) @ L.T

    raise ValueError(f"Unknown factor_cov: {cov_type!r}")


def _generate_true_beta(
    N: int,
    M: int,
    K: int,
    rho_beta: float,
    sign_mix: str,
    sparsity: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the true β matrix and cluster assignment.

    Strategy:

    - Build K cluster templates τ_k ∈ R^M, each with ``n_active`` factors
      non-zero, signs drawn uniformly, magnitudes drawn from U[0.3, 0.7].
    - Assign N assets to clusters approximately uniformly.
    - For asset i in cluster k:
        β_i = ρ_β · τ_k + (1 - ρ_β) · η_i      (magnitudes vary)
      where η_i is a perturbation supported only on τ_k's active set, so
      the sparsity pattern is preserved by construction.
    - Apply the sign_mix rule to active factors only:
        clean         → no change (preserve τ_k's signs)
        mixed         → with probability _MIXED_FLIP_PROB flip one
                        randomly-chosen active factor's sign
        idiosyncratic → for each active factor independently, flip with
                        probability 0.5 (breaks all cluster sign coherence)
    """
    n_active = _SPARSITY_NUM_ACTIVE[sparsity]
    if n_active > M:
        raise ValueError(f"n_active ({n_active}) > M ({M})")

    # Build cluster templates τ ∈ R^{K x M}
    templates = np.zeros((K, M))
    template_active = np.zeros((K, M), dtype=bool)
    for k in range(K):
        active_idx = rng.choice(M, size=n_active, replace=False)
        template_active[k, active_idx] = True
        signs = rng.choice([-1.0, 1.0], size=n_active)
        magnitudes = 0.3 + 0.4 * rng.random(n_active)  # U[0.3, 0.7]
        templates[k, active_idx] = signs * magnitudes

    # Assign assets to clusters: approximately uniform via floor-divide
    # plus shuffle to break index correlation.
    base_assignment = np.tile(np.arange(K), N // K + 1)[:N]
    clusters = rng.permutation(base_assignment)

    # Build per-asset β with magnitude perturbation supported on the
    # cluster's active set, then apply sign_mix to active factors.
    beta = np.zeros((N, M))
    for i in range(N):
        k = clusters[i]
        active = template_active[k]
        # Magnitude blend: ρ · template + (1-ρ) · perturbation on active factors only
        perturbation = np.zeros(M)
        if active.sum() > 0:
            perturbation[active] = (
                0.3 + 0.4 * rng.random(active.sum())
            ) * np.sign(templates[k, active])
        beta[i] = rho_beta * templates[k] + (1.0 - rho_beta) * perturbation

        # Apply sign_mix to active factors
        active_idx = np.where(active)[0]
        if sign_mix == "clean" or len(active_idx) == 0:
            pass
        elif sign_mix == "mixed":
            if rng.random() < _MIXED_FLIP_PROB:
                flip = rng.choice(active_idx)
                beta[i, flip] = -beta[i, flip]
        elif sign_mix == "idiosyncratic":
            for j in active_idx:
                if rng.random() < 0.5:
                    beta[i, j] = -beta[i, j]

    return beta, clusters


def _generate_errors(
    T: int, N: int, residual_cov: str, rng: np.random.Generator,
) -> np.ndarray:
    """Draw residuals ε of shape (T, N) per the requested residual_cov."""
    if residual_cov == "diagonal":
        return rng.standard_normal((T, N))

    if residual_cov == "rank1_plus_diag":
        # Common factor + idiosyncratic component. The common factor
        # introduces residual-covariance structure that the methodology
        # should be robust to (it's not in the modelled X).
        common = rng.standard_normal(T)
        loadings = 0.3 * rng.standard_normal(N)
        idio = rng.standard_normal((T, N))
        return np.outer(common, loadings) + idio

    raise ValueError(f"Unknown residual_cov: {residual_cov!r}")
