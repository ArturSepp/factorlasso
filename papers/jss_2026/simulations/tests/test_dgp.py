"""Tests for the DGP module."""
from __future__ import annotations

import numpy as np
import pytest

from papers.jss_2026.simulations.dgp import (
    DGPConfig,
    make_synthetic_panel,
)

# ── Shape and labelling ──────────────────────────────────────────────


def test_basic_shapes_and_labels():
    cfg = DGPConfig(T=100, N=20, M=9, K=4, seed=0)
    out = make_synthetic_panel(cfg)
    assert out.X.shape == (100, 9)
    assert out.Y.shape == (100, 20)
    assert out.beta_true.shape == (20, 9)
    assert out.clusters_true.shape == (20,)
    assert out.factor_premia.shape == (9,)
    assert list(out.X.columns) == [f"F{i}" for i in range(9)]
    assert list(out.Y.columns) == [f"A{i}" for i in range(20)]
    assert list(out.beta_true.index) == list(out.Y.columns)
    assert list(out.beta_true.columns) == list(out.X.columns)
    assert list(out.clusters_true.index) == list(out.Y.columns)


# ── Reproducibility ──────────────────────────────────────────────────


def test_reproducibility_same_seed():
    cfg = DGPConfig(T=80, N=30, M=9, K=5, seed=123)
    a = make_synthetic_panel(cfg)
    b = make_synthetic_panel(cfg)
    np.testing.assert_array_equal(a.X.values, b.X.values)
    np.testing.assert_array_equal(a.Y.values, b.Y.values)
    np.testing.assert_array_equal(a.beta_true.values, b.beta_true.values)
    np.testing.assert_array_equal(a.clusters_true.values, b.clusters_true.values)


def test_different_seeds_yield_different_data():
    a = make_synthetic_panel(DGPConfig(seed=1))
    b = make_synthetic_panel(DGPConfig(seed=2))
    assert not np.allclose(a.X.values, b.X.values)
    assert not np.allclose(a.beta_true.values, b.beta_true.values)


# ── SNR enforcement ──────────────────────────────────────────────────


@pytest.mark.parametrize("target_snr", [0.10, 0.25, 0.50])
def test_realised_snr_matches_target(target_snr):
    cfg = DGPConfig(T=2000, N=50, M=9, K=6, snr=target_snr, seed=7)
    out = make_synthetic_panel(cfg)
    # Per-asset realised SNR should center on target (large T → tight)
    median_snr = out.realised_snr.median()
    assert abs(median_snr - target_snr) < 0.05, (
        f"Median realised SNR {median_snr:.3f} differs from target "
        f"{target_snr} by more than 0.05"
    )


# ── Sparsity ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("sparsity,expected_active", [
    ("sparse", 2), ("moderate", 4), ("dense", 7),
])
def test_sparsity_controls_active_factor_count(sparsity, expected_active):
    cfg = DGPConfig(N=50, M=9, K=6, sparsity=sparsity, seed=0)
    out = make_synthetic_panel(cfg)
    # Active factors per row should equal expected_active (template-driven)
    active_per_row = (np.abs(out.beta_true.values) > 1e-10).sum(axis=1)
    # Every asset's active count should be exactly expected_active
    # (cluster templates have exactly n_active active factors, magnitude
    # perturbation only affects existing active factors)
    assert (active_per_row == expected_active).all(), (
        f"Expected {expected_active} active factors per row, got "
        f"{np.unique(active_per_row)}"
    )


# ── sign_mix semantics ───────────────────────────────────────────────


def _within_cluster_sign_coherence(beta, clusters):
    """Helper: fraction of (cluster, factor) pairs where all members agree on sign."""
    coh = []
    for k in np.unique(clusters):
        members = beta[clusters == k]
        signs = np.sign(members)
        for j in range(signs.shape[1]):
            col = signs[:, j]
            nonzero = col[col != 0]
            if len(nonzero) == 0:
                continue
            coh.append(len(np.unique(nonzero)) == 1)
    return float(np.mean(coh))


def test_clean_sign_mix_has_perfect_within_cluster_coherence():
    cfg = DGPConfig(N=60, M=9, K=6, sign_mix="clean", sparsity="moderate", seed=0)
    out = make_synthetic_panel(cfg)
    coh = _within_cluster_sign_coherence(
        out.beta_true.values, out.clusters_true.values,
    )
    assert coh == 1.0, f"Expected perfect coherence for sign_mix=clean, got {coh}"


def test_idiosyncratic_sign_mix_breaks_within_cluster_coherence():
    cfg = DGPConfig(
        N=200, M=9, K=6, sign_mix="idiosyncratic", sparsity="moderate", seed=0,
    )
    out = make_synthetic_panel(cfg)
    coh = _within_cluster_sign_coherence(
        out.beta_true.values, out.clusters_true.values,
    )
    # idiosyncratic mode randomises signs independently per asset per factor.
    # With M=9, N=200, K=6 the expected coherence is well below 1.
    assert coh < 0.95, (
        f"Expected coherence well below 1.0 for sign_mix=idiosyncratic, got {coh}"
    )


def test_mixed_sign_mix_is_between_clean_and_idiosyncratic():
    common = dict(N=100, M=9, K=6, sparsity="moderate", seed=0)
    clean = _within_cluster_sign_coherence(
        make_synthetic_panel(DGPConfig(**common, sign_mix="clean")).beta_true.values,
        make_synthetic_panel(DGPConfig(**common, sign_mix="clean")).clusters_true.values,
    )
    mixed = _within_cluster_sign_coherence(
        make_synthetic_panel(DGPConfig(**common, sign_mix="mixed")).beta_true.values,
        make_synthetic_panel(DGPConfig(**common, sign_mix="mixed")).clusters_true.values,
    )
    idio = _within_cluster_sign_coherence(
        make_synthetic_panel(
            DGPConfig(**common, sign_mix="idiosyncratic"),
        ).beta_true.values,
        make_synthetic_panel(
            DGPConfig(**common, sign_mix="idiosyncratic"),
        ).clusters_true.values,
    )
    assert clean >= mixed >= idio, (
        f"Expected coherence ordering clean ≥ mixed ≥ idiosyncratic, "
        f"got {clean:.3f} / {mixed:.3f} / {idio:.3f}"
    )


# ── Cluster assignment ──────────────────────────────────────────────


def test_clusters_assigned_approximately_uniformly():
    cfg = DGPConfig(N=120, K=6, seed=42)
    out = make_synthetic_panel(cfg)
    counts = out.clusters_true.value_counts().sort_index()
    # With N=120 and K=6, exactly 20 per cluster
    assert (counts == 20).all(), f"Expected 20 per cluster, got {counts.to_dict()}"


def test_cluster_count_respected():
    cfg = DGPConfig(N=50, K=3, seed=0)
    out = make_synthetic_panel(cfg)
    assert out.clusters_true.nunique() == 3


# ── Factor covariance ───────────────────────────────────────────────


def test_orthogonal_factor_cov_has_uncorrelated_factors():
    cfg = DGPConfig(T=5000, M=9, factor_cov="orthogonal", seed=0)
    out = make_synthetic_panel(cfg)
    corr = out.X.corr().values
    off_diag = corr[~np.eye(9, dtype=bool)]
    # Large T → off-diagonal correlations should be near zero
    assert np.abs(off_diag).max() < 0.1, (
        f"Expected near-zero off-diagonal corr, got max {np.abs(off_diag).max():.3f}"
    )


def test_block_diag_factor_cov_has_block_structure():
    cfg = DGPConfig(T=5000, M=9, factor_cov="block_diag", seed=0)
    out = make_synthetic_panel(cfg)
    corr = out.X.corr().values
    # Block 0 = F0-F2 should have within-block ρ ~0.4, between-block ~0
    within_block = corr[0, 1], corr[0, 2], corr[1, 2]
    between_block = corr[0, 3], corr[0, 6], corr[3, 6]
    assert all(0.3 < x < 0.5 for x in within_block), (
        f"Within-block correlations not near 0.4: {within_block}"
    )
    assert all(abs(x) < 0.1 for x in between_block), (
        f"Between-block correlations not near 0: {between_block}"
    )


def test_matf_calibrated_produces_panel():
    """The calibrated regime is implemented (Section 6.5 of the paper):
    it delegates to ``_make_calibrated_panel`` and returns a labelled
    panel with ground-truth loadings. The pre-benchmark test asserted
    ``NotImplementedError`` here and went stale when the calibrated
    benchmark landed."""
    out = make_synthetic_panel(DGPConfig(factor_cov="matf_calibrated", seed=0))
    assert out.X.shape[0] == out.Y.shape[0]
    assert out.beta_true.shape == (out.Y.shape[1], out.X.shape[1])
    assert len(out.clusters_true) == out.Y.shape[1]


# ── Validation ──────────────────────────────────────────────────────


def test_invalid_sign_mix_raises():
    with pytest.raises(ValueError, match="sign_mix"):
        DGPConfig(sign_mix="bogus")


def test_invalid_sparsity_raises():
    with pytest.raises(ValueError, match="sparsity"):
        DGPConfig(sparsity="hyper-dense")


def test_invalid_snr_raises():
    with pytest.raises(ValueError, match="snr"):
        DGPConfig(snr=1.5)
    with pytest.raises(ValueError, match="snr"):
        DGPConfig(snr=0.0)


def test_K_larger_than_N_raises():
    with pytest.raises(ValueError, match="K"):
        DGPConfig(N=5, K=10)
