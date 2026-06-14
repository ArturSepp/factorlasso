"""Tests for the metrics module."""
from __future__ import annotations

import math

import numpy as np
import pytest

from papers.jss_2026.simulations.metrics import (
    beta_mse_normalised,
    cluster_recovery_ari,
    cluster_sign_coherence,
    compute_all,
    factor_rp_rmse,
    oos_r2,
    sign_agreement_rate,
    support_recovery_f1,
)

# ── support_recovery_f1 ──────────────────────────────────────────────


def test_support_f1_perfect_recovery():
    beta_true = np.array([[1.0, 0.0, -1.0], [0.0, 2.0, 0.0]])
    beta_hat = np.array([[0.9, 0.0, -1.1], [0.0, 2.1, 0.0]])
    assert support_recovery_f1(beta_true, beta_hat) == 1.0


def test_support_f1_zero_recovery():
    beta_true = np.array([[1.0, -1.0], [2.0, 0.0]])
    beta_hat = np.zeros_like(beta_true)
    assert support_recovery_f1(beta_true, beta_hat) == 0.0


def test_support_f1_all_false_positives():
    beta_true = np.zeros((3, 3))
    beta_hat = np.ones((3, 3))
    # No true support → undefined
    assert math.isnan(support_recovery_f1(beta_true, beta_hat))


def test_support_f1_handles_numerical_tol():
    beta_true = np.array([[1.0, 0.0]])
    beta_hat = np.array([[1.0, 1e-9]])  # tiny non-zero, should count as zero
    assert support_recovery_f1(beta_true, beta_hat, tol=1e-6) == 1.0
    # With stricter tol, the 1e-9 still counts as zero
    assert support_recovery_f1(beta_true, beta_hat, tol=1e-12) < 1.0


# ── sign_agreement_rate ──────────────────────────────────────────────


def test_sign_agreement_perfect():
    beta_true = np.array([[1.0, -1.0], [0.5, -0.5]])
    beta_hat = np.array([[0.9, -0.7], [0.4, -0.6]])
    assert sign_agreement_rate(beta_true, beta_hat) == 1.0


def test_sign_agreement_all_flipped():
    beta_true = np.array([[1.0, -1.0], [0.5, -0.5]])
    beta_hat = -beta_true
    assert sign_agreement_rate(beta_true, beta_hat) == 0.0


def test_sign_agreement_ignores_true_zeros():
    # Only non-zero true cells contribute. β_hat at a zero true cell
    # doesn't matter for sign_agreement.
    beta_true = np.array([[1.0, 0.0, -1.0]])
    beta_hat = np.array([[1.0, 100.0, -1.0]])  # bogus value at the zero cell
    assert sign_agreement_rate(beta_true, beta_hat) == 1.0


def test_sign_agreement_returns_nan_for_all_zero_truth():
    beta_true = np.zeros((2, 3))
    beta_hat = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    assert math.isnan(sign_agreement_rate(beta_true, beta_hat))


# ── beta_mse_normalised ──────────────────────────────────────────────


def test_beta_mse_normalised_perfect_fit():
    beta = np.array([[1.0, -1.0], [2.0, 0.5]])
    assert beta_mse_normalised(beta, beta) == 0.0


def test_beta_mse_normalised_constant_offset():
    beta_true = np.array([[1.0, -1.0], [2.0, 0.5]])
    beta_hat = beta_true + 0.1
    # |Δβ|²_F = 0.01 * 4 = 0.04;  |β|²_F = 1 + 1 + 4 + 0.25 = 6.25
    expected = 0.04 / 6.25
    assert beta_mse_normalised(beta_true, beta_hat) == pytest.approx(expected)


def test_beta_mse_normalised_zero_truth_yields_nan():
    beta_true = np.zeros((3, 3))
    beta_hat = np.ones((3, 3))
    assert math.isnan(beta_mse_normalised(beta_true, beta_hat))


# ── cluster_sign_coherence ───────────────────────────────────────────


def test_cluster_coherence_perfect():
    # 2 clusters of 3 assets each, sharing signs within cluster
    beta = np.array([
        [1.0, -1.0, 0.0],
        [1.0, -0.5, 0.0],
        [0.5, -0.8, 0.0],
        [-1.0, 1.0, 0.0],
        [-0.7, 0.5, 0.0],
        [-0.3, 1.2, 0.0],
    ])
    clusters = np.array([0, 0, 0, 1, 1, 1])
    # Both clusters perfectly coherent on F0 and F1; F2 all-zero
    # All-zero columns are excluded from the average.
    coh = cluster_sign_coherence(beta, clusters)
    assert coh == 1.0


def test_cluster_coherence_broken_one_factor():
    # 2 clusters: cluster 0 fully coherent, cluster 1 has one disagreeing sign
    beta = np.array([
        [1.0, -1.0],
        [1.0, -0.5],
        [-1.0, 1.0],
        [1.0, 1.0],   # disagrees with cluster 1 on F0
    ])
    clusters = np.array([0, 0, 1, 1])
    # Cluster 0: 2/2 coherent (F0, F1). Cluster 1: 1/2 coherent (F0 disagrees, F1 agrees)
    # Mean coherence = (1 + 1 + 0 + 1) / 4 = 0.75
    coh = cluster_sign_coherence(beta, clusters)
    assert coh == pytest.approx(0.75)


def test_cluster_coherence_excludes_all_zero_factors():
    beta = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    clusters = np.array([0, 0])
    # F1 is structurally zero → excluded. F0 is coherent.
    assert cluster_sign_coherence(beta, clusters) == 1.0


# ── oos_r2 ───────────────────────────────────────────────────────────


def test_oos_r2_perfect_prediction():
    y_test = np.random.default_rng(0).standard_normal((50, 3))
    assert oos_r2(y_test, y_test) == pytest.approx(1.0)


def test_oos_r2_mean_baseline_gives_zero():
    y_test = np.random.default_rng(0).standard_normal((100, 3))
    y_pred = np.broadcast_to(y_test.mean(axis=0, keepdims=True), y_test.shape)
    # Predicting the test-window mean gives R² = 0 exactly
    assert oos_r2(y_test, y_pred) == pytest.approx(0.0, abs=1e-10)


def test_oos_r2_constant_y_test_returns_nan():
    y_test = np.ones((10, 3))  # zero variance per column
    y_pred = np.zeros_like(y_test)
    assert math.isnan(oos_r2(y_test, y_pred))


# ── factor_rp_rmse ───────────────────────────────────────────────────


def test_factor_rp_rmse_zero_for_identical():
    beta = np.array([[1.0, -1.0], [0.5, 0.5]])
    lam = np.array([0.1, 0.2])
    assert factor_rp_rmse(beta, beta, lam) == 0.0


def test_factor_rp_rmse_known_value():
    beta_true = np.array([[1.0, 0.0]])
    beta_hat = np.array([[2.0, 0.0]])
    lam = np.array([0.5, 0.3])
    # RP_true = 0.5, RP_hat = 1.0, diff = 0.5, RMSE = 0.5
    assert factor_rp_rmse(beta_true, beta_hat, lam) == pytest.approx(0.5)


# ── cluster_recovery_ari ─────────────────────────────────────────────


def test_ari_identical_clusterings():
    c = np.array([0, 0, 1, 1, 2, 2])
    assert cluster_recovery_ari(c, c) == 1.0


def test_ari_label_permutation_invariant():
    # Same partition, different labels — ARI should be 1
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([5, 5, 9, 9, 7, 7])
    assert cluster_recovery_ari(a, b) == 1.0


def test_ari_completely_different():
    # All-same vs all-different — ARI = 0 (no agreement beyond chance)
    a = np.array([0, 0, 0, 0])
    b = np.array([0, 1, 2, 3])
    # Both are degenerate cases; expect 0 or 1 depending on convention.
    # For all-same and all-different, the standard convention is ARI = 0
    # (in fact for these particular inputs, the formula gives ~1 because
    # both partitions are extremes; specifically when one is a single
    # cluster, ARI is not well-behaved). Test something more interesting:
    a = np.array([0, 0, 1, 1, 0, 0])
    b = np.array([0, 1, 0, 1, 0, 1])
    val = cluster_recovery_ari(a, b)
    assert -0.5 < val < 0.5, f"Expected ARI near 0 for unrelated clusterings, got {val}"


# ── compute_all ──────────────────────────────────────────────────────


def test_compute_all_returns_expected_keys():
    beta_true = np.array([[1.0, 0.0], [0.0, 1.0]])
    beta_hat = beta_true.copy()
    clusters = np.array([0, 1])
    lam = np.array([0.1, 0.2])

    out = compute_all(beta_true, beta_hat, clusters, lam)
    expected_keys = {
        "support_f1", "sign_rate", "beta_mse_norm",
        "cluster_coherence_hat", "factor_rp_rmse",
        "oos_r2", "cluster_ari",
    }
    assert set(out.keys()) == expected_keys


def test_compute_all_oos_r2_nan_without_test_data():
    beta = np.array([[1.0, 0.0]])
    clusters = np.array([0])
    lam = np.array([0.1, 0.2])
    out = compute_all(beta, beta, clusters, lam)
    assert math.isnan(out["oos_r2"])


def test_compute_all_perfect_fit_gives_perfect_metrics():
    beta_true = np.array([[1.0, -0.5, 0.0], [0.0, 0.8, -0.3], [-1.0, 0.0, 0.4]])
    clusters = np.array([0, 0, 1])
    lam = np.array([0.1, 0.2, 0.05])
    out = compute_all(beta_true, beta_true, clusters, lam)
    assert out["support_f1"] == 1.0
    assert out["sign_rate"] == 1.0
    assert out["beta_mse_norm"] == 0.0
    assert out["factor_rp_rmse"] == 0.0
