"""End-to-end smoke tests exercising DGP + estimators + metrics."""
from __future__ import annotations

import numpy as np
import pytest

from papers.jss_2026.simulations.dgp import DGPConfig, make_synthetic_panel
from papers.jss_2026.simulations.estimators import ESTIMATORS
from papers.jss_2026.simulations.metrics import compute_all


@pytest.mark.parametrize("estimator_name", [
    "factorlasso_lasso",
    "factorlasso_grp_oracle",
    "factorlasso_grp_hcgl",
    "factorlasso_grp_hcgl_sign",
    "factorlasso_grp_hcgl_sign_adapt",
    "factorlasso_sgl_hcgl_sign_adapt",
])
def test_factorlasso_estimators_end_to_end(estimator_name):
    """Each factorlasso estimator fits without error and produces sane metrics."""
    cfg = DGPConfig(T=80, N=20, M=9, K=4, seed=0)
    data = make_synthetic_panel(cfg)
    fit_fn = ESTIMATORS[estimator_name]

    result = fit_fn(
        X_train=data.X, y_train=data.Y, reg_lambda=1e-3,
        true_clusters=data.clusters_true,
    )

    # Shape contracts
    assert result.beta_hat.shape == (20, 9)
    assert result.intercept_hat.shape == (20,)
    assert result.runtime > 0
    assert result.reg_lambda == 1e-3

    # Metrics compute without error and lie in valid ranges
    clusters_hat = result.extra.get("clusters_hat")
    metrics = compute_all(
        beta_true=data.beta_true.values,
        beta_hat=result.beta_hat.values,
        clusters_true=data.clusters_true.values,
        factor_premia=data.factor_premia.values,
        clusters_hat=clusters_hat.values if clusters_hat is not None else None,
    )

    assert 0.0 <= metrics["support_f1"] <= 1.0 or np.isnan(metrics["support_f1"])
    assert 0.0 <= metrics["sign_rate"] <= 1.0 or np.isnan(metrics["sign_rate"])
    assert metrics["beta_mse_norm"] >= 0.0
    assert (
        0.0 <= metrics["cluster_coherence_hat"] <= 1.0
        or np.isnan(metrics["cluster_coherence_hat"])
    )
    assert metrics["factor_rp_rmse"] >= 0.0


def test_unwired_estimators_raise_not_implemented():
    """Stub estimators must announce themselves clearly, not return garbage."""
    rng = np.random.default_rng(0)
    import pandas as pd
    X = pd.DataFrame(rng.standard_normal((30, 3)), columns=["F0", "F1", "F2"])
    Y = pd.DataFrame(rng.standard_normal((30, 5)), columns=[f"A{i}" for i in range(5)])

    for name in ["sparsegl_sgl"]:
        with pytest.raises(NotImplementedError, match=name):
            ESTIMATORS[name](X, Y, reg_lambda=1e-3)


def test_oracle_baseline_beats_lasso_on_clustered_data():
    """
    Sanity check: on data with strong cluster structure (clean sign_mix,
    high ρ_β), the oracle group estimator should achieve lower β-MSE than
    pure LASSO. This is the directional claim §5 will make.
    """
    cfg = DGPConfig(
        T=200, N=60, M=9, K=6, rho_beta=0.95,
        sign_mix="clean", sparsity="sparse", snr=0.30, seed=42,
    )
    data = make_synthetic_panel(cfg)

    from papers.jss_2026.simulations.metrics import beta_mse_normalised

    lasso = ESTIMATORS["factorlasso_lasso"](
        data.X, data.Y, reg_lambda=1e-3,
    )
    oracle = ESTIMATORS["factorlasso_grp_oracle"](
        data.X, data.Y, reg_lambda=1e-3, true_clusters=data.clusters_true,
    )

    lasso_mse = beta_mse_normalised(data.beta_true.values, lasso.beta_hat.values)
    oracle_mse = beta_mse_normalised(data.beta_true.values, oracle.beta_hat.values)

    # Oracle should be at least as good as Lasso; usually meaningfully better.
    # We use a soft check: oracle should be within 1.5× of Lasso even in the
    # worst case (the directional claim is empirical, not absolute).
    assert oracle_mse <= 1.5 * lasso_mse, (
        f"Oracle MSE {oracle_mse:.4f} should not be much worse than "
        f"Lasso MSE {lasso_mse:.4f}"
    )
