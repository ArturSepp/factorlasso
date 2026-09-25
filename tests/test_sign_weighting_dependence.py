"""Regression contracts for weighted, masked and dependence-aware signs."""

import numpy as np
import pandas as pd
import pytest

from factorlasso import LassoModel
from factorlasso.lasso_estimator import get_x_y_np
from factorlasso.sign_constraints import (
    _compute_sign_matrix_per_response,
    _compute_sign_vector,
    derive_sign_constraints,
)


def reference(x, y, span):
    """Independent scalar date-score calculation on the original date grid."""
    w = np.ones(len(x)) if span is None else (1 - 2 / (span + 1)) ** np.arange(len(x) - 1, -1, -1)
    numerator = denominator = sw = sw2 = 0.0
    rows = []
    for t in range(len(x)):
        ys = [v for v in y[t] if np.isfinite(v)] if np.isfinite(x[t]) else []
        if ys:
            numerator += w[t] * x[t] * sum(ys)
            denominator += w[t] * x[t] ** 2 * len(ys)
            sw += w[t]
            sw2 += w[t] ** 2
        rows.append(ys)
    beta = numerator / denominator
    ess = sw**2 / sw2
    meat = sum(
        (w[t] * x[t] * sum(v - beta * x[t] for v in ys)) ** 2 for t, ys in enumerate(rows) if ys
    )
    se = np.sqrt(meat * ess / (ess - 1)) / denominator
    return beta, beta / se, ess


@pytest.mark.parametrize("span", [None, 12.0, 60.0])
def test_weighted_date_scores_match_independent_masked_reference(span):
    """Weights retain date age through both response and factor gaps."""
    rng = np.random.default_rng(281)
    x = rng.normal(size=(90, 3))
    y = 0.13 * x[:, :1] + rng.normal(size=(90, 4))
    x[:13, 0] = np.nan
    x[35:40, 1] = np.nan
    y[:40, 0] = np.nan
    y[::7, 2] = np.nan
    signs, slopes, diag = _compute_sign_vector(
        x,
        y,
        ewma_span=span,
        auto_sign_threshold_t=1.0,
        return_diagnostics=True,
        variance_estimator="date",
    )
    for j in range(3):
        beta, t, ess = reference(x[:, j], y, span)
        np.testing.assert_allclose(
            [slopes[j], diag["t_stats"][j], diag["effective_n"][j]], [beta, t, ess], rtol=1e-12
        )
        assert signs[j] == (np.sign(beta) if abs(t) >= 1 else 0)


def test_duplicate_responses_cannot_inflate_gate():
    """Repeating precisely the same history adds no date-level evidence."""
    rng = np.random.default_rng(182)
    x = rng.normal(size=(90, 1))
    e = rng.normal(size=(90, 1))
    e -= x * ((x.T @ e) / (x.T @ x))
    y = 0.91 * np.sqrt((e.T @ e).item() / 89 / (x.T @ x).item()) * x + e
    single = _compute_sign_vector(x, y, auto_sign_threshold_t=1.0, variance_estimator="date")[0]
    ten = _compute_sign_vector(
        x, np.tile(y, (1, 10)), auto_sign_threshold_t=1.0, variance_estimator="date"
    )[0]
    np.testing.assert_array_equal(single, ten)


@pytest.mark.parametrize("span", [None, 20.0])
def test_bulk_matches_single_response_and_factor_clusters(span):
    """The independent response path and singleton factor clusters agree."""
    rng = np.random.default_rng(542)
    x = rng.normal(size=(70, 4))
    y = rng.normal(size=(70, 3))
    y[:20, 0] = np.nan
    x[::8, 1] = np.nan
    s, b, d = _compute_sign_matrix_per_response(
        x, y, 1.0, True, ewma_span=span, return_diagnostics=True, variance_estimator="date"
    )
    for k in range(3):
        ss, bb, dd = _compute_sign_vector(
            x,
            y[:, k : k + 1],
            clusters=np.arange(4),
            ewma_span=span,
            auto_sign_threshold_t=1.0,
            return_diagnostics=True,
            variance_estimator="date",
        )
        np.testing.assert_array_equal(s[k], ss)
        np.testing.assert_allclose(b[k], bb, atol=1e-15)
        np.testing.assert_allclose(d["t_stats"][k], dd["t_stats"], atol=1e-14)


def test_fit_restores_missing_response_and_factor_masks(monkeypatch):
    """Check the actual solver preparation, not only a standalone helper."""
    import factorlasso.sign_constraints as sc

    rng = np.random.default_rng(34)
    x = pd.DataFrame(rng.normal(size=(300, 2)), columns=["x", "z"])
    y = pd.DataFrame({"y": 0.8 * x.x})
    y.iloc[:240] = np.nan
    x.iloc[:70, 1] = np.nan
    seen = []
    original = sc._compute_sign_matrix_per_response

    def spy(x_arr, y_arr, **kwargs):
        """Capture the actual analytics inputs."""
        seen.append((x_arr.copy(), y_arr.copy()))
        return original(x_arr, y_arr, **kwargs)

    monkeypatch.setattr(sc, "_compute_sign_matrix_per_response", spy)
    LassoModel(
        auto_sign_constraints=True,
        auto_sign_adaptive_weights=True,
        demean=False,
        span=None,
        auto_sign_variance="date",
    ).fit(x, y, verbose=False)
    assert np.isnan(seen[0][1][:240]).all()
    assert np.isnan(seen[0][0][:70, 1]).all()


def test_model_fit_span_and_path_route_same_sign_horizon():
    """A native fit override drives signs in single fits and paths."""
    rng = np.random.default_rng(57)
    x = pd.DataFrame(rng.normal(size=(180, 2)), columns=["x", "z"])
    y = pd.DataFrame(
        {"a": 0.7 * x.x + rng.normal(size=180), "b": -0.2 * x.z + rng.normal(size=180)}
    )
    y.iloc[:95, 0] = np.nan
    kwargs = dict(
        auto_sign_constraints=True,
        auto_sign_use_fit_span=True,
        auto_sign_adaptive_weights=True,
        span=60.0,
        reg_lambda=1e-3,
    )
    model = LassoModel(**kwargs, auto_sign_variance="date").fit(x, y, span=24.0, verbose=False)
    xx, yy, valid = get_x_y_np(x, y, span=24.0, demean=True)
    expected = _compute_sign_matrix_per_response(
        xx, np.where(valid > 0, yy, np.nan), 0.75, True, ewma_span=24.0, variance_estimator="date"
    )
    np.testing.assert_allclose(model.sign_slopes_, expected[1])
    assert model.effective_sign_span_ == 24.0
    path = LassoModel(**kwargs, auto_sign_variance="date").fit_reg_lambda_path(
        x, y, [1e-3], span=24.0
    )
    np.testing.assert_allclose(path[0].estimated_betas, model.estimated_betas, atol=1e-7)


def test_degenerate_inputs_and_ewma_validation():
    """No effective sample cannot pass a positive gate; perfect signals can."""
    x = np.arange(1.0, 11.0)[:, None]
    assert derive_sign_constraints(x, 2 * x, ewma_span=1.0, variance_estimator="date")[0] == 0
    assert derive_sign_constraints(x, 2 * x, ewma_span=3.0, variance_estimator="date")[0] == 1
    assert derive_sign_constraints(np.zeros_like(x), x, variance_estimator="date")[0] == 0
    for span in [0.0, -1.0, np.nan, np.inf]:
        with pytest.raises(ValueError):
            derive_sign_constraints(x, x, ewma_span=span, variance_estimator="date")


def test_leading_missing_rows_and_duplicate_weight_scale_invariance():
    """Padding missing history and copying all responses preserve evidence."""
    rng = np.random.default_rng(81)
    x = rng.normal(size=(70, 2))
    y = rng.normal(size=(70, 2))
    _, b, d = _compute_sign_vector(
        x, y, ewma_span=20.0, return_diagnostics=True, variance_estimator="date"
    )
    xp = np.vstack([np.zeros((30, 2)), x])
    yp = np.vstack([np.full((30, 2), np.nan), y])
    _, bp, dp = _compute_sign_vector(
        xp, np.tile(yp, (1, 3)), ewma_span=20.0, return_diagnostics=True, variance_estimator="date"
    )
    np.testing.assert_allclose(b, bp, atol=1e-15)
    np.testing.assert_allclose(d["t_stats"], dp["t_stats"], atol=1e-14)


def test_public_solver_remains_clarabel():
    """MOSEK belongs to private configuration, never the public dependency default."""
    assert LassoModel(auto_sign_variance="date").solver == "CLARABEL"
