"""Diagnostic-norm regression tests (v0.5.0 fix).

Solver weights carry ``sqrt(lambda)`` decay and are squared inside the
quadratic loss, so the *loss* has always been the nominal-span EWMA norm.
In-fit diagnostics must square the weights before any linear-EWMA
statistic; v0.4.x reused them linearly, placing ``alpha`` / ``ss_res`` /
``ss_total`` / ``r2`` at an effective span of ~``2 * span``. These tests
pin the v0.5.0 convention: all in-sample diagnostics live in the
nominal-span EWMA norm.
"""

import numpy as np
import pandas as pd
import pytest

from factorlasso import LassoModel, LassoModelType
from factorlasso.ewm_utils import compute_expanding_power
from factorlasso.lasso_estimator import _compute_solver_weights, get_x_y_np

SPAN = 36.0
LAM = 1.0 - 2.0 / (SPAN + 1.0)


def _toy(seed: int = 7, t: int = 240, n: int = 6, m: int = 3):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(
        rng.standard_normal((t, m)), columns=[f"F{j}" for j in range(m)]
    )
    beta = rng.standard_normal((n, m)) * 0.5
    y = pd.DataFrame(
        x.values @ beta.T + 0.5 * rng.standard_normal((t, n)),
        columns=[f"y{k}" for k in range(n)],
    )
    return x, y


def _fit(x, y, span=SPAN, reg_lambda=1e-6):
    return LassoModel(
        model_type=LassoModelType.LASSO, reg_lambda=reg_lambda, span=span
    ).fit(x=x, y=y)


def test_error_norm_identity():
    """The solver loss is the nominal-span EWMA norm: squared solver
    weights reproduce the ``lambda^k`` profile exactly, and at negligible
    regularisation the fitted betas equal closed-form WLS with
    per-observation weights ``lambda^k``."""
    x, y = _toy()
    model = _fit(x, y, reg_lambda=1e-8)
    xd, yd, _ = get_x_y_np(x, y, span=SPAN, demean=True)
    t = xd.shape[0]
    w_nominal = compute_expanding_power(
        n=t, power_lambda=LAM, reverse_columns=True
    )
    w_solver = compute_expanding_power(
        n=t, power_lambda=np.sqrt(LAM), reverse_columns=True
    )
    assert np.allclose(w_solver**2, w_nominal)
    w_mat = xd.T * w_nominal  # (M, T') row-scaled
    beta_wls = np.linalg.solve(w_mat @ xd, w_mat @ yd).T  # (N, M)
    assert np.allclose(model.coef_.values, beta_wls, atol=1e-3)


def test_alpha_matches_nominal_span_linear_ewma():
    """``result.alpha`` equals the finite-window normalized ``lambda^k``
    EWMA mean of the residuals — the ``pandas.ewm(span, adjust=True)``
    convention (NOT the recursive ``adjust=False`` / InitType.X0 form,
    which differs by O(lambda^T) initialisation terms)."""
    x, y = _toy()
    model = _fit(x, y)
    res = model.estimation_result_
    xd, yd, _ = get_x_y_np(x, y, span=SPAN, demean=True)
    eps = yd - xd @ model.coef_.values.T
    t = xd.shape[0]
    w = compute_expanding_power(n=t, power_lambda=LAM, reverse_columns=True)
    alpha_explicit = (w[:, None] * eps).sum(axis=0) / w.sum()
    assert np.allclose(res.alpha, alpha_explicit)
    alpha_pandas = (
        pd.DataFrame(eps).ewm(span=SPAN, adjust=True).mean().iloc[-1].to_numpy()
    )
    assert np.allclose(res.alpha, alpha_pandas, atol=1e-12)
    ssr_explicit = (w[:, None] * eps**2).sum(axis=0) / w.sum()
    assert np.allclose(res.ss_res, ssr_explicit)


def test_diagnostic_ess_equals_span():
    """Kish ESS of the diagnostic weighting equals the nominal span under
    this convention (ESS = (1+lambda)/(1-lambda) = span exactly). The
    pre-v0.5.0 linear reuse of the sqrt-decay weights gives
    (1+sqrt(lambda))/(1-sqrt(lambda)) ~ 2*span — kept as a regression
    guard against reverting the squaring. Note the exact pre-fix
    effective span is ~2*span, not 2*span + 1."""
    t = int(SPAN * 12)
    mask = np.ones((t, 1))
    w = _compute_solver_weights(t, 1, SPAN, mask)[:, 0]

    def ess(v: np.ndarray) -> float:
        v = v / v.sum()
        return 1.0 / float(np.sum(v**2))

    assert ess(w**2) == pytest.approx(SPAN, rel=1e-3)
    mu = np.sqrt(LAM)
    assert ess(w) == pytest.approx((1 + mu) / (1 - mu), rel=1e-3)
    assert ess(w) == pytest.approx(2 * SPAN, rel=0.02)


def test_snapshot_delta_v050():
    """Frozen fixture documenting the numerical change in ``alpha`` and
    ``r2`` at v0.5.0 (referenced from the CHANGELOG). The v0.4.x values
    were computed at an effective span of ~2*span; the v0.5.0 values are
    the nominal-span statistics. Same seed, data, and fit configuration
    in both rows."""
    x, y = _toy(seed=7)
    model = _fit(x, y, span=36.0, reg_lambda=1e-6)
    res = model.estimation_result_

    alpha_v04x = np.array(
        [0.0271332127, 0.0104356391, -0.0073828531,
         -0.0076664880, 0.0196123940, -0.0315505113]
    )
    r2_v04x = np.array(
        [0.8679699968, 0.7887380972, 0.8816780295,
         0.6954165590, 0.7170607735, 0.6701467553]
    )
    alpha_v050 = np.array(
        [0.0197832870, 0.0018608743, -0.0215583890,
         -0.0227264702, 0.0343118111, -0.0430948538]
    )
    r2_v050 = np.array(
        [0.8949652205, 0.7820504852, 0.8984938714,
         0.7147544367, 0.7209072291, 0.6502924471]
    )

    assert np.allclose(res.alpha, alpha_v050, atol=1e-7)
    assert np.allclose(res.r2, r2_v050, atol=1e-7)
    assert not np.allclose(res.alpha, alpha_v04x, atol=1e-4)
    assert not np.allclose(res.r2, r2_v04x, atol=1e-4)


def test_span_none_unaffected():
    """Uniform weights are idempotent under squaring: ``span=None`` fits
    produce identical diagnostics before and after the v0.5.0 fix, and
    ``alpha`` is ~0 by the OLS first-order condition on demeaned data."""
    x, y = _toy()
    model = _fit(x, y, span=None, reg_lambda=1e-8)
    res = model.estimation_result_
    assert np.allclose(res.alpha, 0.0, atol=1e-6)
