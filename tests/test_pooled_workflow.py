"""Independent synthetic checks of explicit pooled-estimator composition.

These test caller-owned preparation, not a new CV or forecasting API. The
proximal reference does not call the package solver or the paper implementation.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import fcluster

from factorlasso import (
    DistanceTransform, LassoModel, LassoModelType, compute_clusters_from_corr_matrix,
    compute_pooled_sign_gate, fit_pooled_loading_shrinkage,
)
from factorlasso.linear_model._solvers import group_lasso as group_solver


def panel():
    """Unequal groups and nonzero levels in explicitly scaled synthetic units."""
    rng = np.random.default_rng(2026110107)
    x = rng.normal(size=(79, 3))
    x[:, 1] += 0.45 * x[:, 0]
    beta = np.array([[1, -.4, 0], [.6, -.3, 0], [-.7, .8, 0],
                     [.2, 0, .9], [.3, 0, .5]])
    y = (x @ beta.T + .35 * rng.normal(size=(79, 5))) * [1, 2, .4, 3, .7]
    y += [2, -3, 1, 4, -.5]
    x = x * [.4, 2, 3] + [3, -2, 1]
    dates = pd.date_range('2000-01-01', periods=79)
    return (pd.DataFrame(x, index=dates, columns=list('abc')),
            pd.DataFrame(y, index=dates, columns=list('vwxyz')),
            pd.Series(np.exp(np.linspace(-2, 0, 79)), index=dates),
            pd.Series(['g', 'g', 'h', 'i', 'i'], index=list('vwxyz')))


def prepare(x, y, weights):
    """Independent fixed-window weighted centering and unit conversion."""
    w = weights.to_numpy() / weights.mean()
    xm, ym = np.average(x, axis=0, weights=w), np.average(y, axis=0, weights=w)
    xc, yc = x.to_numpy() - xm, y.to_numpy() - ym
    xs = np.sqrt(np.average(xc**2, axis=0, weights=w))
    ys = np.sqrt(np.average(yc**2, axis=0, weights=w))
    return (pd.DataFrame(xc / xs, index=x.index, columns=x.columns),
            pd.DataFrame(yc / ys, index=y.index, columns=y.columns), w, xm, ym, xs, ys)


def prox(values, step, lam, q, restrictions):
    """Exact row sparse-group proximal map intersected with coordinate half-lines."""
    z = np.sign(values) * np.maximum(np.abs(values) - step * lam / 2, 0)
    z[restrictions == 0] = 0
    z[(restrictions == 1) & (z < 0)] = 0
    z[(restrictions == -1) & (z > 0)] = 0
    norms = np.linalg.norm(z, axis=1)
    shrink = np.zeros_like(norms)
    nonzero = norms > 0
    shrink[nonzero] = np.maximum(1 - step * lam * q[nonzero] / (2 * norms[nonzero]), 0)
    return shrink[:, None] * z


def reference(xw, yw, lam, groups, restrictions):
    """FISTA from zero, with an independently measured proximal residual."""
    q = np.array([np.sqrt(np.sum(groups == g) / groups.nunique()) for g in groups])
    step = len(xw) / (2 * np.linalg.norm(xw, 2)**2)
    b = np.zeros((yw.shape[1], xw.shape[1]))
    z, momentum = b.copy(), 1.

    def gradient(v):
        return 2 * (xw @ v.T - yw).T @ xw / len(xw)

    for _ in range(4000):
        new = prox(z - step * gradient(z), step, lam, q, restrictions)
        new_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        z = new + (momentum - 1) / new_momentum * (new - b)
        b, momentum = new, new_momentum
        if np.max(np.abs(b - prox(b - step * gradient(b), step, lam, q,
                                 restrictions))) / step < 1e-10:
            break
    else:
        raise AssertionError('Independent FISTA did not converge')
    return b, q, step, gradient


def fit_bridge(xa, ya, w, groups, restrictions, lam):
    """Use the existing public estimator with all preparation explicit."""
    model = LassoModel(
        model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
        reg_lambda=lam, demean=False, span=None, warmup_period=None,
        auto_sign_constraints=False, apply_ols_prior=False,
        auto_sign_adaptive_weights=False, nonneg=False,
        group_penalty='normalized', l1_weight=.5, loss_normalization='sample',
        factors_beta_loading_signs=restrictions,
    )
    return model.fit(xa.mul(np.sqrt(w), axis=0), ya.mul(np.sqrt(w), axis=0),
                     external_clusters=groups)


@pytest.mark.parametrize('action', ['exclude', 'release', 'zero_only'])
def test_gate_to_weighted_hcgl_matches_independent_proximal_reference(action):
    x, y, weights, groups = panel()
    xa, ya, w, xm, ym, xs, ys = prepare(x, y, weights)
    gate = compute_pooled_sign_gate(xa, ya, groups, weights=weights, action=action)
    assert np.any(gate.signs.to_numpy() == 0) and np.any(gate.signs.to_numpy() != 0)
    lam = .3
    restrictions = gate.restrictions.to_numpy()
    xw, yw = xa.to_numpy() * np.sqrt(w[:, None]), ya.to_numpy() * np.sqrt(w[:, None])
    expected, q, step, gradient = reference(xw, yw, lam, groups, restrictions)
    model = fit_bridge(xa, ya, w, groups, gate.restrictions, lam)
    actual = model.coef_.to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-5)

    def objective(b):
        return (np.sum((xw @ b.T - yw)**2) / len(xw)
                + lam / 2 * (np.abs(b).sum() + np.sum(q * np.linalg.norm(b, axis=1))))

    assert abs(objective(actual) - objective(expected)) < 2e-8
    assert np.max(np.abs(actual[restrictions == 0]), initial=0) < 2e-7
    assert np.min((actual * restrictions)[np.abs(restrictions) == 1], initial=0) > -2e-7
    residual = np.max(np.abs(actual - prox(actual - step * gradient(actual), step,
                                          lam, q, restrictions))) / step
    # This fixture bound is not the paper's stricter universal acceptance claim.
    assert residual < 2e-5
    raw_beta = ys[:, None] * actual / xs[None, :]
    raw_alpha = ym - raw_beta @ xm
    np.testing.assert_allclose(x.to_numpy() @ raw_beta.T + raw_alpha,
                               (xa.to_numpy() @ actual.T) * ys + ym, atol=2e-14)
    pd.testing.assert_series_equal(model.clusters_, groups, check_names=False)


def test_independent_hcgl_reference_detects_penalty_scaling_defect(monkeypatch):
    x, y, weights, groups = panel()
    xa, ya, w, *_ = prepare(x, y, weights)
    gate = compute_pooled_sign_gate(xa, ya, groups, weights=weights, action='release')
    xw, yw = xa.to_numpy() * np.sqrt(w[:, None]), ya.to_numpy() * np.sqrt(w[:, None])
    expected, *_ = reference(xw, yw, .3, groups, gate.restrictions.to_numpy())
    original = group_solver._build_group_lasso_problem

    def defective(x, y, group_loadings, reg_lambda, **kwargs):
        return original(x, y, group_loadings, reg_lambda / len(x), **kwargs)

    monkeypatch.setattr(group_solver, '_build_group_lasso_problem', defective)
    wrong = fit_bridge(xa, ya, w, groups, gate.restrictions, .3).coef_
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(wrong, expected, rtol=0, atol=2e-5)


def prefix_state(x, y, weights, stop):
    """Training-local transforms, Ward recut, evidence and target for one candidate."""
    x, y, weights = x.iloc[:stop], y.iloc[:stop], weights.iloc[:stop]
    xa, ya, w, *_ = prepare(x, y, weights)
    corr = (ya.to_numpy() * w[:, None]).T @ ya.to_numpy() / len(ya)
    corr = pd.DataFrame(corr, index=y.columns, columns=y.columns)
    _, linkage, _ = compute_clusters_from_corr_matrix(
        corr, linkage_method='ward', distance_transform=DistanceTransform.CHORD)
    groups = pd.Series(fcluster(linkage, .6 * linkage[-1, 2], criterion='distance'),
                       index=y.columns)
    gate = compute_pooled_sign_gate(xa, ya, groups, weights=weights)
    fitted = fit_pooled_loading_shrinkage(xa, ya, groups, .2, weights=weights)
    return xa, ya, groups, gate, fitted


def test_prefix_isolation_covers_preparation_groups_evidence_and_target():
    x, y, weights, _ = panel()
    a = prefix_state(x, y, weights, 40)
    changed_x, changed_y = x.copy(), y.copy()
    changed_x.iloc[40:] += 100
    changed_y.iloc[40:] -= 200
    b = prefix_state(changed_x, changed_y, weights, 40)
    for left, right in zip(a[:2], b[:2]):
        pd.testing.assert_frame_equal(left, right)
    pd.testing.assert_series_equal(a[2], b[2])
    pd.testing.assert_frame_equal(a[3].slopes, b[3].slopes)
    pd.testing.assert_frame_equal(a[3].restrictions, b[3].restrictions)
    pd.testing.assert_frame_equal(a[4].pooled_target, b[4].pooled_target)
    pd.testing.assert_frame_equal(a[4].coefficients, b[4].coefficients)
    # Validation outcomes legitimately change scores even though the fixed fit is identical.
    _, _, _, xm, ym, xs, ys = prepare(x.iloc[:40], y.iloc[:40], weights.iloc[:40])

    def validation_loss(factors, responses):
        predicted = ((factors.iloc[40:50].to_numpy() - xm) / xs
                     @ a[4].coefficients.to_numpy().T + a[4].intercept.to_numpy())
        observed = (responses.iloc[40:50].to_numpy() - ym) / ys
        return np.mean((observed - predicted)**2)

    assert not np.isclose(validation_loss(x, y), validation_loss(changed_x, changed_y))
    # Data beyond a complete decision cutoff must also be ignored by its refit.
    changed_x, changed_y = x.copy(), y.copy()
    changed_x.iloc[60:] *= -99
    changed_y.iloc[60:] *= 200
    a, b = prefix_state(x, y, weights, 60), prefix_state(changed_x, changed_y, weights, 60)
    pd.testing.assert_frame_equal(a[4].coefficients, b[4].coefficients)
    pd.testing.assert_frame_equal(a[3].restrictions, b[3].restrictions)
