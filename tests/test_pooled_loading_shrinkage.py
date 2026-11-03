"""Independent references and failure contracts for pooled joint-loading shrinkage."""

from dataclasses import FrozenInstanceError, replace

import cvxpy as cvx
import numpy as np
import pandas as pd
import pytest

import factorlasso.linear_model._pooled_shrinkage as implementation
from factorlasso.linear_model._pooled_shrinkage import fit_pooled_loading_shrinkage


def _panel(t=63, m=3, n=5):
    """Unequal response groups, correlated factors and nonzero intercepts."""
    rng = np.random.default_rng(2026110102)
    xx = rng.normal(size=(t, m))
    if m > 1:
        xx[:, 1] += 0.65 * xx[:, 0]
    xx += np.linspace(-0.4, 0.8, m)
    beta = rng.normal(size=(n, m))
    yy = xx @ beta.T + np.linspace(-0.7, 0.9, n)
    yy += rng.normal(scale=0.17, size=(t, n))
    index = pd.date_range("2010-01-01", periods=t, freq="D", name="date")
    x = pd.DataFrame(xx, index=index, columns=[f"factor_{i}" for i in range(m)])
    y = pd.DataFrame(yy, index=index, columns=[f"asset_{i}" for i in range(n)])
    labels = [1, "1", 1, "last", "last"][:n] if n <= 5 else list(range(n))
    groups = pd.Series(labels, index=y.columns, dtype=object, name="groups")
    weights = pd.Series(np.exp(np.linspace(-3.0, 0.0, t)), index=index, name="date_weight")
    return x, y, groups, weights


def _reference(x, y, groups, weights, penalty):
    """Intercepted augmented least squares, without the producer's centering/solver."""
    xx, yy = np.asarray(x), np.asarray(y)
    t, m = xx.shape
    w = np.ones(t) if weights is None else np.asarray(weights) / np.mean(weights)
    root = np.sqrt(w / t)
    design = np.column_stack((np.ones(t), xx))
    individual = np.linalg.lstsq(root[:, None] * design, root[:, None] * yy,
                                 rcond=None)[0]
    target = np.empty((yy.shape[1], m))
    for label in dict.fromkeys(groups):
        members = [i for i, value in enumerate(groups) if value == label]
        target[members] = np.mean(individual[1:, members], axis=1)
    if np.isposinf(penalty):
        beta = target.copy()
        alpha = np.average(yy - xx @ beta.T, axis=0, weights=w)
    else:
        ridge_rows = np.column_stack((np.zeros(m), np.sqrt(penalty) * np.eye(m)))
        augmented_x = np.vstack((root[:, None] * design, ridge_rows))
        augmented_y = np.vstack((root[:, None] * yy, np.sqrt(penalty) * target.T))
        solution = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
        alpha, beta = solution[0], solution[1:].T
    return beta, alpha, target, individual[1:].T


@pytest.mark.parametrize("penalty", [0.0, 0.03, 0.7, 8.0, np.inf])
@pytest.mark.parametrize("shape", [(63, 3, 5), (4, 3, 2), (21, 2, 1)])
def test_joint_target_fit_matches_intercepted_augmented_least_squares(penalty, shape):
    """General fits, the zero-df target boundary and one response use the exact objective."""
    x, y, groups, weights = _panel(*shape)
    fitted = fit_pooled_loading_shrinkage(x, y, groups, penalty, weights=weights)
    expected = _reference(x, y, groups, weights, penalty)
    for actual, desired in zip((fitted.coefficients, fitted.intercept,
                                fitted.pooled_target, fitted.individual_wls), expected):
        np.testing.assert_allclose(actual, desired, rtol=2e-8, atol=2e-9)
    assert fitted.design_rank == x.shape[1]
    assert fitted.coefficients.index.equals(y.columns)
    assert fitted.coefficients.columns.equals(x.columns)
    assert fitted.date_weights.index.equals(x.index)
    assert fitted.groups.equals(groups)
    assert fitted.date_weights.mean() == pytest.approx(1.0)
    if penalty == 0:
        pd.testing.assert_frame_equal(fitted.coefficients, fitted.individual_wls)
    if np.isinf(penalty):
        pd.testing.assert_frame_equal(fitted.coefficients, fitted.pooled_target)
        assert fitted.penalty == fitted.target_violation == 0.0
        assert fitted.objective is fitted.stationarity_residual is None
    else:
        assert fitted.stationarity_residual <= 1e-7


@pytest.mark.parametrize("penalty", [0.0, 0.42, np.inf])
def test_original_coordinate_diagnostics_and_stationarity(penalty):
    """Reconstruct all reported scalars independently, not from solver residual diagnostics."""
    x, y, groups, weights = _panel()
    fitted = fit_pooled_loading_shrinkage(x, y, groups, penalty, weights=weights)
    w = np.asarray(weights) / weights.mean()
    beta, target = np.asarray(fitted.coefficients), np.asarray(fitted.pooled_target)
    residual = np.asarray(y) - np.asarray(x) @ beta.T - np.asarray(fitted.intercept)
    sse = float(np.sum(w[:, None] * residual**2))
    distance = beta - target
    np.testing.assert_allclose(fitted.weighted_sse, sse, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fitted.penalty, np.sum(distance**2), atol=1e-14)
    assert fitted.target_violation == np.max(np.abs(distance))
    np.testing.assert_allclose(fitted.target_intercept,
                               fitted.y_mean - target @ np.asarray(fitted.x_mean))
    if np.isfinite(penalty):
        assert fitted.objective == pytest.approx(sse / len(x) + penalty * np.sum(distance**2))
        xc = np.asarray(x) - np.average(x, weights=w, axis=0)
        yc = np.asarray(y) - np.average(y, weights=w, axis=0)
        h = xc.T @ (w[:, None] * xc) / len(x)
        c = yc.T @ (w[:, None] * xc) / len(x)
        scale = max(1.0, penalty)
        first, second, third = beta @ h / scale, c / scale, penalty * distance / scale
        independent = np.max(np.abs(first - second + third)) / max(
            1, np.max(np.abs(first)), np.max(np.abs(second)), np.max(np.abs(third)))
        assert fitted.stationarity_residual == pytest.approx(independent, abs=2e-15)


def test_separate_convex_program_with_free_intercept():
    """A direct, unshifted convex objective corroborates the augmented-LS reference."""
    x, y, groups, weights = _panel()
    penalty = 0.36
    expected_beta, expected_alpha, target, _ = _reference(x, y, groups, weights, penalty)
    beta = cvx.Variable((y.shape[1], x.shape[1]))
    alpha = cvx.Variable(y.shape[1])
    w = np.asarray(weights) / weights.mean()
    residual = np.asarray(y) - np.asarray(x) @ beta.T - alpha
    problem = cvx.Problem(cvx.Minimize(
        cvx.sum_squares(cvx.multiply(np.sqrt(w[:, None]), residual)) / len(x)
        + penalty * cvx.sum_squares(beta - target)))
    problem.solve(solver="CLARABEL")
    fitted = fit_pooled_loading_shrinkage(x, y, groups, penalty, weights=weights)
    np.testing.assert_allclose(fitted.coefficients, beta.value, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(fitted.intercept, alpha.value, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(beta.value, expected_beta, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(alpha.value, expected_alpha, rtol=2e-8, atol=2e-9)
    assert fitted.objective == pytest.approx(problem.value, rel=2e-10, abs=1e-11)


@pytest.mark.parametrize("weight_scale", [1e-250, 1e250])
def test_common_weight_rescaling_and_equal_weights(weight_scale):
    """Only relative weights matter; normalization avoids overflowing a raw weight sum."""
    x, y, groups, weights = _panel()
    first = fit_pooled_loading_shrinkage(x, y, groups, 0.2, weights=weights)
    second = fit_pooled_loading_shrinkage(x, y, groups, 0.2, weights=weights * weight_scale)
    np.testing.assert_allclose(first.coefficients, second.coefficients, atol=2e-12)
    np.testing.assert_allclose(first.intercept, second.intercept, atol=2e-12)
    equal = fit_pooled_loading_shrinkage(x, y, groups, 0.2)
    explicit = fit_pooled_loading_shrinkage(x, y, groups, 0.2,
                                           weights=pd.Series(7.0, index=x.index))
    pd.testing.assert_frame_equal(equal.coefficients, explicit.coefficients)


def test_targets_depend_on_groups_and_do_not_impose_signs():
    """Full joint targets distinguish mixed-type groups and singleton pooling."""
    x, y, groups, weights = _panel()
    pooled = fit_pooled_loading_shrinkage(x, y, groups, np.inf, weights=weights)
    assert not np.allclose(pooled.pooled_target.iloc[0], pooled.pooled_target.iloc[1])
    np.testing.assert_allclose(pooled.pooled_target.iloc[0], pooled.pooled_target.iloc[2])
    singletons = pd.Series(range(y.shape[1]), index=y.columns)
    singleton = fit_pooled_loading_shrinkage(x, y, singletons, 100.0, weights=weights)
    np.testing.assert_allclose(singleton.coefficients, singleton.individual_wls, atol=1e-10)
    all_grouped = pd.Series("all", index=y.columns)
    full = fit_pooled_loading_shrinkage(x, y, all_grouped, np.inf, weights=weights)
    np.testing.assert_allclose(full.coefficients,
                               np.repeat(full.individual_wls.mean().to_numpy()[None, :],
                                         len(y.columns), axis=0), atol=1e-12)
    # The independent optimum can cross its target direction; no target-sign cone is imposed.
    changed = y.copy()
    changed.iloc[:, 2] = -5 * y.iloc[:, 0]
    free = fit_pooled_loading_shrinkage(x, changed, groups, 0.01, weights=weights)
    assert (np.asarray(free.coefficients) * np.asarray(free.pooled_target) < 0).any()


def test_original_unit_recovery_and_coordinate_dependence():
    """External response scaling changes group targets; raw prediction recovery is explicit."""
    x, y, groups, weights = _panel()
    w = np.asarray(weights)
    xm, ym = np.average(x, weights=w, axis=0), np.average(y, weights=w, axis=0)
    xs = np.sqrt(np.average((np.asarray(x) - xm)**2, weights=w, axis=0))
    ys = np.array([0.3, 1.2, 2.0, 0.8, 1.7])
    xstar, ystar = (x - xm) / xs, (y - ym) / ys
    fitted = fit_pooled_loading_shrinkage(xstar, ystar, groups, 0.5, weights=weights)
    raw_beta = ys[:, None] * np.asarray(fitted.coefficients) / xs[None, :]
    raw_alpha = ym + ys * np.asarray(fitted.intercept) - raw_beta @ xm
    forecast = np.array([[0.4, -0.6, 0.2], [-0.1, 0.9, 0.3]])
    left = forecast @ raw_beta.T + raw_alpha
    right = (((forecast - xm) / xs) @ np.asarray(fitted.coefficients).T
             + np.asarray(fitted.intercept)) * ys + ym
    np.testing.assert_allclose(left, right, atol=2e-15)
    unscaled = fit_pooled_loading_shrinkage(x, y, groups, 0.5, weights=weights)
    assert np.max(np.abs(raw_beta - np.asarray(unscaled.coefficients))) > 0.05
    assert np.max(np.abs(ys[:, None] * np.asarray(fitted.pooled_target)
                         / xs[None, :] - np.asarray(unscaled.pooled_target))) > 0.05


def test_shifted_levels_recover_intercepts_and_preserve_slopes():
    """An unpenalized intercept removes large offsets without changing the target."""
    x, y, groups, weights = _panel()
    base = fit_pooled_loading_shrinkage(x, y, groups, 0.6, weights=weights)
    shifted = fit_pooled_loading_shrinkage(x + 1e6, y - 2e6, groups, 0.6, weights=weights)
    np.testing.assert_allclose(shifted.coefficients, base.coefficients, atol=3e-10)
    np.testing.assert_allclose(shifted.intercept,
                               base.intercept - 2e6 - shifted.coefficients.sum(axis=1) * 1e6,
                               atol=5e-4)


def test_original_date_subset_weights_are_not_restarted():
    """An irregular subset retains original ages; restarting its decay changes the fit."""
    x, y, groups, weights = _panel()
    chosen = np.r_[0:15:3, 20:45:2, 48:63]
    xx, yy, ww = x.iloc[chosen], y.iloc[chosen], weights.iloc[chosen]
    fitted = fit_pooled_loading_shrinkage(xx, yy, groups, 0.4, weights=ww)
    reference = _reference(xx, yy, groups, ww, 0.4)
    np.testing.assert_allclose(fitted.coefficients, reference[0], atol=2e-9)
    restarted = pd.Series(np.exp(np.arange(len(xx)) * 3 / (len(x) - 1)), index=xx.index)
    wrong = fit_pooled_loading_shrinkage(xx, yy, groups, 0.4, weights=restarted)
    assert np.max(np.abs(fitted.coefficients - wrong.coefficients)) > 1e-3


def test_no_input_or_result_data_aliasing():
    """Endpoint tables and copied metadata are independently writable owned outputs."""
    x, y, groups, weights = _panel()
    copies = [value.copy(deep=True) for value in (x, y, groups, weights)]
    fitted = fit_pooled_loading_shrinkage(x, y, groups, np.inf, weights=weights)
    for actual, before in zip((x, y, groups, weights), copies):
        if isinstance(actual, pd.DataFrame):
            pd.testing.assert_frame_equal(actual, before)
        else:
            pd.testing.assert_series_equal(actual, before)
    target = fitted.pooled_target.copy()
    fitted.coefficients.iloc[0, 0] += 100
    pd.testing.assert_frame_equal(fitted.pooled_target, target)
    fitted.groups.iloc[0] = "mutated"
    fitted.date_weights.iloc[0] = 999
    pd.testing.assert_series_equal(groups, copies[2])
    pd.testing.assert_series_equal(weights, copies[3])
    with pytest.raises(FrozenInstanceError):
        fitted.reg_lambda = 1.0


@pytest.mark.parametrize("penalty", [True, np.bool_(False), -1.0, -np.inf, np.nan,
                                     "inf", 1j, 10**400])
def test_invalid_penalties_are_rejected(penalty):
    """Penalty semantics exclude booleans, strings and invalid real values."""
    x, y, groups, weights = _panel()
    with pytest.raises(ValueError, match="reg_lambda"):
        fit_pooled_loading_shrinkage(x, y, groups, penalty, weights=weights)


@pytest.mark.parametrize("penalty", [0.0, 0.2, np.inf])
@pytest.mark.parametrize("defect", ["constant", "collinear", "too_short"])
def test_target_rank_is_required_at_every_penalty(penalty, defect):
    """Positive shrinkage cannot identify an otherwise undefined joint WLS group target."""
    x, y, groups, weights = _panel()
    if defect == "constant":
        x.iloc[:, 0] = 1.0
    elif defect == "collinear":
        x.iloc[:, 1] = 2 * x.iloc[:, 0]
    else:
        x, y, weights = x.iloc[:3], y.iloc[:3], weights.iloc[:3]
    with pytest.raises(ValueError, match="rank|T >= M"):
        fit_pooled_loading_shrinkage(x, y, groups, penalty, weights=weights)


@pytest.mark.parametrize("defect", ["nan", "infinity", "complex", "bool", "row_order",
                                  "duplicate_rows", "duplicate_columns", "group_order",
                                  "missing_group", "zero_weight", "weight_order"])
def test_strict_input_policy_is_shared_with_the_gate(defect):
    """Reject invalid panels before the more permissive underlying ridge machinery."""
    x, y, groups, weights = _panel()
    if defect == "nan":
        y.iloc[0, 0] = np.nan
    elif defect == "infinity":
        x.iloc[0, 0] = np.inf
    elif defect == "complex":
        x = x.astype(complex)
    elif defect == "bool":
        x = x > 0
    elif defect == "row_order":
        y = y.iloc[::-1]
    elif defect == "duplicate_rows":
        x.index = y.index = pd.Index([0] * len(x))
    elif defect == "duplicate_columns":
        x.columns = ["same"] * x.shape[1]
    elif defect == "group_order":
        groups = groups.iloc[::-1]
    elif defect == "missing_group":
        groups.iloc[0] = None
    elif defect == "zero_weight":
        weights.iloc[0] = 0.0
    else:
        weights = weights.iloc[::-1]
    with pytest.raises(ValueError):
        fit_pooled_loading_shrinkage(x, y, groups, 0.4, weights=weights)


@pytest.mark.parametrize("argument", ["x", "y", "groups", "weights"])
def test_labeled_container_requirement(argument):
    """No ambiguous array/Series versus frame return branches are added."""
    x, y, groups, weights = _panel()
    inputs = dict(x=x, y=y, groups=groups, weights=weights, reg_lambda=0.4)
    inputs[argument] = np.asarray(inputs[argument])
    with pytest.raises(TypeError):
        fit_pooled_loading_shrinkage(**inputs)


def test_solver_reuse_receives_same_penalty_and_target_offset(monkeypatch):
    """Spy on the reused engine to prevent silent changes in penalty normalization."""
    x, y, groups, weights = _panel()
    real_solver = implementation.solve_ridge_cvx_problem
    calls = []

    def spy(**kwargs):
        calls.append({key: value.copy() if isinstance(value, np.ndarray) else value
                      for key, value in kwargs.items()})
        return real_solver(**kwargs)

    monkeypatch.setattr(implementation, "solve_ridge_cvx_problem", spy)
    fitted = fit_pooled_loading_shrinkage(x, y, groups, 0.27, weights=weights)
    assert [call["reg_lambda"] for call in calls] == [0.0, 0.27]
    assert all(call["span"] is None and call["loss_normalization"] == "sample" for call in calls)
    np.testing.assert_allclose(calls[1]["y"],
                               calls[0]["y"] - calls[0]["x"] @ fitted.pooled_target.to_numpy().T,
                               atol=2e-15)
    calls.clear()
    fit_pooled_loading_shrinkage(x, y, groups, np.inf, weights=weights)
    assert [call["reg_lambda"] for call in calls] == [0.0]


@pytest.mark.parametrize("defect", ["nan", "poor_wls", "poor_finite", "wrong_lambda"])
def test_defective_solver_results_fail_independent_certification(monkeypatch, defect):
    """Injected numerical defects are detected even when a solver returns a result object."""
    x, y, groups, weights = _panel()
    real_solver = implementation.solve_ridge_cvx_problem

    def defective(**kwargs):
        penalty = kwargs["reg_lambda"]
        if defect == "wrong_lambda" and penalty > 0:
            kwargs["reg_lambda"] = penalty / len(x)
        result = real_solver(**kwargs)
        beta = result.estimated_beta.copy()
        if defect == "nan":
            beta[0, 0] = np.nan
        elif defect == "poor_wls" and penalty == 0:
            beta[0, 0] += 0.1
        elif defect == "poor_finite" and penalty > 0:
            beta[0, 0] += 0.1
        return replace(result, estimated_beta=beta)

    monkeypatch.setattr(implementation, "solve_ridge_cvx_problem", defective)
    with pytest.raises(RuntimeError, match="invalid coefficients|stationarity"):
        fit_pooled_loading_shrinkage(x, y, groups, 0.4, weights=weights)


def test_mutation_references_distinguish_wrong_targets_and_penalty_norms():
    """The independent numerical assertions fail for plausible, runnable wrong estimators."""
    x, y, groups, weights = _panel()
    penalty = 0.4
    fitted = fit_pooled_loading_shrinkage(x, y, groups, penalty, weights=weights)
    expected, _, target, _ = _reference(x, y, groups, weights, penalty)
    w = np.asarray(weights) / weights.mean()
    xc, yc = np.asarray(x) - fitted.x_mean.to_numpy(), np.asarray(y) - fitted.y_mean.to_numpy()
    h = xc.T @ (w[:, None] * xc) / len(x)
    c = xc.T @ (w[:, None] * yc) / len(x)
    wrong_zero = np.linalg.solve(h + penalty * np.eye(x.shape[1]), c).T
    wrong_scale = _reference(x, y, groups, weights, penalty / len(x))[0]
    beta = cvx.Variable(target.shape)
    loss = cvx.sum_squares(cvx.multiply(np.sqrt(w[:, None]), yc - xc @ beta.T)) / len(x)
    cvx.Problem(cvx.Minimize(loss + penalty * cvx.norm1(beta - target))).solve(solver="CLARABEL")
    for mutated in [wrong_zero, wrong_scale, beta.value]:
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(mutated, expected, rtol=2e-8, atol=2e-9)


def test_solver_failure_and_unrepresentable_geometry_are_explicit(monkeypatch):
    """No solver or arithmetic failure becomes a neutral target or partial return."""
    x, y, groups, weights = _panel()

    def fail(**kwargs):
        raise cvx.error.SolverError("injected backend failure")

    with monkeypatch.context() as context:
        context.setattr(implementation, "solve_ridge_cvx_problem", fail)
        with pytest.raises(RuntimeError, match="solver failed"):
            fit_pooled_loading_shrinkage(x, y, groups, 0.4, weights=weights)
    with pytest.raises(RuntimeError, match="arithmetic|finite"):
        fit_pooled_loading_shrinkage(x * 1e200, y, groups, 0.4, weights=weights)


@pytest.mark.parametrize("settings", [{"solver": None}, {"solver": ""},
                                     {"solver_fallbacks": "CLARABEL"},
                                     {"solver_fallbacks": [None]}])
def test_invalid_solver_settings_are_rejected(settings):
    """Validate the retry-chain structure without changing legacy solver behavior."""
    x, y, groups, weights = _panel()
    with pytest.raises(ValueError, match="solver"):
        fit_pooled_loading_shrinkage(x, y, groups, 0.4, weights=weights, **settings)


def test_explicit_solver_fallback_preserves_the_target_fit():
    """Reuse the existing opt-in retry chain, while a failed primary alone raises."""
    x, y, groups, weights = _panel()
    with pytest.raises(RuntimeError, match="solver failed"):
        fit_pooled_loading_shrinkage(x, y, groups, 0.4, weights=weights,
                                     solver="UNAVAILABLE_TEST_SOLVER")
    recovered = fit_pooled_loading_shrinkage(
        x, y, groups, 0.4, weights=weights, solver="UNAVAILABLE_TEST_SOLVER",
        solver_fallbacks=["CLARABEL"])
    expected = _reference(x, y, groups, weights, 0.4)[0]
    np.testing.assert_allclose(recovered.coefficients, expected, atol=2e-9)


@pytest.mark.skipif(np.finfo(np.longdouble).max <= np.finfo(float).max,
                    reason="longdouble has no extended exponent range on this platform")
def test_finite_extended_precision_penalty_cannot_become_infinite_endpoint():
    """A finite penalty outside float64 range is invalid, not exact complete pooling."""
    x, y, groups, weights = _panel()
    penalty = np.longdouble(np.finfo(float).max) * 2
    assert np.isfinite(penalty)
    with pytest.raises(ValueError, match="representable"):
        fit_pooled_loading_shrinkage(x, y, groups, penalty, weights=weights)
