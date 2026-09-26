"""Research adapters for matched prior-centred L1 regression.

The exported factorlasso.solve_lasso_cvx_problem is the reference estimator.
Its objective is mean squared error plus lambda times the L1 deviation.
For repeated research paths, sklearn.lasso_path solves the same objective at
alpha=lambda/2. Numerical equivalence is checked before simulations.
No production estimator or default is modified.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import lasso_path

from factorlasso import solve_lasso_cvx_problem
from factorlasso.beta_priors import _compute_ols_prior


def prepare(x, y):
    """Centre and scale using the supplied training observations only."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    xm, ym = x.mean(axis=0), y.mean()
    xs, ys = x.std(axis=0), y.std()
    if np.any(xs <= 0) or ys <= 0:
        raise ValueError("Degenerate training data")
    return (x - xm) / xs, (y - ym) / ys, xm, ym, xs, ys


def automatic_target(x, y):
    """Use the package's actual selector on original training observations."""
    slopes, r2, prior = _compute_ols_prior(x, y[:, None], span=None)
    return prior[0], int(np.nanargmax(r2[0])), r2[0]


def path_standardised(x, y, lambdas, prior=None, unpenalised=None):
    """Solve the centred standardised objective, optionally freeing one slope."""
    n, p = x.shape
    target = np.zeros(p) if prior is None else np.asarray(prior)
    coefficients = np.zeros((len(lambdas), p))
    if unpenalised is None:
        design, response = x, y - x @ target
        remaining = np.arange(p)
    else:
        remaining = np.delete(np.arange(p), unpenalised)
        anchor = x[:, unpenalised]
        norm = anchor @ anchor
        # Package reference supports penalty_weights with this entry set to zero.
        # Residualising both design and response solves that same joint problem.
        design = x[:, remaining] - np.outer(anchor, anchor @ x[:, remaining] / norm)
        response = y - anchor * (anchor @ y / norm)
    positive = np.flatnonzero(lambdas > 0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, beta, gaps = lasso_path(
            np.asfortranarray(design), response,
            alphas=lambdas[positive] / 2, tol=1e-11, max_iter=100000,
        )
    if caught:
        raise RuntimeError("; ".join(str(w.message) for w in caught))
    if not np.isfinite(beta).all():
        raise ValueError("Nonfinite path")
    coefficients[np.ix_(positive, remaining)] = beta.T
    if unpenalised is None:
        coefficients += target
    else:
        coefficients[:, unpenalised] = (
            anchor @ y - (anchor @ x[:, remaining]) @ coefficients[:, remaining].T
        ) / norm
    for row in np.flatnonzero(lambdas == 0):
        coefficients[row] = np.linalg.lstsq(x, y, rcond=None)[0]
    return coefficients, float(np.max(np.abs(gaps)))


def fit_paths(x, y, lambdas, external_targets=None, include_relaxed=True):
    """Return native-unit paths and provenance for all comparable methods."""
    z, v, xm, ym, xs, ys = prepare(x, y)
    automatic, winner, r2 = automatic_target(np.asarray(x), np.asarray(y))
    targets = {"M1_zero": np.zeros(z.shape[1]), "M2_auto": automatic}
    if external_targets:
        targets.update(external_targets)
    paths = {}
    for method, native in targets.items():
        b, gap = path_standardised(z, v, lambdas, native * xs / ys)
        paths[method] = {
            "beta": b * ys / xs, "target": native, "gap": gap,
            "winner": winner, "winner_r2": float(r2[winner]),
        }
    b, gap = path_standardised(z, v, lambdas, unpenalised=winner)
    paths["M4_free_winner"] = {
        "beta": b * ys / xs, "target": np.zeros(z.shape[1]), "gap": gap,
        "winner": winner, "winner_r2": float(r2[winner]),
    }
    ols = np.linalg.lstsq(z, v, rcond=None)[0] * ys / xs
    paths["M0_ols"] = {
        "beta": np.repeat(ols[None], len(lambdas), axis=0),
        "target": np.zeros(z.shape[1]), "gap": 0., "winner": winner,
        "winner_r2": float(r2[winner]),
    }
    if include_relaxed:
        baseline = paths["M1_zero"]["beta"] * xs / ys
        relaxed = np.zeros_like(baseline)
        for k, row in enumerate(baseline):
            support = np.flatnonzero(np.abs(row) > 1e-7)
            if len(support):
                relaxed[k, support] = np.linalg.lstsq(z[:, support], v, rcond=None)[0]
        paths["M5_post_lasso"] = {
            "beta": relaxed * ys / xs, "target": np.zeros(z.shape[1]),
            "gap": 0., "winner": winner, "winner_r2": float(r2[winner]),
        }
    for values in paths.values():
        values["intercept"] = ym - values["beta"] @ xm
    return paths


def prediction_losses(x, y, betas, intercepts):
    """Evaluate every path using independent sample sufficient statistics."""
    augmented = np.column_stack([np.ones(len(x)), x])
    coefficients = np.column_stack([intercepts, betas])
    gram = augmented.T @ augmented / len(x)
    cross = augmented.T @ y / len(x)
    return np.maximum(
        np.einsum("ij,jk,ik->i", coefficients, gram, coefficients)
        - 2 * coefficients @ cross + y @ y / len(y), 0.,
    )


def select_index(losses):
    """Choose minimum validation loss; ties favour the descending stronger lambda."""
    tolerance = max(1e-14, abs(float(np.min(losses))) * 1e-12)
    return int(np.flatnonzero(losses <= np.min(losses) + tolerance)[0])


def paired_summary(frame, baseline="M1_zero"):
    """Compute paired seed differences and Monte Carlo confidence intervals."""
    metrics = ["secondary_mse", "scenario_mse", "prediction_mse", "beta_mse"]
    keys = ["rho", "secondary", "n", "seed", "snr"]
    control = frame[frame.method == baseline].set_index(keys)
    output = []
    for method, group in frame.groupby("method"):
        group = group.set_index(keys)
        for cell, part in group.groupby(level=["rho", "secondary", "n", "snr"]):
            base = control.loc[part.index]
            record = dict(zip(["rho", "secondary", "n", "snr"], cell))
            record.update(method=method, seeds=len(part))
            for metric in metrics:
                values = part[metric].to_numpy()
                differences = values - base[metric].to_numpy()
                se = float(differences.std(ddof=1) / np.sqrt(len(part)))
                record.update({
                    metric: float(values.mean()),
                    metric + "_delta": float(differences.mean()),
                    metric + "_delta_se": se,
                    metric + "_delta_low": float(differences.mean() - 1.96 * se),
                    metric + "_delta_high": float(differences.mean() + 1.96 * se),
                })
            record["secondary_zero_rate"] = float(part.secondary_zero.mean())
            record["winner_credit_rate"] = float((part.winner == 1).mean())
            output.append(record)
    return pd.DataFrame(output)


def reference_checks(lambdas):
    """Check fast paths against the exported package solver and negative controls."""
    rng = np.random.default_rng(908001)
    x = rng.normal(size=(120, 3))
    x[:, 1] = .8 * x[:, 0] + .6 * x[:, 1]
    y = x @ np.array([1., .3, -.2]) + rng.normal(size=120)
    z, v, _, _, _, _ = prepare(x, y)
    target = np.array([1., .2, 0.])
    sample = np.array([.4, .05, .005, 0.])
    errors = []
    start = time.perf_counter()
    for prior, free in [(np.zeros(3), None), (target, None), (None, 0)]:
        fast, _ = path_standardised(z, v, sample, prior=prior, unpenalised=free)
        weights = np.ones((1, 3))
        if free is not None:
            weights[0, free] = 0.
        for i, lam in enumerate(sample):
            reference = solve_lasso_cvx_problem(
                z, v[:, None], reg_lambda=float(lam),
                factors_beta_prior=None if prior is None else prior[None],
                penalty_weights=weights, solver="CLARABEL",
            ).estimated_beta[0]
            errors.append(float(np.max(np.abs(reference - fast[i]))))
    maximum = max(errors)
    if maximum > 3e-4:
        raise AssertionError(f"Package path mismatch {maximum}")
    # Deliberately omit the factor of two: this must fail the same comparison.
    good, _ = path_standardised(z, v, np.array([.4]), target)
    broken, _ = path_standardised(z, v, np.array([.8]), target)
    assert np.max(np.abs(good - broken)) > 1e-3
    base = fit_paths(x, y, lambdas)
    units = np.array([3., .2, 7.])
    scaled = fit_paths(x * units, y, lambdas)
    scale_error = max(float(np.max(np.abs(
        base[m]["beta"] - scaled[m]["beta"] * units
    ))) for m in base)
    assert scale_error < 1e-7
    test = rng.normal(size=(31, 3))
    response = rng.normal(size=31)
    coefficients = base["M2_auto"]
    direct = np.mean((response[:, None] - (
        test @ coefficients["beta"].T + coefficients["intercept"]
    )) ** 2, axis=0)
    reduced = prediction_losses(test, response, coefficients["beta"],
                                coefficients["intercept"])
    np.testing.assert_allclose(direct, reduced, atol=1e-12, rtol=1e-12)
    return {
        "package_reference_max_abs_beta_error": maximum,
        "package_reference_tolerance": 3e-4,
        "unit_rescaling_max_abs_error": scale_error,
        "deliberate_lambda_normalisation_defect_rejected": True,
        "direct_prediction_scoring_verified": True,
        "seconds": time.perf_counter() - start,
    }

