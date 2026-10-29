"""Full-study research adapters: common SSE/T loss and explicit chronological tuning.

Core FactorLasso defaults are untouched. Ridge/OLS use the public ridge mode;
HCGL/FCGL objectives and gates retain Stage 1B conventions. The convex path
solver reuses the existing sparse-group proximal maps on cached Gram matrices.
Cooperative Lasso uses positive/negative cluster-factor group norms (Chiquet,
Grandvalet and Charbonnier, 2012, doi:10.1214/11-AOAS520), adapted to response
groups. UniLasso reuses the exact intercepted leave-one-out feature construction
of Chatterjee, Hastie and Tibshirani (2025), doi:10.1162/99608f92.c79ff6db;
its second-stage features are centered, never standardized. No published
inferential guarantee is claimed for learned groups or weighted adaptations.
"""

from __future__ import annotations

from dataclasses import replace
import time

import numpy as np
from scipy.optimize import nnls

if __package__:
    from . import matf_comparators as M
    from . import french_estimators as F
    from .matf_pilot_run import measured_fit
else:
    import matf_comparators as M
    import french_estimators as F
    from matf_pilot_run import measured_fit


def methods(cell, diagnostic_cells):
    """Full deployable procedures plus explicitly separate bounded diagnostics."""
    rows = [
        dict(name=x, family=x, gate="none", eta=0.0, eta0=0.0, span=None)
        for x in ("ols", "ridge", "lasso", "cooperative", "unilasso")
    ]
    for family in ("hcgl", "fcgl"):
        for gate in ("ungated", "pooled_release", "pooled_exclude", "soft_direction"):
            rows.append(
                dict(
                    name=family + "_" + gate,
                    family=family,
                    gate=gate,
                    eta=1.0 if gate == "soft_direction" else 0.0,
                    eta0=0.0,
                    span=None,
                )
            )
    for gate in ("conditional_release", "conditional_exclude"):
        rows.append(
            dict(name="hcgl_" + gate, family="hcgl", gate=gate, eta=0.0, eta0=0.0, span=None)
        )
    if cell in diagnostic_cells:
        for gate, eta, eta0, name in (
            ("singleton_release", 0.0, 0.0, "singleton_release"),
            ("singleton_exclude", 0.0, 0.0, "singleton_exclude"),
            ("pooled_zero_only", 0.0, 0.0, "pooled_zero_only"),
            ("soft_direction", 0.1, 0.0, "soft_eta01"),
            ("soft_direction", 10.0, 0.0, "soft_eta10"),
            ("soft_direction", 1.0, 1.0, "soft_exclusion"),
        ):
            rows.append(
                dict(name="hcgl_" + name, family="hcgl", gate=gate, eta=eta, eta0=eta0, span=None)
            )
        for row in list(rows[:]):
            if row["name"] in (
                "ridge",
                "hcgl_ungated",
                "hcgl_pooled_release",
                "hcgl_pooled_exclude",
                "hcgl_soft_direction",
            ):
                rows.append(dict(row, name=row["name"] + "_ew120", span=120))
    return rows


class Window:
    """Training-local transforms; partitions, gates and features are cached within it only."""

    def __init__(self, x, y, span, cut=0.6, threshold=2.0):
        self.data = M.prepare(x, y, span)
        self.cut, self.threshold = cut, threshold
        self.labels = self.data.weighted.labels(cut)
        self.specs, self.gates = {}, {}
        xw, yw = self.data.weighted.x, self.data.weighted.y
        self.gram = 2 * xw.T @ xw / len(xw)
        self.cross = 2 * yw.T @ xw / len(xw)
        self.constant = float(np.sum(yw**2) / len(xw))
        self.eigenmax = float(np.linalg.eigvalsh(self.gram)[-1])
        self.uni = None

    def spec(self, family):
        if family not in self.specs:
            if family in ("hcgl", "fcgl"):
                spec = M.specification(self.data, family, self.cut)
            else:
                weights = np.array(
                    [
                        np.sqrt(np.sum(self.labels == g) / len(np.unique(self.labels)))
                        for g in self.labels
                    ]
                )
                spec = F.Spec(
                    family, self.labels.copy(), weights, np.full_like(self.cross, np.nan), 0.0
                )
                spec.scale = F.kkt_scale(self.data.weighted, spec)
            self.specs[family] = spec
        return self.specs[family]

    def gate(self, kind):
        if kind not in self.gates:
            labels = np.arange(len(self.labels)) if kind == "singleton" else self.labels
            self.gates[kind] = M.gate(
                self.data,
                labels,
                "conditional" if kind == "conditional" else "marginal",
                self.threshold,
            )
        return self.gates[kind]

    def restrictions(self, method):
        hard, soft = np.full_like(self.cross, np.nan), np.full_like(self.cross, np.nan)
        gate = method["gate"]
        if gate not in ("none", "ungated"):
            kind = (
                "conditional"
                if gate.startswith("conditional")
                else "singleton"
                if gate.startswith("singleton")
                else "marginal"
            )
            signs = self.gate(kind)["signs"]
            if gate == "soft_direction":
                soft = signs.copy()
            else:
                action = (
                    "zero_only"
                    if gate.endswith("zero_only")
                    else ("release" if gate.endswith("release") else "exclude")
                )
                hard = M.hard_signs(signs, action)
        return hard, soft

    def univariate(self):
        if self.uni is None:
            if self.data.span is not None:
                raise ValueError("Published UniLasso reference is equal-weight only in this study")
            eta, mean, slopes, constants = self.data.state.unilasso()
            self.uni = dict(
                eta=eta,
                mean=mean,
                slopes=slopes,
                constants=constants,
                gram=2 * np.einsum("nqi,nqj->qij", eta, eta) / len(eta),
                cross=2 * np.einsum("nqp,nq->qp", eta, self.data.state.y) / len(eta),
            )
        return self.uni


def gradient(window, beta, soft, eta, eta0):
    grad = beta @ window.gram - window.cross
    if eta or eta0:
        grad += M.soft_loss_gradient(beta, soft, eta, eta0)[1]
    return grad


def audit(window, method, beta, lam, hard, soft):
    spec = replace(window.spec(method["family"]), signs=hard)
    grad = gradient(window, beta, soft, method["eta"], method["eta0"])
    lip = max(1.0, window.eigenmax + 2 * max(method["eta"], method["eta0"]))
    prox = F.proximal(beta - grad / lip, spec, lam / lip)
    residual = float(lip * abs(beta - prox).max() / (1 + abs(beta).max() + abs(grad).max()))
    added = M.soft_loss_gradient(beta, soft, method["eta"], method["eta0"])[0]
    objective = (
        window.constant
        + 0.5 * np.sum((beta @ window.gram) * beta)
        - np.sum(window.cross * beta)
        + lam * F.penalty(beta, spec)
        + added
    )
    violation = float(abs(beta - F.cone_project(beta, hard)).max())
    return dict(kkt=residual, objective=float(objective), violation=violation)


def sparse_fit(window, method, lam, initial=None):
    """Cached-Gram FISTA on the unchanged objective; strict stationarity, no silent truncation."""
    start = time.perf_counter()
    hard, soft = window.restrictions(method)
    spec = replace(window.spec(method["family"]), signs=hard)
    current = np.zeros_like(hard) if initial is None else F.cone_project(initial, hard)
    lip = max(1.0, window.eigenmax + 2 * max(method["eta"], method["eta0"]))
    # Exact KKT-zero endpoint is common to all gate arms with nonnegative soft terms.
    zero = np.zeros_like(current)
    if audit(window, method, zero, lam, hard, soft)["kkt"] <= 1e-12:
        current, iteration = zero, 0
    elif lam == 0 and np.isnan(hard).all() and method["eta"] == method["eta0"] == 0:
        current = np.linalg.solve(window.gram, window.cross.T).T
        iteration = 0
    else:
        accelerated, momentum = current.copy(), 1.0
        for iteration in range(1, 20001):
            grad = gradient(window, accelerated, soft, method["eta"], method["eta0"])
            updated = F.proximal(accelerated - grad / lip, spec, lam / lip)
            if iteration % 5 == 0:
                check_grad = gradient(window, updated, soft, method["eta"], method["eta0"])
                prox = F.proximal(updated - check_grad / lip, spec, lam / lip)
                residual = (
                    lip
                    * abs(updated - prox).max()
                    / (1 + abs(updated).max() + abs(check_grad).max())
                )
                if residual <= 2e-8:
                    current = updated
                    break
            next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
            accelerated = updated + (momentum - 1) / next_momentum * (updated - current)
            current, momentum = updated, next_momentum
        else:
            raise RuntimeError("Full-study FISTA did not converge; no approximate fit accepted")
    diagnostics = audit(window, method, current, lam, hard, soft)
    if diagnostics["kkt"] > 1e-7 or diagnostics["violation"] > 1e-9:
        raise RuntimeError("Full-study sparse stationarity/feasibility failed")
    return dict(
        beta=current,
        alpha=np.zeros(len(current)),
        parameter=current,
        diagnostics=dict(
            status="verified_fista",
            iterations=iteration,
            seconds=time.perf_counter() - start,
            **diagnostics,
        ),
    )


def uni_audit(window, theta, lam):
    u = window.univariate()
    grad = np.einsum("qij,qj->qi", u["gram"], theta) - u["cross"] + lam
    violation = float(np.maximum(-theta, 0).max())
    residual = np.where(theta > 1e-10, abs(grad), np.maximum(-grad, 0))
    kkt = float(np.max(residual / (1 + abs(u["cross"]))))
    objective = (
        window.constant
        + 0.5 * np.einsum("qi,qij,qj->", theta, u["gram"], theta)
        - np.sum(u["cross"] * theta)
        + lam * theta.sum()
    )
    return dict(kkt=kkt, violation=violation, objective=float(objective))


def uni_fit(window, lam, initial=None):
    """Batched positive coordinate descent on centered, unscaled LOO predictions."""
    start = time.perf_counter()
    u = window.univariate()
    theta = np.zeros_like(u["cross"]) if initial is None else np.maximum(initial.copy(), 0)
    if lam == 0:
        theta = np.stack(
            [
                nnls(u["eta"][:, i], window.data.state.y[:, i], maxiter=10000)[0]
                for i in range(len(theta))
            ]
        )
        iteration = 0
    elif np.max(u["cross"]) <= lam:
        theta.fill(0.0)
        iteration = 0
    else:
        for iteration in range(1, 20001):
            for j in range(theta.shape[1]):
                denominator = u["gram"][:, j, j]
                residual = u["cross"][:, j] - np.sum(u["gram"][:, j, :] * theta, axis=1)
                theta[:, j] = np.maximum(theta[:, j] + (residual - lam) / denominator, 0)
            if iteration % 5 == 0 and uni_audit(window, theta, lam)["kkt"] <= 2e-8:
                break
        else:
            raise RuntimeError("UniLasso coordinate descent did not converge")
    diagnostics = uni_audit(window, theta, lam)
    if diagnostics["kkt"] > 1e-7:
        raise RuntimeError("UniLasso KKT failed")
    beta = theta * u["slopes"]
    alpha = np.sum(theta * (u["constants"] - u["mean"]), axis=1)
    return dict(
        beta=beta,
        alpha=alpha,
        parameter=theta,
        diagnostics=dict(
            status="verified_unilasso",
            iterations=iteration,
            seconds=time.perf_counter() - start,
            **diagnostics,
        ),
    )


def path(window, method, grid):
    """Fit each grid value once, using adjacent solutions only as numerical starts."""
    family = method["family"]
    results = {}
    initial = None
    scale = 1.0 if family in ("ridge", "ols") else window.spec(family).scale
    for candidate in sorted(grid, key=lambda v: np.inf if v == "inf" else v, reverse=True):
        lam = np.inf if candidate == "inf" else candidate * scale
        if family in ("ridge", "ols"):
            if np.isinf(lam):
                started = time.perf_counter()
                fitted = M.ridge_fit(window.data, lam)
                if np.any(fitted["beta"] != 0):
                    raise RuntimeError("Infinite ridge endpoint is not zero")
                fitted.update(
                    kkt_residual=0.0,
                    elapsed_seconds=time.perf_counter() - started,
                    outer_warnings=[],
                )
            else:
                fitted = measured_fit(window.data, None, lam)
            diagnostics = dict(
                status="package_ridge",
                iterations=0,
                kkt=fitted["kkt_residual"],
                violation=0.0,
                seconds=fitted["elapsed_seconds"],
                warnings=fitted["outer_warnings"],
            )
            result = dict(
                beta=fitted["beta"],
                alpha=fitted["alpha"],
                parameter=fitted["beta"],
                diagnostics=diagnostics,
            )
        elif family == "unilasso":
            result = uni_fit(window, lam, initial)
        else:
            result = sparse_fit(window, method, lam, initial)
        initial = result["parameter"]
        result["lambda"] = "inf" if np.isinf(lam) else float(lam)
        results[candidate] = result
    return results


def select(records, grid):
    """Complete three-fold loss, with the same larger-penalty tie rule for every method."""
    scores = {}
    for value in grid:
        found = [r for r in records if r["candidate"] == value]
        if len(found) != 3 or len({r["stop"] for r in found}) != 3:
            raise ValueError("Missing or duplicate tuning fold")
        scores[value] = float(np.mean([r["mse"] for r in found]))
    if not np.isfinite(list(scores.values())).all():
        raise ValueError("Nonfinite tuning score")
    best = min(scores.values())
    selected = max(
        (v for v in grid if scores[v] <= best + 1e-12), key=lambda v: np.inf if v == "inf" else v
    )
    return selected, scores
