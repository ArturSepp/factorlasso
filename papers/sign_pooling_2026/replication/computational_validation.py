"""Dimension scaling and independent numerical checks for the revised paper.

Each timed regularization path runs in a fresh process. Peak working set includes
the interpreter and numerical libraries; timing includes training preprocessing,
CVXPY canonicalization and all path solves, but excludes process startup/imports.
This measures a seven-point path, not a nested validation run. The accuracy audit
rebuilds the selected objective independently and compares a tighter CLARABEL
solution with the benchmark's MOSEK solution. No package defaults are changed.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import cvxpy as cp
import numpy as np
import pandas as pd

if __package__:
    from . import estimator_benchmark as bench
    from .paths import output_directory
else:
    import estimator_benchmark as bench
    from paths import output_directory


SIZES = ((80, 12, 12), (120, 60, 24), (160, 120, 48), (160, 240, 64))
METHODS = ("hcgl_unconstrained", "hcgl_pooled_zero", "hcgl_pooled_free",
           "fcgl_unconstrained", "cooperative")


def peak_working_set_mb():
    """Windows process peak resident working set, including imported libraries."""
    if os.name != "nt":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters),
                                      counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.PeakWorkingSetSize / 2**20


def worker(n, p, q, seed, method):
    scenario = bench.Scenario("scaling", train_size=n, test_size=5, predictors=p,
                              cluster_sizes=tuple([4]*(q//4)), residual_correlation=.6)
    x, y, _, _, _, _ = bench.simulate(scenario, seed)
    protocol = bench.Protocol(cutoff_fraction=.7)
    start = time.perf_counter()
    prep = bench.prepare(x, y, protocol.cutoff_fraction)
    prepared_at = time.perf_counter()
    betas, _, _, signs = bench.fit_path(prep, method, 2., protocol)
    end = time.perf_counter()
    if not np.isfinite(betas).all():
        raise AssertionError("Nonfinite coefficients")
    violation = max(float(np.max(np.abs(betas[:, signs == 0]), initial=0)),
                    float(np.max(-betas[:, signs == 1], initial=0)),
                    float(np.max(betas[:, signs == -1], initial=0)))
    return dict(n=n, p=p, q=q, seed=seed, method=method,
                groups=len(np.unique(prep.labels)), path_points=len(betas),
                preprocessing_seconds=prepared_at-start, path_seconds=end-prepared_at,
                total_seconds=end-start, peak_working_set_mb=peak_working_set_mb(),
                max_constraint_violation=violation, success=True)


def explicit_objective(prep, method, beta, alpha, lam):
    """Direct NumPy objective; no CVXPY expression or benchmark objective reused."""
    groups = [np.flatnonzero(prep.labels == g) for g in np.unique(prep.labels)]
    penalty = 0.
    for group in groups:
        weight = np.sqrt(len(group)/len(groups))
        if method == "cooperative":
            block = beta[group]
            penalty += weight * (np.linalg.norm(np.maximum(block, 0), axis=0).sum()
                                 + np.linalg.norm(np.minimum(block, 0), axis=0).sum())
        elif method == "fcgl_unconstrained":
            penalty += (1-alpha)*weight*np.linalg.norm(beta[group], axis=0).sum()
        else:
            penalty += (1-alpha)*weight*np.linalg.norm(beta[group], axis=1).sum()
    if method != "cooperative":
        penalty += alpha*np.abs(beta).sum()
    return float(np.sum((prep.y-prep.x@beta.T)**2)/len(prep.x) + lam*penalty)


def independent_fit(prep, method, signs, alpha, lam, tolerance=1e-9,
                    return_diagnostics=False):
    """Scalar row/block construction with an independent, tightly solved backend."""
    q, p = prep.y.shape[1], prep.x.shape[1]
    beta = cp.Variable((q, p))
    groups = [np.flatnonzero(prep.labels == g) for g in np.unique(prep.labels)]
    terms = []
    for group in groups:
        weight = np.sqrt(len(group)/len(groups))
        if method == "cooperative":
            for j in range(p):
                terms.append(weight*(cp.norm(cp.pos(beta[group, j]), 2)
                                     + cp.norm(cp.neg(beta[group, j]), 2)))
        elif method == "fcgl_unconstrained":
            terms.extend((1-alpha)*weight*cp.norm(beta[group, j], 2) for j in range(p))
        else:
            terms.extend((1-alpha)*weight*cp.norm(beta[k, :], 2) for k in group)
    if method != "cooperative":
        terms.append(alpha*cp.norm1(cp.vec(beta, order="C")))
    constraints = []
    for k in range(q):
        for j in range(p):
            s = signs[k, j]
            if s == 0:
                constraints.append(beta[k, j] == 0)
            elif np.isfinite(s):
                constraints.append(s*beta[k, j] >= 0)
    loss = sum(cp.sum_squares(prep.y[:, k]-prep.x@beta[k, :])
               for k in range(q))/len(prep.x)
    problem = cp.Problem(cp.Minimize(loss+lam*sum(terms)), constraints)
    problem.solve(solver="CLARABEL", tol_gap_abs=tolerance, tol_gap_rel=tolerance,
                  tol_feas=tolerance, max_iter=500)
    if problem.status != cp.OPTIMAL:
        raise RuntimeError(f"Independent solve returned {problem.status}")
    if return_diagnostics:
        diagnostics = dict(reference_status=problem.status,
                           reference_tolerance=tolerance,
                           reference_iterations=problem.solver_stats.num_iters)
        solver = problem._solver_cache.get("CLARABEL")
        if solver is not None and hasattr(solver, "get_info"):
            info = solver.get_info()
            for name in ("res_primal", "res_dual", "gap_abs", "gap_rel"):
                diagnostics["reference_conic_"+name] = float(getattr(info, name))
        return beta.value, diagnostics
    return beta.value


def constraint_violation(beta, signs):
    """Largest violation of the actual coefficient restrictions."""
    return max(float(np.max(np.abs(beta[signs == 0]), initial=0)),
               float(np.max(-beta[signs == 1], initial=0)),
               float(np.max(beta[signs == -1], initial=0)))


def proximal_kkt_residual(prep, method, beta, signs, alpha, lam):
    """Proximal stationarity residual in gradient units, independent of solvers.

    The sign cone projection, L1 threshold and disjoint group shrinkage give
    the penalty-plus-constraint proximal map. Cooperative groups shrink their
    positive and negative parts separately. A zero residual is the KKT condition.
    """
    gram = prep.x.T @ prep.x / len(prep.x)
    lipschitz = 2*np.linalg.eigvalsh(gram)[-1]
    gradient = 2*(prep.x.T @ (prep.x@beta.T-prep.y)).T / len(prep.x)
    z = beta-gradient/lipschitz
    z[signs == 0] = 0
    z[signs == 1] = np.maximum(z[signs == 1], 0)
    z[signs == -1] = np.minimum(z[signs == -1], 0)
    groups = [np.flatnonzero(prep.labels == g) for g in np.unique(prep.labels)]
    if method != "cooperative":
        z = np.sign(z)*np.maximum(abs(z)-lam*alpha/lipschitz, 0)
    for group in groups:
        weight = np.sqrt(len(group)/len(groups))
        if method == "cooperative":
            pos, neg = np.maximum(z[group], 0), np.minimum(z[group], 0)
            for block in (pos, neg):
                norm = np.linalg.norm(block, axis=0)
                block *= np.maximum(1-lam*weight/lipschitz/np.maximum(norm, 1e-300), 0)
            z[group] = pos+neg
        elif method == "fcgl_unconstrained":
            norm = np.linalg.norm(z[group], axis=0)
            z[group] *= np.maximum(1-lam*(1-alpha)*weight/lipschitz/
                                  np.maximum(norm, 1e-300), 0)
        else:
            norm = np.linalg.norm(z[group], axis=1)
            z[group] *= np.maximum(1-lam*(1-alpha)*weight/lipschitz/
                                  np.maximum(norm, 1e-300), 0)[:, None]
    return float(lipschitz*np.max(abs(beta-z)))


def accuracy_audit(reference_tolerance=1e-9):
    rows = []
    for seed in range(3):
        x, y, _, _, _, _ = bench.simulate(
            bench.Scenario("accuracy", residual_correlation=.6), seed)
        prep = bench.prepare(x, y, .7)
        protocol = bench.Protocol(cutoff_fraction=.7)
        for method in METHODS:
            betas, _, lambdas, signs = bench.fit_path(prep, method, 2., protocol,
                                                      fractions=(.3, .03, .003))
            for fraction, beta, lam in zip((.3, .03, .003), betas, lambdas):
                reference, diagnostics = independent_fit(
                    prep, method, signs, protocol.alpha, lam,
                    tolerance=reference_tolerance, return_diagnostics=True)
                objective = explicit_objective(prep, method, beta, protocol.alpha, lam)
                reference_obj = explicit_objective(prep, method, reference, protocol.alpha, lam)
                if not np.isfinite([objective, reference_obj]).all():
                    raise AssertionError("Nonfinite independent objective comparison")
                discrepancy = abs(objective-reference_obj)/max(1., abs(reference_obj))
                if discrepancy > 2e-5:
                    raise AssertionError(f"Objective discrepancy {discrepancy}")
                for tol in (1e-8, 1e-6, 1e-4):
                    a, b = np.abs(beta) > tol, np.abs(reference) > tol
                    rows.append(dict(seed=seed, method=method, lambda_value=lam,
                                     lambda_fraction=fraction, primary_status="optimal",
                                     **diagnostics,
                                     zero_tolerance=tol, objective_relative_difference=discrepancy,
                                     primary_objective=objective, reference_objective=reference_obj,
                                     coefficient_max_difference=float(np.max(abs(beta-reference))),
                                     primary_constraint_violation=constraint_violation(beta, signs),
                                     reference_constraint_violation=constraint_violation(
                                         reference, signs),
                                     primary_proximal_kkt=proximal_kkt_residual(
                                         prep, method, beta, signs, protocol.alpha, lam),
                                     reference_proximal_kkt=proximal_kkt_residual(
                                         prep, method, reference, signs, protocol.alpha, lam),
                                     selection_disagreement=float(np.mean(a != b)),
                                     sign_disagreement=float(np.mean(
                                         np.where(a, np.sign(beta), 0)
                                         != np.where(b, np.sign(reference), 0)))))
    return pd.DataFrame(rows)


def run_accuracy_only(out: Path, reference_tolerance=1e-9):
    """Run the 45 small checks without executing any dimension-scaling path."""
    os.environ["FACTORLASSO_PAPER_OUTPUT_DIR"] = str(out.resolve())
    out = output_directory()
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(mode="accuracy_only", seeds=list(range(3)), methods=METHODS,
                    fractions=(.3, .03, .003), cut=.7, alpha=.5, tau=2.,
                    primary_solver="MOSEK", reference_solver="CLARABEL",
                    reference_tolerance=reference_tolerance, reference_max_iter=500,
                    objective_relative_limit=2e-5, zero_tolerances=(1e-8, 1e-6, 1e-4),
                    acceptance="Both solvers OPTIMAL; finite objective comparison <=2e-5. "
                               "Constraint and proximal KKT residuals are reported separately.")
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2))
    data = accuracy_audit(reference_tolerance)
    data.to_csv(out/"accuracy_audit.csv", index=False)
    unique_cases = data.drop_duplicates(["seed", "method", "lambda_fraction"])
    receipt = dict(status="complete", protocol=protocol, small_cases=len(data)//3,
                   reference_status_counts=unique_cases.reference_status.value_counts().to_dict(),
                   python=sys.version, platform=platform.platform(),
                   dependencies={name: importlib.metadata.version(name) for name in
                                 ("numpy", "pandas", "scipy", "cvxpy", "Mosek", "clarabel")},
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   benchmark_sha256=hashlib.sha256(Path(bench.__file__).read_bytes()).hexdigest(),
                   output_sha256={"accuracy_audit.csv": hashlib.sha256(
                       (out/"accuracy_audit.csv").read_bytes()).hexdigest()})
    (out/"receipt.json").write_text(json.dumps(receipt, indent=2))


def run(out: Path, repeats=3, reference_tolerance=1e-9):
    os.environ["FACTORLASSO_PAPER_OUTPUT_DIR"] = str(out.resolve())
    out = output_directory()
    out.mkdir(parents=True, exist_ok=False)
    defaults = {}
    import mosek
    with mosek.Task() as task:
        for name in ("intpnt_co_tol_pfeas", "intpnt_co_tol_dfeas", "intpnt_co_tol_rel_gap"):
            defaults[name] = task.getdouparam(getattr(mosek.dparam, name))
    protocol = dict(sizes=SIZES, methods=METHODS, seeds=list(range(repeats)), tau=2.,
                    cut=.7, alpha=.5, fractions=bench.Protocol().lambda_fractions,
                    solver="MOSEK", solver_threads=1, mosek_default_tolerances=defaults,
                    memory="fresh-process peak working set, including interpreter/imports",
                    timing="preprocessing and seven-point path; excludes imports; no CV",
                    accuracy_reference="CLARABEL independent scalar formulation",
                    accuracy_reference_tolerance=reference_tolerance)
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2))
    records = []
    for n, p, q in SIZES:
        for seed in range(repeats):
            for method in METHODS:
                command = [sys.executable, "-B", "-m",
                           "papers.sign_pooling_2026.replication.computational_validation",
                           "--worker", str(n), str(p), str(q), str(seed), method]
                result = subprocess.run(command, capture_output=True, text=True, check=False)
                if result.returncode:
                    records.append(dict(n=n, p=p, q=q, seed=seed, method=method,
                                        success=False, error=result.stderr[-3000:]))
                else:
                    records.append(json.loads(result.stdout.strip().splitlines()[-1]))
                pd.DataFrame(records).to_csv(out/"scaling_draws.csv", index=False)
                print(f"scaling {len(records)}/{len(SIZES)*repeats*len(METHODS)}", flush=True)
    draws = pd.DataFrame(records)
    numeric = ["total_seconds", "path_seconds", "preprocessing_seconds",
               "peak_working_set_mb", "max_constraint_violation"]
    draws.groupby(["n", "p", "q", "method"])[numeric].agg(["mean", "std", "max"]).to_csv(
        out/"scaling_summary.csv")
    accuracy_audit(reference_tolerance).to_csv(out/"accuracy_audit.csv", index=False)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in out.glob("*.csv")}
    receipt = dict(protocol=protocol, python=sys.version, platform=platform.platform(),
                   cpu=platform.processor(), logical_cpus=os.cpu_count(),
                   dependencies={name: importlib.metadata.version(name) for name in
                                 ("numpy", "pandas", "scipy", "scikit-learn", "cvxpy",
                                  "Mosek", "clarabel")},
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   benchmark_sha256=hashlib.sha256(Path(bench.__file__).read_bytes()).hexdigest(),
                   output_sha256=hashes, failed_paths=int((~draws.success).sum()))
    (out/"receipt.json").write_text(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", nargs=5)
    parser.add_argument("--accuracy-only", action="store_true")
    parser.add_argument("--reference-tolerance", type=float, default=1e-9)
    args = parser.parse_args()
    if not 0 < args.reference_tolerance < 1:
        parser.error("--reference-tolerance must be between zero and one")
    if args.worker:
        n, p, q, seed = map(int, args.worker[:4])
        print(json.dumps(worker(n, p, q, seed, args.worker[4])))
    else:
        if args.out is None:
            parser.error("--out is required")
        if args.accuracy_only:
            run_accuracy_only(args.out, args.reference_tolerance)
        else:
            run(args.out, args.repeats, args.reference_tolerance)


if __name__ == "__main__":
    main()
