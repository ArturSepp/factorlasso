"""Prespecified learned-group stress study and nested tuning sensitivity.

This diagnostic deliberately makes response groups recognizable through two
shared anchors, at population R-squared .65. It is not a model of yeast signal
strength. Partial support and mixed directions affect a weaker third predictor;
predictor and residual correlations are changed in separate scenarios. Truth
enters evaluation only. Every candidate is rebuilt inside each training fold.

Full Cartesian candidate grids: Ward cut .35/.5/.7, sparse-group alpha
.25/.5/.75, signed-method tau 0/1.5/3, and the original seven lambda fractions.
The joint-CV fit selects all applicable settings; paired one-factor sensitivity
fits hold the other settings at cut=.5, alpha=.5, tau=1.5 and tune lambda only.
Gamma is zero throughout. The coefficient objectives and solvers are imported
from the independently checked estimator_benchmark implementation.

Shared-threshold gate curves and truth-calibrated, interpolated equal-null
comparisons are diagnostic evaluations, never a practical threshold selector.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import subprocess
import time
import tomllib

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.metrics import adjusted_rand_score

if __package__:
    from . import estimator_benchmark as B
else:
    import estimator_benchmark as B


@dataclass(frozen=True)
class Design:
    name: str
    train_size: int = 80
    test_size: int = 1000
    predictors: int = 12
    r2: float = 0.65
    predictor_rho: float = 0.0
    residual_rho: float = 0.0
    partial: bool = False
    mixed: bool = False


DESIGNS = (
    Design("coherent"),
    Design("mixed_directions", mixed=True),
    Design("partial_support", partial=True),
    Design("predictor_correlation", predictor_rho=0.6),
    Design("residual_correlation", residual_rho=0.6),
)
METHODS = tuple(m for m in B.METHODS if m != "hcgl_known_zero")
CUTS = (0.35, 0.5, 0.7)
ALPHAS = (0.25, 0.5, 0.75)
TAUS = (0.0, 1.5, 3.0)
GATE_TAUS = tuple(np.arange(0.0, 8.01, 0.5))


@dataclass(frozen=True)
class Candidate:
    cut: float = 0.5
    alpha: float = 0.5
    tau: float = 1.5


def simulate(design, seed):
    """Independent population-parameter/train/test streams; no sample-R2 rescaling."""
    parameters, train_rng, test_rng = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
    )
    p, q = design.predictors, 12
    if p < 12:
        raise ValueError("At least 12 predictors are needed for disjoint anchor blocks")
    labels = np.repeat(np.arange(3), 4)
    permutation = parameters.permutation(p)
    supports = permutation[:9].reshape(3, 3)
    spare = permutation[9:]
    beta = np.zeros((q, p))
    for g in range(3):
        direction = parameters.choice([-1.0, 1.0], 3)
        base = np.array([1.0, 1.0, 0.6]) * parameters.uniform(0.9, 1.1, 3)
        for local, k in enumerate(np.flatnonzero(labels == g)):
            active = supports[g].copy()
            if design.partial and local % 2:
                active[-1] = spare[g % len(spare)]
            values = direction * base * parameters.uniform(0.9, 1.1, 3)
            if design.mixed and local % 2:
                values[-1] *= -1
            beta[k, active] = values
    indices = np.arange(p)
    covariance = design.predictor_rho ** np.abs(indices[:, None] - indices[None, :])
    chol = np.linalg.cholesky(covariance)
    signal_var = np.einsum("ij,jk,ik->i", beta, covariance, beta)
    sd = np.sqrt(signal_var * (1 - design.r2) / design.r2)

    def draw(rng, n):
        x = rng.normal(size=(n, p)) @ chol.T
        independent = rng.normal(size=(n, q))
        common = rng.normal(size=(n, 3))
        errors = (
            np.sqrt(1 - design.residual_rho) * independent
            + np.sqrt(design.residual_rho) * common[:, labels]
        ) * sd
        return x, x @ beta.T + errors

    x, y = draw(train_rng, design.train_size)
    xt, yt = draw(test_rng, design.test_size)
    return x, y, xt, yt, beta, labels


def candidates_for(method):
    if method in ("lasso", "unilasso"):
        return (Candidate(),)
    if method == "cooperative":
        return tuple(Candidate(c) for c in CUTS)
    taus = TAUS if method in B.SIGNED_METHODS else (1.5,)
    return tuple(Candidate(c, a, t) for c, a, t in product(CUTS, ALPHAS, taus))


def candidate_protocol(candidate, solver):
    return B.Protocol(
        cutoff_fraction=candidate.cut,
        alpha=candidate.alpha,
        tau_grid=(candidate.tau,),
        solver=solver,
    )


def tune_surfaces(x, y, method, seed, solver="MOSEK"):
    """No truth or external test argument; fold-only preprocessing for every cut."""
    candidates = candidates_for(method)
    fractions = B.lambda_fractions_for(method, B.Protocol())
    errors = np.zeros((len(candidates), len(fractions)))
    for train, valid in B.cv_splits(len(x), 3, seed):
        prepared = {
            cut: B.prepare(x[train], y[train], cut) for cut in sorted({c.cut for c in candidates})
        }
        for i, candidate in enumerate(candidates):
            prep = prepared[candidate.cut]
            betas, intercepts, _, _ = B.fit_path(
                prep, method, candidate.tau, candidate_protocol(candidate, solver)
            )
            prediction = np.einsum("ij,lkj->lik", prep.transform_x(x[valid]), betas)
            prediction += intercepts[:, None, :]
            errors[i] += (
                np.sum((prediction - prep.transform_y(y[valid])[None]) ** 2, axis=(1, 2))
                / y.shape[1]
            )
    return candidates, fractions, errors / len(x)


def select_variants(candidates, fractions, errors):
    """Cartesian joint choice plus paired one-factor fixed-setting sensitivities."""
    selected = {"joint_cv": np.unravel_index(np.argmin(errors), errors.shape)}
    for i, c in enumerate(candidates):
        changes = int(c.cut != 0.5) + int(c.alpha != 0.5) + int(c.tau != 1.5)
        if changes > 1:
            continue
        name = (
            "reference"
            if not changes
            else f"cut_{c.cut:g}"
            if c.cut != 0.5
            else f"alpha_{c.alpha:g}"
            if c.alpha != 0.5
            else f"tau_{c.tau:g}"
        )
        selected[name] = (i, int(np.argmin(errors[i])))
    return {
        name: (candidates[i], fractions[j], float(errors[i, j]))
        for name, (i, j) in selected.items()
    }


def group_metrics(labels, truth_labels, beta):
    """Pairwise grouping and true sign/support heterogeneity, used after fitting."""
    labels, truth_labels = np.asarray(labels), np.asarray(truth_labels)
    first, second = np.triu_indices(len(labels), 1)
    same, true_same = labels[first] == labels[second], truth_labels[first] == truth_labels[second]
    active_pair = (beta[first] != 0) & (beta[second] != 0) & same[:, None]
    conflict = active_pair & (np.sign(beta[first]) != np.sign(beta[second]))
    mixed_support = ((beta[first] != 0) != (beta[second] != 0)) & same[:, None]
    sizes = np.unique(labels, return_counts=True)[1]
    return dict(
        discovered_clusters=len(sizes),
        adjusted_rand=adjusted_rand_score(truth_labels, labels),
        grouping_precision=np.sum(same & true_same) / max(np.sum(same), 1),
        grouping_recall=np.sum(same & true_same) / max(np.sum(true_same), 1),
        singleton_response_fraction=np.sum(sizes == 1) / len(labels),
        largest_cluster=int(max(sizes)),
        mean_cluster_size=float(np.mean(sizes)),
        sign_conflict_pairs=int(np.sum(conflict)),
        active_within_pairs=int(np.sum(active_pair)),
        within_active_sign_conflict=float(np.sum(conflict) / max(np.sum(active_pair), 1)),
        within_support_disagreement=float(
            np.sum(mixed_support) / max(np.sum(same) * beta.shape[1], 1)
        ),
    )


def gate_metrics(signs, beta):
    active, signed = beta != 0, signs != 0
    correct = active & signed & (signs == np.sign(beta))
    wrong = active & signed & (signs != np.sign(beta))
    return dict(
        active_correct=np.sum(correct) / np.sum(active),
        active_wrong=np.sum(wrong) / np.sum(active),
        active_excluded=np.sum(active & ~signed) / np.sum(active),
        null_retention=np.sum(~active & signed) / np.sum(~active),
        signed_fdp=np.sum(~active & signed) / max(np.sum(signed), 1),
        signed_cells=int(np.sum(signed)),
    )


def gate_curves(x, y, beta, cut=0.5):
    prep = B.prepare(x, y, cut)
    rows = []
    for name, labels in (("singleton", np.arange(y.shape[1])), ("pooled", prep.labels)):
        for tau in GATE_TAUS:
            signs, _ = B.derive_gate(prep.x, prep.y, labels, tau)
            rows.append(dict(gate=name, cut=cut, tau=tau, **gate_metrics(signs, beta)))
    return rows


def equal_null_diagnostics(curves, targets=(0.05, 0.1, 0.2, 0.4)):
    """Oracle, within-seed linear interpolation; not a usable calibration rule.

    For repeated null-retention values the smallest threshold is retained.
    No extrapolation is performed; both curves must bracket the target.
    """
    rows = []
    for target in targets:
        values = {}
        for gate in ("singleton", "pooled"):
            part = pd.DataFrame([r for r in curves if r["gate"] == gate])
            x, index = np.unique(part.null_retention.to_numpy(), return_index=True)
            if not x[0] <= target <= x[-1]:
                break
            values[gate] = {
                metric: float(np.interp(target, x, part[metric].to_numpy()[index]))
                for metric in ("active_correct", "active_wrong", "signed_fdp")
            }
        if len(values) == 2:
            row = {"target_null_retention": target}
            for metric in values["pooled"]:
                row[metric + "_pooled"] = values["pooled"][metric]
                row[metric + "_singleton"] = values["singleton"][metric]
                row[metric + "_difference"] = values["pooled"][metric] - values["singleton"][metric]
            rows.append(row)
    return rows


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def fit_one(job):
    design, seed, out, methods, solver = job
    prefix = Path(out) / "shards" / f"{design.name}_{seed:04d}"
    x, y, xt, yt, truth, true_groups = simulate(design, seed)
    rows, surface, arrays = [], [], {}
    start = time.perf_counter()
    for method in methods:
        method_start = time.perf_counter()
        candidates, fractions, errors = tune_surfaces(x, y, method, seed, solver)
        cv_seconds = time.perf_counter() - method_start
        for i, c in enumerate(candidates):
            surface.extend(
                dict(
                    scenario=design.name,
                    seed=seed,
                    method=method,
                    **asdict(c),
                    lambda_fraction=f,
                    cv_mse=errors[i, j],
                )
                for j, f in enumerate(fractions)
            )
        variants = select_variants(candidates, fractions, errors)
        fit_cache = {}
        for variant, (candidate, fraction, error) in variants.items():
            key = (candidate, fraction)
            if key not in fit_cache:
                prep = B.prepare(x, y, candidate.cut)
                fit_start = time.perf_counter()
                betas, intercepts, lambdas, signs = B.fit_path(
                    prep,
                    method,
                    candidate.tau,
                    candidate_protocol(candidate, solver),
                    fractions=[fraction],
                )
                fitted = B.Fitted(
                    betas[0],
                    intercepts[0],
                    prep,
                    float(lambdas[0]),
                    fraction,
                    candidate.tau,
                    error,
                    cv_seconds + time.perf_counter() - fit_start,
                    signs,
                )
                fit_cache[key] = fitted
            fitted = fit_cache[key]
            descriptor = f"{method}__{variant}"
            arrays[descriptor + "__beta"] = fitted.beta
            arrays[descriptor + "__intercept"] = fitted.intercept
            arrays[descriptor + "__labels"] = fitted.prepared.labels
            arrays[descriptor + "__signs"] = fitted.signs
            for tolerance in B.ZERO_TOLERANCES:
                rows.append(
                    dict(
                        scenario=design.name,
                        seed=seed,
                        method=method,
                        variant=variant,
                        zero_tolerance=tolerance,
                        **asdict(candidate),
                        lambda_fraction=fraction,
                        lambda_value=fitted.lambda_value,
                        cv_mse=error,
                        joint_search_seconds=cv_seconds,
                        refit_seconds=fitted.seconds - cv_seconds,
                        lambda_upper_boundary=int(fraction == max(fractions)),
                        lambda_lower_boundary=int(fraction == min(fractions)),
                        **group_metrics(fitted.prepared.labels, true_groups, truth),
                        **B.evaluate(fitted, xt, yt, truth, tolerance),
                    )
                )
        pd.DataFrame(rows).to_csv(str(prefix) + "_draws.csv", index=False)
    curves = gate_curves(x, y, truth)
    for r in curves:
        r.update(scenario=design.name, seed=seed)
    equal = equal_null_diagnostics(curves)
    for r in equal:
        r.update(scenario=design.name, seed=seed)
    pd.DataFrame(curves).to_csv(str(prefix) + "_gates.csv", index=False)
    pd.DataFrame(equal).to_csv(str(prefix) + "_equal_null.csv", index=False)
    pd.DataFrame(surface).to_csv(str(prefix) + "_cv.csv", index=False)
    arrays.update(
        x_mean=x.mean(axis=0),
        x_scale=x.std(axis=0),
        y_mean=y.mean(axis=0),
        y_scale=y.std(axis=0),
        beta_true=truth,
        true_groups=true_groups,
    )
    np.savez_compressed(str(prefix) + "_coefficients.npz", **arrays)
    metadata = dict(
        scenario=design.name,
        seed=seed,
        completed_utc=datetime.now(timezone.utc).isoformat(),
        total_seconds=time.perf_counter() - start,
        inputs={
            k: array_hash(a)
            for k, a in zip(
                ("x_train", "y_train", "x_test", "y_test", "beta_true", "true_groups"),
                (x, y, xt, yt, truth, true_groups),
            )
        },
    )
    Path(str(prefix) + "_receipt.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def summarize(frame, keys):
    numeric = [c for c in frame.select_dtypes(include=np.number) if c not in set(keys) | {"seed"}]
    groups = frame.groupby(keys, sort=False, dropna=False)
    result = pd.concat([groups[numeric].mean().add_suffix("_mean"),
                        groups[numeric].std(ddof=1).add_suffix("_sd"),
                        groups[numeric].sem(ddof=1).add_suffix("_se"),
                        groups.size().rename("replications")], axis=1)
    return result.reset_index()


def summarize_run(out):
    out = Path(out)
    for name, suffix, keys in (
        ("estimator", "draws", ["scenario", "method", "variant", "zero_tolerance"]),
        ("gate", "gates", ["scenario", "gate", "cut", "tau"]),
        ("oracle_equal_null", "equal_null", ["scenario", "target_null_retention"]),
    ):
        frame = pd.concat(
            [pd.read_csv(p) for p in sorted((out / "shards").glob(f"*_{suffix}.csv"))],
            ignore_index=True,
        )
        frame.to_csv(out / f"{name}_draws.csv", index=False)
        summarize(frame, keys).to_csv(out / f"{name}_summary.csv", index=False)
    draws = pd.read_csv(out / "estimator_draws.csv")
    primary = draws[draws.zero_tolerance == B.PRIMARY_ZERO_TOLERANCE]
    paired = []
    metrics = (
        "prediction_mse_standardized",
        "prediction_r2",
        "beta_mse_standardized",
        "active_sign_recovery",
        "active_sign_flip",
        "support_fdp",
        "support_recall",
    )
    # Joint-CV compares feasible procedures; reference compares identical fixed cut/alpha/tau.
    for variant in ("joint_cv", "reference"):
        part = primary[primary.variant == variant]
        for comparator in ("hcgl_unconstrained", "hcgl_per_response", "lasso"):
            base = part[part.method == comparator].set_index(["scenario", "seed"])
            for (scenario, method), group in part.groupby(["scenario", "method"]):
                if base.empty or method == comparator:
                    continue
                for metric in metrics:
                    diff = np.array(
                        [
                            row[metric] - base.loc[(scenario, row.seed), metric]
                            for _, row in group.iterrows()
                        ]
                    )
                    se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.nan
                    half = student_t.ppf(0.975, len(diff) - 1) * se if len(diff) > 1 else np.nan
                    paired.append(
                        dict(
                            scenario=scenario,
                            variant=variant,
                            method=method,
                            comparator=comparator,
                            metric=metric,
                            replications=len(diff),
                            difference_mean=diff.mean(),
                            difference_se=se,
                            ci_low=diff.mean() - half,
                            ci_high=diff.mean() + half,
                        )
                    )
    pd.DataFrame(paired).to_csv(out / "paired_summary.csv", index=False)
    return draws


def protocol_record(designs, methods, reps, first_seed, solver):
    return dict(
        designs=[asdict(d) for d in designs],
        methods=list(methods),
        reps=reps,
        first_seed=first_seed,
        folds=3,
        cuts=CUTS,
        alphas=ALPHAS,
        taus=TAUS,
        lambda_fractions=B.Protocol().lambda_fractions,
        unilasso_fractions=B.lambda_fractions_for("unilasso", B.Protocol()),
        candidates={m: [asdict(c) for c in candidates_for(m)] for m in methods},
        gate_curve_cut=0.5,
        gate_curve_taus=GATE_TAUS,
        zero_tolerances=B.ZERO_TOLERANCES,
        primary_zero_tolerance=B.PRIMARY_ZERO_TOLERANCE,
        solver=solver,
        solver_threads=1,
        adaptation="gamma=0; nonadaptive matched objectives only",
        design_scope="R2 .65 is a recognizable-group diagnostic, not a yeast model",
        tuning="Full Cartesian inner CV; fixed-setting sensitivity tunes lambda only",
        lambda_scale="HCGL/FCGL conservative sufficient zero bound; other scales exact",
        equal_null="Truth-calibrated within-seed linear interpolation; diagnostic only",
        truth_access="Coefficient truth and partition enter evaluation only",
        solver_accuracy=("Inherited solver defaults and OPTIMAL/finite checks; "
                         "common tolerance sensitivity"),
    )


def run(out, reps=40, first_seed=1000, designs=DESIGNS, methods=METHODS, solver="MOSEK", workers=2):
    out = B.safe_output(out)
    if (out / "protocol.json").exists():
        raise FileExistsError("Choose a fresh run directory; protocols and results are immutable")
    (out / "shards").mkdir()
    protocol = protocol_record(designs, methods, reps, first_seed, solver)
    # This file is written before any simulation or solver invocation.
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    sources = [
        Path(__file__),
        Path(B.__file__),
        B.REPOSITORY_DIR / "src/factorlasso/priors/_signs.py",
    ]
    for path in sources:
        (out / (path.stem + "_snapshot.py")).write_bytes(path.read_bytes())
    receipt = dict(
        started_utc=datetime.now(timezone.utc).isoformat(),
        status="running",
        workers=workers,
        python=platform.python_version(),
        platform=platform.platform(),
        processor=platform.processor(),
        source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        protocol_sha256=hashlib.sha256((out / "protocol.json").read_bytes()).hexdigest(),
        repository_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=B.REPOSITORY_DIR, text=True
        ).strip(),
        factorlasso_source=str(Path(B.factorlasso.__file__).resolve()),
        source_pyproject_version=tomllib.loads(
            (B.REPOSITORY_DIR/"pyproject.toml").read_text(encoding="utf-8"))["project"]["version"],
        versions={
            n: B.package_version(n)
            for n in ("factorlasso", "numpy", "scipy", "pandas", "cvxpy", "scikit-learn", "Mosek")
        },
    )
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    jobs = [
        (d, s, str(out), tuple(methods), solver)
        for d in designs
        for s in range(first_seed, first_seed + reps)
    ]
    completed = []
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(fit_one, job): job[:2] for job in jobs}
            for future in as_completed(futures):
                result = future.result()
                completed.append(result)
                print(
                    f"{result['scenario']} seed {result['seed']} complete "
                    f"({len(completed)}/{len(jobs)}; {result['total_seconds']:.1f}s)",
                    flush=True,
                )
        summarize_run(out)
        receipt.update(
            status="complete",
            completed_replications=len(completed),
            completed_utc=datetime.now(timezone.utc).isoformat(),
            outputs={
                str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(out.rglob("*"))
                if p.is_file() and p.name != "receipt.json"
            },
        )
    except Exception as error:
        receipt.update(status="failed", error=repr(error), completed_replications=len(completed))
        raise
    finally:
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=40)
    parser.add_argument("--first-seed", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--solver", choices=("MOSEK", "CLARABEL"), default="MOSEK")
    parser.add_argument("--scenarios", nargs="+", choices=[d.name for d in DESIGNS])
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    args = parser.parse_args()
    if args.reps < 1 or args.workers < 1:
        parser.error("reps and workers must be positive")
    run(
        args.out,
        args.reps,
        args.first_seed,
        tuple(d for d in DESIGNS if not args.scenarios or d.name in args.scenarios),
        tuple(args.methods or METHODS),
        args.solver,
        args.workers,
    )


if __name__ == "__main__":
    main()
