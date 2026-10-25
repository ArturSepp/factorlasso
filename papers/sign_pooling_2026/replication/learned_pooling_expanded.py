"""Post-inspection joint-CV alpha=1 extension using validated prior CV surfaces.

Run the original learned_pooling_validation first, then pass its run to --prior.
--fresh-cv recomputes the original CV surfaces and checks them against the cache;
the candidate order and tie rule are identical in either mode. Neither mode
changes the prior run. All test outcomes and coefficient truth enter metrics only.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import t

if __package__:
    from . import learned_pooling_validation as L
else:
    import learned_pooling_validation as L

B = L.B
EXTENDED = ("hcgl_unconstrained", "hcgl_per_response", "hcgl_pooled_zero",
            "hcgl_pooled_free", "fcgl_unconstrained")
METRICS = ("prediction_mse_standardized", "prediction_r2", "beta_mse_standardized",
           "support_precision", "support_recall", "support_fdp", "active_sign_recovery",
           "active_sign_flip", "active_abstention")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def endpoint_candidates(method):
    taus = L.TAUS if method in B.SIGNED_METHODS else (1.5,)
    return tuple(L.Candidate(cut, 1., tau) for cut in L.CUTS for tau in taus)


def additional_surface(x, y, method, seed, solver):
    candidates = endpoint_candidates(method)
    fractions = B.lambda_fractions_for(method, B.Protocol())
    errors = np.zeros((len(candidates), len(fractions)))
    for train, valid in B.cv_splits(len(x), 3, seed):
        prepared = {cut: B.prepare(x[train], y[train], cut) for cut in L.CUTS}
        for i, candidate in enumerate(candidates):
            prep = prepared[candidate.cut]
            betas, intercepts, _, _ = B.fit_path(
                prep, method, candidate.tau, L.candidate_protocol(candidate, solver))
            prediction = np.einsum("ij,lkj->lik", prep.transform_x(x[valid]), betas)
            prediction += intercepts[:, None, :]
            errors[i] += np.sum((prediction - prep.transform_y(y[valid])[None]) ** 2,
                                axis=(1, 2)) / y.shape[1]
    return candidates, fractions, errors / len(x)


def prior_surface(frame, method, seed, scenario):
    part = frame[frame.method == method].copy()
    expected = [(c.cut, c.alpha, c.tau, fraction) for c in L.candidates_for(method)
                for fraction in B.lambda_fractions_for(method, B.Protocol())]
    actual = list(part[["cut", "alpha", "tau", "lambda_fraction"]].itertuples(
        index=False, name=None))
    if actual != expected or not (part.seed == seed).all() or not (
            part.scenario == scenario).all() or not np.isfinite(part.cv_mse).all():
        raise ValueError("Prior CV schema, ordering, seed, scenario or finite-loss check failed")
    return part


def fit_one(job):
    design, seed, out, prior, solver, fresh_cv = job
    prefix = Path(out) / "shards" / f"{design.name}_{seed}"
    prior_prefix = Path(prior) / "shards" / f"{design.name}_{seed}"
    start = time.perf_counter()
    x, y, xt, yt, truth, true_groups = L.simulate(design, seed)
    hashes = {key: L.array_hash(value) for key, value in zip(
        ("x_train", "y_train", "x_test", "y_test", "beta_true", "true_groups"),
        (x, y, xt, yt, truth, true_groups))}
    assert hashes == json.loads(Path(str(prior_prefix) + "_receipt.json").read_text())["inputs"]
    original = pd.read_csv(str(prior_prefix) + "_draws.csv")
    original = original[original.variant == "joint_cv"]
    original_cv = pd.read_csv(str(prior_prefix) + "_cv.csv")
    original_arrays = np.load(str(prior_prefix) + "_coefficients.npz")
    rows, surfaces, arrays, selections = [], [], {}, []
    for method in L.METHODS:
        old = original[original.method == method]
        descriptor = method + "__joint_cv"
        if method not in EXTENDED:
            rows.extend(old.to_dict("records"))
            for suffix in ("beta", "intercept", "labels", "signs"):
                arrays[descriptor + "__" + suffix] = original_arrays[descriptor + "__" + suffix]
            continue
        method_start = time.perf_counter()
        prior_part = prior_surface(original_cv, method, seed, design.name)
        if fresh_cv:
            _, _, fresh_errors = L.tune_surfaces(x, y, method, seed, solver)
            np.testing.assert_allclose(fresh_errors.ravel(), prior_part.cv_mse, rtol=0, atol=1e-12)
            prior_part.loc[:, "cv_mse"] = fresh_errors.ravel()
        candidates, fractions, errors = additional_surface(x, y, method, seed, solver)
        additional = pd.DataFrame([
            dict(scenario=design.name, seed=seed, method=method, **asdict(candidate),
                 lambda_fraction=fraction, cv_mse=errors[i, j])
            for i, candidate in enumerate(candidates) for j, fraction in enumerate(fractions)])
        combined = pd.concat([prior_part, additional], ignore_index=True)
        assert np.isfinite(combined.cv_mse).all()
        choice = combined.iloc[int(np.argmin(combined.cv_mse))]
        candidate = L.Candidate(choice.cut, choice.alpha, choice.tau)
        old_winner = old.iloc[0]
        if choice.alpha < 1:
            for key in ("cut", "alpha", "tau", "lambda_fraction"):
                assert choice[key] == old_winner[key]
        prep = B.prepare(x, y, candidate.cut)
        cv_seconds = time.perf_counter() - method_start
        refit_start = time.perf_counter()
        if choice.alpha == 1:
            beta, intercept, lambdas, signs = B.fit_path(
                prep, method, candidate.tau, L.candidate_protocol(candidate, solver),
                fractions=[choice.lambda_fraction])
            beta, intercept, lam = beta[0], intercept[0], lambdas[0]
        else:
            beta = original_arrays[descriptor + "__beta"]
            intercept = original_arrays[descriptor + "__intercept"]
            signs = original_arrays[descriptor + "__signs"]
            lam = old_winner.lambda_value
        fitted = B.Fitted(beta, intercept, prep, lam, choice.lambda_fraction,
                          candidate.tau, choice.cv_mse, time.perf_counter() - method_start, signs)
        for suffix, values in (("beta", beta), ("intercept", intercept),
                               ("labels", prep.labels), ("signs", signs)):
            arrays[descriptor + "__" + suffix] = values
        for tolerance in B.ZERO_TOLERANCES:
            rows.append(dict(scenario=design.name, seed=seed, method=method, variant="joint_cv",
                             zero_tolerance=tolerance, **asdict(candidate),
                             lambda_fraction=choice.lambda_fraction, lambda_value=lam,
                             cv_mse=choice.cv_mse, joint_search_seconds=(
                                 old_winner.joint_search_seconds + cv_seconds),
                             refit_seconds=time.perf_counter() - refit_start,
                             lambda_upper_boundary=int(choice.lambda_fraction == max(fractions)),
                             lambda_lower_boundary=int(choice.lambda_fraction == min(fractions)),
                             **L.group_metrics(prep.labels, true_groups, truth),
                             **B.evaluate(fitted, xt, yt, truth, tolerance)))
        surfaces.append(combined)
        selections.append(dict(method=method, **asdict(candidate),
                               lambda_fraction=choice.lambda_fraction,
                               prior_cv_mse=old_winner.cv_mse, expanded_cv_mse=choice.cv_mse,
                               extension_search_seconds=cv_seconds))
        pd.DataFrame(rows).to_csv(str(prefix) + "_draws.csv", index=False)
    original_arrays.close()
    pd.DataFrame(rows).to_csv(str(prefix) + "_draws.csv", index=False)
    pd.concat(surfaces, ignore_index=True).to_csv(str(prefix) + "_cv.csv", index=False)
    np.savez_compressed(str(prefix) + "_coefficients.npz", **arrays)
    receipt = dict(scenario=design.name, seed=seed, inputs=hashes,
                   total_seconds=time.perf_counter()-start, selections=selections)
    Path(str(prefix) + "_receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def summarize(out, prior):
    draws = pd.concat([pd.read_csv(p) for p in sorted((out / "shards").glob("*_draws.csv"))],
                      ignore_index=True)
    draws.to_csv(out / "estimator_draws.csv", index=False)
    L.summarize(draws, ["scenario", "method", "variant", "zero_tolerance"]).to_csv(
        out / "estimator_summary.csv", index=False)
    original = pd.read_csv(prior / "estimator_draws.csv")
    original = original[original.variant == "joint_cv"]
    keys = ["scenario", "seed", "method", "zero_tolerance"]
    pairs = []
    for comparator in ("hcgl_unconstrained", "hcgl_per_response", "lasso", "original_joint_cv"):
        if comparator == "original_joint_cv":
            merged = draws.merge(original, on=keys, suffixes=("_expanded", "_base"),
                                 validate="one_to_one")
        else:
            base = draws[draws.method == comparator].drop(columns="method")
            merged = draws.merge(base, on=["scenario", "seed", "zero_tolerance"],
                                 suffixes=("_expanded", "_base"), validate="many_to_one")
        for (scenario, method, tolerance), frame in merged.groupby(
                ["scenario", "method", "zero_tolerance"]):
            for metric in METRICS:
                delta = frame[metric + "_expanded"] - frame[metric + "_base"]
                se = delta.std(ddof=1)/np.sqrt(len(delta))
                half = t.ppf(.975, len(delta)-1)*se
                pairs.append(dict(scenario=scenario, method=method, variant="joint_cv",
                                  comparator=comparator, metric=metric, zero_tolerance=tolerance,
                                  replications=len(delta), difference_mean=delta.mean(),
                                  difference_se=se, ci_low=delta.mean()-half,
                                  ci_high=delta.mean()+half))
    pd.DataFrame(pairs).to_csv(out / "paired_summary.csv", index=False)
    primary = draws[draws.zero_tolerance == B.PRIMARY_ZERO_TOLERANCE]
    choices = []
    for method in EXTENDED:
        parameters = ["cut", "alpha", "lambda_fraction"]
        if method in B.SIGNED_METHODS:
            parameters.append("tau")
        for parameter in parameters:
            for (scenario, value), count in primary[primary.method == method].groupby(
                    ["scenario", parameter]).size().items():
                choices.append(dict(scenario=scenario, method=method, parameter=parameter,
                                    value=value, count=count, fraction=count/40))
    pd.DataFrame(choices).to_csv(out / "selection_frequencies.csv", index=False)


def validate_prior(prior):
    receipt = json.loads((prior / "receipt.json").read_text())
    protocol = json.loads((prior / "protocol.json").read_text())
    if receipt["status"] != "complete" or receipt["completed_replications"] != 200:
        raise ValueError("Require completed 200-replication original study")
    assert protocol["designs"] == [asdict(d) for d in L.DESIGNS]
    assert protocol["first_seed"] == 1000 and protocol["reps"] == 40
    for source in [Path(L.__file__), Path(B.__file__),
                   B.REPOSITORY_DIR / "src/factorlasso/priors/_signs.py"]:
        snapshot = prior / (source.stem + "_snapshot.py")
        if digest(source) != digest(snapshot):
            raise ValueError(f"Changed numerical producer since prior run: {source.name}")


def run(prior, out, workers=6, fresh_cv=False):
    prior = Path(prior).resolve()
    validate_prior(prior)
    out = B.safe_output(out)
    if (out / "protocol.json").exists():
        raise FileExistsError("Choose a fresh expanded-grid run directory")
    (out / "shards").mkdir()
    protocol = dict(scope="Post-inspection alpha1 extension of joint CV, original preserved",
                    trigger="Upper-alpha boundary selection and separate endpoint diagnostic",
                    prior=str(prior), prior_draws_sha256=digest(prior / "estimator_draws.csv"),
                    prior_protocol_sha256=digest(prior / "protocol.json"),
                    designs=[asdict(d) for d in L.DESIGNS], seeds=list(range(1000, 1040)),
                    methods=list(L.METHODS), extended_methods=EXTENDED,
                    added_candidates={m: [asdict(c) for c in endpoint_candidates(m)]
                                      for m in EXTENDED},
                    lambda_fractions=B.Protocol().lambda_fractions,
                    selection="Minimum combined inner-CV MSE only; test/truth evaluation only",
                    ordering="Original candidates first in original order, then alpha1 by cut/tau",
                    ties="First candidate/lambda in declared ordering; original wins exact ties",
                    fresh_cv=fresh_cv, workers=workers, solver="MOSEK", solver_threads=1)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    sources = [Path(__file__), Path(L.__file__), Path(B.__file__)]
    for source in sources:
        (out / (source.stem + "_snapshot.py")).write_bytes(source.read_bytes())
    receipt = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                   source_hashes={str(p):digest(p) for p in sources},
                   protocol_sha256=digest(out / "protocol.json"), completed_replications=0,
                   runtime_metadata=str(prior / "receipt.json"))
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    completed = []
    try:
        jobs = [(d, seed, str(out), str(prior), "MOSEK", fresh_cv)
                for d in L.DESIGNS for seed in range(1000, 1040)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fit_one, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                completed.append(result)
                print(f"{result['scenario']} {result['seed']} complete "
                      f"({len(completed)}/200; {result['total_seconds']:.1f}s)", flush=True)
        summarize(out, prior)
        receipt.update(status="complete", completed_replications=len(completed),
                       completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        receipt.update(status="failed", error=repr(error), completed_replications=len(completed))
        raise
    finally:
        receipt["outputs"] = {str(p.relative_to(out)):digest(p) for p in out.rglob("*")
                              if p.is_file() and p.name != "receipt.json"}
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--fresh-cv", action="store_true")
    args = parser.parse_args()
    run(args.prior, args.out, args.workers, args.fresh_cv)
