"""Post-inspection alpha=1 endpoint diagnostic for the learned-pooling study.

This separate study fixes cut=.5 and tau=1.5, selects lambda by the original
threefold training-only CV, and pairs against the frozen alpha=.5/.75 fits.
It does not extend or replace the prespecified primary joint tuning study.
At alpha=1, HCGL and FCGL both reduce to the same cellwise L1 penalty.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

if __package__:
    from . import learned_pooling_validation as L
else:
    import learned_pooling_validation as L

B = L.B
METHODS = ("hcgl_unconstrained", "hcgl_per_response", "hcgl_pooled_zero", "fcgl_unconstrained")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def endpoint_protocol(solver="MOSEK"):
    return B.Protocol(alpha=1.0, cutoff_fraction=0.5, tau_grid=(1.5,), solver=solver)


def fit_one(job):
    design, seed, output, original, solver = job
    start = time.perf_counter()
    prefix = Path(output) / "shards" / f"{design.name}_{seed}"
    x, y, xt, yt, truth, true_groups = L.simulate(design, seed)
    input_hashes = {key: L.array_hash(value) for key, value in zip(
        ("x_train", "y_train", "x_test", "y_test", "beta_true", "true_groups"),
        (x, y, xt, yt, truth, true_groups))}
    original_receipt = Path(original) / "shards" / f"{design.name}_{seed}_receipt.json"
    assert input_hashes == json.loads(original_receipt.read_text())["inputs"]
    rows, arrays = [], {}
    for method in METHODS:
        fitted = B.tune_and_fit(x, y, method, endpoint_protocol(solver), seed)
        arrays[method + "__beta"] = fitted.beta
        arrays[method + "__intercept"] = fitted.intercept
        arrays[method + "__labels"] = fitted.prepared.labels
        arrays[method + "__signs"] = fitted.signs
        for tolerance in B.ZERO_TOLERANCES:
            rows.append(dict(scenario=design.name, seed=seed, method=method,
                             alpha=1.0, cut=0.5, tau=1.5, zero_tolerance=tolerance,
                             lambda_fraction=fitted.lambda_fraction,
                             lambda_value=fitted.lambda_value, cv_mse=fitted.cv_mse,
                             seconds=fitted.seconds,
                             **B.evaluate(fitted, xt, yt, truth, tolerance)))
        pd.DataFrame(rows).to_csv(str(prefix) + "_draws.csv", index=False)
    np.savez_compressed(str(prefix) + "_coefficients.npz", **arrays)
    duplicate = float(np.max(np.abs(arrays["hcgl_unconstrained__beta"]
                                   - arrays["fcgl_unconstrained__beta"])))
    receipt = dict(scenario=design.name, seed=seed, inputs=input_hashes,
                   total_seconds=time.perf_counter() - start,
                   alpha1_hcgl_fcgl_max_coefficient_difference=duplicate)
    Path(str(prefix) + "_receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def summarize(output, original):
    draws = pd.concat([pd.read_csv(path) for path in sorted(
        (output / "shards").glob("*_draws.csv"))], ignore_index=True)
    draws.to_csv(output / "endpoint_draws.csv", index=False)
    L.summarize(draws, ["scenario", "method", "zero_tolerance"]).to_csv(
        output / "endpoint_summary.csv", index=False)
    prior = pd.read_csv(original / "estimator_draws.csv")
    metrics = ["prediction_mse_standardized", "prediction_r2", "beta_mse_standardized",
               "support_precision", "support_recall", "support_fdp",
               "active_sign_recovery", "active_sign_flip", "active_abstention"]
    rows = []
    for variant, alpha in (("reference", 0.5), ("alpha_0.75", 0.75)):
        base = prior[prior.variant == variant]
        merged = draws.merge(base, on=["scenario", "seed", "method", "zero_tolerance"],
                             suffixes=("_endpoint", "_original"), validate="one_to_one")
        assert len(merged) == len(draws)
        for (scenario, method, tolerance), part in merged.groupby(
                ["scenario", "method", "zero_tolerance"]):
            for metric in metrics:
                delta = part[metric + "_endpoint"] - part[metric + "_original"]
                se = delta.std(ddof=1) / np.sqrt(len(delta))
                half = student_t.ppf(.975, len(delta) - 1) * se
                rows.append(dict(scenario=scenario, method=method, zero_tolerance=tolerance,
                                 comparator_alpha=alpha, metric=metric, replications=len(delta),
                                 difference_mean=delta.mean(), difference_se=se,
                                 ci_low=delta.mean() - half, ci_high=delta.mean() + half))
    pd.DataFrame(rows).to_csv(output / "paired_endpoint_summary.csv", index=False)


def run(output, original, workers=6, solver="MOSEK"):
    output, original = B.safe_output(output), Path(original).resolve()
    if (output / "protocol.json").exists():
        raise FileExistsError("Choose a fresh endpoint diagnostic directory")
    (output / "shards").mkdir()
    protocol = dict(scope="Post-inspection endpoint diagnostic; primary study unchanged",
                    reason="Primary grid frequently selected alpha=.75 upper endpoint",
                    designs=[asdict(d) for d in L.DESIGNS], first_seed=1000, seeds=40,
                    methods=METHODS, alpha=1.0, cut=.5, tau=1.5, gamma=0,
                    lambda_fractions=B.Protocol().lambda_fractions, cv_folds=3,
                    tuning="Same training-only transformations and paired folds; lambda only",
                    endpoint_identity="At alpha=1 both HCGL/FCGL group terms vanish; L1 only",
                    lambda_scale="Exact L1 zero bound at alpha1; conservative bound below1",
                    comparisons="Paired vs main_40 reference alpha.5 and alpha_.75 variants",
                    zero_tolerances=B.ZERO_TOLERANCES, solver=solver, workers=workers,
                    primary_input=str(original), primary_draws_sha256=sha256(
                        original / "estimator_draws.csv"))
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    sources = [Path(__file__), Path(L.__file__), Path(B.__file__)]
    for source in sources:
        (output / (source.stem + "_snapshot.py")).write_bytes(source.read_bytes())
    receipt = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                   source_hashes={str(p): sha256(p) for p in sources},
                   protocol_sha256=sha256(output / "protocol.json"),
                   python=platform.python_version(), source_runtime_receipt=str(
                       original / "receipt.json"), completed_replications=0)
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    completed = []
    try:
        jobs = [(design, seed, str(output), str(original), solver)
                for design in L.DESIGNS for seed in range(1000, 1040)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fit_one, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                completed.append(result)
                print(f"{result['scenario']} {result['seed']} complete "
                      f"({len(completed)}/200; {result['total_seconds']:.2f}s)", flush=True)
        summarize(output, original)
        receipt.update(status="complete", completed_replications=len(completed),
                       completed_utc=datetime.now(timezone.utc).isoformat(),
                       max_endpoint_hcgl_fcgl_difference=max(
                           row["alpha1_hcgl_fcgl_max_coefficient_difference"]
                           for row in completed))
    except BaseException as exc:
        receipt.update(status="failed", error=repr(exc), completed_replications=len(completed))
        raise
    finally:
        receipt["outputs"] = {str(p.relative_to(output)): sha256(p)
                              for p in output.rglob("*") if p.is_file()
                              and p.name != "receipt.json"}
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    run(args.output, args.original, args.workers)
