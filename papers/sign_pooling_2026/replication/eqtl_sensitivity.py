"""Prespecified paired, one-factor eQTL sensitivity; settings are not CV-selected.

Each affected fit uses the original nine outer splits and three inner folds,
starting from raw markers on every inner/outer training split. Only the penalty
fraction is tuned. Neighbor settings characterize sensitivity and are not ranked
to choose a new claimed optimum. Unchanged mathematical estimators reuse verified
baseline fits, with an explicit method-by-setting provenance ledger.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import pandas as pd
from sklearn.metrics import adjusted_rand_score

if __package__:
    from . import eqtl_validation as validation
    from . import eqtl_stability as stability
else:
    import eqtl_validation as validation
    import eqtl_stability as stability


PRIMARY = ("lasso", "hcgl_unconstrained", "hcgl_exclude", "hcgl_unrestricted")
ADDITIONS = ("hcgl_singleton_exclude", "fcgl_unconstrained")
NEIGHBORS = (
    ("ld_085", "ld_correlation", 0.85),
    ("ld_098", "ld_correlation", 0.98),
    ("screen_0005", "screen_p", 0.005),
    ("screen_002", "screen_p", 0.02),
    ("cut_05", "cluster_cut", 0.5),
    ("cut_09", "cluster_cut", 0.9),
    ("tau_1", "gate_threshold", 1.0),
    ("tau_3", "gate_threshold", 3.0),
    ("alpha_02", "l1_weight", 0.2),
    ("alpha_08", "l1_weight", 0.8),
)


def affected_methods(parameter: str) -> tuple[str, ...]:
    if parameter in ("ld_correlation", "screen_p"):
        return PRIMARY
    if parameter in ("cluster_cut", "l1_weight"):
        return PRIMARY[1:]
    if parameter == "gate_threshold":
        return ("hcgl_exclude", "hcgl_unrestricted")
    raise ValueError(f"Unknown sensitivity parameter {parameter}")


def make_declaration(prior: Path) -> dict:
    base = validation.Protocol()
    prior_receipt = json.loads((prior / "receipt.json").read_text(encoding="utf-8"))
    if to_protocol(prior_receipt["protocol"]) != base:
        raise ValueError("Prior protocol must match the declared six-method baseline exactly")
    configurations = {"baseline": asdict(replace(base, methods=validation.AVAILABLE_METHODS))}
    tasks = {"baseline_additions": asdict(replace(base, methods=ADDITIONS))}
    for name, parameter, value in NEIGHBORS:
        protocol = replace(base, **{parameter: value}, methods=PRIMARY)
        configurations[name] = asdict(protocol)
        tasks[name] = asdict(replace(protocol, methods=affected_methods(parameter)))
    return dict(
        declared_utc=datetime.now(timezone.utc).isoformat(),
        prior_run=str(prior.resolve()),
        prior_receipt_sha256=validation._sha256(prior / "receipt.json"),
        configurations=configurations,
        fit_tasks=tasks,
        analysis="One-factor sensitivity; no outer outcome chooses any setting",
        reuse=(
            "Baseline outputs are reused only if a parameter has no effect on that method: "
            "Lasso ignores cut/alpha/tau; unconstrained HCGL ignores tau. "
            "All changes to LD or outcome screening refit every primary method."
        ),
        uncertainty="Three repeated split scores and 36 overlapping fold pairs are descriptive",
        stability_tolerance=1e-6,
        coefficient_units="training-standardized",
        maximum_concurrent_tasks=2,
        source_hashes={
            p.name: validation._sha256(p)
            for p in (Path(__file__), Path(stability.__file__), Path(validation.__file__))
        },
    )


def to_protocol(values: dict) -> validation.Protocol:
    return validation.Protocol(
        **{k: tuple(v) if k in ("methods", "lambda_fractions") else v for k, v in values.items()}
    )


def fit_task(name: str, output: str) -> dict:
    root = Path(output)
    declaration = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    for filename, digest in declaration["source_hashes"].items():
        if validation._sha256(Path(__file__).with_name(filename)) != digest:
            raise RuntimeError(f"Source changed after predeclaration: {filename}")
    protocol = to_protocol(declaration["fit_tasks"][name])
    started = time.perf_counter()
    with (root / "logs" / f"{name}.txt").open("w", encoding="utf-8") as log:
        with redirect_stdout(log), redirect_stderr(log):
            print(
                f"Task {name}; protocol SHA256 {validation._sha256(root / 'protocol.json')}",
                flush=True,
            )
            y_all, x, positions = validation.load_data()
            genes, _ = validation.select_mapk_genes(y_all)
            result = validation.run_nested_cv(x, y_all.loc[:, genes], positions, protocol)
            receipt = validation.save_results(
                result, root / "runs" / name, protocol, time.perf_counter() - started
            )
            receipt["declaration_sha256"] = validation._sha256(root / "protocol.json")
            receipt["followup_sources"] = declaration["source_hashes"]
            (root / "runs" / name / "receipt.json").write_text(
                json.dumps(receipt, indent=2), encoding="utf-8"
            )
    return dict(task=name, seconds=time.perf_counter() - started)


def verify_fold_alignment(reference, candidate):
    if len(reference) != len(candidate):
        raise AssertionError("Outer fold counts changed")
    for a, b in zip(reference, candidate):
        for name in ("repeat", "fold", "train_ids", "test_ids"):
            if a[name] != b[name]:
                raise AssertionError(f"Paired outer split mismatch: {name}")
        if len(a["inner"]) != len(b["inner"]):
            raise AssertionError("Inner fold counts changed")
        for ia, ib in zip(a["inner"], b["inner"]):
            for name in ("train_ids", "validation_ids"):
                if ia[name] != ib[name]:
                    raise AssertionError(f"Paired inner split mismatch: {name}")


def combine_configuration(name, declaration, prior, runs):
    protocol = to_protocol(declaration["configurations"][name])
    current = runs["baseline_additions"] if name == "baseline" else runs[name]
    manifest = current["manifest"]
    verify_fold_alignment(prior["manifest"], manifest)
    provenance, sources = [], {}
    newly_fitted = set(current["receipt"]["protocol"]["methods"])
    for method in protocol.methods:
        if method in newly_fitted:
            source, source_name = current, ("baseline_additions" if name == "baseline" else name)
        else:
            source, source_name = prior, "original_nested_v1"
        sources[method] = source
        provenance.append(dict(configuration=name, method=method, source_run=source_name))
        for original, target in zip(source["manifest"], manifest):
            op, tp = original["outer_preprocessing"], target["outer_preprocessing"]
            for item in ("selected_markers", "x_mean", "x_sd", "y_mean", "y_sd"):
                if op[item] != tp[item]:
                    raise AssertionError(f"Unsafe reuse for {name}/{method}: changed {item}")
    frames = {}
    for field in ("predictions", "repeats", "summary", "folds", "tuning", "genes"):
        if field == "genes":
            continue
        frames[field] = pd.concat(
            [
                source[field].loc[source[field].method == method]
                for method, source in sources.items()
            ],
            ignore_index=True,
        )
    # Record the current preprocessing partition even for a partition-invariant reused Lasso.
    clusters = {
        (r["repeat"], r["fold"]): len(set(r["outer_preprocessing"]["cluster_by_gene"].values()))
        for r in manifest
    }
    frames["folds"]["n_clusters"] = [
        clusters[(r.repeat, r.fold)] for r in frames["folds"].itertuples()
    ]
    _, frames["genes"] = validation.summarize_predictions(frames["predictions"])
    coefficients = {}
    for method, source in sources.items():
        coefficients.update(
            {k: v for k, v in source["coefficients"].items() if k.endswith("_" + method)}
        )
    return dict(
        **frames, manifest=manifest, coefficients=coefficients, heldout=current["heldout"]
    ), provenance


def summarize_saved(root: Path):
    """Postprocess completed fits; never refit or choose a protocol from outer scores.

    Numerical support thresholds 1e-8 and 1e-4 are post hoc diagnostics around the
    declared 1e-6 threshold. All thresholds and all configurations are retained.
    Same-fold comparisons isolate a setting change, whereas stability_tables
    compares different overlapping training splits within a setting.
    """
    repeats = pd.read_csv(root / "prediction_repeats.csv")
    paired = []
    for name, frame in repeats.groupby("configuration"):
        indexed = frame.set_index(["method", "repeat"])
        for left, right in (
            ("hcgl_exclude", "hcgl_unconstrained"),
            ("hcgl_unrestricted", "hcgl_exclude"),
            ("hcgl_unrestricted", "hcgl_unconstrained"),
            ("lasso", "hcgl_unconstrained"),
        ):
            for metric in ("r2_gene_median", "r2_gene_mean", "r2_pooled"):
                difference = indexed.loc[left, metric] - indexed.loc[right, metric]
                paired.append(
                    dict(
                        configuration=name,
                        left=left,
                        right=right,
                        metric=metric,
                        mean=difference.mean(),
                        split_sd=difference.std(ddof=1),
                        minimum=difference.min(),
                        maximum=difference.max(),
                    )
                )
    base = stability.read_run(root / "combined" / "baseline")
    structures, coefficients, boundaries, sizes, tolerances, counts = [], [], [], [], [], []
    for folder in sorted((root / "combined").iterdir()):
        current = stability.read_run(folder)
        verify_fold_alignment(base["manifest"], current["manifest"])
        fractions = current["receipt"]["protocol"]["lambda_fractions"]
        for method, frame in current["folds"].groupby("method"):
            boundaries.append(
                dict(
                    configuration=folder.name,
                    method=method,
                    upper_fraction_rate=(frame.fraction == max(fractions)).mean(),
                    lower_fraction_rate=(frame.fraction == min(fractions)).mean(),
                    chosen_fractions=json.dumps(
                        frame.fraction.value_counts().sort_index().to_dict()
                    ),
                )
            )
        for baseline, changed in zip(base["manifest"], current["manifest"]):
            i, j = changed["repeat"], changed["fold"]
            a, b = baseline["outer_preprocessing"], changed["outer_preprocessing"]
            genes = sorted(a["cluster_by_gene"])
            if set(genes) != set(b["cluster_by_gene"]):
                raise ValueError("Same-fold comparisons require identical gene labels")
            markers = sorted(set(a["selected_markers"]) | set(b["selected_markers"]))
            sizes.append(
                dict(
                    configuration=folder.name,
                    repeat=i,
                    fold=j,
                    selected_markers=len(b["selected_markers"]),
                    ld_markers=len(b["ld_markers"]),
                    clusters=len(set(b["cluster_by_gene"].values())),
                )
            )
            structures.append(
                dict(
                    configuration=folder.name,
                    repeat=i,
                    fold=j,
                    ari_to_baseline=adjusted_rand_score(
                        [a["cluster_by_gene"][g] for g in genes],
                        [b["cluster_by_gene"][g] for g in genes],
                    ),
                    selected_marker_jaccard_to_baseline=stability.jaccard(
                        set(a["selected_markers"]), set(b["selected_markers"])
                    ),
                    ld_marker_jaccard_to_baseline=stability.jaccard(
                        set(a["ld_markers"]), set(b["ld_markers"])
                    ),
                )
            )
            for method in PRIMARY:
                key = f"r{i}_f{j}_{method}"
                left = stability.coefficient_states(
                    **base["coefficients"][key],
                    marker_universe=markers,
                    gene_universe=genes,
                    tolerance=1e-6,
                )
                right = stability.coefficient_states(
                    **current["coefficients"][key],
                    marker_universe=markers,
                    gene_universe=genes,
                    tolerance=1e-6,
                )
                coefficients.append(
                    dict(
                        configuration=folder.name,
                        method=method,
                        repeat=i,
                        fold=j,
                        **stability.compare_states(left, right),
                    )
                )
        for tolerance in (1e-8, 1e-6, 1e-4):
            result = stability.stability_tables(
                current["manifest"], current["coefficients"], tolerance
            )
            tolerances.append(
                result["coefficient_summary"].assign(configuration=folder.name, tolerance=tolerance)
            )
            for method, frame in result["coefficient_pairs"].groupby("method"):
                for column in (
                    "present_both",
                    "selected_both",
                    "selected_union",
                    "same_direction",
                    "opposite_direction",
                    "absent_one",
                    "zero_one_present",
                ):
                    counts.append(
                        dict(
                            configuration=folder.name,
                            method=method,
                            tolerance=tolerance,
                            quantity=column,
                            mean=frame[column].mean(),
                            median=frame[column].median(),
                            minimum=frame[column].min(),
                            maximum=frame[column].max(),
                        )
                    )
    tables = dict(
        paired_methods_by_configuration=pd.DataFrame(paired),
        same_fold_partition_marker_sensitivity=pd.DataFrame(structures),
        same_fold_coefficient_sensitivity=pd.DataFrame(coefficients),
        penalty_grid_diagnostics=pd.DataFrame(boundaries),
        preprocessing_sizes=pd.DataFrame(sizes),
        support_tolerance_summary=pd.concat(tolerances, ignore_index=True),
        support_tolerance_denominators=pd.DataFrame(counts),
    )
    for name, frame in tables.items():
        frame.to_csv(root / f"{name}.csv", index=False)
    receipt = dict(
        analysis="Saved-fit paired contrasts and stability; no setting selected or model refit",
        support_tolerance_analysis="Post hoc thresholds 1e-8, 1e-6, 1e-4; all reported",
        uncertainty="Descriptive split SD and overlapping-fold dispersion; not sampling SE/CI",
        source_sha256=validation._sha256(Path(__file__)),
        stability_source_sha256=validation._sha256(Path(stability.__file__)),
        declaration_sha256=validation._sha256(root / "protocol.json"),
        outputs={f"{name}.csv": validation._sha256(root / f"{name}.csv") for name in tables},
    )
    (root / "saved_fit_analysis_receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    return tables


def analyze(root: Path):
    declaration = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    prior = stability.read_run(Path(declaration["prior_run"]))
    runs = {name: stability.read_run(root / "runs" / name) for name in declaration["fit_tasks"]}
    _, raw_x, _ = validation.load_data()
    ledger, all_summary, all_repeats, structure, remaining, fitted = [], [], [], [], [], []
    checks = {}
    for name in declaration["configurations"]:
        result, provenance = combine_configuration(name, declaration, prior, runs)
        ledger.extend(provenance)
        protocol = to_protocol(declaration["configurations"][name])
        destination = root / "combined" / name
        receipt = validation.save_results(result, destination, protocol, elapsed=0)
        receipt["scope"] = "Derived assembly of paired fits; source runs and reuse recorded below"
        receipt["provenance"] = provenance
        receipt["declaration_sha256"] = validation._sha256(root / "protocol.json")
        (destination / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        tables = stability.stability_tables(
            result["manifest"], result["coefficients"], raw_markers=raw_x.columns
        )
        tables["remaining_ld"] = stability.remaining_ld(result["manifest"], raw_x)
        checks[name] = stability.verify_predictions(result["predictions"], result["repeats"])
        stability.write_stability(tables, root / "stability" / name)
        all_summary.append(result["summary"].assign(configuration=name))
        all_repeats.append(result["repeats"].assign(configuration=name))
        structure.append(
            dict(
                configuration=name,
                **{
                    f"{metric}_{stat}": value
                    for metric, values in tables["partition_summary"].items()
                    if isinstance(values, dict)
                    for stat, value in values.items()
                },
            )
        )
        remaining.append(tables["remaining_ld"].assign(configuration=name))
        fitted.append(tables["coefficient_summary"].assign(configuration=name))
    summary, repeats = pd.concat(all_summary), pd.concat(all_repeats)
    summary.to_csv(root / "prediction_summary.csv", index=False)
    repeats.to_csv(root / "prediction_repeats.csv", index=False)
    pd.DataFrame(structure).to_csv(root / "partition_summary.csv", index=False)
    pd.concat(remaining).to_csv(root / "remaining_ld.csv", index=False)
    pd.concat(fitted).to_csv(root / "coefficient_stability_summary.csv", index=False)
    pd.DataFrame(ledger).to_csv(root / "provenance.csv", index=False)
    baseline = repeats.loc[repeats.configuration == "baseline"].set_index(["method", "repeat"])
    deltas = []
    for (name, method), frame in repeats.loc[repeats.configuration != "baseline"].groupby(
        ["configuration", "method"]
    ):
        for metric in ("r2_gene_median", "r2_gene_mean", "r2_pooled"):
            differences = frame.set_index("repeat")[metric] - baseline.loc[method, metric]
            deltas.append(
                dict(
                    configuration=name,
                    method=method,
                    metric=metric,
                    paired_difference_mean=differences.mean(),
                    paired_difference_split_sd=differences.std(ddof=1),
                    minimum=differences.min(),
                    maximum=differences.max(),
                )
            )
    pd.DataFrame(deltas).to_csv(root / "paired_sensitivity.csv", index=False)
    comparison_rows = []
    for left, right in (
        ("hcgl_exclude", "hcgl_singleton_exclude"),
        ("hcgl_exclude", "hcgl_unconstrained"),
        ("hcgl_unrestricted", "hcgl_exclude"),
        ("fcgl_exclude", "fcgl_unconstrained"),
    ):
        for metric in ("r2_gene_median", "r2_gene_mean", "r2_pooled"):
            differences = baseline.loc[left, metric] - baseline.loc[right, metric]
            comparison_rows.append(
                dict(
                    left=left,
                    right=right,
                    metric=metric,
                    difference_mean=differences.mean(),
                    difference_split_sd=differences.std(ddof=1),
                )
            )
    pd.DataFrame(comparison_rows).to_csv(root / "paired_baseline_methods.csv", index=False)
    (root / "verification.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    summarize_saved(root)
    receipt = dict(
        status="completed",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        declaration_sha256=validation._sha256(root / "protocol.json"),
        original_receipt_sha256=declaration["prior_receipt_sha256"],
        followup_sources=declaration["source_hashes"],
        outputs={
            str(p.relative_to(root)): validation._sha256(p)
            for p in root.rglob("*")
            if p.is_file() and p.name != "receipt.json"
        },
        run_receipts={
            name: validation._sha256(root / "runs" / name / "receipt.json") for name in runs
        },
        no_setting_selected_by_outer_outcomes=True,
        uncertainty="Split SD and overlapping fold-pair dispersion are not sampling SE or CI",
    )
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(
        summary[["configuration", "method", "r2_median_mean", "r2_pooled"]].to_string(index=False)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Recompute saved-fit contrasts and support tolerance only; no fits",
    )
    args = parser.parse_args()
    if any(part.lower().startswith("onedrive") for part in args.out.resolve().parts):
        parser.error("Outputs must be outside OneDrive")
    if args.summarize_only:
        summarize_saved(args.out)
        return
    if args.analyze_only:
        analyze(args.out)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    declaration = make_declaration(args.prior)
    (args.out / "protocol.json").write_text(json.dumps(declaration, indent=2), encoding="utf-8")
    for name in ("logs", "runs", "combined", "stability"):
        (args.out / name).mkdir()
    print(
        f"Predeclared {len(declaration['fit_tasks'])} tasks: {args.out / 'protocol.json'}",
        flush=True,
    )
    prior = stability.read_run(args.prior)
    _, raw_x, _ = validation.load_data()
    tables = stability.stability_tables(
        prior["manifest"], prior["coefficients"], raw_markers=raw_x.columns
    )
    tables["remaining_ld"] = stability.remaining_ld(prior["manifest"], raw_x)
    tables["verification"] = stability.verify_predictions(prior["predictions"], prior["repeats"])
    stability.write_stability(tables, args.out / "original_stability")
    print("Original nine-fold stability saved", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fit_task, name, str(args.out)): name
            for name in declaration["fit_tasks"]
        }
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)
    analyze(args.out)


if __name__ == "__main__":
    main()
