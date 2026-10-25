"""Descriptive eQTL stability from saved nested fits, without refitting models.

All pairwise summaries use overlapping training splits of the same subjects;
their dispersion is descriptive, never an independent-sample standard error.
Coefficient states distinguish a marker absent after preprocessing from a retained
marker with a numerically zero fitted coefficient, and from either fitted sign.
The cell universe is all genes crossed with the union of selected markers across
the supplied folds. Markers never selected are reported separately when a raw
marker universe is supplied. Coefficients are in training-standardized units.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

if __package__:
    from . import eqtl_validation as validation
else:
    import eqtl_validation as validation


ABSENT = 2  # Distinct from negative, numerical zero, and positive.


def jaccard(left: set, right: set) -> float:
    """Empty/empty sets agree; preserve the denominator in accompanying counts."""
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def read_run(path: Path) -> dict:
    """Read only trusted local run files and verify their receipt hashes."""
    receipt = json.loads((path / "receipt.json").read_text(encoding="utf-8"))
    for name, digest in receipt["outputs"].items():
        if validation._sha256(path / name) != digest:
            raise ValueError(f"Saved run hash mismatch: {path / name}")
    coefficients = {}
    # These arrays were produced locally by eqtl_validation, not uploaded input.
    with np.load(path / "coefficients.npz", allow_pickle=True) as archive:
        for key in archive.files:
            if key.endswith("_beta"):
                prefix = key[:-5]
                coefficients[prefix] = {
                    field: archive[f"{prefix}_{field}"] for field in ("beta", "markers", "genes")
                }
    return dict(
        receipt=receipt,
        manifest=json.loads((path / "fold_manifest.json").read_text(encoding="utf-8")),
        coefficients=coefficients,
        predictions=pd.read_csv(path / "predictions.csv.gz"),
        repeats=pd.read_csv(path / "repeats.csv"),
        summary=pd.read_csv(path / "summary.csv"),
        folds=pd.read_csv(path / "folds.csv"),
        tuning=pd.read_csv(path / "tuning.csv"),
        heldout=pd.read_csv(path / "heldout.csv"),
    )


def coefficient_states(beta, markers, genes, marker_universe, gene_universe, tolerance):
    """Align one fit without counting an absent marker as a fitted zero."""
    beta = np.asarray(beta)
    markers, genes = list(markers), list(genes)
    if beta.shape != (len(genes), len(markers)) or not np.isfinite(beta).all():
        raise ValueError("Coefficient dimensions and finite values must match labels")
    if len(set(markers)) != len(markers) or len(set(genes)) != len(genes):
        raise ValueError("Coefficient labels must be unique")
    if set(genes) != set(gene_universe) or not set(markers) <= set(marker_universe):
        raise ValueError("Coefficient labels do not match the analysis universe")
    result = np.full((len(gene_universe), len(marker_universe)), ABSENT, dtype=np.int8)
    gi = pd.Index(gene_universe).get_indexer(genes)
    mi = pd.Index(marker_universe).get_indexer(markers)
    states = np.where(np.abs(beta) > tolerance, np.sign(beta), 0).astype(np.int8)
    result[np.ix_(gi, mi)] = states
    return result


def compare_states(left, right) -> dict:
    """Pairwise selection and direction with explicit, different denominators."""
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        raise ValueError("State matrices must have identical dimensions")
    present = (left != ABSENT) & (right != ABSENT)
    l_selected, r_selected = np.abs(left) == 1, np.abs(right) == 1
    both = l_selected & r_selected
    union = l_selected | r_selected
    same = both & (left == right)
    union_n, both_n = int(union.sum()), int(both.sum())
    present_union = union & present
    return dict(
        cells=int(left.size),
        absent_both=int(((left == ABSENT) & (right == ABSENT)).sum()),
        absent_one=int(((left == ABSENT) ^ (right == ABSENT)).sum()),
        present_both=int(present.sum()),
        zero_both=int(((left == 0) & (right == 0)).sum()),
        zero_one_present=int((present & ((left == 0) ^ (right == 0))).sum()),
        selected_union=union_n,
        selected_both=both_n,
        same_direction=int(same.sum()),
        opposite_direction=int((both & (left != right)).sum()),
        selection_jaccard=both_n / union_n if union_n else 1.0,
        selection_jaccard_common_markers=(
            both_n / int(present_union.sum()) if present_union.any() else 1.0
        ),
        direction_agreement_selected_both=float(same.sum() / both_n) if both_n else np.nan,
        signed_overlap_over_selected_union=float(same.sum() / union_n) if union_n else np.nan,
    )


def stability_tables(manifest, coefficients, tolerance=1e-6, raw_markers=None) -> dict:
    """Summarize partitions, marker selection and fitted cells across saved folds."""
    if tolerance <= 0 or len(manifest) < 2:
        raise ValueError("Positive tolerance and at least two fitted folds are required")
    records = {(r["repeat"], r["fold"]): r["outer_preprocessing"] for r in manifest}
    if len(records) != len(manifest):
        raise ValueError("Repeated fold IDs")
    fold_ids = sorted(records)
    genes = sorted(records[fold_ids[0]]["cluster_by_gene"])
    markers = sorted(set().union(*(set(r["selected_markers"]) for r in records.values())))
    labels = []
    for key in fold_ids:
        mapping = records[key]["cluster_by_gene"]
        if set(mapping) != set(genes):
            raise ValueError("Partitions use different gene sets")
        labels.append(np.array([mapping[g] for g in genes]))
    triu = np.triu_indices(len(genes), 1)
    coassignments = np.array([(v[:, None] == v[None, :])[triu] for v in labels])
    structural = []
    for i, j in combinations(range(len(fold_ids)), 2):
        a, b = records[fold_ids[i]], records[fold_ids[j]]
        sa, sb = set(a["selected_markers"]), set(b["selected_markers"])
        structural.append(
            dict(
                repeat_a=fold_ids[i][0],
                fold_a=fold_ids[i][1],
                repeat_b=fold_ids[j][0],
                fold_b=fold_ids[j][1],
                cluster_ari=adjusted_rand_score(labels[i], labels[j]),
                coassignment_agreement=float(np.mean(coassignments[i] == coassignments[j])),
                selected_marker_jaccard=jaccard(sa, sb),
                selected_marker_intersection=len(sa & sb),
                selected_marker_union=len(sa | sb),
                ld_marker_jaccard=jaccard(set(a["ld_markers"]), set(b["ld_markers"])),
            )
        )
    pairs = pd.DataFrame(structural)
    coassignment = pd.DataFrame(
        dict(
            gene_a=np.array(genes)[triu[0]],
            gene_b=np.array(genes)[triu[1]],
            times_same_cluster=coassignments.sum(axis=0),
            coassignment_frequency=coassignments.mean(axis=0),
            n_folds=len(fold_ids),
        )
    )
    parsed = {}
    for key, fields in coefficients.items():
        match = re.fullmatch(r"r(\d+)_f(\d+)_(.+)", key)
        if not match:
            raise ValueError(f"Unrecognized coefficient key: {key}")
        repeat, fold, method = int(match[1]), int(match[2]), match[3]
        expected = records[(repeat, fold)]["selected_markers"]
        if set(fields["markers"]) != set(expected):
            raise ValueError(f"Coefficient markers disagree with manifest: {key}")
        parsed.setdefault(method, {})[(repeat, fold)] = coefficient_states(
            **fields, marker_universe=markers, gene_universe=genes, tolerance=tolerance
        )
    fitted_pairs, frequencies, fit_counts = [], [], []
    for method, method_folds in sorted(parsed.items()):
        if set(method_folds) != set(fold_ids):
            raise ValueError(f"Missing folds for {method}")
        states = np.array([method_folds[key] for key in fold_ids])
        for i, j in combinations(range(len(fold_ids)), 2):
            fitted_pairs.append(
                dict(
                    method=method,
                    repeat_a=fold_ids[i][0],
                    fold_a=fold_ids[i][1],
                    repeat_b=fold_ids[j][0],
                    fold_b=fold_ids[j][1],
                    **compare_states(states[i], states[j]),
                )
            )
        for key, state in zip(fold_ids, states):
            fit_counts.append(
                dict(
                    method=method,
                    repeat=key[0],
                    fold=key[1],
                    absent_cells=int((state == ABSENT).sum()),
                    retained_zero_cells=int((state == 0).sum()),
                    positive_cells=int((state == 1).sum()),
                    negative_cells=int((state == -1).sum()),
                )
            )
        frames = dict(
            method=method,
            gene=np.repeat(genes, len(markers)),
            marker=np.tile(markers, len(genes)),
            n_folds=len(fold_ids),
        )
        for state, name in ((ABSENT, "absent"), (0, "zero"), (1, "positive"), (-1, "negative")):
            frames[f"n_{name}"] = (states == state).sum(axis=0).ravel()
        frequencies.append(pd.DataFrame(frames))
    fitted_pairs = pd.DataFrame(fitted_pairs)
    metric_columns = [
        "selection_jaccard",
        "selection_jaccard_common_markers",
        "direction_agreement_selected_both",
        "signed_overlap_over_selected_union",
    ]
    fitted_summary = fitted_pairs.groupby("method")[metric_columns].agg(["mean", "min", "max"])
    fitted_summary.columns = ["_".join(c) for c in fitted_summary.columns]
    partition_summary = {
        name: {
            "mean": float(pairs[name].mean()),
            "min": float(pairs[name].min()),
            "max": float(pairs[name].max()),
        }
        for name in (
            "cluster_ari",
            "coassignment_agreement",
            "selected_marker_jaccard",
            "ld_marker_jaccard",
        )
    }
    partition_summary.update(
        n_folds=len(fold_ids),
        n_fold_pairs=len(pairs),
        n_genes=len(genes),
        selected_marker_union=len(markers),
        never_selected_markers=len(set(raw_markers) - set(markers))
        if raw_markers is not None
        else None,
        n_raw_markers=len(raw_markers) if raw_markers is not None else None,
        stable_coassignment_fraction=float(
            (
                (coassignment.coassignment_frequency <= 1 / len(fold_ids))
                | (coassignment.coassignment_frequency >= 1 - 1 / len(fold_ids))
            ).mean()
        ),
    )
    return dict(
        partition_pairs=pairs,
        coassignment=coassignment,
        coefficient_pairs=fitted_pairs,
        coefficient_summary=fitted_summary.reset_index(),
        coefficient_frequencies=pd.concat(frequencies, ignore_index=True),
        coefficient_counts=pd.DataFrame(fit_counts),
        partition_summary=partition_summary,
    )


def remaining_ld(manifest, raw_x: pd.DataFrame) -> pd.DataFrame:
    """All-pair remaining marker correlation, computed only on each training fold."""
    rows = []
    for record in manifest:
        prep = record["outer_preprocessing"]
        x = raw_x.loc[prep["train_ids"], prep["selected_markers"]]
        corr = np.abs(x.corr().to_numpy())
        values = corr[np.triu_indices(len(corr), 1)]
        values = values[np.isfinite(values)]
        rows.append(
            dict(
                repeat=record["repeat"],
                fold=record["fold"],
                n_markers=x.shape[1],
                n_pairs=len(values),
                median=float(np.median(values)) if len(values) else np.nan,
                q90=float(np.quantile(values, 0.90)) if len(values) else np.nan,
                q95=float(np.quantile(values, 0.95)) if len(values) else np.nan,
                maximum=float(values.max()) if len(values) else np.nan,
                **{
                    f"fraction_above_{int(threshold * 100)}": float(np.mean(values > threshold))
                    if len(values)
                    else np.nan
                    for threshold in (0.5, 0.8, 0.9, 0.92)
                },
            )
        )
    return pd.DataFrame(rows)


def verify_predictions(predictions, repeats) -> dict:
    """Independent long-form aggregation (no pivot or validation metric helper)."""
    values = predictions.assign(
        squared_error=(predictions.actual - predictions.predicted) ** 2,
        baseline_error=(predictions.actual - predictions.training_mean) ** 2,
    )
    if values.duplicated(["repeat", "method", "sample", "gene"]).any():
        raise ValueError("Repeated out-of-fold predictions")
    totals = values.groupby(["repeat", "method", "gene"])[["squared_error", "baseline_error"]].sum()
    totals["r2"] = 1 - totals.squared_error / totals.baseline_error
    checks = []
    for (repeat, method), frame in totals.groupby(level=["repeat", "method"]):
        reported = repeats.loc[(repeats.repeat == repeat) & (repeats.method == method)].iloc[0]
        independent = dict(
            r2_gene_median=frame.r2.median(),
            r2_gene_mean=frame.r2.mean(),
            r2_pooled=1 - frame.squared_error.sum() / frame.baseline_error.sum(),
        )
        for key, value in independent.items():
            checks.append(abs(value - reported[key]))
    maximum = float(max(checks, default=0))
    if maximum > 1e-10:
        raise AssertionError(f"Independent R2 discrepancy {maximum}")
    return dict(independent_r2_maximum_difference=maximum, n_checked_metrics=len(checks))


def write_stability(tables: dict, destination: Path):
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in tables.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(destination / f"{name}.csv.gz", index=False, compression="gzip")
        else:
            (destination / f"{name}.json").write_text(json.dumps(value, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if any(part.lower().startswith("onedrive") for part in args.out.resolve().parts):
        parser.error("Outputs must be outside OneDrive")
    run = read_run(args.run)
    _, x, _ = validation.load_data()
    tables = stability_tables(run["manifest"], run["coefficients"], raw_markers=x.columns)
    tables["remaining_ld"] = remaining_ld(run["manifest"], x)
    tables["verification"] = verify_predictions(run["predictions"], run["repeats"])
    write_stability(tables, args.out)
    receipt = dict(
        source_run=str(args.run.resolve()),
        source_receipt_sha256=validation._sha256(args.run / "receipt.json"),
        code_sha256=validation._sha256(Path(__file__)),
        outputs={p.name: validation._sha256(p) for p in args.out.iterdir()},
        uncertainty=("Descriptive dispersion of overlapping splits; "
                     "not independent sampling uncertainty"),
        zero_tolerance=1e-6,
        units="training-standardized coefficients",
    )
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(tables["partition_summary"], indent=2), flush=True)
    print(tables["coefficient_summary"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
