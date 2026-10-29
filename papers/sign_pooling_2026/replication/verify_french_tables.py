"""Verify the FF report independently from saved raw coefficients and panel data.

Does not import the report producer or any fitting module. Bootstrap draws are
rebuilt with scalar transitions and recomputed using block multiplicities.
Cluster association uses pair enumeration; residual correlation uses explicit
correlation matrices, independently of the producer's summed-score identity.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .french_protocol import json_bytes, sha256
else:
    from french_protocol import json_bytes, sha256

MATCHED = ["matched__hcgl_singleton_exclude", "matched__hcgl_exclude", "matched__hcgl_unrestricted"]
GATED = {
    "singleton_exclude",
    "hcgl_singleton_exclude",
    "hcgl_exclude",
    "hcgl_unrestricted",
    "fcgl_exclude",
    *MATCHED,
}
KEYS = {
    "metrics": ["lookback", "method"],
    "portfolio_metrics": ["lookback", "method", "portfolio"],
    "weekly_losses": ["lookback", "origin", "method", "date"],
    "block_losses": ["lookback", "origin", "method"],
    "selections": ["lookback", "origin", "method"],
    "partitions": ["lookback", "origin", "method"],
    "cluster_sizes": ["lookback", "origin", "method", "cluster"],
    "coherence": ["lookback", "origin", "method", "factor"],
    "loading_stability": ["lookback", "origin", "method", "factor"],
    "support_levels": ["lookback", "origin", "method", "tolerance"],
    "support_stability": ["lookback", "origin", "method", "tolerance"],
    "gate_transitions": ["lookback", "origin", "method", "before", "after"],
    "gate_coverage": ["lookback", "origin", "method", "factor"],
    "gate_categories": ["lookback", "origin", "method", "factor", "axis", "category"],
    "coefficient_categories": ["lookback", "origin", "method", "factor", "axis", "category"],
    "gate_category_summary": ["lookback", "method", "factor", "axis", "category"],
    "coefficient_category_summary": ["lookback", "method", "factor", "axis", "category"],
    "boundaries": ["lookback", "method"],
    "contrasts": ["lookback", "mean_blocks", "kind", "method_a", "method_b", "metric"],
    "method_intervals": ["lookback", "mean_blocks", "method"],
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, context):
    if expected is None:
        ensure(pd.isna(actual), context + ": expected undefined")
    elif isinstance(expected, (bool, np.bool_)):
        value = actual if not isinstance(actual, str) else actual.lower() == "true"
        ensure(pd.notna(actual) and bool(value) == bool(expected), context + ": boolean")
    elif isinstance(expected, (float, int, np.number)) or expected == "inf":
        ensure(
            np.isclose(float(actual), float(expected), atol=2e-11, rtol=2e-10),
            context + ": numeric",
        )
    else:
        ensure(str(actual) == str(expected), context + ": text")


class Checker:
    def __init__(self, report):
        self.tables, self.rows, self.seen, self.values = {}, {}, {}, 0
        self.calendar = {}
        self.checked_columns = {name: set(keys) for name, keys in KEYS.items()}
        for name, keys in KEYS.items():
            table = pd.read_csv(report / (name + ".csv"), float_precision="round_trip")
            ensure(not table.duplicated(keys).any(), f"Duplicate report keys: {name}")
            self.tables[name] = table
            self.rows[name] = {tuple(row[k] for k in keys): row for row in table.to_dict("records")}
            self.seen[name] = set()

    def check(self, name, key, fields):
        key = tuple(key)
        ensure(key in self.rows[name], f"Missing {name} row: {key}")
        row = self.rows[name][key]
        fields = dict(fields)
        if "origin" in KEYS[name]:
            block, date = self.calendar[int(row["lookback"]), int(row["origin"])]
            fields["block"] = block
            if name != "weekly_losses":
                fields["date"] = date
        for field, expected in fields.items():
            ensure(field in row, f"Missing {name} column: {field}")
            equal(row[field], expected, f"{name}{key}/{field}")
            self.values += 1
            self.checked_columns[name].add(field)
        self.seen[name].add(key)

    def complete(self):
        for name in self.rows:
            ensure(self.seen[name] == set(self.rows[name]), f"Unverified/extra {name} rows")
            ensure(
                set(self.tables[name].columns) == self.checked_columns[name],
                f"Unverified/extra {name} columns",
            )


def pair_ari(a, b):
    both = same_a = same_b = pairs = 0
    for i in range(len(a)):
        for j in range(i):
            aa, bb = a[i] == a[j], b[i] == b[j]
            same_a += int(aa)
            same_b += int(bb)
            both += int(aa and bb)
            pairs += 1
    if not pairs:
        return 1.0
    chance = same_a * same_b / pairs
    scale = (same_a + same_b) / 2 - chance
    return 1.0 if not scale else (both - chance) / scale


def reference_settings(selection, p):
    c, e = selection["candidate"], selection["extension"]
    ref = p["grid"]["refinement"]
    f = list(p["grid"]["fractions"])
    cut = list(p["clustering"]["partitions"])
    tau = list(p["gate"]["thresholds"])
    if e["fraction"]:
        f += ref["fractions_if_smallest_positive_wins"]
    if e["partition"]:
        cut += ref["cuts_if_lowest_or_highest_finite_wins"]
    if e["tau"]:
        tau += ref["thresholds_if_highest_finite_wins"]
    finite_cut = sorted(v for v in cut if isinstance(v, (int, float)))
    finite_tau = sorted(v for v in tau if isinstance(v, (int, float)))
    ridge = sorted(v for v in p["grid"]["ridge_lambdas"] if isinstance(v, (int, float)))
    ff, cc, tt, rr = [c.get(k) for k in ("fraction", "partition", "tau", "ridge_lambda")]
    flags = {
        "fraction_zero": ff == 0,
        "fraction_one": ff == 1,
        "fraction_lowest_positive": ff == min(v for v in f if v > 0),
        "partition_singleton": cc == "singleton",
        "partition_one_cluster": cc == "one_cluster",
        "partition_finite_low": cc == finite_cut[0],
        "partition_finite_high": cc == finite_cut[-1],
        "tau_zero": tt == 0,
        "tau_infinite": tt == "inf",
        "tau_finite_high": tt == finite_tau[-1],
        "ridge_zero": rr == 0,
        "ridge_infinite": rr == "inf",
        "ridge_finite_low": rr == next(v for v in ridge if v > 0),
        "ridge_finite_high": rr == ridge[-1],
        **{"extended_" + k: v for k, v in e.items()},
    }
    for axis, value in (("fraction", ff), ("partition", cc), ("tau", tt), ("ridge", rr)):
        if value is None:
            for key in flags:
                if key.startswith(axis + "_") or key == "extended_" + axis:
                    flags[key] = None
    return {
        "fraction": ff,
        "partition": cc,
        "tau": tt,
        "ridge_lambda": rr,
        "lambda": selection["lambda"],
        "lambda_max": selection["lambda_max"],
        **flags,
    }, flags


def reference_diagnostics(xt, yt, xv, yv, labels, factors, saved):
    """Explicit marginal slopes, group residual correlation matrices and lag products."""
    sx, sy = xt.std(axis=0, ddof=0), yt.std(axis=0, ddof=0)
    a = (xt - np.mean(xt, axis=0)) / sx
    b = (yt - np.mean(yt, axis=0)) / sy
    av = (xv - np.mean(xv, axis=0)) / sx
    bv = (yv - np.mean(yv, axis=0)) / sy
    rows = []

    def sign(v):
        return int(v > 1e-14) - int(v < -1e-14)

    for j, factor in enumerate(factors):
        ct = np.array([np.dot(a[:, j], b[:, k]) / len(a) for k in range(b.shape[1])])
        cv = np.array([np.dot(av[:, j], bv[:, k]) / len(av) for k in range(bv.shape[1])])
        st, sv = np.array([sign(v) for v in ct]), np.array([sign(v) for v in cv])
        ta = va = n = tns = vns = nns = ps = 0
        lag_values = {lag: [] for lag in (1, 4, 13)}
        lag_weights = {lag: [] for lag in (1, 4, 13)}
        pair_correlations = []
        pair_weights = []
        saved_rows = (
            {(r["cluster"], r["factor"]): r for r in saved["by_cluster_factor"]}
            if saved is not None
            else {}
        )
        for group in sorted(set(labels)):
            members = np.where(labels == group)[0]
            k = len(members)
            pt, pv = sign(sum(ct[m] for m in members)), sign(sum(cv[m] for m in members))
            at, ag = sum(st[m] == pt for m in members), sum(sv[m] == pv for m in members)
            ta += at
            va += ag
            n += k
            ps += (pt == pv) * k
            if k > 1:
                tns += at
                vns += ag
                nns += k
            marginal_slope = np.dot(a[:, j], np.mean(b[:, members], axis=1)) / np.dot(
                a[:, j], a[:, j]
            )
            residual = b[:, members] - np.outer(a[:, j], np.full(k, marginal_slope))
            scores = np.array([a[d, j] * sum(residual[d]) for d in range(len(a))])
            scores = scores - np.mean(scores)
            denominator = sum(float(v * v) for v in scores)
            for lag in (1, 4, 13):
                value = (
                    sum(float(scores[d] * scores[d + lag]) for d in range(len(scores) - lag))
                    / denominator
                    if lag < len(scores) and denominator > 0
                    else None
                )
                if value is not None:
                    lag_values[lag].append(value)
                    lag_weights[lag].append(k)
                if saved is not None:
                    equal(
                        saved_rows[int(group), factor]["marginal_score_acf"][str(lag)],
                        value,
                        "Saved Stage 3 marginal-score ACF",
                    )
            good = residual.std(0) > 1e-12
            r = int(good.sum())
            corr = None
            if r > 1:
                matrix = np.corrcoef(residual[:, good], rowvar=False)
                corr = float(np.mean(matrix[np.triu_indices(r, 1)]))
                pair_correlations.append(corr)
                pair_weights.append(r * (r - 1) // 2)
            if saved is not None:
                equal(
                    saved_rows[int(group), factor]["mean_marginal_residual_correlation"],
                    corr,
                    "Saved Stage 3 residual correlation",
                )
        result = {
            "factor": factor,
            "train_agree": ta,
            "test_agree": va,
            "cells": n,
            "train_non_single_agree": tns,
            "test_non_single_agree": vns,
            "non_single_cells": nns,
            "pooled_same": ps,
            "train_agreement": ta / n,
            "heldout_agreement": va / n,
            "train_non_single_agreement": tns / nns if nns else None,
            "heldout_non_single_agreement": vns / nns if nns else None,
            "pooled_train_test_agreement": ps / n,
            "member_train_test_agreement": float(np.mean(st == sv)),
            "within_group_residual_correlation": float(
                np.average(pair_correlations, weights=pair_weights)
            )
            if pair_weights
            else None,
            "residual_correlation_pairs": sum(pair_weights),
        }
        for lag in (1, 4, 13):
            values, weights = lag_values[lag], lag_weights[lag]
            result[f"score_acf_{lag}"] = (
                float(np.average(values, weights=weights)) if weights else None
            )
            result[f"score_abs_acf_{lag}"] = (
                float(np.average(np.abs(values), weights=weights)) if weights else None
            )
        rows.append(result)
    return rows


def verify(panel, fit_run, report):
    p = read(panel / "protocol.json")
    manifest, produced = read(fit_run / "run_manifest.json"), read(report / "report_manifest.json")
    cert, fit_receipt = (
        read(fit_run / "full_validation.json"),
        read(fit_run / "full_run_receipt.json"),
    )
    ensure(cert["status"] == fit_receipt["status"] == "complete", "Incomplete estimation")
    ensure(
        produced["fit_identity"]
        == manifest["identity"]
        == cert["identity"]
        == fit_receipt["identity"],
        "Mixed report/fit identities",
    )
    ensure(
        sha256(fit_run / "full_validation.json")
        == produced["stage3_certificate_sha256"]
        == fit_receipt["validation_sha256"],
        "Certificate hash differs",
    )
    ensure(
        sha256(panel / "manifest.json") == produced["panel_manifest_sha256"]
        and sha256(panel / "protocol.json") == produced["protocol_sha256"],
        "Panel identity differs",
    )
    files = {
        v.relative_to(report).as_posix()
        for v in report.rglob("*")
        if v.is_file() and v.name not in ("report_manifest.json", "verification.json")
    }
    ensure(files == set(produced["files_sha256"]), "Missing/extra report artifacts")
    for name, expected in produced["files_sha256"].items():
        ensure(sha256(report / name) == expected, f"Changed report artifact: {name}")
    if produced.get("source_snapshots"):
        for name, expected in produced["source_sha256"].items():
            ensure(
                sha256(report / "source" / name) == expected, "Changed reporting source snapshot"
            )
    pm = read(panel / "manifest.json")["files_sha256"]
    for name in ("panel/X_weekly.csv", "panel/Y_weekly.csv", "panel/portfolio_labels.csv"):
        ensure(sha256(panel / name) == pm[name], "Changed raw reporting input")
    x, y = [
        pd.read_csv(
            panel / ("panel/" + name + "_weekly.csv"), index_col=0, float_precision="round_trip"
        )
        for name in ("X", "Y")
    ]
    x.index, y.index = pd.to_datetime(x.index), pd.to_datetime(y.index)
    ensure(x.index.equals(y.index) and len(x) == p["calendar"]["weekly_rows"], "Different calendar")
    categories = (
        pd.read_csv(panel / "panel/portfolio_labels.csv").set_index("portfolio").loc[y.columns]
    )
    ck = Checker(report)
    methods = p["methods"] + MATCHED
    origins = list(range(p["calendar"]["outer_burn_in"], len(x), p["calendar"]["outer_horizon"]))
    expected_jobs = {(o, t) for o in origins for t in p["calendar"]["lookbacks"]}
    ensure(
        {(r["origin"], r["lookback"]) for r in manifest["jobs"]} == expected_jobs
        and len(manifest["jobs"]) == len(expected_jobs),
        "Missing/duplicate fit jobs",
    )
    certified = {(a["origin"], a["lookback"]): a for a in cert["audits"]}
    ensure(
        set(certified) == expected_jobs and len(cert["audits"]) == len(expected_jobs),
        "Missing/duplicate certified windows",
    )
    summaries, baseline_blocks = {}, {}
    sums_by_t, counts_by_t, weekly_by_t, settings_records = {}, {}, {}, []
    category_totals = {"gate_category_summary": {}, "coefficient_category_summary": {}}
    for t in p["calendar"]["lookbacks"]:
        errors_by_method = {m: [] for m in methods}
        weekly_by_method = {m: [] for m in methods}
        baseline_blocks[t] = []
        block_sums, counts, previous = [], [], {}
        for block, o in enumerate(origins):
            ck.calendar[t, o] = (block, x.index[o].strftime("%Y-%m-%d"))
            folder = fit_run / f"origin_{o}_T{t}"
            wr = read(folder / "window_receipt.json")
            ensure(
                sha256(folder / "window_receipt.json") == certified[o, t]["window_receipt_sha256"],
                "Different window receipt",
            )
            for name in (
                "outer_arrays.npz",
                "selections.json",
                "outer_preprocessing.json",
                "training_diagnostics.json",
            ):
                value = sha256(folder / name)
                ensure(
                    value
                    == wr["files_sha256"][name]
                    == produced["report_inputs_sha256"][folder.name + "/" + name],
                    "Changed source artifact",
                )
            selected, saved_diag = (
                read(folder / "selections.json"),
                read(folder / "training_diagnostics.json"),
            )
            with np.load(folder / "outer_arrays.npz", allow_pickle=False) as saved:
                arrays = {key: saved[key].copy() for key in saved.files}
            end = min(o + p["calendar"]["outer_horizon"], len(x))
            xt, yt = x.iloc[o - t : o].to_numpy(), y.iloc[o - t : o].to_numpy()
            xv, yv = x.iloc[o:end].to_numpy(), y.iloc[o:end].to_numpy()
            ym, ys, xm, xs = yt.mean(0), yt.std(0), xt.mean(0), xt.std(0)
            np.testing.assert_array_equal(arrays["actual"], yv)
            baseline = yv - ym
            baseline_blocks[t].append(baseline)
            losses_this_block, diag_cache = [], {}
            for method in methods:
                key = (t, o, method)
                beta = arrays[method + "__beta"]
                alpha = arrays[method + "__intercept"] if method in p["methods"] else ym - beta @ xm
                predicted = np.column_stack([xv @ beta[q] + alpha[q] for q in range(len(beta))])
                np.testing.assert_allclose(
                    predicted, arrays[method + "__prediction"], atol=1e-12, rtol=1e-12
                )
                residual = yv - predicted
                norm = residual / ys
                loss = np.einsum("ni,ni->n", norm, norm) / len(ys)
                errors_by_method[method].append(residual)
                weekly_by_method[method].append(loss)
                losses_this_block.append(loss.sum())
                ck.check(
                    "block_losses",
                    key,
                    {
                        "weeks": len(yv),
                        "standardized_loss_sum": loss.sum(),
                        "raw_squared_error_sum": np.sum(residual * residual),
                        "baseline_squared_error_sum": np.sum(baseline * baseline),
                    },
                )
                for date, value in zip(x.index[o:end], loss):
                    ck.check(
                        "weekly_losses",
                        (*key, date.strftime("%Y-%m-%d")),
                        {"standardized_mse": value},
                    )
                std = np.diag(1 / ys) @ beta @ np.diag(xs)
                labels = arrays[
                    (method if method in p["methods"] else "hcgl_unconstrained") + "__labels"
                ]
                gates = arrays[method + "__gates"]
                groups = Counter(labels)
                fields = {
                    "clusters": len(groups),
                    "singleton_response_fraction": sum(v for v in groups.values() if v == 1)
                    / len(labels),
                    "min_cluster_size": min(groups.values()),
                    "max_cluster_size": max(groups.values()),
                    "ari_size": pair_ari(labels, categories.size_decile.to_numpy()),
                    "ari_bm": pair_ari(labels, categories.bm_decile.to_numpy()),
                    "pooling_contrast_available": len(groups) < len(labels),
                }
                ck.check("partitions", key, fields)
                for label, size in groups.items():
                    ck.check("cluster_sizes", (*key, label), {"size": size})
                if method in p["methods"]:
                    fields, flags = reference_settings(selected[method], p)
                    ck.check("selections", key, fields)
                    settings_records.append((t, method, flags))
                if tuple(labels) not in diag_cache:
                    diag_cache[tuple(labels)] = reference_diagnostics(
                        xt, yt, xv, yv, labels, list(x.columns), saved_diag.get(method)
                    )
                for fields in diag_cache[tuple(labels)]:
                    ck.check("coherence", (*key, fields["factor"]), fields)
                for tol in p["reporting"]["support_tolerances_standardized"]:
                    active = np.abs(std) > tol
                    ck.check(
                        "support_levels",
                        (*key, tol),
                        {
                            "cells": active.size,
                            "active": active.sum(),
                            "active_fraction": active.mean(),
                        },
                    )
                if method in previous:
                    old_beta, old_std, old_gates = previous[method]
                    for j, factor in enumerate(x.columns):
                        changes = beta[:, j] - old_beta[:, j]
                        ck.check(
                            "loading_stability",
                            (*key, factor),
                            {
                                "mean_absolute_change": np.mean(np.abs(changes)),
                                "root_mean_square_change": np.linalg.norm(changes)
                                / np.sqrt(len(changes)),
                            },
                        )
                    for tol in p["reporting"]["support_tolerances_standardized"]:
                        new, old = np.abs(std) > tol, np.abs(old_std) > tol
                        intersection, union = np.logical_and(new, old), np.logical_or(new, old)
                        ck.check(
                            "support_stability",
                            (*key, tol),
                            {
                                "cells": new.size,
                                "active": new.sum(),
                                "entered": np.sum(new & ~old),
                                "exited": np.sum(old & ~new),
                                "changed_fraction": np.mean(new != old),
                                "jaccard": intersection.sum() / union.sum() if union.sum() else 1.0,
                                "common_active_sign_flip_fraction": np.mean(
                                    np.sign(std[intersection]) != np.sign(old_std[intersection])
                                )
                                if intersection.any()
                                else None,
                            },
                        )
                    if method in GATED:
                        for before in (-1, 0, 1, 2):
                            for after in (-1, 0, 1, 2):
                                old_mask = (
                                    np.isnan(old_gates) if before == 2 else old_gates == before
                                )
                                new_mask = np.isnan(gates) if after == 2 else gates == after
                                ck.check(
                                    "gate_transitions",
                                    (*key, before, after),
                                    {"cells": gates.size, "count": np.sum(old_mask & new_mask)},
                                )
                previous[method] = beta, std, gates
                if method in GATED:
                    for j, factor in enumerate(x.columns):
                        g = gates[:, j]
                        ck.check(
                            "gate_coverage",
                            (*key, factor),
                            {
                                "cells": len(g),
                                "positive": np.sum(g == 1),
                                "negative": np.sum(g == -1),
                                "excluded": np.sum(g == 0),
                                "unrestricted_failed": np.isnan(g).sum(),
                            },
                        )
                        if factor not in ("SMB", "HML"):
                            continue
                        for axis in ("size_decile", "bm_decile"):
                            for category in sorted(categories[axis].unique()):
                                mask = categories[axis].to_numpy() == category
                                gv, bv = gates[mask, j], std[mask, j]
                                cat_key = (*key, factor, axis, category)
                                counts_gate = {
                                    "cells": mask.sum(),
                                    "positive": np.sum(gv == 1),
                                    "negative": np.sum(gv == -1),
                                    "excluded": np.sum(gv == 0),
                                    "unrestricted_failed": np.isnan(gv).sum(),
                                }
                                counts_coef = {
                                    "cells": mask.sum(),
                                    "positive": np.sum(bv > 1e-6),
                                    "negative": np.sum(bv < -1e-6),
                                    "near_zero": np.sum(np.abs(bv) <= 1e-6),
                                }
                                ck.check("gate_categories", cat_key, counts_gate)
                                ck.check(
                                    "coefficient_categories",
                                    cat_key,
                                    {"tolerance": 1e-6, **counts_coef},
                                )
                                for table, values in (
                                    ("gate_category_summary", counts_gate),
                                    ("coefficient_category_summary", counts_coef),
                                ):
                                    summary_key = (t, method, factor, axis, category)
                                    total = category_totals[table].setdefault(
                                        summary_key, Counter()
                                    )
                                    total.update({k: int(v) for k, v in values.items()})
            block_sums.append(losses_this_block)
            counts.append(end - o)
        sums_by_t[t], counts_by_t[t] = np.asarray(block_sums), np.asarray(counts)
        weekly_by_t[t] = np.column_stack([np.concatenate(weekly_by_method[m]) for m in methods])
        base = np.concatenate(baseline_blocks[t])
        for method in methods:
            residual = np.concatenate(errors_by_method[method])
            loss = np.concatenate(weekly_by_method[method])
            sse, sst = np.einsum("ni,ni->i", residual, residual), np.einsum("ni,ni->i", base, base)
            values = 1 - sse / sst
            metrics = {
                "weeks": len(loss),
                "standardized_mse": sum(loss) / len(loss),
                "standardized_rmse": np.sqrt(sum(loss) / len(loss)),
                "raw_rmse": np.linalg.norm(residual) / np.sqrt(residual.size),
                "pooled_r2": 1 - sum(sse) / sum(sst),
                "mean_portfolio_r2": sum(values) / len(values),
                "median_portfolio_r2": np.median(values),
                "q25_portfolio_r2": np.quantile(values, 0.25),
                "q75_portfolio_r2": np.quantile(values, 0.75),
            }
            ck.check("metrics", (t, method), metrics)
            summaries[t, method] = metrics
            for q, name in enumerate(y.columns):
                ck.check(
                    "portfolio_metrics",
                    (t, method, name),
                    {"raw_sse": sse[q], "baseline_sse": sst[q], "r2": values[q]},
                )
    for name, values in category_totals.items():
        for key, count in values.items():
            fields = dict(count)
            fields.update(
                {
                    state + "_fraction": value / count["cells"]
                    for state, value in count.items()
                    if state != "cells"
                }
            )
            ck.check(name, key, fields)
    for t in p["calendar"]["lookbacks"]:
        for method in p["methods"]:
            flags = [f for tt, mm, f in settings_records if (tt, mm) == (t, method)]
            fields = {}
            for name in flags[0]:
                eligible = [f[name] for f in flags if f[name] is not None]
                fields[name] = sum(eligible) / len(eligible) if eligible else None
            ck.check("boundaries", (t, method), fields)
    verify_bootstrap(report, p, sums_by_t, counts_by_t, weekly_by_t, summaries, methods, ck)
    verify_latex(report, p, ck, summaries)
    ck.complete()
    result = {
        "status": "passed",
        "fit_identity": manifest["identity"],
        "report_manifest_sha256": sha256(report / "report_manifest.json"),
        "verifier_source_sha256": sha256(Path(__file__)),
        "numeric_values_verified": ck.values,
        "report_tables": len(KEYS),
        "windows": len(expected_jobs),
        "metrics_recomputed_from_raw_coefficients": True,
        "bootstrap_recomputed_by_multiplicities": True,
        "all_diagnostic_and_printed_values_verified": True,
        "refitting_performed": False,
    }
    (report / "verification.json").write_bytes(json_bytes(result))
    return result


def verify_bootstrap(report, p, sums_by_t, counts_by_t, weekly_by_t, summaries, methods, ck):
    cfg = p["reporting"]["bootstrap"]
    main = [p["reporting"]["primary_contrast"]] + p["reporting"]["other_contrasts"]
    matched = [[m, "hcgl_unconstrained"] for m in MATCHED] + [[MATCHED[1], MATCHED[0]]]
    with (
        np.load(report / "report_data.npz", allow_pickle=False) as bundle,
        np.load(report / "bootstrap_samples.npz", allow_pickle=False) as saved,
    ):
        expected_bundle, expected_samples = set(), set()
        for t in p["calendar"]["lookbacks"]:
            sums, counts = sums_by_t[t], counts_by_t[t]
            np.testing.assert_allclose(bundle[f"T{t}__block_sums"], sums, atol=1e-10, rtol=1e-12)
            np.testing.assert_array_equal(bundle[f"T{t}__block_counts"], counts)
            np.testing.assert_allclose(
                bundle[f"T{t}__weekly_losses"], weekly_by_t[t], atol=1e-11, rtol=1e-11
            )
            expected_bundle.update(
                f"T{t}__" + suffix for suffix in ("block_sums", "block_counts", "weekly_losses")
            )
            n, b = len(counts), cfg["resamples"]
            mean = cfg["mean_blocks"][str(t)]
            for length in sorted((mean / 2, mean, 2 * mean)):
                rng = np.random.default_rng(cfg["seed"])
                starts = rng.integers(0, n, size=(b, n))
                uniforms = rng.random((b, n))
                indices = np.empty_like(starts)
                for draw in range(b):
                    indices[draw, 0] = starts[draw, 0]
                    for step in range(1, n):
                        indices[draw, step] = (
                            starts[draw, step]
                            if uniforms[draw, step] < 1 / length
                            else (indices[draw, step - 1] + 1) % n
                        )
                tag = f"T{t}_L{length:g}"
                expected_samples.update((tag + "__indices", tag + "__mse"))
                np.testing.assert_array_equal(saved[tag + "__indices"], indices)
                frequencies = np.array([np.bincount(path, minlength=n) for path in indices])
                boot = (frequencies @ sums) / (frequencies @ counts)[:, None]
                np.testing.assert_allclose(saved[tag + "__mse"], boot, atol=1e-11, rtol=1e-11)
                for j, method in enumerate(methods):
                    lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
                    slo, shi = np.percentile(np.sqrt(boot[:, j]), [2.5, 97.5])
                    ck.check(
                        "method_intervals",
                        (t, length, method),
                        {
                            "primary_length": length == mean,
                            "mse_lower": lo,
                            "mse_upper": hi,
                            "srmse_lower": slo,
                            "srmse_upper": shi,
                        },
                    )
                for kind, pairs in (("primary", main), ("matched", matched)):
                    for a, c in pairs:
                        ia, ib = methods.index(a), methods.index(c)
                        for metric in ("mse", "srmse"):
                            if metric == "mse":
                                difference = boot[:, ia] - boot[:, ib]
                                delta = (
                                    summaries[t, a]["standardized_mse"]
                                    - summaries[t, c]["standardized_mse"]
                                )
                            else:
                                difference = np.sqrt(boot[:, ia]) - np.sqrt(boot[:, ib])
                                delta = (
                                    summaries[t, a]["standardized_rmse"]
                                    - summaries[t, c]["standardized_rmse"]
                                )
                            lo, hi = np.percentile(difference, [2.5, 97.5])
                            ck.check(
                                "contrasts",
                                (t, length, kind, a, c, metric),
                                {
                                    "primary_length": length == mean,
                                    "difference": delta,
                                    "lower": lo,
                                    "upper": hi,
                                },
                            )
        ensure(set(bundle.files) == expected_bundle, "Missing/extra reporting arrays")
        ensure(set(saved.files) == expected_samples, "Missing/extra bootstrap arrays")


def verify_latex(report, p, ck, summaries):
    for t in p["calendar"]["lookbacks"]:
        lines = [
            r
            for r in (report / f"table_prediction_T{t}.tex").read_text().splitlines()
            if " & " in r
        ][1:]
        ensure(len(lines) == len(p["methods"]), "Missing printed prediction rows")
        for method, line in zip(p["methods"], lines):
            ensure(line.endswith(chr(92) * 2), "Invalid TeX row terminator")
            cells = line.removesuffix(chr(92) * 2).split(" & ")[1:]
            expected = [
                summaries[t, method][k]
                for k in ("standardized_mse", "standardized_rmse", "raw_rmse", "pooled_r2")
            ]
            ensure(
                [c.strip() for c in cells] == [f"{v:.6f}" for v in expected],
                "Wrong printed metrics",
            )
        printed = [
            r for r in (report / f"table_contrasts_T{t}.tex").read_text().splitlines() if " & " in r
        ][1:]
        values = ck.tables["contrasts"]
        values = values[(values.lookback == t) & values.primary_length & (values.kind == "primary")]
        ensure(len(printed) == len(values), "Missing printed contrast rows")
        for line, row in zip(printed, values.itertuples()):
            ensure(line.endswith(chr(92) * 2), "Invalid contrast TeX row")
            numbers = line.removesuffix(chr(92) * 2).split(" & ")[2:]
            ensure(
                [c.strip() for c in numbers]
                == [f"{v:.6f}" for v in (row.difference, row.lower, row.upper)],
                "Wrong printed contrasts",
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel-run", type=Path, required=True)
    parser.add_argument("--fit-run", type=Path, required=True)
    parser.add_argument("--report-run", type=Path, required=True)
    args = parser.parse_args()
    print(json_bytes(verify(args.panel_run, args.fit_run, args.report_run)).decode())


if __name__ == "__main__":
    main()
