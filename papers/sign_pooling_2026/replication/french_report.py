"""Independent, no-refit scientific report for the frozen French portfolio run.

Politis and Romano (1994), The Stationary Bootstrap, JASA 89:1303-1313,
doi:10.1080/01621459.1994.10476870: circular, geometrically restarted block
indices, applied here to ordered outer-block sums/counts with shared method
draws. These conditional historical intervals do not calibrate sign gates.
Hubert and Arabie (1985), Comparing partitions, Journal of Classification
2:193-218, doi:10.1007/BF01908075: adjusted Rand association with categories.
Categories and descriptive marginal directions are not true partial signs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

if __package__:
    from .french_protocol import REPOSITORY, external_path, json_bytes, sha256
else:
    from french_protocol import REPOSITORY, external_path, json_bytes, sha256

MATCHED = ["matched__hcgl_singleton_exclude", "matched__hcgl_exclude", "matched__hcgl_unrestricted"]
GATED = {
    "singleton_exclude",
    "hcgl_singleton_exclude",
    "hcgl_exclude",
    "hcgl_unrestricted",
    "fcgl_exclude",
    *MATCHED,
}
LABELS = {
    "lasso": "Lasso",
    "singleton_exclude": "Singleton Lasso",
    "hcgl_unconstrained": "HCGL unconstrained",
    "hcgl_singleton_exclude": "HCGL singleton gate",
    "hcgl_exclude": "HCGL pooled exclusion",
    "hcgl_unrestricted": "HCGL pooled release",
    "fcgl_unconstrained": "FCGL unconstrained",
    "fcgl_exclude": "FCGL pooled exclusion",
    "ols": "OLS",
    "ridge": "Ridge",
    "cooperative": "Cooperative Lasso",
    "unilasso": "UniLasso",
    **{m: m.replace("matched__hcgl_", "Matched ") for m in MATCHED},
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frame(path):
    value = pd.read_csv(path, index_col=0, float_precision="round_trip")
    value.index = pd.to_datetime(value.index)
    return value


def conventions(protocol):
    """Report choices fixed before any outer performance summaries are viewed."""
    return {
        "confidence_level": 0.95,
        "interval": "two-sided percentile; linear quantiles",
        "bootstrap": protocol["reporting"]["bootstrap"],
        "bootstrap_rng": "default_rng(seed); per length reset; sorted lengths; common methods",
        "bootstrap_unit": "outer-block sums and week counts; circular continuation",
        "coherence_sign_zero_tolerance": 1e-14,
        "coherence": "member marginal direction vs pooled training-SD-scaled group direction",
        "heldout_coherence": "center within test block; scale by training SD; descriptive only",
        "gate_category_unit": "one fitted portfolio-factor cell per origin; equal origin weights",
        "loading_changes": "raw coefficient differences; no estimation-accuracy interpretation",
        "support_thresholds": protocol["reporting"]["support_tolerances_standardized"],
        "empty_support_jaccard": 1.0,
        "undefined_diagnostics": "blank, never zero-imputed",
        "matched_contrasts": [[m, "hcgl_unconstrained"] for m in MATCHED]
        + [["matched__hcgl_exclude", "matched__hcgl_singleton_exclude"]],
        "no_refitting_or_outer_score_protocol_changes": True,
    }


def load_run(panel, fit_run):
    """Authenticate the Stage 3 certificate and each reporting input, without fitting."""
    p = read_json(panel / "protocol.json")
    manifest = read_json(fit_run / "run_manifest.json")
    cert = read_json(fit_run / "full_validation.json")
    receipt = read_json(fit_run / "full_run_receipt.json")
    binding = manifest["binding"]
    require(cert["status"] == receipt["status"] == "complete", "Incomplete Stage 3 certificate")
    require(
        manifest["identity"] == digest(binding) == cert["identity"] == receipt["identity"],
        "Mixed estimation identities",
    )
    require(
        sha256(fit_run / "full_validation.json") == receipt["validation_sha256"],
        "Corrupted Stage 3 certificate",
    )
    require(
        sha256(panel / "protocol.json") == binding["protocol_sha256"]
        and sha256(panel / "manifest.json") == binding["panel_manifest_sha256"],
        "Different protocol or panel manifest",
    )
    panel_manifest = read_json(panel / "manifest.json")
    for name in ("panel/X_weekly.csv", "panel/Y_weekly.csv", "panel/portfolio_labels.csv"):
        require(
            sha256(panel / name) == panel_manifest["files_sha256"][name],
            "Changed panel reporting input",
        )
    # The Stage 1 verifier owns the complete raw-input lineage. Report inputs
    # below are additionally bound by the weekly panel's manifest hashes.
    x, y = frame(panel / "panel/X_weekly.csv"), frame(panel / "panel/Y_weekly.csv")
    require(
        x.index.equals(y.index) and x.index.is_unique and x.index.is_monotonic_increasing,
        "Invalid panel calendar",
    )
    require(
        len(x) == p["calendar"]["weekly_rows"]
        and y.shape[1] == p["responses"]
        and list(x.columns) == p["predictors"],
        "Panel dimensions/columns differ",
    )
    require(np.isfinite(x.to_numpy()).all() and np.isfinite(y.to_numpy()).all(), "Nonfinite panel")
    labels = pd.read_csv(panel / "panel/portfolio_labels.csv").set_index("portfolio").loc[y.columns]
    for name, value in binding["source_sha256"].items():
        require(sha256(fit_run / "source" / name) == value, "Corrupted estimation source snapshot")
    origins = list(range(p["calendar"]["outer_burn_in"], len(x), p["calendar"]["outer_horizon"]))
    expected = [
        {"origin": o, "lookback": t, "folder": f"origin_{o}_T{t}"}
        for o in origins
        for t in p["calendar"]["lookbacks"]
    ]
    require(
        manifest["jobs"] == expected and len(origins) == p["calendar"]["origins"],
        "Incomplete/duplicate origin inventory",
    )
    require(
        {r["folder"] for r in expected}
        == {f.parent.name for f in fit_run.glob("origin_*/window_receipt.json")},
        "Missing/extra completed windows",
    )
    certified = {(a["origin"], a["lookback"]): a for a in cert["audits"]}
    require(
        len(certified) == len(cert["audits"]) == len(expected), "Duplicate/missing audit records"
    )
    windows, hashes = [], {}
    needed = [
        "outer_arrays.npz",
        "selections.json",
        "outer_preprocessing.json",
        "training_diagnostics.json",
    ]
    for job in expected:
        folder = fit_run / job["folder"]
        wr = read_json(folder / "window_receipt.json")
        require(
            sha256(folder / "window_receipt.json")
            == certified[job["origin"], job["lookback"]]["window_receipt_sha256"],
            "Changed window receipt",
        )
        require(
            wr["identity"] == manifest["identity"] and wr["status"] == "complete",
            "Mixed/incomplete window",
        )
        for name in needed:
            value = sha256(folder / name)
            require(value == wr["files_sha256"][name], f"Corrupted reporting input: {name}")
            hashes[job["folder"] + "/" + name] = value
        with np.load(folder / "outer_arrays.npz", allow_pickle=False) as saved:
            arrays = {k: saved[k].copy() for k in saved.files}
        o, t = job["origin"], job["lookback"]
        end = min(o + p["calendar"]["outer_horizon"], len(x))
        require(wr["test_dates"] == x.index[o:end].strftime("%Y-%m-%d").tolist(), "Misdated test")
        np.testing.assert_array_equal(arrays["actual"], y.iloc[o:end].to_numpy())
        pre = read_json(folder / "outer_preprocessing.json")
        for key, raw in (("x", x.iloc[o - t : o].to_numpy()), ("y", y.iloc[o - t : o].to_numpy())):
            np.testing.assert_allclose(pre[key + "_mean"], raw.mean(0), atol=1e-15, rtol=1e-13)
            np.testing.assert_allclose(pre[key + "_sd"], raw.std(0), atol=1e-15, rtol=1e-13)
        np.testing.assert_array_equal(arrays["training_mean"], pre["y_mean"])
        np.testing.assert_array_equal(arrays["training_sd"], pre["y_sd"])
        selections = read_json(folder / "selections.json")
        require(set(selections) == set(p["methods"]), "Missing selected methods")
        windows.append(
            {
                **job,
                "end": end,
                "arrays": arrays,
                "pre": pre,
                "selections": selections,
                "diagnostics": read_json(folder / needed[-1]),
            }
        )
    return {
        "protocol": p,
        "manifest": manifest,
        "certificate": cert,
        "panel_manifest": panel_manifest,
        "x": x,
        "y": y,
        "labels": labels,
        "windows": windows,
        "input_hashes": hashes,
    }


def stationary_indices(blocks, resamples, mean_blocks, seed):
    """Stationary circular Markov resampling; one index path shared by all methods."""
    require(blocks > 0 and resamples > 0 and mean_blocks >= 1, "Invalid bootstrap dimensions")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, blocks, size=(resamples, blocks))
    restart = rng.random((resamples, blocks)) < 1 / mean_blocks
    indices = starts.copy()
    for k in range(1, blocks):
        indices[:, k] = np.where(restart[:, k], starts[:, k], (indices[:, k - 1] + 1) % blocks)
    return indices


def adjusted_rand(a, b):
    """Chance-adjusted pair-count index; categories are descriptive references."""
    _, aa = np.unique(a, return_inverse=True)
    _, bb = np.unique(b, return_inverse=True)
    n = len(aa)
    require(n == len(bb), "ARI lengths differ")
    if n < 2:
        return 1.0
    table = np.zeros((aa.max() + 1, bb.max() + 1), dtype=np.int64)
    np.add.at(table, (aa, bb), 1)

    def pairs(v):
        return float(np.sum(v * (v - 1) / 2))

    nij, ni, nj = pairs(table), pairs(table.sum(1)), pairs(table.sum(0))
    expected = ni * nj / (n * (n - 1) / 2)
    denominator = 0.5 * (ni + nj) - expected
    return 1.0 if denominator == 0 else (nij - expected) / denominator


def selected_settings(selection, protocol):
    c, extension = selection["candidate"], selection["extension"]
    grid, refinement = protocol["grid"], protocol["grid"]["refinement"]
    fractions = grid["fractions"] + (
        refinement["fractions_if_smallest_positive_wins"] if extension["fraction"] else []
    )
    cuts = protocol["clustering"]["partitions"] + (
        refinement["cuts_if_lowest_or_highest_finite_wins"] if extension["partition"] else []
    )
    taus = protocol["gate"]["thresholds"] + (
        refinement["thresholds_if_highest_finite_wins"] if extension["tau"] else []
    )
    finite_cuts = [v for v in cuts if isinstance(v, (float, int))]
    finite_taus = [v for v in taus if isinstance(v, (float, int))]
    fraction, cut, tau, ridge = (c.get(k) for k in ("fraction", "partition", "tau", "ridge_lambda"))
    result = {
        "fraction": fraction,
        "partition": cut,
        "tau": tau,
        "ridge_lambda": ridge,
        "lambda": selection["lambda"],
        "lambda_max": selection["lambda_max"],
        "fraction_zero": fraction == 0,
        "fraction_one": fraction == 1,
        "fraction_lowest_positive": fraction == min(v for v in fractions if v > 0),
        "partition_singleton": cut == "singleton",
        "partition_one_cluster": cut == "one_cluster",
        "partition_finite_low": cut is not None and cut == min(finite_cuts),
        "partition_finite_high": cut is not None and cut == max(finite_cuts),
        "tau_zero": tau == 0,
        "tau_infinite": tau == "inf",
        "tau_finite_high": tau is not None and tau == max(finite_taus),
        "ridge_zero": ridge == 0,
        "ridge_infinite": ridge == "inf",
        "ridge_finite_low": ridge
        == min(v for v in grid["ridge_lambdas"] if isinstance(v, (float, int)) and v > 0),
        "ridge_finite_high": ridge
        == max(v for v in grid["ridge_lambdas"] if isinstance(v, (float, int))),
        **{"extended_" + k: v for k, v in extension.items()},
    }
    for axis, value in (("fraction", fraction), ("partition", cut), ("tau", tau), ("ridge", ridge)):
        if value is None:
            for key in result:
                if key.startswith(axis + "_") and key != "ridge_lambda":
                    result[key] = None
            if "extended_" + axis in result:
                result["extended_" + axis] = None
    return result


def group_diagnostics(x_train, y_train, x_test, y_test, labels, factors, pre):
    """Descriptive directions and marginal-score dependence; no null/sign accuracy."""
    xs, ys = np.asarray(pre["x_sd"]), np.asarray(pre["y_sd"])
    train_x = (x_train - x_train.mean(0)) / xs
    train_y = (y_train - y_train.mean(0)) / ys
    test_x = (x_test - x_test.mean(0)) / xs
    test_y = (y_test - y_test.mean(0)) / ys
    train_cov, test_cov = train_y.T @ train_x / len(train_x), test_y.T @ test_x / len(test_x)

    def signs(v):
        return np.where(np.abs(v) <= 1e-14, 0, np.sign(v))

    st, sv = signs(train_cov), signs(test_cov)
    rows = []
    for j, factor in enumerate(factors):
        totals = {
            "train_agree": 0,
            "test_agree": 0,
            "cells": 0,
            "train_non_single_agree": 0,
            "test_non_single_agree": 0,
            "non_single_cells": 0,
            "pooled_same": 0,
        }
        corr_sum, corr_pairs = 0.0, 0
        acfs, sizes = [], []
        for label in np.unique(labels):
            members = np.flatnonzero(labels == label)
            size = len(members)
            pt, pv = signs(train_cov[members, j].sum()), signs(test_cov[members, j].sum())
            ta, va = int(np.sum(st[members, j] == pt)), int(np.sum(sv[members, j] == pv))
            totals["train_agree"] += ta
            totals["test_agree"] += va
            totals["cells"] += size
            totals["pooled_same"] += int(pt == pv) * size
            if size > 1:
                totals["train_non_single_agree"] += ta
                totals["test_non_single_agree"] += va
                totals["non_single_cells"] += size
            slope = train_y[:, members].mean(1) @ train_x[:, j] / np.square(train_x[:, j]).sum()
            residual = train_y[:, members] - train_x[:, j, None] * slope
            score = train_x[:, j] * residual.sum(1)
            score -= score.mean()
            denom = score @ score
            acfs.append(
                [
                    float(score[:-lag] @ score[lag:] / denom)
                    if lag < len(score) and denom > 0
                    else np.nan
                    for lag in (1, 4, 13)
                ]
            )
            sizes.append(size)
            centered = residual - residual.mean(0)
            sd = centered.std(0)
            valid = sd > 1e-12
            k = int(valid.sum())
            if k > 1:
                z = centered[:, valid] / sd[valid]
                pair_count = k * (k - 1) // 2
                corr_sum += (np.square(z.sum(1)).mean() - k) / (k * (k - 1)) * pair_count
                corr_pairs += pair_count
        acfs, sizes = np.asarray(acfs), np.asarray(sizes)
        row = {
            "factor": factor,
            **totals,
            "train_agreement": totals["train_agree"] / totals["cells"],
            "heldout_agreement": totals["test_agree"] / totals["cells"],
            "train_non_single_agreement": (
                totals["train_non_single_agree"] / totals["non_single_cells"]
                if totals["non_single_cells"]
                else None
            ),
            "heldout_non_single_agreement": (
                totals["test_non_single_agree"] / totals["non_single_cells"]
                if totals["non_single_cells"]
                else None
            ),
            "pooled_train_test_agreement": totals["pooled_same"] / totals["cells"],
            "member_train_test_agreement": float(np.mean(st[:, j] == sv[:, j])),
            "within_group_residual_correlation": corr_sum / corr_pairs if corr_pairs else None,
            "residual_correlation_pairs": corr_pairs,
        }
        for k, lag in enumerate((1, 4, 13)):
            good = np.isfinite(acfs[:, k])
            row[f"score_acf_{lag}"] = (
                float(np.average(acfs[good, k], weights=sizes[good])) if good.any() else None
            )
            row[f"score_abs_acf_{lag}"] = (
                float(np.average(np.abs(acfs[good, k]), weights=sizes[good]))
                if good.any()
                else None
            )
        rows.append(row)
    return rows


def build_tables(data):
    """Rebuild every held-out prediction from original-unit coefficients/intercepts."""
    p, x, y = data["protocol"], data["x"], data["y"]
    methods = p["methods"] + MATCHED
    rows = {
        name: []
        for name in (
            "metrics",
            "portfolio_metrics",
            "weekly_losses",
            "block_losses",
            "selections",
            "partitions",
            "coherence",
            "loading_stability",
            "support_stability",
            "gate_transitions",
            "gate_categories",
            "coefficient_categories",
            "cluster_sizes",
            "gate_coverage",
            "support_levels",
        )
    }
    bundle = {}
    for t in p["calendar"]["lookbacks"]:
        wins = [w for w in data["windows"] if w["lookback"] == t]
        losses, errors, truths, baselines = [], [], [], []
        previous = {}
        for block, w in enumerate(wins):
            o, end, a, pre = w["origin"], w["end"], w["arrays"], w["pre"]
            truth = y.iloc[o:end].to_numpy()
            xv = x.iloc[o:end].to_numpy()
            base = truth - np.asarray(pre["y_mean"])
            sd = np.asarray(pre["y_sd"])
            block_errors, block_losses, diag_cache = [], [], {}
            for method in methods:
                head = {
                    "lookback": t,
                    "origin": o,
                    "block": block,
                    "date": x.index[o].strftime("%Y-%m-%d"),
                    "method": method,
                }
                beta = a[method + "__beta"]
                intercept = (
                    a[method + "__intercept"]
                    if method in p["methods"]
                    else np.asarray(pre["y_mean"]) - beta @ np.asarray(pre["x_mean"])
                )
                predicted = xv @ beta.T + intercept
                np.testing.assert_allclose(
                    predicted, a[method + "__prediction"], atol=1e-12, rtol=1e-12
                )
                error = truth - predicted
                loss = np.square(error / sd).mean(1)
                np.testing.assert_allclose(
                    loss, a[method + "__weekly_standardized_loss"], atol=1e-12, rtol=1e-12
                )
                block_errors.append(error)
                block_losses.append(loss)
                rows["block_losses"].append(
                    {
                        **head,
                        "weeks": len(truth),
                        "standardized_loss_sum": float(loss.sum()),
                        "raw_squared_error_sum": float(np.square(error).sum()),
                        "baseline_squared_error_sum": float(np.square(base).sum()),
                    }
                )
                rows["weekly_losses"].extend(
                    {**head, "date": date.strftime("%Y-%m-%d"), "standardized_mse": float(value)}
                    for date, value in zip(x.index[o:end], loss)
                )
                std_beta = beta * np.asarray(pre["x_sd"])[None, :] / sd[:, None]
                if method in p["methods"]:
                    np.testing.assert_allclose(
                        std_beta, a[method + "__beta_standardized"], atol=1e-12, rtol=1e-12
                    )
                    labels = a[method + "__labels"]
                    rows["selections"].append(
                        {**head, **selected_settings(w["selections"][method], p)}
                    )
                else:
                    labels = a["hcgl_unconstrained__labels"]
                gates = a[method + "__gates"]
                for tol in p["reporting"]["support_tolerances_standardized"]:
                    active = np.abs(std_beta) > tol
                    rows["support_levels"].append(
                        {
                            **head,
                            "tolerance": tol,
                            "cells": active.size,
                            "active": int(active.sum()),
                            "active_fraction": float(active.mean()),
                        }
                    )
                if method in GATED:
                    for j, factor in enumerate(x.columns):
                        g = gates[:, j]
                        rows["gate_coverage"].append(
                            {
                                **head,
                                "factor": factor,
                                "cells": len(g),
                                "positive": int((g == 1).sum()),
                                "negative": int((g == -1).sum()),
                                "excluded": int((g == 0).sum()),
                                "unrestricted_failed": int(np.isnan(g).sum()),
                            }
                        )
                group_sizes = np.array([np.sum(labels == v) for v in np.unique(labels)])
                rows["partitions"].append(
                    {
                        **head,
                        "clusters": len(group_sizes),
                        "singleton_response_fraction": float(
                            np.sum(group_sizes == 1) / len(labels)
                        ),
                        "min_cluster_size": int(group_sizes.min()),
                        "max_cluster_size": int(group_sizes.max()),
                        "ari_size": adjusted_rand(labels, data["labels"]["size_decile"]),
                        "ari_bm": adjusted_rand(labels, data["labels"]["bm_decile"]),
                        "pooling_contrast_available": bool(len(group_sizes) < len(labels)),
                    }
                )
                rows["cluster_sizes"].extend(
                    {**head, "cluster": int(label), "size": int(np.sum(labels == label))}
                    for label in np.unique(labels)
                )
                key = labels.tobytes()
                if key not in diag_cache:
                    diag_cache[key] = group_diagnostics(
                        x.iloc[o - t : o].to_numpy(),
                        y.iloc[o - t : o].to_numpy(),
                        xv,
                        truth,
                        labels,
                        list(x.columns),
                        pre,
                    )
                rows["coherence"].extend({**head, **r} for r in diag_cache[key])
                if method in previous:
                    old_beta, old_std, old_gates = previous[method]
                    for j, factor in enumerate(x.columns):
                        delta = beta[:, j] - old_beta[:, j]
                        rows["loading_stability"].append(
                            {
                                **head,
                                "factor": factor,
                                "mean_absolute_change": float(np.abs(delta).mean()),
                                "root_mean_square_change": float(np.sqrt(np.square(delta).mean())),
                            }
                        )
                    for tol in p["reporting"]["support_tolerances_standardized"]:
                        now, old = np.abs(std_beta) > tol, np.abs(old_std) > tol
                        union, common = now | old, now & old
                        rows["support_stability"].append(
                            {
                                **head,
                                "tolerance": tol,
                                "cells": now.size,
                                "active": int(now.sum()),
                                "entered": int((now & ~old).sum()),
                                "exited": int((old & ~now).sum()),
                                "changed_fraction": float(np.mean(now != old)),
                                "jaccard": float(common.sum() / union.sum())
                                if union.any()
                                else 1.0,
                                "common_active_sign_flip_fraction": float(
                                    np.mean(np.sign(std_beta[common]) != np.sign(old_std[common]))
                                )
                                if common.any()
                                else None,
                            }
                        )
                    if method in GATED:
                        new_state, old_state = (
                            np.nan_to_num(gates, nan=2),
                            np.nan_to_num(old_gates, nan=2),
                        )
                        for before in (-1, 0, 1, 2):
                            for after in (-1, 0, 1, 2):
                                rows["gate_transitions"].append(
                                    {
                                        **head,
                                        "before": before,
                                        "after": after,
                                        "cells": gates.size,
                                        "count": int(
                                            np.sum((old_state == before) & (new_state == after))
                                        ),
                                    }
                                )
                previous[method] = beta.copy(), std_beta.copy(), gates.copy()
                if method in GATED:
                    for j, factor in enumerate(x.columns):
                        if factor not in ("SMB", "HML"):
                            continue
                        for axis in ("size_decile", "bm_decile"):
                            for category in sorted(data["labels"][axis].unique()):
                                mask = data["labels"][axis].to_numpy() == category
                                g = gates[mask, j]
                                b = std_beta[mask, j]
                                common = {
                                    **head,
                                    "factor": factor,
                                    "axis": axis,
                                    "category": int(category),
                                    "cells": int(mask.sum()),
                                }
                                rows["gate_categories"].append(
                                    {
                                        **common,
                                        "positive": int((g == 1).sum()),
                                        "negative": int((g == -1).sum()),
                                        "excluded": int((g == 0).sum()),
                                        "unrestricted_failed": int(np.isnan(g).sum()),
                                    }
                                )
                                rows["coefficient_categories"].append(
                                    {
                                        **common,
                                        "tolerance": 1e-6,
                                        "positive": int((b > 1e-6).sum()),
                                        "negative": int((b < -1e-6).sum()),
                                        "near_zero": int((np.abs(b) <= 1e-6).sum()),
                                    }
                                )
            losses.append(np.stack(block_losses, axis=1))
            errors.append(np.stack(block_errors, axis=2))
            truths.append(truth)
            baselines.append(base)
        loss = np.concatenate(losses)
        error, base = np.concatenate(errors), np.concatenate(baselines)
        require(len(loss) == p["calendar"]["test_weeks"], "Incomplete reporting weeks")
        block_counts = np.array([len(v) for v in losses], dtype=int)
        block_sums = np.stack([v.sum(0) for v in losses])
        bundle[f"T{t}__weekly_losses"] = loss
        bundle[f"T{t}__block_sums"] = block_sums
        bundle[f"T{t}__block_counts"] = block_counts
        for j, method in enumerate(methods):
            sse = np.square(error[:, :, j]).sum(0)
            sst = np.square(base).sum(0)
            require(np.all(sst > 0), "Undefined portfolio baseline R2")
            r2 = 1 - sse / sst
            rows["metrics"].append(
                {
                    "lookback": t,
                    "method": method,
                    "weeks": len(loss),
                    "standardized_mse": float(loss[:, j].mean()),
                    "standardized_rmse": float(np.sqrt(loss[:, j].mean())),
                    "raw_rmse": float(np.sqrt(np.square(error[:, :, j]).mean())),
                    "pooled_r2": float(1 - sse.sum() / sst.sum()),
                    "mean_portfolio_r2": float(r2.mean()),
                    "median_portfolio_r2": float(np.median(r2)),
                    "q25_portfolio_r2": float(np.quantile(r2, 0.25)),
                    "q75_portfolio_r2": float(np.quantile(r2, 0.75)),
                }
            )
            rows["portfolio_metrics"].extend(
                {
                    "lookback": t,
                    "method": method,
                    "portfolio": portfolio,
                    "raw_sse": float(sse[k]),
                    "baseline_sse": float(sst[k]),
                    "r2": float(r2[k]),
                }
                for k, portfolio in enumerate(y.columns)
            )
    tables = {name: pd.DataFrame(values) for name, values in rows.items()}
    sel = tables["selections"]
    bool_columns = [
        c
        for c in sel
        if c.startswith(("fraction_", "partition_", "tau_", "ridge_", "extended_"))
        and c != "ridge_lambda"
    ]
    numeric = sel[["lookback", "method"] + bool_columns].copy()
    for column in bool_columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    tables["boundaries"] = (
        numeric.groupby(["lookback", "method"], sort=False)[bool_columns].mean().reset_index()
    )
    for source, target, states in (
        (
            "gate_categories",
            "gate_category_summary",
            ["positive", "negative", "excluded", "unrestricted_failed"],
        ),
        (
            "coefficient_categories",
            "coefficient_category_summary",
            ["positive", "negative", "near_zero"],
        ),
    ):
        grouped = tables[source].groupby(
            ["lookback", "method", "factor", "axis", "category"], sort=False
        )
        summary = grouped[["cells"] + states].sum().reset_index()
        for state in states:
            summary[state + "_fraction"] = summary[state] / summary["cells"]
        tables[target] = summary
    return tables, bundle


def bootstrap_tables(protocol, bundle):
    methods = protocol["methods"] + MATCHED
    main = [protocol["reporting"]["primary_contrast"]] + protocol["reporting"]["other_contrasts"]
    matched = conventions(protocol)["matched_contrasts"]
    rows, intervals, samples = [], [], {}
    b = protocol["reporting"]["bootstrap"]
    for t in protocol["calendar"]["lookbacks"]:
        sums, counts = bundle[f"T{t}__block_sums"], bundle[f"T{t}__block_counts"]
        observed = sums.sum(0) / counts.sum()
        length = b["mean_blocks"][str(t)]
        for mean in sorted([length * 0.5, length, length * 2]):
            idx = stationary_indices(len(counts), b["resamples"], mean, b["seed"])
            estimates = sums[idx].sum(1) / counts[idx].sum(1)[:, None]
            tag = f"T{t}_L{mean:g}"
            samples[tag + "__indices"] = idx
            samples[tag + "__mse"] = estimates
            for j, method in enumerate(methods):
                lo, hi = np.quantile(estimates[:, j], [0.025, 0.975])
                slo, shi = np.quantile(np.sqrt(estimates[:, j]), [0.025, 0.975])
                intervals.append(
                    {
                        "lookback": t,
                        "mean_blocks": mean,
                        "method": method,
                        "primary_length": mean == length,
                        "mse_lower": lo,
                        "mse_upper": hi,
                        "srmse_lower": slo,
                        "srmse_upper": shi,
                    }
                )
            for kind, pairs in (("primary", main), ("matched", matched)):
                for a, c in pairs:
                    i, j = methods.index(a), methods.index(c)
                    for metric in ("mse", "srmse"):
                        values = (
                            estimates[:, i] - estimates[:, j]
                            if metric == "mse"
                            else np.sqrt(estimates[:, i]) - np.sqrt(estimates[:, j])
                        )
                        delta = (
                            observed[i] - observed[j]
                            if metric == "mse"
                            else np.sqrt(observed[i]) - np.sqrt(observed[j])
                        )
                        lo, hi = np.quantile(values, [0.025, 0.975])
                        rows.append(
                            {
                                "lookback": t,
                                "mean_blocks": mean,
                                "kind": kind,
                                "primary_length": mean == length,
                                "method_a": a,
                                "method_b": c,
                                "metric": metric,
                                "difference": float(delta),
                                "lower": lo,
                                "upper": hi,
                            }
                        )
    return pd.DataFrame(rows), pd.DataFrame(intervals), samples


def write_latex(tables, out, protocol):
    for t in protocol["calendar"]["lookbacks"]:
        metrics = tables["metrics"].query("lookback == @t").set_index("method")
        lines = [
            r"\begin{tabular}{lrrrr}\toprule",
            r"Method & Std. MSE & Std. RMSE & Raw RMSE & Pooled $R^2$\\\midrule",
        ]
        for method in protocol["methods"]:
            r = metrics.loc[method]
            lines.append(
                f"{LABELS[method]} & {r.standardized_mse:.6f} & "
                f"{r.standardized_rmse:.6f} & {r.raw_rmse:.6f} & {r.pooled_r2:.6f} " + chr(92) * 2
            )
        lines.append(r"\bottomrule\end{tabular}")
        (out / f"table_prediction_T{t}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        contrasts = tables["contrasts"].query(
            "lookback == @t and primary_length and kind == 'primary'"
        )
        lines = [
            r"\begin{tabular}{llrrr}\toprule",
            r"Contrast & Metric & Difference & Lower & Upper\\\midrule",
        ]
        for r in contrasts.itertuples():
            name = LABELS[r.method_a] + " minus " + LABELS[r.method_b]
            lines.append(
                f"{name} & {r.metric.upper()} & {r.difference:.6f} & "
                f"{r.lower:.6f} & {r.upper:.6f} " + chr(92) * 2
            )
        lines.append(r"\bottomrule\end{tabular}")
        (out / f"table_contrasts_T{t}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def figures(tables, out, protocol):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.dpi": 170,
        }
    )
    out.mkdir(exist_ok=True)
    lookbacks = protocol["calendar"]["lookbacks"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)
    for col, t in enumerate(lookbacks):
        for row, metric in enumerate(("mse", "srmse")):
            ax = axes[row, col]
            selected = tables["contrasts"].query(
                "lookback == @t and kind == 'primary' and metric == @metric"
            )
            pairs = list(dict.fromkeys(zip(selected.method_a, selected.method_b)))
            lengths = sorted(selected.mean_blocks.unique())
            for k, length in enumerate(lengths):
                part = (
                    selected[selected.mean_blocks == length]
                    .set_index(["method_a", "method_b"])
                    .loc[pairs]
                )
                yy = np.arange(len(pairs)) + (k - 1) * 0.16
                color = f"C{k}"
                ax.hlines(yy, part.lower, part.upper, color=color, lw=1)
                ax.plot(
                    part.difference, yy, "o", ms=3, color=color, label=f"Mean run {length:g} blocks"
                )
            ax.axvline(0, color=".4", lw=0.8)
            ax.set_yticks(
                np.arange(len(pairs)),
                [
                    ("FCGL: " if a.startswith("fcgl") else "HCGL: ")
                    + LABELS[a].removeprefix("HCGL ").removeprefix("FCGL ")
                    + " − "
                    + LABELS[b].removeprefix("HCGL ").removeprefix("FCGL ")
                    for a, b in pairs
                ],
            )
            ax.set_title(f"T={t}: paired Δ{metric.upper()}")
            ax.set_xlabel("Negative favors the first procedure; 95% percentile intervals")
            if row == 0:
                ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / ("ff_paired_contrasts." + suffix), bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, t in zip(axes, lookbacks):
        weekly = tables["weekly_losses"].query("lookback == @t")
        for method in ("hcgl_exclude", "hcgl_unconstrained", "ridge", "ols"):
            part = weekly[weekly.method == method]
            ax.plot(
                pd.to_datetime(part.date),
                part.standardized_mse.rolling(13, min_periods=13).mean(),
                label=LABELS[method],
                lw=1.0,
            )
        ax.set_ylabel(f"T={t}\nStandardized MSE")
        ax.legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("Test week; trailing 13-week means for display only")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / ("ff_calendar_loss." + suffix), bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex="col")
    for col, t in enumerate(lookbacks):
        parts = tables["partitions"].query("lookback == @t and method == 'hcgl_exclude'")
        selection = tables["selections"].query("lookback == @t and method == 'hcgl_exclude'")
        dates = pd.to_datetime(parts.date).to_numpy()
        axes[0, col].step(dates, parts.clusters, where="post", label="Clusters")
        axes[0, col].set_title(f"Pooled HCGL exclusion, T={t}")
        axes[0, col].set_ylabel("Learned response groups")
        vals = np.array([np.nan if v == "inf" else float(v) for v in selection.tau])
        axes[1, col].step(dates, vals, where="post", label="Finite threshold")
        infinite = selection.tau.astype(str).eq("inf").to_numpy()
        if infinite.any():
            axes[1, col].scatter(
                dates[infinite], np.repeat(7, infinite.sum()), marker="^", label="∞ endpoint"
            )
        axes[1, col].set_ylabel("Selected gate threshold")
        axes[1, col].legend(fontsize=7)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / ("ff_selected_pooling." + suffix), bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex="col")
    for col, t in enumerate(lookbacks):
        for method in ("hcgl_exclude", "hcgl_unconstrained", "ridge", "ols"):
            load = tables["loading_stability"].query("lookback == @t and method == @method")
            agg = load.groupby("date", sort=True).mean_absolute_change.mean()
            axes[0, col].plot(pd.to_datetime(agg.index), agg, lw=1, label=LABELS[method])
            support = tables["support_stability"].query(
                "lookback == @t and method == @method and tolerance == 0.0001"
            )
            axes[1, col].plot(pd.to_datetime(support.date), support.changed_fraction, lw=1)
        axes[0, col].set_title(f"Consecutive-origin changes, T={t}")
        axes[0, col].set_ylabel("Mean absolute raw loading change")
        axes[0, col].legend(fontsize=7)
        axes[1, col].set_ylabel("Estimated support change fraction")
        axes[1, col].set_xlabel("Standardized support tolerance 1e−4; stability is not accuracy")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / ("ff_stability." + suffix), bbox_inches="tight")
    plt.close(fig)


def run(panel, fit_run, out):
    out = external_path(out)
    require(not out.exists() or not list(out.iterdir()), "Choose a fresh report directory")
    out.mkdir(parents=True, exist_ok=True)
    p = read_json(panel / "protocol.json")
    (out / "analysis_conventions.json").write_bytes(json_bytes(conventions(p)))
    data = load_run(panel, fit_run)
    tables, bundle = build_tables(data)
    contrasts, intervals, samples = bootstrap_tables(p, bundle)
    tables.update(contrasts=contrasts, method_intervals=intervals)
    for name, table in tables.items():
        table.to_csv(out / (name + ".csv"), index=False, float_format="%.17g")
    np.savez_compressed(out / "report_data.npz", **bundle)
    np.savez_compressed(out / "bootstrap_samples.npz", **samples)
    write_latex(tables, out, p)
    figures(tables, out / "figures", p)
    sources = {Path(__file__).relative_to(REPOSITORY).as_posix(): sha256(Path(__file__))}
    verifier = Path(__file__).with_name("verify_french_tables.py")
    if verifier.exists():
        sources[verifier.relative_to(REPOSITORY).as_posix()] = sha256(verifier)
    for name in sources:
        target = out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPOSITORY / name).read_bytes())
    files = {
        v.relative_to(out).as_posix(): sha256(v) for v in sorted(out.rglob("*")) if v.is_file()
    }
    manifest = {
        "status": "produced; independent verification required",
        "schema": 1,
        "fit_identity": data["manifest"]["identity"],
        "panel_manifest_sha256": sha256(panel / "manifest.json"),
        "protocol_sha256": sha256(panel / "protocol.json"),
        "stage3_certificate_sha256": sha256(fit_run / "full_validation.json"),
        "report_inputs_sha256": data["input_hashes"],
        "source_sha256": sources,
        "source_snapshots": True,
        "files_sha256": files,
        "python": sys.version,
        "platform": platform.platform(),
        "estimator_defaults": data["manifest"]["binding"]["defaults_observed"],
        "refitting_performed": False,
        "raw_input_distribution_rights_established": False,
    }
    (out / "report_manifest.json").write_bytes(json_bytes(manifest))
    return {
        "status": manifest["status"],
        "lookbacks": p["calendar"]["lookbacks"],
        "metric_rows": len(tables["metrics"]),
        "output": str(out),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel-run", type=Path, required=True)
    parser.add_argument("--fit-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json_bytes(run(args.panel_run, args.fit_run, args.out)).decode())


if __name__ == "__main__":
    main()
