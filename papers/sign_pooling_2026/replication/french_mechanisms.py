"""Exploratory, separately frozen mechanism and complexity study of the FF panel.

Original reconstruction results remain primary. Date-score variances follow
Cameron and Miller (2015), doi:10.3368/jhr.50.2.317. Ratios below compare empirical
score variances, not calibrated tests or guaranteed effective sample sizes.
Stationary bootstrap conventions follow the separately frozen FF protocol.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

if __package__:
    from . import french_estimators as F
    from . import french_rolling_validation as E
    from . import french_report as R
    from .french_protocol import external_path, sha256, REPOSITORY
else:
    import french_estimators as F
    import french_rolling_validation as E
    import french_report as R
    from french_protocol import external_path, sha256, REPOSITORY

VARIANTS = (
    "unconstrained",
    "singleton_exclude",
    "pooled_exclude",
    "pooled_release",
    "pooled_direction",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def protocol():
    """All new choices, fixed before this study's refits or simulation results."""
    return {
        "study": "ff_mechanisms_v1",
        "exploratory_after_original_outer_results": True,
        "original_outcomes_preserved": True,
        "ward_cut": 0.7,
        "gate_tau": 2,
        "matched_families": ["hcgl", "fcgl"],
        "matched_variants": list(VARIANTS),
        "matched_pairs": [[v, "unconstrained"] for v in VARIANTS[1:]]
        + [["pooled_exclude", "singleton_exclude"], ["pooled_exclude", "pooled_release"]],
        "pooling_pairs": [
            ["pooled_exclude", "singleton_exclude"],
            ["pooled_exclude", "pooled_release"],
        ],
        "matched_selection": "unconstrained inner MSE at fixed Ward cut; same outer numeric lambda",
        "fractions": [1, 0.5, 0.3, 0.17, 0.1, 0.05, 0.03, 0],
        "complexity_budgets": [0.5, 0.65, 0.8],
        "complexity_selection": (
            "minimum pooled inner MSE subject to mean inner active fraction <= budget"
        ),
        "complexity_axes": (
            "fixed Ward 0.7 and tau 2; original 12 methods; no outer-score selection"
        ),
        "budget_is": "inner ceiling, not a guarantee on full-training achieved support",
        "support_tolerance": 1e-4,
        "reference_thresholds": [1e-4, 0.05],
        "truth_labels_on_real_panel": False,
        "bootstrap": {"resamples": 5000, "seed": 20261026, "mean_blocks": {"52": 4, "104": 8}},
        "simulation": {
            "calibration_rows": 104,
            "calibration_source": "first 104 weeks before original outer tests",
            "truth": "thresholded standardized training OLS; magnitude <0.05 set to exact zero",
            "coherence": ["empirical", "coherent"],
            "truth_groups": (
                "four size/BM quadrants; support and within-row magnitude ratios preserved"
            ),
            "signal_control": (
                "positive row rescaling holds calibrated signal variance across factorial cells"
            ),
            "coherent_sign": "sign of group sum; exact zero sum uses positive",
            "factor_correlation_mix": [0, 1],
            "error_dependence_mix": [0, 1],
            "error_covariance": "0.9 empirical OLS residual covariance + 0.1 diagonal",
            "dates": "iid Gaussian; no financial serial-dependence or heavy-tail claim",
            "lookbacks": [52, 104],
            "replications": 40,
            "test_rows": 512,
            "seed": 20261026,
            "pairing": (
                "SeedSequence(seed,lookback,replication); same innovations across factorial cells"
            ),
            "partitions": ["known_quadrants", "learned_ward"],
            "families": ["hcgl", "fcgl"],
            "variants": list(VARIANTS),
            "penalty_fractions": [0, 0.1],
            "penalty": (
                "unconstrained KKT scale per family/partition; same numeric lambda across gates"
            ),
            "purpose": "matched mechanism effects, not prediction-tuned benchmark rankings",
            "metrics": [
                "loading MSE",
                "population excess prediction risk",
                "test MSE",
                "correct active fitted signs",
                "false exclusion",
                "support precision/recall",
                "gate active direction/coverage",
                "partition ARI",
            ],
            "mc_intervals": (
                "paired replicate mean +/- 1.96 Monte Carlo SE; descriptive finite simulation"
            ),
        },
        "defaults": {
            "auto_sign_constraints": False,
            "auto_sign_threshold_t": 0.75,
            "auto_sign_variance": "independent",
        },
    }


def freeze(panel, fits, original_report, out):
    out = external_path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "freeze.json").exists():
        return validate_freeze(out)
    v = read(original_report / "verification.json")
    if v["status"] != "passed" or v["report_manifest_sha256"] != sha256(
        original_report / "report_manifest.json"
    ):
        raise ValueError("Original scientific report not authenticated")
    source = {}
    for name in (
        "french_mechanisms.py",
        "french_semisynthetic.py",
        "verify_french_mechanisms.py",
        "tests/test_french_mechanisms.py",
    ):
        path = Path(__file__).parent / name
        relative = path.relative_to(REPOSITORY).as_posix()
        source[relative] = sha256(path)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    original = read(fits / "run_manifest.json")
    for name, value in original["binding"]["source_sha256"].items():
        if sha256(REPOSITORY / name) != value:
            raise ValueError("Estimation source changed: " + name)
    E.atomic_json(out / "protocol.json", protocol())
    binding = {
        "protocol_sha256": sha256(out / "protocol.json"),
        "source_sha256": source,
        "original_fit_identity": original["identity"],
        "original_report_manifest_sha256": v["report_manifest_sha256"],
        "panel_manifest_sha256": sha256(panel / "manifest.json"),
        "panel": str(panel),
        "fits": str(fits),
        "original_report": str(original_report),
    }
    E.atomic_json(
        out / "freeze.json",
        {"binding": binding, "identity": E.digest(binding), "new_outcomes_accessed": False},
    )
    return validate_freeze(out)


def validate_freeze(out):
    frozen = read(out / "freeze.json")
    b = frozen["binding"]
    if frozen["identity"] != E.digest(b) or sha256(out / "protocol.json") != b["protocol_sha256"]:
        raise ValueError("Changed mechanism freeze")
    for name, value in b["source_sha256"].items():
        if sha256(REPOSITORY / name) != value or sha256(out / "source" / name) != value:
            raise ValueError("Mechanism source differs from freeze: " + name)
    return frozen


def candidate(method, p):
    grouped = method.startswith(("hcgl", "fcgl")) or method == "cooperative"
    return {
        "method": method,
        "partition": p["ward_cut"] if grouped else None,
        "tau": p["gate_tau"] if method in R.GATED else None,
    }


def choices(method, p, original):
    c = candidate(method, p)
    if method == "ols":
        return [c]
    if method == "ridge":
        return [{**c, "ridge_lambda": lam} for lam in original["grid"]["ridge_lambdas"]]
    return [{**c, "fraction": f} for f in p["fractions"]]


def cached_evaluate(
    state, cs, original, old_folder, old_receipt, new_folder, identity, *, numeric=None
):
    """Read only authenticated old fits; all new fits/checkpoints go to new_folder."""
    evaluator = E.Evaluator(state, original, new_folder, identity)
    legacy = E.Evaluator(state, original, old_folder, old_receipt["identity"])
    used = {}
    for j, c in enumerate(cs):
        spec = evaluator.spec(c)
        lam = (
            numeric[j]
            if numeric is not None
            else (
                np.inf
                if c.get("ridge_lambda") == "inf"
                else float(c.get("ridge_lambda", c.get("fraction", 0) * spec.scale))
            )
        )
        key = evaluator.fit_key(spec, lam)
        old = old_folder / (key + ".npz")
        if old.is_file():
            relative = old.relative_to(old_folder.parent).as_posix()
            if old_receipt["files_sha256"].get(relative) != sha256(old):
                raise ValueError("Changed frozen fit: " + relative)
            value = legacy._load(key)
            evaluator.results[key] = value
            evaluator._save(key, value)
            used[relative] = sha256(old)
    return evaluator.evaluate(cs, numerical_lambdas=numeric), used


def choose_budget(rows, budget, original):
    eligible = [r for r in rows if r["inner_active_fraction"] <= budget + 1e-12]
    return E.choose(eligible, original)[0] if eligible else None


def gate_spec(state, base, variant, tau):
    """Keep the objective labels/weights; alter only the restrictions."""
    spec = copy.deepcopy(base)
    if variant == "unconstrained":
        return spec
    labels = np.arange(len(base.labels)) if variant == "singleton_exclude" else base.labels
    signs = state.gates(labels, 0 if variant == "pooled_direction" else tau).copy()
    if variant == "pooled_release":
        signs[signs == 0] = np.nan
    spec.signs = signs
    # The KKT endpoint shortcut must use the new cone, even at a matched lambda.
    spec.scale = F.kkt_scale(state, spec)
    return spec


def gate_precision(state, labels):
    """Explicit date scores; reference singleton variances use their own slopes."""
    n = len(state.x)
    rows = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        if len(members) < 2:
            continue
        for j, factor in enumerate(state.factors):
            x = state.x[:, j]
            den = x @ x
            yy = state.y[:, members]
            slopes = x @ yy / den
            pooled = float(slopes.mean())
            cell_scores = x[:, None] * (yy - x[:, None] * pooled)
            date_variance = (
                n / (n - 1) * np.square(cell_scores.sum(1)).sum() / (len(members) * den) ** 2
            )
            cell_variance = n / (n - 1) * np.square(cell_scores).sum() / (len(members) * den) ** 2
            single_scores = x[:, None] * (yy - x[:, None] * slopes)
            single = n / (n - 1) * np.square(single_scores).sum(0) / den**2
            rows.append(
                {
                    "cluster": int(label),
                    "factor": factor,
                    "size": len(members),
                    "pooled_slope": pooled,
                    "date_variance": float(date_variance),
                    "independent_score_variance": float(cell_variance),
                    "mean_singleton_variance": float(single.mean()),
                    "precision_ratio": float(np.sqrt(single.mean() / date_variance))
                    if date_variance > 0
                    else np.nan,
                    "variance_inflation": float(date_variance / cell_variance)
                    if cell_variance > 0
                    else np.nan,
                }
            )
    return rows


def train_select(x, y, original, p, old, receipt, folder, identity):
    """API receives no outer observations; common support ceilings use inner fits."""
    table, ledgers, cache = [], [], {}
    t = len(x)
    for method in original["methods"]:
        cs = choices(method, p, original)
        totals = {E.digest(c): [0.0, 0.0, 0] for c in cs}
        for fold, (end, vend) in enumerate(
            zip(
                original["calendar"]["inner_train_ends"][str(t)],
                original["calendar"]["inner_validation_ends"][str(t)],
            )
        ):
            state = F.prepare(x.iloc[:end], y.iloc[:end])
            results, used = cached_evaluate(
                state,
                cs,
                original,
                old / f"inner_{fold}_fits",
                receipt,
                folder / f"inner_{fold}_fits",
                identity,
            )
            cache.update(used)
            xx = (x.iloc[end:vend].to_numpy() - state.x_mean) / state.x_sd
            yy = (y.iloc[end:vend].to_numpy() - state.y_mean) / state.y_sd
            for c, spec, result, key in results:
                pred = xx @ result["beta"].T + result["alpha"]
                sse = float(np.square(yy - pred).sum())
                active = float((np.abs(result["beta"]) > p["support_tolerance"]).mean())
                z = totals[E.digest(c)]
                z[0] += sse
                z[1] += active
                z[2] += yy.size
                ledgers.append(
                    {
                        "method": method,
                        "fold": fold,
                        "candidate": c,
                        "fit_key": key,
                        "sse": sse,
                        "cells": yy.size,
                        "active_fraction": active,
                        "lambda": result["lambda"],
                        "status": result["status"],
                        "train_end": end,
                        "validation_end": vend,
                    }
                )
        table.extend(
            {
                "candidate": c,
                "candidate_id": E.digest(c),
                "score": totals[E.digest(c)][0] / totals[E.digest(c)][2],
                "inner_active_fraction": totals[E.digest(c)][1] / 3,
            }
            for c in cs
        )
    selected = {}
    for method in original["methods"]:
        rows = [r for r in table if r["candidate"]["method"] == method]
        for budget in p["complexity_budgets"]:
            selected[f"budget_{budget}__{method}"] = choose_budget(rows, budget, original)
        if method in ("hcgl_unconstrained", "fcgl_unconstrained"):
            selected["matched_base__" + method] = E.choose(rows, original)[0]
    return selected, table, ledgers, cache


def real_job(job):
    panel, fits, out = (Path(job[k]) for k in ("panel", "fits", "out"))
    o, t, identity = job["origin"], job["lookback"], job["identity"]
    folder, old = out / f"origin_{o}_T{t}", fits / f"origin_{o}_T{t}"
    done = folder / "receipt.json"
    if done.exists():
        saved = read(done)
        if saved["identity"] != identity:
            raise ValueError("Stale mechanism job")
        for name, value in saved["files_sha256"].items():
            if sha256(folder / name) != value:
                raise ValueError("Corrupt mechanism job")
        return {"job": folder.name, "resumed": True}
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    p, original = read(out / "protocol.json"), read(panel / "protocol.json")
    receipt = read(old / "window_receipt.json")
    x, y = R.frame(panel / "panel/X_weekly.csv"), R.frame(panel / "panel/Y_weekly.csv")
    selected, surface, ledger, used = train_select(
        x.iloc[o - t : o], y.iloc[o - t : o], original, p, old, receipt, folder, identity
    )
    state = F.prepare(x.iloc[o - t : o], y.iloc[o - t : o])
    arrays = {
        "training_sd": state.y_sd,
        "training_mean": state.y_mean,
        "x_sd": state.x_sd,
        "x_mean": state.x_mean,
        "x_train": state.x,
        "y_train": state.y,
    }
    if sha256(old / "outer_arrays.npz") != receipt["files_sha256"]["outer_arrays.npz"]:
        raise ValueError("Changed frozen OLS reference")
    with np.load(old / "outer_arrays.npz", allow_pickle=False) as saved:
        arrays["reference_ols_standardized"] = saved["ols__beta_standardized"].copy()
    descriptors, names = [], []
    for name, value in selected.items():
        if value is not None:
            names.append(name)
            descriptors.append(value["candidate"])
    results, more = cached_evaluate(
        state, descriptors, original, old / "outer_fits", receipt, folder / "outer_fits", identity
    )
    used.update(more)
    fit_metadata = {}

    def save(name, spec, result):
        beta, intercept = F.original_units(state, result)
        arrays[name + "__beta"] = beta
        arrays[name + "__intercept"] = intercept
        arrays[name + "__beta_standardized"] = result["beta"]
        arrays[name + "__gates"] = spec.signs
        arrays[name + "__labels"] = spec.labels
        fit_metadata[name] = {k: v for k, v in result.items() if k not in ("beta", "alpha")}
        fit_metadata[name]["family"] = spec.family
        fit_metadata[name]["weights"] = spec.weights.tolist()

    for name, (_, spec, result, _) in zip(names, results):
        save(name, spec, result)
        if name.startswith("matched_base"):
            family = name.split("__")[1].split("_")[0]
            for variant in VARIANTS:
                gate = gate_spec(state, spec, variant, p["gate_tau"])
                value = (
                    result
                    if variant == "unconstrained"
                    else F.fit_path(state, gate, [result["lambda"]], original)[0]
                )
                save("matched__" + family + "__" + variant, gate, value)
    # Test outcomes first enter scoring after all training choices are final.
    end = min(o + original["calendar"]["outer_horizon"], len(x))
    arrays["x_test"], arrays["actual"] = x.iloc[o:end].to_numpy(), y.iloc[o:end].to_numpy()
    for name in fit_metadata:
        arrays[name + "__prediction"] = (
            arrays["x_test"] @ arrays[name + "__beta"].T + arrays[name + "__intercept"]
        )
    E.atomic_npz(folder / "arrays.npz", arrays)
    E.atomic_json(
        folder / "selection.json", {"selected": selected, "surface": surface, "ledger": ledger}
    )
    E.atomic_json(folder / "fit_metadata.json", E.safe_json(fit_metadata))
    precision = gate_precision(state, state.labels(p["ward_cut"]))
    E.atomic_json(folder / "precision.json", E.safe_json(precision))
    files = {f.relative_to(folder).as_posix(): sha256(f) for f in folder.rglob("*") if f.is_file()}
    E.atomic_json(
        done,
        {
            "identity": identity,
            "origin": o,
            "lookback": t,
            "test_dates": x.index[o:end].strftime("%Y-%m-%d").tolist(),
            "used_frozen_fits": used,
            "files_sha256": files,
            "seconds": time.perf_counter() - started,
            "status": "complete",
        },
    )
    return {"job": folder.name, "resumed": False, "seconds": time.perf_counter() - started}


def aggregate_real(out):
    """Descriptive metrics, primary-length paired intervals and frontier figures."""
    frozen, p = validate_freeze(out), read(out / "protocol.json")
    original = read(Path(frozen["binding"]["panel"]) / "protocol.json")
    jobs = read(Path(frozen["binding"]["fits"]) / "run_manifest.json")["jobs"]
    rows, precision, changes, saved, missing = [], [], [], {}, []
    disagreement, conflicts, partial_coherence = [], [], []
    prior = {}
    for job in jobs:
        folder = out / job["folder"]
        receipt = read(folder / "receipt.json")
        if receipt["identity"] != frozen["identity"]:
            raise ValueError("Mixed mechanism jobs")
        for name, value in receipt["files_sha256"].items():
            if sha256(folder / name) != value:
                raise ValueError("Changed mechanism output: " + name)
        t, o = job["lookback"], job["origin"]
        a = dict(np.load(folder / "arrays.npz", allow_pickle=False))
        selection, meta = read(folder / "selection.json"), read(folder / "fit_metadata.json")
        ols = a["reference_ols_standardized"]
        marginal = a["y_train"].T @ a["x_train"] / len(a["x_train"])
        corr = a["x_train"].T @ a["x_train"] / len(a["x_train"])
        np.testing.assert_allclose(marginal, ols @ corr, atol=2e-11, rtol=2e-11)
        for tol in p["reference_thresholds"]:
            for j, factor in enumerate(original["predictors"]):
                active = np.abs(ols[:, j]) > tol
                disagreement.append(
                    {
                        "lookback": t,
                        "origin": o,
                        "factor": factor,
                        "tolerance": tol,
                        "reference_active": int(active.sum()),
                        "opposite_marginal": int((active & (ols[:, j] * marginal[:, j] < 0)).sum()),
                    }
                )
        for name, value in selection["selected"].items():
            if value is None:
                missing.append(
                    {
                        "lookback": t,
                        "origin": o,
                        "method": name,
                        "reason": "no inner-budget-feasible candidate",
                    }
                )
        for row in read(folder / "precision.json"):
            precision.append({"lookback": t, "origin": o, **row})
        for name in meta:
            pred = a["x_test"] @ a[name + "__beta"].T + a[name + "__intercept"]
            np.testing.assert_allclose(pred, a[name + "__prediction"], atol=1e-12, rtol=1e-12)
            losses = np.square((a["actual"] - pred) / a["training_sd"])
            b = a[name + "__beta_standardized"]
            rows.append(
                {
                    "lookback": t,
                    "origin": o,
                    "method": name,
                    "weeks": len(pred),
                    "loss_sum": float(losses.mean(1).sum()),
                    "raw_sse": float(np.square(a["actual"] - pred).sum()),
                    "active_fraction": float((np.abs(b) > p["support_tolerance"]).mean()),
                    "clusters": len(np.unique(a[name + "__labels"])),
                    "lambda": meta[name]["lambda"],
                }
            )
            key = (t, name)
            if name.startswith("matched__"):
                gates, labels = a[name + "__gates"], a[name + "__labels"]
                for tol in p["reference_thresholds"]:
                    for j, factor in enumerate(original["predictors"]):
                        active = np.abs(ols[:, j]) > tol
                        conflicts.append(
                            {
                                "lookback": t,
                                "origin": o,
                                "method": name,
                                "factor": factor,
                                "tolerance": tol,
                                "reference_active": int(active.sum()),
                                "opposite_signed_gate": int(
                                    (active & (gates[:, j] * ols[:, j] < 0)).sum()
                                ),
                                "excluded_reference_active": int(
                                    (active & (gates[:, j] == 0)).sum()
                                ),
                            }
                        )
                        minority = eligible = mixed = total = 0
                        for label in np.unique(labels):
                            group = np.flatnonzero(labels == label)
                            if len(group) < 2:
                                continue
                            pos, neg = (
                                int((ols[group, j] > tol).sum()),
                                int((ols[group, j] < -tol).sum()),
                            )
                            minority += min(pos, neg)
                            total += pos + neg
                            eligible += int(pos + neg > 0)
                            mixed += int(pos > 0 and neg > 0)
                        partial_coherence.append(
                            {
                                "lookback": t,
                                "origin": o,
                                "method": name,
                                "factor": factor,
                                "tolerance": tol,
                                "active_non_singleton_cells": total,
                                "minority_cells": minority,
                                "eligible_groups": eligible,
                                "mixed_groups": mixed,
                            }
                        )
            if key in prior:
                pb, raw = prior[key]
                active, oldactive = (
                    np.abs(b) > p["support_tolerance"],
                    np.abs(pb) > p["support_tolerance"],
                )
                changes.append(
                    {
                        "lookback": t,
                        "origin": o,
                        "method": name,
                        "support_change": float(np.mean(active != oldactive)),
                        "raw_loading_change": float(np.abs(a[name + "__beta"] - raw).mean()),
                    }
                )
            prior[key] = (b, a[name + "__beta"])
        saved[job["folder"]] = sha256(folder / "receipt.json")
    blocks = pd.DataFrame(rows)
    summary = []
    for (t, name), g in blocks.groupby(["lookback", "method"], sort=False):
        summary.append(
            {
                "lookback": t,
                "method": name,
                "windows": len(g),
                "weeks": int(g.weeks.sum()),
                "standardized_mse": float(g.loss_sum.sum() / g.weeks.sum()),
                "standardized_rmse": float(np.sqrt(g.loss_sum.sum() / g.weeks.sum())),
                "raw_rmse": float(np.sqrt(g.raw_sse.sum() / g.weeks.sum() / original["responses"])),
                "mean_active_fraction": float(g.active_fraction.mean()),
                "non_singleton_windows": int((g.clusters < original["responses"]).sum()),
            }
        )
    metrics = pd.DataFrame(summary)
    intervals = []
    for t in original["calendar"]["lookbacks"]:
        for family in p["matched_families"]:
            for variant, reference in p["matched_pairs"]:
                base = f"matched__{family}__{reference}"
                name = f"matched__{family}__{variant}"
                left = blocks.query("lookback==@t and method==@name").sort_values("origin")
                right = blocks.query("lookback==@t and method==@base").sort_values("origin")
                np.testing.assert_array_equal(left.origin, right.origin)
                indices = R.stationary_indices(
                    len(left),
                    p["bootstrap"]["resamples"],
                    p["bootstrap"]["mean_blocks"][str(t)],
                    p["bootstrap"]["seed"],
                )
                counts = left.weeks.to_numpy()
                diff = left.loss_sum.to_numpy() - right.loss_sum.to_numpy()
                sample = diff[indices].sum(1) / counts[indices].sum(1)
                low, high = np.quantile(sample, [0.025, 0.975])
                intervals.append(
                    {
                        "lookback": t,
                        "method": name,
                        "reference": base,
                        "difference": float(diff.sum() / counts.sum()),
                        "lower": low,
                        "upper": high,
                    }
                )
    tables = {
        "real_blocks": blocks,
        "real_metrics": metrics,
        "real_contrasts": pd.DataFrame(intervals),
        "precision": pd.DataFrame(precision),
        "stability": pd.DataFrame(changes),
        "marginal_partial": pd.DataFrame(disagreement),
        "gate_conflicts": pd.DataFrame(conflicts),
        "partial_coherence": pd.DataFrame(partial_coherence),
        "budget_unavailable": pd.DataFrame(
            missing, columns=["lookback", "origin", "method", "reason"]
        ),
    }
    for name, table in tables.items():
        table.to_csv(out / (name + ".csv"), index=False, float_format="%.17g")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, t in zip(axes, original["calendar"]["lookbacks"]):
        subset = metrics[(metrics.lookback == t) & metrics.method.str.startswith("budget_")]
        for method in original["methods"]:
            g = subset[subset.method.str.endswith("__" + method)].sort_values(
                "mean_active_fraction"
            )
            if len(g):
                ax.plot(g.mean_active_fraction, g.standardized_mse, ".-", label=method)
        ax.set(
            title=f"T={t}; inner support ceilings",
            xlabel="Achieved mean active fraction",
            ylabel="Held-out standardized MSE",
        )
    axes[1].legend(fontsize=7, bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / ("complexity_frontier." + suffix), bbox_inches="tight")
    plt.close(fig)
    stability = (
        tables["stability"]
        .groupby(["lookback", "method"])[["support_change", "raw_loading_change"]]
        .mean()
        .reset_index()
    )
    stability = stability.merge(metrics, on=["lookback", "method"])
    stability.to_csv(out / "stability_error.csv", index=False, float_format="%.17g")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, t in zip(axes, original["calendar"]["lookbacks"]):
        for method in original["methods"]:
            g = stability[
                (stability.lookback == t)
                & stability.method.str.startswith("budget_")
                & stability.method.str.endswith("__" + method)
            ].sort_values("raw_loading_change")
            if len(g):
                ax.plot(g.raw_loading_change, g.standardized_mse, ".-", label=method)
        ax.set(
            title=f"T={t}; stability is not accuracy",
            xlabel="Mean raw loading change",
            ylabel="Held-out standardized MSE",
        )
    axes[1].legend(fontsize=7, bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / ("stability_frontier." + suffix), bbox_inches="tight")
    plt.close(fig)
    E.atomic_json(
        out / "real_report_manifest.json",
        {
            "identity": frozen["identity"],
            "status": "produced; verification required",
            "window_receipt_sha256": saved,
            "files_sha256": {
                f.name: sha256(f)
                for f in out.iterdir()
                if f.is_file() and f.suffix in (".csv", ".png", ".pdf")
            },
        },
    )
    return metrics


def run_real(out, workers=4):
    frozen = validate_freeze(out)
    b = frozen["binding"]
    # Re-authenticate original reporting inputs and exact expected windows.
    data = R.load_run(Path(b["panel"]), Path(b["fits"]))
    jobs = [
        {
            "panel": b["panel"],
            "fits": b["fits"],
            "out": str(out),
            "identity": frozen["identity"],
            "origin": w["origin"],
            "lookback": w["lookback"],
        }
        for w in data["windows"]
    ]
    del data
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(real_job, jobs), 1):
            print(json.dumps({"completed": index, "expected": len(jobs), **result}), flush=True)
    aggregate_real(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--fits", type=Path)
    parser.add_argument("--original-report", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = external_path(args.out)
    if args.freeze:
        if any(v is None for v in (args.panel, args.fits, args.original_report)):
            parser.error("Freeze requires --panel --fits --original-report")
        print(json.dumps(freeze(args.panel, args.fits, args.original_report, out), indent=2))
    if args.run:
        run_real(out, args.workers)


if __name__ == "__main__":
    main()
