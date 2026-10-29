"""FF-calibrated factorial mechanism experiment with explicit known loadings.

This is an exploratory conditional simulation, not a new financial validation.
Gaussian independent dates deliberately separate factor correlation, partial-sign
coherence and contemporaneous error dependence. Fits use fixed matched numerical
penalties and the existing audited study engine, not prediction-tuned rankings.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from . import french_estimators as F
    from . import french_rolling_validation as E
    from . import french_report as R
    from . import french_mechanisms as M
    from .french_protocol import sha256, external_path
else:
    import french_estimators as F
    import french_rolling_validation as E
    import french_report as R
    import french_mechanisms as M
    from french_protocol import sha256, external_path


def calibration(out):
    frozen = M.validate_freeze(out)
    path = out / "calibration.npz"
    if path.exists():
        receipt = M.read(out / "calibration.json")
        if receipt["identity"] != frozen["identity"] or receipt["sha256"] != sha256(path):
            raise ValueError("Changed simulation calibration")
        return dict(np.load(path, allow_pickle=False))
    p = M.read(out / "protocol.json")["simulation"]
    panel = Path(frozen["binding"]["panel"])
    x, y = R.frame(panel / "panel/X_weekly.csv"), R.frame(panel / "panel/Y_weekly.csv")
    state = F.prepare(x.iloc[: p["calibration_rows"]], y.iloc[: p["calibration_rows"]])
    beta = np.linalg.lstsq(state.x, state.y, rcond=None)[0].T
    residual = state.y - state.x @ beta.T
    cov = residual.T @ residual / len(residual)
    noise = 0.9 * cov + 0.1 * np.diag(np.diag(cov))
    corr = state.x.T @ state.x / len(state.x)
    beta[np.abs(beta) < 0.05] = 0
    labels = pd.read_csv(panel / "panel/portfolio_labels.csv").set_index("portfolio").loc[y.columns]
    truth_labels = F.canonical_labels(
        2 * (labels.size_decile.to_numpy() > 5) + (labels.bm_decile.to_numpy() > 5)
    )
    signal = np.einsum("ij,jk,ik->i", beta, corr, beta)
    arrays = {
        "beta_seed": beta,
        "factor_corr": corr,
        "noise_covariance": noise,
        "truth_labels": truth_labels,
        "signal_variance": signal,
        "calibration_x": state.x,
        "calibration_y": state.y,
    }
    for coherence, fx, ex in itertools.product(
        p["coherence"], p["factor_correlation_mix"], p["error_dependence_mix"]
    ):
        b = beta.copy()
        if coherence == "coherent":
            for label in np.unique(truth_labels):
                group = np.flatnonzero(truth_labels == label)
                signs = np.where(b[group].sum(0) < 0, -1.0, 1.0)
                b[group] = np.abs(b[group]) * signs
        r = (1 - fx) * np.eye(len(corr)) + fx * corr
        current = np.einsum("ij,jk,ik->i", b, r, b)
        b *= np.sqrt(signal / current)[:, None]
        sigma = (1 - ex) * np.diag(np.diag(noise)) + ex * noise
        key = f"{coherence}_fx{fx}_ex{ex}"
        arrays[key + "__truth"] = b
        arrays[key + "__factor_cov"] = r
        arrays[key + "__error_cov"] = sigma
        np.testing.assert_allclose(
            np.einsum("ij,jk,ik->i", b, r, b), signal, atol=1e-12, rtol=1e-12
        )
        if np.linalg.eigvalsh(r).min() <= 0 or np.linalg.eigvalsh(sigma).min() <= 0:
            raise ValueError("Calibration covariance is not positive definite")
    E.atomic_npz(path, arrays)
    E.atomic_json(
        out / "calibration.json",
        {
            "identity": frozen["identity"],
            "sha256": sha256(path),
            "rows": p["calibration_rows"],
            "first_date": str(x.index[0].date()),
            "last_date": str(x.index[p["calibration_rows"] - 1].date()),
            "no_original_outer_outcomes_used": True,
            "null_cells": int((beta == 0).sum()),
            "cells": beta.size,
            "known_truth_only_inside_this_simulation": True,
        },
    )
    return arrays


def inventory(p):
    return [
        {
            "lookback": t,
            "replication": rep,
            "coherence": coh,
            "fx": fx,
            "ex": ex,
            "key": f"T{t}_r{rep:03d}_{coh}_fx{fx}_ex{ex}",
        }
        for t, rep, coh, fx, ex in itertools.product(
            p["lookbacks"],
            range(p["replications"]),
            p["coherence"],
            p["factor_correlation_mix"],
            p["error_dependence_mix"],
        )
    ]


def generate(cal, job, p):
    key = f"{job['coherence']}_fx{job['fx']}_ex{job['ex']}"
    beta, corr, noise = (cal[key + "__" + name] for name in ("truth", "factor_cov", "error_cov"))
    rng = np.random.default_rng(
        np.random.SeedSequence([p["seed"], job["lookback"], job["replication"]])
    )
    n, q, k, nt = job["lookback"], beta.shape[0], beta.shape[1], p["test_rows"]
    zx, ze, zt, zet = (rng.standard_normal(shape) for shape in ((n, k), (n, q), (nt, k), (nt, q)))
    x, xt = zx @ np.linalg.cholesky(corr).T, zt @ np.linalg.cholesky(corr).T
    y, yt = (
        x @ beta.T + ze @ np.linalg.cholesky(noise).T,
        xt @ beta.T + zet @ np.linalg.cholesky(noise).T,
    )
    return x, y, xt, yt, beta, corr, noise


def known_metrics(beta, intercept, truth, corr, gates, active, ytest, pred, tolerance):
    delta = beta - truth
    selected = np.abs(beta) > tolerance
    tp, fp = int((selected & active).sum()), int((selected & ~active).sum())
    a, n = int(active.sum()), int((~active).sum())
    return {
        "loading_mse": float(np.square(delta).mean()),
        "excess_risk": float(
            (np.einsum("ij,jk,ik->", delta, corr, delta) + intercept @ intercept) / len(beta)
        ),
        "test_mse": float(np.square(ytest - pred).mean()),
        "correct_active_fitted_sign": float((active & selected & (beta * truth > 0)).sum() / a),
        "support_precision": tp / (tp + fp) if tp + fp else None,
        "support_recall": tp / a,
        "support_false_positive_rate": fp / n if n else None,
        "active_fraction": float(selected.mean()),
        "gate_correct_active": float((active & (gates * truth > 0)).sum() / a),
        "gate_opposite_active": float((active & (gates * truth < 0)).sum() / a),
        "false_exclusion": float((active & (gates == 0)).sum() / a),
        "active_unrestricted": float((active & np.isnan(gates)).sum() / a),
        "true_active_cells": a,
        "true_null_cells": n,
    }


def simulation_job(job):
    out = Path(job["out"])
    folder = out / "simulation" / job["key"]
    done = folder / "receipt.json"
    if done.exists():
        receipt = M.read(done)
        if receipt["identity"] != job["identity"]:
            raise ValueError("Stale simulation checkpoint")
        for name, value in receipt["files_sha256"].items():
            if sha256(folder / name) != value:
                raise ValueError("Corrupt simulation checkpoint")
        return {"key": job["key"], "resumed": True}
    folder.mkdir(parents=True, exist_ok=True)
    p = M.read(out / "protocol.json")
    sim = p["simulation"]
    original = M.read(Path(M.read(out / "freeze.json")["binding"]["panel"]) / "protocol.json")
    cal = dict(np.load(out / "calibration.npz", allow_pickle=False))
    x, y, xt, yt, truth, corr, noise = generate(cal, job, sim)
    dates = pd.date_range("2000-01-05", periods=len(x), freq="W-WED")
    state = F.prepare(pd.DataFrame(x, index=dates), pd.DataFrame(y, index=dates))
    arrays = {
        "x": x,
        "y": y,
        "x_test": xt,
        "y_test": yt,
        "truth": truth,
        "factor_cov": corr,
        "error_cov": noise,
        "training_y_sd": state.y_sd,
    }
    rows, ledger, cache = [], {}, {}
    for partition in sim["partitions"]:
        labels = (
            cal["truth_labels"] if partition == "known_quadrants" else state.labels(p["ward_cut"])
        )
        ari = R.adjusted_rand(labels, cal["truth_labels"])
        weights = np.array(
            [np.sqrt(np.sum(labels == label) / len(np.unique(labels))) for label in labels]
        )
        for family in sim["families"]:
            base = F.Spec(
                "row" if family == "hcgl" else "block",
                labels,
                weights,
                np.full_like(truth, np.nan),
                0.0,
            )
            base.scale = F.kkt_scale(state, base)
            for fraction in sim["penalty_fractions"]:
                lam = fraction * base.scale
                for variant in M.VARIANTS:
                    spec = M.gate_spec(state, base, variant, p["gate_tau"])
                    key = (
                        ("zero" if lam == 0 else spec.key())
                        + F.array_hash(np.nan_to_num(spec.signs, nan=2.0))
                        + float(lam).hex()
                    )
                    if key not in cache:
                        cache[key] = F.fit_path(state, spec, [lam], original)[0]
                    fitted = cache[key]
                    beta, intercept = F.original_units(state, fitted)
                    name = f"{partition}__{family}__{fraction}__{variant}"
                    pred = xt @ beta.T + intercept
                    # Use raw coefficient tolerance matching the truth's fixed units.
                    metrics = known_metrics(
                        beta,
                        intercept,
                        truth,
                        corr,
                        spec.signs,
                        truth != 0,
                        yt,
                        pred,
                        p["support_tolerance"],
                    )
                    rows.append(
                        {
                            **{
                                k: job[k]
                                for k in ("lookback", "replication", "coherence", "fx", "ex")
                            },
                            "partition": partition,
                            "family": family,
                            "fraction": fraction,
                            "variant": variant,
                            "lambda": lam,
                            "clusters": len(np.unique(labels)),
                            "ari": ari,
                            **metrics,
                        }
                    )
                    arrays[name + "__beta"], arrays[name + "__intercept"] = beta, intercept
                    arrays[name + "__gates"], arrays[name + "__labels"] = spec.signs, labels
                    ledger[name] = {k: v for k, v in fitted.items() if k not in ("beta", "alpha")}
    E.atomic_npz(folder / "arrays.npz", arrays)
    E.atomic_json(folder / "metrics.json", E.safe_json(rows))
    E.atomic_json(folder / "solver_ledger.json", E.safe_json(ledger))
    E.atomic_json(
        done,
        {
            "identity": job["identity"],
            "status": "complete",
            "design": job,
            "files_sha256": {
                name: sha256(folder / name)
                for name in ("arrays.npz", "metrics.json", "solver_ledger.json")
            },
        },
    )
    return {"key": job["key"], "resumed": False}


def aggregate(out):
    frozen = M.validate_freeze(out)
    p = M.read(out / "protocol.json")["simulation"]
    rows, receipts = [], {}
    for job in inventory(p):
        folder = out / "simulation" / job["key"]
        receipt = M.read(folder / "receipt.json")
        if receipt["identity"] != frozen["identity"]:
            raise ValueError("Mixed simulation jobs")
        for name, value in receipt["files_sha256"].items():
            if sha256(folder / name) != value:
                raise ValueError("Changed simulation result")
        rows.extend(M.read(folder / "metrics.json"))
        receipts[job["key"]] = sha256(folder / "receipt.json")
    df = pd.DataFrame(rows)
    keys = ["lookback", "coherence", "fx", "ex", "partition", "family", "fraction"]
    cols = [
        "loading_mse",
        "excess_risk",
        "test_mse",
        "correct_active_fitted_sign",
        "support_precision",
        "support_recall",
        "support_false_positive_rate",
        "active_fraction",
        "gate_correct_active",
        "gate_opposite_active",
        "false_exclusion",
        "active_unrestricted",
    ]
    summary = []
    for key, g in df.groupby(keys + ["variant"], sort=False):
        base = df
        for name, value in zip(keys, key[:-1]):
            base = base[base[name] == value]
        base = base[base.variant == "unconstrained"].set_index("replication")
        g = g.set_index("replication").sort_index()
        if len(g) != p["replications"] or not g.index.is_unique:
            raise ValueError("Missing/duplicate simulation replications")
        for metric in cols:
            values = g[metric].to_numpy(dtype=float)
            diffs = values - base.loc[g.index, metric].to_numpy(dtype=float)
            valid, valid_diff = values[np.isfinite(values)], diffs[np.isfinite(diffs)]
            se = float(valid.std(ddof=1) / np.sqrt(len(valid))) if len(valid) > 1 else np.nan
            dse = (
                float(valid_diff.std(ddof=1) / np.sqrt(len(valid_diff)))
                if len(valid_diff) > 1
                else np.nan
            )
            mean, dm = (
                float(valid.mean()) if len(valid) else np.nan,
                float(valid_diff.mean()) if len(valid_diff) else np.nan,
            )
            summary.append(
                {
                    **dict(zip(keys + ["variant"], key)),
                    "metric": metric,
                    "replications": len(g),
                    "defined": len(valid),
                    "mean": mean,
                    "mc_se": se,
                    "paired_difference": dm,
                    "paired_mc_se": dse,
                    "paired_lower": dm - 1.96 * dse,
                    "paired_upper": dm + 1.96 * dse,
                }
            )
    df.to_csv(out / "simulation_replications.csv", index=False, float_format="%.17g")
    table = pd.DataFrame(summary)
    table.to_csv(out / "simulation_summary.csv", index=False, float_format="%.17g")
    pairs = []
    for key, g in df.groupby(keys, sort=False):
        for variant, reference in M.read(out / "protocol.json")["pooling_pairs"]:
            left = g[g.variant == variant].set_index("replication").sort_index()
            right = g[g.variant == reference].set_index("replication").sort_index()
            for metric in cols:
                difference = (left[metric] - right[metric]).dropna()
                mean = float(difference.mean()) if len(difference) else np.nan
                se = (
                    float(difference.std(ddof=1) / np.sqrt(len(difference)))
                    if len(difference) > 1
                    else np.nan
                )
                pairs.append(
                    {
                        **dict(zip(keys, key)),
                        "variant": variant,
                        "reference": reference,
                        "metric": metric,
                        "defined": len(difference),
                        "difference": mean,
                        "mc_se": se,
                        "lower": mean - 1.96 * se,
                        "upper": mean + 1.96 * se,
                    }
                )
    pd.DataFrame(pairs).to_csv(
        out / "simulation_pooling_contrasts.csv", index=False, float_format="%.17g"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, t in enumerate(p["lookbacks"]):
        for col, coh in enumerate(p["coherence"]):
            ax = axes[row, col]
            selected = table[
                (table.lookback == t)
                & (table.coherence == coh)
                & (table.partition == "learned_ward")
                & (table.fraction == 0.1)
                & (table.metric == "loading_mse")
                & table.variant.isin(["pooled_exclude", "pooled_release"])
            ]
            for index, r in enumerate(selected.itertuples()):
                ax.errorbar(
                    r.paired_difference,
                    index,
                    xerr=1.96 * r.paired_mc_se,
                    fmt="o",
                    color="C0" if r.variant == "pooled_exclude" else "C1",
                )
            ax.set_yticks(
                range(len(selected)),
                [
                    f"{r.family} F{r.fx} E{r.ex} {r.variant.replace('pooled_', '')}"
                    for r in selected.itertuples()
                ],
                fontsize=7,
            )
            ax.axvline(0, color="grey", lw=1)
            ax.set(title=f"T={t}; {coh}", xlabel="Loading-MSE difference vs matched unconstrained")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / ("simulation_loading_effects." + suffix), bbox_inches="tight")
    plt.close(fig)
    E.atomic_json(
        out / "simulation_manifest.json",
        {
            "identity": frozen["identity"],
            "status": "produced; verification required",
            "jobs": len(receipts),
            "rows": len(df),
            "receipt_sha256": receipts,
            "calibration_sha256": sha256(out / "calibration.npz"),
            "files_sha256": {
                name: sha256(out / name)
                for name in (
                    "simulation_replications.csv",
                    "simulation_summary.csv",
                    "simulation_pooling_contrasts.csv",
                    "simulation_loading_effects.png",
                    "simulation_loading_effects.pdf",
                )
            },
        },
    )


def run(out, workers=4):
    frozen = M.validate_freeze(out)
    calibration(out)
    jobs = [
        {**j, "out": str(out), "identity": frozen["identity"]}
        for j in inventory(M.read(out / "protocol.json")["simulation"])
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(simulation_job, jobs), 1):
            print(json.dumps({"completed": index, "expected": len(jobs), **result}), flush=True)
    aggregate(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(external_path(args.out), args.workers)


if __name__ == "__main__":
    main()
