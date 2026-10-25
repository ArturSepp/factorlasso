"""Paired variance sensitivity for existing pooled-sign applications.

Run from the repository root with an external output directory::

    python -m benchmarks.compare_sign_variance --out C:/runtime/sign-variance

The two fits in every pair differ only in ``auto_sign_variance``. This is
conditional sensitivity, not tuning or a performance-superiority experiment.
Canonical examples and archived paper results are never changed. The yeast
comparison uses the revised, training-only preprocessing on nine outer splits;
its fixed penalty fraction is not nested model selection or a replacement for
the paper's nested benchmark.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import runpy
import subprocess
import time

import numpy as np
import pandas as pd

import factorlasso as fl
from factorlasso.utils._panel import get_x_y_np
from examples.docs import app_multi_asset_credit_attribution as credit
from examples.docs import gated_cluster_pooled_signs as gated
from examples.docs import quickstart
from papers.sign_pooling_2026.replication import eqtl_validation as eqtl
from papers.sign_pooling_2026.replication.cvxpy_shape_compat import safe_cvxpy_sum_shape


REPO = Path(__file__).resolve().parents[1]
TOLERANCES = (1e-6, 1e-5, 1e-4)
VARIANCES = ("independent", "date")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_revision(repo: Path) -> str:
    """Report Git provenance when available; exported source is hash-identified."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable (source export; use source_hashes_start)"


def json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.to_dict()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=json_default), encoding="utf-8")


def reference_statistics(x: np.ndarray, y: np.ndarray, span, variance: str):
    """Direct residual-cell calculation, independent of package moment expansions."""
    weights = (np.ones(len(x)) if span is None else
               (1 - 2 / (span + 1)) ** np.arange(len(x) - 1, -1, -1))
    slopes, t_stats = [], []
    for predictor in x.T:
        observed = np.isfinite(predictor[:, None]) & np.isfinite(y)
        dates = observed.any(axis=1)
        xp = np.where(np.isfinite(predictor), predictor, 0.)
        counts = observed.sum(axis=1)
        denominator = float(np.sum(weights * counts * xp ** 2))
        numerator = float(np.sum(weights[:, None] * xp[:, None]
                                 * np.where(observed, y, 0.)))
        slope = numerator / denominator if denominator else 0.
        residual = np.where(observed, y - slope * xp[:, None], 0.)
        mass = float(weights[dates].sum())
        ess = mass ** 2 / np.sum(weights[dates] ** 2) if mass else 0.
        if denominator == 0 or ess <= 1 + 1e-12:
            standard_error = np.inf
        elif variance == "date":
            score = weights * xp * residual.sum(axis=1)
            standard_error = np.sqrt(np.sum(score ** 2) * ess / (ess - 1)) / denominator
        elif span is None:
            df = max(int(observed.sum() - observed.any(axis=0).sum()), 1)
            standard_error = np.sqrt(np.sum(residual ** 2) / (df * denominator))
        else:
            scores = weights[:, None] * xp[:, None] * residual
            cell_mass = np.sum(weights * counts)
            cell_ess = cell_mass ** 2 / np.sum(weights ** 2 * counts)
            q = observed.any(axis=0).sum()
            correction = cell_ess / (cell_ess - q) if cell_ess > q + 1e-12 else 1.
            standard_error = np.sqrt(np.sum(scores ** 2) * correction) / denominator
        statistic = (slope / standard_error if standard_error > 0 else
                     np.sign(slope) * np.inf if slope else 0.)
        slopes.append(slope)
        t_stats.append(statistic)
    return np.array(slopes), np.array(t_stats)


def verify_statistics(model, x, y) -> dict:
    """Check each fitted pool against explicit residual sums on the fitted panel."""
    xp, yp, valid = get_x_y_np(x, y, span=model.span, demean=model.demean)
    offset = len(x) - len(xp)
    xp = np.where(x.iloc[offset:].notna(), xp, np.nan)
    yp = np.where(valid > 0, yp, np.nan)
    labels = (model.clusters_.to_numpy() if model.clusters_ is not None
              else np.arange(y.shape[1]))
    max_slope_error, max_statistic_error = 0., 0.
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        slope, statistic = reference_statistics(
            xp, yp[:, members], model.effective_sign_span_, model.auto_sign_variance)
        actual_slopes = model.sign_slopes_.iloc[members].to_numpy()
        actual_stats = model.sign_t_stats_.iloc[members].to_numpy()
        np.testing.assert_allclose(actual_slopes, np.broadcast_to(slope, actual_slopes.shape),
                                   rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(actual_stats, np.broadcast_to(statistic, actual_stats.shape),
                                   rtol=1e-9, atol=1e-10)
        signs = np.sign(slope)
        if model.auto_sign_threshold_t is not None and model.auto_sign_threshold_t > 0:
            signs[np.abs(statistic) < model.auto_sign_threshold_t] = 0.
        np.testing.assert_array_equal(model.detected_signs_.iloc[members].to_numpy(),
                                      np.broadcast_to(signs, actual_slopes.shape))
        max_slope_error = max(max_slope_error, float(np.max(abs(actual_slopes - slope))))
        max_statistic_error = max(max_statistic_error,
                                  float(np.max(abs(actual_stats - statistic))))
    return dict(max_slope_error=max_slope_error, max_statistic_error=max_statistic_error,
                pools=int(len(np.unique(labels))))


def relative_change(before, after) -> float:
    denominator = np.linalg.norm(before)
    return float(np.linalg.norm(after - before) / denominator) if denominator else float("nan")


def sign_counts(values) -> dict:
    """Keep finite signed constraints, zero exclusions and NaN releases distinct."""
    signs = np.asarray(values, dtype=float)
    return dict(signed=int(np.count_nonzero(np.isfinite(signs) & (signs != 0))),
                excluded=int(np.count_nonzero(signs == 0)),
                unrestricted=int(np.count_nonzero(np.isnan(signs))))


def changed_signs(before, after) -> np.ndarray:
    """Two NaN releases express the same constraint, so they are unchanged."""
    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    return ~((before == after) | (np.isnan(before) & np.isnan(after)))


def metrics(actual: np.ndarray, predicted: np.ndarray, baseline: np.ndarray) -> dict:
    valid = np.isfinite(actual)
    square_error = np.where(valid, (actual - predicted) ** 2, 0.)
    baseline_error = np.where(valid, (actual - baseline) ** 2, 0.)
    ratios = np.divide(square_error.sum(axis=0), baseline_error.sum(axis=0),
                       out=np.full(actual.shape[1], np.nan),
                       where=baseline_error.sum(axis=0) > 0)
    return dict(mse=float(square_error.sum() / valid.sum()),
                pooled_r2=float(1 - square_error.sum() / baseline_error.sum()),
                median_response_r2=float(np.nanmedian(1 - ratios)))


class Comparison:
    def __init__(self, out):
        self.out = out
        self.rows = []
        self.manifests = []

    @safe_cvxpy_sum_shape()
    def pair(self, case, x, y, template, *, test_x=None, test_y=None, metadata=None,
             external_clusters=None, covariance=False, portfolio_weights=None,
             prediction_scale=None, prediction_mean=None):
        """Fit both variance modes with identical data and remaining settings."""
        started = time.perf_counter()
        folder = self.out / case
        folder.mkdir()
        model_config = template.get_params()
        write_json(folder / "configuration.json", model_config)
        x.to_csv(folder / "train_x.csv")
        y.to_csv(folder / "train_y.csv")
        if test_x is not None:
            test_x.to_csv(folder / "test_x.csv")
            test_y.to_csv(folder / "test_y.csv")
        models, references, predictions, covariances = {}, {}, {}, {}
        actual = y if test_y is None else test_y
        target = x if test_x is None else test_x
        baseline = (y.mean().to_numpy() if prediction_mean is None else prediction_mean)
        row = dict(case=case, n_train=len(x), n_test=0 if test_x is None else len(test_x),
                   n_responses=y.shape[1], n_factors=x.shape[1],
                   evaluation="in_sample" if test_x is None else "held_out",
                   reg_lambda=template.reg_lambda)
        for variance in VARIANCES:
            fitted = template.copy(kwargs={"auto_sign_variance": variance})
            kwargs = {} if external_clusters is None else dict(external_clusters=external_clusters)
            fitted.fit(x=x, y=y, **kwargs)
            models[variance] = fitted
            references[variance] = verify_statistics(fitted, x, y)
            fitted.coef_.to_csv(folder / f"coef_{variance}.csv")
            fitted.detected_signs_.to_csv(folder / f"detected_signs_{variance}.csv")
            fitted.derived_signs_.to_csv(folder / f"final_signs_{variance}.csv")
            fitted.sign_t_stats_.to_csv(folder / f"statistics_{variance}.csv")
            fitted.sign_slopes_.to_csv(folder / f"slopes_{variance}.csv")
            manual = target.to_numpy() @ fitted.coef_.to_numpy().T
            if fitted.demean:
                manual += fitted.alpha_const_.to_numpy()
            prediction = fitted.predict(target).to_numpy()
            np.testing.assert_allclose(prediction, manual, rtol=1e-12, atol=1e-12)
            if prediction_scale is not None:
                prediction = prediction * prediction_scale + prediction_mean
            predictions[variance] = prediction
            pd.DataFrame(prediction, index=actual.index, columns=actual.columns).to_csv(
                folder / f"predictions_{variance}.csv")
            score = metrics(actual.to_numpy(), prediction, baseline)
            row.update({f"{key}_{variance}": value for key, value in score.items()})
            for stage, signs in (("detected", fitted.detected_signs_),
                                 ("final", fitted.derived_signs_)):
                counts = sign_counts(signs)
                row.update({f"{stage}_{kind}_{variance}": count
                            for kind, count in counts.items()})
                # Preserve the original column as an alias for finite signed cells.
                row[f"{stage}_active_{variance}"] = counts["signed"]
            for tolerance in TOLERANCES:
                support = np.abs(fitted.coef_.to_numpy()) > tolerance
                row[f"support_{tolerance:g}_{variance}"] = int(support.sum())
            if covariance:
                container = quickstart.assemble_covariance(fitted, x)
                covariance_array = container.get_y_covar().to_numpy()
                direct = (fitted.coef_.to_numpy() @ container.x_covar.to_numpy()
                          @ fitted.coef_.to_numpy().T
                          + np.diag(12 * fitted.estimation_result_.ss_res))
                np.testing.assert_allclose(covariance_array, direct, rtol=1e-12, atol=1e-12)
                covariances[variance] = covariance_array
                np.savetxt(folder / f"covariance_{variance}.csv", covariance_array, delimiter=",")
            if portfolio_weights is not None:
                exposure = portfolio_weights @ fitted.coef_
                exposure.to_csv(folder / f"portfolio_exposure_{variance}.csv")
                for factor, value in exposure.items():
                    row[f"portfolio_{factor}_{variance}"] = value
        first, second = (models[v] for v in VARIANCES)
        np.testing.assert_allclose(first.sign_slopes_, second.sign_slopes_, rtol=0, atol=0)
        if first.clusters_ is not None:
            pd.testing.assert_series_equal(first.clusters_, second.clusters_)
            first.clusters_.to_csv(folder / "clusters.csv")
        raw_a, raw_b = first.detected_signs_.to_numpy(), second.detected_signs_.to_numpy()
        final_a, final_b = first.derived_signs_.to_numpy(), second.derived_signs_.to_numpy()
        raw_changed = changed_signs(raw_a, raw_b)
        row.update(detected_changed=int(np.count_nonzero(raw_changed)),
                   detected_fraction=float(np.mean(raw_changed)),
                   newly_excluded=int(np.count_nonzero(
                       np.isfinite(raw_a) & (raw_a != 0) & (raw_b == 0))),
                   newly_retained=int(np.count_nonzero(
                       (raw_a == 0) & np.isfinite(raw_b) & (raw_b != 0))),
                   final_constraints_changed=int(np.count_nonzero(changed_signs(final_a, final_b))),
                   coefficient_relative_change=relative_change(first.coef_, second.coef_),
                   coefficient_max_abs_change=float(np.max(abs(second.coef_ - first.coef_))),
                   prediction_relative_change=relative_change(predictions["independent"],
                                                               predictions["date"]),
                   mse_difference=row["mse_date"] - row["mse_independent"],
                   pooled_r2_difference=row["pooled_r2_date"] - row["pooled_r2_independent"])
        for tolerance in TOLERANCES:
            a = np.abs(first.coef_.to_numpy()) > tolerance
            b = np.abs(second.coef_.to_numpy()) > tolerance
            row[f"support_changed_{tolerance:g}"] = int(np.count_nonzero(a != b))
        if covariance:
            row["covariance_relative_change"] = relative_change(covariances["independent"],
                                                               covariances["date"])
        row["seconds"] = time.perf_counter() - started
        record = dict(case=case, metadata=metadata or {},
                      train_ids=[str(i) for i in x.index],
                      test_ids=[] if test_x is None else [str(i) for i in test_x.index],
                      reference_checks=references, result=row)
        write_json(folder / "receipt.json", record)
        self.rows.append(row)
        self.manifests.append(record)
        pd.DataFrame(self.rows).to_csv(self.out / "paired_results.csv", index=False)
        print(json.dumps(row, default=json_default), flush=True)


def synthetic_applications(comparison):
    x, y, _ = credit.make_panel()
    for method, mode, centre in [
        ("hcgl_zero", fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO, None),
        ("hcgl_prior", fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
         credit.credit_prior(y.columns)),
        ("fcgl_prior", fl.LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
         credit.credit_prior(y.columns)),
    ]:
        for penalty in credit.REG_LAMBDAS:
            template = credit.build_model(mode, penalty, centre)
            comparison.pair(f"credit_{method}_{penalty:g}", x, y, template,
                            metadata=dict(source="examples/docs/app_multi_asset_credit_attribution.py",
                                          seed=credit.SEED, data="synthetic monthly returns",
                                          penalty="unchanged canonical four-point path"))
    x, y = quickstart.make_panel()
    for suffix, train_size in [("full", len(x)), ("holdout", 96)]:
        train_x, train_y = x.iloc[:train_size], y.iloc[:train_size]
        selector = quickstart.select_and_fit(train_x, train_y)
        selector.cv_scores_.to_csv(comparison.out / f"quickstart_{suffix}_training_cv.csv")
        comparison.pair(f"quickstart_{suffix}", train_x, train_y, selector.best_model_,
                        test_x=None if suffix == "full" else x.iloc[train_size:],
                        test_y=None if suffix == "full" else y.iloc[train_size:], covariance=True,
                        metadata=dict(source="examples/docs/quickstart.py", seed=quickstart.SEED,
                                      data="synthetic monthly decimal returns",
                                      selection="independent CV on this training set only; "
                                                "same selected numerical lambda for both modes",
                                      penalty_grid=quickstart.PENALTY_GRID.tolist()))
    # Importing this teaching script runs its unchanged 30-draw example. Reuse
    # its exact input panels and unfitted template; compare only point estimates.
    production = runpy.run_path(str(REPO / "examples/docs/production_bootstrap.py"))
    x, y, template = (production[name] for name in ("x", "y", "model"))
    for suffix, train_size in [("full", len(x)), ("holdout", 54)]:
        comparison.pair(f"production_{suffix}", x.iloc[:train_size], y.iloc[:train_size], template,
                        test_x=None if suffix == "full" else x.iloc[train_size:],
                        test_y=None if suffix == "full" else y.iloc[train_size:],
                        portfolio_weights=production["weights"],
                        metadata=dict(source="examples/docs/production_bootstrap.py", seed=401,
                                      data="synthetic; first 12 Asset B values missing",
                                      scope="point estimator; intervals not compared",
                                      sign_weighting="equal weights on EWMA-demeaned fit panel"))
    x, y, _, groups = gated.make_panel(np.random.default_rng(gated.SEED))
    template = fl.LassoModel(model_type=fl.LassoModelType.GROUP_LASSO, group_data=groups,
                            reg_lambda=1e-3, auto_sign_constraints=True, auto_sign_threshold_t=.75)
    comparison.pair("gate_teaching_panel", x, y, template,
                    metadata=dict(source="examples/docs/gated_cluster_pooled_signs.py",
                                  seed=gated.SEED, data="synthetic known coherent groups"))


def yeast_application(comparison, solver):
    y_all, x, positions = eqtl.load_data()
    genes, _ = eqtl.select_mapk_genes(y_all)
    y = y_all[genes]
    protocol = eqtl.Protocol(solver=solver)
    for repeat in range(protocol.repeats):
        for fold, (train, test) in enumerate(eqtl.splits(
                len(x), protocol.outer_folds, protocol.seed + 1009 * repeat)):
            state = eqtl.fit_preprocessing(x.iloc[train], y.iloc[train], positions, protocol)
            for method in ("hcgl", "fcgl"):
                maximum = eqtl.lambda_max(state, method + "_unconstrained", protocol)
                template = fl.LassoModel(
                    model_type=(fl.LassoModelType.GROUP_LASSO if method == "hcgl" else
                                fl.LassoModelType.FACTOR_CLUSTER_GROUP_LASSO),
                    group_data=state.clusters if method == "hcgl" else None,
                    reg_lambda=.3 * maximum, demean=False, warmup_period=0,
                    solver=solver, solver_fallbacks=None, loss_normalization="sample",
                    auto_sign_constraints=True, auto_sign_threshold_t=protocol.gate_threshold,
                    auto_sign_adaptive_weights=False, group_penalty="normalized",
                    l1_weight=protocol.l1_weight)
                comparison.pair(
                    f"yeast_{method}_r{repeat}_f{fold}", state.x, state.y, template,
                    test_x=state.transform_x(x.iloc[test]), test_y=y.iloc[test],
                    external_clusters=None if method == "hcgl" else state.clusters,
                    prediction_scale=state.y_sd.to_numpy(), prediction_mean=state.y_mean.to_numpy(),
                    metadata=dict(source="papers/sign_pooling_2026/replication/eqtl_validation.py",
                                  data="Brem and Kruglyak yeast cross; 64 measured MAPK genes",
                                  protocol=asdict(protocol), preprocessing=state.manifest(),
                                  repeat=repeat, fold=fold, lambda_max_unconstrained=maximum,
                                  penalty_fraction=.3, penalty="common unconstrained KKT scale; "
                                  "fixed sensitivity setting, not nested tuning",
                                  implementation="HCGL supplied learned groups via GROUP_LASSO; "
                                  "FCGL uses same external groups"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--case", choices=("all", "synthetic", "yeast"), default="all")
    parser.add_argument("--yeast-solver", default="MOSEK")
    args = parser.parse_args()
    out = args.out.resolve()
    if "onedrive" in str(out).lower():
        raise ValueError("Write generated results outside OneDrive")
    out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), *sorted((REPO / "src/factorlasso").rglob("*.py")),
               *(REPO / "examples/docs" / name for name in (
                   "app_multi_asset_credit_attribution.py", "quickstart.py",
                   "production_bootstrap.py",
                   "gated_cluster_pooled_signs.py")),
               *(REPO / "papers/sign_pooling_2026/replication" / name for name in (
                   "eqtl_pipeline.py", "eqtl_validation.py", "cvxpy_shape_compat.py"))]
    inputs = [REPO / "papers/sign_pooling_2026/replication/data" / name
              for name in ("yeast_full.npz", "yeast.rda")]
    start_sources = {str(path.relative_to(REPO)): sha256(path) for path in sources}
    for source in sources:
        snapshot = out / "source_snapshot" / source.relative_to(REPO)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
    comparison = Comparison(out)
    if args.case in ("all", "synthetic"):
        synthetic_applications(comparison)
    if args.case in ("all", "yeast"):
        yeast_application(comparison, args.yeast_solver)
    end_sources = {str(path.relative_to(REPO)): sha256(path) for path in sources}
    changed = {name: dict(before=start_sources[name], after=digest)
               for name, digest in end_sources.items() if start_sources[name] != digest}
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
        package_source=str(Path(fl.__file__).resolve()),
        declared_checkout_version=re.search(r'^version\s*=\s*"([^"]+)"',
            (REPO / "pyproject.toml").read_text(encoding="utf-8"), flags=re.MULTILINE).group(1),
        versions={name: importlib.metadata.version(name)
                  for name in ("factorlasso", "numpy", "pandas", "scipy", "cvxpy", "clarabel")},
        git_head=source_revision(REPO),
        source_hashes_start=start_sources, source_hashes_end=end_sources,
        sources_changed_during_run=changed,
        input_hashes={str(path.relative_to(REPO)): sha256(path) for path in inputs},
        cases=comparison.manifests, support_tolerances=TOLERANCES,
        interpretation="paired fixed-setting sensitivity, no general superiority claim; "
                       "synthetic full fits are in-sample; yeast splits are not replication",
        output_hashes={str(path.relative_to(out)): sha256(path)
                       for path in sorted(out.rglob("*")) if path.is_file()})
    write_json(out / "MANIFEST.json", manifest)
    if changed:
        raise RuntimeError(f"Source files changed during run; inspect manifest: {list(changed)}")
    print(f"Completed {len(comparison.rows)} variance pairs: {out}", flush=True)


if __name__ == "__main__":
    main()
