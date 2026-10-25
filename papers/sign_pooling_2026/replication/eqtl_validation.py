"""Leakage-free nested validation of the current sign gate on yeast eQTL data.

This is separate from ``eqtl_pipeline.py``, which preserves the archived experiment.
Every inner and outer training split starts from the raw marker panel and refits
LD thinning, outcome screening, scaling, response clustering and sign derivation.
No outer-held-out outcome participates in selection or tuning. The predeclared
gate and cluster settings are not tuned; only the penalty fraction is selected.

The loss is SSE / n, with no adaptive weights. HCGL uses response-row norms,
FCGL uses cluster-by-marker norms; both use sqrt(cluster size / cluster count).
The lambda grid is relative to each method's training-only zero-solution KKT
threshold. Thus the same grid fractions do not imply the same numerical penalty.
The date-score gate assumes independent segregants and is a screen, not a test
of conditional coefficient signs. All clustering uses Euclidean distances between
training-standardized response vectors, divided by sqrt(n), and Ward linkage.
The cut is a fraction of the final linkage height, not maximum pairwise distance.

Run as a module from the repository root with an explicit external output folder.
Outputs are new research results; frozen reference files are never read or changed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

import factorlasso
from factorlasso import LassoModel, LassoModelType, derive_sign_constraints

if __package__:
    from .eqtl_pipeline import ld_block_markers, load_data, screen_markers, select_mapk_genes
    from .paths import DATA_DIR, REPOSITORY_DIR
    from .cvxpy_shape_compat import safe_cvxpy_sum_shape
else:
    from eqtl_pipeline import ld_block_markers, load_data, screen_markers, select_mapk_genes
    from paths import DATA_DIR, REPOSITORY_DIR
    from cvxpy_shape_compat import safe_cvxpy_sum_shape


METHODS = (
    "lasso",
    "singleton_exclude",
    "hcgl_unconstrained",
    "hcgl_exclude",
    "hcgl_unrestricted",
    "fcgl_exclude",
)
# Keep the original six-method protocol stable; follow-up studies opt into these.
AVAILABLE_METHODS = METHODS + ("hcgl_singleton_exclude", "fcgl_unconstrained")


@dataclass(frozen=True)
class Protocol:
    """Predeclared design; repeated splits describe split variation, not iid trials."""

    repeats: int = 3
    outer_folds: int = 3
    inner_folds: int = 3
    seed: int = 20261024
    lambda_fractions: tuple[float, ...] = (1.0, 0.3, 0.1, 0.03)
    methods: tuple[str, ...] = METHODS
    solver: str = "MOSEK"
    gate_threshold: float = 2.0
    cluster_cut: float = 0.7
    l1_weight: float = 0.5
    ld_correlation: float = 0.92
    screen_p: float = 0.01
    screen_min_genes: int = 2
    coefficient_tolerance: float = 1e-6
    random_partitions: int = 100

    def __post_init__(self):
        if self.repeats < 1 or min(self.outer_folds, self.inner_folds) < 2:
            raise ValueError("At least one repeat and two inner/outer folds are required")
        if not self.methods or any(m not in AVAILABLE_METHODS for m in self.methods):
            raise ValueError(f"methods must be a nonempty subset of {AVAILABLE_METHODS}")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("methods must not contain duplicates")
        if not self.lambda_fractions or any(not 0 < f <= 1 for f in self.lambda_fractions):
            raise ValueError("lambda_fractions must lie in (0, 1]")
        if len(set(self.lambda_fractions)) != len(self.lambda_fractions):
            raise ValueError("lambda_fractions must not contain duplicates")
        if not 0 < self.l1_weight <= 1 or not 0 < self.cluster_cut <= 1:
            raise ValueError("l1_weight and cluster_cut must lie in (0, 1]")
        if not 0 < self.screen_p < 1 or not 0 < self.ld_correlation <= 1:
            raise ValueError("Invalid screening or LD threshold")
        if min(self.screen_min_genes, self.random_partitions) < 1:
            raise ValueError("screen_min_genes and random_partitions must be positive")
        if self.gate_threshold < 0 or self.coefficient_tolerance <= 0:
            raise ValueError("Invalid gate threshold or coefficient tolerance")


@dataclass
class TrainingState:
    """All learned preprocessing for one particular training set."""

    x: pd.DataFrame
    y: pd.DataFrame
    x_mean: pd.Series
    x_sd: pd.Series
    y_mean: pd.Series
    y_sd: pd.Series
    clusters: pd.Series
    singleton_signs: pd.DataFrame
    pooled_signs: pd.DataFrame
    ld_markers: list[str]

    def transform_x(self, x: pd.DataFrame) -> pd.DataFrame:
        return (x.loc[:, self.x.columns] - self.x_mean) / self.x_sd

    def manifest(self) -> dict:
        return dict(
            train_ids=self.x.index.tolist(),
            ld_markers=self.ld_markers,
            selected_markers=self.x.columns.tolist(),
            cluster_by_gene={str(k): int(v) for k, v in self.clusters.items()},
            x_mean=self.x_mean.to_dict(),
            x_sd=self.x_sd.to_dict(),
            y_mean=self.y_mean.to_dict(),
            y_sd=self.y_sd.to_dict(),
        )


def _clusters(y_z: pd.DataFrame, cut: float) -> pd.Series:
    if y_z.shape[1] == 1:
        labels = np.ones(1, dtype=int)
    else:
        distances = pdist(y_z.to_numpy().T) / np.sqrt(len(y_z))
        tree = linkage(distances, method="ward")
        labels = fcluster(tree, cut * tree[-1, 2], criterion="distance")
    return pd.Series(labels, index=y_z.columns, name="cluster")


def _signs(x: pd.DataFrame, y: pd.DataFrame, labels: pd.Series, tau: float) -> pd.DataFrame:
    signs = pd.DataFrame(0.0, index=y.columns, columns=x.columns)
    if x.shape[1]:
        for label in labels.unique():
            genes = labels.index[labels == label]
            signs.loc[genes] = derive_sign_constraints(
                x,
                y.loc[:, genes],
                auto_sign_threshold_t=tau,
                variance_estimator="date",
                ewma_span=None,
            )
    return signs


def fit_preprocessing(
    x: pd.DataFrame, y: pd.DataFrame, marker_pos: pd.DataFrame, protocol: Protocol
) -> TrainingState:
    """Fit using only the rows supplied; callers pass a strict training subset."""
    if len(x) < 4 or not x.index.equals(y.index):
        raise ValueError("Aligned training samples with at least four rows are required")
    if not np.isfinite(x.to_numpy()).all() or not np.isfinite(y.to_numpy()).all():
        raise ValueError("This eQTL protocol requires complete finite observations")
    variable = x.std(ddof=0) > 1e-12
    x_variable = x.loc[:, variable]
    positions = marker_pos.loc[x_variable.columns]
    ld_idx = ld_block_markers(x_variable, positions, protocol.ld_correlation)
    x_ld = x_variable.iloc[:, ld_idx]
    selected = screen_markers(x_ld, y, protocol.screen_p, protocol.screen_min_genes)
    chosen = x_ld.iloc[:, selected]
    x_mean, x_sd = chosen.mean(), chosen.std(ddof=0).replace(0, 1.0)
    y_mean, y_sd = y.mean(), y.std(ddof=0).replace(0, 1.0)
    x_z, y_z = (chosen - x_mean) / x_sd, (y - y_mean) / y_sd
    clusters = _clusters(y_z, protocol.cluster_cut)
    singleton = pd.Series(np.arange(y.shape[1]), index=y.columns)
    return TrainingState(
        x_z,
        y_z,
        x_mean,
        x_sd,
        y_mean,
        y_sd,
        clusters,
        _signs(x_z, y_z, singleton, protocol.gate_threshold),
        _signs(x_z, y_z, clusters, protocol.gate_threshold),
        x_ld.columns.tolist(),
    )


def constraint_signs(state: TrainingState, method: str) -> pd.DataFrame | None:
    """A failed gate means zero, except in the explicitly unrestricted variant."""
    if method in ("lasso", "hcgl_unconstrained", "fcgl_unconstrained"):
        return None
    if method in ("singleton_exclude", "hcgl_singleton_exclude"):
        return state.singleton_signs.copy()
    signs = state.pooled_signs.copy()
    return signs.replace(0.0, np.nan) if method == "hcgl_unrestricted" else signs


def lambda_max(state: TrainingState, method: str, protocol: Protocol) -> float:
    """Exact zero-solution KKT threshold (bisection for sparse-group norms).

    At zero, the negative loss gradient is H=2 Y'X/n. Project H onto
    the signed tangent cone. Each disjoint group must satisfy
    ||soft_threshold(H, lambda*alpha)||_2 <= lambda*(1-alpha)*weight.
    LASSO is the alpha=1 special case. All targets and adaptive weights are zero/one.
    """
    if not state.x.shape[1]:
        return 0.0
    h = 2.0 * state.y.to_numpy().T @ state.x.to_numpy() / len(state.x)
    signs = constraint_signs(state, method)
    magnitude = np.abs(h)
    if signs is not None:
        s = signs.to_numpy()
        magnitude = np.where(s == 0, 0.0, magnitude)
        magnitude = np.where(s == 1, np.maximum(h, 0), magnitude)
        magnitude = np.where(s == -1, np.maximum(-h, 0), magnitude)
    if method in ("lasso", "singleton_exclude") or state.y.shape[1] == 1:
        return float(magnitude.max(initial=0.0))
    labels = state.clusters.to_numpy()
    unique = np.unique(labels)
    blocks = []
    for label in unique:
        rows = magnitude[labels == label]
        weight = np.sqrt(len(rows) / len(unique))
        groups = rows.T if method.startswith("fcgl_") else rows
        blocks.extend((group, weight) for group in groups)
    alpha = protocol.l1_weight
    upper = float(magnitude.max(initial=0.0) / alpha)
    lower = 0.0
    for _ in range(55):
        middle = (lower + upper) / 2
        valid = all(
            np.linalg.norm(np.maximum(group - alpha * middle, 0)) <= middle * (1 - alpha) * weight
            for group, weight in blocks
        )
        if valid:
            upper = middle
        else:
            lower = middle
    return upper


@safe_cvxpy_sum_shape()
def fit_path(
    state: TrainingState,
    x_valid: pd.DataFrame,
    method: str,
    fractions: tuple[float, ...],
    protocol: Protocol,
) -> list[dict]:
    """Fit the package estimators, with preprocessing and signs held fixed on this split."""
    start = time.perf_counter()
    lam_max = lambda_max(state, method, protocol)
    penalties = [lam_max * ratio for ratio in fractions]
    if not state.x.shape[1] or lam_max == 0:
        betas = [np.zeros((state.y.shape[1], state.x.shape[1])) for _ in fractions]
    else:
        common = dict(
            demean=False,
            warmup_period=0,
            solver=protocol.solver,
            solver_fallbacks=None,
            loss_normalization="sample",
            auto_sign_constraints=False,
            auto_sign_adaptive_weights=False,
            factors_beta_loading_signs=constraint_signs(state, method),
            group_penalty="normalized",
            l1_weight=protocol.l1_weight,
        )
        if method in ("lasso", "singleton_exclude"):
            mode = LassoModelType.LASSO
        elif method.startswith("fcgl_"):
            mode = LassoModelType.FACTOR_CLUSTER_GROUP_LASSO
        else:
            # A training-learned partition supplied to GROUP_LASSO has exactly
            # the HCGL row penalty, and permits public regularization-path reuse.
            mode = LassoModelType.GROUP_LASSO
            common["group_data"] = state.clusters
        model = LassoModel(model_type=mode, **common)
        if method.startswith("fcgl_"):
            models = []
            for penalty in penalties:
                fitted = LassoModel(model_type=mode, reg_lambda=penalty, **common)
                fitted.fit(state.x, state.y, external_clusters=state.clusters)
                models.append(fitted)
        else:
            models = model.fit_reg_lambda_path(state.x, state.y, penalties)
        betas = [np.asarray(m.coef_, dtype=float) for m in models]
    elapsed = (time.perf_counter() - start) / len(fractions)
    x_z = state.transform_x(x_valid).to_numpy()
    output = []
    for fraction, penalty, beta in zip(fractions, penalties, betas):
        if not np.isfinite(beta).all():
            raise RuntimeError(f"Nonfinite coefficients for {method}, lambda={penalty}")
        pred = (x_z @ beta.T) * state.y_sd.to_numpy() + state.y_mean.to_numpy()
        output.append(
            dict(
                fraction=fraction,
                reg_lambda=penalty,
                lambda_max=lam_max,
                predictions=pred,
                beta=beta,
                seconds=elapsed,
            )
        )
    return output


def splits(n: int, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if k > n // 2:
        raise ValueError("Each validation fold needs at least two observations")
    order = np.random.default_rng(seed).permutation(n)
    return [(np.setdiff1d(np.arange(n), test), test) for test in np.array_split(order, k)]


def prediction_metrics(actual: np.ndarray, predicted: np.ndarray, baseline: np.ndarray) -> dict:
    """Pool squared errors before calculating R2; baseline is fold-training mean."""
    sse = np.square(actual - predicted).sum(axis=0)
    baseline_sse = np.square(actual - baseline).sum(axis=0)
    r2 = np.divide(sse, baseline_sse, out=np.full_like(sse, np.nan), where=baseline_sse > 0)
    r2 = 1.0 - r2
    return dict(
        r2_by_gene=r2,
        r2_gene_mean=float(np.nanmean(r2)),
        r2_gene_median=float(np.nanmedian(r2)),
        r2_pooled=float(1.0 - sse.sum() / baseline_sse.sum())
        if baseline_sse.sum()
        else float("nan"),
        rmse=float(np.sqrt(sse.sum() / actual.size)),
    )


def _marginal_signs(x: pd.DataFrame, y: pd.DataFrame) -> np.ndarray:
    """Held-out association signs; no predictions or training preprocessing use these."""
    xz = (x - x.mean()) / x.std(ddof=0).replace(0, np.inf)
    yz = (y - y.mean()) / y.std(ddof=0).replace(0, np.inf)
    r = np.clip(yz.to_numpy().T @ xz.to_numpy() / len(x), -1, 1)
    t = r * np.sqrt((len(x) - 2) / np.maximum(1 - r**2, 1e-12))
    return np.where(2 * stats.t.sf(np.abs(t), len(x) - 2) <= 0.05, np.sign(r), 0)


def _coherence(signs: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
    values = []
    for cluster in np.unique(labels):
        block = signs[labels == cluster]
        positive, negative = (block > 0).sum(axis=0), (block < 0).sum(axis=0)
        count = positive + negative
        keep = count >= 2
        values.extend((np.maximum(positive[keep], negative[keep]) / count[keep]).tolist())
    return (float(np.mean(values)) if values else float("nan"), len(values))


def heldout_diagnostics(
    state: TrainingState, x: pd.DataFrame, y: pd.DataFrame, protocol: Protocol, seed: int
) -> dict:
    signs = _marginal_signs(x.loc[:, state.x.columns], y)
    labels = state.clusters.to_numpy()
    coherence, eligible = _coherence(signs, labels)
    rng = np.random.default_rng(seed)
    random_values = [
        _coherence(signs, rng.permutation(labels))[0] for _ in range(protocol.random_partitions)
    ]
    finite = np.asarray(random_values)[np.isfinite(random_values)]
    train = state.pooled_signs.to_numpy()
    overlap = (train != 0) & (signs != 0)
    return dict(
        heldout_coherence=coherence,
        eligible_cluster_marker_cells=eligible,
        random_coherence_mean=float(finite.mean()) if len(finite) else np.nan,
        random_coherence_sd=float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
        random_valid_partitions=len(finite),
        train_test_sign_agreement=float((train[overlap] == signs[overlap]).mean())
        if overlap.any()
        else np.nan,
        train_test_signed_cells=int(overlap.sum()),
    )


def run_nested_cv(
    x: pd.DataFrame,
    y: pd.DataFrame,
    marker_pos: pd.DataFrame,
    protocol: Protocol,
    *,
    fitter: Callable = fit_path,
    progress: bool = True,
) -> dict:
    """Use paired outer/inner folds for all methods and retain all out-of-fold predictions."""
    if not x.index.equals(y.index) or not x.index.is_unique:
        raise ValueError("x and y require identical unique sample indexes")
    manifest, tuning_rows, fold_rows, prediction_rows, diagnostic_rows = [], [], [], [], []
    coefficients = {}
    fractions = tuple(sorted(protocol.lambda_fractions, reverse=True))
    for repeat in range(protocol.repeats):
        outer = splits(len(x), protocol.outer_folds, protocol.seed + 1009 * repeat)
        for fold, (train, test) in enumerate(outer):
            started = time.perf_counter()
            outer_x, outer_y = x.iloc[train], y.iloc[train]
            fold_record = dict(
                repeat=repeat,
                fold=fold,
                train_ids=x.index[train].tolist(),
                test_ids=x.index[test].tolist(),
                inner=[],
            )
            scores = {method: np.zeros(len(fractions)) for method in protocol.methods}
            count = 0
            for inner_fold, (itr, iva) in enumerate(
                splits(
                    len(train),
                    protocol.inner_folds,
                    protocol.seed + 1009 * repeat + 97 * (fold + 1),
                )
            ):
                state = fit_preprocessing(
                    outer_x.iloc[itr], outer_y.iloc[itr], marker_pos, protocol
                )
                record = state.manifest()
                record.update(inner_fold=inner_fold, validation_ids=outer_x.index[iva].tolist())
                fold_record["inner"].append(record)
                truth = outer_y.iloc[iva].to_numpy()
                for method in protocol.methods:
                    results = fitter(state, outer_x.iloc[iva], method, fractions, protocol)
                    for candidate, result in enumerate(results):
                        squared = np.square((truth - result["predictions"]) / state.y_sd.to_numpy())
                        scores[method][candidate] += squared.sum()
                        tuning_rows.append(
                            dict(
                                repeat=repeat,
                                fold=fold,
                                inner_fold=inner_fold,
                                method=method,
                                fraction=result["fraction"],
                                reg_lambda=result["reg_lambda"],
                                lambda_max=result["lambda_max"],
                                standardized_mse=float(squared.mean()),
                                n_validation=len(iva),
                                n_markers=state.x.shape[1],
                            )
                        )
                count += truth.size
            # All inner transformations above were fit afresh from the raw panel.
            # Refit every transformation once more on the entire outer training set.
            state = fit_preprocessing(outer_x, outer_y, marker_pos, protocol)
            fold_record["outer_preprocessing"] = state.manifest()
            truth = y.iloc[test].to_numpy()
            baseline = np.broadcast_to(state.y_mean.to_numpy(), truth.shape)
            for method in protocol.methods:
                # Descending fractions make an exact tie choose stronger regularization.
                best = int(np.argmin(scores[method]))
                result = fitter(state, x.iloc[test], method, (fractions[best],), protocol)[0]
                predicted = result["predictions"]
                metrics = prediction_metrics(truth, predicted, baseline)
                metrics.pop("r2_by_gene")
                beta = result["beta"]
                signs = constraint_signs(state, method)
                fold_rows.append(
                    dict(
                        repeat=repeat,
                        fold=fold,
                        method=method,
                        n_train=len(train),
                        n_test=len(test),
                        n_markers=state.x.shape[1],
                        n_clusters=state.clusters.nunique(),
                        fraction=fractions[best],
                        reg_lambda=result["reg_lambda"],
                        lambda_max=result["lambda_max"],
                        inner_standardized_mse=float(scores[method][best] / count),
                        active_coefficients=int(
                            (np.abs(beta) > protocol.coefficient_tolerance).sum()
                        ),
                        forced_zero_cells=int((signs == 0).to_numpy().sum())
                        if signs is not None
                        else 0,
                        signed_cells=int(signs.isin([-1, 1]).to_numpy().sum())
                        if signs is not None
                        else 0,
                        fit_seconds=result["seconds"],
                        **metrics,
                    )
                )
                prediction_rows.append(
                    pd.DataFrame(
                        dict(
                            repeat=repeat,
                            fold=fold,
                            method=method,
                            sample=np.repeat(x.index[test].to_numpy(), y.shape[1]),
                            gene=np.tile(y.columns.to_numpy(), len(test)),
                            actual=truth.ravel(),
                            predicted=predicted.ravel(),
                            training_mean=baseline.ravel(),
                        )
                    )
                )
                coefficients[f"r{repeat}_f{fold}_{method}"] = dict(
                    beta=beta, markers=state.x.columns.to_numpy(), genes=y.columns.to_numpy()
                )
            diagnostic_rows.append(
                dict(
                    repeat=repeat,
                    fold=fold,
                    **heldout_diagnostics(
                        state,
                        x.iloc[test],
                        y.iloc[test],
                        protocol,
                        protocol.seed + 7919 * repeat + 613 * fold,
                    ),
                )
            )
            manifest.append(fold_record)
            if progress:
                print(
                    f"eQTL repeat {repeat + 1}/{protocol.repeats}, "
                    f"fold {fold + 1}/{protocol.outer_folds}: "
                    f"{state.x.shape[1]} markers, {state.clusters.nunique()} clusters, "
                    f"{time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    predictions = pd.concat(prediction_rows, ignore_index=True)
    repeats, genes = summarize_predictions(predictions)
    summary = (
        repeats.groupby("method")
        .agg(
            repeats=("repeat", "size"),
            r2_median_mean=("r2_gene_median", "mean"),
            r2_median_split_sd=("r2_gene_median", "std"),
            r2_mean=("r2_gene_mean", "mean"),
            r2_mean_split_sd=("r2_gene_mean", "std"),
            r2_pooled=("r2_pooled", "mean"),
            r2_pooled_split_sd=("r2_pooled", "std"),
            rmse=("rmse", "mean"),
            rmse_split_sd=("rmse", "std"),
        )
        .reset_index()
    )
    return dict(
        predictions=predictions,
        folds=pd.DataFrame(fold_rows),
        tuning=pd.DataFrame(tuning_rows),
        repeats=repeats,
        genes=genes,
        summary=summary,
        heldout=pd.DataFrame(diagnostic_rows),
        manifest=manifest,
        coefficients=coefficients,
    )


def summarize_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    repeat_rows, gene_rows = [], []
    for (repeat, method), frame in predictions.groupby(["repeat", "method"], sort=False):
        pivots = {
            column: frame.pivot(index="sample", columns="gene", values=column)
            for column in ("actual", "predicted", "training_mean")
        }
        metrics = prediction_metrics(*(pivots[c].to_numpy() for c in pivots))
        r2 = metrics.pop("r2_by_gene")
        repeat_rows.append(dict(repeat=repeat, method=method, **metrics))
        gene_rows.extend(
            dict(repeat=repeat, method=method, gene=gene, r2=value)
            for gene, value in zip(pivots["actual"].columns, r2)
        )
    return pd.DataFrame(repeat_rows), pd.DataFrame(gene_rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    # A deposited source export legitimately has no Git metadata. Its receipt
    # still hashes every scientific source and input; do not invent a commit.
    if not (REPOSITORY_DIR / ".git").exists():
        return "unavailable (source export; see source hashes)"
    return subprocess.check_output(["git", *args], cwd=REPOSITORY_DIR, text=True).strip()


def save_results(result: dict, destination: Path, protocol: Protocol, elapsed: float) -> dict:
    """Save into a new external directory, with source/input hashes and actual versions."""
    destination = destination.resolve()
    if any(part.lower().startswith("onedrive") for part in destination.parts):
        raise ValueError("Research outputs must be outside OneDrive")
    if destination == REPOSITORY_DIR or REPOSITORY_DIR in destination.parents:
        raise ValueError("Research outputs must be outside the checkout")
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("folds", "tuning", "repeats", "genes", "summary", "heldout"):
        result[name].to_csv(destination / f"{name}.csv", index=False)
    result["predictions"].to_csv(
        destination / "predictions.csv.gz", index=False, compression="gzip"
    )
    (destination / "fold_manifest.json").write_text(
        json.dumps(result["manifest"], indent=2), encoding="utf-8"
    )
    arrays = {
        f"{key}_{field}": value
        for key, fields in result["coefficients"].items()
        for field, value in fields.items()
    }
    np.savez_compressed(destination / "coefficients.npz", **arrays)
    source_paths = [Path(__file__), Path(__file__).with_name("eqtl_pipeline.py")]
    source_paths.extend((REPOSITORY_DIR / "src/factorlasso").rglob("*.py"))
    receipt = dict(
        status="completed",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=elapsed,
        protocol=asdict(protocol),
        python=sys.executable,
        python_version=platform.python_version(),
        source_head=_git("rev-parse", "HEAD"),
        source_status=_git("status", "--short"),
        factorlasso_import_path=factorlasso.__file__,
        factorlasso_import_version=factorlasso.__version__,
        installed_distribution_version=importlib.metadata.version("factorlasso"),
        declared_project_version=next(
            line.strip()
            for line in (REPOSITORY_DIR / "pyproject.toml").read_text().splitlines()
            if line.startswith("version =")
        ),
        dependencies={
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "cvxpy", "scikit-learn", "rdata")
        },
        inputs={
            str(path.relative_to(REPOSITORY_DIR)): _sha256(path)
            for path in (DATA_DIR / "yeast_full.npz", DATA_DIR / "yeast.rda")
        },
        sources={str(path.relative_to(REPOSITORY_DIR)): _sha256(path) for path in source_paths},
        outputs={path.name: _sha256(path) for path in destination.iterdir() if path.is_file()},
        scope=("Exploratory repeated nested validation on one 112-segregant data set; "
               "not independent replication"),
        uncertainty=("Split SD is descriptive across repeats of the same subjects; "
                     "it is not a sampling confidence interval"),
        tuning=("Minimize pooled inner standardized squared error; training-only "
                "method-specific KKT lambda scale; exact ties choose larger fraction"),
        prediction=("For each repeat pool outer-held-out SSE by gene; R2 denominator "
                    "sums squared deviations from each fold's training response mean"),
        clustering=("Ward on Euclidean training-standardized response vectors/sqrt(n); "
                    "cut fraction of final merge height"),
        signs=("Date-score gate; equal weights; failed gate excluded except hcgl_unrestricted; "
               "no adaptive weights or prior targets"),
        interpretation=("Held-out marginal-sign agreement/coherence is descriptive association "
                        "stability, not validation of conditional coefficient signs"),
    )
    (destination / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--outer-folds", type=int, default=3)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--solver", default="MOSEK")
    parser.add_argument("--lambda-fractions", nargs="+", type=float, default=[1, 0.3, 0.1, 0.03])
    parser.add_argument("--methods", nargs="+", choices=AVAILABLE_METHODS, default=list(METHODS))
    parser.add_argument("--gate-threshold", type=float, default=2)
    parser.add_argument("--cluster-cut", type=float, default=0.7)
    parser.add_argument("--l1-weight", type=float, default=0.5)
    parser.add_argument("--random-partitions", type=int, default=100)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output directory already exists; choose a fresh directory")
    protocol = Protocol(
        **{
            key: tuple(value) if key in ("methods", "lambda_fractions") else value
            for key, value in vars(args).items()
            if key != "out"
        }
    )
    started = time.perf_counter()
    y_all, x, positions = load_data()
    genes, _ = select_mapk_genes(y_all)
    result = run_nested_cv(x, y_all.loc[:, genes], positions, protocol)
    receipt = save_results(result, args.out, protocol, time.perf_counter() - started)
    print(result["summary"].to_string(index=False), flush=True)
    print(f"Saved {receipt['status']} run to {args.out}", flush=True)


if __name__ == "__main__":
    main()
