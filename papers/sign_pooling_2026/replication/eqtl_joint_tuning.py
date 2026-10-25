"""Joint (cut, threshold, penalty) tuning of the nested yeast eQTL protocol.

Extends ``eqtl_validation.py`` without changing it. The repeats, outer folds and inner
folds are the same splits (seed 20261024, 3 x 3 x 3), the preprocessing is the same
training-only chain (LD thinning at |r| <= 0.92, marginal screen p <= 0.01 in at least
two genes, training standardisation), and the estimators are the same eight package
fits. The difference is the selection rule. Each method selects its own candidate
(kappa, tau, lambda fraction) by pooled inner validation squared error in
training-standardised response units, where kappa is the Ward cut fraction and tau
the date-score gate threshold, over the grid that applies to it:

  lasso                      lambda only
  singleton_exclude          tau x lambda
  hcgl_unconstrained         kappa x lambda          (kappa enters the row-norm multiplier)
  hcgl_singleton_exclude     kappa x tau x lambda
  hcgl_exclude               kappa x tau x lambda
  hcgl_unrestricted          kappa x tau x lambda    (release of failed gates)
  fcgl_unconstrained         kappa x lambda
  fcgl_exclude               kappa x tau x lambda

Two arms share one solver and one set of splits. The ``reference`` arm fixes
kappa = 0.7, tau = 2 and the archived four fractions {1, 0.3, 0.1, 0.03}, which is the
protocol of ``eqtl_validation.py`` and serves as a cross-solver reproduction of the
archived MOSEK run. The ``joint`` arm searches kappa in {0.35, 0.5, 0.7, 0.9}, tau in
{1, 2, 3} and seven fractions {1, 0.5, 0.3, 0.17, 0.1, 0.05, 0.03}. Ties choose the
earliest candidate in grid order with fractions descending, so an exact tie prefers
stronger regularisation, as in the archived protocol.

Units and conventions. Loss SSE / n in training-standardised units, no adaptive
weights, zero prior targets, l1_weight 0.5 for HCGL/FCGL, date-score gate with equal
weights, Ward on Euclidean training-standardised response vectors / sqrt(n) with the
cut as a fraction of the final merge height. Held-out R2 pools squared errors over the
outer test folds of a repeat per gene with the fold-training mean as baseline.

Execution. Checkpoints one pickle per (repeat, fold) under the output directory, so
the run resumes after an interruption and ``--worker k --workers n`` splits the nine
outer folds across processes. ``--finalize`` merges the checkpoints into the CSV
outputs of ``eqtl_validation.py`` plus ``selections.csv`` and a receipt. Outputs are
new research results; frozen reference files are never read or changed.

Not in this module: the archived fixed-setting producer, the sensitivity neighbours,
the stability diagnostics and the held-out coherence diagnostics, which remain in
``eqtl_validation.py`` and ``eqtl_stability.py``.
"""
# packages
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import platform
import sys
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import factorlasso
# project
if __package__:
    from .eqtl_pipeline import load_data, select_mapk_genes
    from .eqtl_validation import (AVAILABLE_METHODS, Protocol, TrainingState, _clusters,
                                  _signs, constraint_signs, fit_path, fit_preprocessing,
                                  prediction_metrics, splits, summarize_predictions)
    from .paths import DATA_DIR
else:
    from eqtl_pipeline import load_data, select_mapk_genes
    from eqtl_validation import (AVAILABLE_METHODS, Protocol, TrainingState, _clusters,
                                 _signs, constraint_signs, fit_path, fit_preprocessing,
                                 prediction_metrics, splits, summarize_predictions)
    from paths import DATA_DIR


class Arm(str, Enum):
    REFERENCE = "reference"
    JOINT = "joint"


METHODS: Tuple[str, ...] = (
    "lasso",
    "singleton_exclude",
    "hcgl_unconstrained",
    "hcgl_singleton_exclude",
    "hcgl_exclude",
    "hcgl_unrestricted",
    "fcgl_unconstrained",
    "fcgl_exclude",
)
USES_CUT = {"hcgl_unconstrained", "hcgl_singleton_exclude", "hcgl_exclude",
            "hcgl_unrestricted", "fcgl_unconstrained", "fcgl_exclude"}
USES_TAU = {"singleton_exclude", "hcgl_singleton_exclude", "hcgl_exclude",
            "hcgl_unrestricted", "fcgl_exclude"}


@dataclass(frozen=True)
class Grid:
    """Candidate hyperparameters of one arm; the reference arm has single cut and tau."""

    arm: Arm
    cuts: Tuple[float, ...]
    taus: Tuple[float, ...]
    fractions: Tuple[float, ...]        # descending, relative to each method's KKT scale

    def __post_init__(self):
        if any(not 0 < c <= 1 for c in self.cuts):
            raise ValueError(f"cuts must lie in (0, 1], got {self.cuts}")
        if any(t < 0 for t in self.taus):
            raise ValueError(f"taus must be nonnegative, got {self.taus}")
        if list(self.fractions) != sorted(set(self.fractions), reverse=True):
            raise ValueError(f"fractions must be distinct and descending, got {self.fractions}")

    def candidates(self, method: str) -> List[Tuple[Optional[float], Optional[float]]]:
        cuts = list(self.cuts) if method in USES_CUT else [None]
        taus = list(self.taus) if method in USES_TAU else [None]
        return [(c, t) for c in cuts for t in taus]


GRIDS: Dict[Arm, Grid] = {
    Arm.REFERENCE: Grid(Arm.REFERENCE, cuts=(0.7,), taus=(2.0,), fractions=(1.0, 0.3, 0.1, 0.03)),
    Arm.JOINT: Grid(Arm.JOINT, cuts=(0.35, 0.5, 0.7, 0.9), taus=(1.0, 2.0, 3.0),
                    fractions=(1.0, 0.5, 0.3, 0.17, 0.1, 0.05, 0.03)),
}


class VariantCache:
    """Clusters and gate signs for one training set at every (cut, tau), computed once."""

    def __init__(self, base: TrainingState):
        self.base = base
        self._clusters: Dict[Optional[float], pd.Series] = {None: base.clusters}
        self._singleton: Dict[Optional[float], pd.DataFrame] = {None: base.singleton_signs}
        self._pooled: Dict[Tuple[Optional[float], Optional[float]], pd.DataFrame] = {
            (None, None): base.pooled_signs}
        self._singleton_labels = pd.Series(np.arange(base.y.shape[1]), index=base.y.columns)

    def clusters(self, cut: Optional[float]) -> pd.Series:
        if cut not in self._clusters:
            self._clusters[cut] = _clusters(self.base.y, cut)
        return self._clusters[cut]

    def state(self, cut: Optional[float], tau: Optional[float]) -> TrainingState:
        clusters = self.clusters(cut)
        if tau not in self._singleton:
            self._singleton[tau] = _signs(self.base.x, self.base.y, self._singleton_labels, tau)
        if (cut, tau) not in self._pooled:
            self._pooled[(cut, tau)] = _signs(self.base.x, self.base.y, clusters, tau)
        return replace(self.base, clusters=clusters, singleton_signs=self._singleton[tau],
                       pooled_signs=self._pooled[(cut, tau)])


def tune_fold(x: pd.DataFrame,
              y: pd.DataFrame,
              marker_pos: pd.DataFrame,
              protocol: Protocol,
              grid: Grid,
              repeat: int,
              fold: int,
              methods: Tuple[str, ...],
              progress: bool = True,
              partial_path: Optional[Path] = None
              ) -> dict:
    """One outer fold: inner selection over the grid, outer refit and held-out prediction.

    ``partial_path`` holds the inner-fold state (scores, tuning rows, completed inner
    folds) so that an interrupted fold resumes at the next inner fold.
    """
    outer = splits(len(x), protocol.outer_folds, protocol.seed + 1009 * repeat)
    train, test = outer[fold]
    outer_x, outer_y = x.iloc[train], y.iloc[train]
    inner = splits(len(train), protocol.inner_folds, protocol.seed + 1009 * repeat + 97 * (fold + 1))
    fractions = grid.fractions
    scores = {m: {c: np.zeros(len(fractions)) for c in grid.candidates(m)} for m in methods}
    tuning_rows, count, started, completed = [], 0, time.perf_counter(), 0
    if partial_path is not None and partial_path.exists():
        with open(partial_path, "rb") as handle:
            partial = pickle.load(handle)
        scores, tuning_rows, count, completed = (partial["scores"], partial["tuning_rows"],
                                                 partial["count"], partial["completed"])
        if progress:
            print(f"[{grid.arm.value}] r{repeat} f{fold}: resuming after inner fold {completed}",
                  flush=True)
    for inner_fold, (itr, iva) in enumerate(inner):
        if inner_fold < completed:
            continue
        base = fit_preprocessing(outer_x.iloc[itr], outer_y.iloc[itr], marker_pos, protocol)
        cache = VariantCache(base)
        truth = outer_y.iloc[iva].to_numpy()
        inner_started = time.perf_counter()
        for method in methods:
            for cut, tau in grid.candidates(method):
                state = cache.state(cut, tau)
                results = fit_path(state, outer_x.iloc[iva], method, fractions, protocol)
                for k, result in enumerate(results):
                    squared = np.square((truth - result["predictions"]) / state.y_sd.to_numpy())
                    scores[method][(cut, tau)][k] += squared.sum()
                    tuning_rows.append(dict(
                        arm=grid.arm.value, repeat=repeat, fold=fold, inner_fold=inner_fold,
                        method=method, cut=cut, tau=tau, fraction=result["fraction"],
                        reg_lambda=result["reg_lambda"], lambda_max=result["lambda_max"],
                        standardized_mse=float(squared.mean()), n_validation=len(iva),
                        n_markers=state.x.shape[1], n_clusters=int(state.clusters.nunique()),
                        fit_seconds=result["seconds"]))
        count += truth.size
        if partial_path is not None:
            with open(partial_path.with_suffix(".tmp"), "wb") as handle:
                pickle.dump(dict(scores=scores, tuning_rows=tuning_rows, count=count,
                                 completed=inner_fold + 1), handle)
            partial_path.with_suffix(".tmp").rename(partial_path)
        if progress:
            print(f"[{grid.arm.value}] r{repeat} f{fold} inner {inner_fold + 1}/{len(inner)}: "
                  f"{base.x.shape[1]} markers, {time.perf_counter() - inner_started:.0f}s",
                  flush=True)
    base = fit_preprocessing(outer_x, outer_y, marker_pos, protocol)
    cache = VariantCache(base)
    truth = y.iloc[test].to_numpy()
    baseline = np.broadcast_to(base.y_mean.to_numpy(), truth.shape)
    fold_rows, prediction_frames, coefficients = [], [], {}
    for method in methods:
        # Earliest candidate in grid order with descending fractions wins exact ties.
        best_key, best_k, best_score = None, None, np.inf
        for key in grid.candidates(method):
            k = int(np.argmin(scores[method][key]))
            if scores[method][key][k] < best_score:
                best_key, best_k, best_score = key, k, scores[method][key][k]
        cut, tau = best_key
        state = cache.state(cut, tau)
        result = fit_path(state, x.iloc[test], method, (fractions[best_k],), protocol)[0]
        predicted = result["predictions"]
        metrics = prediction_metrics(truth, predicted, baseline)
        metrics.pop("r2_by_gene")
        beta, signs = result["beta"], constraint_signs(state, method)
        fold_rows.append(dict(
            arm=grid.arm.value, repeat=repeat, fold=fold, method=method,
            n_train=len(train), n_test=len(test), n_markers=state.x.shape[1],
            n_clusters=int(state.clusters.nunique()), cut=cut, tau=tau,
            fraction=fractions[best_k], reg_lambda=result["reg_lambda"],
            lambda_max=result["lambda_max"],
            inner_standardized_mse=float(best_score / count),
            active_coefficients=int((np.abs(beta) > protocol.coefficient_tolerance).sum()),
            forced_zero_cells=int((signs == 0).to_numpy().sum()) if signs is not None else 0,
            signed_cells=int(signs.isin([-1, 1]).to_numpy().sum()) if signs is not None else 0,
            fit_seconds=result["seconds"], **metrics))
        prediction_frames.append(pd.DataFrame(dict(
            arm=grid.arm.value, repeat=repeat, fold=fold, method=method,
            sample=np.repeat(x.index[test].to_numpy(), y.shape[1]),
            gene=np.tile(y.columns.to_numpy(), len(test)),
            actual=truth.ravel(), predicted=predicted.ravel(), training_mean=baseline.ravel())))
        coefficients[f"{grid.arm.value}_r{repeat}_f{fold}_{method}"] = dict(
            beta=beta, markers=state.x.columns.to_numpy(), genes=y.columns.to_numpy())
    if progress:
        print(f"[{grid.arm.value}] r{repeat} f{fold} done in {time.perf_counter() - started:.0f}s",
              flush=True)
    return dict(folds=fold_rows, tuning=tuning_rows, predictions=pd.concat(prediction_frames),
                coefficients=coefficients, seconds=time.perf_counter() - started)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize(out: Path, protocol: Protocol, arms: List[Arm], methods: Tuple[str, ...]) -> None:
    """Merge checkpoints into CSV outputs and write the receipt."""
    folds, tuning, predictions, coefficients, seconds = [], [], [], {}, 0.0
    for arm in arms:
        for path in sorted((out / arm.value).glob("fold_r*_f*.pkl")):
            with open(path, "rb") as handle:
                chunk = pickle.load(handle)
            folds.extend(chunk["folds"])
            tuning.extend(chunk["tuning"])
            predictions.append(chunk["predictions"])
            coefficients.update(chunk["coefficients"])
            seconds += chunk["seconds"]
    if not folds:
        raise FileNotFoundError(f"no checkpoints under {out}")
    predictions = pd.concat(predictions, ignore_index=True)
    folds_df, tuning_df = pd.DataFrame(folds), pd.DataFrame(tuning)
    repeats, genes = [], []
    for arm, frame in predictions.groupby("arm", sort=False):
        rep, gen = summarize_predictions(frame.drop(columns="arm"))
        rep.insert(0, "arm", arm)
        gen.insert(0, "arm", arm)
        repeats.append(rep)
        genes.append(gen)
    repeats, genes = pd.concat(repeats, ignore_index=True), pd.concat(genes, ignore_index=True)
    summary = (repeats.groupby(["arm", "method"], sort=False)
               .agg(repeats=("repeat", "size"),
                    r2_median_mean=("r2_gene_median", "mean"),
                    r2_median_split_sd=("r2_gene_median", "std"),
                    r2_mean=("r2_gene_mean", "mean"),
                    r2_mean_split_sd=("r2_gene_mean", "std"),
                    r2_pooled=("r2_pooled", "mean"),
                    r2_pooled_split_sd=("r2_pooled", "std"),
                    rmse=("rmse", "mean"),
                    rmse_split_sd=("rmse", "std"))
               .reset_index())
    selections = (folds_df.groupby(["arm", "method", "cut", "tau", "fraction"], dropna=False)
                  .size().rename("outer_fits").reset_index())
    for name, frame in (("folds", folds_df), ("tuning", tuning_df), ("repeats", repeats),
                        ("genes", genes), ("summary", summary), ("selections", selections)):
        frame.to_csv(out / f"{name}.csv", index=False)
    predictions.to_csv(out / "predictions.csv.gz", index=False, compression="gzip")
    arrays = {f"{key}_{field}": value for key, fields in coefficients.items()
              for field, value in fields.items()}
    np.savez_compressed(out / "coefficients.npz", **arrays)
    here = Path(__file__).resolve()
    receipt = dict(
        status="completed",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        fold_seconds_total=seconds,
        protocol=asdict(protocol),
        grids={arm.value: asdict(GRIDS[arm]) for arm in arms},
        methods=list(methods),
        python=sys.executable, python_version=platform.python_version(),
        platform=platform.platform(),
        factorlasso_import_path=factorlasso.__file__,
        factorlasso_import_version=factorlasso.__version__,
        installed_distribution_version=importlib.metadata.version("factorlasso"),
        dependencies={name: importlib.metadata.version(name)
                      for name in ("numpy", "pandas", "scipy", "cvxpy", "clarabel", "rdata")},
        inputs={path.name: _sha256(path) for path in (DATA_DIR / "yeast_full.npz", DATA_DIR / "yeast.rda")},
        sources={name: _sha256(here.with_name(name)) for name in
                 ("eqtl_joint_tuning.py", "eqtl_validation.py", "eqtl_pipeline.py",
                  "cvxpy_shape_compat.py")},
        outputs={path.name: _sha256(path) for path in out.iterdir() if path.is_file()},
        scope=("Repeated nested validation on one 112-segregant data set with the splits of "
               "eqtl_validation.py; the joint arm tunes cut, threshold and penalty per method"),
        uncertainty="Split SD across the three repeats is descriptive, not a sampling standard error",
        tuning=("Minimize pooled inner standardized squared error over each method's candidate "
                "grid; training-only KKT lambda scale per candidate; ties choose the earliest "
                "grid candidate with the larger fraction"),
    )
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(4).to_string(index=False))
        print(selections.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=[a.value for a in Arm], nargs="+",
                        default=[Arm.REFERENCE.value, Arm.JOINT.value])
    parser.add_argument("--solver", default="CLARABEL")
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--methods", nargs="+", choices=AVAILABLE_METHODS, default=list(METHODS))
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="one outer fold, one inner fold, two methods; writes nothing")
    args = parser.parse_args()
    arms = [Arm(a) for a in args.arm]
    protocol = Protocol(seed=args.seed, solver=args.solver, methods=tuple(args.methods))
    methods = tuple(args.methods)
    if args.finalize:
        finalize(args.out, protocol, arms, methods)
        return
    y_all, x, positions = load_data()
    genes, _ = select_mapk_genes(y_all)
    y = y_all.loc[:, genes]
    if args.smoke:
        smoke = Protocol(seed=args.seed, solver=args.solver, methods=methods, inner_folds=2)
        grid = Grid(Arm.JOINT, cuts=(0.5, 0.7), taus=(2.0,), fractions=(1.0, 0.3))
        chunk = tune_fold(x, y, positions, smoke, grid, repeat=0, fold=0,
                          methods=("hcgl_exclude", "lasso"))
        print(pd.DataFrame(chunk["folds"])[["method", "cut", "tau", "fraction", "r2_gene_median",
                                            "fit_seconds"]].to_string(index=False))
        return
    jobs = [(arm, r, f) for arm in arms for r in range(protocol.repeats)
            for f in range(protocol.outer_folds)]
    for arm, r, f in jobs[args.worker::args.workers]:
        target = args.out / arm.value / f"fold_r{r}_f{f}.pkl"
        if target.exists():
            print(f"skip {target.name} (checkpoint exists)", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f"fold_r{r}_f{f}.partial")
        chunk = tune_fold(x, y, positions, protocol, GRIDS[arm], r, f, methods,
                          partial_path=partial)
        with open(target.with_suffix(".tmp"), "wb") as handle:
            pickle.dump(chunk, handle)
        target.with_suffix(".tmp").rename(target)
        partial.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
