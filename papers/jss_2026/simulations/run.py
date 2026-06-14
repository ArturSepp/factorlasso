"""
Simulation study orchestration.

Reads a YAML config that enumerates regimes, seeds, estimators, and a
λ grid, then runs the Cartesian product in parallel and writes:

- ``results_long.parquet`` — one row per (regime, seed, estimator, λ)
  with all metrics and per-fit runtime
- ``results_oracle_lambda.parquet`` — for each (regime, seed, estimator)
  the row with minimum ``beta_mse_norm`` over the λ grid (oracle-λ
  selection used in the paper's main tables)
- ``manifest.json`` — config snapshot, fit count, wall-clock, factorlasso
  version, timestamp; the reviewer-facing reproducibility receipt

Usage (preferred — module form, run from repository root)::

    python -m papers.jss_2026.simulations.run \\
        --config papers/jss_2026/simulations/study.yaml \\
        --output papers/jss_2026/simulations/results

Direct invocation also works (e.g. PyCharm's Run button on this file)::

    python papers/jss_2026/simulations/run.py

With no arguments it falls back to the JSS 2026 study config and writes
output into that study's ``results/`` directory.

Use ``--dry-run`` to print the cell count and the wired/unwired
estimator status without executing.
"""
from __future__ import annotations

# ── Path bootstrap for direct invocation
# When this file is executed as a script (``python papers/jss_2026/simulations/run.py``)
# rather than via ``python -m papers.jss_2026.simulations.run``, Python puts
# only ``simulations/`` on ``sys.path``, so the
# ``from papers.jss_2026.simulations.X import Y`` lines below cannot resolve.
# Detect that case and prepend the repo root (four levels up:
# simulations/ → jss_2026/ → papers/ → repo root) so the imports work
# regardless of how the file was launched.
if __name__ == "__main__" and __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _repo_root = _Path(__file__).resolve().parents[3]
    if str(_repo_root) not in _sys.path:
        _sys.path.insert(0, str(_repo_root))

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

import factorlasso
from papers.jss_2026.simulations.dgp import DGPConfig, make_synthetic_panel
from papers.jss_2026.simulations.estimators import ESTIMATORS, is_wired
from papers.jss_2026.simulations.metrics import compute_all

LOG = logging.getLogger("simulations")

# Default config and output: the JSS 2026 study under this package.
# Resolved at module load via ``__file__`` so they work regardless of the
# caller's current working directory.
_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _PKG_DIR / "study.yaml"
_DEFAULT_OUTPUT = _PKG_DIR / "results"


# ── Single-cell worker ────────────────────────────────────────────────


def fit_one_cell(
    regime_id: str,
    regime_params: dict,
    seed: int,
    estimator_name: str,
    reg_lambda: float,
) -> dict:
    """
    Run one (regime, seed, estimator, λ) cell. Returns a dict ready for
    long-form DataFrame insertion. Catches and records solver errors so
    one bad cell does not abort the study.
    """
    try:
        cfg = DGPConfig(seed=seed, **regime_params)
        data = make_synthetic_panel(cfg)
        result = ESTIMATORS[estimator_name](
            X_train=data.X,
            y_train=data.Y,
            reg_lambda=reg_lambda,
            true_clusters=data.clusters_true,
        )

        metrics = compute_all(
            beta_true=data.beta_true.values,
            beta_hat=result.beta_hat.values,
            clusters_true=data.clusters_true.values,
            factor_premia=data.factor_premia.values,
            clusters_hat=(
                result.extra["clusters_hat"].values
                if "clusters_hat" in result.extra else None
            ),
        )

        return {
            "regime_id": regime_id,
            "seed": seed,
            "estimator": estimator_name,
            "reg_lambda": reg_lambda,
            "runtime": result.runtime,
            "solver_used": result.extra.get("solver_used", "n/a"),
            "status": "ok",
            "error": None,
            **metrics,
            **regime_params,
        }

    except NotImplementedError as exc:
        return {
            "regime_id": regime_id, "seed": seed, "estimator": estimator_name,
            "reg_lambda": reg_lambda, "runtime": float("nan"),
            "status": "not_implemented", "error": str(exc),
        }
    except Exception as exc:
        return {
            "regime_id": regime_id, "seed": seed, "estimator": estimator_name,
            "reg_lambda": reg_lambda, "runtime": float("nan"),
            "status": "failed", "error": f"{type(exc).__name__}: {exc}",
        }


# ── Config expansion ──────────────────────────────────────────────────


def expand_config(cfg: dict) -> tuple[list[tuple[str, dict]], list[int], list[float], list[str]]:
    """
    Parse a loaded YAML into (regimes, seeds, lambda_grid, estimators).

    Each regime is (regime_id, params_dict) where params_dict has all
    DGPConfig fields except ``seed``. Defaults from
    ``study.defaults`` are merged with per-regime overrides; the
    per-regime entry wins.
    """
    study = cfg["study"]
    defaults = study.get("defaults", {})
    regimes = []
    for r in study["regimes"]:
        regime_id = r["id"]
        params = {**defaults}
        for k, v in r.items():
            if k != "id":
                params[k] = v
        regimes.append((regime_id, params))

    seeds = list(study["seeds"])
    lambda_grid = [float(x) for x in study["lambda_grid"]]
    estimators = list(study["estimators"])

    unknown = [e for e in estimators if e not in ESTIMATORS]
    if unknown:
        raise KeyError(
            f"Unknown estimators in YAML: {unknown}. "
            f"Available: {sorted(ESTIMATORS.keys())}"
        )

    return regimes, seeds, lambda_grid, estimators


def apply_oracle_lambda(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Pick the λ minimising ``beta_mse_norm`` for each (regime, seed,
    estimator) triple. Returns the subset of df_long carrying the
    chosen rows.

    Failed cells (``status != 'ok'``) are excluded from the min;
    triples where every λ failed produce no oracle row.
    """
    ok = df_long[df_long["status"] == "ok"]
    if ok.empty:
        return ok.copy()
    idx = (
        ok.groupby(["regime_id", "seed", "estimator"])["beta_mse_norm"]
        .idxmin()
        .dropna()
        .astype(int)
    )
    return ok.loc[idx].reset_index(drop=True)


# ── Output: format fallback chain ────────────────────────────────────


def _has_parquet_engine() -> bool:
    """True iff ``pyarrow`` or ``fastparquet`` is importable."""
    for mod in ("pyarrow", "fastparquet"):
        try:
            __import__(mod)
            return True
        except ImportError:
            continue
    return False


def _save_results_table(df: pd.DataFrame, base_path: Path, stem: str) -> Path:
    """
    Save ``df`` to ``base_path/stem.<ext>`` with format fallback:

        parquet  →  csv  →  pickle

    Returns the path actually written. Auto-fallback ensures that heavy
    computation runs (e.g. the full JSS 2026 study with 2 850 fits) are
    never lost to a missing optional dependency. The fallback also
    catches per-format write errors (disk full, permissions, etc.) and
    proceeds to the next format rather than re-raising.

    A WARNING is logged whenever a format is skipped.
    """
    formats = [
        ("parquet", "parquet", lambda p: df.to_parquet(p, index=False)),
        ("csv",     "csv",     lambda p: df.to_csv(p, index=False)),
        ("pickle",  "pkl",     lambda p: df.to_pickle(p)),
    ]
    last_exc: Optional[Exception] = None
    for fmt, ext, writer in formats:
        path = base_path / f"{stem}.{ext}"
        try:
            writer(path)
            return path
        except ImportError as exc:
            LOG.warning(
                "Writer %s unavailable (%s); trying next format. "
                "Install missing deps with: pip install -e \".[simulations]\"",
                fmt, exc,
            )
            last_exc = exc
        except Exception as exc:
            LOG.warning(
                "Writing %s failed (%s: %s); trying next format.",
                fmt, type(exc).__name__, exc,
            )
            last_exc = exc
    # All formats failed — this is genuinely catastrophic
    raise RuntimeError(
        f"All output formats failed for {stem}. Last error: {last_exc}"
    )


# ── Main ─────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m papers.jss_2026.simulations.run",
        description="Run the factorlasso simulation study described by a YAML config.",
    )
    parser.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG,
        help=(
            "Study YAML to run. "
            "Default: papers/jss_2026/simulations/study.yaml"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=_DEFAULT_OUTPUT,
        help=(
            "Output directory for results_*.parquet + manifest.json. "
            "Default: papers/jss_2026/simulations/results"
        ),
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Joblib parallelism (-1 = all cores, default)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print cell count and estimator wiring status, then exit",
    )
    parser.add_argument(
        "--seeds-limit", type=int, default=None,
        help="Use only the first N seeds (for fast smoke testing)",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Logging verbosity (-v = INFO, -vv = DEBUG)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=[logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    regimes, seeds, lambda_grid, estimators = expand_config(cfg)
    if args.seeds_limit is not None:
        seeds = seeds[: args.seeds_limit]

    total_cells = len(regimes) * len(seeds) * len(estimators) * len(lambda_grid)

    print(f"Study:          {cfg['study'].get('name', '<unnamed>')}")
    print(f"Regimes:        {len(regimes)}")
    print(f"Seeds:          {len(seeds)}  {seeds}")
    print(f"λ grid:         {len(lambda_grid)}  {lambda_grid}")
    print(f"Estimators:     {len(estimators)}")
    for e in estimators:
        status = "wired" if is_wired(e) else "STUB (NotImplementedError)"
        print(f"                  {e:<40s}  {status}")
    print(f"Total cells:    {total_cells}")
    print(f"factorlasso:    v{factorlasso.__version__}")
    if not _has_parquet_engine():
        print()
        print("NOTE: pyarrow/fastparquet not installed in this Python")
        print("      environment. Results will be written as CSV instead")
        print("      of parquet (slightly larger, otherwise equivalent).")
        print("      For the optimal output format:")
        print("        pip install -e \".[simulations]\"")
        print("      or just:")
        print("        pip install pyarrow")
    print()

    if args.dry_run:
        return 0

    args.output.mkdir(parents=True, exist_ok=True)

    jobs = [
        delayed(fit_one_cell)(rid, params, seed, est, lam)
        for rid, params in regimes
        for seed in seeds
        for est in estimators
        for lam in lambda_grid
    ]

    print(f"Running {len(jobs)} fits on n_jobs={args.n_jobs}...")
    t0 = time.perf_counter()
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(jobs)
    wall_clock = time.perf_counter() - t0

    df = pd.DataFrame(results)
    long_path = _save_results_table(df, args.output, "results_long")

    df_oracle = apply_oracle_lambda(df)
    oracle_path = _save_results_table(df_oracle, args.output, "results_oracle_lambda")

    # Reproducibility manifest
    manifest: dict[str, Any] = {
        "study_name": cfg["study"].get("name"),
        "config_path": str(args.config.resolve()),
        "config": cfg,
        "n_cells_requested": len(jobs),
        "n_cells_ok": int((df["status"] == "ok").sum()),
        "n_cells_failed": int((df["status"] == "failed").sum()),
        "n_cells_not_implemented": int((df["status"] == "not_implemented").sum()),
        "wall_clock_seconds": wall_clock,
        "factorlasso_version": factorlasso.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": platform.machine(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    with open(args.output / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print()
    print(f"Done. Wall clock: {wall_clock:.1f}s")
    print(f"  ok:               {manifest['n_cells_ok']}")
    print(f"  failed:           {manifest['n_cells_failed']}")
    print(f"  not_implemented:  {manifest['n_cells_not_implemented']}")
    print(f"Output: {args.output.resolve()}")
    print(f"  long table:    {long_path.name}")
    print(f"  oracle-λ table: {oracle_path.name}")
    print("  manifest:      manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
