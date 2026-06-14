#!/usr/bin/env python
"""
replicate.py -- single standalone replication script for the JSS article
"factorlasso: Hierarchical Clustering Group LASSO (HCGL) with Cluster-Pooled
Sign Derivation for Multi-Asset Factor Models in Python".

This is the JSS-recommended single entry point. It regenerates every computed
exhibit in the manuscript and writes a captured output log with a session-
information footer (package versions + platform), the Python analogue of the
``code.html`` / ``sessionInfo()`` convention requested for R submissions.

Exhibits and the stage that produces each
-----------------------------------------
  Stage 1  simulations.run                  -> simulations/results/*.parquet
  Stage 2  paper/analysis.py                -> Table 2 (table2_headline_ablation.csv)
                                               Figures 1-4 (fig1..fig4 *.png)
  Stage 3  applications/etf_simulation_study.py
                                            -> tab:competitor
                                               (etf_competitor_study_{raw,summary}.csv)
  Stage 4  applications/plot_competitor_study.py
                                            -> Figures etf_study_selector_contrast,
                                               etf_study_t_robustness
  Stage 5  applications/run_etf_study.py    -> tab:etf-credit (etf_empirical_credit_betas.csv)
                                               Figure etf_credit_beta_vs_lambda

Stages 1-2 are fully self-contained (seeded synthetic data, no external inputs).
Stages 3-5 require the multi-asset input data described under "Input data"
below; when those files are absent the stage is reported as SKIPPED and the run
continues, so the synthetic half always reproduces.

Input data (required only for Stages 3-5)
-----------------------------------------
  applications/data/futures_risk_factors.csv   factor NAV series (Stages 3, 5)
  applications/data/etf_excess_logreturns.csv  ETF excess monthly log returns (Stage 5)

The ETF panel is rebuilt by ``applications/fetch_etf_panel.py`` from the public
ticker list ``applications/etf_universe.csv`` (requires network + yfinance). The
factor NAV series must be supplied by the user; see the manuscript's
reproduction appendix for its construction.

Modes
-----
  --quick (default)  One simulation seed; the ETF stages run in their own
                     reduced-seed mode. Finishes within roughly the
                     one-hour-on-a-regular-PC guideline of the JSS author
                     instructions. Figures and the ETF tables reproduce to the
                     displayed precision; the synthetic tables reproduce in
                     shape and qualitative pattern with wider Monte Carlo error.
  --full             All simulation seeds and the full ETF seed counts;
                     reproduces every number in the manuscript exactly. Takes
                     substantially longer (minutes to tens of minutes).

Usage
-----
    python papers/jss_2026/replicate.py                 # quick
    python papers/jss_2026/replicate.py --full          # exact
    python papers/jss_2026/replicate.py --skip-sim      # reuse existing parquet
    python papers/jss_2026/replicate.py --log out.txt
"""
from __future__ import annotations

import argparse
import io
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode characters such as
# the lambda symbol the stages print. Reconfigure to UTF-8 so the driver never
# crashes echoing captured output; the console may show mojibake but the log
# file is written as clean UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001  (older interpreters / non-reconfigurable streams)
        pass

HERE = Path(__file__).resolve().parent            # papers/jss_2026
SIM = HERE / "simulations"
APP = HERE / "applications"
PAPER = HERE / "paper"
RESULTS_CAL = SIM / "results_calibrated"
REPO_ROOT = HERE.parents[1]                        # repository root

# Input data required by the ETF stages (Stages 3-5).
FACTOR_NAV = APP / "data" / "futures_risk_factors.csv"
ETF_PANEL = APP / "data" / "etf_excess_logreturns.csv"
ETF_RAW = RESULTS_CAL / "etf_competitor_study_raw.csv"


def _run(cmd: list[str], title: str, tee) -> float:
    """Run a subprocess from the repo root, tee-ing combined output."""
    banner = f"\n{'=' * 72}\n# {title}\n# $ {' '.join(cmd)}\n{'=' * 72}"
    print(banner)
    tee.write(banner + "\n")
    t0 = time.perf_counter()
    # Force UTF-8 in the child so its lambda/dash prints do not crash under a
    # cp1252 Windows console, and decode the captured pipe as UTF-8 to match.
    child_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", env=child_env,
    )
    dt = time.perf_counter() - t0
    print(proc.stdout, end="")
    tee.write(proc.stdout)
    tail = f"[{title}] exit={proc.returncode} elapsed={dt:.1f}s"
    print(tail)
    tee.write(tail + "\n")
    if proc.returncode != 0:
        raise SystemExit(f"Stage failed: {title} (exit {proc.returncode})")
    return dt


def _skip(title: str, missing: list[Path], tee) -> None:
    """Record a skipped data-dependent stage with a clear remediation note."""
    lines = [
        f"\n{'=' * 72}",
        f"# {title} -- SKIPPED (required input data not found)",
        f"{'=' * 72}",
        "Missing:",
        *[f"  - {p.relative_to(REPO_ROOT)}" for p in missing],
        "The synthetic stages (1-2) are unaffected. To reproduce this stage,",
        "supply the factor NAV series and rebuild the ETF panel with",
        "`python papers/jss_2026/applications/fetch_etf_panel.py`, then re-run.",
    ]
    msg = "\n".join(lines)
    print(msg)
    tee.write(msg + "\n")


def _session_info() -> str:
    lines = [
        "\n" + "=" * 72,
        "# Session information",
        "=" * 72,
        f"platform : {platform.platform()}",
        f"python   : {sys.version.split()[0]} ({platform.python_implementation()})",
    ]
    import importlib.metadata as _md
    for pkg in [
        "factorlasso", "numpy", "pandas", "scipy", "cvxpy", "sklearn",
        "matplotlib", "joblib", "asgl", "skglm", "adelie", "yfinance",
    ]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", None)
            if ver is None:
                try:
                    ver = _md.version(pkg)        # packages without __version__ (e.g. asgl)
                except Exception:  # noqa: BLE001
                    ver = "unknown"
        except Exception:  # noqa: BLE001
            ver = "not installed"
        lines.append(f"{pkg:12s}: {ver}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Standalone JSS replication for the factorlasso article."
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run all seeds (exact reproduction). Default is --quick.",
    )
    parser.add_argument(
        "--log", type=Path, nargs="?",
        const=HERE / "replication_output.txt",
        default=HERE / "replication_output.txt",
        help="Path for the captured output log "
             "(bare --log uses the default path).",
    )
    parser.add_argument(
        "--skip-sim", action="store_true",
        help="Skip the (slow) synthetic simulation stage; reuse existing parquet.",
    )
    args = parser.parse_args(argv)

    py = sys.executable
    quick = not args.full
    t_start = time.perf_counter()
    buf = io.StringIO()
    ran: list[str] = []
    skipped: list[str] = []

    header = (
        f"factorlasso JSS replication -- mode={'quick' if quick else 'full'}"
        f" -- started {time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(header)
    buf.write(header + "\n")

    # ── Stage 1: synthetic simulation study ──────────────────────────────
    if not args.skip_sim:
        sim_cmd = [
            py, "-m", "papers.jss_2026.simulations.run",
            "--output", str(SIM / "results"),
        ]
        if quick:
            sim_cmd += ["--seeds-limit", "1"]
        _run(sim_cmd, "Stage 1/5: synthetic simulation study", buf)
        ran.append("Stage 1 (parquet results)")
    else:
        note = "\n[Stage 1/5] skipped (--skip-sim); reusing existing parquet."
        print(note)
        buf.write(note + "\n")
        ran.append("Stage 1 (reused existing parquet)")

    # ── Stage 2: synthetic figures + Table 2 ─────────────────────────────
    _run(
        [py, str(PAPER / "analysis.py"),
         "--results", str(SIM / "results"),
         "--output", str(PAPER / "figures")],
        "Stage 2/5: Table 2 and Figures 1-4", buf,
    )
    ran.append("Stage 2 (Table 2, Figures 1-4)")

    # ── Stage 3: ETF calibrated competitor study (tab:competitor) ────────
    if FACTOR_NAV.exists():
        cmd = [py, str(APP / "etf_simulation_study.py")]
        if quick:
            cmd += ["--quick"]
        else:
            cmd += ["--seeds", "15", "--seed-start", "101"]  # paper tab:competitor config
        _run(cmd, "Stage 3/5: ETF calibrated competitor study", buf)
        ran.append("Stage 3 (tab:competitor)")
    else:
        _skip("Stage 3/5: ETF calibrated competitor study", [FACTOR_NAV], buf)
        skipped.append("Stage 3 (tab:competitor)")

    # ── Stage 4: ETF competitor figures ──────────────────────────────────
    # Plots from the raw CSV produced by Stage 3, or from a committed copy.
    if ETF_RAW.exists():
        _run(
            [py, str(APP / "plot_competitor_study.py"),
             "--raw", str(ETF_RAW), "--figdir", str(PAPER / "figures")],
            "Stage 4/5: ETF competitor figures", buf,
        )
        ran.append("Stage 4 (etf_study_selector_contrast, etf_study_t_robustness)")
    else:
        _skip("Stage 4/5: ETF competitor figures", [ETF_RAW], buf)
        skipped.append("Stage 4 (ETF competitor figures)")

    # ── Stage 5: ETF empirical credit application (tab:etf-credit) ───────
    missing5 = [p for p in (FACTOR_NAV, ETF_PANEL) if not p.exists()]
    if not missing5:
        cmd = [py, str(APP / "run_etf_study.py")]
        if quick:
            cmd += ["--quick"]
        _run(cmd, "Stage 5/5: ETF empirical credit application", buf)
        ran.append("Stage 5 (tab:etf-credit, etf_credit_beta_vs_lambda)")
    else:
        _skip("Stage 5/5: ETF empirical credit application", missing5, buf)
        skipped.append("Stage 5 (tab:etf-credit, etf_credit_beta_vs_lambda)")

    # ── Reproducibility summary ──────────────────────────────────────────
    summary = ["\n" + "=" * 72, "# Reproducibility summary", "=" * 72,
               "Reproduced:"]
    summary += [f"  [ok]   {s}" for s in ran]
    if skipped:
        summary.append("Skipped (input data not present):")
        summary += [f"  [skip] {s}" for s in skipped]
    msg = "\n".join(summary)
    print(msg)
    buf.write(msg + "\n")

    info = _session_info()
    print(info)
    buf.write(info + "\n")
    total = time.perf_counter() - t_start
    footer = f"\nReplication complete in {total:.1f}s ({total / 60:.1f} min)."
    print(footer)
    buf.write(footer + "\n")

    args.log.write_text(buf.getvalue(), encoding="utf-8")
    print(f"\nCaptured output written to {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
