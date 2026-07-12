"""Build factorlasso_jss_replication.zip — the JSS submission bundle.

Packages papers/jss_2026/ re-rooted under factorlasso_jss_replication/,
matching the curated layout of the submitted replication materials:

    paper/          LaTeX source, jss.cls/bst, refs.bib, article.pdf, figures
    simulations/    harness code, study.yaml, tests, committed results,
                    results_calibrated (benchmark + empirical + timing CSVs)
    applications/   ETF study scripts, panel data, universe definition
    replicate.py    the standalone six-stage replication script
    replication_output.txt   captured log of the canonical --full run

Deliberately excluded (not part of the paper pipeline): __pycache__,
LaTeX build artefacts, and the bundle itself. The exploratory and
non-pipeline scripts that earlier versions excluded by name have been
removed from the tree.

Usage (from repository root)::

    python papers/jss_2026/make_replication_zip.py

The script refuses nothing but warns loudly if replication_output.txt
looks older than the code it certifies.
"""
from __future__ import annotations

import hashlib
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent          # papers/jss_2026
ROOT_NAME = "factorlasso_jss_replication"
OUT = HERE / f"{ROOT_NAME}.zip"

EXCLUDE_NAMES = {
    OUT.name, Path(__file__).name,
}
EXCLUDE_DIRS = {"__pycache__", ".ipynb_checkpoints"}
EXCLUDE_SUFFIXES = {".pyc", ".aux", ".bbl", ".blg", ".log", ".out",
                    ".toc", ".fls", ".fdb_latexmk", ".synctex.gz"}


def _keep(p: Path) -> bool:
    if p.name in EXCLUDE_NAMES:
        return False
    if any(part in EXCLUDE_DIRS for part in p.parts):
        return False
    if any(str(p).endswith(suf) for suf in EXCLUDE_SUFFIXES):
        return False
    return True


def main() -> int:
    files = sorted(p for p in HERE.rglob("*") if p.is_file() and _keep(p))

    # Staleness guard: the captured log certifies the code it shipped with.
    log = HERE / "replication_output.txt"
    if log.exists():
        log_m = log.stat().st_mtime
        newer = [p for p in files
                 if p.suffix == ".py" and p.stat().st_mtime > log_m]
        if newer:
            print("[warn] replication_output.txt is OLDER than:",
                  ", ".join(p.name for p in newer))
            print("[warn] re-run 'python papers/jss_2026/replicate.py --full' "
                  "before submitting.")
    else:
        print("[warn] replication_output.txt missing — run replicate.py --full.")

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            arc = f"{ROOT_NAME}/{p.relative_to(HERE).as_posix()}"
            z.write(p, arc)

    size_mb = OUT.stat().st_size / 1e6
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()[:16]
    print(f"wrote {OUT}")
    print(f"  files: {len(files)}   size: {size_mb:.1f} MB "
          f"(JSS limit 50 MB)   sha256[:16]: {sha}")
    print(f"  built: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if size_mb > 50:
        print("[error] bundle exceeds the JSS 50 MB upload limit")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
