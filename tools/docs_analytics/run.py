"""List, generate and verify the documentation exhibits of factorlasso.

Run from the repository root (or a local source export) with the interpreter in ``AGENTS.md``::

    python -m tools.docs_analytics.run --list
    python -m tools.docs_analytics.run --all --output-root <new-directory-outside-the-checkout>
    python -m tools.docs_analytics.run --verify

``--list`` validates the registry and the displayed-image coverage with the standard library
only. ``--all`` runs every implemented producer into a fresh directory, writes the PNG previews,
one CSV per table and ``analytics_manifest.json``, and validates the bundle before finalising it.
``--verify`` compares the previews committed under ``docs/images/`` with the committed manifest.

A successful run is not a visual review. Publication is a manual copy of inspected PNG files and
the manifest into ``docs/images/``; see ``docs/documentation_standard.md``.

Exit codes: 0 success, 1 failed check or producer, 2 nothing can be generated (no implemented
producer, or a registered producer is still pending).
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from tools.docs_analytics.registry import MANIFEST_PATH, ROOT, load_registry, source_file

MANIFEST_NAME = "analytics_manifest.json"
BASE_DEPENDENCIES = ("factorlasso", "numpy", "pandas", "scipy", "cvxpy", "matplotlib")


def sha256_file(path: Path) -> str:
    """Hash one file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity(root: Path, files: Sequence[str]) -> dict:
    """Record the code that actually ran: per-file hashes, package hash, commit and dirty flag."""
    package = hashlib.sha256()
    for path in sorted((root / "src" / "factorlasso").glob("*.py")):
        package.update(path.name.encode())
        package.update(path.read_bytes())
    identity = {
        "files": {name: sha256_file(source_file(root, name)) for name in sorted(set(files))},
        "package_source_sha256": package.hexdigest(),
        "commit": None,
        "dirty": None,
    }
    try:
        identity["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        ).stdout
        identity["dirty"] = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        pass  # a source export has no git metadata; the hashes above still identify the code
    return identity


def environment() -> dict:
    """Record the interpreter, platform and the versions of the imported dependencies."""
    versions = {}
    for name in BASE_DEPENDENCIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": versions,
    }


def _run_producer(name: str, spec: dict, registry: dict, staging: Path) -> dict:
    """Run one producer, write its figures and tables, and return its manifest record."""
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = registry["rendering"]["font_family"]
    module = importlib.import_module(f"tools.docs_analytics.{name}")
    result = module.produce(spec["configuration"])
    if set(result) != {"figures", "tables", "configuration", "checks"}:
        raise ValueError(f"{name}: produce() must return figures, tables, configuration, checks")
    expected = {
        Path(asset["path"]).name
        for asset in registry["assets"]
        if asset["kind"] == "synthetic" and asset["producer"] == name
    }
    if set(result["figures"]) != expected:
        raise ValueError(
            f"{name}: figures {sorted(result['figures'])} != registered {sorted(expected)}"
        )
    failed = [check for check, passed in result["checks"].items() if passed is not True]
    if not result["checks"] or failed:
        raise ValueError(f"{name}: numerical checks failed or absent: {failed}")
    record = {
        "configuration": result["configuration"],
        "checks": result["checks"],
        "images": {},
        "tables": {},
    }
    for basename, figure in result["figures"].items():
        path = staging / "images" / basename
        figure.savefig(path, dpi=registry["rendering"]["dpi"], facecolor=figure.get_facecolor())
        record["images"][basename] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    for table, frame in result["tables"].items():
        if frame.empty:
            raise ValueError(f"{name}: table {table} is empty")
        path = staging / "tables" / name / f"{table}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, float_format="%.17g", lineterminator="\n")
        record["tables"][table] = {"sha256": sha256_file(path), "shape": list(frame.shape)}
    return record


def generate(root: Path, output_root: Path) -> int:
    """Generate the complete bundle into ``output_root``, which must not exist yet."""
    registry = load_registry(root, require_previews=False)
    producers = registry["producers"]
    pending = sorted(name for name, spec in producers.items() if spec["status"] == "pending")
    if pending or not producers:
        print(f"Nothing generated: pending producers {pending}" if pending
              else "Nothing generated: no producer is registered yet.")
        return 2
    output_root = output_root.resolve()
    if output_root.exists():
        print(f"Refusing an existing output directory: {output_root}")
        return 1
    if output_root.is_relative_to(root.resolve()):
        print("Write generated output outside the source checkout.")
        return 1
    staging = output_root.with_name(f"{output_root.name}.staging-{os.getpid()}")
    (staging / "images").mkdir(parents=True)
    try:
        records = {name: _run_producer(name, producers[name], registry, staging)
                   for name in sorted(producers)}
        files = [f"tools/docs_analytics/{name}.py" for name in producers]
        files += [path for spec in producers.values() for path in spec["fixture_files"]]
        files += ["tools/docs_analytics/registry.json", "tools/docs_analytics/run.py"]
        manifest = {
            "schema_version": 1,
            "kind": "factorlasso_documentation_analytics",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_kind": "Synthetic teaching exhibits; not market or paper-replication evidence.",
            "rendering": registry["rendering"],
            "source": source_identity(root, files),
            "environment": environment(),
            "producers": records,
            "review": {"status": "pending", "note": "Set by the person who inspects the images."},
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        from tools.docs_analytics.validate import validate_bundle

        validate_bundle(staging, root)
    except Exception as exc:  # keep the partial output for diagnosis, never finalise it
        (staging / "FAILED.json").write_text(
            json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2) + "\n", encoding="utf-8"
        )
        print(f"FAILED: {type(exc).__name__}: {exc}\nPartial output kept in {staging}")
        return 1
    staging.rename(output_root)
    print(f"Bundle written and validated: {output_root}")
    return 0


def verify(root: Path) -> int:
    """Compare committed synthetic previews with the committed manifest."""
    registry = load_registry(root)
    synthetic = [asset for asset in registry["assets"] if asset["kind"] == "synthetic"]
    preview_dir = root / "docs" / "images"
    present = {path.name for path in preview_dir.glob("*.png")} if preview_dir.is_dir() else set()
    registered = {Path(asset["path"]).name for asset in synthetic}
    problems = [f"unregistered preview: {name}" for name in sorted(present - registered)]
    if synthetic:
        manifest = json.loads(source_file(root, MANIFEST_PATH).read_text(encoding="utf-8"))
        recorded = {
            basename: image["sha256"]
            for record in manifest["producers"].values()
            for basename, image in record["images"].items()
        }
        for name in sorted(registered):
            if name not in recorded:
                problems.append(f"preview missing from the manifest: {name}")
            elif sha256_file(preview_dir / name) != recorded[name]:
                problems.append(f"preview bytes differ from the manifest: {name}")
        problems += [f"manifest records an unregistered image: {name}"
                     for name in sorted(set(recorded) - registered)]
    for problem in problems:
        print(problem)
    print(f"{'FAIL' if problems else 'PASS'}: {len(registered)} synthetic previews verified.")
    return 1 if problems else 0


def list_assets(root: Path) -> int:
    """Print the registered exhibits after validating registry and coverage."""
    registry = load_registry(root)
    for asset in registry["assets"]:
        origin = asset.get("producer") or asset.get("producer_script")
        consumers = ", ".join(asset["documents"]) or f"planned: {', '.join(asset['planned_for'])}"
        print(f"{asset['kind']:<9} {asset['path']}  <- {origin}  [{consumers}]")
    print(f"PASS: {len(registry['assets'])} exhibits and {len(registry['non_analytics'])} "
          f"non-analytics images cover every displayed image.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Dispatch one of ``--list``, ``--all`` or ``--verify``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="Validate registry and coverage.")
    action.add_argument("--all", action="store_true", help="Generate the complete bundle.")
    action.add_argument("--verify", action="store_true", help="Check committed previews.")
    parser.add_argument("--output-root", type=Path, help="New directory outside the checkout.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Checkout or source export.")
    args = parser.parse_args(argv)
    try:
        if args.list:
            return list_assets(args.root.resolve())
        if args.verify:
            return verify(args.root.resolve())
        if args.output_root is None:
            parser.error("--all requires --output-root")
        return generate(args.root.resolve(), args.output_root)
    except ValueError as exc:
        print(f"FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
