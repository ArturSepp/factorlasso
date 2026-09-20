"""Read a generated documentation-analytics bundle back and check it against its manifest.

    python -m tools.docs_analytics.validate --run-root <bundle-directory>

The validator rejects missing or extra files, changed bytes, blank or undersized PNG files,
malformed or mis-shaped CSV tables, failed numerical checks, images that are not registered, and
a manifest whose recorded source no longer matches the checkout. A pass means the bundle is
internally consistent with the code that produced it. It is not a visual review and it does not
recompute the analysis.
"""

import argparse
import csv
import hashlib
import json
import struct
from pathlib import Path
from typing import Optional, Sequence

from tools.docs_analytics.registry import ROOT, load_registry, source_file

MANIFEST_NAME = "analytics_manifest.json"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MINIMUM_PIXELS = (600, 300)


def _sha256(path: Path) -> str:
    """Hash one file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def png_size(path: Path) -> tuple[int, int]:
    """Read width and height from the IHDR chunk; raise when the file is not a PNG."""
    header = path.read_bytes()[:24]
    if len(header) < 24 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise ValueError(f"Not a PNG file: {path.name}")
    return struct.unpack(">II", header[16:24])


def validate_bundle(run_root: Path, root: Path = ROOT) -> dict:
    """Validate one bundle directory and return its parsed manifest.

    Raises
    ------
    ValueError
        On the first inconsistency between the files, the manifest, the registry and the source.
    """
    run_root = run_root.resolve()
    manifest_path = run_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"Bundle has no {MANIFEST_NAME}: {run_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("generated_at_utc"):
        raise ValueError("Unsupported manifest schema or missing generation time")
    registry = load_registry(root, require_previews=False)
    registered = {
        Path(asset["path"]).name: asset["producer"]
        for asset in registry["assets"]
        if asset["kind"] == "synthetic"
    }
    expected_files = {MANIFEST_NAME}
    seen_images = {}
    for name, record in manifest["producers"].items():
        if name not in registry["producers"]:
            raise ValueError(f"Manifest names an unregistered producer: {name}")
        if record["configuration"] != registry["producers"][name]["configuration"]:
            raise ValueError(f"{name}: applied configuration differs from the registry")
        failed = [check for check, passed in record["checks"].items() if passed is not True]
        if not record["checks"] or failed:
            raise ValueError(f"{name}: numerical checks failed or absent: {failed}")
        for basename, image in record["images"].items():
            path = run_root / "images" / basename
            expected_files.add(f"images/{basename}")
            if registered.get(basename) != name:
                raise ValueError(f"{name}: image is not registered to this producer: {basename}")
            if not path.is_file() or _sha256(path) != image["sha256"]:
                raise ValueError(f"Image missing or changed since generation: {basename}")
            width, height = png_size(path)
            if width < MINIMUM_PIXELS[0] or height < MINIMUM_PIXELS[1]:
                raise ValueError(f"Image too small to read: {basename} is {width}x{height}")
            seen_images[basename] = name
        for table, entry in record["tables"].items():
            path = run_root / "tables" / name / f"{table}.csv"
            expected_files.add(f"tables/{name}/{table}.csv")
            if not path.is_file() or _sha256(path) != entry["sha256"]:
                raise ValueError(f"Table missing or changed since generation: {name}/{table}")
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            widths = {len(row) for row in rows}
            if len(rows) != entry["shape"][0] + 1 or len(widths) != 1:
                raise ValueError(f"Table is malformed or mis-shaped: {name}/{table}")
    if set(seen_images) != set(registered):
        raise ValueError(
            f"Bundle images {sorted(seen_images)} != registered previews {sorted(registered)}"
        )
    present = {
        path.relative_to(run_root).as_posix() for path in run_root.rglob("*") if path.is_file()
    }
    if present != expected_files:
        raise ValueError(
            f"Unexpected files {sorted(present - expected_files)}; "
            f"missing files {sorted(expected_files - present)}"
        )
    for name, recorded in manifest["source"]["files"].items():
        if _sha256(source_file(root, name)) != recorded:
            raise ValueError(f"Source changed since generation: {name}")
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Validate the bundle named by ``--run-root``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-root", type=Path, required=True, help="Generated bundle directory.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Checkout or source export.")
    args = parser.parse_args(argv)
    try:
        manifest = validate_bundle(args.run_root, args.root.resolve())
    except ValueError as exc:
        print(f"FAIL: {exc}")
        return 1
    images = sum(len(record["images"]) for record in manifest["producers"].values())
    tables = sum(len(record["tables"]) for record in manifest["producers"].values())
    print(f"PASS: {images} images and {tables} tables match the manifest generated "
          f"{manifest['generated_at_utc']}. Visual review is a separate step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
