"""Read the documentation image registry without importing analytical code.

The image-reference parser and the coverage comparison are adapted from
``OptimalPortfolios/tools/docs_analytics/registry.py``, which adapts
``QuantInvestStrats/tools/docs_analytics/run.py``:

    Copyright (c) Artur Sepp. Those files are distributed under the MIT License; the
    permission notice of that licence is retained for the adapted portions. This file, as
    part of factorlasso, is distributed under GPL-3.0-or-later.

Two kinds of asset are registered. A ``paper`` asset is an exhibit committed under ``papers/``
with its own producer script and frozen inputs; documentation tooling displays it and never
regenerates it. A ``synthetic`` asset is a teaching exhibit written to ``docs/images/`` by a
producer module in this directory. Every image displayed in ``README.md`` or ``docs/`` must be
one of these or an explicitly classified non-analytics image such as a badge.

Only the standard library is imported here, so listing and coverage checks run anywhere.
"""

import json
import re
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = "tools/docs_analytics/registry.json"
PREVIEW_ROOT = PurePosixPath("docs/images")
MANIFEST_PATH = "docs/images/analytics_manifest.json"
PAPER_ROOT = PurePosixPath("papers")
IDENTIFIER = re.compile(r"[a-z][a-z0-9_]*")


def relative_path(value: str) -> PurePosixPath:
    """Require a canonical repository-relative path with portable separators."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"Expected a nonempty relative path: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "\\" in value
        or ":" in value
        or any(ord(char) < 32 for char in value)
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return path


def source_file(root: Path, value: str) -> Path:
    """Resolve an existing file with exact case, rejecting source-tree escapes."""
    path = root
    for part in relative_path(value).parts:
        if not path.is_dir() or part not in {entry.name for entry in path.iterdir()}:
            raise ValueError(f"Missing source file or incorrect case: {value}")
        path = path / part
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Missing file or source-tree escape: {value}")
    return path


def image_references(text: str) -> list[str]:
    """Read Markdown, HTML, MyST and RST image references outside code examples."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(
        r"(?m)^\.\. (?:code-block|code|testcode|testoutput)::[^\n]*\n(?:[ \t]+[^\n]*\n|\n)*",
        "",
        text,
    )
    visible: list[str] = []
    references: list[str] = []
    fence = None
    for line in text.splitlines():
        match = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line)
        if fence:
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence):
                fence = None
            continue
        if match:
            fence = match[1]
            directive = re.match(r"\{(?:image|figure)\}\s+(\S+)", match[2])
            if directive:
                references.append(directive[1])
            continue
        directive = re.match(
            r"^\s*(?::{3,}\{(?:image|figure)\}|\.\. (?:\|[^|]+\| )?(?:image|figure)::)\s+(\S+)",
            line,
        )
        if directive:
            references.append(directive[1])
        visible.append(line)
    prose = re.sub(r"(`+)[^\n]*?\1", "", "\n".join(visible))
    references += re.findall(r"!\[[^\]]*\]\(\s*<?([^\s)>]+)>?(?:\s+[^)]*)?\)", prose)

    class ImageParser(HTMLParser):
        """Collect HTML image sources, including self-closing tags."""

        def handle_starttag(self, tag, attrs):
            """Append each visible ``img`` source."""
            if tag.lower() == "img" and dict(attrs).get("src"):
                references.append(dict(attrs)["src"])

    ImageParser().feed(prose)
    definitions = {
        key.strip().casefold(): url
        for key, url in re.findall(r"^\s*\[([^\]]+)\]:\s*<?([^\s>]+)>?", prose, flags=re.M)
    }
    for match in re.finditer(r"!\[([^\]]+)\](?:\[([^\]]*)\])?(?!\()", prose):
        label = (match[2] or match[1]).strip().casefold()
        if label not in definitions:
            raise ValueError(f"Unresolved image reference: {label}")
        references.append(definitions[label])
    return references


def documents(root: Path) -> list[Path]:
    """Discover the README and the human pages under docs/, excluding Sphinx build output."""
    paths = {source_file(root, "README.md")}
    for path in (root / "docs").rglob("*"):
        relative = path.relative_to(root / "docs")
        if path.is_file() and path.suffix in {".md", ".rst"} and relative.parts[0] != "_build":
            paths.add(path)
    return sorted(paths)


def displayed_images(root: Path) -> set[tuple[str, str]]:
    """Return ``(document, target)`` for every displayed image, targets repository-relative."""
    observed = set()
    for document in documents(root):
        name = document.relative_to(root).as_posix()
        for url in image_references(document.read_text(encoding="utf-8")):
            parsed = urlsplit(url)
            if parsed.scheme or parsed.netloc:
                observed.add((name, url))
                continue
            if parsed.path.startswith(("/", "\\")):
                raise ValueError(f"Absolute image path: {name}: {url}")
            resolved = (document.parent / unquote(parsed.path)).resolve()
            if not resolved.is_relative_to(root.resolve()):
                raise ValueError(f"Image escapes the source tree: {name}: {url}")
            observed.add((name, resolved.relative_to(root.resolve()).as_posix()))
    return observed


def check_coverage(registry: dict, root: Path = ROOT) -> None:
    """Require a classification for every displayed image and a consumer for every record."""
    expected = {
        (document, asset["path"]) for asset in registry["assets"] for document in asset["documents"]
    }
    expected |= {(item["document"], item["url"]) for item in registry["non_analytics"]}
    observed = displayed_images(root)
    if observed != expected:
        raise ValueError(
            f"Image coverage mismatch: unregistered={sorted(observed - expected)}, "
            f"not displayed={sorted(expected - observed)}"
        )


def _unique_object(pairs: list) -> dict:
    """Reject duplicate JSON keys instead of silently keeping the last value."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate registry key: {key}")
        result[key] = value
    return result


def load_registry(root: Path = ROOT, *, require_previews: bool = True) -> dict:
    """Validate the registry and, unless generating, the presence of every registered file.

    Parameters
    ----------
    root : Path
        Repository checkout or local source export.
    require_previews : bool, default True
        Whether synthetic previews must already exist under ``docs/images/``. Generation passes
        False, because a new exhibit is produced before it can be committed.

    Returns
    -------
    dict
        Parsed registry with verified paths and complete displayed-image coverage.

    Raises
    ------
    ValueError
        If the registry, an asset record or the image coverage is invalid.
    """
    registry = json.loads(
        source_file(root, REGISTRY_PATH).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
    )
    if not isinstance(registry, dict) or registry.get("schema_version") != 1:
        raise ValueError("Unsupported analytics registry schema")
    producers = registry.get("producers")
    assets = registry.get("assets")
    badges = registry.get("non_analytics")
    rendering = registry.get("rendering")
    if not isinstance(producers, dict) or not isinstance(assets, list) or not assets:
        raise ValueError("Registry needs a producers mapping and a nonempty assets list")
    if not isinstance(badges, list) or not isinstance(rendering, dict):
        raise ValueError("Registry needs non_analytics and rendering records")
    if not isinstance(rendering.get("dpi"), int) or not rendering.get("font_family"):
        raise ValueError("Rendering needs an integer dpi and a font family")
    for name, spec in producers.items():
        if not IDENTIFIER.fullmatch(name) or not isinstance(spec, dict):
            raise ValueError(f"Invalid producer name: {name}")
        if spec.get("status") not in {"pending", "implemented"}:
            raise ValueError(f"Invalid producer status: {name}")
        if not isinstance(spec.get("configuration"), dict) or not spec["configuration"]:
            raise ValueError(f"Producer needs an explicit configuration: {name}")
        fixture = spec.get("fixture_files")
        if not isinstance(fixture, list) or not fixture:
            raise ValueError(f"Producer needs the fixture files it reads: {name}")
        for path in fixture:
            source_file(root, path)
        if spec["status"] == "implemented":
            source_file(root, f"tools/docs_analytics/{name}.py")
    identities, paths, used = set(), set(), set()
    for asset in assets:
        if not isinstance(asset, dict) or not IDENTIFIER.fullmatch(str(asset.get("id"))):
            raise ValueError("Invalid asset identity")
        path = relative_path(asset.get("path"))
        if asset["id"] in identities or str(path).casefold() in paths:
            raise ValueError(f"Duplicate asset identity or path: {path}")
        identities.add(asset["id"])
        paths.add(str(path).casefold())
        consumers = asset.get("documents")
        planned = asset.get("planned_for", [])
        if not isinstance(consumers, list) or len(consumers) != len(set(consumers)):
            raise ValueError(f"Invalid consumer list: {path}")
        if not consumers and not planned:
            raise ValueError(f"Asset has neither a consumer nor a planned article: {path}")
        for document in consumers:
            source_file(root, document)
        for field in ("caption_class", "alt"):
            if not asset.get(field):
                raise ValueError(f"Asset needs {field}: {path}")
        if asset.get("kind") == "paper":
            if PAPER_ROOT not in path.parents or path.suffix.lower() != ".png":
                raise ValueError(f"Paper exhibits are PNG files under papers/: {path}")
            source_file(root, str(path))
            source_file(root, asset.get("producer_script", ""))
            if not asset.get("replication_command") or not asset.get("study"):
                raise ValueError(f"Paper exhibit needs a replication command and study: {path}")
        elif asset.get("kind") == "synthetic":
            if path.parent != PREVIEW_ROOT or path.suffix != ".png":
                raise ValueError(f"Synthetic previews are docs/images/*.png: {path}")
            if asset.get("producer") not in producers:
                raise ValueError(f"Unknown producer for {path}")
            used.add(asset["producer"])
            if require_previews:
                source_file(root, str(path))
        else:
            raise ValueError(f"Asset kind must be 'paper' or 'synthetic': {path}")
    if used != set(producers):
        raise ValueError(f"Producers without a registered exhibit: {sorted(set(producers) - used)}")
    classified = set()
    for item in badges:
        if not isinstance(item, dict) or not item.get("reason") or not item.get("url"):
            raise ValueError("Non-analytics images need a URL and a reason")
        if urlsplit(item["url"]).scheme != "https":
            raise ValueError("Non-analytics classification requires an HTTPS image URL")
        source_file(root, item["document"])
        key = (item["document"], item["url"])
        if key in classified:
            raise ValueError("Duplicate non-analytics classification")
        classified.add(key)
    check_coverage(registry, root)
    return registry
