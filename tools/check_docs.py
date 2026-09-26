"""Check factorlasso documentation sources without importing the numerical stack.

The source checks (front matter, byline, software links, heading order, portable display
mathematics, local links) are adapted from ``OptimalPortfolios/tools/check_docs.py`` and
``QuantInvestStrats/tools/check_docs.py``:

    Copyright (c) Artur Sepp. Those files are distributed under the MIT License; the
    permission notice of that licence is retained for the adapted portions. This file, as
    part of factorlasso, is distributed under GPL-3.0-or-later.

Local additions are the ``retained`` page status for legacy RST pages, the ``planned`` article
list, the ``case_study`` page form, and three ownership checks:

* every name in ``factorlasso.__all__`` is owned by exactly one methodology article, and an adopted
  article must name each symbol it owns under its "Implementation in factorlasso" heading;
* every ``LassoModel`` configuration parameter is owned by exactly one methodology article, and an
  adopted article names each parameter it owns;
* the paper ledger records one citation title per paper, taken from the LaTeX source, and no
  reader-facing page may carry a retired title.

Default: validate adopted pages and report what is pending. ``--files`` checks a selected batch
regardless of adoption status. ``--all`` is the completion gate and fails while any article is
planned or pending, or any public symbol is undocumented. Rendering, numerical correctness,
bibliography accuracy and external links are separate checks.
"""

import argparse
import ast
import json
import re
import textwrap
from datetime import date
from pathlib import Path
from typing import NamedTuple, Optional, Sequence
from urllib.parse import unquote, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_URL = "https://github.com/ArturSepp/factorlasso"
CITATION_URL = f"{PROJECT_URL}/blob/main/CITATION.cff"
IMPLEMENTATION_HEADING = "Implementation in factorlasso"
ARTICLE_HEADINGS = (
    "Overview",
    "Inputs, notation, and assumptions",
    "Methodology",
    "Worked example",
    IMPLEMENTATION_HEADING,
    "Interpretation and limitations",
    "See also",
    "References",
)
CASE_STUDY_HEADINGS = (
    "Overview",
    "Study design and data",
    "Configuration",
    "Results",
    "What the study does and does not show",
    "Reproduce",
    "See also",
    "References",
)
FORM_SECTIONS = {"methodology": ARTICLE_HEADINGS, "case_study": CASE_STUDY_HEADINGS}
PAGE_STATES = {
    ("methodology", "pending"),
    ("methodology", "adopted"),
    ("case_study", "pending"),
    ("case_study", "adopted"),
    ("utility", "pending"),
    ("utility", "adopted"),
    ("utility", "retained"),
    ("api", "api"),
}
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
# GitHub Markdown applies backslash escapes before it hands math to MathJax, so ``\,`` reaches
# the renderer as a comma, ``\|`` as a single bar and ``\{`` as a bare brace. Verified against the
# GitHub Markdown API on 2026-09-20. Commands spelled with letters survive in every viewer.
ESCAPED_PUNCTUATION = re.compile(r"\\[^A-Za-z0-9\s]")
INLINE_MATH = re.compile(r"(?<![\\$])\$(?!\$)(.+?)(?<![\\$])\$(?!\$)")
# A display-math line that opens with a list, quote or heading marker ends the $$ block on GitHub:
# a continuation line starting with "+ " became a bullet list in the 2026-09-20 viewer check.
BLOCK_MARKER = re.compile(r"^\s{0,3}(?:[-+*>]\s|#{1,6}\s|\d+[.)]\s|=+\s*$|-+\s*$)")
BLOCK_MARKER_MESSAGE = (
    "A display-math line must not start with a list, quote or heading marker (+, -, *, >, #, 1.); "
    "end the previous line with the operator instead."
)
DELIMITER_MESSAGE = (
    "GitHub renders inline math only when the opening $ follows a space or '(' and the closing "
    "$ is not followed by a letter or digit; reword so the delimiters stand clear."
)
ESCAPE_MESSAGE = (
    "GitHub drops the backslash before punctuation inside math; use a letter command "
    "(\\lVert, \\lbrace, \\quad) or omit the spacing."
)
# A Python block in an article is either an excerpt of the article's canonical script, so that the
# code a reader sees is code the test suite runs, or an explicitly marked non-runnable fragment.
FRAGMENT_MARKER = "<!-- fragment -->"
EXCERPT_MESSAGE = (
    "Python block is not a verbatim excerpt of {example}; copy the lines from the script, or put "
    "'<!-- fragment -->' on the line before a fragment that is not meant to run."
)
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
LINK = re.compile(r"(?<!!)\[[^\]\n]+\]\((https://[^\s)]+)\)")
BYLINE = re.compile(
    r"^\*Author: \[[^\]\n]+\]\(https://github\.com/[A-Za-z0-9-]+\)"
    r"(?: / First recorded: \[(?P<date>\d{4}-\d{2}-\d{2})\]"
    r"\(https://github\.com/ArturSepp/[Ff]actor[Ll]asso/commit/[0-9a-f]{40}\))?\*$"
)


class Issue(NamedTuple):
    """One source-level documentation problem.

    Attributes
    ----------
    line : int
        One-based source line, or one for a document-wide problem.
    message : str
        Explanation of the violated convention.
    """

    line: int
    message: str


def prose_lines(text: str) -> tuple[list[tuple[int, str]], list[Issue], str]:
    """Extract prose while respecting YAML front matter, code fences and display mathematics.

    Parameters
    ----------
    text : str
        Markdown source. No file is modified.

    Returns
    -------
    visible : list of (int, str)
        Line numbers and text of prose lines outside fences, math blocks and comments.
    issues : list of Issue
        Delimiter and fence problems found while scanning.
    metadata : str
        Body of the YAML front matter, or an empty string.
    """
    lines = text.splitlines()
    issues: list[Issue] = []
    metadata = ""
    first = 0
    if lines and lines[0] == "---":
        closing = next((i for i in range(1, len(lines)) if lines[i] == "---"), None)
        if closing is None:
            return [], [Issue(1, "Unclosed YAML front matter.")], ""
        metadata = "\n".join(lines[1:closing])
        first = closing + 1
    visible: list[tuple[int, str]] = []
    fence = ""
    fence_line = 0
    math_line = 0
    comment = False
    for index in range(first, len(lines)):
        line = lines[index]
        number = index + 1
        matched = FENCE.match(line)
        if fence:
            if (
                matched
                and matched[1][0] == fence[0]
                and len(matched[1]) >= len(fence)
                and not matched[2].strip()
            ):
                fence = ""
            continue
        if math_line:
            if line.strip() == "$$":
                math_line = 0
                if index + 1 < len(lines) and lines[index + 1].strip():
                    issues.append(Issue(number, "Put a blank line after display mathematics."))
            else:
                if ESCAPED_PUNCTUATION.search(line):
                    issues.append(Issue(number, ESCAPE_MESSAGE))
                if BLOCK_MARKER.match(line):
                    issues.append(Issue(number, BLOCK_MARKER_MESSAGE))
            continue
        # HTML comments must not satisfy a missing byline, section, or citation.
        if comment:
            if "-->" not in line:
                continue
            comment = False
            line = line.split("-->", 1)[1]
        while "<!--" in line:
            before, after = line.split("<!--", 1)
            if "-->" in after:
                line = before + after.split("-->", 1)[1]
            else:
                line = before
                comment = True
        matched = FENCE.match(line)
        if matched:
            fence, fence_line = matched[1], number
            if re.match(r"^(?:\{math\}|math)(?:\s|$)", matched[2].strip()):
                issues.append(Issue(number, "Use standalone $$ display blocks, not math fences."))
            continue
        if line.startswith(("    ", "\t", ">")):
            continue  # indented code and quoted examples are not article structure
        for code in re.finditer(r"(`+).*?\1", line):
            if re.search(r"\{(?:math|eq)\}$", line[: code.start()]):
                issues.append(Issue(number, "Use dollar math and ordinary section links."))
        without_code = re.sub(r"(`+).*?\1", "", line)
        if "$$" in without_code:
            if line.strip() == "$$":
                math_line = number
                if index > 0 and lines[index - 1].strip():
                    issues.append(Issue(number, "Put a blank line before display mathematics."))
            else:
                issues.append(Issue(number, "Put each display $$ delimiter on its own line."))
            continue
        if re.search(r"\\[\[\]()]", without_code):
            issues.append(Issue(number, "Use dollar delimiters for portable mathematics."))
        if any(ESCAPED_PUNCTUATION.search(math) for math in INLINE_MATH.findall(without_code)):
            issues.append(Issue(number, ESCAPE_MESSAGE))
        if not inline_delimiters_stand_clear(without_code):
            issues.append(Issue(number, DELIMITER_MESSAGE))
        if len(re.findall(r"(?<!\\)\$", without_code)) % 2:
            issues.append(Issue(number, "Unclosed inline mathematics; pair dollars on one line."))
        visible.append((number, line))
    if fence:
        issues.append(Issue(fence_line, "Unclosed code fence."))
    if math_line:
        issues.append(Issue(math_line, "Unclosed display mathematics."))
    if comment:
        issues.append(Issue(len(lines), "Unclosed HTML comment."))
    return visible, issues, metadata


def inline_delimiters_stand_clear(line: str) -> bool:
    """Check the GitHub placement rule for the ``$`` delimiters of inline math on one line.

    Established against the GitHub Markdown API on 2026-09-20: ``fixed-$p$``, ``the $p$th``,
    ``"$p$"`` and ``a=$p$`` stay raw text, while ``a $p$-value``, ``($p$)`` and ``$p$,`` render.
    """
    positions = [m.start() for m in re.finditer(r"(?<!\\)\$", line)]
    for order, position in enumerate(positions):
        if order % 2 == 0:
            before = line[position - 1] if position else " "
            if not (before.isspace() or before == "("):
                return False
        else:
            after = line[position + 1] if position + 1 < len(line) else " "
            if after.isalnum():
                return False
    return True


def valid_byline(line: str) -> bool:
    """Check the linked author byline and an optional calendar-valid first-recorded date."""
    match = BYLINE.fullmatch(line)
    if match is None:
        return False
    if match["date"] is not None:
        try:
            date.fromisoformat(match["date"])
        except ValueError:
            return False
    return True


def _headings(visible: list[tuple[int, str]]) -> list[tuple[int, int, str]]:
    """Return ``(line, level, title)`` for every ATX heading in the visible prose."""
    return [
        (number, len(match[1]), match[2])
        for number, line in visible
        if (match := HEADING.match(line))
    ]


def check_document(
    text: str,
    *,
    methodology: bool = False,
    sections: Optional[Sequence[str]] = None,
) -> list[Issue]:
    """Check one Markdown page's metadata, structure, byline, software links and math source.

    Parameters
    ----------
    text : str
        Complete Markdown source.
    methodology : bool, default False
        Whether the eight methodology H2 sections are required, once each and in order.
    sections : sequence of str, optional
        Required H2 sections of another page form, such as a case study; overrides
        ``methodology``.

    Returns
    -------
    list of Issue
        Source issues. An empty list means these source checks passed, not that TeX rendered.
    """
    visible, issues, metadata = prose_lines(text)
    description = re.search(r"(?m)^[ \t]+description:[ \t]*([^\n]*)", metadata)
    if not description or "html_meta:" not in metadata or "myst:" not in metadata:
        issues.append(Issue(1, "Provide a myst.html_meta.description in front matter."))
    elif description[1].strip() in ("", "''", '""'):
        issues.append(Issue(1, "The page description must not be empty."))
    elif description[1].strip() in (">", ">-", "|", "|-"):
        if not metadata[description.end():].strip():
            issues.append(Issue(1, "The page description must not be empty."))
    headings = _headings(visible)
    titles = [heading for heading in headings if heading[1] == 1]
    if len(titles) != 1 or not headings or headings[0][1] != 1:
        issues.append(Issue(1, "Start with exactly one H1 title."))
    previous = 0
    for number, level, _ in headings:
        if level > previous + 1:
            issues.append(Issue(number, "Do not skip heading levels."))
        previous = level
    title_line = titles[0][0] if titles else 0
    opening = [line for number, line in visible if title_line < number <= title_line + 12]
    if not any(valid_byline(line) for line in opening):
        issues.append(
            Issue(title_line or 1, "Use a linked author byline; any date needs a commit link.")
        )
    prose = "\n".join(line for _, line in visible)
    links = {link.rstrip("/").lower() for link in LINK.findall(re.sub(r"(`+).*?\1", "", prose))}
    if PROJECT_URL.lower() not in links:
        issues.append(Issue(1, "Link to the factorlasso project repository in prose."))
    if CITATION_URL.lower() not in links:
        issues.append(Issue(1, "Link to the canonical factorlasso CITATION.cff in prose."))
    required = list(sections) if sections is not None else (
        list(ARTICLE_HEADINGS) if methodology else None
    )
    if required is not None:
        found = [title for _, level, title in headings if level == 2]
        if found != required:
            issues.append(Issue(1, "Use each required H2 once, in the standard order."))
    return issues


def python_blocks(text: str) -> list[tuple[int, list[str], bool]]:
    """Return ``(line, body, is_fragment)`` for each fenced block tagged ``python``."""
    lines = text.splitlines()
    blocks = []
    index = 0
    while index < len(lines):
        opening = FENCE.match(lines[index])
        if not opening:
            index += 1
            continue
        start = index
        body: list[str] = []
        index += 1
        while index < len(lines):
            closing = FENCE.match(lines[index])
            if (
                closing
                and closing[1][0] == opening[1][0]
                and len(closing[1]) >= len(opening[1])
                and not closing[2].strip()
            ):
                break
            body.append(lines[index])
            index += 1
        index += 1
        if opening[2].strip() != "python":
            continue
        previous = next((line.strip() for line in reversed(lines[:start]) if line.strip()), "")
        blocks.append((start + 1, body, previous == FRAGMENT_MARKER))
    return blocks


def check_code_excerpts(text: str, example_source: str, example_name: str) -> list[Issue]:
    """Require every unmarked Python block to be a contiguous, dedented run of the example's lines.

    Parameters
    ----------
    text : str
        Markdown source of the page.
    example_source : str
        Source of the canonical script named by the page's inventory entry.
    example_name : str
        Repository-relative path of that script, for the message.

    Returns
    -------
    list of Issue
        One issue per Python block that is neither an excerpt nor marked as a fragment.
    """
    script = [line.rstrip() for line in example_source.splitlines()]
    issues = []
    for number, body, is_fragment in python_blocks(text):
        if is_fragment:
            continue
        block = textwrap.dedent("\n".join(line.rstrip() for line in body)).splitlines()
        size = len(block)
        found = size > 0 and any(
            textwrap.dedent("\n".join(script[first:first + size])).splitlines() == block
            for first in range(len(script) - size + 1)
        )
        if not found:
            issues.append(Issue(number, EXCERPT_MESSAGE.format(example=example_name)))
    return issues


def check_local_links(text: str, path: Path, root: Path) -> list[Issue]:
    """Check that local inline and reference-style Markdown links resolve inside the repository.

    Fragment targets, external URLs and HTML attributes need Sphinx or a viewer and are skipped.
    """
    visible, _, _ = prose_lines(text)
    issues = []
    for number, line in visible:
        prose = re.sub(r"(`+).*?\1", "", line)
        links = re.findall(r"\[[^\]\n]*\]\((<[^>\n]+>|[^\s)]+)(?:\s+[^)]*)?\)", prose)
        definition = re.match(r"^\s{0,3}\[[^\]^]+\]:\s*(<[^>\n]+>|\S+)", prose)
        if definition:
            links.append(definition[1])
        for link in links:
            try:
                target = urlsplit(link.strip("<>"))
            except ValueError:
                issues.append(Issue(number, f"Invalid link target: {link}"))
                continue
            if target.scheme or target.netloc or not target.path:
                continue
            decoded = unquote(target.path)
            destination = (
                root / decoded.lstrip("/") if decoded.startswith("/") else path.parent / decoded
            ).resolve()
            if not destination.is_relative_to(root.resolve()):
                issues.append(Issue(number, f"Local link leaves the repository: {link}"))
                continue
            candidates = [destination]
            if destination.suffix == ".html":
                candidates.extend(destination.with_suffix(suffix) for suffix in (".md", ".rst"))
            if not any(candidate.exists() for candidate in candidates):
                issues.append(Issue(number, f"Missing local link target: {link}"))
    return issues


def public_symbols(root: Path) -> list[str]:
    """Read ``factorlasso.__all__`` from source with ``ast``; the package is never imported."""
    source = (root / "src" / "factorlasso" / "__init__.py").read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            names = ast.literal_eval(node.value)
            if len(names) != len(set(names)):
                raise ValueError("factorlasso.__all__ contains a duplicate name.")
            return list(names)
    raise ValueError("factorlasso.__all__ was not found.")


def lasso_model_fields(root: Path) -> tuple[list[str], list[str]]:
    """Read the ``LassoModel`` dataclass fields from source with ``ast``; nothing is imported.

    Returns
    -------
    configuration : list of str
        Constructor fields without a trailing underscore, in declaration order.
    fitted : list of str
        Constructor fields with a trailing underscore; they hold fitted state.
    """
    source = (root / "src" / "factorlasso" / "lasso_estimator.py").read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == "LassoModel":
            configuration, fitted = [], []
            for item in node.body:
                if not (isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)):
                    continue
                value = item.value
                excluded = (
                    isinstance(value, ast.Call)
                    and getattr(value.func, "id", getattr(value.func, "attr", "")) == "field"
                    and any(
                        keyword.arg == "init"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is False
                        for keyword in value.keywords
                    )
                )
                if excluded:
                    continue
                name = item.target.id
                (fitted if name.endswith("_") else configuration).append(name)
            return configuration, fitted
    raise ValueError("class LassoModel was not found in lasso_estimator.py.")


def check_parameter_ownership(inventory: dict, root: Path) -> tuple[list[str], dict[str, str]]:
    """Require every ``LassoModel`` configuration parameter to have exactly one owning article.

    Returns
    -------
    errors : list of str
        Ownership defects; like unowned exports they fail every run.
    owners : dict
        Parameter name mapped to its owning article path.
    """
    pages, planned = inventory["pages"], inventory["planned"]
    parameters = inventory.get("parameters")
    if not isinstance(parameters, dict):
        return ["tools/docs_inventory.json:1: Provide a parameters mapping."], {}
    errors = []
    owners: dict[str, str] = {}
    for article, names in parameters.items():
        is_methodology = article in planned or pages.get(article, {}).get("form") == "methodology"
        if not is_methodology:
            errors.append(f"{article}:1: Parameter owners must be planned or methodology articles.")
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            errors.append(f"{article}:1: Owned parameters must be a list of names.")
            continue
        for name in names:
            if name in owners:
                errors.append(f"{article}:1: Parameter {name} is already owned by {owners[name]}.")
            owners[name] = article
    configuration, _ = lasso_model_fields(root)
    for name in configuration:
        if name not in owners:
            errors.append(
                f"tools/docs_inventory.json:1: LassoModel parameter {name} has no owning article."
            )
    for name in sorted(set(owners) - set(configuration)):
        errors.append(f"{owners[name]}:1: Owned parameter {name} is not a LassoModel parameter.")
    return errors, owners


def check_paper_titles(inventory: dict, root: Path, pages: Sequence[str]) -> list[str]:
    """Reject a retired paper title on any reader-facing page or citation file.

    The ledger in ``tools/docs_inventory.json`` records one title per paper, taken from the LaTeX
    source of the manuscript. Retired titles are matched case-insensitively after whitespace is
    collapsed, so a title wrapped across lines is still found.
    """
    papers = inventory.get("papers", {})
    if not isinstance(papers, dict):
        return ["tools/docs_inventory.json:1: 'papers' must map a key to a title record."]
    errors = []
    retired: list[tuple[str, str]] = []
    for key, record in papers.items():
        if not isinstance(record, dict) or not record.get("title") or not record.get("status"):
            errors.append(f"tools/docs_inventory.json:1: Paper {key} needs a title and a status.")
            continue
        retired.extend((key, title) for title in record.get("retired_titles", []))
    targets = set(pages) | {"CITATION.cff"}
    targets.update(
        path.relative_to(root).as_posix() for path in (root / "papers").glob("*/README.md")
    )
    for name in sorted(targets):
        path = root / name
        if not path.is_file():
            continue
        text = " ".join(path.read_text(encoding="utf-8").split()).lower()
        for key, title in retired:
            if " ".join(title.split()).lower() in text:
                errors.append(
                    f"{name}:1: Retired title of paper {key}; use "
                    f"'{papers[key]['title']}' from the LaTeX source."
                )
    return errors


def implementation_section(text: str) -> str:
    """Return the prose under the implementation H2, or an empty string when it is absent."""
    visible, _, _ = prose_lines(text)
    collected: list[str] = []
    inside = False
    for _, line in visible:
        match = HEADING.match(line)
        if match and len(match[1]) <= 2:
            inside = len(match[1]) == 2 and match[2] == IMPLEMENTATION_HEADING
            continue
        if inside:
            collected.append(line)
    return "\n".join(collected)


def load_inventory(root: Path) -> tuple[dict, list[str]]:
    """Read and validate explicit page ownership; adoption is never inferred from content."""
    path = root / "tools" / "docs_inventory.json"
    try:
        inventory = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {}, [f"tools/docs_inventory.json:1: Cannot read documentation inventory: {exc}"]
    if not isinstance(inventory, dict) or inventory.get("schema_version") != 1:
        return {}, ["tools/docs_inventory.json:1: Expected inventory schema_version 1."]
    pages = inventory.get("pages")
    planned = inventory.get("planned")
    symbols = inventory.get("symbols")
    if not isinstance(pages, dict) or not pages:
        return {}, ["tools/docs_inventory.json:1: Provide a nonempty pages mapping."]
    if not isinstance(planned, dict) or not isinstance(symbols, dict):
        return {}, ["tools/docs_inventory.json:1: Provide planned and symbols mappings."]
    errors = []
    for name, entry in {**planned, **pages}.items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or relative.suffix not in {".md", ".rst"}
        ):
            errors.append(f"{name}:1: Inventory paths must be repository-relative Markdown or RST.")
    for name, entry in pages.items():
        state = (entry.get("form"), entry.get("status")) if isinstance(entry, dict) else None
        if state not in PAGE_STATES:
            errors.append(f"{name}:1: Invalid page form or adoption status.")
            continue
        if entry["status"] == "api" and name != "docs/api.rst":
            errors.append(f"{name}:1: Only docs/api.rst has API ownership.")
        if entry["status"] == "adopted" and Path(name).suffix != ".md":
            errors.append(f"{name}:1: Adopted human pages use portable Markdown.")
        if not (root / name).is_file():
            errors.append(f"{name}:1: Missing documentation page.")
        example = entry.get("example")
        if example is not None and (
            not isinstance(example, str)
            or not example.startswith("examples/docs/")
            or not example.endswith(".py")
            or ".." in Path(example).parts
            or not (root / example).is_file()
        ):
            errors.append(f"{name}:1: 'example' must name an existing script under examples/docs/.")
    for name, entry in planned.items():
        if name in pages:
            errors.append(f"{name}:1: A page is either planned or inventoried, not both.")
        if not isinstance(entry, dict) or not entry.get("id") or not entry.get("title"):
            errors.append(f"{name}:1: A planned article needs an id and a title.")
        if not name.startswith("docs/") or Path(name).suffix != ".md":
            errors.append(f"{name}:1: Planned articles are Markdown pages under docs/.")
        if (root / name).exists():
            errors.append(f"{name}:1: The file exists; move it from planned to pages.")
    return inventory, errors


def check_symbol_ownership(inventory: dict, root: Path) -> tuple[list[str], dict[str, str]]:
    """Require every public name to be owned by exactly one methodology article.

    Returns
    -------
    errors : list of str
        Ownership defects. These fail every run, because an export without an owning article is
        the drift this check exists to catch.
    owners : dict
        Public name mapped to its owning article path.
    """
    pages, planned, symbols = inventory["pages"], inventory["planned"], inventory["symbols"]
    errors = []
    owners: dict[str, str] = {}
    for article, names in symbols.items():
        is_methodology = article in planned or pages.get(article, {}).get("form") == "methodology"
        if not is_methodology:
            errors.append(f"{article}:1: Symbol owners must be planned or methodology articles.")
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            errors.append(f"{article}:1: Owned symbols must be a list of names.")
            continue
        for name in names:
            if name in owners:
                errors.append(f"{article}:1: {name} is already owned by {owners[name]}.")
            owners[name] = article
    exported = public_symbols(root)
    for name in sorted(set(exported) - set(owners)):
        errors.append(
            f"tools/docs_inventory.json:1: Public symbol {name} has no owning article."
        )
    for name in sorted(set(owners) - set(exported)):
        errors.append(
            f"{owners[name]}:1: Owned symbol {name} is not in factorlasso.__all__."
        )
    return errors, owners


def discover_pages(root: Path) -> set[str]:
    """Find the README and every human page under docs/, excluding Sphinx build output."""
    found = {"README.md"} if (root / "README.md").is_file() else set()
    for path in (root / "docs").rglob("*"):
        if path.is_file() and path.suffix in {".md", ".rst"}:
            relative = path.relative_to(root / "docs")
            if relative.parts[0] not in {"_build", "_generated"}:
                found.add(path.relative_to(root).as_posix())
    return found


def run(root: Path, files: Optional[Sequence[Path]], require_all: bool) -> tuple[int, list[str]]:
    """Run the checks against one checkout and return the exit code with its report lines.

    Parameters
    ----------
    root : Path
        Repository checkout or source export.
    files : sequence of Path, optional
        Human Markdown pages to check regardless of adoption status.
    require_all : bool
        Whether planned articles, pending pages and undocumented symbols are failures.
    """
    inventory, errors = load_inventory(root)
    if errors:
        return 1, errors
    pages, planned = inventory["pages"], inventory["planned"]
    discovered = discover_pages(root)
    for name in sorted(discovered - pages.keys()):
        errors.append(f"{name}:1: Add this page to the explicit documentation inventory.")
    for name in sorted(pages.keys() - discovered):
        errors.append(f"{name}:1: Inventory entry is outside the reader-facing discovery scope.")
    stems: dict[str, str] = {}
    for name in sorted(discovered):
        if name.startswith("docs/"):
            stem = str(Path(name).with_suffix(""))
            if stem in stems:
                errors.append(f"{name}:1: Duplicate Sphinx source basename with {stems[stem]}.")
            stems[stem] = name
    ownership_errors, owners = check_symbol_ownership(inventory, root)
    errors.extend(ownership_errors)
    parameter_errors, parameter_owners = check_parameter_ownership(inventory, root)
    errors.extend(parameter_errors)
    errors.extend(check_paper_titles(inventory, root, sorted(discovered)))

    adopted = {name for name, entry in pages.items() if entry["status"] == "adopted"}
    pending = {name for name, entry in pages.items() if entry["status"] == "pending"}
    if files:
        selected = set()
        for requested in files:
            path = (root / requested).resolve()
            if not path.is_relative_to(root.resolve()):
                raise SystemExit(f"Expected a repository-relative page: {requested}")
            name = path.relative_to(root.resolve()).as_posix()
            if name not in pages or path.suffix != ".md" or pages[name]["status"] == "retained":
                raise SystemExit(f"Expected an inventoried Markdown article: {requested}")
            selected.add(name)
    else:
        selected = adopted | pending if require_all else adopted

    documented = set()
    for name in sorted(selected):
        path = root / name
        source = path.read_text(encoding="utf-8")
        issues = check_document(source, sections=FORM_SECTIONS.get(pages[name]["form"]))
        issues.extend(check_local_links(source, path, root))
        example = pages[name].get("example")
        if example:
            script = (root / example).read_text(encoding="utf-8")
            issues.extend(check_code_excerpts(source, script, example))
        errors.extend(f"{name}:{issue.line}: {issue.message}" for issue in issues)
        section = implementation_section(source)
        for symbol in inventory["symbols"].get(name, []):
            if re.search(rf"`(?:factorlasso\.|fl\.)?{re.escape(symbol)}(?:\(\))?`", section):
                documented.add(symbol)
            else:
                errors.append(
                    f"{name}:1: Name `{symbol}` under the heading '{IMPLEMENTATION_HEADING}'."
                )
        for parameter in inventory.get("parameters", {}).get(name, []):
            if not re.search(rf"`{re.escape(parameter)}(?:=[^`]*)?`", source):
                errors.append(f"{name}:1: Name the owned LassoModel parameter `{parameter}`.")
    if not files:
        for name in sorted(adopted - selected):
            documented.update(inventory["symbols"].get(name, []))
    undocumented = sorted(set(owners) - documented) if not files else []
    if require_all:
        for name in sorted(pending):
            errors.append(f"{name}:1: Pending page; mark adopted only after verification.")
        for name in sorted(planned):
            errors.append(f"{name}:1: Planned article {planned[name]['id']} is not written.")
        for symbol in undocumented:
            errors.append(f"{owners[symbol]}:1: Public symbol {symbol} is not documented yet.")
    report = list(errors)
    report.append(
        f"{'FAIL' if errors else 'PASS'}: {len(selected)} selected pages; {len(errors)} issues."
    )
    if not files:
        report.append(
            f"SYMBOLS: {len(owners)} public names owned; {len(owners) - len(undocumented)} "
            f"documented by the checked articles; {len(undocumented)} awaiting their article."
        )
        report.append(
            f"PARAMETERS: {len(parameter_owners)} LassoModel configuration parameters owned by "
            f"{len(set(parameter_owners.values()))} articles."
        )
        if planned:
            ordered = sorted(planned, key=lambda name: planned[name]["id"])
            ids = ", ".join(f"{planned[name]['id']} {name}" for name in ordered)
            report.append(f"PLANNED (not written): {len(planned)} articles: {ids}")
        if pending:
            names = ", ".join(sorted(pending))
            report.append(f"PENDING (not adopted): {len(pending)} pages: {names}")
    return (1 if errors else 0), report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Validate selected sources; ``--all`` is the complete-adoption gate.

    Returns
    -------
    int
        Zero for a passing selection and one for failed checks.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--files", nargs="+", type=Path, help="Markdown articles to check now.")
    selection.add_argument("--all", action="store_true", help="Require complete adoption.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Checkout to inspect.")
    args = parser.parse_args(argv)
    code, report = run(args.root.resolve(), args.files, args.all)
    for line in report:
        print(line)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
