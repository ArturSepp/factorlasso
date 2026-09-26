"""Regression tests for the documentation checker and the exhibit registry.

Each check in ``tools/`` is shown to fail on the defect it exists to catch. A check that cannot
fail reads as a guarantee it does not give.
"""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPOSITORY_ROOT / "tools"

pytestmark = pytest.mark.skipif(
    not (TOOLS_ROOT / "check_docs.py").is_file(),
    reason="documentation tooling is not shipped with this test layout",
)

ARTICLE = """---
myst:
  html_meta:
    description: >-
      A factual description of the topic.
---

# Topic

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

## Overview

Let $x_i$ be observation $i$.

$$
\\bar{x} = \\frac{1}{n}\\sum_i x_i.
$$

## Inputs, notation, and assumptions

## Methodology

## Worked example

## Implementation in factorlasso

Call `diagnose_residuals` on the residual panel.

## Interpretation and limitations

## See also

## References
"""


@pytest.fixture(scope="module")
def check_docs():
    """Load ``tools/check_docs.py`` by path so the test works under any import mode."""
    spec = importlib.util.spec_from_file_location("check_docs", TOOLS_ROOT / "check_docs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def analytics(monkeypatch):
    """Import the ``tools.docs_analytics`` namespace from the repository root."""
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT))
    for name in [key for key in sys.modules if key == "tools" or key.startswith("tools.")]:
        monkeypatch.delitem(sys.modules, name)
    import tools.docs_analytics.registry as registry
    import tools.docs_analytics.run as run

    return registry, run


def _messages(issues):
    return [issue.message for issue in issues]


# --- source checker -------------------------------------------------------------------------


def test_template_article_passes(check_docs):
    assert check_docs.check_document(ARTICLE, methodology=True) == []


def test_missing_byline_is_reported(check_docs):
    broken = ARTICLE.replace("*Author: [Artur Sepp](https://github.com/ArturSepp)*\n", "")
    assert any("byline" in message for message in _messages(
        check_docs.check_document(broken, methodology=True)))


def test_first_recorded_date_needs_a_full_commit_link(check_docs):
    dated = "*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-09-20]"
    assert check_docs.valid_byline(
        dated + "(https://github.com/ArturSepp/factorlasso/commit/" + "a" * 40 + ")*")
    assert not check_docs.valid_byline(
        dated + "(https://github.com/ArturSepp/factorlasso/commit/abc1234)*")
    assert not check_docs.valid_byline(
        dated.replace("2026-09-20", "2026-02-30")
        + "(https://github.com/ArturSepp/factorlasso/commit/" + "a" * 40 + ")*")


def test_section_order_is_enforced(check_docs):
    swapped = ARTICLE.replace(
        "## Methodology\n\n## Worked example", "## Worked example\n\n## Methodology")
    assert any("standard order" in message for message in _messages(
        check_docs.check_document(swapped, methodology=True)))
    assert check_docs.check_document(swapped, methodology=False) == []


def test_software_links_are_required(check_docs):
    broken = ARTICLE.replace("https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff", "https://example.org")
    assert any("CITATION.cff" in message for message in _messages(
        check_docs.check_document(broken, methodology=True)))


def test_display_math_must_stand_alone(check_docs):
    inline = ARTICLE.replace("$$\n\\bar{x}", "$$ \\bar{x}")
    assert any("own line" in message for message in _messages(
        check_docs.check_document(inline, methodology=True)))
    crowded = ARTICLE.replace("observation $i$.\n\n$$", "observation $i$.\n$$")
    assert any("blank line before" in message for message in _messages(
        check_docs.check_document(crowded, methodology=True)))


def test_backslash_punctuation_inside_math_is_reported(check_docs):
    """GitHub turns ``\\,`` into a comma and ``\\|`` into a single bar before MathJax runs."""
    display = ARTICLE.replace("\\bar{x} =", "\\bar{x} \\, =")
    inline = ARTICLE.replace("Let $x_i$ be", "Let $\\|x_i\\|$ be")
    portable = ARTICLE.replace("Let $x_i$ be", "Let $\\lVert x_i \\rVert$ be")
    for broken in (display, inline):
        assert any("GitHub drops the backslash" in message for message in _messages(
            check_docs.check_document(broken, methodology=True)))
    assert check_docs.check_document(portable, methodology=True) == []


def test_display_math_line_must_not_open_with_a_block_marker(check_docs):
    """GitHub reads a continuation line starting with ``+ `` as a bullet and ends the formula."""
    broken = ARTICLE.replace("\\bar{x} = \\frac{1}{n}\\sum_i x_i.", "\\bar{x} = a\n+ b")
    assert any("list, quote or heading marker" in message for message in _messages(
        check_docs.check_document(broken, methodology=True)))
    portable = ARTICLE.replace("\\bar{x} = \\frac{1}{n}\\sum_i x_i.", "\\bar{x} = a +\nb")
    assert check_docs.check_document(portable, methodology=True) == []


def test_inline_math_delimiters_must_stand_clear(check_docs):
    """GitHub leaves ``fixed-$p$`` and ``the $n$th`` as raw text; Sphinx and VS Code do not."""
    for broken in ("the fixed-$x_i$ limit", "the $x_i$th term", 'quoted "$x_i$" text'):
        text = ARTICLE.replace("Let $x_i$ be", broken + " and let $x_i$ be")
        assert any("stand clear" in message for message in _messages(
            check_docs.check_document(text, methodology=True))), broken
    for portable in ("a $x_i$-value", "in ($x_i$) parentheses", "ends with $x_i$, then"):
        text = ARTICLE.replace("Let $x_i$ be", portable + " and let $x_i$ be")
        assert check_docs.check_document(text, methodology=True) == [], portable


def test_front_matter_description_is_required(check_docs):
    broken = ARTICLE.replace("      A factual description of the topic.\n", "")
    assert any("description" in message for message in _messages(
        check_docs.check_document(broken, methodology=True)))


def test_missing_local_link_is_reported(check_docs):
    page = REPOSITORY_ROOT / "docs" / "probe.md"
    issues = check_docs.check_local_links("[gone](no_such_page.md)\n", page, REPOSITORY_ROOT)
    assert any("Missing local link" in message for message in _messages(issues))
    assert check_docs.check_local_links("[home](index.rst)\n", page, REPOSITORY_ROOT) == []


def test_implementation_section_is_isolated(check_docs):
    section = check_docs.implementation_section(ARTICLE)
    assert "`diagnose_residuals`" in section
    assert "observation" not in section


SCRIPT = """import factorlasso as fl


def fit(x, y):
    model = fl.LassoModel(reg_lambda=1e-4)
    return model.fit(x=x, y=y)


def main():
    x, y = make_panel()
    model = fit(x, y)
    print(model.coef_)
"""


def test_python_blocks_must_be_excerpts_of_the_canonical_script(check_docs):
    excerpt = "```python\ndef fit(x, y):\n    model = fl.LassoModel(reg_lambda=1e-4)\n```\n"
    dedented = "```python\nx, y = make_panel()\nmodel = fit(x, y)\n```\n"
    assert check_docs.check_code_excerpts(excerpt + dedented, SCRIPT, "examples/docs/a.py") == []

    drifted = excerpt.replace("1e-4", "1e-5")
    issues = check_docs.check_code_excerpts(drifted, SCRIPT, "examples/docs/a.py")
    assert [issue.line for issue in issues] == [1]
    assert "verbatim excerpt of examples/docs/a.py" in issues[0].message
    # lines that exist in the script but are not contiguous there are not an excerpt either
    spliced = "```python\nimport factorlasso as fl\nx, y = make_panel()\n```\n"
    assert len(check_docs.check_code_excerpts(spliced, SCRIPT, "examples/docs/a.py")) == 1


def test_marked_fragments_and_other_languages_are_not_excerpt_checked(check_docs):
    fragment = "<!-- fragment -->\n```python\nmodel.coef_   # not in the script\n```\n"
    console = "```console\npython examples/docs/a.py\n```\n"
    assert check_docs.check_code_excerpts(fragment + console, SCRIPT, "examples/docs/a.py") == []
    unmarked = fragment.replace("<!-- fragment -->\n", "")
    assert len(check_docs.check_code_excerpts(unmarked, SCRIPT, "examples/docs/a.py")) == 1


def test_inventory_example_must_be_a_script_under_examples_docs(check_docs, tmp_path):
    inventory, _ = check_docs.load_inventory(REPOSITORY_ROOT)
    declared = {name: entry["example"] for name, entry in inventory["pages"].items()
                if "example" in entry}
    assert declared and all((REPOSITORY_ROOT / path).is_file() for path in declared.values())

    (tmp_path / "tools").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "page.md").write_text("# Page\n", encoding="utf-8")
    broken = {"schema_version": 1, "planned": {}, "symbols": {}, "pages": {
        "docs/page.md": {"form": "utility", "status": "pending", "example": "tools/check_docs.py"}}}
    (tmp_path / "tools" / "docs_inventory.json").write_text(json.dumps(broken), encoding="utf-8")
    _, errors = check_docs.load_inventory(tmp_path)
    assert any("'example' must name an existing script" in error for error in errors)


# --- inventory and symbol ownership ---------------------------------------------------------


def test_repository_inventory_passes(check_docs):
    code, report = check_docs.run(REPOSITORY_ROOT, None, False)
    assert code == 0, "\n".join(report)


def test_all_gate_tracks_outstanding_work(check_docs):
    inventory, errors = check_docs.load_inventory(REPOSITORY_ROOT)
    assert errors == []
    outstanding = bool(inventory["planned"]) or any(
        entry["status"] == "pending" for entry in inventory["pages"].values())
    code, _ = check_docs.run(REPOSITORY_ROOT, None, True)
    assert code == (1 if outstanding else 0)


def test_every_public_symbol_has_exactly_one_owner(check_docs):
    inventory, _ = check_docs.load_inventory(REPOSITORY_ROOT)
    errors, owners = check_docs.check_symbol_ownership(inventory, REPOSITORY_ROOT)
    assert errors == []
    assert set(owners) == set(check_docs.public_symbols(REPOSITORY_ROOT))


def test_unowned_public_symbol_fails(check_docs):
    inventory, _ = check_docs.load_inventory(REPOSITORY_ROOT)
    broken = copy.deepcopy(inventory)
    article, names = next(iter(broken["symbols"].items()))
    removed = names.pop()
    errors, _ = check_docs.check_symbol_ownership(broken, REPOSITORY_ROOT)
    assert any(f"Public symbol {removed} has no owning article" in error for error in errors)


def test_doubly_owned_and_unknown_symbols_fail(check_docs):
    inventory, _ = check_docs.load_inventory(REPOSITORY_ROOT)
    broken = copy.deepcopy(inventory)
    articles = list(broken["symbols"])
    broken["symbols"][articles[1]].append(broken["symbols"][articles[0]][0])
    broken["symbols"][articles[0]].append("not_a_public_name")
    errors, _ = check_docs.check_symbol_ownership(broken, REPOSITORY_ROOT)
    assert any("already owned" in error for error in errors)
    assert any("not_a_public_name is not in factorlasso.__all__" in error for error in errors)


def test_adopted_articles_name_their_symbols(check_docs):
    inventory, _ = check_docs.load_inventory(REPOSITORY_ROOT)
    for name, entry in inventory["pages"].items():
        if entry["form"] != "methodology" or entry["status"] != "adopted":
            continue
        section = check_docs.implementation_section(
            (REPOSITORY_ROOT / name).read_text(encoding="utf-8"))
        for symbol in inventory["symbols"].get(name, []):
            assert f"`{symbol}`" in section or f"`{symbol}()`" in section, (name, symbol)


# --- exhibit registry -----------------------------------------------------------------------


def test_registry_covers_every_displayed_image(analytics):
    registry, _ = analytics
    loaded = registry.load_registry(REPOSITORY_ROOT)
    assert {asset["kind"] for asset in loaded["assets"]} <= {"paper", "synthetic"}


def test_unregistered_image_fails_coverage(analytics):
    registry, _ = analytics
    loaded = registry.load_registry(REPOSITORY_ROOT)
    broken = copy.deepcopy(loaded)
    displayed = next(asset for asset in broken["assets"] if asset["documents"])
    broken["assets"].remove(displayed)
    with pytest.raises(ValueError, match="unregistered"):
        registry.check_coverage(broken, REPOSITORY_ROOT)


def test_image_parser_reads_markdown_rst_and_html_but_not_code(analytics):
    registry, _ = analytics
    text = (
        "![alt](images/a.png)\n\n```python\n![no](images/in_code.png)\n```\n\n"
        ".. image:: images/b.png\n\n<img src='images/c.png'>\n\n`![no](images/inline.png)`\n"
    )
    found = sorted(registry.image_references(text))
    assert found == ["images/a.png", "images/b.png", "images/c.png"]


def test_unsafe_registry_paths_are_rejected(analytics):
    registry, _ = analytics
    for value in ("../outside.png", "/absolute.png", "docs\\images\\x.png", "C:/x.png"):
        with pytest.raises(ValueError):
            registry.relative_path(value)


def test_committed_previews_match_the_manifest(analytics, capsys):
    _, run = analytics
    assert run.verify(REPOSITORY_ROOT) == 0, capsys.readouterr().out


def test_generation_refuses_pending_producers_and_the_checkout(analytics, monkeypatch, tmp_path):
    registry, run = analytics
    loaded = registry.load_registry(REPOSITORY_ROOT)
    assert run.generate(REPOSITORY_ROOT, REPOSITORY_ROOT / "docs" / "generated_here") == 1
    pending = copy.deepcopy(loaded)
    for spec in pending["producers"].values():
        spec["status"] = "pending"
    monkeypatch.setattr(run, "load_registry", lambda *args, **kwargs: pending)
    assert run.generate(REPOSITORY_ROOT, tmp_path / "bundle") == 2
    assert not (tmp_path / "bundle").exists()


def test_bundle_generates_validates_and_detects_tampering(analytics, tmp_path):
    pytest.importorskip("matplotlib")
    _, run = analytics
    from tools.docs_analytics.validate import validate_bundle

    bundle = tmp_path / "bundle"
    assert run.generate(REPOSITORY_ROOT, bundle) == 0
    manifest = validate_bundle(bundle, REPOSITORY_ROOT)
    assert manifest["review"]["status"] == "pending"
    assert all(all(record["checks"].values()) for record in manifest["producers"].values())
    assert run.generate(REPOSITORY_ROOT, bundle) == 1          # never overwrite a bundle

    image = next((bundle / "images").glob("*.png"))
    image.write_bytes(image.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="changed since generation"):
        validate_bundle(bundle, REPOSITORY_ROOT)

    record = json.loads((bundle / "analytics_manifest.json").read_text(encoding="utf-8"))
    assert record["source"]["files"] and "factorlasso" in record["environment"]["dependencies"]
