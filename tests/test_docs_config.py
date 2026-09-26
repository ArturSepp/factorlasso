"""Tests for the Sphinx documentation configuration."""

import importlib.metadata
import json
import re
import runpy
from pathlib import Path

import pytest

DOCS = Path(__file__).parents[1] / "docs"


def _config(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.0.0")
    return runpy.run_path(str(DOCS / "conf.py"))


def test_readthedocs_canonical_url(monkeypatch):
    """Use the canonical URL supplied by Read the Docs."""
    canonical_url = "https://factorlasso.readthedocs.io/en/latest/"
    monkeypatch.setenv("READTHEDOCS_CANONICAL_URL", canonical_url)

    assert _config(monkeypatch)["html_baseurl"] == canonical_url


def test_local_docs_canonical_url(monkeypatch):
    """Use the live latest-version URL when building outside Read the Docs."""
    monkeypatch.delenv("READTHEDOCS_CANONICAL_URL", raising=False)

    assert _config(monkeypatch)["html_baseurl"] == "https://factorlasso.readthedocs.io/en/latest/"


def test_indexing_extension_and_verification_tag_are_configured(monkeypatch):
    """Canonical URLs and the sitemap come from the extension; every page carries the tag."""
    config = _config(monkeypatch)

    assert "factorlasso_indexing" in config["extensions"]
    assert (DOCS / "_ext" / "factorlasso_indexing.py").is_file()
    template = (DOCS / "_templates" / "base.html").read_text(encoding="utf-8")
    assert "google-site-verification" in template
    assert config["html_context"]["google_site_verification"] == config["GOOGLE_SITE_VERIFICATION"]
    # The tag must not depend on rst_prolog, which Markdown pages such as index.md never see.
    assert "google-site-verification" not in config["rst_prolog"]


def test_markdown_articles_use_portable_dollar_math(monkeypatch):
    """Articles are MyST Markdown with dollar math and Mermaid fences; the API page stays RST."""
    config = _config(monkeypatch)

    assert {"myst_parser", "numpydoc", "sphinx.ext.doctest", "sphinxcontrib.mermaid"} <= set(
        config["extensions"]
    )
    assert config["source_suffix"] == {".rst": "restructuredtext", ".md": "markdown"}
    assert "dollarmath" in config["myst_enable_extensions"]
    assert config["myst_heading_anchors"] >= 2
    assert config["myst_fence_as_directive"] == ["mermaid"]
    assert "_generated" in config["exclude_patterns"]


def test_copyright_names_the_package_owner(monkeypatch):
    """The rendered footer reads 'Copyright 2026, Artur Sepp'; authorship is a separate field."""
    assert _config(monkeypatch)["copyright"] == "2026, Artur Sepp"


def test_generated_api_reference_documents_every_export_once(monkeypatch, tmp_path):
    """Every public name appears once, under its owning article, with the parameter map."""
    pytest.importorskip("cvxpy")
    import factorlasso

    config = _config(monkeypatch)
    text = config["_write_api_reference"](tmp_path).read_text(encoding="utf-8")

    documented = re.findall(r"^\.\. auto(?:class|function):: factorlasso\.(\w+)$", text, re.M)
    assert sorted(documented) == sorted(factorlasso.__all__)
    inventory = json.loads(
        (DOCS.parent / "tools" / "docs_inventory.json").read_text(encoding="utf-8")
    )
    parameters = [name for names in inventory["parameters"].values() for name in names]
    for name in parameters:
        assert f"``{name}=" in text, name
    assert "(planned article)" in text or all(
        (DOCS.parent / page).is_file() for page in inventory["symbols"]
    )


def test_api_page_includes_the_generated_reference():
    """The tracked page is a stable entry point; the generated body is git-ignored."""
    page = (DOCS / "api.rst").read_text(encoding="utf-8")
    assert ".. include:: _generated/api_reference.rst" in page
    assert "_generated/" in (DOCS / ".gitignore").read_text(encoding="utf-8")
