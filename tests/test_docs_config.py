"""Tests for the Sphinx documentation configuration."""

import importlib.metadata
import runpy
from pathlib import Path
from types import SimpleNamespace


def test_readthedocs_canonical_url(monkeypatch):
    """Use the canonical URL supplied by Read the Docs."""
    canonical_url = "https://factorlasso.readthedocs.io/en/latest/"
    monkeypatch.setenv("READTHEDOCS_CANONICAL_URL", canonical_url)
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.0.0")

    config = runpy.run_path(str(Path(__file__).parents[1] / "docs" / "conf.py"))

    assert config["html_baseurl"] == canonical_url


def test_local_docs_canonical_url(monkeypatch):
    """Use the live latest-version URL when building outside Read the Docs."""
    monkeypatch.delenv("READTHEDOCS_CANONICAL_URL", raising=False)
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.0.0")

    config = runpy.run_path(str(Path(__file__).parents[1] / "docs" / "conf.py"))

    assert config["html_baseurl"] == "https://factorlasso.readthedocs.io/en/latest/"


def test_index_page_uses_root_canonical(monkeypatch):
    """Collapse the landing page canonical from index.html to the version root."""
    canonical_url = "https://factorlasso.readthedocs.io/en/latest/"
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.0.0")
    config = runpy.run_path(str(Path(__file__).parents[1] / "docs" / "conf.py"))
    context = {"pageurl": f"{canonical_url}index.html"}

    config["_use_root_canonical"](
        SimpleNamespace(config=SimpleNamespace(html_baseurl=canonical_url)),
        "index",
        "page.html",
        context,
        None,
    )

    assert context["pageurl"] == canonical_url
