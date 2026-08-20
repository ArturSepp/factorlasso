"""Sphinx configuration for the FactorLasso documentation."""

import os
import sys
from importlib.metadata import version as package_version
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

project = "factorlasso"
author = "Artur Sepp and Mika Kastenholz"
copyright = "2026, Artur Sepp and Mika Kastenholz"
version = package_version("factorlasso")
release = package_version("factorlasso")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "numpydoc",
]

root_doc = "index"
source_suffix = ".rst"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
nitpicky = False
autodoc_typehints = "description"
autodoc_member_order = "bysource"
numpydoc_show_class_members = False
doctest_test_doctest_blocks = ""

rst_prolog = """
.. |t| replace:: t
.. |beta| replace:: beta
"""

html_theme = "sphinx_rtd_theme"
html_title = f"factorlasso {release}"
html_baseurl = os.environ.get(
    "READTHEDOCS_CANONICAL_URL",
    "https://factorlasso.readthedocs.io/en/latest/",
)
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}


def _use_root_canonical(app, pagename, templatename, context, doctree) -> None:
    """Use the HTTPS version root, rather than index.html, as the landing canonical."""
    if pagename == "index":
        context["pageurl"] = app.config.html_baseurl


def setup(app) -> None:
    """Register documentation build hooks."""
    app.connect("html-page-context", _use_root_canonical)


linkcheck_anchors = True
linkcheck_timeout = 20
