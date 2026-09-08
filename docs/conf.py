"""Sphinx configuration for the FactorLasso documentation."""

import os
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 metadata tests use the compatible parser.
    import tomli as tomllib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

project = "factorlasso"
author = "Artur Sepp and Mika Kastenholz"
copyright = "2026, Artur Sepp and Mika Kastenholz"
release = tomllib.loads(
    (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
)["project"]["version"]
version = release

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

GOOGLE_SITE_VERIFICATION = "cddUZk3Gsd1MySw42Rwuq_rMzUDcMNkJWekObx-QS9Y"

rst_prolog = f"""
.. meta::
   :google-site-verification: {GOOGLE_SITE_VERIFICATION}

.. |t| replace:: t
.. |beta| replace:: beta
"""

html_theme = "furo"
html_title = "factorlasso - sparse factor-model estimation"
html_baseurl = os.environ.get(
    "READTHEDOCS_CANONICAL_URL",
    "https://factorlasso.readthedocs.io/en/latest/",
)


def _use_root_canonical(app, pagename, templatename, context, doctree) -> None:
    """Use the HTTPS version root, rather than index.html, as the landing canonical."""
    if pagename == "index":
        context["pageurl"] = app.config.html_baseurl


def setup(app) -> None:
    """Register documentation build hooks."""
    app.connect("html-page-context", _use_root_canonical)


linkcheck_anchors = True
# Archived research links occasionally exceed 20 seconds; keep checks bounded and retry once.
linkcheck_timeout = 30
linkcheck_retries = 2
