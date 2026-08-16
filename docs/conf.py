"""Sphinx configuration for the FactorLasso documentation."""

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

rst_prolog = """
.. |t| replace:: t
.. |beta| replace:: beta
"""

html_theme = "sphinx_rtd_theme"
html_title = f"factorlasso {release}"
html_baseurl = "https://factorlasso.readthedocs.io/"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}

linkcheck_anchors = True
linkcheck_timeout = 20
