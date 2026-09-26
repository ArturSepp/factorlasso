"""Keep canonical HTML URLs and the public page sitemap consistent.

Adapted from ``QuantInvestStrats/docs/_ext/qis_indexing.py``:

    Copyright (c) Artur Sepp. That file is distributed under the MIT License; the permission
    notice of that licence is retained for the adapted portions. This file, as part of
    factorlasso, is distributed under GPL-3.0-or-later.

Read the Docs serves the default version's sitemap at the domain root. Listing the actual pages
gives crawlers one preferred discovery path instead of only the version roots. The moving
``latest`` and ``stable`` aliases describe the same default documentation, so both use ``latest``
as their canonical URL, the version the site links to. Numbered releases keep their own canonical
base URL, because their API documentation can differ. qis consolidates onto ``stable`` instead;
the choice is per package.
"""

from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree

SITEMAP_NAMESPACE = "http://www.sitemaps.org/schemas/sitemap/0.9"
READTHEDOCS_ALIAS_PATHS = {"/en/latest", "/en/stable"}
CANONICAL_ALIAS_PATH = "/en/latest"
EXCLUDED_PAGES = {"search", "genindex", "py-modindex"}


def canonical_baseurl(baseurl: str) -> str:
    """Consolidate Read the Docs' moving aliases onto the ``latest`` URL.

    Parameters
    ----------
    baseurl : str
        Configured ``html_baseurl``, with or without a trailing slash.

    Returns
    -------
    str
        Base URL with a trailing slash. Numbered release paths and deployments outside Read the
        Docs are returned unchanged apart from the slash.
    """
    parts = urlsplit(baseurl)
    path = parts.path.rstrip("/")
    if parts.netloc.endswith(".readthedocs.io") and path in READTHEDOCS_ALIAS_PATHS:
        path = CANONICAL_ALIAS_PATH
    return urlunsplit((parts.scheme, parts.netloc, path + "/", parts.query, parts.fragment))


def canonical_url(app: Any, pagename: str) -> str:
    """Return the same preferred URL for HTML metadata and sitemap entries.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        Application with an HTML builder and a configured ``html_baseurl``.
    pagename : str
        Sphinx document name, without the output suffix.

    Returns
    -------
    str
        Absolute URL, using the directory URL for the site's ``index.html``.
    """
    uri = app.builder.get_target_uri(pagename)
    if uri == "index.html":
        uri = ""
    return urljoin(canonical_baseurl(app.config.html_baseurl), quote(uri, safe="/%"))


def set_canonical_url(
    app: Any, pagename: str, templatename: str, context: dict, doctree: Any
) -> None:
    """Set the canonical URL consumed by the HTML theme, including the landing page.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        Running application.
    pagename : str
        Current document name.
    templatename : str
        HTML template selected by Sphinx; unchanged.
    context : dict
        Template variables; ``pageurl`` holds the canonical URL.
    doctree : docutils.nodes.document or None
        Parsed document, or None for generated helper pages; unused.
    """
    if app.builder.name == "html" and app.config.html_baseurl:
        context["pageurl"] = canonical_url(app, pagename)


def write_sitemap(app: Any, exception: Optional[Exception]) -> None:
    """Write a deterministic sitemap only after a successful HTML build.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        Application with the complete discovered document inventory.
    exception : Exception or None
        Build failure, if any; a failed build must not publish a sitemap.
    """
    if exception is not None or app.builder.name != "html" or not app.config.html_baseurl:
        return
    namespace = SITEMAP_NAMESPACE
    root = ElementTree.Element(f"{{{namespace}}}urlset")
    pages = (
        name
        for name in app.env.found_docs
        if name not in EXCLUDED_PAGES and not name.startswith("_modules/")
    )
    for url in sorted({canonical_url(app, name) for name in pages}):
        entry = ElementTree.SubElement(root, f"{{{namespace}}}url")
        ElementTree.SubElement(entry, f"{{{namespace}}}loc").text = url
    ElementTree.indent(root)
    ElementTree.ElementTree(root).write(
        Path(app.outdir) / "sitemap.xml",
        encoding="utf-8",
        xml_declaration=True,
        default_namespace=namespace,
    )


def setup(app: Any) -> dict:
    """Register the page-metadata and end-of-build sitemap callbacks.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        Application being configured.

    Returns
    -------
    dict
        Extension metadata; sitemap generation uses the complete merged inventory.
    """
    app.connect("html-page-context", set_canonical_url)
    app.connect("build-finished", write_sitemap)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
