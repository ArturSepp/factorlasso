Public API reference
====================

.. meta::
   :description: Reference for the supported top-level factorlasso estimators, diagnostics,
      clustering tools, covariance containers and solver helpers, grouped by the methodology
      article that explains each name, with a map of the LassoModel parameters by topic.

The names exported by ``factorlasso.__all__`` define the supported top-level public API for the
current release line. Modules and names that are importable only through implementation paths are
not additional public entry points; see the `compatibility policy
<https://github.com/ArturSepp/factorlasso/blob/main/COMPATIBILITY.md>`_ for details.

This page is generated when the documentation is built, from ``factorlasso.__all__`` and the
ownership recorded in ``tools/docs_inventory.json``. Each public name belongs to exactly one
methodology article, and the sections follow the order of the sidebar. A test fails when a new
export or a new ``LassoModel`` parameter has no owning article, so the page cannot drift from the
package.

.. py:module:: factorlasso

.. include:: _generated/api_reference.rst
