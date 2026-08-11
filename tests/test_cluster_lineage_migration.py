"""Migration, dependency-surface, and docstring tests for cluster lineage."""

import ast
import inspect
from pathlib import Path

import factorlasso
import factorlasso.cluster_lineage as cluster_lineage


def test_cluster_lineage_public_exports_are_canonical() -> None:
    """The four canonical lineage symbols are exported from the package root."""
    assert factorlasso.analyze_cluster_lineage is cluster_lineage.analyze_cluster_lineage
    assert factorlasso.run_cluster_lineage_report is cluster_lineage.run_cluster_lineage_report
    assert factorlasso.RiskClusterReport is cluster_lineage.RiskClusterReport
    assert factorlasso.TaxonomyConfig is cluster_lineage.TaxonomyConfig


def test_cluster_lineage_runtime_import_surface_stays_leaf_only() -> None:
    """Package code imports only the scientific core, factorlasso internals, and stdlib."""
    source = Path(inspect.getfile(cluster_lineage)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    allowed = {
        "__future__",
        "dataclasses",
        "typing",
        "numpy",
        "pandas",
        "scipy",
        "factorlasso",
    }
    imported_roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots <= allowed
    assert "matplotlib" not in imported_roots


def test_cluster_lineage_docstrings_use_numpydoc_sections() -> None:
    """The moved module contains no Google-style public docstring sections."""
    source = Path(inspect.getfile(cluster_lineage)).read_text(encoding="utf-8")
    assert "Args:" not in source
    assert "Returns:" not in source
    assert "Raises:" not in source
