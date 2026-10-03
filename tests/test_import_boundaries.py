"""Import boundaries between the capability subpackages.

The package is layered: ``utils`` below ``cluster`` and ``priors``; ``covariance`` above
``cluster``; ``linear_model`` above ``cluster`` and ``priors``; ``diagnostics`` above
``covariance``; ``model_selection`` on top. Package code imports the private module that owns
a name, never a historical facade, the package root or another subpackage's ``__init__``.
The checks parse the installed sources with :mod:`ast` (standard library only), so they also
run against an installed wheel. Each check is shown to fail on a synthetic violation.
"""

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import factorlasso

PACKAGE_ROOT = Path(factorlasso.__file__).resolve().parent

#: Capability subpackage -> the other capabilities it may import.
ALLOWED = {
    "utils": set(),
    "cluster": {"utils"},
    "priors": {"utils"},
    "covariance": {"utils", "cluster"},
    "linear_model": {"utils", "cluster", "priors"},
    "diagnostics": {"utils", "covariance"},
    "model_selection": {"utils", "linear_model", "diagnostics"},
}

#: Imports against the layering that exist only for type checking: (importer, imported).
TYPING_ONLY_EXCEPTIONS = {
    ("factorlasso.cluster._smoothing", "factorlasso.linear_model._estimator"),
}

#: Modules of the 0.23.0 flat layout, removed in 1.0.
LEGACY = {
    "beta_priors", "cluster_lineage", "cluster_smoothing", "cluster_standardization",
    "cluster_statistics", "cluster_utils", "cv", "dependence_utils", "diagonality",
    "ewm_utils", "expert_prior_map", "factor_covar", "lasso_estimator", "prior_bounds",
    "prior_inference", "prior_risk", "residual_covar", "residual_diagnostics",
    "sign_constraints",
}


@dataclass(frozen=True)
class Edge:
    """One import of a factorlasso module by a factorlasso module."""
    importer: str
    imported: str
    typing_only: bool
    module_level: bool


def _module_name(path: Path) -> str:
    """Dotted module name of a package source file."""
    parts = path.relative_to(PACKAGE_ROOT.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _sources() -> Dict[str, Path]:
    """Every module of the installed package."""
    return {_module_name(path): path for path in sorted(PACKAGE_ROOT.rglob("*.py"))}


def _resolve(importer: str, is_package: bool, node: ast.ImportFrom) -> str:
    """Absolute module named by a (possibly relative) ``from`` import."""
    if not node.level:
        return node.module or ""
    base = importer.split(".") if is_package else importer.split(".")[:-1]
    base = base[: len(base) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def _is_type_checking(test: ast.expr) -> bool:
    """``if TYPE_CHECKING:`` or ``if typing.TYPE_CHECKING:``."""
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


def edges(sources: Optional[Dict[str, Path]] = None) -> List[Edge]:
    """All factorlasso-to-factorlasso imports, at any nesting, with their context."""
    sources = _sources() if sources is None else sources
    found: List[Edge] = []
    for name, path in sources.items():
        is_package = path.name == "__init__.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))

        def visit(node, typing_only: bool, module_level: bool):
            if isinstance(node, ast.If) and _is_type_checking(node.test):
                for child in node.body:
                    visit(child, True, module_level)
                for child in node.orelse:
                    visit(child, typing_only, module_level)
                return
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "factorlasso":
                        found.append(Edge(name, alias.name, typing_only, module_level))
                return
            if isinstance(node, ast.ImportFrom):
                target = _resolve(name, is_package, node)
                if target.split(".")[0] != "factorlasso":
                    return
                for alias in node.names:
                    submodule = f"{target}.{alias.name}"
                    imported = submodule if submodule in sources else target
                    found.append(Edge(name, imported, typing_only, module_level))
                return
            nested = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for child in ast.iter_child_nodes(node):
                visit(child, typing_only, module_level and not nested)

        for statement in tree.body:
            visit(statement, False, True)
    return found


def _capability(module: str) -> Optional[str]:
    """Capability subpackage of ``module``, or None outside those packages."""
    parts = module.split(".")
    return parts[1] if len(parts) > 1 and parts[1] in ALLOWED else None


def violations(found: Iterable[Edge],
               exceptions: Set[Tuple[str, str]] = TYPING_ONLY_EXCEPTIONS) -> List[str]:
    """Boundary violations, including typing-only exceptions that are no longer used."""
    problems: List[str] = []
    used: Set[Tuple[str, str]] = set()
    for edge in found:
        source = _capability(edge.importer)
        if edge.importer == "factorlasso._compat":
            problems.append(f"{edge.importer} imports {edge.imported}")
            continue
        if source is None:
            continue                      # root initializer
        target = _capability(edge.imported)
        last = edge.imported.split(".")[-1]
        if edge.imported == "factorlasso" or (target is None and last in LEGACY):
            problems.append(f"{edge.importer} imports the facade/root {edge.imported}")
            continue
        if target is None:
            problems.append(f"{edge.importer} imports {edge.imported}")
            continue
        if target != source and edge.imported == f"factorlasso.{target}":
            problems.append(f"{edge.importer} imports the {target} package initializer")
            continue
        if target == source or target in ALLOWED[source]:
            continue
        pair = (edge.importer, edge.imported)
        if edge.typing_only and pair in exceptions:
            used.add(pair)
            continue
        kind = "typing-only" if edge.typing_only else "run-time"
        problems.append(f"{edge.importer} imports {edge.imported} ({kind}, {source} -> {target})")
    problems += [f"obsolete typing-only exception {pair}" for pair in sorted(exceptions - used)]
    return problems


def test_package_respects_the_layering():
    """No upward, sideways, facade, root or initializer import in package code."""
    assert violations(edges()) == []


def test_module_level_imports_are_acyclic():
    """The run-time module graph among private modules has no cycle."""
    graph: Dict[str, Set[str]] = {}
    for edge in edges():
        if edge.module_level and not edge.typing_only and _capability(edge.importer):
            graph.setdefault(edge.importer, set()).add(edge.imported)
    state: Dict[str, int] = {}

    def visit(node: str, trail: Tuple[str, ...]) -> None:
        if state.get(node) == 1:
            raise AssertionError(" -> ".join(trail + (node,)))
        if state.get(node) == 2:
            return
        state[node] = 1
        for target in sorted(graph.get(node, ())):
            visit(target, trail + (node,))
        state[node] = 2

    for node in sorted(graph):
        visit(node, ())




def test_checker_rejects_injected_violations():
    """An upward import, a facade import, an initializer import and a stale exception fail."""
    upward = Edge("factorlasso.utils._ewm", "factorlasso.cluster._hierarchical", False, False)
    facade = Edge("factorlasso.cluster._smoothing", "factorlasso.lasso_estimator", False, True)
    initializer = Edge("factorlasso.linear_model._preparation", "factorlasso.cluster",
                       False, True)
    typing_edge = Edge("factorlasso.cluster._smoothing", "factorlasso.linear_model._estimator",
                       True, True)
    assert len(violations([upward, typing_edge])) == 1
    assert len(violations([facade, typing_edge])) == 1
    assert len(violations([initializer, typing_edge])) == 1
    assert violations([typing_edge]) == []
    stale = {("factorlasso.priors._signs", "factorlasso.linear_model._types")}
    assert violations([typing_edge], TYPING_ONLY_EXCEPTIONS | stale) == [
        f"obsolete typing-only exception {sorted(stale)[0]}"]
    runtime_edge = Edge(typing_edge.importer, typing_edge.imported, False, False)
    assert len(violations([runtime_edge])) == 2      # violation, and the unused exception
