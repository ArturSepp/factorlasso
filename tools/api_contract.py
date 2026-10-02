"""Record and check the importable API contract of ``factorlasso``.

The contract is what a restructuring of the package must keep: the root exports, the call
signatures and dataclass fields of every exported object, enum members, and every
factorlasso-owned name that each historical module made importable. It is generated once
from a reviewed revision and stored in ``tests/data/api_contract.json``;
``tests/test_api_contract.py`` rebuilds it from the installed package and requires equality.

Values are serialised semantically (``None``, literals, enum names, factory names), never by
``repr``, so the contract is stable across Python versions and object addresses. Annotations
are not recorded: they render differently across the supported Python versions.

Usage::

    python tools/api_contract.py --write tests/data/api_contract.json
    python tools/api_contract.py --check tests/data/api_contract.json

``--write`` regenerates the reviewed fixture and is run only when the contract is meant to
change. Legacy names are derived from module source with :mod:`ast` at generation time; a check
uses the names recorded in the fixture.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import enum
import importlib
import inspect
import json
import math
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

#: Modules that formed the flat 0.23.0 layout. Each stays importable with its names.
LEGACY_MODULES = (
    "beta_priors",
    "cluster_lineage",
    "cluster_smoothing",
    "cluster_standardization",
    "cluster_statistics",
    "cluster_utils",
    "cv",
    "dependence_utils",
    "diagonality",
    "ewm_utils",
    "expert_prior_map",
    "factor_covar",
    "lasso_estimator",
    "prior_bounds",
    "prior_inference",
    "prior_risk",
    "residual_covar",
    "residual_diagnostics",
    "sign_constraints",
)

_NO_DEFAULT = "<no default>"


def serialise_value(value: Any) -> Any:
    """Return a JSON-compatible, address-free description of a default or constant."""
    if value is inspect.Parameter.empty or value is dataclasses.MISSING:
        return _NO_DEFAULT
    if type(value).__name__ == "_HAS_DEFAULT_FACTORY_CLASS":
        # dataclass ``__init__`` signatures show this sentinel for ``default_factory`` fields
        return "<default factory>"
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, enum.Enum):
        return {"enum": type(value).__qualname__, "name": value.name,
                "value": serialise_value(value.value)}
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"float": repr(value)}
    if isinstance(value, (tuple, list, frozenset, set)):
        items = [serialise_value(item) for item in value]
        if isinstance(value, (frozenset, set)):
            items = sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
        return {type(value).__name__: items}
    if isinstance(value, dict):
        return {"dict": [[serialise_value(key), serialise_value(item)]
                         for key, item in value.items()]}
    if isinstance(value, type):
        return {"type": value.__qualname__}
    if callable(value):
        return {"callable": getattr(value, "__qualname__", type(value).__qualname__)}
    return {"instance": type(value).__qualname__}


def signature_record(obj: Any) -> Any:
    """Describe a callable's parameters as ``[name, kind, default]`` triples."""
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return "<no signature>"
    return [[name, parameter.kind.name, serialise_value(parameter.default)]
            for name, parameter in signature.parameters.items()]


def _is_own(obj: Any) -> bool:
    """Whether ``obj`` was defined inside the factorlasso package."""
    module = getattr(obj, "__module__", None) or ""
    return module == "factorlasso" or module.startswith("factorlasso.")


def _field_record(field: dataclasses.Field) -> Dict[str, Any]:
    """Describe one dataclass field's construction contract."""
    factory = field.default_factory
    return {
        "name": field.name,
        "init": field.init,
        "default": serialise_value(field.default),
        "factory": None if factory is dataclasses.MISSING else serialise_value(factory),
    }


def _members(cls: type) -> Dict[str, Any]:
    """Public methods and properties that factorlasso classes define on ``cls``."""
    members: Dict[str, Any] = {}
    enum_members = getattr(cls, "__members__", {}) if issubclass(cls, enum.Enum) else {}
    for name in sorted(dir(cls)):
        if name.startswith("_") or name in enum_members:
            continue
        owner = next((base for base in cls.__mro__ if name in vars(base)), None)
        if owner is None or not _is_own(owner):
            continue
        raw = vars(owner)[name]
        if isinstance(raw, property):
            members[name] = {"property": True, "setter": raw.fset is not None}
        elif isinstance(raw, staticmethod):
            members[name] = {"staticmethod": signature_record(raw.__func__)}
        elif isinstance(raw, classmethod):
            members[name] = {"classmethod": signature_record(getattr(cls, name))}
        elif inspect.isfunction(raw):
            members[name] = {"method": signature_record(raw)}
    return members


def class_record(cls: type) -> Dict[str, Any]:
    """Describe an exported class: enum members, dataclass fields, constructor and members."""
    record: Dict[str, Any] = {"kind": "class"}
    if issubclass(cls, enum.Enum):
        record["kind"] = "enum"
        record["enum_members"] = [[member.name, serialise_value(member.value)]
                                  for member in cls]
    if dataclasses.is_dataclass(cls):
        params = getattr(cls, "__dataclass_params__", None)
        record["dataclass"] = {
            "frozen": bool(getattr(params, "frozen", False)),
            "fields": [_field_record(field) for field in dataclasses.fields(cls)],
        }
    if record["kind"] == "class":
        init_owner = next((base for base in cls.__mro__ if "__init__" in vars(base)), object)
        record["init"] = (signature_record(cls.__init__) if _is_own(init_owner)
                          else "<inherited>")
    record["members"] = _members(cls)
    return record


def export_record(obj: Any) -> Dict[str, Any]:
    """Describe one exported object."""
    if isinstance(obj, type):
        return class_record(obj)
    if callable(obj):
        return {"kind": "function", "signature": signature_record(obj)}
    return {"kind": "constant", "value": serialise_value(obj)}


def legacy_names_from_source(path: Path) -> Dict[str, Optional[str]]:
    """Top-level factorlasso names of a module source, mapped to their source module.

    Module-defined functions, classes and assigned names map to ``None``. Names imported from
    another factorlasso module map to ``"<dotted module>:<original name>"``. Imports of
    third-party or standard-library modules, dunder names and conditional (``if``/``try``)
    blocks are skipped.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: Dict[str, Optional[str]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names[node.name] = None
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names[target.id] = None
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names[node.target.id] = None
        elif isinstance(node, ast.ImportFrom) and node.module and (
            node.module == "factorlasso" or node.module.startswith("factorlasso.")
        ):
            for alias in node.names:
                names[alias.asname or alias.name] = f"{node.module}:{alias.name}"
    return {name: source for name, source in sorted(names.items())
            if not (name.startswith("__") and name.endswith("__"))}


def _legacy_entry(module: types.ModuleType, name: str, root: types.ModuleType,
                  source: Optional[str]) -> Dict[str, Any]:
    """Describe one name of a historical module as seen through ``module``."""
    if not hasattr(module, name):
        return {"missing": True}
    obj = getattr(module, name)
    entry: Dict[str, Any] = {"source": source}
    if name in root.__all__:
        entry["is_root_export"] = getattr(root, name) is obj
    if isinstance(obj, type) or inspect.isfunction(obj) or inspect.isbuiltin(obj):
        entry["qualname"] = obj.__qualname__
    elif isinstance(obj, types.ModuleType):
        entry["module"] = obj.__name__
    else:
        entry["value"] = serialise_value(obj)
    if source is not None:
        origin_module, origin_name = source.split(":")
        origin = importlib.import_module(origin_module)
        entry["is_source_object"] = getattr(origin, origin_name, None) is obj
    return entry


def build_contract(legacy: Optional[Dict[str, Dict[str, Optional[str]]]] = None) -> Dict[str, Any]:
    """Build the contract from the importable package.

    Parameters
    ----------
    legacy : dict, optional
        ``{module: {name: source}}`` as stored in a fixture. ``None`` derives it from the
        current module sources, which is correct only on the revision that defines the
        contract.
    """
    root = importlib.import_module("factorlasso")
    if legacy is None:
        legacy = {}
        for module_name in LEGACY_MODULES:
            module = importlib.import_module(f"factorlasso.{module_name}")
            legacy[module_name] = legacy_names_from_source(Path(module.__file__))
    exports = {name: export_record(getattr(root, name)) for name in root.__all__}
    legacy_records: Dict[str, Any] = {}
    for module_name, names in legacy.items():
        module = importlib.import_module(f"factorlasso.{module_name}")
        legacy_records[module_name] = {
            "root_attribute": getattr(root, module_name, None) is module,
            "__all__": list(getattr(module, "__all__", [])) or None,
            "names": {name: _legacy_entry(module, name, root, source)
                      for name, source in names.items()},
        }
    return {
        "root_all": list(root.__all__),
        "exports": exports,
        "legacy_sources": legacy,
        "legacy": legacy_records,
    }


def differences(expected: Any, actual: Any, path: str = "") -> List[str]:
    """List the paths at which two contract trees differ."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        out: List[str] = []
        for key in sorted(set(expected) | set(actual), key=str):
            if key not in actual:
                out.append(f"{path}/{key}: missing")
            elif key not in expected:
                out.append(f"{path}/{key}: unexpected")
            else:
                out.extend(differences(expected[key], actual[key], f"{path}/{key}"))
        return out
    if (isinstance(expected, list) and isinstance(actual, list)
            and len(expected) == len(actual) and expected != actual):
        out = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            out.extend(differences(left, right, f"{path}[{index}]"))
        return out
    if expected != actual:
        return [f"{path}: expected {json.dumps(expected)[:200]}, got {json.dumps(actual)[:200]}"]
    return []


def load(path: Path) -> Dict[str, Any]:
    """Load a stored contract."""
    return json.loads(path.read_text(encoding="utf-8"))


def check(path: Path) -> List[str]:
    """Rebuild the contract with the fixture's legacy names and list differences."""
    expected = load(path)
    actual = build_contract(expected["legacy_sources"])
    return differences(expected, actual)


def _dump(contract: Dict[str, Any]) -> str:
    """Serialise a contract deterministically."""
    return json.dumps(contract, indent=1, sort_keys=False, ensure_ascii=True) + "\n"


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", type=Path, help="write the contract derived from this revision")
    group.add_argument("--check", type=Path, help="compare the package against a stored contract")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(_dump(build_contract()), encoding="utf-8")
        print(f"wrote {args.write}")
        return 0
    found = check(args.check)
    for line in found:
        print(line)
    print(f"{len(found)} difference(s)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
