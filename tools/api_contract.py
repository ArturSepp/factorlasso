"""Record and check the importable public API contract of ``factorlasso``.

The reviewed fixture pins root exports, call signatures, dataclass fields, enum members,
class module paths and capability subpackage exports. Annotations are omitted because
their rendering varies across Python versions; defaults are serialised semantically.

Use ``--check tests/data/api_contract.json`` to validate the installed package. ``--write``
and ``--update`` generate a fixture only for an intended, reviewed public API change.
"""

from __future__ import annotations

import argparse
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


#: Capability subpackages; each exports a subset of the root ``__all__``.
SUBPACKAGES = (
    "utils", "linear_model", "cluster", "priors", "covariance", "diagnostics",
    "model_selection",
)

_NO_DEFAULT = "<no default>"


def _is_typing_alias(value: Any) -> bool:
    """Whether ``value`` is a typing construct, such as ``Union[np.ndarray, pd.DataFrame]``.

    Their runtime representation changes across Python versions (3.14 made ``Union`` objects
    non-callable), so only their kind is recorded.
    """
    return (isinstance(value, getattr(types, "UnionType", ()))
            or isinstance(value, getattr(types, "GenericAlias", ()))
            or type(value).__module__ == "typing")


def serialise_value(value: Any) -> Any:
    """Return a JSON-compatible, address-free description of a default or constant.

    Floats are kept to 12 significant digits: computed constants (such as a ``logspace``
    grid) can differ in the last bit between NumPy builds and platforms.
    """
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
        return float(f"{value:.12g}") if math.isfinite(value) else {"float": repr(value)}
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
    if _is_typing_alias(value):
        return {"typing": "alias"}
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
    """Describe an exported class: enum members, dataclass fields, constructor and members.

    ``module`` is the class's ``__module__``: pickles store it, so it is pinned once the
    layout is final and a later move is visible in review.
    """
    record: Dict[str, Any] = {"kind": "class", "module": cls.__module__}
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


def build_contract() -> Dict[str, Any]:
    """Build the public root and capability subpackage contract."""
    root = importlib.import_module("factorlasso")
    exports = {name: export_record(getattr(root, name)) for name in root.__all__}
    subpackages = {name: list(importlib.import_module(f"factorlasso.{name}").__all__)
                   for name in SUBPACKAGES}
    return {"root_all": list(root.__all__), "exports": exports, "subpackages": subpackages}


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
    """Rebuild the public contract and list differences."""
    expected = load(path)
    actual = build_contract()
    return differences(expected, actual)


def _dump(contract: Dict[str, Any]) -> str:
    """Serialise a contract deterministically."""
    return json.dumps(contract, indent=1, sort_keys=False, ensure_ascii=True) + "\n"


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", type=Path, help="write the contract derived from this revision")
    group.add_argument("--update", type=Path,
                       help="rewrite a stored contract for a reviewed public API change")
    group.add_argument("--check", type=Path, help="compare the package against a stored contract")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(_dump(build_contract()), encoding="utf-8", newline="\n")
        print(f"wrote {args.write}")
        return 0
    if args.update:
        args.update.write_text(_dump(build_contract()), encoding="utf-8", newline="\n")
        print(f"updated {args.update}")
        return 0
    found = check(args.check)
    for line in found:
        print(line)
    print(f"{len(found)} difference(s)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
