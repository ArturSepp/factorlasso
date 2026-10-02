"""Support for the historical flat module layout.

The modules of the 0.23.0 layout (``factorlasso.lasso_estimator``, ``factorlasso.ewm_utils``,
...) remain importable as facades that re-export every name from its canonical owner. A facade
holds references, not the objects the package uses internally, so assigning one of its names
(``monkeypatch.setattr(factorlasso.lasso_estimator, "_solve_with_fallback", ...)``) would have
no effect on factorlasso and a patched run would silently compute the unpatched result. Facades
therefore reject assignment and deletion of their re-exported names with an
:class:`AttributeError`. Importing stays silent.

In 0.23 one module held each name, so one assignment changed every internal call. A helper can
now be imported by name into several implementation modules, and each of them looks the name up
in its own namespace. :func:`patch_points` lists those modules; patching the name in all of them
reproduces the historical effect. The error raised by a facade names them too.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Dict, List


class _LegacyModule(types.ModuleType):
    """Module type of a historical facade: re-exported names are read-only."""

    def _redirect(self, name: str) -> None:
        """Raise for a re-exported ``name``; other attributes behave normally."""
        owners: Dict[str, str] = self.__dict__.get("_legacy_owners", {})
        if name in owners:
            targets = ", ".join(module.__name__ for module in patch_points(self.__name__, name))
            raise AttributeError(
                f"{self.__name__}.{name} is a compatibility re-export from {owners[name]}; "
                f"changing it here would not affect factorlasso. Patch {name} in every module "
                f"where factorlasso looks it up: {targets} "
                f"(factorlasso._compat.patch_points lists them)."
            )

    def __setattr__(self, name: str, value) -> None:
        """Reject assignment of a re-exported name."""
        self._redirect(name)
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        """Reject deletion of a re-exported name."""
        self._redirect(name)
        super().__delattr__(name)


def _is_implementation(dotted: str) -> bool:
    """Whether ``dotted`` names a private implementation module of the package."""
    parts = dotted.split(".")
    return (parts[0] == "factorlasso" and dotted != "factorlasso._compat"
            and any(part.startswith("_") for part in parts[1:]))


def patch_points(module_name: str, name: str) -> List[types.ModuleType]:
    """Implementation modules in which factorlasso looks up ``name`` of a historical module.

    Parameters
    ----------
    module_name : str
        A historical module, e.g. ``"factorlasso.lasso_estimator"``.
    name : str
        A name it re-exports, e.g. ``"_solve_with_fallback"``.

    Returns
    -------
    list of module
        Every loaded private module (a module path with an underscore component) that holds
        the same object under ``name``, sorted by module name. Assigning the replacement in all
        of them reproduces an assignment on the 0.23 module. Helpers that factorlasso imports
        when they are called, such as the sign derivation, have one entry: their defining
        module.

    Raises
    ------
    KeyError
        If ``name`` is not a re-export of ``module_name``.

    Examples
    --------
    >>> from unittest import mock
    >>> from factorlasso._compat import patch_points
    >>> targets = patch_points("factorlasso.lasso_estimator", "_solve_with_fallback")
    >>> patches = [mock.patch.object(module, "_solve_with_fallback") for module in targets]
    """
    facade = importlib.import_module(module_name)
    owners: Dict[str, str] = facade.__dict__.get("_legacy_owners", {})
    if name not in owners:
        raise KeyError(f"{module_name}.{name} is not a compatibility re-export")
    value = facade.__dict__[name]
    return [module for dotted, module in sorted(list(sys.modules.items()))
            if module is not None and _is_implementation(dotted)
            and not isinstance(module, _LegacyModule)
            and module.__dict__.get(name, None) is value]


def guard_legacy_module(module_name: str) -> None:
    """Turn the already imported module ``module_name`` into a read-only facade.

    Every non-dunder global of the module at call time is treated as a re-export. Its owner
    is the ``__module__`` of the object when it has one (functions, classes, enums) and
    otherwise the canonical module that holds the same object under the same name.
    """
    module = sys.modules[module_name]
    namespace = module.__dict__
    namespace.pop("guard_legacy_module", None)
    owners: Dict[str, str] = {}
    for name, value in namespace.items():
        if name.startswith("__") and name.endswith("__"):
            continue
        owner = getattr(value, "__module__", None) if not isinstance(value, types.ModuleType) \
            else value.__name__
        if not isinstance(owner, str) or not owner.startswith("factorlasso"):
            owner = next((other for other, loaded in list(sys.modules.items())
                          if other.startswith("factorlasso.") and loaded is not module
                          and not isinstance(loaded, _LegacyModule)
                          and getattr(loaded, name, None) is value), module_name)
        owners[name] = owner
    namespace["_legacy_owners"] = owners
    module.__class__ = _LegacyModule
