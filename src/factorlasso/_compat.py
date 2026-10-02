"""Support for the historical flat module layout.

The modules of the 0.23.0 layout (``factorlasso.lasso_estimator``, ``factorlasso.ewm_utils``,
...) remain importable as facades that re-export every name from its canonical owner. A facade
holds references, not the objects the package uses internally, so assigning one of its names
(``monkeypatch.setattr(factorlasso.lasso_estimator, "_solve_with_fallback", ...)``) would have
no effect on factorlasso and a patched run would silently compute the unpatched result. Facades
therefore reject assignment and deletion of their re-exported names with an
:class:`AttributeError` that names the module to patch instead. Importing stays silent.
"""

from __future__ import annotations

import sys
import types
from typing import Dict


class _LegacyModule(types.ModuleType):
    """Module type of a historical facade: re-exported names are read-only."""

    def _redirect(self, name: str) -> None:
        """Raise for a re-exported ``name``; other attributes behave normally."""
        owners: Dict[str, str] = self.__dict__.get("_legacy_owners", {})
        if name in owners:
            raise AttributeError(
                f"{self.__name__}.{name} is a compatibility re-export from {owners[name]}; "
                f"changing it here would not affect factorlasso. Patch {owners[name]}.{name} "
                f"instead."
            )

    def __setattr__(self, name: str, value) -> None:
        """Reject assignment of a re-exported name."""
        self._redirect(name)
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        """Reject deletion of a re-exported name."""
        self._redirect(name)
        super().__delattr__(name)


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
