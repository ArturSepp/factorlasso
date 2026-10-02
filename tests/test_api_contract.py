"""The importable API contract recorded in ``tests/data/api_contract.json``.

The fixture pins the root exports, signatures, dataclass fields and enum members of every
exported object, and every factorlasso name that the historical flat modules made importable.
It is regenerated with ``tools/api_contract.py --write`` only when a contract change is intended
and reviewed. Each check is shown to fail on the defect it exists to catch.
"""

import dataclasses
import importlib
import importlib.util
import inspect
import json
import types
from pathlib import Path

import pytest

import factorlasso

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent / "data" / "api_contract.json"


@pytest.fixture(scope="module")
def contract_tool():
    """Load ``tools/api_contract.py`` by path so the test works under any import mode."""
    path = REPOSITORY_ROOT / "tools" / "api_contract.py"
    if not path.is_file():
        pytest.skip("the contract tool is not available outside a checkout")
    spec = importlib.util.spec_from_file_location("api_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_package_matches_recorded_contract(contract_tool):
    """Root exports, signatures, fields, enums and legacy imports equal the reviewed fixture."""
    found = contract_tool.check(FIXTURE)
    assert not found, "API contract differences:\n" + "\n".join(found[:50])


def test_contract_detects_a_changed_default(contract_tool):
    """A changed keyword default is a contract difference."""

    def before(x, span=None, demean=True):
        """Reference signature."""

    def after(x, span=None, demean=False):
        """Same signature with one changed default."""

    assert contract_tool.differences(
        contract_tool.signature_record(before), contract_tool.signature_record(after)
    )


def test_contract_detects_a_changed_dataclass_field(contract_tool):
    """Reordering a field or changing its default changes the class record."""

    @dataclasses.dataclass
    class Before:
        """Reference fields."""

        a: int = 1
        b: int = 2

    @dataclasses.dataclass
    class After:
        """Same fields, changed default."""

        a: int = 1
        b: int = 3

    assert contract_tool.differences(
        contract_tool.class_record(Before)["dataclass"],
        contract_tool.class_record(After)["dataclass"],
    )


def test_contract_detects_a_missing_legacy_name(contract_tool):
    """A historical module that no longer provides a recorded name is reported."""
    module = types.ModuleType("factorlasso.fake_legacy")
    entry = contract_tool._legacy_entry(module, "removed_name", factorlasso, None)
    assert entry == {"missing": True}


def test_legacy_modules_remain_root_attributes():
    """``import factorlasso`` keeps every historical module reachable as an attribute."""
    for name in (
        "beta_priors", "cluster_lineage", "cluster_smoothing", "cluster_standardization",
        "cluster_statistics", "cluster_utils", "cv", "dependence_utils", "diagonality",
        "ewm_utils", "expert_prior_map", "factor_covar", "lasso_estimator", "prior_bounds",
        "prior_inference", "prior_risk", "residual_covar", "residual_diagnostics",
        "sign_constraints",
    ):
        assert isinstance(getattr(factorlasso, name), types.ModuleType), name


def _facades():
    """The historical modules that are compatibility facades."""
    from factorlasso._compat import _LegacyModule

    modules = [getattr(factorlasso, name) for name in sorted(
        json.loads(FIXTURE.read_text(encoding="utf-8"))["legacy"])]
    return [module for module in modules if isinstance(module, _LegacyModule)]


def test_facades_reject_assignment_of_reexported_names(monkeypatch):
    """Patching a facade would not reach the implementation, so it fails loudly instead."""
    facades = _facades()
    assert facades
    for module in facades:
        name = sorted(module._legacy_owners)[0]
        owner = module._legacy_owners[name]
        with pytest.raises(AttributeError, match=owner.replace(".", r"\.")):
            setattr(module, name, None)
        with pytest.raises(AttributeError):
            delattr(module, name)
        with pytest.raises(AttributeError):
            monkeypatch.setattr(module, name, None)
        assert getattr(module, name) is getattr(importlib.import_module(owner), name, None) \
            or owner == module.__name__


def test_facades_keep_ordinary_attribute_behaviour():
    """Names that are not re-exports can still be set and removed."""
    module = _facades()[0]
    module.not_a_reexport = 1
    del module.not_a_reexport
    assert not hasattr(module, "not_a_reexport")


def test_estimator_patch_target_reaches_the_fit(monkeypatch):
    """Patching the canonical owner affects ``fit``; the facade name is the same object."""
    import factorlasso.linear_model._estimator as owner
    from factorlasso import LassoModelType

    calls = []
    original = owner.solve_lasso_cvx_problem

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, "solve_lasso_cvx_problem", spy)
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.standard_normal((30, 2)), columns=["f0", "f1"])
    y = pd.DataFrame(rng.standard_normal((30, 2)), columns=["a", "b"])
    factorlasso.LassoModel(model_type=LassoModelType.LASSO).fit(x, y)
    assert calls == [1]


def test_preparation_seam_is_retained():
    """Tests and research scripts wrap ``LassoModel._prepare_fit`` and read its record."""
    from factorlasso.lasso_estimator import LassoModel, _PreparedFit

    parameters = list(inspect.signature(LassoModel._prepare_fit).parameters)
    assert parameters == [
        "self", "x", "y", "x_np", "y_np", "valid_mask", "eff_span",
        "eff_cluster_correlation_span", "external_clusters", "external_linkage",
        "external_cutoff",
    ]
    assert [field.name for field in dataclasses.fields(_PreparedFit)] == [
        "asset_clusters", "linkage", "cutoff", "is_lasso_mode", "signs_np", "prior_np",
        "penalty_weights_np", "row_weights_np", "col_weights_np", "lower_bounds_np",
        "upper_bounds_np",
    ]
