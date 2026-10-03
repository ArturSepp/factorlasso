"""The importable API contract recorded in ``tests/data/api_contract.json``.

The fixture pins the root exports, signatures, dataclass fields and enum members of every
exported object, and their capability subpackage ownership.
It is regenerated with ``tools/api_contract.py --write`` only when a contract change is intended
and reviewed. Each check is shown to fail on the defect it exists to catch.
"""

import dataclasses
import importlib
import importlib.util
import inspect
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
    """Root exports, signatures, fields, enums and subpackage exports equal the reviewed fixture."""
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


def test_contract_values_are_portable(contract_tool):
    """Platform last-bit float noise and Python-version typing reprs do not change the record."""
    import typing

    import numpy as np

    serialise = contract_tool.serialise_value
    assert serialise(6.158482110660267e-06) == serialise(6.1584821106602665e-06)
    assert serialise(1e-5) != serialise(2e-5)
    assert serialise(typing.Union[np.ndarray, list]) == {"typing": "alias"}
    assert serialise(typing.Optional[int]) == {"typing": "alias"}


def test_subpackages_are_homes_of_root_names(contract_tool):
    """Each root export has exactly one capability subpackage, which re-exports that object.

    Subpackages promote nothing beyond ``factorlasso.__all__``; COMPATIBILITY.md defines the
    root as the stable surface.
    """
    homes = {}
    for name in contract_tool.SUBPACKAGES:
        subpackage = importlib.import_module(f"factorlasso.{name}")
        assert len(set(subpackage.__all__)) == len(subpackage.__all__), name
        for export in subpackage.__all__:
            assert export in factorlasso.__all__, f"{name}.{export} is not a root export"
            assert getattr(subpackage, export) is getattr(factorlasso, export)
            homes.setdefault(export, []).append(name)
    assert sorted(homes) == sorted(factorlasso.__all__)
    assert {name: places for name, places in homes.items() if len(places) > 1} == {}


def test_estimator_patch_target_reaches_the_fit(monkeypatch):
    """Patching the solver name where dispatch looks it up affects ``fit``."""
    import factorlasso.linear_model._dispatch as owner
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
    from factorlasso import LassoModel
    from factorlasso.linear_model._preparation import _PreparedFit

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


def test_removed_modules_are_not_shipped_or_importable():
    """The v1 package has no historical facades, guard, or root module aliases."""
    removed = (
        "beta_priors", "cluster_lineage", "cluster_smoothing", "cluster_standardization",
        "cluster_statistics", "cluster_utils", "cv", "dependence_utils", "diagonality",
        "ewm_utils", "expert_prior_map", "factor_covar", "lasso_estimator", "prior_bounds",
        "prior_inference", "prior_risk", "residual_covar", "residual_diagnostics",
        "sign_constraints", "_compat",
    )
    package = Path(factorlasso.__file__).resolve().parent
    for name in removed:
        assert not (package / f"{name}.py").exists(), name
        assert importlib.util.find_spec(f"factorlasso.{name}") is None, name
        assert not hasattr(factorlasso, name), name
