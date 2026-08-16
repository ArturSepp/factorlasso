"""Tests for the repository's ``src`` package layout."""

from pathlib import Path

import pytest

import factorlasso


def test_checkout_declares_and_uses_src_layout():
    """A checkout must discover the import package exclusively below ``src``."""
    root = Path(__file__).resolve().parents[1]
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("installed tests have no repository metadata")

    package_init = root / "src" / "factorlasso" / "__init__.py"
    assert package_init.is_file()
    assert not (root / "factorlasso").exists()

    discovery = pyproject.read_text(encoding="utf-8").split(
        "[tool.setuptools.packages.find]", maxsplit=1
    )[1].split("\n[", maxsplit=1)[0]
    assert 'where = ["src"]' in discovery

    imported = Path(factorlasso.__file__).resolve()
    if root in imported.parents:
        assert imported == package_init.resolve()
