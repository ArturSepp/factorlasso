"""Execute every example script offline.

``examples/`` is documentation: each script must run top to bottom with the dependencies it
declares, without network access, and its own assertions must hold. Scripts under
``examples/docs/`` are the canonical worked examples included by the methodology articles, so a
failure here means an article no longer tells the truth.
"""

import runpy
import socket
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPOSITORY_ROOT / "examples"
EXAMPLE_SCRIPTS = sorted(EXAMPLES_ROOT.rglob("*.py")) if EXAMPLES_ROOT.is_dir() else []


def _refuse_network(*args, **kwargs):
    """Fail an example that opens a socket instead of using its own seeded inputs."""
    raise RuntimeError("examples must run offline; a network connection was attempted")


def test_example_scripts_are_discovered():
    """Guard against a silently empty parametrisation when run from a source checkout."""
    if not EXAMPLES_ROOT.is_dir():
        pytest.skip("examples/ is not shipped with this test layout")
    assert EXAMPLE_SCRIPTS, "examples/ exists but contains no Python scripts"


@pytest.mark.parametrize(
    "script",
    EXAMPLE_SCRIPTS,
    ids=[path.relative_to(EXAMPLES_ROOT).as_posix() for path in EXAMPLE_SCRIPTS],
)
def test_example_runs_offline(script, monkeypatch, tmp_path):
    """Run one script as ``__main__`` in a scratch directory with sockets disabled."""
    if "matplotlib" in script.read_text(encoding="utf-8"):
        pytest.importorskip("matplotlib")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(socket, "create_connection", _refuse_network)
    monkeypatch.setattr(socket.socket, "connect", _refuse_network)

    runpy.run_path(str(script), run_name="__main__")

    leftovers = sorted(path.name for path in tmp_path.iterdir())
    assert not leftovers, f"example wrote files into its working directory: {leftovers}"
