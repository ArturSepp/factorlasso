"""Shared inputs for the synthetic documentation exhibits.

The worked example of an article is the fixture of its exhibit. A producer loads the canonical
script under ``examples/docs/`` and calls the same functions the article includes, so a figure, its
supporting table and the numbers quoted in the prose come from one calculation. A second
implementation written only to draw a chart is what this avoids.

A producer names the scripts it loads under ``fixture_files`` in ``registry.json``; their hashes
enter the manifest. Generators that no example owns may be added here in later stages, each with
a fixed seed that is never changed once an exhibit depends on it.
"""

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Validated on the light chart surface with adjacent-pair colour-vision-deficiency separation.
# Identity is never carried by colour alone: producers also vary the marker and label directly.
SURFACE = "#fcfcfb"
INK = "#1f1f1e"
INK_MUTED = "#52514e"
GRID = "#e4e3dd"
SERIES = ("#2a78d6", "#eb6834")


def load_example(basename: str) -> dict:
    """Return the namespace of ``examples/docs/<basename>.py`` without running its ``main``.

    Parameters
    ----------
    basename : str
        Article basename, for example ``"residual_diagnostics"``.

    Returns
    -------
    dict
        Module globals: the example's constants and functions.
    """
    path = ROOT / "examples" / "docs" / f"{basename}.py"
    if not path.is_file():
        raise ValueError(f"No canonical example for {basename!r}: {path}")
    return runpy.run_path(str(path), run_name=f"docs_example_{basename}")
