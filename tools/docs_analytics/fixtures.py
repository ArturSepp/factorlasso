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

import numpy as np

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


def verify_parameters(example: dict, parameters: dict) -> None:
    """Fail when the registry describes a calculation other than the one the example runs.

    Parameters
    ----------
    example : dict
        Namespace returned by :func:`load_example`.
    parameters : dict
        The exhibit's ``parameters`` record from ``registry.json``. Each key is the lower-case name
        of a module constant of the example, for instance ``"reg_lambda"`` for ``REG_LAMBDA``.

    Raises
    ------
    ValueError
        If a declared value differs from the example's constant.
    """
    applied = {}
    for key in parameters:
        value = example[key.upper()]
        applied[key] = value.tolist() if isinstance(value, np.ndarray) else value
    if applied != parameters:
        raise ValueError(f"Registry parameters {parameters} differ from the example's {applied}")


def style_axis(axis) -> None:
    """Recessive frame and grid; text in ink, never in a series colour."""
    axis.set_facecolor(SURFACE)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=INK_MUTED, labelsize=10.5)
    axis.xaxis.label.set_color(INK)
    axis.xaxis.label.set_size(11.5)
    axis.yaxis.label.set_size(11.5)
    axis.yaxis.label.set_color(INK)
    axis.title.set_color(INK)


def loading_colormap():
    """Diverging map for loadings: second series colour below zero, surface at zero, first above."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("loadings", [SERIES[1], SURFACE, SERIES[0]])
