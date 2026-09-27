"""Shared test configuration.

Figures are drawn with Matplotlib's non-interactive Agg backend, as CI does through
``MPLBACKEND=Agg``. A local run otherwise picks the platform default, TkAgg on Windows, whose GUI
state made figure-producing tests such as the cluster lineage report fail intermittently. The
backend is chosen here, before any test module imports ``matplotlib.pyplot``.
"""

try:
    import matplotlib
except ImportError:  # the core-install job runs without Matplotlib
    matplotlib = None

if matplotlib is not None:
    matplotlib.use("Agg")
