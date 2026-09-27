"""Read-only inputs and external output locations for the CSDA replication."""

import os
from pathlib import Path
import tempfile

PAPER_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = PAPER_DIR.parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"
REFERENCE_DIR = DATA_DIR / "reference"


def output_directory() -> Path:
    """Select a runtime destination outside the checkout and OneDrive."""
    override = os.environ.get("FACTORLASSO_PAPER_OUTPUT_DIR")
    if override:
        result = Path(override).expanduser()
        if not result.is_absolute():
            raise ValueError("FACTORLASSO_PAPER_OUTPUT_DIR must be an absolute path")
    elif os.environ.get("AGENT_LOCAL_ROOT"):
        result = Path(os.environ["AGENT_LOCAL_ROOT"]) / "outputs" / "sign_pooling_2026"
    else:
        base = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
        result = base / "FactorLasso" / "sign_pooling_2026"
    result = result.resolve()
    if result == REPOSITORY_DIR or REPOSITORY_DIR in result.parents:
        raise ValueError("Paper outputs must be outside the source checkout")
    if any(part.casefold().startswith("onedrive") for part in result.parts):
        raise ValueError("Paper outputs must be outside OneDrive")
    return result


OUTPUT_DIR = output_directory()
RESULTS_DIR = OUTPUT_DIR / "results"
EXHIBITS_DIR = OUTPUT_DIR / "paper"
