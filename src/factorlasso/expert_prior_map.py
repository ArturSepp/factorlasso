"""Historical location of the expert prior mapping recipe.

Compatibility module of the 0.23.0 layout. Every name is re-exported from its canonical
owner; new code imports from :mod:`factorlasso` or :mod:`factorlasso.priors`. Assigning a
name here raises :class:`AttributeError` (see :mod:`factorlasso._compat`).
"""

# ruff: noqa: F401

from factorlasso._compat import guard_legacy_module
from factorlasso.priors._expert_map import (
    _BROAD_EQUITY_NAME, _BROAD_MSCI_MARKETS, _commodity_selection, ExpertPriorResolution,
    _is_equity_index_name, _is_listed_real_estate_name, map_expert_factor_priors,
    _MSCI_RETURN_MARKERS, _NON_EQUITY_NAME, _normalise_name, _normalise_ticker,
    _selection_from_name,
)

guard_legacy_module(__name__)
