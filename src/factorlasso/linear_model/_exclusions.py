"""Labelled hard-zero restrictions, independent of sign detection and prior selection."""
from collections.abc import Mapping

import numpy as np
import pandas as pd

from factorlasso.linear_model._types import _mode_spec


def selected_exclusions(value) -> tuple:
    """Parse one response's unique factor labels without modifying caller data."""
    if pd.api.types.is_scalar(value):
        return () if pd.isna(value) else (value,)
    if not isinstance(value, (list, tuple)):
        raise ValueError('excluded_factors requires a label or list/tuple of labels')
    if any(not pd.api.types.is_scalar(v) or pd.isna(v) for v in value):
        raise ValueError('excluded_factors requires nonmissing scalar labels')
    if pd.Index(value).has_duplicates:
        raise ValueError('excluded_factors must contain unique factor labels')
    return tuple(value)


def validate_exclusion_settings(model) -> None:
    """Validate reusable declarations before any observations are supplied."""
    declarations = model.excluded_factors
    if declarations is None:
        return
    if not isinstance(declarations, (Mapping, pd.Series)):
        raise TypeError('excluded_factors must be a mapping or pandas Series')
    if isinstance(declarations, pd.Series) and not declarations.index.is_unique:
        raise ValueError('excluded_factors response labels must be unique')
    for value in declarations.values() if isinstance(declarations, Mapping) else declarations:
        if selected_exclusions(value) and not _mode_spec(model.model_type).hard_constraints:
            raise ValueError('excluded_factors requires a solver supporting hard constraints')


def exclusion_mask(model, x: pd.DataFrame, y: pd.DataFrame):
    """Resolve current responses; permit extra response rows for rolling/cadence subsets."""
    validate_exclusion_settings(model)
    if model.excluded_factors is None:
        return None
    mask = pd.DataFrame(False, index=y.columns, columns=x.columns)
    for response, value in model.excluded_factors.items():
        factors = selected_exclusions(value)
        missing = [factor for factor in factors if factor not in x.columns]
        if missing:
            raise ValueError(f'excluded_factors[{response!r}] names unknown factors {missing!r}')
        if response not in y.columns or not factors:
            continue
        if model.factor_for_prior is not None:
            prior = selected_exclusions(model.factor_for_prior.get(response))
            conflicts = [factor for factor in factors if factor in prior]
            if conflicts:
                raise ValueError(
                    f'{response!r}: excluded factors also named as prior: {conflicts!r}')
        if model.factors_beta_prior is not None:
            values = model.factors_beta_prior.loc[response, list(factors)]
            if (values.notna() & values.ne(0)).any():
                raise ValueError(f'{response!r}: excluded factors have a nonzero explicit prior')
        mask.loc[response, list(factors)] = True
    return mask


def overlay_exclusions(explicit_signs, mask):
    """Hard exclusions take precedence over both automatic and explicit sign directions."""
    if mask is None or not mask.to_numpy().any():
        return explicit_signs
    signs = (np.full(mask.shape, np.nan) if explicit_signs is None
             else explicit_signs.copy())
    signs[mask.to_numpy()] = 0.0
    return signs
