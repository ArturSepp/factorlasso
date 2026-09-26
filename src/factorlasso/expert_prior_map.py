"""Resolve economic descriptions into per-response OLS-prior factor labels.

This optional metadata helper does not estimate beta magnitudes. It produces
the ``factor_for_prior`` mapping consumed by :class:`factorlasso.LassoModel`;
unrecognised or mixed instruments retain automatic highest-R-squared selection.
It depends on factor labels, not on a consumer's factor-model class or CMA data.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Collection, Mapping
from dataclasses import dataclass

import pandas as pd


_BROAD_EQUITY_NAME = re.compile(
    r'^(?:eq )?(?:acwi|us|uk|europe|eu|japan|swiss|ac asia|asia|em|world|'
    r'north america|global equity)'
    r'(?: chf| xuk| xswiss| sli| msci| xjp| xasia| ex japan| ex asia| dm| hedged)*$'
)


@dataclass(frozen=True)
class ExpertPriorResolution:
    """Factor labels and provenance for one metadata-to-prior resolution.

    Attributes
    ----------
    selection : pd.Series
        Response-indexed scalar or ordered tuple of factor labels. Missing
        entries defer to the estimator's automatic factor selection.
    audit : pd.DataFrame
        Response-indexed name, readable selection, source, rule and optional
        caller-supplied policy version.
    """

    selection: pd.Series
    audit: pd.DataFrame


def _normalise_name(value: object) -> str:
    """Return a token-preserving, case-insensitive instrument description."""
    if not isinstance(value, str):
        return ''
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', value.casefold()).split())


def _normalise_ticker(value: object) -> str:
    """Canonicalise an instrument identifier for override matching."""
    return ' '.join(str(value).upper().split())


def _selection_from_name(name: str) -> tuple[str | tuple[str, str] | None, str]:
    """Recognise unambiguous fixed-income descriptions."""
    il = bool(re.search(r'\b(?:il bonds?|inflation linked bonds?|tips)\b', name))
    ig = bool(re.search(
        r'\b(?:ig (?:agg|corp|global|sbi|bonds?)|investment grade)\b', name))
    hy = bool(re.search(r'\b(?:hy (?:global|us|europe|bonds?|credit)|high yield)\b', name))
    em = bool(re.search(
        r'\b(?:em (?:hc )?bonds?|jpm em (?:latam )?corp|emerging markets? bonds?)\b',
        name))
    candidates = [
        (('Rates', 'Inflation'), 'inflation_linked', il),
        ('Credit IG', 'investment_grade', ig),
        ('Credit HY', 'high_yield', hy),
        ('Credit EM', 'em_bonds', em),
    ]
    matches = [(selection, rule) for selection, rule, matched in candidates if matched]
    credit_tokens = sum(bool(re.search(rf'\b{token}\b', name))
                        for token in ('ig', 'hy', 'em'))
    if len(matches) > 1 or (matches and credit_tokens > 1):
        return None, 'mixed_factor_name'
    return matches[0] if matches else (None, 'no_rule')


def map_expert_factor_priors(
        ticker_to_name: pd.Series,
        factor_names: Collection[str] | Mapping[str, object],
        *,
        asset_class: pd.Series | None = None,
        ticker_overrides: Mapping[str, str | list[str] | tuple[str, ...]] | None = None,
        policy_version: str | None = None,
) -> ExpertPriorResolution:
    """Select OLS-prior factors from instrument descriptions and reviewed overrides.

    Parameters
    ----------
    ticker_to_name : pd.Series
        Instrument names indexed by unique ticker or response identifier.
    factor_names : collection of str or mapping
        Available factor labels. A mapping contributes its keys, not values;
        pass ``x.columns`` or a consumer model's factor-name list directly.
    asset_class : pd.Series, optional
        Response-indexed broad asset classes. When supplied, name rules are
        gated to equity or fixed-income classes.
    ticker_overrides : mapping, optional
        Explicit response-to-factor selection, taking precedence over names.
        A scalar selects univariate OLS; an ordered list or tuple selects joint OLS.
    policy_version : str, optional
        Caller-owned policy identifier copied into the audit. It is distinct
        from the factorlasso package version.

    Returns
    -------
    ExpertPriorResolution
        A nullable ``selection`` suitable for ``LassoModel.factor_for_prior``
        and an auditable account of each decision. Unknown model factors warn
        and fall back to automatic selection.
    """
    if not isinstance(ticker_to_name, pd.Series):
        raise TypeError('ticker_to_name must be a pandas Series')
    if not ticker_to_name.index.is_unique:
        raise ValueError('expert-prior tickers must be unique')
    tickers = pd.Index([_normalise_ticker(ticker) for ticker in ticker_to_name.index])
    if not tickers.is_unique or any(ticker in ('', 'NAN', 'NONE') for ticker in tickers):
        raise ValueError('expert-prior tickers must be nonempty and unique after normalisation')
    if asset_class is not None:
        if not isinstance(asset_class, pd.Series):
            raise TypeError('asset_class must be a pandas Series')
        if (not asset_class.index.is_unique
                or not ticker_to_name.index.isin(asset_class.index).all()):
            raise ValueError('asset_class must uniquely cover every requested ticker')

    if isinstance(factor_names, str) or not isinstance(factor_names, Collection):
        raise TypeError('factor names must be a collection of strings or mapping keys')
    factors = list(factor_names)
    if (not factors or any(not isinstance(factor, str) or not factor.strip()
                           for factor in factors) or len(set(factors)) != len(factors)):
        raise ValueError('factor names must be nonempty, distinct string labels')
    known_factors = set(factors)

    if ticker_overrides is not None and not isinstance(ticker_overrides, Mapping):
        raise TypeError('ticker_overrides must be a mapping')
    overrides = {_normalise_ticker(ticker): selected
                 for ticker, selected in (ticker_overrides or {}).items()}
    if len(overrides) != len(ticker_overrides or {}):
        raise ValueError('expert-prior override tickers must be unique after normalisation')

    selection = pd.Series(None, index=ticker_to_name.index, name='factor_for_prior', dtype=object)
    audit_rows = []
    unavailable = {}
    for ticker, normalised_ticker in zip(ticker_to_name.index, tickers):
        name = _normalise_name(ticker_to_name.loc[ticker])
        selected = overrides.get(normalised_ticker)
        source = 'override' if selected is not None else 'automatic'
        rule = 'reviewed_override' if selected is not None else 'no_rule'
        if selected is None:
            category = '' if asset_class is None else _normalise_name(asset_class.loc[ticker])
            if ((category in ('equity', 'equities') or (not category and name.startswith('eq ')))
                    and _BROAD_EQUITY_NAME.fullmatch(name)):
                selected, rule, source = 'Equity', 'broad_equity_name', 'name'
            elif category in ('', 'bonds', 'bond', 'fixed income'):
                selected, rule = _selection_from_name(name)
                source = 'name' if selected is not None else (
                    'ambiguous' if rule == 'mixed_factor_name' else 'automatic')
        if isinstance(selected, (list, tuple)):
            if (not selected or any(not isinstance(factor, str) or not factor.strip()
                                    for factor in selected)
                    or len(set(selected)) != len(selected)):
                raise ValueError('expert-prior selections require distinct factor names')
            selected = tuple(selected)
        elif selected is not None and (not isinstance(selected, str) or not selected.strip()):
            raise ValueError('expert-prior selections require factor names')
        selected_factors = selected if isinstance(selected, tuple) else (selected,)
        absent = [factor for factor in selected_factors
                  if factor is not None and factor not in known_factors]
        if absent:
            unavailable[str(ticker)] = absent
            selected, source, rule = None, 'unavailable', 'factor_absent_from_model'
        selection.at[ticker] = selected
        audit_rows.append({
            'name': ticker_to_name.loc[ticker],
            'selection': ' + '.join(selected) if isinstance(selected, tuple) else selected,
            'source': source,
            'rule': rule,
            'policy_version': policy_version,
        })
    if unavailable:
        warnings.warn(
            f'expert prior factors absent from this model; retaining automatic '
            f'selection for {unavailable!r}', UserWarning, stacklevel=2)
    audit = pd.DataFrame(audit_rows, index=ticker_to_name.index)
    return ExpertPriorResolution(selection=selection, audit=audit)
