"""Portable metadata-to-OLS-prior selections for factor models."""

import pandas as pd
import pytest

from factorlasso.expert_prior_map import map_expert_factor_priors


FACTORS = ['Equity', 'Rates', 'Inflation', 'Credit IG', 'Credit HY', 'Credit EM']


def test_names_select_factors_without_a_consumer_model_object():
    """Only factor labels and descriptive asset metadata are needed."""
    names = pd.Series({
        'A': 'EQ ACWI', 'B': 'IG Corp Global EUR',
        'C': 'IL Bonds Global EUR', 'D': 'EQ Value',
    })
    classes = pd.Series({'A': 'Equities', 'B': 'Bonds',
                         'C': 'Fixed Income', 'D': 'Equities'})
    result = map_expert_factor_priors(names, FACTORS, asset_class=classes)
    assert result.selection['A'] == 'Equity'
    assert result.selection['B'] == 'Credit IG'
    assert result.selection['C'] == ('Rates', 'Inflation')
    assert pd.isna(result.selection['D'])
    assert result.audit.loc['A', 'source'] == 'name'


def test_mapping_keys_can_supply_factor_names_and_override_wins():
    """A factor-name dictionary uses keys; explicit ticker judgment wins."""
    names = pd.Series({'i13913us index': 'Structured Credit'})
    result = map_expert_factor_priors(
        names, dict.fromkeys(FACTORS),
        ticker_overrides={'I13913US INDEX': 'Credit HY'},
        policy_version='client-v1',
    )
    assert result.selection['i13913us index'] == 'Credit HY'
    assert result.audit.loc['i13913us index', 'source'] == 'override'
    assert result.audit.loc['i13913us index', 'policy_version'] == 'client-v1'


def test_ordered_list_override_matches_estimator_selection_contract():
    """An ordered factor list is accepted and normalised for joint OLS."""
    result = map_expert_factor_priors(
        pd.Series({'LINKER': 'IL Bonds Global EUR'}), FACTORS,
        ticker_overrides={'LINKER': ['Rates', 'Inflation']},
    )
    assert result.selection['LINKER'] == ('Rates', 'Inflation')
    assert result.audit.loc['LINKER', 'selection'] == 'Rates + Inflation'


def test_missing_factor_falls_back_to_automatic_selection():
    """Do not return labels that the estimation factor panel lacks."""
    with pytest.warns(UserWarning, match='absent from this model'):
        result = map_expert_factor_priors(
            pd.Series({'HY': 'HY Global EUR'}), ['Equity', 'Rates'])
    assert pd.isna(result.selection['HY'])
    assert result.audit.loc['HY', 'source'] == 'unavailable'


def test_mixed_names_and_nonbond_classes_do_not_force_credit():
    """Ambiguity and class conflicts leave FactorLasso's selector in control."""
    names = pd.Series({'MIX': 'IG and HY Bonds', 'STYLE': 'High Yield Equity Strategy'})
    classes = pd.Series({'MIX': 'Bonds', 'STYLE': 'Equity'})
    result = map_expert_factor_priors(names, FACTORS, asset_class=classes)
    assert result.selection.isna().all()
    assert result.audit.loc['MIX', 'source'] == 'ambiguous'


def test_invalid_factor_names_and_colliding_override_tickers_rejected():
    """A malformed factor universe or override map must not silently select."""
    names = pd.Series({'A': 'EQ ACWI'})
    with pytest.raises(ValueError, match='factor names'):
        map_expert_factor_priors(names, ['Equity', 'Equity'])
    with pytest.raises(ValueError, match='override tickers'):
        map_expert_factor_priors(
            names, FACTORS, ticker_overrides={'A': 'Equity', 'a': 'Equity'})
