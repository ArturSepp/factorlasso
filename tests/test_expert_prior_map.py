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


@pytest.mark.parametrize(('name', 'asset_class', 'expected'), [
    ('MSCI ACWI Net Total Return USD Index', 'Equities', 'Equity'),
    ('MSCI UK Net Total Return Local Index', 'Equity', 'Equity'),
    ('MSCI Europe Ex UK ex Switzerland Net EUR Index', 'Equity', 'Equity'),
    ('MSCI Emerging Markets India Net Total Return Local Index', 'Equity', 'Equity'),
    ('Bloomberg United Kingdom Large & Mid Cap Net Return Index Hedged CHF',
     'Equity', 'Equity'),
    ('SLI SWISS LEADER PERFORM', 'Equity', 'Equity'),
    ('Bloomberg Global-Aggregate Total Return Index Value Hedged EUR',
     'Bonds', 'Credit IG'),
    ('Bloomberg Global Aggregate Corporate Total Return Index Hedged EUR',
     'Fixed Income', 'Credit IG'),
    ('Bloomberg Euro Corporate Unh EUR', 'Bonds', 'Credit IG'),
    ('Bloomberg Global High Yield Corporate Total Return Index Hedged EUR',
     'Bonds', 'Credit HY'),
    ('Bloomberg EM Hard Currency Aggregate Total Return Index Hedged EUR',
     'Bonds', 'Credit EM'),
    ('J.P. Morgan EMBI Global Diversified Hedged EUR', 'Bonds', 'Credit EM'),
    ('J.P. Morgan CEMBI Broad Diversified Composite Index Hedged EUR',
     'Bonds', 'Credit EM'),
    ('Bloomberg Global Inflation-Linked 1-10yrs Total Return Index Hedged EUR',
     'Bonds', ('Rates', 'Inflation')),
    ('Bloomberg US Treasury Inflation Notes TR Index Value Unhedged USD',
     'Bonds', ('Rates', 'Inflation')),
])
def test_full_provider_index_names_select_economic_factors(name, asset_class, expected):
    """Full reference names select the same economic priors as short labels."""
    result = map_expert_factor_priors(
        pd.Series({'asset': name}), FACTORS,
        asset_class=pd.Series({'asset': asset_class}),
    )
    assert result.selection['asset'] == expected
    assert result.audit.loc['asset', 'source'] == 'name'


def test_full_names_do_not_force_mixed_credit_or_non_equity_indices():
    """Economic class gates and mixed-credit rules remain conservative."""
    names = pd.Series({
        'BOND': 'MSCI World Bond Index',
        'WRONG_CLASS': 'MSCI World Bond Index',
        'GOVT': 'Bloomberg Series-E Euro Govt 5-10 Yr Bond Index',
        'GOVT_AGG': 'Bloomberg Global Aggregate Treasuries Total Return Index Hedged EUR',
        'MIXED': 'Emerging Markets High Yield Corporate Bond Index',
        'ALT': 'HFRX Global Hedge Fund EUR Index',
        'STYLE': 'MSCI World Momentum Net Total Return USD Index',
        'SECTOR': 'MSCI EMU Banks Net Return EUR Index',
        'SECTOR_DAILY': 'MSCI Daily TR EMU Net Chemicals Local',
    })
    classes = pd.Series({
        'BOND': 'Fixed Income', 'WRONG_CLASS': 'Equity',
        'GOVT': 'Fixed Income', 'GOVT_AGG': 'Bonds',
        'MIXED': 'Bonds', 'ALT': 'Alternatives',
        'STYLE': 'Equity', 'SECTOR': 'Equity', 'SECTOR_DAILY': 'Equity',
    })
    result = map_expert_factor_priors(names, FACTORS, asset_class=classes)
    assert result.selection.loc[['GOVT', 'GOVT_AGG']].eq('Rates').all()
    assert result.selection.drop(['GOVT', 'GOVT_AGG']).isna().all()
    assert result.audit.loc['MIXED', 'source'] == 'ambiguous'


@pytest.mark.parametrize(('name', 'category', 'expected'), [
    ('Global IG corporates, USD hedged', 'Fixed Income', 'Credit IG'),
    ('Global IG corporates 1-3 years, USD hedged', 'Bonds', 'Credit IG'),
    ('Global HY CoCos, USD hedged', 'Fixed Income', 'Credit HY'),
    ('Global IG CoCos, USD hedged', 'Hybrids', 'Credit IG'),
    ('EM hard-currency aggregate, USD hedged', 'Fixed Income', 'Credit EM'),
    ('Global government bonds, USD unhedged', 'Bonds', 'Rates'),
    ('US municipal bonds', 'Fixed Income', 'Rates'),
    ('US securitized bonds: MBS/ABS/CMBS/covered', 'Fixed Income', 'Rates'),
    ('Global convertibles, USD hedged', 'Hybrids', ('Rates', 'Equity')),
    ('Global convertible bonds, USD hedged', 'Fixed Income', ('Rates', 'Equity')),
    ('US preferred and hybrid securities (PFF)', 'Fixed Income', ('Rates', 'Equity')),
    ('Preferred stock index', 'Hybrids', ('Rates', 'Equity')),
    ('MSCI Emerging Markets net total return, USD', 'Equity', 'Equity'),
    ('MSCI World Real Estate net total return, USD', 'Equity', ('Rates', 'Equity')),
    ('Global listed REIT index', 'Equities', ('Rates', 'Equity')),
    ('SPDR Gold Shares (GLD)', 'Commodities', 'Gold'),
    ('United States Oil Fund (USO)', 'Commodities', 'Oil'),
    ('Bloomberg Gold Subindex Total Return', 'Commodity', 'Gold'),
    ('Bloomberg WTI Crude Oil Subindex Total Return', 'Commodity', 'Oil'),
])
def test_bond_hybrid_equity_and_commodity_metadata_coverage(name, category, expected):
    """Economic mandates resolve without per-security overrides or return data."""
    result = map_expert_factor_priors(
        pd.Series({'asset': name}), FACTORS + ['Gold', 'Oil', 'Commodities'],
        asset_class=pd.Series({'asset': category}),
    )
    assert result.selection['asset'] == expected
    assert result.audit.loc['asset', 'source'] == 'name'


@pytest.mark.parametrize(('name', 'category'), [
    ('EM local-currency government IG, USD unhedged', 'Fixed Income'),
    ('Emerging Markets Local Currency Government Index', 'Bonds'),
    ('Global HY and IG CoCos', 'Hybrids'),
    ('Global contingent convertible bonds', 'Fixed Income'),
    ('US floating-rate preferred securities', 'Hybrids'),
    ('Government and corporate bonds', 'Fixed Income'),
    ('US Treasury floating rate notes', 'Fixed Income'),
    ('Global convertible arbitrage', 'Hybrids'),
    ('High Yield Municipal Bonds', 'Bonds'),
    ('MSCI World Real Estate Debt Index', 'Equity'),
    ('Private Real Estate Fund', 'Alternatives'),
    ('Gold Mining Equity Fund', 'Commodities'),
    ('Oil and Gas Producers Fund', 'Commodities'),
    ('Gold and Silver Fund', 'Commodity'),
    ('Gold and Oil Basket', 'Commodity'),
    ('SPDR Gold Shares', 'Equity'),
    ('Invesco DB Agriculture Fund', 'Commodities'),
    ('Invesco DB Base Metals Fund', 'Commodities'),
])
def test_new_rules_leave_conflicting_or_unsupported_exposures_automatic(name, category):
    """Do not turn name coverage into false economic certainty."""
    result = map_expert_factor_priors(
        pd.Series({'asset': name}), FACTORS + ['Gold', 'Oil', 'Commodities'],
        asset_class=pd.Series({'asset': category}),
    )
    assert pd.isna(result.selection['asset'])


def test_joint_metadata_rule_requires_every_factor_and_override_still_wins():
    """An incomplete joint target falls back as a whole; explicit review wins."""
    names = pd.Series({'asset': 'Global convertible bonds'})
    classes = pd.Series({'asset': 'Hybrids'})
    with pytest.warns(UserWarning, match='absent from this model'):
        result = map_expert_factor_priors(names, ['Equity'], asset_class=classes)
    assert pd.isna(result.selection['asset'])
    assert result.audit.loc['asset', 'source'] == 'unavailable'
    result = map_expert_factor_priors(
        names, FACTORS, asset_class=classes, ticker_overrides={'ASSET': 'Credit HY'})
    assert result.selection['asset'] == 'Credit HY'
    assert result.audit.loc['asset', 'source'] == 'override'
