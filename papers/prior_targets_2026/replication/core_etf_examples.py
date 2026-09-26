"""Reconstruct ten illustrative production ETF estimates from frozen FI evidence."""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .build_paper import markdown_table

BASE = Path(r'C:\Users\artur\AppData\Local\AgentWork\ARTURDESKTOP\Rosaa\analyses')
FUNDS = BASE / 'fi_prior_funds_20260922_v2'
COMPARE = BASE / 'fi_prior_span36_comparison_20260922_v1'
# A coverage-based illustration chosen after the study, not a prespecified cohort.
ROSTER = [
    ('IEI US Equity', 'IEI', 'US Treasury, 3-7 years'),
    ('TLT US Equity', 'TLT', 'US Treasury, 20+ years'),
    ('LQDE LN Equity', 'LQDE', 'USD investment grade'),
    ('SDIG LN Equity', 'SDIG', 'USD short-duration investment grade'),
    ('IEAC LN Equity', 'IEAC', 'EUR investment grade'),
    ('IHYU LN Equity', 'IHYU', 'USD high yield'),
    ('IHYG LN Equity', 'IHYG', 'EUR high yield'),
    ('IEMB LN Equity', 'IEMB', 'Emerging-market hard-currency debt'),
    ('EMDD LN Equity', 'EMDD', 'Emerging-market local-currency debt'),
    ('CWB US Equity', 'CWB', 'Convertible bonds'),
]
FACTORS = ['Equity', 'Rates', 'Credit IG US', 'Credit HY US', 'Credit EM',
           'Carry G10', 'Carry EM', 'Inflation', 'Commodities', 'Private Equity',
           'Rates Vol', 'Fx']


def evidence_blocks():
    """Recompute scores and check endpoint betas before generating Table 9."""
    hashes = {}
    for folder, names in [
        (FUNDS, ['june2026_betas.csv', 'june2026_fits.csv', 'june2026_targets.csv',
                 'june2026_endpoint_summary.json', 'predictions_all_lambda.csv',
                 'verified_fund_outcomes.csv']),
        (COMPARE, ['endpoint_beta_comparison.csv']),
    ]:
        manifest = json.loads((folder / 'evidence_manifest.json').read_text())
        for name in names:
            path = folder / name
            actual = sha256(path.read_bytes()).hexdigest()
            assert actual == manifest[name], f'Source hash mismatch: {path}'
            hashes[str(path)] = actual
    summary = json.loads((FUNDS / 'june2026_endpoint_summary.json').read_text())
    assert summary['date'] == '2026-06-30' and summary['matched_clusters_signs']
    tickers = [r[0] for r in ROSTER]
    beta = pd.read_csv(FUNDS / 'june2026_betas.csv')
    beta = beta.loc[beta.ticker.isin(tickers)]
    assert len(beta) == 240 and not beta.duplicated(['arm', 'ticker', 'factor']).any()
    for _, group in beta.groupby(['arm', 'ticker']):
        assert set(group.factor) == set(FACTORS)
    fits = pd.read_csv(FUNDS / 'june2026_fits.csv').set_index(['ticker', 'arm'])
    fits = fits.loc[pd.IndexSlice[tickers, :], :]
    assert len(fits) == 20 and fits.kind.eq('Passive fund').all()
    betas = beta.set_index(['ticker', 'arm', 'factor']).beta.sort_index()
    targets = pd.read_csv(FUNDS / 'june2026_targets.csv').set_index('ticker')
    endpoint = pd.read_csv(COMPARE / 'endpoint_beta_comparison.csv')
    endpoint = endpoint.loc[endpoint.lane.eq('Fund') & endpoint.ticker.isin(tickers)]
    assert len(endpoint) == 120
    endpoint = endpoint.set_index(['ticker', 'factor'])
    endpoint_error = 0.0
    for arm in ['P0', 'P1']:
        left = beta.loc[beta.arm.eq(arm)].set_index(['ticker', 'factor']).beta
        right = endpoint[f'beta60_{arm}'].reindex(left.index)
        assert right.notna().all()
        np.testing.assert_allclose(left, right, rtol=0, atol=1e-13)
        endpoint_error = max(endpoint_error, float((left - right).abs().max()))

    # Independent scores use actual returns and predictions, not the saved errors.
    pred = pd.read_csv(FUNDS / 'predictions_all_lambda.csv')
    pred = pred.loc[pred.ticker.isin(tickers) &
                    np.isclose(pred.reg_lambda, 1e-5, rtol=0, atol=1e-14)].copy()
    assert len(pred) == 1680 and set(pred.arm) == {'P0', 'P1'}
    assert not pred.duplicated(['ticker', 'arm', 'month']).any()
    pred['month'] = pd.to_datetime(pred.month)
    pred['fit_date'] = pd.to_datetime(pred.fit_date)
    assert (pred.fit_date < pred.month).all()
    quarter_gap = pred.month.dt.to_period('Q').astype('int64') - pred.fit_date.dt.to_period('Q').astype('int64')
    assert quarter_gap.eq(1).all()
    assert pred.fit_date.dt.is_quarter_end.all()
    wanted_months = pd.period_range('2019-07', '2026-06', freq='M')
    for _, group in pred.groupby(['ticker', 'arm']):
        assert set(group.month.dt.to_period('M')) == set(wanted_months)
        assert len(group) == 84 and group.fit_date.nunique() == 28
        assert group.groupby('fit_date').size().eq(3).all()
    paired = pred.pivot(index=['ticker', 'month'], columns='arm', values=['actual', 'benchmark', 'fit_date'])
    for name in ['actual', 'benchmark', 'fit_date']:
        pd.testing.assert_series_equal(paired[name]['P0'], paired[name]['P1'], check_names=False)
    pred['sse_check'] = (pred.actual - pred.prediction) ** 2
    pred['baseline_check'] = (pred.actual - pred.benchmark) ** 2
    np.testing.assert_allclose(pred.sse_check, pred.squared_error, rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(pred.baseline_check, pred.baseline_squared_error, rtol=1e-11, atol=1e-15)
    errors = pred.groupby(['ticker', 'arm'])[['sse_check', 'baseline_check']].sum()
    assert errors.baseline_check.gt(0).all()
    scores = (1 - errors.sse_check / errors.baseline_check).unstack('arm').loc[tickers]
    saved = pd.read_csv(FUNDS / 'verified_fund_outcomes.csv').set_index('ticker').loc[tickers]
    np.testing.assert_allclose(scores[['P0', 'P1']], saved[['P0', 'P1']], rtol=0, atol=1e-12)
    score_error = float(np.abs(scores[['P0', 'P1']] - saved[['P0', 'P1']]).to_numpy().max())
    gain = scores.P1 - scores.P0
    sse = errors.sse_check.unstack('arm').loc[tickers]
    rmse_reduction = 100 * (1 - np.sqrt(sse.P1 / sse.P0))

    rows, records = [], []
    for ticker, label, segment in ROSTER:
        for arm in ['P0', 'P1']:
            b = betas.loc[(ticker, arm)].reindex(FACTORS)
            rows.append([label, arm, *[f'{v:.3f}' for v in b.iloc[:5]],
                         f'{fits.loc[(ticker, arm), "fitted_r2"]:.4f}',
                         f'{scores.loc[ticker, arm]:.4f}'])
            records.append(dict(ticker=ticker, label=label, segment=segment, arm=arm,
                endpoint_date=summary['date'], name=fits.loc[(ticker, arm), 'name'],
                n_training_months=int(fits.loc[(ticker, arm), 'n_months']),
                span=60, reg_lambda=1e-5, **{f'beta_{k}': float(v) for k, v in b.items()},
                fitted_r2=float(fits.loc[(ticker, arm), 'fitted_r2']),
                oos_r2=float(scores.loc[ticker, arm]), n_oos_months=84,
                automatic_winner=targets.loc[ticker, 'winner'] if arm == 'P1' else '',
                automatic_target=float(targets.loc[ticker, 'target']) if arm == 'P1' else 0.0))
    blocks = {'core_etf_estimates': markdown_table(
        ['ETF', 'Arm', 'Equity', 'Rates', 'IG', 'HY', 'EM', 'Fitted $R^2$', 'OOS $R^2$'], rows)}
    fit_mean = fits.fitted_r2.groupby('arm').mean()
    blocks['core_etf_results'] = (
        f'Within these ten examples, automatic centring improves pooled conditional out-of-sample $R^2$ for {int(gain.gt(0).sum())} of 10 ETFs. '
        f'The equal-ETF mean rises from {scores.P0.mean():.4f} to {scores.P1.mean():.4f}, '
        f'a gain of {gain.mean():.4f} $R^2$ points ({100 * gain.mean():.2f} percentage points). '
        f'The corresponding mean endpoint fitted $R^2$ rises from {fit_mean["P0"]:.4f} to {fit_mean["P1"]:.4f}. '
        f'For LQDE and IEMB, the held-out RMSE reductions are {rmse_reduction["LQDE LN Equity"]:.1f}% and {rmse_reduction["IEMB LN Equity"]:.1f}%, respectively. '
        f'EMDD is the adverse example: its out-of-sample $R^2$ falls by {abs(gain["EMDD LN Equity"]):.4f} points '
        f'and its RMSE increases by {abs(rmse_reduction["EMDD LN Equity"]):.1f}%. '
        'IEAC improves out of sample despite a lower endpoint fitted score. '
        'The ten-ETF mean is descriptive of this illustrative subset; Table 4 supplies the broader 16-passive-fund comparison.')
    audit = dict(status='passed', endpoint=summary['date'], n_etfs=10, n_fits=20,
        n_factor_cells=240, n_prediction_rows=len(pred), n_paired_asset_months=840,
        oos_start='2019-07', oos_end='2026-06', n_quarters=28,
        endpoint_reference_max_abs_error=endpoint_error, oos_reference_max_abs_error=score_error,
        mean_oos_r2_P0=float(scores.P0.mean()), mean_oos_r2_P1=float(scores.P1.mean()),
        mean_gain=float(gain.mean()), n_oos_improved=int(gain.gt(0).sum()),
        rmse_reduction_pct=rmse_reduction.to_dict(), source_hashes=hashes,
        selection='Illustrative coverage-based subset, chosen after the study; not prespecified.',
        records=records)
    return blocks, audit


def validate_source(source, blocks):
    """Reject any changed number or wording in the generated example blocks."""
    for key, expected in blocks.items():
        actual = re.findall(r'<!-- evidence: ' + key + r' -->\n(.*?)\n<!-- /evidence -->', source, re.S)
        assert actual == [expected], f'Core ETF evidence drift: {key}'
