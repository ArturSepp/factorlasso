"""Controlled frozen-publication replay of the revised sign procedure.

Uses the existing FactorLasso research legacy adapter and production CMA APIs.
All economic assumptions, priors, return panels and factor covariance are frozen.
Run with Rosaa's external environment and FactorLasso on PYTHONPATH.
"""
from pathlib import Path
import json
import os
import warnings

import numpy as np
import pandas as pd

from papers.prior_targets_2026.replication import fi_validation as f
from papers.prior_targets_2026.replication.sign_revision import old_inputs
from optimalportfolios.covar_estimation.factor_covar_estimator import (
    estimate_lasso_factor_covar_data,
)
from ramen.quant_model_mvp.rosaa.data.factors.beta_priors import build_factor_for_prior
from ramen.quant_model_mvp.rosaa.data.cmas.assets.global_saa import _resolve_factors_beta_prior


def main():
    """Fit both native cadences, verify the old endpoint and export paired impacts."""
    out = Path(os.environ['AGENT_LOCAL_ROOT']) / 'analyses/sign_revision_cma_impact_20260925'
    out.mkdir(parents=True, exist_ok=True)
    source = Path('C:/Users/artur/AppData/Local/AgentWork/ARTURDESKTOP/FactorLasso/analyses/fi_sign_validation_20260925')
    assets = f.final_assets()
    bundle = f.factor_bundle()
    economic = bundle.select(f.MATF_CUSTOM_IG_HY).for_asset_model(f.MATF_CUSTOM_IG_HY)
    meta = pd.read_csv(source / 'inputs/full_metadata.csv', index_col=0)
    names = meta.index[~meta.universe.eq('core_liquidity')]
    cash = meta.index[meta.universe.eq('core_liquidity')]
    panels = {}
    for freq in ('ME', 'QE'):
        cols = names[meta.loc[names, 'Rebalancing'].eq(freq)]
        panel = assets.excess_logreturns.loc[:f.CUT, cols]
        if freq == 'QE':
            # These are already quarterly returns: select quarter dates, never sum.
            panel = panel.loc[panel.index.is_quarter_end]
        panels[freq] = panel
        print('INPUT', freq, panel.shape, panel.index[0], panel.index[-1], flush=True)
    signs = f.compute_beta_loading_signs_for_matf(
        meta.loc[names, 'LongOnlyBetas'], meta.loc[names, 'PEfactorExposure'],
        f.MATF_CUSTOM_IG_HY,
    )
    priors = _resolve_factors_beta_prior(meta, f.MATF_CUSTOM_IG_HY, use_beta_priors=False, pe_beta_prior=1.)
    mapping = build_factor_for_prior(meta, f.MATF_CUSTOM_IG_HY)
    spec = f.get_cma_covar_estimation_spec()
    clean_meta = meta.astype(object).where(meta.notna(), None)
    original_snap = assets.covar_data.get_snapshot(
        alpha_span=spec.alpha_span, asset_frequencies=meta.Rebalancing)
    # Publication reporting suppresses native USD cash alpha after covariance
    # construction. Preserve its published display as well as its zero admission.
    original_snap.loc[cash, 'stat_alpha'] = assets.cma_metadata.loc[cash, 'stat_alpha']
    receipt = {'cutoff': str(f.CUT), 'solver': spec.solver,
               'scope': {'monthly': len(panels['ME'].columns),
                         'quarterly': len(panels['QE'].columns), 'cash_unchanged': len(cash)},
               'held_fixed': ['factor prices and premia', 'factor covariance', 'asset excess log returns',
                              'prior mappings and targets', 'hard sign constraints', 'penalty',
                              'alpha admission and regional settings'], 'warnings': {}}
    results = {}
    covars = {}
    for label in ('old', 'revised'):
        model = f.get_covar_estimator(spec, factors_beta_loading_signs=signs,
                                     factors_beta_prior=priors, factor_for_prior=mapping).lasso_model
        model.auto_sign_use_fit_span = label == 'revised'
        model.auto_sign_variance = 'date' if label == 'revised' else 'independent'
        with old_inputs(label == 'old'), warnings.catch_warnings(record=True) as caught:
            covar = estimate_lasso_factor_covar_data(
                risk_factor_prices=bundle.factor_prices.loc[:f.CUT, f.FACTORS],
                asset_returns_dict=panels, lasso_model=model, assets=names,
                x_covar=assets.covar_data.x_covar, estimation_date=f.CUT,
                factor_returns_freq=spec.factor_returns_freq,
                factor_covar_span=spec.factor_covar_span,
            )
        receipt['warnings'][label] = [str(w.message) for w in caught]
        snap = covar.get_snapshot(alpha_span=spec.alpha_span, asset_frequencies=meta.Rebalancing)
        snap = pd.concat([snap, original_snap.loc[cash]]).reindex(meta.index)
        cmas, _ = economic.estimate_asset_universe_cma(clean_meta, snap, 'USD', f.CUT, concat_metadata=False)
        attribution = economic.estimate_factor_attribution(clean_meta, snap, 'USD', f.CUT)
        for name, table in [('snapshot', snap), ('cmas', cmas), ('attribution', attribution),
                            ('betas', covar.y_betas), ('variances', covar.y_variances),
                            ('clusters', covar.clusters), ('signs', covar.derived_signs)]:
            table.to_csv(out / f'{label}_{name}.csv', float_format='%.17g')
        covars[label] = covar
        results[label] = (snap, cmas, attribution)
        if label == 'old':
            checks = {
                'beta_max_error': float((covar.y_betas-assets.covar_data.y_betas.loc[names]).abs().max().max()),
                'variance_max_error': float((covar.y_variances.select_dtypes('number')-assets.covar_data.y_variances.loc[names].select_dtypes('number')).abs().max().max()),
                'cma_max_error': float((cmas.base_total_cma-assets.cma_metadata.base_total_cma).abs().max()),
                'alpha_max_error': float((snap.stat_alpha-assets.cma_metadata.stat_alpha).abs().max()),
            }
            receipt['baseline_parity'] = checks
            (out / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
            print('BASELINE', json.dumps(checks), flush=True)
            assert max(checks.values()) < 2e-7, checks
        print('DONE', label, flush=True)
    pd.testing.assert_series_equal(covars['old'].clusters, covars['revised'].clusters)
    report = meta[['Index', 'name', 'asset_class', 'sub_asset_class', 'universe', 'Rebalancing', 'alpha_weight']].copy()
    for label, (snap, cmas, attrib) in results.items():
        for col in ('base_total_cma', 'base_excess_factor_cma', 'excess_alpha_cma', 'base_excess_sharpe'):
            report[label+'_'+col] = cmas[col]
        for col in ('total_vol', 'r2', 'stat_alpha'):
            report[label+'_'+col] = snap[col]
        report[label+'_sharpe_rf0'] = cmas.base_total_cma / snap.total_vol
    for col in ('base_total_cma', 'base_excess_factor_cma', 'excess_alpha_cma', 'total_vol'):
        report['delta_'+col+'_bp'] = (report['revised_'+col]-report['old_'+col])*10000
    report['delta_r2_pp'] = (report.revised_r2-report.old_r2)*100
    delta_beta = covars['revised'].y_betas-covars['old'].y_betas
    pure_delta = delta_beta.mul(economic.factor_excess_cma.loc[f.CUT], axis=1).sum(axis=1)*10000
    report['delta_pure_factor_bp'] = pure_delta.reindex(meta.index).fillna(0.)
    report['delta_regional_bp'] = report.delta_base_excess_factor_cma_bp-report.delta_pure_factor_bp
    np.testing.assert_allclose(report.delta_base_total_cma_bp,
                               report.delta_pure_factor_bp+report.delta_regional_bp+report.delta_excess_alpha_cma_bp,
                               atol=1e-8)
    report.to_csv(out / 'cma_impact.csv', float_format='%.17g')
    delta_beta.to_csv(out / 'beta_delta.csv', float_format='%.17g')
    receipt['same_clusters'] = True
    receipt['decomposition_max_error_bp'] = float((report.delta_base_total_cma_bp-report.delta_pure_factor_bp-report.delta_regional_bp-report.delta_excess_alpha_cma_bp).abs().max())
    receipt['status'] = 'complete'
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    cols = ['name','old_base_total_cma','revised_base_total_cma','delta_base_total_cma_bp','delta_pure_factor_bp','delta_regional_bp','delta_excess_alpha_cma_bp','delta_r2_pp']
    print(report.loc[report.delta_base_total_cma_bp.abs().sort_values(ascending=False).index, cols].head(35).to_string(), flush=True)
    print('OUTPUT', out, flush=True)


if __name__ == '__main__':
    main()
