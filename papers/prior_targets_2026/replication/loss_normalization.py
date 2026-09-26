"""Controlled loss-normalization calibration and CMA/FI validation.

Run with the ROSAA external environment and FactorLasso on PYTHONPATH. All
generated output is C-local; economic assumptions and production settings stay fixed.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from papers.prior_targets_2026.replication import fi_validation as f
from factorlasso import lasso_estimator as le
import optimalportfolios.covar_estimation.factor_covar_estimator as covariance
from ramen.quant_model_mvp.rosaa.data.factors.beta_priors import build_factor_for_prior
from ramen.quant_model_mvp.rosaa.data.cmas.assets.global_saa import _resolve_factors_beta_prior

SOURCE = Path('C:/Users/artur/AppData/Local/AgentWork/ARTURDESKTOP/FactorLasso/analyses/fi_sign_validation_20260925')
OLD_FI = SOURCE.parent / 'fi_prior_validation_20260924'
REVISED = f.BASE / 'analyses/sign_revision_cma_impact_20260925'
OUT = Path(os.environ['AGENT_LOCAL_ROOT']) / 'analyses/loss_normalization_20260925'


def save_json(path, value):
    """Write finite JSON evidence in the analysis workspace."""
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False)+'\n', encoding='utf-8')


def full_mass(t, span):
    """Independent geometric weight mass on a complete observation grid."""
    return float(t) if span is None else float(np.sum((1-2/(span+1))**np.arange(t)))


@contextmanager
def frequency_calibration(mode, multiplier, reference, diagnostics):
    """Apply fixed cadence-specific penalties only within this experiment.

    The existing owner has no cadence-specific lambda argument. Intercept its
    frequency boundary locally; all fitting, covariance and CMA arithmetic still
    execute in the owner. Restore the function and model settings after each call.
    """
    original = covariance._fit_lasso_frequency

    def call(**kwargs):
        """Set the approved experimental loss/penalty and collect native-fit masses."""
        model, freq = kwargs['lasso_model'], kwargs['freq']
        old_lambda, old_mode = model.reg_lambda, model.loss_normalization
        model.loss_normalization = mode
        model.reg_lambda = (old_lambda if mode == 'sample' else reference[freq]['lambda']*multiplier)
        try:
            result = original(**kwargs)
            diag = pd.DataFrame({'weight_mass': model.loss_weight_mass_,
                                 'denominator': model.loss_denominator_,
                                 'valid_count': model.valid_mask_.sum(axis=0)})
            diag['frequency'] = freq
            diag['rows'] = model.n_loss_rows_
            diag['span'] = model.effective_span_
            diag['lambda'] = model.reg_lambda
            diagnostics.append(diag)
            if mode == 'sample':
                t, span = model.n_loss_rows_, model.effective_span_
                mass = full_mass(t, span)
                reference[freq] = dict(rows=t, span=span, full_mass=mass,
                                       old_lambda=old_lambda, conversion=t/mass,
                                       **{'lambda': old_lambda*t/mass})
            return result
        finally:
            model.reg_lambda, model.loss_normalization = old_lambda, old_mode

    covariance._fit_lasso_frequency = call
    try:
        yield
    finally:
        covariance._fit_lasso_frequency = original


def cma():
    """Reconcile the revised-sign endpoint, then isolate loss and penalty changes."""
    out = OUT / 'cma'
    out.mkdir(parents=True, exist_ok=True)
    assets, bundle = f.final_assets(), f.factor_bundle()
    economic = bundle.select(f.MATF_CUSTOM_IG_HY).for_asset_model(f.MATF_CUSTOM_IG_HY)
    meta = pd.read_csv(SOURCE/'inputs/full_metadata.csv', index_col=0)
    names = meta.index[~meta.universe.eq('core_liquidity')]
    cash = meta.index[meta.universe.eq('core_liquidity')]
    panels = {}
    for freq in ('ME', 'QE'):
        cols = names[meta.loc[names, 'Rebalancing'].eq(freq)]
        panel = assets.excess_logreturns.loc[:f.CUT, cols]
        panels[freq] = panel.loc[panel.index.is_quarter_end] if freq == 'QE' else panel
    signs = f.compute_beta_loading_signs_for_matf(
        meta.loc[names, 'LongOnlyBetas'], meta.loc[names, 'PEfactorExposure'], f.MATF_CUSTOM_IG_HY)
    priors = _resolve_factors_beta_prior(meta, f.MATF_CUSTOM_IG_HY, False, pe_beta_prior=1.)
    mapping = build_factor_for_prior(meta, f.MATF_CUSTOM_IG_HY)
    spec = f.get_cma_covar_estimation_spec()
    clean_meta = meta.astype(object).where(meta.notna(), None)
    cash_snapshot = assets.covar_data.get_snapshot(alpha_span=spec.alpha_span,
                                                  asset_frequencies=meta.Rebalancing).loc[cash]
    cash_snapshot['stat_alpha'] = assets.cma_metadata.loc[cash, 'stat_alpha']
    reference, runs, summaries = {}, {}, []
    paper = pd.read_csv(REVISED/'paper_18_cma_impact.csv', index_col=0)
    variants = [('legacy', 'sample', 1.), ('normalized_matched', 'weight_sum', 1.),
                ('normalized_half', 'weight_sum', .5), ('normalized_double', 'weight_sum', 2.),
                ('normalized_zero', 'weight_sum', 0.)]
    for label, mode, multiplier in variants:
        model = f.get_covar_estimator(spec, factors_beta_loading_signs=signs,
                                     factors_beta_prior=priors, factor_for_prior=mapping).lasso_model
        model.solver = 'MOSEK'
        model.auto_sign_use_fit_span = True
        model.auto_sign_variance = 'date'
        diagnostics = []
        with frequency_calibration(mode, multiplier, reference, diagnostics), warnings.catch_warnings(record=True) as caught:
            risk = covariance.estimate_lasso_factor_covar_data(
                risk_factor_prices=bundle.factor_prices.loc[:f.CUT, f.FACTORS],
                asset_returns_dict=panels, lasso_model=model, assets=names,
                x_covar=assets.covar_data.x_covar, estimation_date=f.CUT,
                factor_returns_freq=spec.factor_returns_freq, factor_covar_span=spec.factor_covar_span)
        snap = risk.get_snapshot(alpha_span=spec.alpha_span, asset_frequencies=meta.Rebalancing)
        snap = pd.concat([snap, cash_snapshot]).reindex(meta.index)
        result, _ = economic.estimate_asset_universe_cma(clean_meta, snap, 'USD', f.CUT, concat_metadata=False)
        pd.concat(diagnostics).to_csv(out/f'{label}_loss_diagnostics.csv')
        for key, data in [('betas', risk.y_betas), ('cmas', result), ('snapshot', snap)]:
            data.to_csv(out/f'{label}_{key}.csv', float_format='%.17g')
        runs[label] = (risk, snap, result)
        if label == 'legacy':
            baseline_cmas = pd.read_csv(REVISED/'revised_cmas.csv', index_col=0)
            baseline_betas = pd.read_csv(REVISED/'revised_betas.csv', index_col=0)
            errors = dict(beta=float((risk.y_betas-baseline_betas).abs().max().max()),
                          total_cma=float((result.base_total_cma-baseline_cmas.base_total_cma).abs().max()))
            assert max(errors.values()) < 2e-7, errors
            save_json(out/'baseline_parity.json', errors)
        base_risk, base_snap, base = runs['legacy']
        pd.testing.assert_series_equal(risk.clusters, base_risk.clusters)
        pd.testing.assert_frame_equal(risk.derived_signs, base_risk.derived_signs)
        table = meta[['name', 'asset_class', 'sub_asset_class', 'Rebalancing', 'alpha_weight']].copy()
        for col in ['base_total_cma', 'base_excess_factor_cma', 'excess_alpha_cma']:
            table['old_'+col] = base[col]
            table['new_'+col] = result[col]
            table['delta_'+col+'_bp'] = 10000*(result[col]-base[col])
        table['vol_before'] = base_snap.total_vol
        table['vol_after'] = snap.total_vol
        table['r2_before'] = base_snap.r2
        table['r2_after'] = snap.r2
        np.testing.assert_allclose(table.delta_base_total_cma_bp,
            table.delta_base_excess_factor_cma_bp+table.delta_excess_alpha_cma_bp, atol=1e-8)
        table.to_csv(out/f'{label}_impact.csv', float_format='%.17g')
        paper_table = table.loc[paper.index].copy()
        paper_table.insert(0, 'paper_name', paper.paper_name)
        paper_table.to_csv(out/f'{label}_paper_18.csv', float_format='%.17g')
        for scope, selection in [('all_182', table), ('paper_18', paper_table),
                                 ('bonds', table[table.asset_class.eq('Bonds') & ~meta.universe.eq('core_liquidity')])]:
            delta = selection.delta_base_total_cma_bp
            summaries.append(dict(variant=label, scope=scope, n=len(selection),
                mean_delta_bp=float(delta.mean()), median_absolute_bp=float(delta.abs().median()),
                max_absolute_bp=float(delta.abs().max()), warnings=len(caught)))
        print('CMA', label, summaries[-2], flush=True)
    pd.DataFrame(summaries).to_csv(out/'summary.csv', index=False)
    save_json(out/'calibration.json', reference)
    save_json(out/'receipt.json', dict(status='complete', solver='MOSEK', same_clusters=True,
        same_effective_signs=True, economic_assumptions='frozen pre-exhibit calibration',
        baseline='revised sign methodology', convention='per-response valid EWMA weight mass',
        production_changed=False, paper_changed=False))


def oos():
    """Compare fixed normalized penalties over the existing 28 held-out quarters.

    Set the reference conversion at the FIRST fit date using only its row count
    and span. Hold normalized lambda fixed afterwards; do not reconvert each date.
    This isolates the old penalty's mechanical growth as the panel expands.
    """
    out = OUT / 'oos'
    out.mkdir(parents=True, exist_ok=True)
    f.ROOT = OLD_FI
    records, protocol = [], []
    # A separately labelled retrospective check of the current-cut conversion.
    endpoint_lambda = json.loads((OUT/'cma/calibration.json').read_text('utf-8'))['ME']['lambda']
    for lane in ('indices', 'funds'):
        x, y, roster = f.load(lane)
        span = 60
        first_rows = len(x.loc[:f.QUARTERS[0]])-1
        normalized_lambda = 1e-5*first_rows/full_mass(first_rows, span)
        protocol.append(dict(lane=lane, first_fit=str(f.QUARTERS[0]), first_rows=first_rows,
                             span=span, fixed_normalized_lambda=normalized_lambda,
                             retrospective_endpoint_lambda=endpoint_lambda))
        for q in f.QUARTERS:
            xx, yy = x.loc[:q], y.reindex(x.loc[:q].index)
            names = yy.columns[(yy.notna().sum() >= 60) & yy.iloc[-1].notna()]
            yy, rr = yy[names], roster.loc[names]
            dates = pd.date_range(q+pd.offsets.MonthEnd(1), periods=3, freq='ME')
            xt, yt = x.loc[dates], y.loc[dates, names]
            clusters, hard, prior_signs = None, None, None
            for label, mode, lam in [
                ('legacy', 'sample', 1e-5),
                ('normalized_half', 'weight_sum', normalized_lambda*.5),
                ('normalized_matched', 'weight_sum', normalized_lambda),
                ('normalized_double', 'weight_sum', normalized_lambda*2),
                ('normalized_zero', 'weight_sum', 0.),
                ('normalized_endpoint', 'weight_sum', endpoint_lambda),
                ('normalized_endpoint_half', 'weight_sum', endpoint_lambda*.5)]:
                model = f.make_model(rr, 'E2', lam=lam, span=span, signs=hard)
                model.solver = 'MOSEK'
                model.loss_normalization = mode
                model.auto_sign_use_fit_span = True
                model.auto_sign_variance = 'date'
                model.fit(xx, yy, span=span, external_clusters=clusters)
                if clusters is None:
                    clusters = model.clusters_.copy()
                    prior_signs = model.derived_signs_.copy()
                pd.testing.assert_frame_equal(model.derived_signs_, prior_signs)
                predicted = model.predict(xt)
                for name in names:
                    valid = yt[name].notna()
                    error = predicted.loc[valid, name]-yt.loc[valid, name]
                    records.append(dict(lane=lane, fit_date=str(q.date()), variant=label,
                                        ticker=name, n=int(valid.sum()), sse=float(np.square(error).sum()),
                                        **{'lambda': lam}))
            print('OOS', lane, str(q.date()), flush=True)
    detail = pd.DataFrame(records)
    detail.to_csv(out/'scores.csv', index=False)
    summary = detail.groupby(['lane', 'variant'])[['n', 'sse']].sum()
    summary['monthly_rmse_pct'] = 100*np.sqrt(summary.sse/summary.n)
    summary.to_csv(out/'summary.csv')
    save_json(out/'protocol.json', dict(reference=protocol, solver='MOSEK',
        prior_policy='Rates-conditioned E2 research policy', held_fixed=['signs', 'targets', 'clusters'],
        limitations=['fixed-vintage conditional return reconstruction',
                     'research subset, not the full CMA prior mapping',
                     'sensitivity comparison, not independent validation of a selected lambda',
                     'endpoint-labelled variants use a retrospective current-cut penalty conversion']))
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['cma', 'oos'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (cma if args.stage == 'cma' else oos)()
