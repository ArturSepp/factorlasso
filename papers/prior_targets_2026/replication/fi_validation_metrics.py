"""Independent FI-study verification, attribution diagnostics and supporting simulations."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json, time, warnings, shutil
import numpy as np
import pandas as pd
import qis
from . import fi_validation as f

def full_replay(root):
    """Replay the unchanged monthly full-universe block, then change IL priors alone."""
    from ramen.quant_model_mvp.rosaa.data.factors.beta_priors import build_factor_for_prior
    from ramen.quant_model_mvp.rosaa.data.cmas.assets.global_saa import _resolve_factors_beta_prior
    assets=f.final_assets();bundle=f.factor_bundle();meta=pd.read_csv(root/'inputs/full_metadata.csv',index_col=0)
    names=meta.index[meta.Rebalancing.eq('ME') & ~meta.universe.eq('core_liquidity')]
    y=assets.excess_logreturns[names]
    prices=bundle.factor_prices[f.FACTORS].reindex(y.index,method='ffill').ffill()
    x=qis.to_returns(prices,is_log_returns=True,is_first_zero=False,drop_first=False,freq=None)
    signs=f.compute_beta_loading_signs_for_matf(meta.loc[names,'LongOnlyBetas'],meta.loc[names,'PEfactorExposure'],f.MATF_CUSTOM_IG_HY)
    prior=_resolve_factors_beta_prior(meta,f.MATF_CUSTOM_IG_HY,use_beta_priors=False,pe_beta_prior=1.)
    mapping=build_factor_for_prior(meta,f.MATF_CUSTOM_IG_HY)
    spec=f.get_cma_covar_estimation_spec()
    models={};out=root/'R1/full_replay';out.mkdir(parents=True,exist_ok=True)
    for case in ['current_joint_IL','univariate_IL']:
        select=mapping.copy()
        il=meta.index[meta.sub_asset_class.eq('IL Bonds')]
        if case=='univariate_IL':select.loc[il]='Inflation'
        model=f.get_covar_estimator(spec,factors_beta_loading_signs=signs,factors_beta_prior=prior,factor_for_prior=select).lasso_model
        model.fit(x,y,span=60,external_clusters=models['current_joint_IL'].clusters_ if models else None)
        models[case]=model
        f.save_model(model,out,case,{'n_monthly_assets':len(names),'other_prior_choices_unchanged':True})
    original=assets.covar_data.y_betas.loc[names]
    # The revised procedure is intentionally different. Reconstruct the archived
    # caller first, rather than requiring new estimates to equal old estimates.
    from .sign_revision import old_inputs
    legacy = models['current_joint_IL'].copy()
    legacy.auto_sign_use_fit_span = False
    legacy.auto_sign_variance = 'independent'
    with old_inputs(True):
        legacy.fit(x, y, span=60, external_clusters=models['current_joint_IL'].clusters_)
    baseline=models['current_joint_IL'] if (root/'owner_inputs').exists() else legacy
    maxdiff=float((original-baseline.estimated_betas).abs().max().max())
    f.csv(models['current_joint_IL'].estimated_betas-original, out/'sign_revision_beta_delta.csv')
    assert maxdiff<1e-7,('full-replay mismatch',maxdiff)
    a=models['current_joint_IL'];b=models['univariate_IL']
    untouched=names.difference(il)
    np.testing.assert_allclose(a.ols_beta_prior_.loc[untouched],b.ols_beta_prior_.loc[untouched],atol=1e-13)
    pd.testing.assert_frame_equal(a.derived_signs_.loc[untouched],b.derived_signs_.loc[untouched])
    delta=a.estimated_betas-b.estimated_betas
    comparison=pd.DataFrame({'name':meta.loc[names,'name'],'is_IL':names.isin(il),
        'r2_current':a.estimation_result_.r2,'r2_univariate':b.estimation_result_.r2,
        'max_beta_change':delta.abs().max(axis=1),
        'factor_cma_change_bp':delta.mul(bundle.cmas.loc[f.CUT,f.FACTORS],axis=1).sum(axis=1)*10000})
    f.csv(comparison,out/'comparison.csv')
    f.jsave(out/'receipt.json',dict(status='complete',n_monthly_assets=len(names),
        full_182_universe_context=True,unaffected_cadence_and_cash_blocks='not refitted; no shared penalty with monthly main block',
        baseline_max_beta_difference=maxdiff,untouched_priors_and_signs_identical=True))
    print('Full-universe replay verified; max baseline beta error',maxdiff,flush=True)

def diagnostics(root):
    """Summarise factor attribution, prior/sign controls, OAD and CMA sensitivity."""
    roster=pd.read_csv(root/'inputs/roster.csv',index_col=0)
    out=root/'summary';out.mkdir(exist_ok=True)
    endpoint=[]
    for lane in ['indices','funds']:
        x,y,rr=f.load(lane)
        hard=f.compute_beta_loading_signs_for_matf(pd.Series(True,index=rr.index),pd.Series(False,index=rr.index),f.MATF_CUSTOM_IG_HY)
        f.csv(hard,root/f'R1/{lane}/hard_signs.csv')
        detected=pd.read_csv(root/f'R1/{lane}/Z_signs.csv',index_col=0).where(hard.isna())
        # Detector signs on hard-constrained cells are immaterial and not exposed by owner API.
        # Keep them missing rather than refit a different unrestricted estimator.
        f.csv(detected,root/f'R1/{lane}/detected_signs_on_unrestricted_cells.csv')
        for arm in f.ARMS:
            path=root/f'R1/{lane}';b=pd.read_csv(path/f'{arm}_betas.csv',index_col=0)
            d=pd.read_csv(path/f'{arm}_diagnostics.csv',index_col=0)
            entry=b.join(d).join(roster[['group','name','kind']]);entry['arm']=arm;entry['lane']=lane
            entry['ticker']=entry.index;endpoint.append(entry)
    ep=pd.concat(endpoint);ep.to_csv(out/'endpoint.csv',index=False,float_format='%.17g')
    group=ep.groupby(['lane','group','arm']).fit_r2_owner.agg(['count','mean','median'])
    f.csv(group,out/'endpoint_groups.csv')
    cov=pd.read_csv(root/'inputs/factor_covar.csv',index_col=0)
    annual_factor_vol=pd.Series(np.sqrt(np.diag(cov)),index=cov.index)
    zeros=[];robust=[]
    for lane in ['indices','funds']:
        for arm in f.ARMS:
            d=ep[(ep.lane==lane)&(ep.arm==arm)].set_index('ticker')
            for tol in [1e-4,1e-3]:
                zeros.append(dict(lane=lane,arm=arm,tolerance=tol,
                    no_credit_share=float(d[['Credit IG','Credit HY','Credit EM']].abs().max(axis=1).le(tol).mean()),
                    active_factor_mean=float(d[f.FACTORS].abs().gt(tol).sum(axis=1).mean())))
            material=d[f.FACTORS].abs().mul(annual_factor_vol).gt(.001)
            f.csv(material,out/f'{lane}_{arm}_material_exposures.csv')
            labels=['separable_'+arm]+(['pe_only_'+arm] if lane=='indices' else [])
            for label in labels:
                diag=pd.read_csv(root/f'R1/{lane}/{label}_diagnostics.csv',index_col=0)
                robust.append(dict(lane=lane,variant=label,mean_fit_r2=float(diag.fit_r2_owner.mean())))
    pd.DataFrame(zeros).to_csv(out/'zero_tolerance_sensitivity.csv',index=False)
    pd.DataFrame(robust).to_csv(out/'robustness_fit.csv',index=False)
    rows=[]
    for label in ['E2','rating_alternative','hybrid_three_factor']:
        b=pd.read_csv(root/f'R1/indices/{label}_betas.csv',index_col=0)
        d=pd.read_csv(root/f'R1/indices/{label}_diagnostics.csv',index_col=0)
        for t in ['I12881US Index','I05040US Index','I05039US Index','H24641US Index']:
            rows.append(dict(variant=label,ticker=t,**b.loc[t].to_dict(),fit_r2=float(d.loc[t,'fit_r2_owner'])))
    pd.DataFrame(rows).to_csv(out/'mapping_sensitivities.csv',index=False)
    oad=[]
    characteristics=root/'characteristics';characteristics.mkdir(exist_ok=True)
    source=f.BASE/'analyses/brandon_fi_risk_models_20260922/resources'
    for p in source.glob('*_oads.csv'):
        target=characteristics/p.name
        if not target.exists():shutil.copy2(p,target)
        assert f.sha(p)==f.sha(target)
    f.jsave(characteristics/'manifest.json',{p.name:f.sha(p) for p in characteristics.glob('*.csv')})
    for p in characteristics.glob('*oads.csv'):
        data=f.read(p).loc[:f.CUT]
        data.columns=[str(t)+' Index' for t in data.columns]
        for t in data:
            if t in roster.index:
                obs=data[t].dropna()
                if len(obs):oad.append(dict(ticker=t,oad=obs.iloc[-1],date=str(obs.index[-1].date()),source=p.name))
    oad=pd.DataFrame(oad).drop_duplicates('ticker').set_index('ticker')
    joined=ep.query("lane=='indices'").join(oad,on='ticker');joined.to_csv(out/'oad_comparison.csv',index=False)
    # Correlation is descriptive: OAD was not used to tune the coefficients.
    duration=[]
    for arm in f.ARMS:
        s=joined.query('arm==@arm').dropna(subset=['oad'])
        duration.append(dict(arm=arm,n=len(s),spearman_rates_oad=float(s[['Rates','oad']].corr(method='spearman').iloc[0,1])))
    pd.DataFrame(duration).to_csv(out/'oad_rank_association.csv',index=False)
    if (root/'R2/cmas.csv').exists():
        cma=pd.read_csv(root/'R2/cmas.csv');cma=cma.join(roster[['group','name']],on='ticker')
        cma.to_csv(out/'cmas.csv',index=False,float_format='%.17g')
        weights=roster.query("lane=='indices'").groupby('group').size()
        portfolio=[];sensitivity=[]
        premium=f.read(root/'inputs/premia.csv').iloc[0]
        for arm in f.ARMS:
            table=cma.query('arm==@arm').set_index('ticker')
            w=table.group.map(lambda g:1/len(weights)/weights[g]);assert abs(w.sum()-1)<1e-12
            cov=pd.read_csv(root/f'R2/{arm}_covar.csv',index_col=0)
            # QIS RiskModel owns portfolio quadratic risk calculations.
            rm=qis.RiskModel(covar={f.CUT:cov})
            portfolio.append(dict(arm=arm,total_cma=float(w@table.base_total_cma),
                excess_cma=float(w@(table.base_total_cma-table.rf_rate)),
                **_portfolio_risk(rm,w)))
            b=pd.read_csv(root/f'R1/indices/{arm}_betas.csv',index_col=0)
            credit=b[['Credit IG','Credit HY','Credit EM']].mul(premium,axis=1).sum(axis=1)
            for q in [.8,1.,1.2]:
                for t in table.index:sensitivity.append(dict(arm=arm,ticker=t,q=q,total_cma=table.loc[t,'base_total_cma']+(q-1)*credit[t]))
        pd.DataFrame(portfolio).to_csv(out/'fixed_portfolio.csv',index=False)
        pd.DataFrame(sensitivity).to_csv(out/'credit_unit_sensitivity.csv',index=False)
    f.jsave(out/'endpoint_summary.json',dict(mean_fit_r2=ep.groupby(['lane','arm']).fit_r2_owner.mean().unstack().to_dict('index'),
        oad_comparisons=duration))

def _portfolio_risk(rm,w):
    """Use the existing QIS portfolio risk API; introspection is checked by the probe."""
    return dict(vol=float(rm.compute_tre_at_date(
        benchmark_weights=pd.Series(0.,index=w.index),portfolio_weights=w,date=f.CUT)))

# Paired held-out contrasts. The last two compare against the sign-constrained zero-penalty fit
# with detected signs, which isolates what targets and penalty add beyond least squares.
CONTRASTS=[('A','Z'),('E1','Z'),('E2','Z'),('E1','A'),('E2','A'),
           ('E2','Z_lambda_zero'),('E2_lambda_zero','Z_lambda_zero')]

def scores(root,lane):
    """Independently rebuild OOS scores from observations and paired quarterly resamples."""
    path=root/f'R3/{lane}';out=root/f'summary/{lane}';out.mkdir(parents=True,exist_ok=True)
    p=pd.concat([pd.read_csv(z) for z in sorted(path.glob('*_predictions.csv'))],ignore_index=True)
    assert p.groupby(['span','arm']).fit_date.nunique().eq(28).all()
    assert (pd.to_datetime(p.month)>pd.to_datetime(p.fit_date)).all()
    keys=['span','fit_date','month','ticker']
    ref=p.query("arm=='Z'").sort_values(keys)
    arms=list(f.ARMS)+list(f.CONTROLS)
    assert set(p.arm)==set(arms),sorted(set(p.arm))
    for arm in arms[1:]:
        other=p.query('arm==@arm').sort_values(keys)
        pd.testing.assert_frame_equal(ref[keys+['actual','benchmark']].reset_index(drop=True),other[keys+['actual','benchmark']].reset_index(drop=True))
    p['error']=p.actual-p.predicted;p['sse']=p.error**2;p['denom']=(p.actual-p.benchmark)**2
    pooled=p.groupby(['span','arm','ticker','group','kind']).agg(sse=('sse','sum'),denom=('denom','sum'),n=('sse','size'),bias=('error','mean'))
    pooled['oos_r2']=1-pooled.sse/pooled.denom;pooled['rmse']=np.sqrt(pooled.sse/pooled.n)
    f.csv(pooled,out/'asset_scores.csv')
    group=pooled.reset_index().groupby(['span','arm','group','kind']).agg(n=('ticker','size'),mean_r2=('oos_r2','mean'),mean_rmse=('rmse','mean'))
    f.csv(group,out/'category_scores.csv')
    beta=pd.concat([pd.read_csv(z) for z in sorted(path.glob('*_betas.csv'))])
    beta=beta.sort_values('fit_date');beta['movement']=beta.groupby(['span','arm','ticker','factor']).beta.diff().abs()
    f.csv(beta.groupby(['span','arm']).movement.agg(['mean','median','max']),out/'stability.csv')
    beta['sign_switch']=beta.groupby(['span','arm','ticker','factor']).sign.diff().abs().gt(0)
    f.csv(beta.groupby(['span','arm']).sign_switch.mean(),out/'sign_switches.csv')
    a=beta.query("arm=='A'").pivot(index=['span','fit_date','ticker'],columns='factor',values='prior')
    winner=a.abs().idxmax(axis=1).where(a.abs().max(axis=1)>1e-10,'blocked/zero')
    winner=winner.rename('winner').reset_index().sort_values('fit_date')
    previous=winner.groupby(['span','ticker']).winner.shift()
    winner['switch']=winner.winner.ne(previous).where(previous.notna())
    winner.to_csv(out/'automatic_winners.csv',index=False)
    p['year']=pd.to_datetime(p.month).dt.year
    episode=p[p.year.isin([2020,2022,2023])].groupby(['span','arm','year']).agg(sse=('sse','sum'),denom=('denom','sum'),n=('sse','size'))
    episode['r2']=1-episode.sse/episode.denom;f.csv(episode,out/'episodes.csv')
    # Fixed initial cohort; entire calendar quarters jointly resampled across assets/arms.
    bootstrap=[]
    for span in [60,36]:
        panel=p.query('span==@span')
        common=panel.groupby(['arm','ticker']).size().unstack(0).eq(84).all(axis=1)
        common=common.index[common]
        panel=panel[panel.ticker.isin(common)&panel.group.ne('Rates')]
        grouped=panel.groupby(['fit_date','ticker','arm'])[['sse','denom']].sum()
        assets=sorted(panel.ticker.unique());dates=sorted(panel.fit_date.unique())
        idx=pd.MultiIndex.from_product([dates,assets,arms],names=['fit_date','ticker','arm'])
        arr=grouped.reindex(idx).to_numpy().reshape(28,len(assets),len(arms),2);assert np.isfinite(arr).all()
        meta=panel.drop_duplicates('ticker').set_index('ticker').reindex(assets)
        cats=sorted(meta.group.unique());cat_weights=np.array([1/len(cats)/(meta.group==g).sum() for g in meta.group])
        base_r2=1-arr[:,:,:,0].sum(axis=0)/arr[:,:,:,1].sum(axis=0)
        for block in [2,4,8]:
            indices=qis.generate_bootstrapped_indices(num_data_index=28,bootstrap_type=qis.BootstrapType.STATIONARY,
                num_samples=5000,index_length=28,block_size=block,seed=20260924+block)
            if indices.shape==(28,5000):indices=indices.T
            assert indices.shape==(5000,28)
            sums=arr[indices].sum(axis=1);r2=1-sums[:,:,:,0]/sums[:,:,:,1]
            for arm,comparator in CONTRASTS:
                j,k=arms.index(arm),arms.index(comparator)
                paired=(r2[:,:,j]-r2[:,:,k])@cat_weights
                ci=np.quantile(paired,[.025,.975]);point=float((base_r2[:,j]-base_r2[:,k])@cat_weights)
                bootstrap.append(dict(span=span,arm=arm,comparator=comparator,block=block,cohort=len(assets),delta_r2=point,low=ci[0],high=ci[1],positive_fraction=float((paired>0).mean())))
    pd.DataFrame(bootstrap).to_csv(out/'bootstrap.csv',index=False)
    # Deliberate data defect must fail the same date guard before trusting the clean audit.
    bad=p.iloc[:1].copy();bad['fit_date']=bad.month
    assert not (pd.to_datetime(bad.month)>pd.to_datetime(bad.fit_date)).all()
    audit=dict(status='complete',rows=len(p),all_paired_masks_identical=True,quarters=28,
        corrupted_date_rejected=True,mean_r2=pooled.reset_index().groupby(['span','arm']).oos_r2.mean().unstack().to_dict('index'))
    f.jsave(out/'receipt.json',audit);print('OOS summary',lane,audit['mean_r2'],flush=True)

def run(stage,root,lane):
    """Dispatch additional approved research stages."""
    f.ROOT=root
    if stage=='full':full_replay(root)
    elif stage=='summary':diagnostics(root)
    elif stage=='scores':scores(root,lane)
    elif stage=='checks':final_checks(root)
    elif stage in ['pilot','R4','examples']:
        from . import fi_validation_synthetic
        fi_validation_synthetic.run(root,pilot=stage=='pilot',examples=stage=='examples')
    elif stage=='prepare':
        raise ValueError('The legacy template prepare is retired; run refresh_current --stage prepare, '
                         'which fills evidence blocks and values in manuscript.md without rewriting prose.')
    elif stage=='R5':
        from . import build_latex
        candidates=[root/f'R5/build_v{i}' for i in range(1,10)]
        build_latex.build(next(p for p in candidates if not p.exists()),False)
    else:raise ValueError(stage)

def final_checks(root):
    """Verify the saved experiment independently, including adverse focal MC assets."""
    for lane in ['indices','funds']:
        path=root/f'R1/{lane}'
        hard=pd.read_csv(path/'hard_signs.csv',index_col=0)
        detected=pd.read_csv(path/'Z_signs.csv',index_col=0)
        for arm in ['A','E1','E2']:
            raw=pd.read_csv(path/f'{arm}_raw_prior.csv',index_col=0)
            signs=pd.read_csv(path/f'{arm}_signs.csv',index_col=0)
            expected=detected.copy();override=hard.isna()&raw.abs().gt(1e-14)&np.isfinite(raw)
            expected=expected.where(~override,np.sign(raw))
            np.testing.assert_allclose(signs,expected,atol=1e-12,equal_nan=True)
            f.csv(override,path/f'{arm}_prior_sign_override_cells.csv')
    cma=pd.read_csv(root/'R2/cmas.csv')
    np.testing.assert_allclose(cma.base_total_cma,cma.pure_factor_total_cma+cma.regional_adjustment,atol=1e-12)
    for a in f.ARMS:
        cov=pd.read_csv(root/f'R2/{a}_covar.csv',index_col=0)
        np.testing.assert_allclose(cov,cov.T,atol=1e-12)
        assert np.linalg.eigvalsh(cov).min()>0
    for rho in [-.8,.25,.9,.999]:
        a=.4;np.testing.assert_allclose((a*(1-rho))**2+(a*np.sqrt(1-rho*rho))**2,2*a*a*(1-rho),atol=1e-15)
    assert abs(.45+.9*(-.8)-(-.27))<1e-12
    from .fi_validation_synthetic import CASES,ARMS as MC_ARMS,ASSETS as MC_ASSETS
    mc=pd.concat([pd.read_csv(root/f'R4/{c}_{n}.csv') for c in CASES for n in [60,120]])
    expected_rows=len(CASES)*2*len(MC_ARMS)*200*len(MC_ASSETS)
    assert len(mc)==expected_rows and set(mc.arm)==set(MC_ARMS)
    assert mc.groupby(['case','n','arm']).seed.nunique().eq(200).all()
    assert all(json.loads((root/f'R4/{c}_{n}_audit.json').read_text())['failures']==0 for c in CASES for n in [60,120])
    mc['abs_cma_bp']=mc.cma_error_bp.abs()
    adverse=[]
    for case,asset in [('credit_high','IG'),('credit_wrong','IG'),('credit_omitted','Convertible'),('il_weak','IL')]:
        d=mc[(mc.case==case)&(mc.asset==asset)&(mc.n==120)]
        for arm,g in d.groupby('arm'):
            adverse.append(dict(case=case,asset=asset,arm=arm,mean_absolute_cma_error_bp=float(g.abs_cma_bp.mean()),
                mean_beta_mse=float(g.beta_mse.mean()),inflation_sign_wrong_fraction=float((g.inflation_beta>=0).mean()) if asset=='IL' else None))
    pd.DataFrame(adverse).to_csv(root/'summary/mc_focal_cases.csv',index=False)
    f.jsave(root/'final_numerical_checks.json',dict(status='passed',saved_sign_priority_reconstructed=True,
        cma_adjustments_reconciled=True,covariance_positive_definite=True,analytical_identities=True,
        monte_carlo_rows=expected_rows,paired_draws=2400,solver_failures=0))
    print(pd.DataFrame(adverse).to_string(index=False),flush=True)
