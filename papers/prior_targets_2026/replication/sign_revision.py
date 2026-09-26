"""Frozen-data FI sign ablations, MOSEK fits, held-out and stress controls.

Run as a module after copying to prior_targets_2026/replication. Original
observations, mappings, priors, clusters, penalties and premiums are retained.
The zero-filled old input is reproduced only inside this research context.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import json
import warnings
import numpy as np
import pandas as pd
import qis
import factorlasso.sign_constraints as signs
from . import fi_validation as f

VARIANTS=('old','mask','ewma','date')
LABELS={'old':'Archived sign procedure','mask':'Valid observations only',
        'ewma':'Valid observations and EWMA','date':'EWMA and date-dependent gate'}
OLD=Path('C:/Users/artur/AppData/Local/AgentWork/ARTURDESKTOP/FactorLasso/analyses/fi_prior_validation_20260924')
STRESS=Path('C:/Users/artur/AppData/Local/AgentWork/ARTURDESKTOP/FactorLasso/analyses/fi_prior_stress_20260925/evidence')


@contextmanager
def old_inputs(enabled):
    """Reproduce the historical caller defect without a production compatibility switch."""
    originals={n:getattr(signs,n) for n in ('_compute_sign_vector','_compute_sign_matrix_per_response')}
    def wrap(fn):
        """Adapt only research inputs, preserving the original helper callable."""
        def call(x_arr,y_arr,*args,**kwargs):
            """Restore historical zero-filled sign input."""
            return fn(np.nan_to_num(x_arr),np.nan_to_num(y_arr),*args,**kwargs)
        return call
    try:
        if enabled:
            for name,fn in originals.items():setattr(signs,name,wrap(fn))
        yield
    finally:
        for name,fn in originals.items():setattr(signs,name,fn)


def fit(x,y,roster,arm,variant,span=60,clusters=None,lam=None,hard=None):
    """Select the sign ablation explicitly while keeping the numerical fit owner."""
    model=f.make_model(roster,arm,lam=lam,span=span,signs=hard)
    model.solver='MOSEK'
    model.auto_sign_use_fit_span=variant in ('ewma','date')
    model.auto_sign_variance='date' if variant=='date' else 'independent'
    with old_inputs(variant=='old'),warnings.catch_warnings(record=True) as caught:
        model.fit(x,y,span=span,external_clusters=clusters,verbose=False)
    assert model.solver=='MOSEK' and np.isfinite(model.estimated_betas).all().all()
    return model,[str(w.message) for w in caught]


def endpoint(out,source=OLD):
    """Separate missingness, weighting and gate effects on all four policies."""
    f.ROOT=Path(source)
    premia=f.read(f.ROOT/'inputs/premia.csv').iloc[0]
    shocks=f.read((f.ROOT/'stress' if (f.ROOT/'stress').exists() else STRESS)/'factor_log_shocks.csv')
    rows=[];details=[];baseline=[]
    for lane in ('indices','funds'):
        x,y,roster=f.load(lane)
        clusters=pd.read_csv(f.ROOT/f'R1/{lane}/Z_clusters.csv',index_col=0).iloc[:,0]
        reference={}
        for variant in VARIANTS:
            dest=out/'endpoint'/variant/lane;dest.mkdir(parents=True,exist_ok=True)
            for arm in f.ARMS:
                m,warnings_=fit(x,y,roster,arm,variant,clusters=clusters)
                if variant=='old':reference[arm]=m
                b=m.estimated_betas; old=reference[arm];delta=b-old.estimated_betas
                for attr in ('estimated_betas','derived_signs_','detected_signs_','sign_slopes_',
                             'sign_t_stats_','sign_effective_n_','sign_valid_counts_','sign_penalty_weights_'):
                    getattr(m,attr).to_csv(dest/f'{arm}_{attr}.csv')
                np.savetxt(dest/f'{arm}_block_weights.csv',m.sign_block_weights_,delimiter=',')
                result=qis.project_factor_scenarios(betas=b,amounts=pd.Series(1.,index=b.index),factor_log_shocks=shocks)
                oldstress=qis.project_factor_scenarios(betas=old.estimated_betas,
                    amounts=pd.Series(1.,index=b.index),factor_log_shocks=shocks)
                result.asset_pnl.to_csv(dest/f'{arm}_stress.csv')
                mu=b.mul(premia,axis=1).sum(axis=1)
                delta_mu=delta.mul(premia,axis=1).sum(axis=1)*10000
                record=dict(lane=lane,variant=variant,arm=arm,
                    detected_changes=int(m.detected_signs_.ne(old.detected_signs_).sum().sum()),
                    effective_changes=int(m.derived_signs_.fillna(9).ne(old.derived_signs_.fillna(9)).sum().sum()),
                    adaptive_changes=int((m.sign_penalty_weights_-old.sign_penalty_weights_).abs().gt(1e-10).sum().sum()),
                    block_changes=int(np.sum(np.abs(m.sign_block_weights_-old.sign_block_weights_)>1e-10)),
                    max_beta_change=float(delta.abs().max().max()),mean_abs_beta_change=float(delta.abs().mean().mean()),
                    mean_r2=float(np.mean(m.estimation_result_.r2)),
                    max_abs_cma_change_bp=float(delta_mu.abs().max()),mean_cma_change_bp=float(delta_mu.mean()),
                    max_abs_stress_change_pp=float((result.asset_pnl-oldstress.asset_pnl).abs().max().max()*100),
                    warnings=len(warnings_))
                rows.append(record)
                for t in b.index:details.append(dict(lane=lane,variant=variant,arm=arm,ticker=t,
                    name=roster.loc[t,'name'],factor_premium=float(mu[t]),cma_delta_bp=float(delta_mu[t]),
                    r2=float(m.estimation_result_.r2[b.index.get_loc(t)])))
                if variant==('date' if (f.ROOT/'owner_inputs').exists() else 'old'):
                    saved=pd.read_csv(f.ROOT/f'R1/{lane}/{arm}_betas.csv',index_col=0)
                    err=float((b-saved).abs().max().max())
                    assert err<2e-5,(lane,arm,err)
                    baseline.append(dict(lane=lane,arm=arm,baseline_beta_max_error=err))
                if variant=='date':
                    full,_=fit(x,y,roster,arm,variant)
                    pd.testing.assert_series_equal(full.clusters_,m.clusters_,check_names=False,check_dtype=False)
                    np.testing.assert_allclose(full.estimated_betas,b,atol=2e-5)
                print('endpoint',lane,variant,arm,round(record['max_abs_cma_change_bp'],2),'bp',flush=True)
    summary=pd.DataFrame(rows)
    # The archived variant patches helpers that FactorLasso imports at call time. If that import moves,
    # the patch silently stops working and 'old' equals 'mask'; the late-entrant rows make them differ.
    effect=summary.query("variant=='mask'").groupby('lane')[['max_beta_change','adaptive_changes']].max()
    assert ((effect.max_beta_change>0)|(effect.adaptive_changes>0)).all(),f'archived sign-input patch had no effect: {effect}'
    summary.to_csv(out/'endpoint_summary.csv',index=False)
    pd.DataFrame(details).to_csv(out/'endpoint_details.csv',index=False)
    (out/'baseline_parity.json').write_text(json.dumps(baseline,indent=2),encoding='utf-8')


def target_predict(model,x,y,xt,span):
    """Feasible target-only control with the owner's weighted-intercept convention.

    No exported FactorLasso prediction API accepts unfitted arbitrary target
    coefficients. Use the documented weighted raw residual mean here, and verify
    it independently against the fitted owner's alpha before applying the target.
    """
    b=model.effective_beta_prior_.copy()
    s=model.derived_signs_
    b=b.mask(s.eq(0),0.).mask(s.eq(1)&b.lt(0),0.).mask(s.eq(-1)&b.gt(0),0.)
    w=pd.Series((1-2/(span+1))**np.arange(len(x)-1,-1,-1),index=x.index)
    def alpha(beta):
        """Average raw residuals over each response's weighted valid observations."""
        e=y-x@beta.T
        return e.mul(w,axis=0).sum()/e.notna().mul(w,axis=0).sum()
    np.testing.assert_allclose(alpha(model.estimated_betas),model.alpha_const_,atol=1e-12)
    return (xt@b.T).add(alpha(b),axis=1)


def oos(out,source=OLD):
    """Repeat all 28 quarterly splits at both horizons with paired controls."""
    f.ROOT=Path(source)
    for lane in ('indices','funds'):
        x,y,roster=f.load(lane)
        for span in (60,36):
            for q in f.QUARTERS:
                dest=out/'oos'/f'{lane}_{span}_{q:%Y%m%d}.csv';dest.parent.mkdir(exist_ok=True)
                if dest.exists():continue
                xx=x.loc[:q]; yy=y.reindex(xx.index)
                names=yy.columns[(yy.notna().sum()>=60)&yy.iloc[-1].notna()]
                yy=yy[names];rr=roster.loc[names]
                dates=pd.date_range(q+pd.offsets.MonthEnd(1),periods=3,freq='ME')
                xt=x.loc[dates];yt=y.loc[dates,names]
                base,unused=fit(xx,yy,rr,'Z','old',span)
                rows=[]
                for variant in VARIANTS:
                    for arm in f.ARMS:
                        model,unused=fit(xx,yy,rr,arm,variant,span,clusters=base.clusters_)
                        predictions={arm:model.predict(xt)}
                        if variant=='date' and arm in ('A','E1','E2'):
                            predictions[arm+'_target_only']=target_predict(model,xx,yy,xt,span)
                        if variant=='date' and arm in ('Z','E2'):
                            zero,unused=fit(xx,yy,rr,arm,variant,span,clusters=base.clusters_,lam=0.,hard=model.derived_signs_)
                            pd.testing.assert_frame_equal(zero.derived_signs_,model.derived_signs_)
                            predictions[arm+'_lambda_zero']=zero.predict(xt)
                        for policy,pred in predictions.items():
                            for ticker in names:
                                mask=yt[ticker].notna();err=pred.loc[mask,ticker]-yt.loc[mask,ticker]
                                rows.append(dict(lane=lane,span=span,fit_date=str(q.date()),variant=variant,
                                    arm=policy,ticker=ticker,n=int(mask.sum()),sse=float((err*err).sum())))
                pd.DataFrame(rows).to_csv(dest,index=False)
                print('OOS',lane,span,q.date(),flush=True)
    data=pd.concat([pd.read_csv(p) for p in (out/'oos').glob('*.csv')])
    aggregate=data.groupby(['lane','span','variant','arm'])[['n','sse']].sum()
    aggregate['monthly_rmse_pct']=100*np.sqrt(aggregate.sse/aggregate.n)
    aggregate.to_csv(out/'oos_summary.csv')


def main():
    """Run an explicit stage and record immutable data and runtime provenance."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stage',choices=['endpoint','oos'],required=True)
    args=parser.parse_args();out=args.out.resolve()
    assert 'OneDrive' not in str(out);out.mkdir(parents=True,exist_ok=True)
    import factorlasso, cvxpy
    (out/'protocol.json').write_text(json.dumps(dict(variants=LABELS,solver='MOSEK',
        factorlasso_path=factorlasso.__file__,cvxpy_version=cvxpy.__version__,
        observation_manifest=f.sha(OLD/'inputs/manifest.json'),
        loss_normalization=f.get_cma_covar_estimation_spec().loss_normalization,reg_lambda=f.MAIN_LAMBDA,
        gate='date-level score sandwich; independent dates, no HAC',
        cutoff='2026-06-30',prior_policies=list(f.ARMS),threshold=1.,adaptive_floor=.5,
        spans=[60,36],frozen_data=str(OLD),clusters='fixed to old within each fit date',
        limitations=['retrospective fixed factor vintage','target rules selected earlier','no external carry validation']),indent=2),encoding='utf-8')
    (endpoint if args.stage=='endpoint' else oos)(out)


if __name__=='__main__':main()
