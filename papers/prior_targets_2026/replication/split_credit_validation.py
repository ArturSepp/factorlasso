"""Split-credit rolling replication and independent numerical audits."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import shutil
import time
import warnings
import numpy as np
import pandas as pd
from .split_credit_research import ROOT,HERE,FACTORS,CLASS,CREDIT,SETTINGS,METHODS,METRICS,PROFILES,data,draw,external_targets,restricted,save,manifest,check,digest,setup


def panels():
    """Load the frozen complete empirical panel and exact response metadata."""
    x=pd.read_csv(ROOT/'calibration/factor_returns.csv',index_col=0,parse_dates=True)
    y=pd.read_csv(ROOT/'calibration/response_returns.csv',index_col=0,parse_dates=True)
    uni=pd.read_csv(ROOT/'inputs/etf_universe.csv').set_index('ticker')
    assert x.columns.tolist()==FACTORS and 'Credit' not in x and x.index.equals(y.index)
    return x,y,uni


def targets(x,y,asset,subclass,span):
    """Use only the chosen instrument's past rows to construct each target."""
    from papers.prior_targets_2026.replication.robustness import fit_weighted
    fixed=np.zeros(12)
    if subclass in CLASS:
        j,value=CLASS[subclass];fixed[j]=value
        names4=['Rates',*FACTORS[2:5]];names5=['Equity',*names4]
    else:
        names4=names5=['Rates'] if asset in ['TLT','SHY'] else ['Equity']
    return fit_weighted(x,y,np.r_[np.geomspace(10,1e-6,36),0.],span,
        dict(economic=fixed,rates_credit=restricted(x,y,names4,span),equity_rates_credit=restricted(x,y,names5,span)))


def fit_at(x,y,asset,subclass,span,end):
    """Replicate the original 84-month rolling design with twelve regressors."""
    from papers.prior_targets_2026.replication.estimators import prediction_losses,select_index
    start=end-84;assert start>=0
    xt=x.iloc[start:end].to_numpy();yt=y.iloc[start:end].to_numpy()
    grid=np.r_[np.geomspace(10,1e-6,36),0.];losses={};cutoffs=[]
    for stop in [60,72]:
        paths=targets(xt[:stop],yt[:stop],asset,subclass,span)
        for name,path in paths.items():
            loss=prediction_losses(xt[stop:stop+12],yt[stop:stop+12],path['beta'],path['intercept'])
            losses[name]=losses.get(name,0.)+loss
        cutoffs.append(dict(span=span or 0,end=end,train_end=str(x.index[start+stop-1]),
            validation_start=str(x.index[start+stop]),validation_end=str(x.index[start+stop+11]),outer_fit_end=str(x.index[end-1])))
    final=targets(xt,yt,asset,subclass,span);chosen={}
    for method,path in final.items():
        k=len(grid)-1 if method=='M0_ols' else select_index(losses[method])
        chosen[method]=dict(beta=path['beta'][k],intercept=path['intercept'][k],target=path['target'],
            winner=path['winner'],lambda_index=int(k),reg_lambda=float(grid[k]),validation_mse=float(losses[method][k]/2))
    return chosen,cutoffs


def rolling_asset(asset,span):
    """Reconstruct each held-out month for one asset at one weighting span."""
    setup();x,y,uni=panels();subclass=uni.loc[asset,'sub_asset_class'];scale=float(y[asset].iloc[:60].var(ddof=0))
    rows=[];folds=[]
    for end in range(84,113):
        fit,cutoffs=fit_at(x,y[asset],asset,subclass,span,end)
        if asset=='LQD':folds+=cutoffs
        for method,result in fit.items():
            prediction=float(x.iloc[end]@result['beta']+result['intercept']);actual=float(y[asset].iloc[end])
            row=dict(lane='MATF_CUSTOM_IG_HY',span=span or 0,date=str(x.index[end].date()),asset=asset,
                method=method,cohort='credit' if subclass in CLASS else 'control',
                train_start=str(x.index[end-84]),train_end=str(x.index[end-1]),actual=actual,prediction=prediction,
                squared_error=(actual-prediction)**2,nmse=(actual-prediction)**2/scale,response_var=scale,
                winner=FACTORS[result['winner']],reg_lambda=result['reg_lambda'],lambda_index=result['lambda_index'],intercept=result['intercept'])
            for j in range(12):row.update({f'beta{j}':result['beta'][j],f'target{j}':result['target'][j]})
            rows.append(row)
    return rows,folds


def rolling(workers):
    """Run all 21 responses, two spans and 29 out-of-sample months."""
    setup();check(ROOT/'calibration');check(ROOT/'inputs');out=ROOT/'rolling'
    if out.exists():raise FileExistsError(out)
    out.mkdir();shutil.copy2(Path(__file__),out/'split_credit_validation.py')
    d=data();names=list(d['tickers'][d['credit_indices']])+['TLT','SHY','VGK','AAXJ']
    rows=[];folds=[];failures=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(rolling_asset,name,span):(name,span) for span in [None,36] for name in names}
        for future in as_completed(jobs):
            name,span=jobs[future]
            try:
                a,b=future.result();rows+=a;folds+=b
            except Exception as e:failures.append(dict(asset=name,span=span,error=repr(e)))
            print(f'rolling: {name} span={span}; failures={len(failures)}',flush=True)
    save(out/'failures.json',failures);assert not failures
    frame=pd.DataFrame(rows);frame.to_csv(out/'rolling_results.csv',index=False)
    pd.DataFrame(folds).to_csv(out/'fold_cutoffs.csv',index=False)
    from .rolling import bootstrap_summary
    config=dict(uncertainty=dict(mean_block_lengths=[3,6,12],replications=5000,seed=990001))
    boot=bootstrap_summary(frame,config);boot.to_csv(out/'paired_bootstrap.csv',index=False)
    frame.groupby(['span','cohort','method']).nmse.mean().rename('mean_nmse').to_csv(out/'method_summary.csv')
    assert len(frame)==21*2*29*8 and not frame.duplicated(['span','date','asset','method']).any()
    x,y,uni=panels();X=x.reindex(pd.to_datetime(frame.date)).to_numpy()
    b=frame[[f'beta{j}' for j in range(12)]].to_numpy()
    manual=(X*b).sum(axis=1)+frame.intercept.to_numpy()
    np.testing.assert_allclose(manual,frame.prediction,atol=1e-12)
    np.testing.assert_allclose(frame.nmse,(frame.actual-manual)**2/frame.response_var,atol=1e-12)
    assert (pd.to_datetime(frame.train_end)<pd.to_datetime(frame.date)).all()
    cut=pd.DataFrame(folds)
    assert (pd.to_datetime(cut.train_end)<pd.to_datetime(cut.validation_start)).all()
    assert (pd.to_datetime(cut.validation_end)<=pd.to_datetime(cut.outer_fit_end)).all()
    mutation=0
    for span in [None,36]:
        for end in [84,98,112]:
            base,_=fit_at(x,y.LQD,'LQD',uni.loc['LQD','sub_asset_class'],span,end)
            fx=x.copy();fy=y.LQD.copy();fx.iloc[end:]+=1000;fy.iloc[end:]-=500
            altered,_=fit_at(fx,fy,'LQD',uni.loc['LQD','sub_asset_class'],span,end)
            truncated,_=fit_at(x.iloc[:end],y.LQD.iloc[:end],'LQD',uni.loc['LQD','sub_asset_class'],span,end)
            for method in base:
                np.testing.assert_allclose(base[method]['beta'],altered[method]['beta'],atol=1e-12)
                np.testing.assert_allclose(base[method]['beta'],truncated[method]['beta'],atol=1e-12)
                assert base[method]['lambda_index']==altered[method]['lambda_index']==truncated[method]['lambda_index']
            mutation+=1
    receipt=dict(rows=len(frame),factors=FACTORS,dates=29,assets=21,methods=8,
        first='2024-02',last='2026-06',future_mutation_and_truncation_cases=mutation,
        independent_prediction_max_gap=float(np.max(abs(manual-frame.prediction))),failures=0,
        seconds=time.perf_counter()-started)
    save(out/'verification.json',receipt);manifest(out);print(json.dumps(receipt,indent=2),flush=True)


def numerical():
    """Cross-check L1 paths against package CVXPY and independent residual formulas."""
    d=data()
    from factorlasso import solve_lasso_cvx_problem
    from .estimators import prepare,fit_paths,prediction_losses,select_index
    out=ROOT/'numerical_audit';out.mkdir(exist_ok=True);rows=[]
    for profile in PROFILES:
        samples,truth,_,_,_=draw(d,profile,112,949999)
        (xt,yt),(xv,yv),_=samples
        for i in [int(d['credit_indices'][0]),int(d['credit_indices'][6]),int(d['credit_indices'][11])]:
            grid=np.r_[np.geomspace(10,1e-6,36),0.]
            paths=fit_paths(xt,yt[:,i],grid,external_targets=external_targets(xt,yt[:,i],d,i,truth[i],949999,112))
            z,v,_,_,xs,ys=prepare(xt,yt[:,i])
            for method,path in paths.items():
                loss=prediction_losses(xv,yv[:,i],path['beta'],path['intercept'])
                direct=np.mean((yv[:,i,None]-xv@path['beta'].T-path['intercept'])**2,axis=0)
                np.testing.assert_allclose(loss,direct,atol=1e-12)
                k=len(grid)-1 if method=='M0_ols' else select_index(loss)
                estimated=path['beta'][k]*xs/ys
                if method=='M5_post_lasso':
                    support=np.flatnonzero(abs(paths['M1_zero']['beta'][k]*xs/ys)>1e-7)
                    ref=np.zeros(12)
                    if len(support):ref[support]=np.linalg.lstsq(z[:,support],v,rcond=None)[0]
                else:
                    weights=np.ones((1,12))
                    if method=='M4_free_winner':weights[0,path['winner']]=0
                    result=solve_lasso_cvx_problem(z,v[:,None],reg_lambda=float(grid[k]),solver='CLARABEL',
                        factors_beta_prior=(path['target']*xs/ys)[None],penalty_weights=weights)
                    ref=result.estimated_beta[0]
                gap=float(np.max(abs(estimated-ref)));assert gap<3e-4,(profile,method,gap)
                rows.append(dict(profile=profile,asset=i,method=method,maximum_standardised_beta_gap=gap))
    pd.DataFrame(rows).to_csv(out/'l1_package_references.csv',index=False)
    save(out/'l1_verification.json',dict(comparisons=len(rows),max_gap=max(r['maximum_standardised_beta_gap'] for r in rows),tolerance=3e-4,prediction_losses_direct=True))
    print(json.dumps(json.loads((out/'l1_verification.json').read_text()),indent=2))


def group_audit():
    """Independently refit pilot selections and every warned/fallback confirmation point."""
    d=data();out=ROOT/'numerical_audit';out.mkdir(exist_ok=True)
    from .groups import model_for
    from .estimators import select_index
    rows=[];checks=[];grid=np.r_[np.geomspace(10,1e-6,15),0.]
    for phase in ['pilot','confirmation']:
        stage=ROOT/f'groups_{phase}';check(stage)
        panels=pd.read_csv(stage/'panel_results.csv');assets=pd.read_csv(stage/'asset_results.csv')
        events=pd.read_csv(stage/'solver_events.csv');paths=pd.read_csv(stage/'validation_paths.csv')
        for (profile,seed,setting,target),event in events.groupby(['profile','seed','setting','target']):
            selected=panels.loc[panels.profile.eq(profile)&panels.seed.eq(seed)&panels.setting.eq(setting)&panels.target.eq(target)].iloc[0]
            candidates=set(event.loc[event.status.ne('optimal')|event.fallback,'lambda_index'].astype(int))
            if phase=='pilot' and seed==960000:candidates.add(int(selected.lambda_index))
            if not candidates:continue
            samples,_,_,_,_=draw(d,profile,112,int(seed));(x,y),(xv,yv),_=samples
            xm,ym=x.mean(0),y.mean(0);xs,ys=x.std(0),y.std(0)
            z=pd.DataFrame((x-xm)/xs,columns=FACTORS);v=pd.DataFrame((y-ym)/ys,columns=d['tickers'])
            prior=pd.DataFrame(d['prior']*xs[None]/ys[:,None],index=d['tickers'],columns=FACTORS)
            groups=pd.Series(d['groups'],index=d['tickers'])
            spec=next(s for s in SETTINGS if s['name']==setting)
            track=paths.loc[paths.profile.eq(profile)&paths.seed.eq(seed)&paths.setting.eq(setting)&paths.target.eq(target)].sort_values('lambda_index')
            revised=track.validation_nmse.to_numpy().copy()
            for k in candidates:
                model=model_for(spec,target,prior,groups,solver='ECOS');model.reg_lambda=float(grid[k])
                with warnings.catch_warnings(record=True) as caught:model.fit(z,v)
                b=model.coef_.to_numpy()*ys[:,None]/xs[None];alpha=ym+model.alpha_const_.to_numpy()*ys-b@xm
                revised[k]=np.mean(np.mean((yv-xv@b.T-alpha)**2,axis=0)/y.var(0))
                gap=None
                if k==int(selected.lambda_index):
                    saved=assets.loc[assets.profile.eq(profile)&assets.seed.eq(seed)&assets.setting.eq(setting)&assets.target.eq(target)].sort_values('asset')
                    gap=float(np.max(abs((saved[[f'beta{j}' for j in range(12)]].to_numpy()-b)*xs[None]/ys[:,None])))
                    assert gap<5e-4,(phase,profile,seed,setting,target,gap)
                rows.append(dict(phase=phase,profile=profile,seed=int(seed),setting=setting,target=target,lambda_index=k,selected=k==int(selected.lambda_index),standardised_beta_gap=gap,warnings=[str(w.message) for w in caught]))
            k2=select_index(revised)
            checks.append(dict(phase=phase,profile=profile,seed=int(seed),setting=setting,target=target,original=int(selected.lambda_index),refreshed=k2))
            assert k2==int(selected.lambda_index),checks[-1]
    save(out/'group_solver_checks.json',rows);save(out/'group_selection_checks.json',checks)
    receipt=dict(refits=len(rows),selected_refits=sum(r['selected'] for r in rows),max_selected_beta_gap=max(r['standardised_beta_gap'] or 0 for r in rows),all_selection_choices_preserved=True)
    save(out/'group_verification.json',receipt);manifest(out);print(json.dumps(receipt,indent=2))


def main():
    """Run a named independent audit or empirical replacement stage."""
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['rolling','numerical','group_audit']);p.add_argument('--workers',type=int,default=2);args=p.parse_args()
    if args.stage=='rolling':rolling(args.workers)
    elif args.stage=='numerical':numerical()
    else:group_audit()


if __name__=='__main__':main()
