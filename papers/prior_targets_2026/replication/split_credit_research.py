"""Replacement prior-target experiments using only MATF_CUSTOM_IG_HY factors.

The earlier research adapters implement standardised target-centred L1 and the
exported FactorLasso group path. Reuse these generic solvers; regenerate every
model-dependent calibration, target, metric and empirical path in this module.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import warnings
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=Path(r'C:\Users\artur\AppData\Local\AgentWork\ARTURDESKTOP\FactorLasso\outputs\prior_targets_2026\20260922_split_credit_v3')
OLD=ROOT.parent/'20260921_evidence_v1'
FI=Path(r'C:\Users\artur\AppData\Local\AgentWork\ARTURDESKTOP\Rosaa\analyses')
FACTORS=['Equity','Rates','Credit IG US','Credit HY US','Credit EM','Carry G10','Carry EM','Inflation','Commodities','Private Equity','Rates Vol','Fx']
CREDIT=np.array([2,3,4])
CLASS={'Global IG Bonds':(2,.2),'Global HY Bonds':(3,.4),'EM Bonds':(4,.3)}
METHODS=['M0_ols','M1_zero','M2_auto','economic','economic_noisy','rates_credit','equity_rates_credit','oracle','M4_free_winner','M5_post_lasso']
METRICS=['credit_mse','credit_ig_mse','credit_hy_mse','credit_em_mse','own_credit_mse','beta_mse','scenario_mse','prediction_nmse','credit_prediction_nmse','full_covar_error']
PROFILES=['baseline','zero_credit_exposure','rates_dominant']
SETTINGS=[dict(name='HCGL',geometry='HCGL',signs=False,adaptive=False,known=False),
 dict(name='HCGL_sign',geometry='HCGL',signs=True,adaptive=False,known=False),
 dict(name='HCGL_sign_adaptive',geometry='HCGL',signs=True,adaptive=True,known=False),
 dict(name='FCGL_sign_adaptive',geometry='FCGL',signs=True,adaptive=True,known=False),
 dict(name='Supplied_groups_sign_adaptive',geometry='GROUP',signs=True,adaptive=True,known=True)]


def digest(p):
    """Hash an immutable input or output."""
    return sha256(Path(p).read_bytes()).hexdigest()


def save(p,obj):
    """Save explicit UTF-8 JSON."""
    Path(p).write_text(json.dumps(obj,indent=2,default=str,allow_nan=True)+'\n',encoding='utf-8')


def manifest(p):
    """Freeze every file in a completed stage."""
    save(p/'manifest.json',{str(f.relative_to(p)):digest(f) for f in p.rglob('*') if f.is_file() and f.name!='manifest.json'})


def check(p):
    """Require every frozen stage file to match its recorded hash."""
    for name,value in json.loads((p/'manifest.json').read_text()).items():assert digest(p/name)==value,(p,name)


def setup():
    """Use the frozen estimator source before importing any research adapter."""
    sys.path.insert(0,str(ROOT/'source/src'))


def data():
    """Read the calibrated twelve-factor arrays."""
    setup()
    with np.load(ROOT/'calibration/calibration.npz') as f:return {k:f[k].copy() for k in f.files}


def init():
    """Freeze the single factor specification and calibrate without using a prior."""
    if ROOT.exists():raise FileExistsError(ROOT)
    ROOT.mkdir(parents=True);src=ROOT/'source';src.mkdir()
    shutil.copytree(HERE.parents[2]/'src/factorlasso',src/'src/factorlasso',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['split_credit_research.py','estimators.py','robustness.py','groups.py','rolling.py']:
        shutil.copy2(HERE/name,src/name)
    inp=ROOT/'inputs';inp.mkdir()
    inputs={'factor_prices_monthly_q2.csv':FI/'fi_prior_dataset_20260922_v3/factor_prices_monthly_q2.csv',
        'etf_excess_logreturns.csv':OLD/'source_snapshot/data/etf_excess_logreturns.csv',
        'etf_universe.csv':OLD/'source_snapshot/data/etf_universe.csv'}
    for name,p in inputs.items():shutil.copy2(p,inp/name)
    config=dict(factors=FACTORS,generic_credit_absent=True,model='MATF_CUSTOM_IG_HY',profiles=PROFILES,
        calibration='OLS with intercept on common 113 months; artificial fixed simulation truth, not economic ground truth',
        sizes=[60,112,240],test_size=2000,l1_methods=METHODS,l1_grid=list(np.r_[np.geomspace(10,1e-6,36),0.]),
        group_settings=SETTINGS,group_grid=list(np.r_[np.geomspace(10,1e-6,15),0.]),
        l1_pilot_seeds=[940000,940004],l1_confirmation_seeds=[950000,950049],
        group_pilot_seeds=[960000,960004],group_confirmation_seeds=[970000,970049],
        credit_metric='Mean squared native beta error over 17 cohort assets and all three credit factors; component errors also retained',
        source_input_hashes={str(p):digest(p) for p in inputs.values()},
        rolling='21 ETFs, uniform/span36,84-month fit,60/72 inner trains each next12 validate,29 outer months; same pre-existing design; no proxy lane')
    save(ROOT/'protocol.json',config);manifest(src);manifest(inp)
    prices=pd.read_csv(inp/'factor_prices_monthly_q2.csv',index_col=0,parse_dates=True)
    assert prices.columns.tolist()==FACTORS and 'Credit' not in prices
    y=pd.read_csv(inp/'etf_excess_logreturns.csv',index_col=0,parse_dates=True)
    x=np.log(prices/prices.shift(1)).reindex(y.index).dropna();y=y.loc[x.index]
    assert x.shape==(113,12) and y.shape==(113,102)
    uni=pd.read_csv(inp/'etf_universe.csv').set_index('ticker').reindex(y.columns)
    design=np.column_stack([np.ones(len(x)),x]);b=np.linalg.lstsq(design,y,rcond=None)[0]
    xc=x-x.mean();yc=y-y.mean();reference=np.linalg.solve(xc.T@xc,xc.T@yc)
    np.testing.assert_allclose(reference,b[1:],atol=1e-11)
    sigma=x.cov().to_numpy()*12;resid=y.to_numpy()-design@b
    noise=np.mean(resid**2,axis=0)*12;assert np.all(noise>0)
    assert np.linalg.matrix_rank(xc)==12
    np.testing.assert_allclose(np.linalg.cholesky(sigma)@np.linalg.cholesky(sigma).T,sigma,atol=1e-14)
    prior=np.zeros((102,12));own=np.full(102,-1);cohort=[]
    for i,cl in enumerate(uni.sub_asset_class):
        if cl in CLASS:
            j,value=CLASS[cl];prior[i,j]=value;own[i]=j;cohort.append(i)
    assert len(cohort)==17
    out=ROOT/'calibration';out.mkdir()
    x.to_csv(out/'factor_returns.csv');y.to_csv(out/'response_returns.csv')
    np.savez(out/'calibration.npz',beta=b[1:].T,sigma_annual=sigma,residual_var_annual=noise,
        prior=prior,credit_indices=np.array(cohort),own_credit=own,
        tickers=np.array(y.columns,dtype=str),factors=np.array(FACTORS),groups=uni.sub_asset_class.to_numpy(dtype=str))
    pd.DataFrame(b[1:].T,index=y.columns,columns=FACTORS).to_csv(out/'calibration_betas.csv')
    x.corr().to_csv(out/'factor_correlations.csv')
    save(out/'summary.json',dict(observations=113,assets=102,factors=FACTORS,credit_cohort=list(y.columns[cohort]),
        start=str(x.index[0]),end=str(x.index[-1]),condition=float(np.linalg.cond(x.corr())),
        augmented_centred_ols_max_gap=float(np.max(abs(np.asarray(reference)-b[1:]))),
        min_residual_annual_variance=float(noise.min()),source_hash=digest(ROOT/'protocol.json')))
    with np.load(out/'calibration.npz') as verify:
        assert all(verify[k].dtype!=object for k in verify.files)
    manifest(out)
    print(json.dumps(json.loads((out/'summary.json').read_text()),indent=2),flush=True)


def draw(d,profile,n,seed):
    """Draw paired independent train, validation and test samples in twelve dimensions."""
    beta=d['beta'].copy();cohort=d['credit_indices']
    if profile=='zero_credit_exposure':beta[np.ix_(cohort,CREDIT)]=0.
    elif profile=='rates_dominant':beta[cohort,1]*=4
    else:assert profile=='baseline'
    sigma=d['sigma_annual']/12;noise=d['residual_var_annual']/12
    variance=np.einsum('ij,jk,ik->i',beta,sigma,beta)+noise
    samples=[]
    for stream,size in enumerate([n,n,2000]):
        rng=np.random.default_rng(np.random.SeedSequence([seed,n,stream]))
        x=rng.normal(size=(size,12))@np.linalg.cholesky(sigma).T
        y=x@beta.T+rng.normal(size=(size,len(beta)))*np.sqrt(noise)
        samples.append((x,y))
    return samples,beta,sigma,noise,variance


def restricted(x,y,names,span=None):
    """Embed a train-only four/five-factor regression in the complete factor vector."""
    from papers.prior_targets_2026.replication.rolling import restricted_target
    return restricted_target(x,y,[FACTORS.index(name) for name in names],span)


def external_targets(x,y,d,i,truth,seed,n):
    """Specify class-specific economic and restricted targets before tuning."""
    econ=d['prior'][i];noisy=econ.copy();own=int(d['own_credit'][i])
    if own>=0:
        rng=np.random.default_rng(np.random.SeedSequence([seed,n,3,i]));noisy[own]+=rng.normal(scale=.1)
    small4=restricted(x,y,['Rates',*FACTORS[2:5]]) if own>=0 else np.zeros(12)
    small5=restricted(x,y,['Equity','Rates',*FACTORS[2:5]]) if own>=0 else np.zeros(12)
    return dict(economic=econ,economic_noisy=noisy,rates_credit=small4,equity_rates_credit=small5,oracle=truth)


def score(d,profile,n,seed,method,beta,alpha,truth,sigma,noise,variance,samples,selection):
    """Save all loading components and check covariance assembly independently."""
    from papers.prior_targets_2026.replication.multiasset import assemble
    (xt,yt),_,(xe,ye)=samples
    error=beta-truth;test=np.mean((ye-xe@beta.T-alpha)**2,axis=0)
    residual=np.mean((yt-xt@beta.T-alpha)**2,axis=0)
    cov=assemble(beta,np.cov(xt,rowvar=False,ddof=1),residual,d)
    cov_true=assemble(truth,sigma,noise,d)
    assets=[]
    for i,ticker in enumerate(d['tickers']):
        row=dict(profile=profile,n=n,seed=seed,method=method,asset=i,ticker=ticker,is_credit=i in d['credit_indices'],
            credit_mse=float(np.mean(error[i,CREDIT]**2)),credit_ig_mse=error[i,2]**2,
            credit_hy_mse=error[i,3]**2,credit_em_mse=error[i,4]**2,
            own_credit_mse=error[i,int(d['own_credit'][i])]**2 if d['own_credit'][i]>=0 else np.nan,
            beta_mse=float(np.mean(error[i]**2)),
            scenario_mse=float(np.mean(error[i]**2*np.diag(sigma))/variance[i]),
            prediction_nmse=test[i]/variance[i],intercept=alpha[i],residual_var=residual[i],**selection[i])
        for j in range(12):row.update({f'beta{j}':beta[i,j],f'truth{j}':truth[i,j]})
        assets.append(row)
    frame=pd.DataFrame(assets);credit=frame.loc[frame.is_credit]
    row=dict(profile=profile,n=n,seed=seed,method=method,
        **{m:float(credit[m].mean()) for m in ['credit_mse','credit_ig_mse','credit_hy_mse','credit_em_mse','own_credit_mse']},
        **{m:float(frame[m].mean()) for m in ['beta_mse','scenario_mse','prediction_nmse']},
        credit_prediction_nmse=float(credit.prediction_nmse.mean()),
        full_covar_error=float(np.linalg.norm(cov-cov_true)/np.linalg.norm(cov_true)))
    return assets,row


def l1_job(profile,n,seed):
    """Fit all matched per-asset L1 targets on a complete paired panel."""
    d=data()
    from papers.prior_targets_2026.replication.estimators import fit_paths,prediction_losses,select_index
    samples,truth,sigma,noise,variance=draw(d,profile,n,seed)
    (xt,yt),(xv,yv),_=samples;grid=np.r_[np.geomspace(10,1e-6,36),0.]
    fitted={m:np.zeros_like(truth) for m in METHODS};alphas={m:np.zeros(len(truth)) for m in METHODS}
    selected={m:[] for m in METHODS}
    for i in range(len(truth)):
        paths=fit_paths(xt,yt[:,i],grid,external_targets=external_targets(xt,yt[:,i],d,i,truth[i],seed,n))
        assert set(paths)==set(METHODS)
        for method,path in paths.items():
            losses=prediction_losses(xv,yv[:,i],path['beta'],path['intercept'])
            k=len(grid)-1 if method=='M0_ols' else select_index(losses)
            fitted[method][i]=path['beta'][k];alphas[method][i]=path['intercept'][k]
            entry=dict(reg_lambda=float(grid[k]),lambda_index=int(k),validation_mse=float(losses[k]),winner=int(path['winner']),dual_gap=float(path['gap']))
            entry.update({f'target{j}':path['target'][j] for j in range(12)})
            selected[method].append(entry)
    assets=[];panels=[]
    for method in METHODS:
        a,p=score(d,profile,n,seed,method,fitted[method],alphas[method],truth,sigma,noise,variance,samples,selected[method])
        assets+=a;panels.append(p)
    return dict(assets=assets,panels=panels,warnings=[],solvers=[])


def group_job(profile,n,seed):
    """Fit five existing geometry/sign ablations using the twelve-factor DGP."""
    d=data()
    from papers.prior_targets_2026.replication.groups import fit_paths
    from papers.prior_targets_2026.replication.estimators import select_index
    samples,truth,sigma,noise,variance=draw(d,profile,n,seed)
    (xt,yt),(xv,yv),_=samples;grid=np.r_[np.geomspace(10,1e-6,15),0.]
    assets=[];panels=[];messages=[];events_all=[];paths=[]
    for setting in SETTINGS:
        for target in ['zero','auto','economic']:
            models,betas,intercepts,raw,effective,caught,events=fit_paths(xt,yt,d,setting,target,grid)
            val=np.mean((yv[None]-np.einsum('tf,kaf->kta',xv,betas)-intercepts[:,None])**2,axis=1)
            loss=np.mean(val/yt.var(axis=0)[None],axis=1);k=select_index(loss)
            select=[]
            for i in range(len(truth)):
                entry=dict(setting=setting['name'],target=target,reg_lambda=float(grid[k]),lambda_index=int(k),
                    validation_mse=float(loss[k]),cluster=str(models[k].clusters_.iloc[i]))
                for j in range(12):entry.update({f'target{j}':effective[k,i,j],f'raw_target{j}':raw[k,i,j]})
                select.append(entry)
            a,p=score(d,profile,n,seed,setting['name']+'__'+target,betas[k],intercepts[k],truth,sigma,noise,variance,samples,select)
            p.update(setting=setting['name'],target=target,reg_lambda=float(grid[k]),lambda_index=int(k),selected_solver=events[k]['final_solver'],warning_count=len(caught))
            assets+=a;panels.append(p)
            messages += [dict(profile=profile,seed=seed,setting=setting['name'],target=target,message=w) for w in caught]
            events_all += [dict(profile=profile,seed=seed,setting=setting['name'],target=target,**e) for e in events]
            paths += [dict(profile=profile,seed=seed,setting=setting['name'],target=target,lambda_index=k,reg_lambda=float(lam),validation_nmse=float(loss[k])) for k,lam in enumerate(grid)]
    return dict(assets=assets,panels=panels,warnings=messages,solvers=events_all,paths=paths)


def paired(panels,group=False):
    """Average paired differences over simulation seeds, never over asset rows."""
    rows=[];keys=['profile','n']+(['setting'] if group else [])
    for index,block in panels.groupby(keys):
        key=dict(zip(keys,index));base_name=key['setting']+'__zero' if group else 'M1_zero'
        base=block.loc[block.method.eq(base_name)].set_index('seed')
        for method,part in block.groupby('method'):
            part=part.set_index('seed').sort_index();base=base.sort_index();assert part.index.equals(base.index)
            row=dict(**key,method=method,seeds=len(part))
            for metric in METRICS:
                delta=part[metric]-base[metric];se=delta.std(ddof=1)/np.sqrt(len(delta))
                row.update({metric:float(part[metric].mean()),metric+'_delta':float(delta.mean()),
                    metric+'_low':float(delta.mean()-1.96*se),metric+'_high':float(delta.mean()+1.96*se)})
            rows.append(row)
    return pd.DataFrame(rows)


def validate_stage(out,kind,phase):
    """Recompute saved errors, aggregate rows and fail on missing simulation cells."""
    d=data();a=pd.read_csv(out/'asset_results.csv');p=pd.read_csv(out/'panel_results.csv')
    nseed=5 if phase=='pilot' else 50
    nprofile=2 if kind=='groups' else 3;nsizes=1 if kind=='groups' else 3;nmethod=15 if kind=='groups' else 10
    assert len(p)==nseed*nprofile*nsizes*nmethod and len(a)==len(p)*102
    keys=['profile','n','seed','method'];assert not p.duplicated(keys).any() and not a.duplicated(keys+['asset']).any()
    assert a.groupby(keys).asset.nunique().eq(102).all()
    errors=a[[f'beta{j}' for j in range(12)]].to_numpy()-a[[f'truth{j}' for j in range(12)]].to_numpy()
    np.testing.assert_allclose(a.credit_mse,np.mean(errors[:,CREDIT]**2,axis=1),atol=1e-11)
    np.testing.assert_allclose(a.beta_mse,np.mean(errors**2,axis=1),atol=1e-11)
    for j,name in zip(CREDIT,['credit_ig_mse','credit_hy_mse','credit_em_mse']):
        np.testing.assert_allclose(a[name],errors[:,j]**2,atol=1e-11)
    cohort=a.loc[a.is_credit]
    agg=cohort.groupby(keys).credit_mse.mean();np.testing.assert_allclose(agg,p.set_index(keys).loc[agg.index,'credit_mse'],atol=1e-11)
    assert np.isfinite(p[METRICS]).all().all()
    if kind=='groups':
        events=pd.read_csv(out/'solver_events.csv');assert len(events)==len(p)*16
        assert events.status.isin(['optimal','optimal_inaccurate']).all()
    reject=False
    try:np.testing.assert_allclose(a.credit_mse.to_numpy()+.01,np.mean(errors[:,CREDIT]**2,axis=1),atol=1e-11)
    except AssertionError:reject=True
    assert reject
    table=paired(p,kind=='groups');table.to_csv(out/'paired_summary.csv',index=False)
    return dict(asset_rows=len(a),panel_method_rows=len(p),panels=nseed*nprofile*nsizes,factors=FACTORS,
        all_credit_components_verified=True,independent_cohort_aggregation=True,corrupted_metric_rejected=True)


def run_stage(kind,phase,workers):
    """Execute every frozen panel with resumable, individually recorded chunks."""
    check(ROOT/'source');check(ROOT/'inputs');check(ROOT/'calibration')
    for p in (ROOT/'source').glob('*.py'):
        assert digest(p)==digest(HERE/p.name),f'Research helper drift: {p.name}'
    if phase=='confirmation':check(ROOT/f'{kind}_pilot')
    out=ROOT/f'{kind}_{phase}'
    if (out/'manifest.json').exists():raise FileExistsError(out)
    out.mkdir(exist_ok=True);(out/'chunks').mkdir(exist_ok=True)
    start=time.perf_counter();begin=(960000 if kind=='groups' else 940000)+(10000 if phase=='confirmation' else 0)
    seeds=range(begin,begin+(50 if phase=='confirmation' else 5))
    profiles=PROFILES[:2] if kind=='groups' else PROFILES;sizes=[112] if kind=='groups' else [60,112,240]
    job=group_job if kind=='groups' else l1_job;failures=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs={}
        for profile in profiles:
            for n in sizes:
                for seed in seeds:
                    file=out/'chunks'/f'{profile}_{n}_{seed}.json'
                    if not file.exists():jobs[pool.submit(job,profile,n,seed)]=(file,profile,n,seed)
        for future in as_completed(jobs):
            file,profile,n,seed=jobs[future]
            try:save(file,future.result())
            except Exception as error:
                failures.append(dict(profile=profile,n=n,seed=seed,error=repr(error),trace=traceback.format_exc()))
            print(f'{kind}/{phase}: {profile}, n={n}, seed={seed}, failures={len(failures)}',flush=True)
    save(out/'failures.json',failures)
    if failures:raise RuntimeError(f'{len(failures)} failed panels; retained failure records')
    merged={k:[] for k in ['assets','panels','warnings','solvers','paths']}
    for file in sorted((out/'chunks').glob('*.json')):
        obj=json.loads(file.read_text())
        for key in merged:merged[key]+=obj.get(key,[])
    for key,name in [('assets','asset_results'),('panels','panel_results'),('solvers','solver_events'),('paths','validation_paths')]:
        if merged[key]:pd.DataFrame(merged[key]).to_csv(out/f'{name}.csv',index=False)
    save(out/'warnings.json',merged['warnings'])
    receipt=validate_stage(out,kind,phase);receipt.update(seconds=time.perf_counter()-start,warnings=len(merged['warnings']),failures=0)
    save(out/'verification.json',receipt);manifest(out);print(json.dumps(receipt,indent=2),flush=True)


def main():
    """Dispatch an explicitly bounded frozen research stage."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['init','l1','groups'])
    parser.add_argument('--phase',choices=['pilot','confirmation'],default='pilot')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.stage=='init':init()
    else:run_stage(args.stage,args.phase,args.workers)


if __name__=='__main__':main()
