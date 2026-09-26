"""Independent reconstruction of every split-credit score and paired contrast."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .split_credit_research import ROOT,FACTORS,CREDIT,METRICS,PROFILES,data,draw,save,check,manifest


def audit_stage(name):
    """Recompute saved scores from frozen draws and coefficients, not scoring helpers."""
    check(ROOT/name);d=data();stage=ROOT/name
    a=pd.read_csv(stage/'asset_results.csv');p=pd.read_csv(stage/'panel_results.csv')
    summary=pd.read_csv(stage/'paired_summary.csv')
    credit=d['credit_indices'];keys=['profile','n','seed','method'];count=0
    max_gap=0.;start=time.perf_counter()
    expected=102*len(p)
    assert len(a)==expected and not a.duplicated(keys+['asset']).any()
    assert set(a.ticker)==set(d['tickers'])
    indexed=p.set_index(keys)
    # Independent references intentionally do not call score(), assemble() or paired().
    for (profile,n,seed),panel in a.groupby(keys[:3],sort=False):
        samples,truth,sigma,noise,variance=draw(d,profile,int(n),int(seed))
        (xt,yt),_,(xe,ye)=samples
        loaded=truth@np.linalg.cholesky(sigma)
        actual_cov=loaded@loaded.T+np.diag(noise)
        for method,raw in panel.groupby('method',sort=False):
            raw=raw.sort_values('asset');assert raw.asset.tolist()==list(range(102))
            b=raw[[f'beta{j}' for j in range(12)]].to_numpy()
            recorded_truth=raw[[f'truth{j}' for j in range(12)]].to_numpy()
            np.testing.assert_allclose(recorded_truth,truth,atol=1e-13)
            alpha=raw.intercept.to_numpy();error=b-truth
            training_residual=yt-xt@b.T-alpha
            estimated_cov=np.cov(xt@b.T,rowvar=False,ddof=1)+np.diag(np.mean(training_residual**2,axis=0))
            values=dict(beta_mse=np.mean(error**2,axis=1),
                credit_mse=np.mean(error[:,CREDIT]**2,axis=1),
                credit_ig_mse=error[:,2]**2,credit_hy_mse=error[:,3]**2,credit_em_mse=error[:,4]**2,
                scenario_mse=(error**2*np.diag(sigma)).sum(axis=1)/(12*variance),
                prediction_nmse=np.mean((ye-(xe@b.T+alpha))**2,axis=0)/variance,
                residual_var=np.mean(training_residual**2,axis=0))
            own=(b[credit,d['own_credit'][credit]]-truth[credit,d['own_credit'][credit]])**2
            np.testing.assert_allclose(raw.own_credit_mse.to_numpy()[credit],own,atol=1e-11)
            assert raw.is_credit.to_numpy().tolist()==np.isin(np.arange(102),credit).tolist()
            for metric,ref in values.items():
                max_gap=max(max_gap,float(np.max(abs(raw[metric].to_numpy()-ref))))
                np.testing.assert_allclose(raw[metric],ref,atol=1e-10,rtol=1e-10)
            scalar={k:float(v[credit].mean() if k.startswith('credit_') else v.mean()) for k,v in values.items() if k!='residual_var'}
            scalar.update(own_credit_mse=own.mean(),credit_prediction_nmse=values['prediction_nmse'][credit].mean(),
                full_covar_error=np.sqrt(np.sum((estimated_cov-actual_cov)**2)/np.sum(actual_cov**2)))
            refrow=indexed.loc[(profile,n,seed,method)]
            for metric,ref in scalar.items():np.testing.assert_allclose(refrow[metric],ref,atol=1e-10,rtol=1e-10)
            count+=1
    # Pivot independently establishes identical seed masks and paired contrasts.
    grouping=['profile','n']+(['setting'] if name.startswith('groups') else [])
    contrasts=0
    for idx,part in p.groupby(grouping):
        identity=dict(zip(grouping,idx));zero=identity.get('setting','')+'__zero' if 'setting' in identity else 'M1_zero'
        mask=np.logical_and.reduce([summary[k].eq(v) for k,v in identity.items()])
        saved=summary.loc[mask].set_index('method')
        for metric in METRICS:
            pivot=part.pivot(index='seed',columns='method',values=metric)
            assert not pivot.isna().any().any()
            for method in pivot:
                delta=(pivot[method]-pivot[zero]).to_numpy()
                mean=delta.sum()/len(delta);se=np.sqrt(np.sum((delta-mean)**2)/(len(delta)-1)/len(delta))
                np.testing.assert_allclose(saved.loc[method,[metric,metric+'_delta',metric+'_low',metric+'_high']].to_numpy(dtype=float),
                    [pivot[method].mean(),mean,mean-1.96*se,mean+1.96*se],atol=1e-12)
                contrasts+=1
    # Negative controls show missing rows and corrupted beta/metric contracts fail.
    rejected=[]
    try:assert len(a.iloc[:-1])==expected
    except AssertionError:rejected.append('missing_asset_row')
    altered=a.iloc[0].copy();altered.beta2+=.1
    try:np.testing.assert_allclose(altered.credit_ig_mse,(altered.beta2-altered.truth2)**2,atol=1e-11)
    except AssertionError:rejected.append('changed_credit_beta')
    assert len(rejected)==2
    out=ROOT/'metric_audit';out.mkdir(exist_ok=True)
    result=dict(stage=name,asset_rows=len(a),panel_method_scores=count,paired_metric_contrasts=contrasts,
        all_saved_metrics_reconstructed=True,max_asset_metric_gap=max_gap,negative_controls_rejected=rejected,
        seconds=time.perf_counter()-start)
    save(out/(name+'.json'),result);print(json.dumps(result,indent=2),flush=True)


def main():
    """Audit completed stages selected by explicit names."""
    parser=argparse.ArgumentParser();parser.add_argument('stages',nargs='+')
    for stage in parser.parse_args().stages:audit_stage(stage)


if __name__=='__main__':main()
