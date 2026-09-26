"""Independently audit weighted targets and date-block summaries on the new panel."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .split_credit_research import ROOT,FACTORS,CLASS,save
from .split_credit_validation import panels


def main():
    """Rebuild target coefficients and bootstrap intervals without fitting helpers."""
    x,y,uni=panels()
    frame=pd.read_csv(ROOT/'rolling/rolling_results.csv')
    target_checks=0;max_gap=0.
    for (asset,span,date),part in frame.groupby(['asset','span','date']):
        end=x.index.get_loc(pd.Timestamp(date))
        xt=x.iloc[end-84:end].to_numpy();yt=y[asset].iloc[end-84:end].to_numpy()
        age=np.arange(84)[::-1];raw=np.ones(84) if span==0 else ((span-1)/(span+1))**age
        w=raw/raw.sum();xm=w@xt;ym=w@yt;xc=xt-xm;yc=yt-ym
        variance_x=np.sum(w[:,None]*xc**2,axis=0);variance_y=np.sum(w*yc**2)
        cov=np.sum(w[:,None]*xc*yc[:,None],axis=0);r2=cov**2/(variance_x*variance_y)
        winner=int(r2.argmax());auto=np.zeros(12);auto[winner]=cov[winner]/variance_x[winner]
        cl=uni.loc[asset,'sub_asset_class'];econ=np.zeros(12)
        if cl in CLASS:
            j,val=CLASS[cl];econ[j]=val
            restrictions={'rates_credit':[1,2,3,4],'equity_rates_credit':[0,1,2,3,4]}
        else:
            small=[1] if asset in ['TLT','SHY'] else [0]
            restrictions={'rates_credit':small,'equity_rates_credit':small}
        expected={'M2_auto':auto,'economic':econ}
        for method,idx in restrictions.items():
            design=xc[:,idx];b=np.linalg.solve(design.T@(w[:,None]*design),design.T@(w*yc))
            target=np.zeros(12);target[idx]=b;expected[method]=target
        for _,row in part.iterrows():
            assert row.winner==FACTORS[winner]
            ref=expected.get(row.method,np.zeros(12))
            actual=row[[f'target{j}' for j in range(12)]].to_numpy(dtype=float)
            max_gap=max(max_gap,float(np.max(abs(ref-actual))))
            np.testing.assert_allclose(actual,ref,atol=1e-10)
            target_checks+=1
    # Replay the saved RNG with an independent last-restart vectorisation.
    boot=pd.read_csv(ROOT/'rolling/paired_bootstrap.csv')
    comparisons=0
    for (span,cohort),part in frame.groupby(['span','cohort']):
        months=part.groupby(['date','method']).nmse.mean().unstack().sort_index()
        assert months.shape==(29,8)
        for block in [3,6,12]:
            rng=np.random.default_rng(990001+block)
            starts=rng.integers(0,29,size=(5000,29))
            restart=rng.random((5000,29))<1/block;restart[:,0]=True
            t=np.arange(29)[None,:]
            last=np.maximum.accumulate(np.where(restart,t,0),axis=1)
            indices=(np.take_along_axis(starts,last,axis=1)+t-last)%29
            for method in months:
                diff=(months[method]-months.M1_zero).to_numpy()
                interval=np.percentile(diff[indices].mean(axis=1),[2.5,97.5])
                saved=boot.loc[boot.span.eq(span)&boot.cohort.eq(cohort)&boot.method.eq(method)&boot.block.eq(block)].iloc[0]
                np.testing.assert_allclose([saved.paired_delta,saved.bootstrap_low,saved.bootstrap_high],[diff.mean(),*interval],atol=1e-12)
                comparisons+=1
    result=dict(target_vectors_checked=target_checks,max_native_target_gap=max_gap,
        independent_stationary_bootstrap_comparisons=comparisons,highest_r2_and_class_targets_verified=True)
    save(ROOT/'metric_audit/rolling_targets_bootstrap.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
