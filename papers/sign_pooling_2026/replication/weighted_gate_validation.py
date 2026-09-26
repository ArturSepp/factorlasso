"""Independent replication-level validation of the revised date-score sign gate."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from factorlasso.sign_constraints import _compute_sign_vector


def run(out,reps=500):
    """Compare null gates and active recovery, with separate threshold calibration draws."""
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for case in ('independent','overlapping','duplicate','unequal_histories','heteroskedastic','regime_change'):
        for span in (None,60.):
            for rep in range(reps):
                rng=np.random.default_rng(20260925+rep)
                x=rng.normal(size=(240,2));common=rng.normal(size=(240,1))
                rho=0. if case=='independent' else 1. if case=='duplicate' else .8
                noise=np.sqrt(rho)*common+np.sqrt(1-rho)*rng.normal(size=(240,8))
                if case=='heteroskedastic':noise*=np.exp(.8*x[:,0,None])
                beta=np.full((240,1),.15)
                if case=='regime_change':beta[:160]=.3;beta[160:]=-.3
                y=beta*x[:,:1]+noise
                if case=='unequal_histories':
                    for k in range(8):y[:k*24,k]=np.nan
                    x[::11,1]=np.nan
                truth=-1. if case=='regime_change' and span is not None else 1.
                for gate in ('independent','date'):
                    _,slopes,d=_compute_sign_vector(x,y,ewma_span=span,variance_estimator=gate,return_diagnostics=True)
                    rows.append(dict(case=case,span=0 if span is None else span,rep=rep,gate=gate,
                        active_t=float(d['t_stats'][0]),null_t=float(d['t_stats'][1]),
                        active_correct=bool(np.sign(slopes[0])==truth),
                        null_effective_dates=float(d['effective_n'][1])))
            print('gate MC',case,span,reps,'paired draws',flush=True)
    draws=pd.DataFrame(rows);draws.to_csv(out/'draws.csv',index=False)
    results=[];matched=[]
    for (case,span,gate),group in draws.groupby(['case','span','gate']):
        for tau in (.5,.75,1.,1.5,2.,3.):
            false=group.null_t.abs().ge(tau)
            recovery=group.active_t.abs().ge(tau)&group.active_correct
            results.append(dict(case=case,span=span,gate=gate,threshold=tau,
                false_sign=float(false.mean()),active_recovery=float(recovery.mean()),
                false_mcse=float(false.std(ddof=1)/np.sqrt(len(false))),reps=len(group)))
        calibration=group[group.rep<reps//2];evaluation=group[group.rep>=reps//2]
        tau=float(calibration.null_t.abs().quantile(.8))
        matched.append(dict(case=case,span=span,gate=gate,calibrated_threshold=tau,
            calibration_null_rate_target=.2,evaluation_draws=len(evaluation),
            evaluation_false_sign=float(evaluation.null_t.abs().ge(tau).mean()),
            evaluation_active_recovery=float((evaluation.active_t.abs().ge(tau)&evaluation.active_correct).mean())))
    pd.DataFrame(results).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(matched).to_csv(out/'matched_false_sign.csv',index=False)
    # Deterministic duplication check on one ordinary and one incomplete history.
    check=[]
    for span in (None,60.):
        rng=np.random.default_rng(91);x=rng.normal(size=(120,2));y=.03*x[:,:1]+rng.normal(size=(120,1))
        y[:45]=np.nan
        reference=None
        for copies in (1,2,10):
            _,b,d=_compute_sign_vector(x,np.tile(y,(1,copies)),ewma_span=span,variance_estimator='date',return_diagnostics=True)
            if reference is None:reference=d['t_stats']
            np.testing.assert_allclose(d['t_stats'],reference,atol=1e-13)
            check.append(dict(span=span,copies=copies,t_stats=d['t_stats'].tolist(),slopes=b.tolist()))
    (out/'receipt.json').write_text(json.dumps(dict(status='passed',seed_base=20260925,
        reps_per_cell=reps,paired_designs=6*2*reps,gate_evaluations=len(draws),duplicate_checks=check,
        variance='date-score sandwich with Kish-date HC1 scaling; dates assumed independent',
        calibration='first half thresholds, disjoint second-half evaluation; 20% null target',
        regime_target='full-history positive average if unweighted; recent negative loading for span 60'),indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.out)
