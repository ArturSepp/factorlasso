"""Offline, known-truth FI prior experiments; only public FactorLasso is required.

Each design draws twelve correlated monthly factors, x_t = diag(s) L z_t with z_t ~ N(0, I_14),
and six responses y_ti = x_t' b_i + e_ti. The residual scale is set per asset so that the
population R-squared b_i' Sigma_x b_i / (b_i' Sigma_x b_i + sd_i^2) equals TARGET_R2, close to
the median endpoint fit of the empirical Conditional prior. Arms: zero centre (Z), automatic
highest-R-squared target (A), named target (E1), Rates-conditioned target (E2), the zero-penalty
control with Z's final signs (Z_lambda_zero) and the true-target oracle. Loading error does not
depend on factor premia; CMA error multiplies loading error by the frozen premium vector.

Run ``python -m papers.prior_targets_2026.replication.fi_validation_synthetic --output-root PATH``.
Use ``--examples`` for the two small OSS demonstrations. No market data are used.
"""
from pathlib import Path
import argparse, json, time, warnings
import numpy as np
import pandas as pd
from factorlasso import LassoModel, LassoModelType

FACTORS=['Equity','Rates','Credit IG','Credit HY','Credit EM','Carry G10',
         'Carry EM','Inflation','Commodities','Private Equity','Rates Vol','Fx']
ASSETS=['Government','IG','HY','EM','Convertible','IL']
ARMS=['Z','A','E1','E2','Z_lambda_zero','Oracle']
CASES=['credit_low','credit_high','credit_wrong','credit_omitted','il_sign','il_weak']
# A standalone, disclosed calibration; the refresh runner binds the current frozen premia.
PREMIA=np.array([.038201927387822528,.01003744865803164,.016831122378396519,.02366909875718759,.019828729812175719,.010197449387631889,.01171710655193524,.00672839131527838,.022822125990553142,.034814528123620528,.01130857230232258,0.])
MAIN_LAMBDA=0.000104264890398061
# Population R-squared of every synthetic asset: the Conditional prior's median endpoint fit
# R-squared is 0.79 for indices and 0.80 for funds (R1 E2 diagnostics).
TARGET_R2=0.80
FACTOR_SCALES=np.array([.04,.025,.02,.025,.025,.02,.025,.025,.04,.04,.02,.02])

def factor_loadings(case):
    """Return L (12 x 14) such that the unscaled factors are L z for standard normal z."""
    rho=.25 if case=='credit_low' else .9
    L=np.zeros((12,14));L[:,:12]=np.eye(12)
    for j in [2,3,4]:L[j,j]=np.sqrt(1-rho);L[j,12]=np.sqrt(rho)
    L[7,:]=0.;L[7,1]=-.8;L[7,7]=.6
    L[0,:]=0.;L[0,12]=.4;L[0,0]=np.sqrt(.84)
    return L

def residual_scales(case,truth):
    """Per-asset residual volatility giving population R-squared TARGET_R2."""
    L=factor_loadings(case)*FACTOR_SCALES[:,None]
    systematic=np.einsum('ij,jk,ik->i',truth.to_numpy(),L@L.T,truth.to_numpy())
    return np.sqrt(systematic*(1-TARGET_R2)/TARGET_R2)

def sample(case,n,seed):
    """Generate a design independent of estimated empirical betas, with held-out draws."""
    rng=np.random.default_rng(seed);z=rng.normal(size=(n+240,14))
    # Credit factors share a shock (pairwise correlation rho); Inflation loads on the Rates shock.
    x=z@(factor_loadings(case)*FACTOR_SCALES[:,None]).T
    truth=pd.DataFrame(0.,index=ASSETS,columns=FACTORS)
    truth.loc['Government','Rates']=.8
    for name,credit in [('IG','Credit IG'),('HY','Credit HY'),('EM','Credit EM')]:
        truth.loc[name,['Rates',credit]]=[.6,.4]
    truth.loc['Convertible',['Rates','Credit HY','Equity']]=[.25,.2,.55]
    truth.loc['IL',['Rates','Inflation']]=[.9,-.08 if case=='il_weak' else .45]
    eps=rng.normal(size=(len(x),len(ASSETS)))*residual_scales(case,truth)
    y=x@truth.to_numpy().T+eps
    dates=pd.date_range('2000-01-31',periods=len(x),freq='ME')
    return pd.DataFrame(x,index=dates,columns=FACTORS),pd.DataFrame(y,index=dates,columns=ASSETS),truth

def make(arm,case,truth,solver='CLARABEL',signs=None):
    """Use FCGL, estimated target magnitudes, and explicit hard signs from the FI policy.

    ``Z_lambda_zero`` is the zero centre at lambda=0 with the supplied final signs of ``Z``.
    """
    control=arm=='Z_lambda_zero'
    arm='Z' if control else arm
    hard=pd.DataFrame(np.nan,index=ASSETS,columns=FACTORS)
    hard[['Equity','Rates','Credit IG','Credit HY','Credit EM','Carry G10','Carry EM']]=1.
    hard['Private Equity']=0.;hard['Rates Vol']=-1.
    select={'Government':'Rates','IG':'Credit IG','HY':'Credit HY','EM':'Credit EM',
            'Convertible':'Equity','IL':('Rates','Inflation')}
    if arm=='E2':
        for t in ['IG','HY','EM','Convertible']:select[t]=('Rates',select[t])
    if case=='credit_wrong':select['IG']=('Rates','Credit EM') if arm=='E2' else 'Credit EM'
    if case=='credit_omitted':select['Convertible']=('Rates','Credit HY') if arm=='E2' else 'Credit HY'
    if control and signs is None:raise ValueError('Z_lambda_zero requires the final signs of Z')
    return LassoModel(model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
        cutoff_fraction=.6,reg_lambda=0. if control else MAIN_LAMBDA,loss_normalization='weight_sum',span=60,solver=solver,demean=True,
        auto_sign_constraints=True,auto_sign_threshold_t=1.,auto_sign_adaptive_weights=True,
        auto_sign_adaptive_floor=.5,auto_sign_use_fit_span=True,auto_sign_variance='date',
        factors_beta_loading_signs=signs if control else hard,
        apply_ols_prior=arm in ['A','E1','E2'],
        factor_for_prior=select if arm in ['E1','E2'] else None,
        factors_beta_prior=truth if arm=='Oracle' else None)

def draw(case,n,seed,solver):
    """Fit paired policies and independently reconstruct prediction and CMA errors."""
    x,y,truth=sample(case,n,seed);rows=[];clusters=None;models={};warn=[]
    for arm in ARMS:
        model=make(arm,case,truth,solver,signs=models['Z'].derived_signs_ if arm=='Z_lambda_zero' else None)
        xx=x.iloc[:n];yy=y.iloc[:n]
        # Omitted-factor case drops Equity for every arm, including the oracle.
        if case=='credit_omitted':
            xx=xx.drop(columns='Equity')
            if arm=='Oracle':model.factors_beta_prior=truth[xx.columns]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(xx,yy,external_clusters=clusters)
        warn.extend(str(w.message) for w in caught)
        if clusters is None:clusters=model.clusters_
        if arm=='Z_lambda_zero':pd.testing.assert_frame_equal(model.derived_signs_,models['Z'].derived_signs_)
        models[arm]=model
        b=model.estimated_betas.reindex(columns=FACTORS,fill_value=0.)
        assert np.isfinite(b).all().all()
        pred=model.predict(x.iloc[n:][xx.columns])
        np.testing.assert_allclose(pred,x.iloc[n:].to_numpy()@b.T.to_numpy()+model.alpha_const_.to_numpy(),atol=1e-12)
        delta=b-truth
        cma=delta.to_numpy()@PREMIA
        reference=np.array([sum(float(delta.loc[t,k])*PREMIA[j] for j,k in enumerate(FACTORS)) for t in ASSETS])
        np.testing.assert_allclose(cma,reference,atol=1e-14)
        if model.apply_ols_prior:
            for t in ASSETS:
                select=model.factor_for_prior.get(t) if model.factor_for_prior else model.ols_r2_.loc[t].idxmax()
                select=[select] if isinstance(select,str) else list(select)
                w=np.sqrt((1-2/61)**np.arange(n-1,-1,-1))
                design=np.column_stack([np.ones(n),xx[select]])
                ref=np.linalg.lstsq(design*w[:,None],yy[t].to_numpy()*w,rcond=None)[0][1:]
                np.testing.assert_allclose(ref,model.ols_beta_prior_.loc[t,select],atol=1e-9)
        for i,t in enumerate(ASSETS):
            nonzero=truth.loc[t].abs()>1e-10
            rows.append(dict(case=case,n=n,seed=seed,arm=arm,asset=t,
                beta_mse=float((delta.loc[t]**2).mean()),cma_error_bp=float(cma[i]*10000),
                prediction_mse=float(((pred[t]-y.iloc[n:][t])**2).mean()),
                scenario_rmse=float(np.sqrt(np.mean((delta.loc[t]*.01)**2))),
                sign_accuracy=float((np.sign(b.loc[t,nonzero])==np.sign(truth.loc[t,nonzero])).mean()),
                false_credit=float((b.loc[t,['Credit IG','Credit HY','Credit EM']].abs()>.01).sum()) if t=='Government' else 0.,
                inflation_beta=float(b.loc[t,'Inflation'])))
    spill={}
    if case.startswith('il_'):
        # All other choices and the learned cluster labels remain unchanged.
        m=make('E2',case,truth,solver);m.factor_for_prior['IL']='Inflation'
        m.fit(x.iloc[:n],y.iloc[:n],external_clusters=clusters)
        diff=models['E2'].estimated_betas-m.estimated_betas
        spill=dict(non_il_max_beta_change=float(diff.drop(index='IL').abs().max().max()),
            ig_cma_change_bp=float(diff.loc['IG'].to_numpy()@PREMIA*10000))
    return rows,spill,sorted(set(warn)),models

def run(root,pilot=False,examples=False,solver='MOSEK'):
    """Run resumable paired cells; preserve every failure instead of dropping seeds."""
    assert root.is_absolute() and 'OneDrive' not in root.parts
    out=root/('OSS' if examples else 'R4_pilot' if pilot else 'R4');out.mkdir(parents=True,exist_ok=True)
    reps=5 if pilot else 200
    protocol=dict(cases=CASES,train_lengths=[60,120],test_length=240,repetitions=reps,
        seed_base=90260924 if pilot else 20260924,solver=solver,penalty=MAIN_LAMBDA,loss_normalization='weight_sum',premia=PREMIA.tolist(),span=60,
        arms=ARMS,residual_noise=f'population R-squared {TARGET_R2} per asset',
        zero_penalty_control='Z final signs and clusters, lambda=0',
        oracle='true soft target, same constraints; not a competitor with equal information',
        wrong_mapping='IG target selects EM; true factor remains eligible',
        omitted_factor='Equity omitted from every candidate including oracle; zero-filled for truth errors',
        true_weak_inflation_beta=-.08,truth_is_not_empirically_fitted=True)
    if examples:protocol.update(cases=['credit_high','il_sign'],train_lengths=[120],repetitions=1,seed=20260924)
    p=out/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    else:p.write_text(json.dumps(protocol,indent=2)+'\n')
    if examples:
        records=[]
        for case in ['credit_high','il_sign']:
            rows,spill,warn,models=draw(case,120,20260924,solver)
            for arm,m in models.items():m.estimated_betas.to_csv(out/f'example_{case}_{arm}.csv')
            records.extend(rows)
        pd.DataFrame(records).to_csv(out/'offline_examples.csv',index=False)
        print('Two offline public-API examples passed',flush=True);return
    start=time.time()
    for ci,case in enumerate(CASES):
        for n in [60,120]:
            dest=out/f'{case}_{n}.csv'
            if dest.exists():continue
            records=[];spills=[];warning_set=set()
            for r in range(reps):
                seed=protocol['seed_base']+ci*10000+n*100+r
                rows,spill,warn,_=draw(case,n,seed,solver);records.extend(rows)
                if spill:spills.append(dict(case=case,n=n,seed=seed,**spill))
                warning_set.update(warn)
            pd.DataFrame(records).to_csv(dest,index=False,float_format='%.17g')
            pd.DataFrame(spills).to_csv(out/f'{case}_{n}_spill.csv',index=False)
            (out/f'{case}_{n}_audit.json').write_text(json.dumps(dict(reps=reps,failures=0,warnings=sorted(warning_set)),indent=2))
            print('MC',case,n,reps,'draws elapsed',round(time.time()-start,1),'seconds',flush=True)
    records=pd.concat([pd.read_csv(out/f'{c}_{n}.csv') for c in CASES for n in [60,120]])
    records['abs_cma_bp']=records.cma_error_bp.abs()
    # The independent replication, not individual correlated assets, is the sampling unit.
    draws=records.groupby(['case','n','arm','seed'])[['beta_mse','prediction_mse','abs_cma_bp','scenario_rmse','sign_accuracy']].mean()
    summaries=draws.groupby(['case','n','arm']).agg(['mean','std','count'])
    for metric in draws:summaries[(metric,'mcse')]=summaries[(metric,'std')]/np.sqrt(summaries[(metric,'count')])
    summaries.to_csv(out/'summary.csv');draws.to_csv(out/'draws.csv')
    (out/'receipt.json').write_text(json.dumps(dict(status='complete',reps=reps,cells=12,
        fits=int(12*reps*len(ARMS)+4*reps),failures=0,seconds=time.time()-start,
        independent_OLS_and_prediction_and_CMA_checks=True),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--examples',action='store_true');parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--solver',default='CLARABEL')
    args=parser.parse_args();run(args.output_root,pilot=args.pilot,examples=args.examples,solver=args.solver)
