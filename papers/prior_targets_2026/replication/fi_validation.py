"""Practical FI prior validation using the existing FactorLasso and Rosaa owners.

All inputs and outputs are explicit C-local artifacts. No live setting is mutated.
The experiment is fixed-vintage conditional validation, not a tradable backtest.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, shutil, time, warnings
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
import qis
from factorlasso import LassoModelType
from factorlasso.factor_covar import CurrentFactorCovarData, VarianceColumns
from ramen.quant_model_mvp.rosaa.data.cmas.assets.models import AssetUniverseCmaData
from ramen.quant_model_mvp.rosaa.data.cmas.factors.models import FactorCmaModelData
from ramen.quant_model_mvp.rosaa.data.factors.matf.models import MATF_CUSTOM_IG_HY
from ramen.quant_model_mvp.rosaa.data.factors.matf.beta_signs import compute_beta_loading_signs_for_matf
from ramen.quant_model_mvp.rosaa.specs.covar_estimator_spec import get_cma_covar_estimation_spec, get_covar_estimator

ROOT=Path(os.environ.get('FI_VALIDATION_ROOT','.')).resolve()
BASE=Path('C:/Users/artur/AppData/Local/AgentWork/ARTURDESKTOP/Rosaa')
DATA=BASE/'analyses/fi_prior_dataset_20260922_v3'
FINAL=BASE/'analyses/publication_cma_20260924/paper'
CUT=pd.Timestamp('2026-06-30')
FACTORS=MATF_CUSTOM_IG_HY.factor_names()
ARMS=('Z','A','E1','E2')
# Held-out zero-penalty controls: each refits the named policy's final sign set with lambda=0.
CONTROL_BASE={'Z_lambda_zero':'Z','E2_lambda_zero':'E2'}
CONTROLS=tuple(CONTROL_BASE)
MAIN_LAMBDA=get_cma_covar_estimation_spec().reg_lambda
GRID=[0.]+[float(MAIN_LAMBDA*10.**(-3+.5*i)) for i in range(13)]
QUARTERS=pd.date_range('2019-06-30','2026-03-31',freq='QE')
FUND_IDS=['IEI US Equity','LQDE LN Equity','FRTGUSD LX Equity','IHYU LN Equity',
          'HHGI2AU LX Equity','IEMB LN Equity','ACMEMI2 LX Equity','TIP US Equity',
          'LGTGILI LE Equity','CWB US Equity','OBJCGAU FP Equity','NBCHEPA ID Equity']

def sha(p):
    """Return an exact file identity."""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def jsave(p,obj):
    """Persist JSON with nonfinite values rejected."""
    Path(p).write_text(json.dumps(obj,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8')

def read(p):
    """Load a saved dated matrix without rounding its entries."""
    return pd.read_csv(p,index_col=0,parse_dates=True,float_precision='round_trip')

def csv(frame,p):
    """Persist full-precision tabular evidence."""
    frame.to_csv(p,float_format='%.17g')

def final_assets():
    """Read the frozen final CMA calculation."""
    current=ROOT/'owner_inputs'
    if current.exists():
        return AssetUniverseCmaData.load(file_name='global_saa_universe_data_cmas_usd_2026q2',local_path=str(current))
    return AssetUniverseCmaData.load(file_name='global_saa_universe_data_cmas_usd_publication',local_path=str(FINAL/'outputs'))

def factor_bundle():
    """Read the frozen final factor history and premiums."""
    return FactorCmaModelData.load(local_path=str(ROOT/'owner_inputs' if (ROOT/'owner_inputs').exists() else FINAL/'outputs'))

def mapping(row,arm):
    """Select economic factors before inspecting fitted outcomes; uncertain hybrids fall back."""
    group=row['group']; ticker=row.name
    if arm in ('Z','A'): return None
    if group=='Rates': return 'Rates'
    if group=='IL': return ('Rates','Inflation')
    selected={'IG':'Credit IG','HY':'Credit HY','EM':'Credit EM'}.get(group)
    if ticker in ['H30914US Index','H04401US Index','H04402US Index']: selected='Credit IG'
    if ticker in ['H30919US Index','I38941US Index','I13913US Index']: selected='Credit HY'
    if ticker in ['H24641US Index','CWB US Equity','OBJCGAU FP Equity']:
        return 'Equity' if arm=='E1' else ('Rates','Equity')
    if selected: return selected if arm=='E1' else ('Rates',selected)
    return None

def r0():
    """Freeze roster, masked returns, conventions, economic priors and source lineage."""
    dest=ROOT/'inputs'; dest.mkdir(exist_ok=False)
    text=(ROOT/'approved_roadmap.md').read_text(encoding='utf-8')
    matches=re.findall(r'^\| (\d+) \| (Rates|IG|HY|EM|IL|Hybrid|Specialist) \| (CMA|Brandon) \| ([A-Z0-9]+) \|',text,re.M)
    assert len(matches)==30
    hist=pd.read_csv(DATA/'universe_history.csv').set_index('asset_id')
    rows=[]
    for number,group,source,ticker in matches:
        key=f'{source.upper()}::{ticker} Index'; row=hist.loc[key].to_dict()
        row.update(ticker=ticker+' Index',group=group,lane='indices',source_key=key)
        rows.append(row)
    for ticker in FUND_IDS:
        key='APAC::'+ticker; row=hist.loc[key].to_dict()
        group={'Rates controls':'Rates','Hybrids / specialist':'Hybrid'}.get(row['research_stratum'],row['research_stratum'])
        if ticker in ['TIP US Equity','LGTGILI LE Equity']: group='IL'
        row.update(ticker=ticker,group=group,lane='funds',source_key=key); rows.append(row)
    roster=pd.DataFrame(rows).set_index('ticker')
    assert roster.index.is_unique and roster.return_currency.eq('USD').all()
    # Snapshot financial inputs before computing any new outcome.
    assets=final_assets(); factors=factor_bundle()
    prices=factors.factor_prices[FACTORS].resample('ME').last()
    x=qis.to_returns(prices,is_log_returns=True).loc['2000-01-31':CUT]
    reported=read(DATA/'returns_monthly_reported.csv')
    simple=reported[roster.source_key].copy(); simple.columns=roster.index
    from qis.market_data import FxRatesData
    fx=FxRatesData.load(local_path=str(FINAL/'resources'))
    annual=fx.domestic_rates.USD.reindex(simple.index,method='ffill')
    cash=np.log1p((annual/12).shift(1))
    y=np.log1p(simple).sub(cash,axis=0).reindex(x.index)
    # CMA response columns use the already independently audited final economic returns.
    for ticker in roster.index[roster.source.eq('CMA')]:
        y[ticker]=assets.excess_logreturns[ticker].reindex(x.index)
    for ticker,row in roster.iterrows():
        y.loc[y.index<pd.Timestamp(row.first_main_return),ticker]=np.nan
    assert x.notna().all().all() and y.iloc[-1].notna().all()
    assert len(roster)==42
    assert all(y.loc[y.index<pd.Timestamp(row.first_main_return),ticker].isna().all() for ticker,row in roster.iterrows())
    roster['E1']=roster.apply(lambda row: json.dumps(mapping(row,'E1')),axis=1)
    roster['E2']=roster.apply(lambda row: json.dumps(mapping(row,'E2')),axis=1)
    roster['mapping_basis']=roster.apply(lambda r:'explicit rated index label' if r.source=='BRANDON' else 'source mandate/class; automatic for ambiguous hybrids',axis=1)
    roster['hard_sign_basis']=np.where(roster.source.eq('BRANDON'),
        'research long-risk policy for long cash-bond indices; original missing flag retained; PE-only sensitivity required',
        'existing CMA or APAC mandate flags; check saved evidence')
    # Preserve actual fund flags, including the two newly reinstated IL funds.
    import openpyxl
    flags={};flag_sources=[]
    res=Path('C:/Users/artur/OneDrive/analytics/my_github/Rosaa/resources')
    for p in sorted(res.glob('20260901_APAC_*Mandate*.xlsx')):
        wb=openpyxl.load_workbook(p,read_only=True,data_only=True)
        if 'taa_fund' in wb:
            it=iter(wb['taa_fund'].values);head=list(next(it))
            for row in it:
                ticker=row[head.index('Ticker')]
                if ticker in FUND_IDS:
                    pair=(row[head.index('LongOnlyBetas')],row[head.index('PEfactorExposure')])
                    assert ticker not in flags or flags[ticker]==pair
                    flags[ticker]=pair
        wb.close();flag_sources.append(p)
    assert set(flags)==set(FUND_IDS) and all(v==(True,False) for v in flags.values()),flags
    meta=pd.read_csv(FINAL/'outputs/global_saa_universe_data_metadata.csv',index_col=0)
    assert meta.loc[roster.index[roster.source.eq('CMA')],'alpha_weight'].eq(0).all()
    csv(roster,dest/'roster.csv');csv(x,dest/'x.csv');csv(y,dest/'y.csv')
    csv(assets.excess_logreturns,dest/'full_y.csv');csv(meta,dest/'full_metadata.csv')
    csv(assets.covar_data.x_covar,dest/'factor_covar.csv')
    csv(assets.covar_data.y_betas,dest/'full_betas.csv')
    csv(assets.covar_data.y_variances,dest/'full_variances.csv')
    csv(assets.covar_data.clusters,dest/'full_clusters.csv')
    csv(factors.select(MATF_CUSTOM_IG_HY).cmas.loc[[CUT]],dest/'premia.csv')
    for p in (DATA/'source_snapshot').glob('brandon*oads.csv'):shutil.copy2(p,dest/p.name)
    cov=(assets.covar_data.x_covar)
    factor_unit=json.loads((FINAL/'receipt.json').read_text())
    sources=[DATA/'universe_history.csv',DATA/'returns_monthly_reported.csv',FINAL/'outputs/factor_cma2026.xlsx',
             FINAL/'outputs/factor_cma2026_factor_prices.csv',FINAL/'outputs/global_saa_universe_data_cmas_usd_publication.xlsx',
             FINAL/'outputs/global_saa_universe_data_metadata.csv']+flag_sources
    protocol=dict(cutoff=str(CUT.date()),factor_names=FACTORS,arms=ARMS,lambda_grid=GRID,span=60,lambda_main=1e-5,
        quarters=[str(d.date()) for d in QUARTERS],zero_tolerance=1e-4,zero_sensitivity=1e-3,
        economic_materiality_annual_shock=.001,seed=20260924,mc_reps=200,mc_pilot_reps=5,
        factor_vintage='final 2026 MATF-CMA; fixed-vintage retrospective reconstruction, historical availability not certified',
        expert_information='factor selection only; magnitudes estimated within training sample',
        fund_selection='12 named long-history category examples: one rates control, paired IG/HY/EM/IL/convertibles, one corporate hybrid; selected by role not new results',
        sign_resolution='CMA/APAC documented flags; Brandon shared research long-risk policy inferred from cash-bond index design, not claimed vendor beta metadata; PE-only sensitivity',
        sources={str(p):sha(p) for p in sources},frozen_runtime=factor_unit['versions'],
        methodology_sources=['https://data.bloomberglp.com/professional/sites/10/Bloomberg-Index-Publications-Fixed-Income-Index-Methodology.pdf',
          'https://www.spglobal.com/spdji/en/documents/methodologies/methodology-iboxx-contingent-convertible.pdf'],
        cma_policy='18 approved index metadata rows; Brandon-only rows illustrative zero-alpha, no regional overlays')
    jsave(ROOT/'protocol.json',protocol)
    jsave(dest/'manifest.json',{p.name:sha(p) for p in dest.iterdir() if p.is_file()})
    jsave(ROOT/'R0_receipt.json',dict(status='complete_with_disclosed_vintage_and_sign_assumptions',indices=30,funds=12,
        mapped_E1=int(roster.E1.ne('null').sum()),native_history_masks=True,fund_flags=flags,
        source_hashes_verified=True,factor_covariance_positive_definite=bool(np.linalg.eigvalsh(cov).min()>0)))
    print('R0 complete: 30 indices, 12 funds; inputs and policy frozen.',flush=True)

def load(lane):
    """Read immutable inputs and verify each on every run."""
    inp=ROOT/'inputs'
    manifest=json.loads((inp/'manifest.json').read_text())
    for name,digest in manifest.items():assert sha(inp/name)==digest,name
    roster=pd.read_csv(inp/'roster.csv',index_col=0).query('lane == @lane')
    return read(inp/'x.csv'),read(inp/'y.csv')[roster.index],roster

def make_model(roster,arm,lam=None,span=60,signs=None,prior=None,separable=False,pe_only=False):
    """Construct the actual owner estimator under one explicitly named policy."""
    if signs is None:
        signs=compute_beta_loading_signs_for_matf(pd.Series(True,index=roster.index),pd.Series(False,index=roster.index),MATF_CUSTOM_IG_HY)
        if pe_only: signs.loc[:,signs.columns!='Private Equity']=np.nan
    selections={ticker:mapping(row,arm) for ticker,row in roster.iterrows()}
    selections={k:v for k,v in selections.items() if v is not None}
    spec=get_cma_covar_estimation_spec(apply_ols_prior=arm!='Z',pe_beta_prior=None,
        reg_lambda=MAIN_LAMBDA if lam is None else lam)
    model=get_covar_estimator(spec,factors_beta_loading_signs=signs,factors_beta_prior=prior,
        factor_for_prior=pd.Series(selections,dtype=object) if selections else None).lasso_model
    if separable:model.model_type=LassoModelType.LASSO
    return model

def independent_prior(x,y,selected,span):
    """Independent weighted design-matrix least squares, not the owner moment formula."""
    xx=x[list(selected)];mask=xx.notna().all(axis=1)&y.notna()
    weights=(1-2/(span+1))**np.arange(len(x)-1,-1,-1)
    design=np.column_stack([np.ones(mask.sum()),xx.loc[mask]])
    root=np.sqrt(weights[mask])
    b=np.linalg.lstsq(design*root[:,None],y.loc[mask].to_numpy()*root,rcond=None)[0]
    return b[1:]

def fit(x,y,roster,arm,lam=None,span=60,clusters=None,**kwargs):
    """Fit one policy and check hard restrictions and independently reconstructed priors."""
    model=make_model(roster,arm,lam,span,**kwargs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');model.fit(x,y,span=span,external_clusters=clusters)
    assert np.isfinite(model.estimated_betas).all().all()
    signs=model.derived_signs_
    betas=model.estimated_betas
    for label in (1.,-1.,0.):
        values=betas.where(signs.eq(label)).stack()
        if values.empty:continue
        if label==0:assert values.abs().max()<1e-4
        else:assert (label*values).min()>-1e-4
    errors=[]
    if model.apply_ols_prior:
        raw=model.ols_beta_prior_
        for ticker in y:
            selection=mapping(roster.loc[ticker],arm)
            if selection is None:
                selection=[model.ols_r2_.loc[ticker].idxmax()]
            elif isinstance(selection,str):selection=[selection]
            expected=independent_prior(x,y[ticker],selection,span)
            actual=raw.loc[ticker,list(selection)].to_numpy()
            errors.append(float(np.max(np.abs(actual-expected))))
        assert max(errors,default=0)<1e-8,max(errors,default=0)
    return model,dict(prior_reference_error=max(errors,default=0),warnings=[str(w.message) for w in caught])

def save_model(model,out,arm,audit):
    """Save all coefficients, prior centres, signs and residual diagnostic values."""
    out.mkdir(parents=True,exist_ok=True)
    csv(model.estimated_betas,out/f'{arm}_betas.csv')
    csv(model.derived_signs_,out/f'{arm}_signs.csv')
    csv(model.clusters_ if model.clusters_ is not None else pd.Series('separable',index=model.estimated_betas.index),out/f'{arm}_clusters.csv')
    csv(model.alpha_const_,out/f'{arm}_alpha.csv')
    result=model.estimation_result_
    summary=pd.DataFrame({'fit_r2_owner':np.asarray(result.r2),'residual_var_monthly':np.asarray(result.ss_res)},index=model.estimated_betas.index)
    csv(summary,out/f'{arm}_diagnostics.csv')
    for name,attr in [('raw_prior','ols_beta_prior_'),('effective_prior','effective_beta_prior_'),('marginal_r2','ols_r2_')]:
        value=getattr(model,attr,None)
        if value is not None:csv(value,out/f'{arm}_{name}.csv')
    audit=dict(audit,solver=model.solver,reg_lambda=model.reg_lambda,loss_normalization=model.loss_normalization,
        sign_use_fit_span=model.auto_sign_use_fit_span,sign_variance=model.auto_sign_variance,
        min_loss_mass=float(model.loss_weight_mass_.min()),max_loss_mass=float(model.loss_weight_mass_.max()))
    jsave(out/f'{arm}_audit.json',audit)

def r1(lane):
    """Endpoint policies, common-sign controls, lambda path and independent checks."""
    x,y,roster=load(lane);out=ROOT/'R1'/lane;out.mkdir(parents=True,exist_ok=True)
    models={};audits={}
    for arm in ARMS:
        m,a=fit(x,y,roster,arm,clusters=models['Z'].clusters_ if models else None)
        models[arm]=m;audits[arm]=a;save_model(m,out,arm,a)
        print('R1',lane,arm,'mean R2',m.estimation_result_.r2.mean(),flush=True)
    # A 2x2 mechanism study keeps hard constraints fixed while controlling soft signs.
    control={}
    for sign_name,reference in [('detected',models['Z']),('prior',models['E2'])]:
        s=reference.derived_signs_.copy()
        for centre in ['Z','E2']:
            label=f'{centre}_{sign_name}'
            m,a=fit(x,y,roster,centre,signs=s,clusters=models['Z'].clusters_)
            save_model(m,out,label,a);control[label]=m
        z,_=fit(x,y,roster,'Z',lam=0,signs=s,clusters=models['Z'].clusters_)
        e,_=fit(x,y,roster,'E2',lam=0,signs=s,clusters=models['Z'].clusters_)
        np.testing.assert_allclose(z.predict(x),e.predict(x),atol=1e-7,rtol=1e-6)
    for arm in ARMS:
        m,a=fit(x,y,roster,arm,separable=True);save_model(m,out,'separable_'+arm,a)
        path=make_model(roster,arm).fit_reg_lambda_path(x,y,GRID,span=60)
        records=[]
        for lam,m in zip(GRID,path):
            records.append(pd.DataFrame({'lambda':lam,'ticker':y.columns,'r2':m.estimation_result_.r2}))
            if lam==MAIN_LAMBDA:np.testing.assert_allclose(m.estimated_betas,models[arm].estimated_betas,atol=2e-5)
        pd.concat(records).to_csv(out/f'{arm}_lambda_scores.csv',index=False)
    if lane=='indices':
        for arm in ARMS:
            m,a=fit(x,y,roster,arm,pe_only=True);save_model(m,out,'pe_only_'+arm,a)
        # Rating-vs-geography and convertible three-factor cases are fixed in R0/roadmap.
        mapping3={t:mapping(row,'E2') for t,row in roster.iterrows() if mapping(row,'E2') is not None}
        mapping3['H24641US Index']=('Rates','Credit HY','Equity')
        m=make_model(roster,'E2');m.factor_for_prior=pd.Series(mapping3,dtype=object)
        m.fit(x,y,span=60,external_clusters=models['Z'].clusters_)
        save_model(m,out,'hybrid_three_factor',{'note':'prespecified sensitivity, HY choice assumed, not validated rating'})
        for t in ['I12881US Index','I05040US Index','I05039US Index']:
            mapping3[t]=('Rates','Credit IG' if t=='I12881US Index' else 'Credit HY')
        mapping3['H24641US Index']=('Rates','Equity')
        m=make_model(roster,'E2');m.factor_for_prior=pd.Series(mapping3,dtype=object)
        m.fit(x,y,span=60,external_clusters=models['Z'].clusters_)
        save_model(m,out,'rating_alternative',{'note':'prespecified rating alternative, no winner selection'})
    jsave(out/'receipt.json',dict(status='complete',asset_count=len(y.columns),prior_and_hard_sign_checks=audits,
        common_sign_lambda_zero_invariance=True,sign_effect_separate=True))

def r2():
    """Use Rosaa's actual CMA arithmetic with fixed premiums and owner risk assembly."""
    bundle=factor_bundle().select(MATF_CUSTOM_IG_HY).for_asset_model(MATF_CUSTOM_IG_HY)
    roster=pd.read_csv(ROOT/'inputs/roster.csv',index_col=0).query("lane == 'indices'")
    meta=pd.read_csv(ROOT/'inputs/full_metadata.csv',index_col=0)
    rows={}
    for t,row in roster.iterrows():
        if row.source=='CMA':rows[t]=meta.loc[t].to_dict()
        else:rows[t]=dict(name=row['name'],currency='USD',asset_class='Bonds',universe='research_fi',alpha_weight=0.)
    research_meta=pd.DataFrame.from_dict(rows,orient='index')
    # Unknown optional overlays must remain None, not float NaN.
    research_meta=research_meta.astype(object).where(research_meta.notna(),None)
    covariance=pd.read_csv(ROOT/'inputs/factor_covar.csv',index_col=0,float_precision='round_trip')
    fixed_vol=final_assets().covar_data.get_model_vols()[VarianceColumns.TOTAL_VOL.value]
    for t in research_meta.index:
        region=research_meta.loc[t,'regional_cma_rates']
        if isinstance(region,str):
            research_meta.loc[t,'regional_cma_rates_load']=float(fixed_vol[t]/bundle.factor_model.vol_targets()[bundle.factor_model.regional_rates_factor])
    out=ROOT/'R2';out.mkdir(exist_ok=True)
    all_cmas=[];all_contrib=[]
    for arm in ARMS:
        b=pd.read_csv(ROOT/f'R1/indices/{arm}_betas.csv',index_col=0)
        diag=pd.read_csv(ROOT/f'R1/indices/{arm}_diagnostics.csv',index_col=0)
        variance=pd.DataFrame({VarianceColumns.RESIDUAL_VARS.value:12*diag.residual_var_monthly})
        risk=CurrentFactorCovarData(covariance,b,variance)
        vols=risk.get_model_vols();snap=b.join(vols);snap[VarianceColumns.ALPHA.value]=0.
        cmas,_=bundle.estimate_asset_universe_cma(research_meta,snap,'USD',CUT,concat_metadata=False)
        attrib=bundle.estimate_factor_attribution(research_meta,snap,'USD',CUT)
        cmas=cmas.reindex(b.index);attrib=attrib.reindex(b.index)
        expected=b.mul(bundle.factor_excess_cma.loc[CUT],axis=1).sum(axis=1)
        cmas['pure_factor_total_cma']=expected+cmas.rf_rate
        # Independent scalar equation verifies the owner, including the frozen rates load.
        adjusted=expected.copy()
        for t in b.index:
            region=research_meta.loc[t,'regional_cma_rates']
            if isinstance(region,str):
                scale=research_meta.loc[t,'regional_cma_rates_load']
                adjusted[t]+=(scale-1)*b.loc[t,'Rates']*bundle.factor_excess_cma.loc[CUT,'Rates']+scale*(1-b.loc[t,'Rates'])*bundle.rates_cma_excess.loc[CUT,region]
        np.testing.assert_allclose(cmas.base_excess_factor_cma,adjusted,atol=1e-12)
        np.testing.assert_allclose(cmas.base_total_cma,cmas.rf_rate+adjusted,atol=1e-12)
        np.testing.assert_allclose(attrib.total_cma,cmas.base_total_cma,atol=1e-12)
        cmas['regional_adjustment']=adjusted-expected
        cmas['vol']=vols[VarianceColumns.TOTAL_VOL.value];cmas['sharpe_rf0']=cmas.base_total_cma/cmas.vol
        cmas['arm']=arm;cmas['ticker']=cmas.index;cmas['scope']=np.where(roster.reindex(b.index).source.eq('CMA'),'CMA metadata','illustrative zero-alpha')
        all_cmas.append(cmas);attrib['arm']=arm;attrib['ticker']=attrib.index;all_contrib.append(attrib)
        csv(risk.get_y_covar(),out/f'{arm}_covar.csv')
    pd.concat(all_cmas).to_csv(out/'cmas.csv',index=False,float_format='%.17g')
    pd.concat(all_contrib).to_csv(out/'attribution.csv',index=False,float_format='%.17g')
    jsave(out/'receipt.json',dict(status='complete',owner_cma_reconciliation=True,alpha_admission=0,
        covariance='fixed final factor covariance; changed fitted betas and residuals',factor_units='disclosed 1x inner notional/NAV'))
    print('R2 complete: 120 CMA rows reconciled.',flush=True)

def r3(lane):
    """Quarterly held-out conditional return explanation under all four policies."""
    x,y,roster=load(lane);out=ROOT/'R3'/lane;out.mkdir(parents=True,exist_ok=True)
    for span in [60,36]:
        rows=[];betas=[];audits=[]
        for q in QUARTERS:
            prefix=f'{span}_{q:%Y%m%d}';piece=out/f'{prefix}_predictions.csv'
            if piece.exists():continue
            xx=x.loc[:q];yy=y.reindex(xx.index)
            names=yy.columns[(yy.notna().sum()>=60)&yy.iloc[-1].notna()];yy=yy[names];rr=roster.loc[names]
            dates=pd.date_range(q+pd.offsets.MonthEnd(1),periods=3,freq='ME');xt=x.loc[dates];yt=y.loc[dates,names]
            weights=(1-2/61)**np.arange(len(xx)-1,-1,-1)
            mu=pd.Series({t:np.average(yy[t].dropna(),weights=weights[yy[t].notna()]) for t in names})
            clusters=None;quarter_rows=[];quarter_betas=[];quarter_audit={};models={}
            def record(label,m,prior_free):
                """Store held-out predictions and fitted loadings for one policy or control."""
                pred=m.predict(xt)
                np.testing.assert_allclose(pred,xt.to_numpy()@m.estimated_betas.T.to_numpy()+m.alpha_const_.to_numpy(),atol=1e-12)
                for t in names:
                    for d in dates:
                        if pd.notna(yt.loc[d,t]):quarter_rows.append(dict(fit_date=str(q.date()),month=str(d.date()),ticker=t,arm=label,span=span,
                            group=rr.loc[t,'group'],kind=rr.loc[t,'kind'],actual=yt.loc[d,t],predicted=pred.loc[d,t],benchmark=mu[t]))
                    for k in FACTORS:quarter_betas.append(dict(fit_date=str(q.date()),ticker=t,arm=label,span=span,factor=k,beta=m.estimated_betas.loc[t,k],
                        prior=0. if prior_free else m.effective_beta_prior_.loc[t,k],sign=m.derived_signs_.loc[t,k]))
            for arm in ARMS:
                m,a=fit(xx,yy,rr,arm,span=span,clusters=clusters)
                if clusters is None:clusters=m.clusters_
                models[arm]=m;record(arm,m,arm=='Z');quarter_audit[arm]=a
            # Zero-penalty controls keep the reference policy's clusters and final signs; only lambda changes.
            for control,base in CONTROL_BASE.items():
                m,a=fit(xx,yy,rr,base,lam=0.,span=span,clusters=clusters,signs=models[base].derived_signs_)
                pd.testing.assert_frame_equal(m.derived_signs_,models[base].derived_signs_)
                assert m.reg_lambda==0.
                record(control,m,True);quarter_audit[control]=a
            pd.DataFrame(quarter_rows).to_csv(piece,index=False,float_format='%.17g')
            pd.DataFrame(quarter_betas).to_csv(out/f'{prefix}_betas.csv',index=False,float_format='%.17g')
            jsave(out/f'{prefix}_audit.json',quarter_audit)
            print('R3',lane,span,q.date(),len(names),'done',flush=True)
    # Future mutation check on the actual input-preparation boundary.
    tests=[]
    for q in [QUARTERS[0],QUARTERS[14],QUARTERS[-1]]:
        for arm in ARMS:
            def past_fit(xx,yy):
                a=xx.loc[:q];b=yy.reindex(a.index);names=b.columns[(b.notna().sum()>=60)&b.iloc[-1].notna()]
                return fit(a,b[names],roster.loc[names],arm)[0]
            original=past_fit(x,y);mx=x.copy();my=y.copy();mx.loc[mx.index>q]*=100;my.loc[my.index>q]*=-100
            altered=past_fit(mx,my)
            np.testing.assert_allclose(original.estimated_betas,altered.estimated_betas,atol=1e-12)
            tests.append(f'{q.date()}:{arm}')
    jsave(out/'receipt.json',dict(status='complete',n_quarters=28,spans=[60,36],future_mutation_passed=tests,
        prediction_reference=True,scope='retrospective fixed-vintage conditional explanation',
        zero_penalty_controls={k:f'{v} final signs and clusters, lambda=0' for k,v in CONTROL_BASE.items()}))

def main():
    """Dispatch an approved stage to an explicit C-local workspace."""
    global ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True);parser.add_argument('--lane',default='indices')
    parser.add_argument('--output-root',type=Path,required=True);args=parser.parse_args()
    ROOT=args.output_root.resolve();assert ROOT.is_absolute() and 'OneDrive' not in str(ROOT)
    if args.stage=='R0':r0()
    elif args.stage=='R1':r1(args.lane)
    elif args.stage=='R2':r2()
    elif args.stage=='R3':r3(args.lane)
    else:
        from . import fi_validation_metrics
        fi_validation_metrics.run(args.stage,ROOT,args.lane)

if __name__=='__main__':main()
