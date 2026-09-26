"""Historical factor-shock replay and presentation helpers for the FI prior note."""
from pathlib import Path
import hashlib,json,math,re
import numpy as np
import pandas as pd

POLICY_NAMES={'Z':'Zero prior','A':'Max-R2 prior','E1':'Expert prior','E2':'Conditional prior'}
ARMS=list(POLICY_NAMES)
REFERENCE=['LEGATRUH Index','H23059US Index','H04386US Index']

def sha(path):
    """Return evidence identity."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path,value):
    """Write strict JSON evidence."""
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n',encoding='utf-8')

def read(root,name,**kwargs):
    """Preserve saved input precision."""
    return pd.read_csv(Path(root)/name,float_precision='round_trip',**kwargs)

def policy_cell(value):
    """Expand policy codes in table cells, preserving the Tier 2 A rating label."""
    text=str(value)
    if text=='Tier 2 A':return text
    text=re.sub(r'\b(E1|E2|Z|A)\b',lambda m:POLICY_NAMES[m[1]],text)
    return 'Policy' if text=='Arm' else text

def compute(root,out):
    """Select six observed adverse months and replay all endpoint models through QIS."""
    import qis
    from optimalportfolios.covar_estimation.ewma_covar_estimator import estimate_current_ewma_covar
    from ramen.quant_model_mvp.rosaa.specs.covar_estimator_spec import get_cma_covar_estimation_spec
    from . import fi_validation as study
    from .fi_validation_exhibits import SHORT_NAMES
    root,out=Path(root),Path(out);out.mkdir(exist_ok=False)
    study.ROOT=root
    x,y,roster=study.load('indices')
    assert pd.infer_freq(x.index)=='ME' and x.index.equals(y.index)
    _,fund_y,fund_roster=study.load('funds')
    assert pd.infer_freq(fund_y.index)=='ME' and x.index.equals(fund_y.index)
    basket=np.expm1(y[REFERENCE].dropna()).mean(axis=1)
    ranked=basket.sort_values(kind='stable')
    march=pd.Timestamp('2020-03-31')
    others=ranked.drop(index=march).head(5).index.tolist()
    selected=[march]+others
    shocks=x.loc[selected].copy()
    assert shocks.notna().all().all() and len(shocks)==6
    shocks.to_csv(out/'factor_log_shocks.csv',index_label='date')
    selection=pd.DataFrame({'reference_return':basket,'stress_rank':basket.rank(method='first')})
    selection['selected']=selection.index.isin(selected)
    selection.sort_values('stress_rank').to_csv(out/'selection_audit.csv',index_label='date')
    scenarios=pd.DataFrame({'scenario':[d.strftime('%b %Y') for d in selected],
        'reference_return':basket.loc[selected],'stress_rank':selection.loc[selected,'stress_rank']},index=selected)
    scenarios.to_csv(out/'scenarios.csv',index_label='date')
    records=[];max_error=0.;max_log_error=0.
    for lane,assets in [('indices',roster),('funds',fund_roster)]:
        for arm in ARMS:
            beta=read(root,f'R1/{lane}/{arm}_betas.csv',index_col=0).loc[assets.index,x.columns]
            result=qis.project_factor_scenarios(betas=beta,amounts=pd.Series(1.,index=beta.index),factor_log_shocks=shocks)
            result.asset_pnl.to_csv(out/f'{lane}_{arm}_returns.csv',index_label='date')
            result.factor_attribution.to_csv(out/f'{lane}_{arm}_factor_attribution.csv',index_label='date')
            for date in selected:
                for ticker in beta.index:
                    # Independent scalar math.fsum/expm1 verifies QIS broadcast valuation.
                    log_return=math.fsum(float(beta.loc[ticker,k])*float(shocks.loc[date,k]) for k in beta.columns)
                    reference=math.expm1(log_return)
                    error=abs(reference-result.asset_pnl.loc[date,ticker]);max_error=max(max_error,error)
                    max_log_error=max(max_log_error,abs(log_return-result.asset_log_returns.loc[date,ticker]))
                    assert error<1e-12 and np.isfinite(reference)
                    records.append(dict(lane=lane,arm=arm,policy=POLICY_NAMES[arm],ticker=ticker,
                        short_name=SHORT_NAMES[ticker],date=date,scenario=date.strftime('%b %Y'),
                        log_return=log_return,stress_return=reference))
    all_results=pd.DataFrame(records);all_results.to_csv(out/'stress_returns.csv',index=False)
    assert len(all_results)==42*4*6
    # The risk covariance is the frozen CMA weekly estimate, distinct from monthly regression inputs.
    spec=get_cma_covar_estimation_spec()
    factors=study.factor_bundle()
    prices=factors.factor_prices[x.columns].loc[:study.CUT]
    reconstructed=estimate_current_ewma_covar(prices,returns_freq=spec.factor_returns_freq,
        span=spec.factor_covar_span,demean=True,apply_an_factor=False)
    reconstructed*=qis.get_annualisation_conversion_factor(from_freq=spec.factor_returns_freq,to_freq='YE')
    frozen=read(root,'inputs/factor_covar.csv',index_col=0).loc[x.columns,x.columns]
    covariance_error=float((reconstructed-frozen).abs().max().max())
    assert covariance_error<1e-12,covariance_error
    frequencies=dict(beta_response_frequency='ME',beta_factor_frequency='ME',
        factor_covariance_frequency=spec.factor_returns_freq,factor_covariance_span=spec.factor_covar_span,
        beta_span_months=spec.span_freq_dict['ME'],model_type=str(spec.model_type),reg_lambda=spec.reg_lambda,
        cutoff_fraction=spec.cluster_cutoff_fraction,auto_sign_threshold=spec.auto_sign_threshold_t,
        adaptive_floor=spec.auto_sign_adaptive_floor,solver=spec.solver,
        factor_covariance_reference_max_error=covariance_error,
        research_variations='four prior policies; span 36, sign and penalty sensitivities; synthetic MC is separate')
    dump(out/'frequency_audit.json',frequencies)
    example=all_results.query("lane=='indices' and date==@march").pivot(index='short_name',columns='arm',values='stress_return')
    example.to_csv(out/'march2020_summary.csv')
    dump(out/'checks.json',dict(status='passed',selection_rule='March 2020 plus five lowest other reference-basket months',
        reference_indices=REFERENCE,reference_weights=[1/3]*3,selection_from=str(basket.index.min().date()),
        selection_to=str(basket.index.max().date()),selection_months=len(basket),
        reference_return_convention='mean of expm1(monthly excess log returns); constant equal weights; complete cases',
        selected_dates=[str(d.date()) for d in selected],predictions=len(all_results),
        stress_convention='expm1(sum beta times historical factor log shock); negative is loss',
        excluded_components=['cash','intercept','alpha','regional expected-return overlay','residual shock'],
        endpoint_betas='2026-06-30; fixed historical-shock replay, not contemporaneous forecast',
        valuation_reference_max_error=max_error,log_reference_max_error=max_log_error,
        frequency_audit=frequencies))
    dump(out/'manifest.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print(scenarios.to_string(),flush=True)
    print(example.loc[['Global IG agg','Global IG corp','Global HY','EM hard currency','Global IL','Convertibles']].to_string(),flush=True)
    print('Frequency audit:',json.dumps(frequencies),flush=True)

def verify(root):
    """Verify every stress exhibit input and receipt."""
    root=Path(root)
    for name,value in json.loads((root/'manifest.json').read_text()).items():assert sha(root/name)==value,name
    checks=json.loads((root/'checks.json').read_text())
    assert checks['status']=='passed' and checks['predictions']==1008
    return checks

def table_block(root,table):
    """Provide the six scenario-selection records and six principal factor shocks."""
    root=Path(root);verify(root)
    scenarios=read(root,'scenarios.csv',index_col=0)
    shocks=read(root,'factor_log_shocks.csv',index_col=0)
    factors=['Rates','Credit IG','Credit HY','Credit EM','Inflation','Equity']
    rows=[[r.scenario,f'{100*r.reference_return:.2f}',int(r.stress_rank),
        *[f'{100*np.expm1(shocks.loc[d,k]):+.2f}' for k in factors]] for d,r in scenarios.iterrows()]
    return table(['Month','Reference %','Rank','Rates %','IG %','HY %','EM %','Infl. %','Equity %'],rows)

def figures(root,out):
    """Display index and fund stress comparisons on one shared loss/gain scale."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from .fi_validation_exhibits import SHORT_NAMES
    root,out=Path(root),Path(out);verify(root)
    scenarios=read(root,'scenarios.csv',index_col=0)
    all_results=read(root,'stress_returns.csv')
    scale=max(5.,float(np.ceil(all_results.stress_return.abs().max()*100/5)*5))
    norm=TwoSlopeNorm(vmin=-scale,vcenter=0.,vmax=scale)
    layouts=[('indices',ARMS[:2],'fi_stress_indices_baseline',1,2,10.1),
             ('indices',ARMS[2:],'fi_stress_indices_expert',1,2,10.1),
             ('funds',ARMS,'fi_stress_funds',2,2,9.4)]
    for lane,arms,name,nrows,ncols,height in layouts:
        fig,axes=plt.subplots(nrows,ncols,figsize=(10.0,height),sharey=True,layout='constrained')
        flat=np.asarray(axes).reshape(-1)
        for ax,arm in zip(flat,arms):
            frame=read(root,f'{lane}_{arm}_returns.csv',index_col=0).loc[scenarios.index].T*100
            im=ax.imshow(frame,cmap='RdBu',norm=norm,aspect='auto')
            ax.set_title(POLICY_NAMES[arm],fontsize=11)
            ax.set_xticks(range(6),scenarios.scenario,rotation=50,ha='right',fontsize=9)
            ax.set_yticks(range(len(frame)),[SHORT_NAMES[t] for t in frame.index],fontsize=9)
            ax.set_xticks(np.arange(-.5,6,1),minor=True)
            ax.set_yticks(np.arange(-.5,len(frame),1),minor=True)
            ax.grid(which='minor',color='white',linewidth=.3);ax.tick_params(which='minor',left=False,bottom=False)
            for i in range(len(frame)):
                for j in range(6):
                    value=frame.iloc[i,j]
                    ax.text(j,i,f'{value:.1f}' if abs(value)>=.05 else '0.0',ha='center',va='center',
                        fontsize=8.1,color='white' if abs(value)>.6*scale else '#171717')
        fig.colorbar(im,ax=flat.tolist(),label='Factor-implied monthly return (%); negative = loss',shrink=.65,pad=.02)
        for extension in ['pdf','png']:fig.savefig(out/f'{name}.{extension}',dpi=180,bbox_inches='tight')
        plt.close(fig)

def shade_loading_tables(body):
    """Shade only beta cells in Tables 1, 13 and 14, leaving numbers unchanged."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm,to_hex
    norm=TwoSlopeNorm(vmin=-.5,vcenter=0,vmax=1.5)
    counts=[]
    table_index=0
    def process(match):
        """Apply a shared signed color scale to each selected loading table."""
        nonlocal table_index
        table_index+=1
        original=match[0]
        if table_index not in [1,13,14]:return original
        active=False;changed=[];count=0
        for line in original.splitlines():
            if line==r'\midrule':active=True
            elif line==r'\bottomrule':active=False
            elif active and ' & ' in line:
                cells=line.split(' & ')
                for i in range(2,8):
                    value=float(cells[i])
                    color='#ffffff' if value==0 else to_hex(plt.get_cmap('RdBu_r')(norm(np.clip(value,-.5,1.5))))
                    rgb=np.array([int(color[j:j+2],16)/255 for j in [1,3,5]])
                    ink='white' if rgb.dot([.2126,.7152,.0722])<.48 else 'black'
                    cells[i]=r'\cellcolor[HTML]{'+color[1:].upper()+'}'+r'\textcolor{'+ink+'}{'+cells[i]+'}'
                    count+=1
                line=' & '.join(cells)
            changed.append(line)
        counts.append(count)
        return '\n'.join(changed)
    result=re.sub(r'\\begin\{table\}.*?\\end\{table\}',process,body,flags=re.S)
    assert counts==[144,180,72],counts
    return result

def wrap_policy_headers(body):
    """Keep descriptive policy names legible in comparison-table headers."""
    def table(match):
        """Wrap only the header between top and middle rules."""
        content=match[0]
        a=content.index(r'\toprule');b=content.index(r'\midrule')
        head=content[a:b]
        cells=head.split(' & ')
        for i,cell in enumerate(cells):
            for name in POLICY_NAMES.values():
                if name in cell:
                    first,last=name.split(' ',1)
                    cell=cell.replace(name,r'\shortstack{'+first+r'\\'+last+'}')
            # The long policy contrast deserves two short lines, preserving its meaning.
            cell=cell.replace(r'\shortstack{Conditional\\prior}-\shortstack{Max-R2\\prior}, bp',
                r'\shortstack{Conditional minus\\Max-R2, bp}')
            cells[i]=cell
        return content[:a]+' & '.join(cells)+content[b:]
    return re.sub(r'\\begin\{table\}.*?\\end\{table\}',table,body,flags=re.S)
