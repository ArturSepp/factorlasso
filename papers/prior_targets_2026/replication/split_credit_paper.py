"""Evidence blocks and scientific figures for the single split-credit manuscript."""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .split_credit_research import ROOT,FACTORS,METHODS,SETTINGS,PROFILES,check
from .build_paper import markdown_table

LABELS={'M0_ols':'OLS','M1_zero':'Zero','M2_auto':'Automatic winner','economic':'Economic',
    'economic_noisy':'Economic + noise','rates_credit':'Rates + three credits',
    'equity_rates_credit':'Equity + Rates + three credits','oracle':'Oracle truth',
    'M4_free_winner':'Free winner','M5_post_lasso':'Post-LASSO'}


def evidence_blocks(root=ROOT):
    """Derive every replacement exhibit and headline directly from paired source rows."""
    assert root==ROOT
    l1=pd.read_csv(root/'l1_confirmation/paired_summary.csv').set_index(['profile','n','method'])
    group=pd.read_csv(root/'groups_confirmation/paired_summary.csv').set_index(['profile','setting','method'])
    rolling=pd.read_csv(root/'rolling/method_summary.csv').set_index(['span','cohort','method'])
    bootstrap=pd.read_csv(root/'rolling/paired_bootstrap.csv').set_index(['span','cohort','method','block'])
    blocks={};receipts=[]
    def value(frame,idx,key):
        """Record a numerical source locator whenever a displayed number is retrieved."""
        result=float(frame.loc[idx,key]);receipts.append(dict(index=idx,metric=key,value=result))
        return result
    def at(profile,method,metric='credit_mse',n=112):
        """Read a complete-panel simulation mean or paired contrast."""
        return value(l1,(profile,n,method),metric)
    def ratio(profile,method,metric='credit_mse',n=112):
        """Compare equally weighted paired-panel means."""
        return at(profile,method,metric,n)/at(profile,'M1_zero',metric,n)
    def ci(profile,method,metric='credit_mse'):
        """Render the paired Monte Carlo interval."""
        return '['+', '.join(f'{at(profile,method,metric+s):+.6f}' for s in ['_low','_high'])+']'
    blocks['split_simulation']=markdown_table(
        ['Target or control','Calibration','Zero true credit','Rates multiplied by 4'],
        [[LABELS[m],*[f'{at(p,m):.6f}' for p in PROFILES]] for m in METHODS])
    table_methods=['M1_zero','M2_auto','economic','rates_credit','equity_rates_credit','M4_free_winner','M5_post_lasso']
    blocks['split_components']=markdown_table(
        ['Target or control','IG MSE','HY MSE','EM MSE','Full covariance error'],
        [[LABELS[m],*[f'{at("baseline",m,k):.6f}' for k in ['credit_ig_mse','credit_hy_mse','credit_em_mse','full_covar_error']]] for m in table_methods])
    blocks['abstract_result']=(
        f'In the twelve-factor calibration, an economic target reduces aggregate IG/HY/EM loading mean squared error by '
        f'{100*(1-ratio("baseline","economic")):.1f}%, with a paired interval that includes zero. '
        f'The automatic target raises that error by {100*(ratio("baseline","M2_auto")-1):.1f}%. '
        f'When all three true credit exposures are zero, economic targeting produces '
        f'{ratio("zero_credit_exposure","economic"):.2f} times the zero-target loading error. '
        'All twelve regressors, including IG, HY and EM credit, remain available in this stress case.')
    blocks['split_simulation_result']=(
        f'At 112 training observations, the economic-minus-zero credit-MSE difference is '
        f'{at("baseline","economic","credit_mse_delta"):+.6f}, with pointwise 95% interval {ci("baseline","economic")}. '
        f'The automatic-minus-zero difference is {at("baseline","M2_auto","credit_mse_delta"):+.6f} '
        f'{ci("baseline","M2_auto")}. The latter is positive for each of the three credit components. '
        f'When true credit exposures are zero, the economic target increases credit MSE by '
        f'{at("zero_credit_exposure","economic","credit_mse_delta"):+.6f} '
        f'{ci("zero_credit_exposure","economic")}, while cohort prediction NMSE rises by only '
        f'{100*(ratio("zero_credit_exposure","economic","credit_prediction_nmse")-1):.2f}%. '
        'This contrast illustrates why return reconstruction alone can understate false attribution.')
    blocks['split_covariance_result']=(
        f'Automatic targeting changes relative full-covariance error from '
        f'{at("baseline","M1_zero","full_covar_error"):.6f} to {at("baseline","M2_auto","full_covar_error"):.6f}, '
        f'with paired difference {at("baseline","M2_auto","full_covar_error_delta"):+.6f} '
        f'{ci("baseline","M2_auto","full_covar_error")}. '
        f'At the same time, credit MSE rises from {at("baseline","M1_zero"):.6f} to {at("baseline","M2_auto"):.6f}. '
        'The covariance point estimate and credit-loading loss therefore have different rankings; '
        'the covariance interval must be considered before describing its small change as an improvement.')
    parts=[]
    for n in [60,112,240]:
        parts.append(f'{n} observations: automatic {100*(ratio("baseline","M2_auto",n=n)-1):+.1f}% and '
                     f'economic {100*(ratio("baseline","economic",n=n)-1):+.1f}%')
    blocks['split_sizes_result']='Baseline credit-MSE changes relative to zero are '+ '; '.join(parts)+'. These are changes in the means, not separate declarations of statistical significance.'
    base_group=group.loc['baseline']
    econ=[];auto=[]
    for spec in SETTINGS:
        s=spec['name']
        econ.append(value(group,('baseline',s,s+'__economic'),'credit_mse_high')<0)
        auto.append(value(group,('baseline',s,s+'__auto'),'credit_mse_low')>0)
    group_text=(f'Under the calibrated generating loadings, {sum(econ)} of five economic-minus-zero intervals are wholly below zero, '
        f'and {sum(auto)} of five automatic-minus-zero intervals are wholly above zero. ')
    for s in ['HCGL','FCGL_sign_adaptive']:
        vals=[value(group,('zero_credit_exposure',s,s+'__'+t),'credit_mse') for t in ['zero','economic']]
        low=value(group,('zero_credit_exposure',s,s+'__economic'),'credit_mse_low')
        high=value(group,('zero_credit_exposure',s,s+'__economic'),'credit_mse_high')
        label='HCGL' if s=='HCGL' else 'FCGL with signs and adaptive weights'
        group_text+=f'With zero true credit exposure, {label} has economic/zero credit-MSE ratio {vals[1]/vals[0]:.2f}, with paired difference interval [{low:+.6f}, {high:+.6f}]. '
    blocks['split_group_result']=group_text.strip()
    def roll(span,m,key='mean_nmse'):
        """Retrieve a rolling estimate with cohort and weighting explicit."""
        return value(rolling,(span,'credit',m),key)
    roll_methods=['M1_zero','M0_ols','M2_auto','economic','rates_credit','equity_rates_credit','M4_free_winner','M5_post_lasso']
    rows=[]
    for m in roll_methods:
        row=[LABELS[m]]
        for s in [0,36]:
            v=roll(s,m);base=roll(s,'M1_zero')
            row += [f'{v:.6f}',f'{100*(v/base-1):+.2f}%']
        rows.append(row)
    blocks['split_rolling']=markdown_table(['Target or control','Uniform NMSE','Change','Span-36 NMSE','Change'],rows)
    text=[]
    for s,label in [(0,'uniform'),(36,'span-36')]:
        for m,name in [('M2_auto','automatic'),('economic','economic')]:
            vals=[value(bootstrap,(s,'credit',m,6),key) for key in ['paired_delta','bootstrap_low','bootstrap_high']]
            text.append(f'The {name}-minus-zero NMSE difference under {label} weights is {vals[0]:+.6f} [{vals[1]:+.6f}, {vals[2]:+.6f}].')
    blocks['split_rolling_result']=' '.join(text)+' These intervals resample dates jointly across all methods and cohort members.'
    return blocks,receipts


def validate_source(source,blocks,refs):
    """Reject stale factor specifications, missing references and changed evidence."""
    for key,expected in blocks.items():
        actual=re.findall(r'<!-- evidence: '+key+r' -->\n(.*?)\n<!-- /evidence -->',source,re.S)
        assert actual==[expected],f'Split-credit evidence drift: {key}'
    forbidden=['nine factors','nine-factor','Credit removed','removed-Credit','28.5%','4.89 times','24,000 datasets','ETF-proxy exercise','362 numerical claims']
    for phrase in forbidden:assert phrase not in source,f'Stale claim: {phrase}'
    assert 'MATF_CUSTOM_IG_HY' in source and all(f in source for f in FACTORS)
    for k,ref in enumerate(refs,1):
        assert f'[{k}]({ref["url"]})' in source,(k,ref['key'])
        assert ref['url'] in source.split('## References')[1]
    assert len(re.findall(r'^\*\*Table \d+\.',source,re.M))==9
    assert len(re.findall(r'^\*\*Figure \d+\.',source,re.M))==2


def verify_evidence(root=ROOT):
    """Require the completed audit gates and exact factor membership in both studies."""
    counts={}
    for name in ['source','inputs','calibration','l1_pilot','l1_confirmation','groups_pilot','groups_confirmation','rolling','numerical_audit','research_support']:
        check(root/name);counts[name]=len(json.loads((root/name/'manifest.json').read_text()))
    for stage in ['l1_pilot','l1_confirmation','groups_pilot','groups_confirmation']:
        audit=json.loads((root/'metric_audit'/f'{stage}.json').read_text())
        assert audit['all_saved_metrics_reconstructed'] and len(audit['negative_controls_rejected'])==2
        result=json.loads((root/stage/'verification.json').read_text())
        assert result['failures']==0 and result['factors']==FACTORS
    for name in ['l1_verification.json','group_verification.json']:
        assert (root/'numerical_audit'/name).exists()
    check(root/'metric_audit')
    empirical=json.loads((root/'metric_audit/rolling_targets_bootstrap.json').read_text())
    assert empirical['target_vectors_checked']==9744 and empirical['independent_stationary_bootstrap_comparisons']==96
    protocol=json.loads((root/'protocol.json').read_text())
    assert protocol['factors']==FACTORS and protocol['generic_credit_absent']
    compare=Path(r'C:\Users\artur\AppData\Local\AgentWork\ARTURDESKTOP\Rosaa\analyses\fi_prior_span36_comparison_20260922_v1')
    fi={}
    for lane in ['index','fund']:
        df=pd.read_csv(compare/f'{lane}_beta_cell_comparison.csv')
        assert set(df.factor)==set(FACTORS) and 'Credit' not in set(df.factor)
        assert df.groupby(['fit_date','ticker']).factor.nunique().eq(12).all()
        fi[lane]=dict(rows=len(df),factors=sorted(df.factor.unique()),all_asset_quarters_complete=True)
    return dict(stage_manifests=counts,fi_factor_membership=fi,protocol_sha256=sha256((root/'protocol.json').read_bytes()).hexdigest())


def figures(root,out):
    """Render paired group comparisons and three distinct LQD credit-loading paths."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.dates import MonthLocator,DateFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    dest=out/'figures';dest.mkdir()
    g=pd.read_csv(root/'groups_confirmation/paired_summary.csv')
    names=[s['name'] for s in SETTINGS]
    labels=['HCGL','HCGL + signs','HCGL + signs\n+ adaptive','FCGL + signs\n+ adaptive','Supplied groups\n+ signs + adaptive']
    fig,axes=plt.subplots(1,2,figsize=(9.2,4.2),sharey=True,layout='constrained')
    for ax,profile,title in zip(axes,PROFILES[:2],['Calibrated 12-factor loadings','Zero true IG/HY/EM exposures']):
        ax.axvline(0,color='#727272',lw=.8)
        for target,color,offset,label in [('auto','#376eaa',-.12,'Automatic minus zero'),('economic','#b56035',.12,'Economic minus zero')]:
            p=g.loc[g.profile.eq(profile)&g.method.str.endswith('__'+target)].set_index('setting').loc[names]
            x=p.credit_mse_delta.to_numpy()
            ax.errorbar(x,np.arange(5)+offset,xerr=np.vstack([x-p.credit_mse_low,p.credit_mse_high-x]),fmt='o',color=color,label=label,capsize=3,ms=4)
        ax.set_title(title,fontsize=11,pad=11);ax.set_xlabel('Change in aggregate IG/HY/EM\nloading MSE',fontsize=10)
        ax.grid(axis='x',alpha=.18);ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,3))
    axes[0].set_yticks(np.arange(5),labels,fontsize=10);axes[0].invert_yaxis()
    axes[1].legend(loc='best',fontsize=9,frameon=False)
    fig.savefig(dest/'groups.png',bbox_inches='tight');plt.close(fig)
    r=pd.read_csv(root/'rolling/rolling_results.csv')
    p=r.loc[r.asset.eq('LQD')&r.span.eq(0)].copy();p['date']=pd.to_datetime(p.date)
    fig,axes=plt.subplots(3,1,figsize=(9.2,6.4),sharex=True,layout='constrained')
    for j,ax in zip([2,3,4],axes):
        for method,color,label in [('M1_zero','#303b46','Zero'),('M2_auto','#376eaa','Automatic'),('economic','#b56035','Economic')]:
            b=p.loc[p.method.eq(method)].sort_values('date')
            ax.plot(b.date,b[f'beta{j}'],label=label,color=color,lw=1.5)
        ax.axhline(0,color='#b0b0b0',lw=.65);ax.set_ylabel(FACTORS[j]+'\nnative beta');ax.grid(alpha=.18)
    axes[0].legend(ncol=3,frameon=False,loc='best');axes[0].set_title('LQD: held-out month and loadings fitted through the preceding month')
    axes[-1].xaxis.set_major_locator(MonthLocator(interval=4));axes[-1].xaxis.set_major_formatter(DateFormatter('%b %Y'))
    fig.savefig(dest/'rolling.png',bbox_inches='tight');plt.close(fig)
    return {p.name:sha256(p.read_bytes()).hexdigest() for p in dest.glob('*.png')}
