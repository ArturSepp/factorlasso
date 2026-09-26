"""Evidence tables, tagged values and CAS build for the practical FI validation edition.

manuscript.md is the only editorial source. ``evidence`` rebuilds every numerical table and
``values`` every tagged in-text number from the evidence root; ``build`` verifies both against
the manuscript, rejects a corrupted copy of each, and renders the PDF. Nothing here writes prose.
"""
from pathlib import Path
import hashlib, json, re, shutil, subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from .fi_validation_exhibits import SHORT_NAMES
from .fi_validation_stress import POLICY_NAMES,policy_cell

ARMS=['Z','A','E1','E2']
COLORS=['#9aa4ad','#3e6595','#cf9c45','#23867e']
# Zero-penalty controls refit a policy's final signs with lambda=0; 'Zero penalty' uses the Zero prior's signs.
ZERO_PENALTY='Z_lambda_zero'
DISPLAY={'Z_lambda_zero':'Zero penalty','E2_lambda_zero':'Conditional zero penalty'}
MC_ARMS=ARMS+[ZERO_PENALTY]
MC_COLORS=COLORS+['#7a5c99']
FEATURES=['Rates','Credit IG','Credit HY','Credit EM','Inflation','Equity']

def sha(p):
    """Return file identity for numerical and editorial evidence."""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(p,obj):
    """Write auditable JSON without nonfinite values."""
    Path(p).write_text(json.dumps(obj,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8')

def table(headers,rows):
    """Build a stable, human-readable Markdown evidence table."""
    headers=[policy_cell(v) for v in headers]
    rows=[[policy_cell(str(v)[1:] if re.fullmatch(r'-0\.0+',str(v)) else str(v)) for v in row] for row in rows]
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])

def read(root,path,**kwargs):
    """Read saved evidence without changing numerical precision."""
    return pd.read_csv(root/path,float_precision='round_trip',**kwargs)

def evidence(root,exhibit_root=None,stress_root=None):
    """Reconstruct every published numerical table from validated stage records."""
    ep=read(root,'summary/endpoint.csv');roster=read(root,'inputs/roster.csv',index_col=0)
    cma=read(root,'summary/cmas.csv');blocks={};claims={}
    short={t:SHORT_NAMES[t] for t in ['LEGATRUH Index','LGCPTRUH Index','H23059US Index',
        'H04386US Index','LF94TRUH Index','H24641US Index']}
    rows=[]
    for t,label in short.items():
        for a in ARMS:
            r=ep[(ep.ticker==t)&(ep.arm==a)].iloc[0]
            rows.append([label if a=='Z' else '',a,*[f'{r[k]:.3f}' for k in FEATURES],f'{100*r.fit_r2_owner:.2f}'])
    blocks['cases']=table(['Index','Arm','Rates','IG','HY','EM','Infl.','Equity','Fit R2 %'],rows)
    levels=[]
    for lane in ['indices','funds']:
        scores=read(root,f'summary/{lane}/asset_scores.csv')
        for span in [60,36]:
            s=scores[scores.span==span]
            vals=s.groupby('arm').oos_r2.mean()
            levels.append([lane.title(),str(span),str(s.ticker.nunique()),*[f'{100*vals[a]:.2f}' for a in ARMS]])
            claims[f'{lane}_span{span}_mean_oos_r2']=vals.to_dict()
    blocks['oos']=table(['Panel','Span','Assets','Z %','A %','E1 %','E2 %'],levels)
    rows=[]
    for lane in ['indices','funds']:
        boot=read(root,f'summary/{lane}/bootstrap.csv').query('span==60 and block==4')
        for arm,comp in [('A','Z'),('E1','A'),('E2','A'),('E2','Z'),('E2',ZERO_PENALTY)]:
            r=boot[(boot.arm==arm)&(boot.comparator==comp)].iloc[0]
            rows.append([lane.title(),str(int(r.cohort)),arm+' minus '+DISPLAY.get(comp,comp),f'{100*r.delta_r2:+.2f}',f'[{100*r.low:+.2f}, {100*r.high:+.2f}]'])
    blocks['bootstrap']=table(['Panel','Cohort','Contrast','Gain, pp','95% block interval, pp'],rows)
    rows=[]
    for t,label in short.items():
        s=cma[cma.ticker==t].set_index('arm');d=s.loc['E2','base_total_cma']-s.loc['A','base_total_cma']
        rows.append([label,*[f'{100*s.loc[a,"base_total_cma"]:.2f}' for a in ARMS],f'{10000*d:+.1f}',f'{100*s.loc["E2","vol"]:.2f}'])
    blocks['cma_cases']=table(['Index','Z %','A %','E1 %','E2 %','E2-A, bp','E2 vol %'],rows)
    p=read(root,'summary/fixed_portfolio.csv').set_index('arm')
    blocks['portfolio']=table(['Arm','Total CMA %','Excess CMA %','Model vol %','Sharpe (rf=0)','Excess Sharpe'],
        [[a,f'{100*p.loc[a,"total_cma"]:.2f}',f'{100*p.loc[a,"excess_cma"]:.2f}',f'{100*p.loc[a,"vol"]:.2f}',
          f'{p.loc[a,"total_cma"]/p.loc[a,"vol"]:.2f}',f'{p.loc[a,"excess_cma"]/p.loc[a,"vol"]:.2f}'] for a in ARMS])
    claims['fixed_portfolio']=p.to_dict('index')
    rows=[]
    for label in ['Z_detected','E2_detected','Z_prior','E2_prior']:
        d=read(root,f'R1/indices/{label}_diagnostics.csv',index_col=0)
        b=read(root,f'R1/indices/{label}_betas.csv',index_col=0)
        rows.append([label.replace('_',' / '),f'{100*d.fit_r2_owner.mean():.2f}',
            f'{b.loc["LF94TRUH Index","Inflation"]:.3f}',f'{100*d.loc["LEGATRUH Index","fit_r2_owner"]:.2f}'])
    blocks['mechanism']=table(['Centre / sign layer','Mean fit R2 %','Global IL Inflation','IG aggregate R2 %'],rows)
    replay=read(root,'R1/full_replay/comparison.csv',index_col=0)
    blocks['spillover']=table(['Full monthly block','Univariate IL fit %','Joint IL fit %','Direct factor CMA change, bp'],
        [[short.get(t,t),f'{100*replay.loc[t,"r2_univariate"]:.2f}',f'{100*replay.loc[t,"r2_current"]:.2f}',f'{replay.loc[t,"factor_cma_change_bp"]:+.1f}']
         for t in ['LF94TRUH Index','LEGATRUH Index','LGCPTRUH Index','H23059US Index','H04386US Index']])
    claims['full_universe_IL_replay']=replay.loc[list(short.keys())].to_dict('index')
    rows=[]
    for lane in ['indices','funds']:
        st=read(root,f'summary/{lane}/stability.csv').query('span==60').set_index('arm')
        rows.append([lane.title(),*[f'{st.loc[a,"mean"]:.4f}' for a in MC_ARMS]])
    blocks['stability']=table(['Panel','Z','A','E1','E2','Zero penalty'],rows)
    duration=read(root,'summary/oad_rank_association.csv').set_index('arm')
    blocks['oad']=table(['Arm','Paired observations','Spearman: Rates / OAD'],
        [[a,str(int(duration.loc[a,'n'])),f'{duration.loc[a,"spearman_rates_oad"]:+.3f}'] for a in ARMS])
    rows=[]
    for lane in ['indices','funds']:
        s=read(root,f'summary/{lane}/asset_scores.csv').query('span==60')
        for (group,kind),d in s.groupby(['group','kind'],sort=False):
            vals=d.groupby('arm').oos_r2.mean()
            rows.append([group,kind,str(d.ticker.nunique()),*[f'{100*vals[a]:.1f}' for a in ARMS]])
    blocks['categories']=table(['Group','Kind','n','Z %','A %','E1 %','E2 %'],rows)
    for lane in ['indices','funds']:
        s=read(root,f'summary/{lane}/asset_scores.csv').query('span==60').pivot(index='ticker',columns='arm',values='oos_r2')
        rows=[]
        for t,r in roster[roster.lane==lane].iterrows():
            e=ep[(ep.ticker==t)&(ep.arm=='E2')].iloc[0]
            cm=cma[(cma.ticker==t)&(cma.arm=='E2')]
            rows.append([SHORT_NAMES[t],t.replace(' Index','') if lane=='indices' else r['group'],*[f'{e[k]:.3f}' for k in ['Rates','Credit IG','Credit HY','Credit EM','Inflation','Equity']],
                f'{100*(s.loc[t,"E2"]-s.loc[t,"A"]):+.1f}',f'{100*cm.iloc[0].base_total_cma:.2f}' if len(cm) else '-'])
        headers=['Short name','Bloomberg' if lane=='indices' else 'Group','Rates','IG','HY','EM','Infl.','Equity','OOS gain, pp','CMA %']
        if lane=='indices':
            assert exhibit_root is not None,'The current edition requires computed frontier evidence'
            assets=read(Path(exhibit_root),'assets.csv',index_col=0)
            for row,t in zip(rows,roster[roster.lane==lane].index):row.append(f'{100*assets.loc[t,"factor_vol"]:.2f}')
            headers.append('Factor vol %')
        blocks['roster_'+lane]=table(headers,rows)
    mc=read(root,'R4/draws.csv')
    # Loading RMSE (x100) per replication: premium-free, unlike CMA error.
    mc['loading_rmse']=100*np.sqrt(mc.beta_mse)
    rows=[]
    for (case,n),df in mc.groupby(['case','n'],sort=False):
        vals=df.groupby('arm').loading_rmse.agg(['mean','std','count']);se=vals['std']/np.sqrt(vals['count'])
        rows.append([case.replace('credit_','Credit ').replace('il_','IL ').replace('_',' '),str(n),
            *[f'{vals.loc[a,"mean"]:.2f}' for a in MC_ARMS],f'{vals.loc["Oracle","mean"]:.2f}',f'{se.loc["E2"]:.2f}'])
    blocks['mc']=table(['Design','Months','Z','A','E1','E2','Zero penalty','Oracle','E2 MCSE'],rows)
    # Paired uncertainty uses seed-level errors, not assets as independent replications.
    mcrows=[]
    for (case,n),df in mc.groupby(['case','n'],sort=False):
        wide=df.pivot(index='seed',columns='arm',values='abs_cma_bp');delta=wide.E2-wide.A
        mcrows.append(dict(case=case,n=int(n),E2_minus_A=float(delta.mean()),mcse=float(delta.std(ddof=1)/np.sqrt(len(delta)))))
    claims['MC_paired_cma_error']=mcrows
    focal=read(root,'summary/mc_focal_cases.csv')
    rows=[]
    for case,asset in [('credit_high','IG'),('credit_wrong','IG'),('credit_omitted','Convertible'),('il_weak','IL')]:
        d=focal[(focal.case==case)&(focal.asset==asset)].set_index('arm')
        rows.append([case.replace('credit_','').replace('il_','IL '),asset,*[f'{d.loc[a,"mean_absolute_cma_error_bp"]:.1f}' for a in MC_ARMS]])
    blocks['mc_focal']=table(['Design','Affected asset','Z','A','E1','E2','Zero penalty'],rows)
    claims['MC_focal_cases']=focal.where(focal.notna(),None).to_dict('records')
    # pandas float columns retain NaN under where; JSON must use explicit nulls.
    claims['MC_focal_cases']=[{k:(None if pd.isna(v) else v) for k,v in row.items()} for row in claims['MC_focal_cases']]
    if stress_root is not None:
        from .fi_validation_stress import table_block,verify as verify_stress
        blocks['stress_scenarios']=table_block(stress_root,table)
        claims['stress_checks']=verify_stress(stress_root)
    if (Path(root)/'sign_revision').exists():
        from .sign_revision_paper import evidence as revision_evidence
        extra, extra_claims=revision_evidence(root,table)
        blocks.update(extra);claims.update(extra_claims)
    return blocks,claims

def verify(source,blocks):
    """Reject numerical drift before generating the manuscript mirror."""
    for key,value in blocks.items():
        found=re.findall(r'<!-- evidence: '+key+r' -->\n(.*?)\n<!-- /evidence -->',source,re.S)
        assert found==[value],f'Evidence mismatch: {key}'

def figures(root,out,exhibit_root,stress_root):
    """Render ten evidence-linked scientific exhibits in native PDF and PNG."""
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
        'savefig.bbox':'tight','pdf.fonttype':42})
    def save(fig,name):
        """Save vector master and readable preview from the same figure."""
        fig.savefig(out/(name+'.pdf'));fig.savefig(out/(name+'.png'),dpi=170);plt.close(fig)
    ep=read(root,'summary/endpoint.csv');roster=read(root,'inputs/roster.csv',index_col=0).query("lane=='indices'")
    fig,axes=plt.subplots(1,4,figsize=(9.8,8.8),sharey=True,layout='constrained')
    for ax,a in zip(axes,ARMS):
        d=ep[(ep.arm==a)&(ep.lane=='indices')].set_index('ticker').loc[roster.index,FEATURES]
        im=ax.imshow(d,norm=TwoSlopeNorm(vmin=-.5,vcenter=0,vmax=1.5),cmap='RdBu_r',aspect='auto')
        ax.set_title(POLICY_NAMES[a].replace(' prior','\nprior'));ax.set_xticks(range(6),['Rates','IG','HY','EM','Infl.','Eq.'],rotation=50)
        ax.set_yticks(range(30),[SHORT_NAMES[t] for t in roster.index],fontsize=9)
    fig.colorbar(im,ax=axes,label='Beta in native factor-return units',shrink=.6)
    save(fig,'fi_loadings')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.1),sharey=True,layout='constrained')
    for ax,lane in zip(axes,['indices','funds']):
        df=read(root,f'summary/{lane}/asset_scores.csv').query('span==60')
        groups=list(dict.fromkeys(df.group));width=.19
        for j,a in enumerate(ARMS):
            vals=df[df.arm==a].groupby('group').oos_r2.mean().reindex(groups)*100
            ax.bar(np.arange(len(groups))+(j-1.5)*width,vals,width,label=POLICY_NAMES[a],color=COLORS[j])
        ax.set_xticks(range(len(groups)),groups,rotation=35);ax.set_title(lane.title());ax.axhline(0,color='#666',lw=.6)
        ax.set_ylabel('Mean pooled held-out R² (%)');ax.grid(axis='y',alpha=.15);ax.legend(ncol=2,fontsize=8)
    save(fig,'fi_oos')
    cma=read(root,'summary/cmas.csv');premia=read(root,'inputs/premia.csv',index_col=0).iloc[0]
    a=ep.query("lane=='indices' and arm=='A'").set_index('ticker');e=ep.query("lane=='indices' and arm=='E2'").set_index('ticker')
    delta=(e[FEATURES]-a[FEATURES]).mul(premia,axis=1).reindex(roster.index)[FEATURES]*10000
    fig,ax=plt.subplots(figsize=(10.5,7.8),layout='constrained');pos=np.zeros(30);neg=pos.copy()
    for j,k in enumerate(FEATURES):
        v=delta[k].to_numpy();left=np.where(v>=0,pos,neg);ax.barh(range(30),v,left=left,label=k)
        pos+=np.maximum(v,0);neg+=np.minimum(v,0)
    total=cma.pivot(index='ticker',columns='arm',values='base_total_cma')
    ax.scatter((total.E2-total.A).reindex(roster.index)*10000,range(30),marker='D',color='black',s=17,label='Total CMA change')
    ax.set_yticks(range(30),[SHORT_NAMES[t] for t in roster.index],fontsize=9);ax.invert_yaxis()
    ax.axvline(0,color='black',lw=.6);ax.set_xlabel('Conditional prior minus Max-R2 prior: annual expected return (basis points)');ax.legend(ncol=4,fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.09))
    save(fig,'fi_cma')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4),layout='constrained')
    for ax,lane in zip(axes,['indices','funds']):
        b=pd.concat([pd.read_csv(p) for p in (root/f'R3/{lane}').glob('60_*_betas.csv')])
        target='LF94TRUH Index' if lane=='indices' else 'TIP US Equity'
        for j,a in enumerate(ARMS):
            d=b[(b.ticker==target)&(b.factor=='Inflation')&(b.arm==a)].sort_values('fit_date')
            ax.plot(pd.to_datetime(d.fit_date),d.beta,label=POLICY_NAMES[a],color=COLORS[j],lw=1.6)
        ax.set_title(SHORT_NAMES[target]);ax.set_ylabel('Inflation beta');ax.axhline(0,color='#666',lw=.6);ax.legend(ncol=2,fontsize=8)
    save(fig,'fi_il_path')
    mc=read(root,'R4/draws.csv');mc['loading_rmse']=100*np.sqrt(mc.beta_mse)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5),layout='constrained')
    for ax,metric,label in zip(axes,['loading_rmse','abs_cma_bp'],['Root mean squared loading error (x100)','Mean absolute CMA error (bp)']):
        d=mc.query('n==120');cases=list(dict.fromkeys(d.case));width=.15
        for j,a in enumerate(MC_ARMS):
            vals=d[d.arm==a].groupby('case')[metric].agg(['mean','std','count']).reindex(cases)
            ax.bar(np.arange(len(cases))+(j-2)*width,vals['mean'],width,yerr=1.96*vals['std']/np.sqrt(vals['count']),
                label=POLICY_NAMES.get(a,DISPLAY.get(a,a)),color=MC_COLORS[j],capsize=2)
        ax.set_xticks(range(len(cases)),[c.replace('credit_','').replace('il_','IL ') for c in cases],rotation=35)
        ax.set_ylabel(label);ax.legend(ncol=2,fontsize=8);ax.grid(axis='y',alpha=.15)
    save(fig,'fi_mc')
    from .fi_validation_exhibits import figures as extra_figures
    extra_figures(exhibit_root,out)
    from .fi_validation_stress import figures as stress_figures
    stress_figures(stress_root,out)
    if (Path(root)/'sign_revision').exists():
        from .sign_revision_paper import figures as revision_figures
        revision_figures(root,out)
    return {p.name:sha(p) for p in out.iterdir() if p.is_file()}

VALUE_PATTERN=r'<!-- value: ([A-Za-z0-9_.-]+) -->(.*?)<!-- /value -->'

def values(root,stress_root):
    """Evidence-bound numbers quoted in the prose, keyed by the manuscript's value tags."""
    root=Path(root);vals={}
    scores=read(root,'sign_revision/oos_summary.csv').query('span==60 and variant=="date"').set_index(['lane','arm']).monthly_rmse_pct
    for lane in ['indices','funds']:
        z,z0,e2,e20=(scores[lane,a] for a in ['Z','Z_lambda_zero','E2','E2_lambda_zero'])
        vals.update({f'rmse_{lane}_zero_prior':f'{z:.3f}',f'rmse_{lane}_zero_penalty':f'{z0:.3f}',
            f'rmse_{lane}_conditional':f'{e2:.3f}',f'rmse_{lane}_conditional_zero_penalty':f'{e20:.3f}',
            f'penalty_cost_{lane}_zero_prior':f'{100*(z/z0-1):.1f}',
            f'conditional_vs_zero_penalty_{lane}':f'{100*abs(e2/z0-1):.1f}',
            f'prior_signs_vs_detected_{lane}':f'{100*abs(e20/z0-1):.1f}'})
        boot=read(root,f'summary/{lane}/bootstrap.csv')
        boot=boot[(boot.span==60)&(boot.block==4)&(boot.arm=='E2')&(boot.comparator==ZERO_PENALTY)]
        assert len(boot)==1,f'{lane}: rerun the oos stage to create the zero-penalty bootstrap contrast'
        r=boot.iloc[0]
        vals[f'boot_gain_{lane}_conditional_vs_zero_penalty']=f'{100*r.delta_r2:+.2f}'
        vals[f'boot_ci_{lane}_conditional_vs_zero_penalty']=f'[{100*r.low:+.2f}, {100*r.high:+.2f}]'
        st=read(root,f'summary/{lane}/stability.csv').query('span==60').set_index('arm')['mean']
        vals[f'stability_{lane}_conditional']=f'{st["E2"]:.4f}'
        vals[f'stability_{lane}_zero_penalty']=f'{st[ZERO_PENALTY]:.4f}'
        fit=read(root,f'R1/{lane}/E2_diagnostics.csv',index_col=0).fit_r2_owner
        vals[f'endpoint_median_r2_{lane}']=f'{fit.median():.2f}'
    focal=read(root,'summary/mc_focal_cases.csv').query('case=="credit_wrong" and asset=="IG"').set_index('arm')
    vals.update(mc_wrong_beta_mse_conditional=f'{focal.loc["E2","mean_beta_mse"]:.4f}',
        mc_wrong_beta_mse_automatic=f'{focal.loc["A","mean_beta_mse"]:.4f}',
        mc_wrong_cma_conditional=f'{focal.loc["E2","mean_absolute_cma_error_bp"]:.1f}',
        mc_wrong_cma_automatic=f'{focal.loc["A","mean_absolute_cma_error_bp"]:.1f}')
    stress=read(Path(stress_root),'march2020_summary.csv',index_col=0)
    for name,label in [('Global IL','global_il'),('Global IG agg','ig_agg'),('Global HY','global_hy')]:
        for arm in ['A','E2']:vals[f'stress_mar2020_{label}_{arm}']=f'{100*stress.loc[name,arm]:+.2f}'
    from . import fi_validation as f
    # Section 2.3 states the calibration 1e-5 * 318 rows / 30.49924 complete-grid mass (mass rounded to 7 digits).
    assert abs(1e-5*318/30.49924/f.MAIN_LAMBDA-1)<1e-6,f.MAIN_LAMBDA
    vals['monthly_penalty']=repr(float(f.MAIN_LAMBDA))
    premia=read(root,'inputs/premia.csv',index_col=0).iloc[0]
    for factor,label in [('Credit IG','ig'),('Credit HY','hy'),('Credit EM','em')]:
        vals[f'premium_{label}']=f'{100*premia[factor]:.4f}'
    return vals

def fill_values(source,vals):
    """Replace every value tag's text with its evidence value; unknown keys raise."""
    return re.sub(VALUE_PATTERN,lambda m:f'<!-- value: {m[1]} -->{vals[m[1]]}<!-- /value -->',source,flags=re.S)

def verify_values(source,vals):
    """Every tagged number must equal its evidence value, and every value must be cited."""
    tags=re.findall(VALUE_PATTERN,source,re.S)
    unknown={k for k,_ in tags}-set(vals);unused=set(vals)-{k for k,_ in tags}
    assert not unknown,f'Unknown value tags: {sorted(unknown)}'
    assert not unused,f'Evidence values not cited: {sorted(unused)}'
    wrong=[k for k,v in tags if v!=vals[k]]
    assert not wrong,f'Value drift: {sorted(set(wrong))}'

def build(out,publish=False,here=None):
    """Verify current editorial source, compile CAS, and record convergence checks."""
    from . import build_latex as parser
    here=Path(here or Path(__file__).resolve().parents[1])
    meta=json.loads((here/'paper_sources.json').read_text(encoding='utf-8'));root=Path(meta['study_root'])
    exhibit_root=Path(meta['exhibit_root'])
    from .fi_validation_exhibits import verify as verify_exhibits
    exhibit_checks=verify_exhibits(exhibit_root)
    stress_root=Path(meta['stress_root'])
    source=(here/'manuscript.md').read_text(encoding='utf-8');blocks,claims=evidence(root,exhibit_root,stress_root)
    verify(source,blocks)
    vals=values(root,stress_root);verify_values(source,vals)
    for key in vals:
        bad=re.sub(r'(<!-- value: '+re.escape(key)+r' -->)(.*?)(<!-- /value -->)',lambda m:m[1]+'ALTERED'+m[3],source,count=1,flags=re.S)
        try:verify_values(bad,vals)
        except AssertionError:pass
        else:raise AssertionError('Value corruption accepted: '+key)
    for name,value in json.loads((root/'evidence_manifest.json').read_text()).items():assert sha(root/name)==value,name
    rejected=[]
    for key,value in blocks.items():
        bad=source.replace(value,value.replace('|','| ALTERED ',1))
        try:verify(bad,blocks)
        except AssertionError:rejected.append(key)
        else:raise AssertionError('Numerical corruption accepted: '+key)
    assert len(rejected)==len(blocks)
    assert out.is_absolute() and 'OneDrive' not in out.parts
    out.mkdir(parents=True,exist_ok=False)
    figure_hashes=figures(root,out/'figures',exhibit_root,stress_root)
    template=here.parent/'sign_pooling_2026/paper'
    for name in ['cas-sc.cls','cas-common.sty','cas-model2-names.bst']:shutil.copy2(template/name,out/name)
    refs=meta['references'];abstract=source.split('**Abstract.** ',1)[1].split('## 1.',1)[0].strip()
    abstract=re.sub(r'<!--.*?-->','',abstract,flags=re.S)
    preamble=r'''\documentclass[a4paper,review]{cas-sc}
\usepackage[authoryear,longnamesfirst]{natbib}
\usepackage{amssymb,bm,booktabs,adjustbox,placeins,setspace}
\usepackage{colortbl}
\ExplSyntaxOn
\cs_set:Npn \__first_footerline: {\small\itshape Research~note,~26~September~2026}
\ExplSyntaxOff
\renewcommand{\printorcid}{}
\hypersetup{pdftitle={Selecting Priors for Fixed-Income Factor Models},pdfauthor={Artur Sepp}}
\begin{document}
\let\WriteBookmarks\relax
\def\floatpagepagefraction{1}
\def\textpagefraction{.001}
\shorttitle{Selecting priors for fixed-income factor models}
\shortauthors{Sepp}
\title[mode=title]{Selecting Priors for Fixed-Income Factor Models: Validation and Capital Market Assumptions}
\tnotemark[1]
\author{Artur Sepp}
\tnotetext[1]{Research note. Revised 26 September 2026. Internal empirical draft.}
\begin{abstract}
'''
    front=preamble+parser.inline(' '.join(abstract.split()),refs)+r'''
\end{abstract}
\begin{highlights}
\item Thirty indices and twelve funds test four practical prior-selection policies.
\item Unpenalized sign-constrained fits match the priors' held-out explanation.
\item Duration diagnostics, fund uncertainty and wrong mappings qualify the gains.
\item Shared penalties transmit IL prior changes to untouched IG estimates.
\end{highlights}
\begin{keywords}
fixed income \sep factor models \sep prior selection \sep capital market assumptions
\end{keywords}
\maketitle
'''
    body=parser.body_to_tex(source,refs)
    from .fi_validation_stress import shade_loading_tables,wrap_policy_headers
    body=shade_loading_tables(body)
    body=wrap_policy_headers(body)
    # Keep the complete identifier lookup readable without shrinking unrelated tables.
    body=body.replace(r'\caption{Thirty-index conditioned estimates',
        r'\setlength{\tabcolsep}{3pt}'+'\n'+r'\caption{Thirty-index conditioned estimates')
    body=body.replace(r'\begin{tabular}{lrrrrrrrrrr}',r'\begin{tabular}{llrrrrrrrrr}')
    body=body.replace(r'\section{Appendix:',r'\FloatBarrier\section{Appendix:')
    body=body.replace(r'\subsection{Historical stress scenarios}',r'\FloatBarrier\subsection{Historical stress scenarios}')
    body=body.replace(r'\section*{Computational provenance',r'\FloatBarrier\clearpage\section*{Computational provenance')
    tex='% Generated from manuscript.md by build_latex.py; edit Markdown.\n'+front+body+r'''
\FloatBarrier
\begingroup
\setstretch{1}
\bibliographystyle{cas-model2-names}
\bibliography{refs}
\endgroup
\end{document}
'''
    (out/'article.tex').write_text(tex,encoding='utf-8');(out/'refs.bib').write_text(parser.bibliography(refs),encoding='utf-8')
    def command(args):
        """Run a compiler with captured, reviewable output."""
        result=subprocess.run(args,cwd=out,capture_output=True,text=True)
        (out/(Path(args[0]).stem+'_stdout.txt')).write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if result.returncode:raise RuntimeError(result.stdout[-5000:]+result.stderr)
    binary=Path('C:/Users/artur/AppData/Local/Programs/MiKTeX/miktex/bin/x64')
    latex=[str(binary/'pdflatex.exe'),'-interaction=nonstopmode','-halt-on-error','article.tex']
    command(latex);command([str(binary/'bibtex.exe'),'article'])
    for i in range(5):
        previous=sha(out/'article.aux');command(latex)
        if sha(out/'article.aux')==previous:break
    else:raise AssertionError('Auxiliary references did not converge')
    log=(out/'article.log').read_text(encoding='utf-8',errors='replace')
    assert not re.search(r'undefined references|Citation .* undefined|Reference .* undefined',log)
    overflows=re.findall(r'Overfull \\hbox.*',log)
    assert len(overflows)==1 and '117.0831pt' in overflows[0],overflows
    assert 'Overfull \\vbox' not in log
    if publish:
        for name in ['article.tex','refs.bib']:shutil.copy2(out/name,here/'paper'/name)
    for name in ['manuscript.md','paper_sources.json','build_latex.py','fi_validation.py','fi_validation_metrics.py','fi_validation_synthetic.py','fi_validation_paper.py','fi_validation_exhibits.py','fi_validation_stress.py','sign_revision.py','sign_revision_paper.py']:
        shutil.copy2((here/'replication' if name.endswith('.py') else here)/name,out/name)
    for name in ['protocol.json','numerical_claims.json','evidence_manifest.json']:shutil.copy2(root/name,out/name)
    dump(out/'verification.json',dict(status='compiled_pending_visual_review',verified_blocks=list(blocks),verified_values=list(vals),
        rejected_corruptions=rejected,source_sha256=sha(here/'manuscript.md'),pdf_sha256=sha(out/'article.pdf'),
        figure_hashes=figure_hashes,aux_convergence_passes=i+1,known_CAS_keyword_diagnostic=overflows,
        exhibit_checks=exhibit_checks,exhibit_manifest_sha256=sha(exhibit_root/'manifest.json'),
        stress_checks=claims['stress_checks'],stress_manifest_sha256=sha(stress_root/'manifest.json'),
        shaded_loading_cells=396))
    print('Compiled',out/'article.pdf',flush=True)
