"""Build the evidence-checked prior-target manuscript in the sign paper's CAS style.

Markdown is the editorial source; paper/article.tex is its generated LaTeX mirror.
All compilation state and plots are staged under the explicit C-local output root.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
import numpy as np
import pandas as pd
from . import build_paper as legacy
from . import split_credit_paper as replacement
from . import core_etf_examples as examples

HERE = Path(__file__).resolve().parent
PAPER_ROOT = HERE.parent
BASE = Path(r'C:\Users\artur\AppData\Local\AgentWork\ARTURDESKTOP\Rosaa\analyses')
COMPARE = BASE / 'fi_prior_span36_comparison_20260922_v1'


def digest(path):
    """Return a file's SHA-256 digest."""
    return sha256(path.read_bytes()).hexdigest()


def read_json(path):
    """Read UTF-8 JSON."""
    return json.loads(path.read_text(encoding='utf-8'))


def table(headers, rows):
    """Produce the same Markdown table representation as the legacy verifier."""
    return legacy.markdown_table(headers, rows)


def fi_blocks():
    """Reconstruct manuscript exhibits from matched asset and beta records."""
    summary = read_json(COMPARE/'summary.json')
    secondary = read_json(COMPARE/'secondary_summary.json')
    blocks = {}
    levels=[]
    for lane, filename, kind in [('Indices','index','Index'),('Active funds','fund','Active fund'),('Passive funds','fund','Passive fund')]:
        df=pd.read_csv(COMPARE/f'{filename}_asset_comparison.csv')
        df=df.loc[df.kind.eq(kind)]
        expected=secondary['absolute_oos_r2_levels_common_benchmark'][kind]
        vals=[]
        for span in (60,36):
            for arm in ('P0','P1'):
                value=df[f'r2_{span}_common_{arm}'].mean()
                np.testing.assert_allclose(value,expected[f'mean_r2_{arm.lower()}_{span}'],atol=1e-13)
                vals.append(value)
        np.testing.assert_allclose(df.prior_gain_span36_minus60,
            df.p1_span36_minus60_r2-df.p0_span36_minus60_r2,atol=1e-13)
        levels.append([lane,str(len(df)),*[f'{x:.5f}' for x in vals[:2]],f'{vals[1]-vals[0]:+.5f}',
                       *[f'{x:.5f}' for x in vals[2:]],f'{vals[3]-vals[2]:+.5f}'])
    blocks['fi_levels']=table(['Panel','n','P0, 60','P1, 60','Gain, 60','P0, 36','P1, 36','Gain, 36'],levels)
    beta_values=[]
    temporal=[]
    for lane in ('index','fund'):
        df=pd.read_csv(COMPARE/f'{lane}_beta_cell_comparison.csv')
        raw=[]
        for span in (60,36):
            delta=df[f'beta{span}_P1']-df[f'beta{span}_P0']
            np.testing.assert_allclose(delta,df[f'prior_beta_delta_{span}'],atol=1e-13)
            absolute=delta.abs()
            l1=absolute.groupby([df.fit_date,df.ticker]).sum().mean()
            raw.append([absolute.mean(),absolute.median(),absolute.quantile(.95),l1,(absolute>.01).mean()])
        beta_values.append(raw)
        # Recompute consecutive-quarter movement from the underlying beta cells.
        df['fit_date']=pd.to_datetime(df.fit_date)
        df=df.sort_values(['ticker','factor','fit_date'])
        group=df.groupby(['ticker','factor'])
        consecutive=df.fit_date.dt.to_period('Q').astype('int64').sub(
            group.fit_date.shift().dt.to_period('Q').astype('int64')).eq(1)
        saved=pd.read_csv(COMPARE/f'{lane}_turnover_comparison.csv')
        for arm in ('P0','P1'):
            vals=[]
            for span in (60,36):
                value=group[f'beta{span}_{arm}'].diff().abs().loc[consecutive].mean()
                ref=saved.loc[saved.span.eq(span)&saved.arm.eq(arm),'mean_abs_qoq_beta_change'].iloc[0]
                np.testing.assert_allclose(value,ref,atol=1e-13)
                vals.append(value)
            temporal.append([lane.title(),arm,*[f'{v:.5f}' for v in vals]])
    rows=[]
    names=['Mean absolute shift','Median absolute shift','95th percentile','Mean asset-quarter L1','Share above 0.01']
    for k,name in enumerate(names):
        fmt=(lambda x:f'{100*x:.1f}%') if k==4 else (lambda x:f'{x:.6f}' if k==1 else f'{x:.5f}')
        rows.append([name,*[fmt(beta_values[lane][span][k]) for lane in (0,1) for span in (0,1)]])
    blocks['fi_betas']=table(['Statistic','Indices, 60','Indices, 36','Funds, 60','Funds, 36'],rows)
    blocks['fi_turnover']=table(['Panel','Arm','Span 60','Span 36'],temporal)
    boot=secondary['bootstrap_headline']
    rows=[]
    for key,label in [('active_funds','Active funds'),('indices','Indices')]:
        if key not in boot:
            key=next(k for k in boot if 'index' in k or 'cma' in k)
        b=boot[key];p=b['point'];ci=b['block4']
        interval=lambda key:'['+', '.join(f'{v:+.5f}' for v in ci[key]['percentile_95'])+']'
        rows.append([label,f"{p['prior_gain_36']:+.5f}",interval('prior_gain_36'),
                     f"{p['extra_prior_gain_36_vs60']:+.5f}",interval('extra_prior_gain_36_vs60')])
    blocks['fi_bootstrap']=table(['Panel','Gain at 36','95% interval','Gain at 36 minus 60','95% interval'],rows)
    hybrid=pd.read_csv(COMPARE/'hybrid_asset_outcomes.csv')
    labels={'BGCLTRUH Index':'Capital debt index','H13203US Index':'Corporate subordinated index',
        'H24641US Index':'Convertible index','H30902US Index':'Global CoCo index','IBXXC1D3 Index':'AT1 CoCo index',
        'CWB US Equity':'Convertible ETF','NBCHEPA ID Equity':'Neuberger corporate hybrid',
        'OBJCGAU FP Equity':'Lazard convertible fund'}
    rows=[]
    for _,r in hybrid.iterrows():
        label=labels.get(r.ticker,r.ticker)
        rows.append([label,f'{r.prior_delta_r2_60:+.5f}',f'{r.prior_delta_r2_36:+.5f}',f'{r.p1_span36_minus60_r2:+.5f}'])
    blocks['fi_hybrids']=table(['Instrument','Prior gain, 60','Prior gain, 36','P1: 36 minus 60'],rows)
    return blocks


def validate_fi(source, blocks):
    """Reject drift in every new empirical table."""
    for key,expected in blocks.items():
        match=re.findall(r'<!-- evidence: '+key+r' -->\n(.*?)\n<!-- /evidence -->',source,re.S)
        assert match==[expected],f'FI evidence drift: {key}'


def escape_text(text):
    """Escape prose while preserving inline mathematical expressions."""
    parts=re.split(r'(\$[^$]+\$)',text)
    for i in range(0,len(parts),2):
        s=parts[i]
        s=s.replace('â†’',r'$\to$').replace('Â²',r'$^2$').replace('Î”',r'$\Delta$')
        s=s.replace('â€”','---').replace('â€“','--').replace('âˆ’','-')
        s=s.replace('%',r'\%').replace('&',r'\&').replace('_',r'\_').replace('#',r'\#')
        s=re.sub(r'\*\*(.*?)\*\*',lambda m:r'\textbf{'+m[1]+'}',s)
        s=re.sub(r'(?<!\*)\*([^*]+)\*',lambda m:r'\emph{'+m[1]+'}',s)
        s=re.sub(r'`([^`]+)`',lambda m:r'\texttt{'+m[1]+'}',s)
        parts[i]=s
    return ''.join(parts)


def inline(text,refs):
    """Translate numbered literature links to CAS author-year citations."""
    text=re.sub(r'\[(\d+)\]\([^)]+\)',lambda m:r'\citep{'+refs[int(m[1])-1]['key']+'}',text)
    # Protect citation commands from prose underscore escaping.
    stash={}
    def link(m):
        """Protect a LaTeX hyperlink while escaping its visible title."""
        token=f'LINKPLACEHOLDER{len(stash)}'
        stash[token]=r'\href{'+m[2]+r'}{'+escape_text(m[1])+'}'
        return token
    text=re.sub(r'\[([^]]+)\]\(([^)]+)\)',link,text)
    result=escape_text(text)
    for key,value in stash.items():result=result.replace(key,value)
    return result


def body_to_tex(source,refs):
    """Translate the manuscript's restrained scientific Markdown to LaTeX."""
    source=re.sub(r'<!--.*?-->','',source,flags=re.S)
    source=source.split('## References')[0]
    source=source[source.index('## 1. '):]
    lines=source.splitlines();out=[];i=0;pending_caption=None
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('## '):
            title=re.sub(r'^\d+\.\s*','',line[3:])
            command='section*' if title=='Computational provenance and availability' else 'section'
            out.append('\\'+command+'{'+inline(title,refs)+'}')
        elif line.startswith('### '):
            title=re.sub(r'^\d+\.\d+\s*','',line[4:])
            out.append(r'\subsection{'+inline(title,refs)+'}')
        elif line=='$$':
            eq=[];i+=1
            while lines[i].strip()!='$$':eq.append(lines[i]);i+=1
            out.append('\\begin{equation}\n'+'\n'.join(eq)+'\n\\end{equation}')
        elif line.startswith('**Table '):
            pending_caption=re.sub(r'^\*\*Table \d+\.\s*(.*?)\*\*',r'\1',line)
        elif line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                row=[v.strip() for v in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+',v) for v in row):rows.append(row)
                i+=1
            i-=1
            n=len(rows[0]);tab=[r'\begin{table}[pos=htbp]',r'\centering',r'\small',r'\caption{'+inline(pending_caption or 'Empirical results.',refs)+'}',r'\begin{adjustbox}{max width=\linewidth}',r'\begin{tabular}{l'+'r'*(n-1)+'}',r'\toprule']
            for k,row in enumerate(rows):
                tab.append(' & '.join(inline(v,refs) for v in row)+r' \\')
                if k==0:tab.append(r'\midrule')
            tab += [r'\bottomrule',r'\end{tabular}',r'\end{adjustbox}',r'\end{table}']
            out.append('\n'.join(tab));pending_caption=None
        elif line.startswith('!['):
            match=re.match(r'!\[([^]]*)\]\(([^)]+)\)',line)
            name=Path(match[2]).name
            i+=1
            while i<len(lines) and not lines[i].strip():i+=1
            caption=re.sub(r'^\*\*Figure \d+\.\s*(.*?)\*\*',r'\1',lines[i].strip())
            out.append('\n'.join([r'\begin{figure}[pos=htbp]',r'\centering',r'\includegraphics[width=.94\linewidth]{figures/'+name+'}',r'\caption{'+inline(caption,refs)+'}',r'\end{figure}']))
        else:
            paragraph=[line]
            while i+1<len(lines) and lines[i+1].strip() and not lines[i+1].startswith(('#','$$','|','![')):
                i+=1;paragraph.append(lines[i].strip())
            out.append(inline(' '.join(paragraph),refs))
        i+=1
    return '\n\n'.join(out)


def bibliography(refs):
    """Write publisher-checked bibliography metadata without inventing page ranges."""
    authors=['Tibshirani, R.','Bastani, H.','Takada, M. and Fujisawa, H.','Craig, E. and others',
        'Chatterjee, S. and Hastie, T. and Tibshirani, R.','Meinshausen, N.',
        'Cosemans, M. and Frehen, R. and Schotman, P. C. and Bauer, R.',
        'Ledoit, O. and Wolf, M.','Politis, D. N. and Romano, J. P.']
    venues=[('Journal of the Royal Statistical Society, Series B','58','1','267--288'),
        ('Management Science','67','5','2964--2984'),('Advances in Neural Information Processing Systems','33','',''),
        ('Journal of the Royal Statistical Society, Series B','88','1','261--281'),
        ('Harvard Data Science Review','7','3',''),('Computational Statistics & Data Analysis','52','1','374--393'),
        ('Review of Financial Studies','29','4','1072--1112'),('Journal of Empirical Finance','10','5','603--621'),
        ('Journal of the American Statistical Association','89','428','1303--1313')]
    result=[]
    for r,a,(journal,volume,number,pages) in zip(refs,authors,venues):
        fields=dict(author=a,title='{'+r['title']+'}',year=r['year'],journal=journal.replace('&',r'\&'),volume=volume,number=number,pages=pages,doi=r.get('doi',''),url=r['url'])
        result.append('@article{'+r['key']+',\n'+',\n'.join(f'  {k} = {{{v}}}' for k,v in fields.items() if v)+'\n}')
    return '\n\n'.join(result)+'\n'


def build(out,publish):
    """Verify evidence and compile the manuscript with the existing CAS template."""
    if read_json(PAPER_ROOT/'paper_sources.json').get('edition') == 'fi_validation_20260924':
        from .fi_validation_paper import build as practical_build
        return practical_build(out,publish,here=PAPER_ROOT)
    assert out.is_absolute() and 'OneDrive' not in out.parts
    if out.exists():raise FileExistsError(out)
    source=(PAPER_ROOT/'manuscript.md').read_text(encoding='utf-8')
    metadata=read_json(PAPER_ROOT/'paper_sources.json');refs=metadata['references'];root=Path(metadata['study_root'])
    new,receipts=replacement.evidence_blocks(root)
    replacement.validate_source(source,new,refs)
    split_audit=replacement.verify_evidence(root)
    blocks=fi_blocks();validate_fi(source,blocks)
    etf_blocks,etf_audit=examples.evidence_blocks();examples.validate_source(source,etf_blocks)
    rejected=[]
    for key in ['split_simulation','split_components','split_rolling','fi_levels','fi_betas','core_etf_estimates']:
        b=(new|blocks|etf_blocks)[key]
        corrupted=b.replace('0.001','9.999',1) if key=='core_etf_estimates' else b.replace('|','| ALTERED ',1)
        assert corrupted!=b
        changed=source.replace(b,corrupted)
        try:
            replacement.validate_source(changed,new,refs);validate_fi(changed,blocks);examples.validate_source(changed,etf_blocks)
        except AssertionError:rejected.append(key)
        else:raise AssertionError(f'Corruption accepted: {key}')
    manifests={}
    for name in ['fi_prior_empirics_20260922_v3','fi_prior_funds_20260922_v2',
        'fi_prior_empirics_span36_20260922_v1','fi_prior_funds_span36_20260922_v1',
        'fi_prior_empirics_span36_endpoints_20260922_v1','fi_prior_span36_comparison_20260922_v1']:
        p=BASE/name;manifest=read_json(p/'evidence_manifest.json')
        for filename,expected in manifest.items():assert digest(p/filename)==expected,(name,filename)
        manifests[name]=len(manifest)
    for filename,expected in read_json(COMPARE/'summary.json')['source_hashes'].items():
        assert digest(Path(filename))==expected,filename
    algebra=legacy.algebra_checks()
    # Independent finite-geometric sums verify ESS and loss-rescaling identities.
    details={}
    for s in (36,60):
        r=1-2/(s+1);n=120;w=r**np.arange(n)
        from factorlasso.lasso_estimator import _compute_solver_weights
        solver_weights=_compute_solver_weights(n,1,s,np.ones((n,1)))[:,0]
        np.testing.assert_allclose(solver_weights**2,w[::-1],atol=1e-14)
        ess=w.sum()**2/(w@w);formula=s*(1-r**n)/(1+r**n)
        np.testing.assert_allclose(ess,formula,atol=1e-12)
        errors=np.cos(np.arange(n));lhs=(w*errors**2).sum()/n
        rhs=w.sum()/n*np.average(errors**2,weights=w)
        np.testing.assert_allclose(lhs,rhs,atol=1e-15)
        details[s]=dict(half_life=float(np.log(.5)/np.log(r)),finite_ess=ess,loss_multiplier=w.sum()/n)
    out.mkdir(parents=True)
    (out/'core_etf_verification.json').write_text(json.dumps(etf_audit,indent=2)+'\n',encoding='utf-8')
    pd.DataFrame(etf_audit['records']).to_csv(out/'core_etf_endpoint_and_scores.csv',index=False,float_format='%.17g')
    figure_hashes=replacement.figures(root,out)
    template=PAPER_ROOT.parent/'sign_pooling_2026/paper'
    for name in ['cas-sc.cls','cas-common.sty','cas-model2-names.bst']:shutil.copy2(template/name,out/name)
    abstract=source.split('**Abstract.** ',1)[1].split('## 1. ',1)[0]
    abstract=re.sub(r'<!--.*?-->','',abstract,flags=re.S).strip()
    preamble=r'''\documentclass[a4paper,review]{cas-sc}
\usepackage[authoryear,longnamesfirst]{natbib}
\usepackage{amssymb}
\usepackage{bm}
\usepackage{booktabs}
\usepackage{adjustbox}
\ExplSyntaxOn
\cs_set:Npn \__first_footerline: {\small\itshape Research~note,~23~September~2026}
\ExplSyntaxOff
\renewcommand{\printorcid}{}
\hypersetup{pdftitle={Regularisation Targets for Economic Factor Attribution},pdfauthor={Artur Sepp}}
\begin{document}
\let\WriteBookmarks\relax
\def\floatpagepagefraction{1}
\def\textpagefraction{.001}
\shorttitle{Regularisation targets for factor attribution}
\shortauthors{Sepp}
\title[mode=title]{Regularisation Targets for Economic Factor Attribution}
\tnotemark[1]
\author{Artur Sepp}
\tnotetext[1]{Research note. Revised 23 September 2026. Internal empirical draft.}
\begin{abstract}
'''
    front=preamble+inline(' '.join(abstract.split()),refs)+r'''
\end{abstract}
\begin{highlights}
\item Every numerical design uses MATF custom with separate IG, HY and EM credit.
\item Automatic targets improve average fixed-penalty FI explanation in the main sample.
\item A shorter EWMA span increases relative prior gains but reduces absolute fit.
\item Beta deviations, hybrid failures and weight scaling qualify the comparison.
\end{highlights}
\begin{keywords}
factor models \sep regularisation targets \sep fixed income \sep group lasso \sep out-of-sample evaluation
\end{keywords}
\maketitle
'''
    tex='% Generated from manuscript.md by build_latex.py; edit the Markdown source.\n'+front+body_to_tex(source,refs)+r'''
\bibliographystyle{cas-model2-names}
\bibliography{refs}
\end{document}
'''
    (out/'article.tex').write_text(tex,encoding='utf-8')
    (out/'refs.bib').write_text(bibliography(refs),encoding='utf-8')
    shutil.copy2(PAPER_ROOT/'manuscript.md',out/'manuscript.md')
    shutil.copy2(HERE/'build_latex.py',out/'build_latex.py')
    for name in ['split_credit_paper.py','core_etf_examples.py','paper_sources.json']:shutil.copy2((HERE if name.endswith('.py') else PAPER_ROOT)/name,out/name)
    def compile_command(command):
        """Run one compiler step and retain its complete diagnostics."""
        result=subprocess.run(command,cwd=out,text=True,capture_output=True)
        (out/(command[0]+'_last_stdout.txt')).write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if result.returncode:raise RuntimeError(result.stdout[-3500:]+result.stderr)
    latex=['pdflatex','-interaction=nonstopmode','-halt-on-error','article.tex']
    compile_command(latex);compile_command(['bibtex','article'])
    # CAS writes its last-page footer to the aux file without a rerun warning.
    # Require aux convergence rather than assuming two post-BibTeX passes suffice.
    for convergence_pass in range(1,6):
        before=digest(out/'article.aux');compile_command(latex)
        if digest(out/'article.aux')==before:break
    else:raise AssertionError('LaTeX auxiliary state did not converge')
    log=(out/'article.log').read_text(encoding='utf-8',errors='replace')
    assert not re.search(r'undefined references|Citation .* undefined|Reference .* undefined',log)
    overflows=re.findall(r'Overfull \\hbox \(([^)]*)\) detected at line (\d+)',log)
    # CAS deliberately places the keyword box in a zero-width hbox. The reference
    # sign paper logs the identical 117.0831pt diagnostic; inspect the rendered
    # title page and reject every other horizontal overflow.
    title_line=str(tex.splitlines().index(r'\maketitle')+1)
    assert overflows==[('117.0831pt too wide',title_line)],overflows
    assert log.count('Overfull \\hbox')==1
    assert 'Overfull \\vbox' not in log
    if publish:
        dest=PAPER_ROOT/'paper';dest.mkdir(exist_ok=True)
        for name in ['article.tex','refs.bib']:shutil.copy2(out/name,dest/name)
    receipt=dict(status='compiled_pending_visual_review',split_credit_numerical_receipts=len(receipts),
        fi_verified_blocks=list(blocks),core_etf_verified_blocks=list(etf_blocks),core_etf_audit={k:v for k,v in etf_audit.items() if k!='records'},fi_manifests=manifests,split_credit_audit=split_audit,cas_keyword_box_diagnostic=overflows,
        latex_aux_convergence_passes=convergence_pass,numerical_corruption_rejected=rejected,algebra=algebra,ewma_algebra=details,
        source_sha256=digest(PAPER_ROOT/'manuscript.md'),figure_hashes=figure_hashes,
        template_hashes={name:digest(template/name) for name in ['cas-sc.cls','cas-common.sty','cas-model2-names.bst']},
        pdf_sha256=digest(out/'article.pdf'))
    (out/'verification.json').write_text(json.dumps(receipt,indent=2,default=str)+'\n',encoding='utf-8')
    (out/'numerical_receipts.json').write_text(json.dumps(receipts,indent=2,default=str)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2,default=str))


def main():
    """Parse the explicit build destination and optional source-mirror request."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--publish-source',action='store_true')
    args=parser.parse_args();build(args.output_root,args.publish_source)


if __name__=='__main__':main()
