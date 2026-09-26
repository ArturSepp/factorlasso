"""Build the evidence-led paper from frozen results; never rerun an experiment."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import sys

import cvxpy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser, math_to_image
import numpy as np
import pandas as pd

from papers.prior_targets_2026.replication.run import HERE, PAPER_ROOT, digest, read_json, save_json, validate_root


class Evidence:
    """Resolve every displayed data value to a unique frozen table row."""

    def __init__(self, root):
        """Retain the immutable source root and numerical receipts."""
        self.root, self.frames, self.receipts = root, {}, []

    def value(self, file, where, column):
        """Return a scalar and record its exact provenance."""
        if file not in self.frames:
            self.frames[file] = pd.read_csv(self.root / file)
        selected = self.frames[file]
        for key, value in where.items():
            selected = selected[selected[key] == value]
        assert len(selected) == 1, (file, where)
        value = float(selected.iloc[0][column])
        self.receipts.append(dict(file=file, sha256=digest(self.root/file),
                                  where=where, column=column, value=value))
        return value


def markdown_table(headers, rows):
    """Return a stable portable Markdown table."""
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(row) + ' |' for row in rows])


def evidence_blocks(e, refs):
    """Reconstruct paper exhibits and numerical conclusions without estimation."""
    blocks = {}
    def e2(method, col):
        """Read the declared target-quality stress cell."""
        return e.value('E2_robustness_v1/paired_summary.csv',
                       dict(cell='quality_r0.95_b0.3_n60', method=method), col)
    def l1(method, col, profile='baseline'):
        """Read the n112 per-asset calibration comparison."""
        return e.value('E3_L1_confirmation_v1/paired_summary.csv',
                       dict(profile=profile, n=112, method=method), col)
    def group(setting, target, col, profile='baseline'):
        """Read a within-specification group comparison."""
        return e.value('E3_groups_confirmation_v2/paired_summary.csv',
                       dict(profile=profile, setting=setting, target=target), col)
    def restricted(method, col, profile='baseline'):
        """Read the separately seeded restricted-target sensitivity."""
        return e.value('E3_restricted_v2/paired_summary.csv',
                       dict(profile=profile, method=method, comparator='M1_zero'), col)
    def rolling(method, col, lane='production_snapshot', span=0):
        """Read a paired Credit-cohort rolling statistic."""
        return e.value('E4_independent_review_v1/paired_effects.csv',
                       dict(lane=lane, span=span, cohort='credit', method=method, block=6), col)
    rows = []
    for method, label in [('M1_zero','Zero'), ('M2_auto','Automatic winner'),
                          ('M3_noisy_std025','Noisy truth, SD 0.25'),
                          ('Q_scale1_sd0.5','Noisy truth, SD 0.50'),
                          ('Q_wrong_factor','Wrong-factor target'),
                          ('M4_free_winner','Free winner'), ('M5_post_lasso','Post-LASSO')]:
        interval = ('Reference' if method == 'M1_zero' else
                    f"{e2(method,'secondary_mse_delta'):+.4f} "
                    f"[{e2(method,'secondary_mse_low'):+.4f}, {e2(method,'secondary_mse_high'):+.4f}]")
        rows.append([label, f"{e2(method,'secondary_mse'):.4f}", interval,
                     f"{e2(method,'prediction_mse'):.4f}"])
    blocks['controlled'] = markdown_table(
        ['Target or control','Secondary MSE','Paired difference [95% interval]','Test MSE'], rows)
    rows = []
    for method, label in [('M1_zero','Zero'), ('M2_auto','Automatic winner'),
                          ('M3_economic','Economic'), ('M3_economic_noisy','Economic + noise'),
                          ('M3_small_model','Rates + Credit'),
                          ('M4_free_winner','Free winner'), ('M5_post_lasso','Post-LASSO')]:
        rows.append([label]+[f"{l1(method,'credit_mse',p):.6f}"
                             for p in ['baseline','credit_removed','rates_dominant']])
    blocks['calibrated'] = markdown_table(
        ['Target or control','Original profile','Credit removed','Rates multiplied by 4'], rows)
    rows = []
    for method, label in [('M0_ols','OLS'), ('M2_auto','Automatic winner'),
                          ('M3_fixed','Fixed economic'), ('M3_small2','Rates + Credit'),
                          ('M3_small3','Equity + Rates + Credit'),
                          ('M4_free_winner','Free winner'), ('M5_post_lasso','Post-LASSO')]:
        rows.append([label]+[f"{rolling(method,'relative_change_pct',lane,span):+.2f}%"
                             for lane in ['production_snapshot','public_etf_proxies']
                             for span in [0,36]])
    blocks['rolling'] = markdown_table(
        ['Target or control','NAV uniform','NAV EWMA','ETF uniform','ETF EWMA'], rows)
    improvement = 100*(1-l1('M3_economic','credit_mse')/l1('M1_zero','credit_mse'))
    harm = l1('M3_economic','credit_mse','credit_removed')/l1('M1_zero','credit_mse','credit_removed')
    blocks['abstract_result'] = (
        f'In a calibrated panel, an economic target reduces Credit-loading mean squared error '
        f'by {improvement:.1f}% when its interpretation aligns with the generating model; '
        f'removing true Credit exposure raises its error to {harm:.2f} times the zero-target error. '
        'An automatic univariate target can improve aggregate covariance accuracy while worsening '
        'the Credit loading.')
    blocks['controlled_result'] = (
        f"The wrong-factor target increases secondary-loading error by "
        f"{100*(e2('Q_wrong_factor','secondary_mse')/e2('M1_zero','secondary_mse')-1):.1f}% "
        f"while reducing test prediction error by "
        f"{100*(1-e2('Q_wrong_factor','prediction_mse')/e2('M1_zero','prediction_mse')):.2f}%. "
        'Validation on prediction therefore does not reliably reject this attribution failure.')
    blocks['calibrated_result'] = (
        f'The economic target reduces original-profile Credit MSE by {improvement:.1f}%. '
        f"Its paired difference is {l1('M3_economic','credit_mse_delta'):+.6f}, "
        f"with interval [{l1('M3_economic','credit_mse_low'):+.6f}, "
        f"{l1('M3_economic','credit_mse_high'):+.6f}]. "
        f'When Credit is removed, its error is {harm:.2f} times zero-target error, '
        f"but cohort prediction error rises only "
        f"{100*(l1('M3_economic','credit_prediction_nmse','credit_removed')/l1('M1_zero','credit_prediction_nmse','credit_removed')-1):.2f}%.")
    blocks['covariance_result'] = (
        f"Automatic targeting reduces relative full-covariance error from "
        f"{l1('M1_zero','full_covar_error'):.6f} to {l1('M2_auto','full_covar_error'):.6f}, "
        f"while raising Credit MSE from {l1('M1_zero','credit_mse'):.6f} "
        f"to {l1('M2_auto','credit_mse'):.6f}. "
        'A better covariance aggregate cannot certify an economically relevant coefficient.')
    blocks['restricted_result'] = (
        f"Adding Equity to the restricted target reduces original-profile Credit MSE by "
        f"{100*(1-restricted('small3','credit_mse')/restricted('M1_zero','credit_mse')):.2f}%. "
        f"The paired difference is {restricted('small3','credit_mse_delta'):+.6f} "
        f"[{restricted('small3','credit_mse_low'):+.6f}, {restricted('small3','credit_mse_high'):+.6f}]. "
        f"Removed-Credit error remains {restricted('small3','credit_mse','credit_removed')/restricted('M1_zero','credit_mse','credit_removed'):.2f} "
        'times zero-target error. Broader scenario MSE increases in all three profiles.')
    blocks['group_result'] = (
        'All five original-profile economic-minus-zero intervals are below zero; all automatic-minus-zero '
        'intervals are above zero. With Credit removed, economic targeting gives '
        f"{group('HCGL','economic','credit_mse','credit_removed')/group('HCGL','zero','credit_mse','credit_removed'):.2f} "
        'times the HCGL zero-target error and '
        f"{group('FCGL_sign_adaptive','economic','credit_mse','credit_removed')/group('FCGL_sign_adaptive','zero','credit_mse','credit_removed'):.2f} "
        'times the FCGL error. Signed HCGL specifications largely attenuate this harm; '
        'their economic-minus-zero intervals include zero. Automatic targeting improves removed-Credit '
        'attribution in the HCGL variants, so its adverse original-profile result is not universal.')
    blocks['rolling_result'] = (
        'The automatic target has higher mean reconstruction error in all four settings. '
        f"For NAV factors with uniform weights, its paired NMSE difference is "
        f"{rolling('M2_auto','paired_delta'):+.6f} "
        f"[{rolling('M2_auto','low'):+.6f}, {rolling('M2_auto','high'):+.6f}]. "
        'For ETF proxies with EWMA weights, the difference is '
        f"{rolling('M2_auto','paired_delta','public_etf_proxies',36):+.6f} "
        f"[{rolling('M2_auto','low','public_etf_proxies',36):+.6f}, "
        f"{rolling('M2_auto','high','public_etf_proxies',36):+.6f}]. "
        'The latter interval remains positive at each tested block length. These are exploratory '
        'pointwise intervals over a short evaluation period.')
    blocks['references'] = '\n\n'.join(
        f"{j}. {r['authors']} ({r['year']}). {r['title']}. {r['venue']}. "
        f"[Publisher]({r['url']})." for j,r in enumerate(refs,1))
    return blocks


def validate_source(source, blocks, refs):
    """Reject edited numbers, missing exhibits and unidentified literature links."""
    assert '{{' not in source and 'TODO' not in source
    for key, expected in blocks.items():
        pattern = r'<!-- evidence: '+re.escape(key)+r' -->\n(.*?)\n<!-- /evidence -->'
        found = re.findall(pattern, source, flags=re.S)
        assert found == [expected], f'Evidence block drift: {key}'
    urls = {r['url'] for r in refs}
    for number, url in re.findall(r'\[(\d+)\]\(([^)]+)\)', source):
        assert refs[int(number)-1]['url'] == url, f'Broken reference: {number}'
    assert set(re.findall(r'\[Publisher\]\(([^)]+)\)', source)) == urls
    for url in re.findall(r'!\[[^]]*\]\(([^)]+)\)', source):
        assert Path(url).name in {'groups.png','rolling.png'}


def algebra_checks():
    """Check three elementary identities by independent deterministic computations."""
    t = np.arange(1,42,dtype=float)
    x = np.column_stack([np.sin(t/3),np.cos(t/7),t/40])
    x = (x-x.mean(0))/x.std(0)
    y = x@np.array([.7,-.2,.35]) + .11*np.cos(t*1.7)
    y = y-y.mean()
    target = np.array([.5,0,.1])
    b,d = cp.Variable(3),cp.Variable(3)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(y-x@b)/len(t)+.07*cp.norm1(b-target)))
    centred = cp.Problem(cp.Minimize(cp.sum_squares(y-x@target-x@d)/len(t)+.07*cp.norm1(d)))
    for p in [problem,centred]:
        p.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_feas=1e-10,tol_gap_rel=1e-10)
        assert p.status == 'optimal'
    recenter = float(np.max(abs(b.value-target-d.value)))
    np.testing.assert_allclose(b.value,target+d.value,atol=2e-7,rtol=0)
    ols = np.linalg.solve(x.T@x,x.T@y)
    maximum = 0.
    for lam in [0.,.01,1.,10.]:
        p = cp.Problem(cp.Minimize(cp.sum_squares(y-x@b)/len(t)+lam*cp.norm1(b-ols)))
        p.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_feas=1e-10,tol_gap_rel=1e-10)
        maximum = max(maximum,float(np.max(abs(b.value-ols))))
    assert maximum < 2e-7
    errors = []
    for rho in [0.,.5,.95,.99,1.]:
        covariance = np.array([[1.,rho],[rho,1.]])
        deviation = np.array([.3,-.3])
        direct = float(deviation@covariance@deviation)
        formula = 2*.3**2*(1-rho)
        np.testing.assert_allclose(direct,formula,atol=1e-14)
        errors.append(abs(direct-formula))
    try:
        np.testing.assert_allclose(b.value,ols+.1,atol=2e-7,rtol=0)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupted identity was accepted')
    return dict(recentering_max_error=recenter, ols_identity_max_error=maximum,
                prediction_identity_max_error=max(errors), corruption_rejected=True)


def figures(root, out):
    """Redraw published evidence summaries; perform no new fitting or inference."""
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'font.family':'DejaVu Sans','savefig.dpi':220})
    dest=out/'figures'
    dest.mkdir()
    frame=pd.read_csv(root/'E3_groups_confirmation_v2/paired_summary.csv')
    settings=['HCGL','HCGL_sign','HCGL_sign_adaptive','FCGL_sign_adaptive','Known_HCGL_sign_adaptive']
    labels=['HCGL','HCGL + signs','HCGL + signs/adaptive','FCGL + signs/adaptive','Supplied groups + signs/adaptive']
    fig,axes=plt.subplots(1,2,figsize=(9.1,3.5),sharey=True,layout='constrained')
    for ax,profile,title in zip(axes,['baseline','credit_removed'],['Original profile','Credit removed']):
        for target,color,shift,label in [('auto','#ad5128',-.11,'Automatic'),('economic','#256b82',.11,'Economic')]:
            part=frame[(frame.profile==profile)&(frame.target==target)].set_index('setting').loc[settings]
            ax.errorbar(part.credit_mse_delta,np.arange(5)+shift,
                        xerr=np.vstack([part.credit_mse_delta-part.credit_mse_low,
                                        part.credit_mse_high-part.credit_mse_delta]),
                        fmt='o',ms=4,capsize=2,label=label,color=color)
        ax.axvline(0,color='#777777',lw=.8)
        ax.set(title=title,xlabel='Credit MSE difference vs zero')
        ax.ticklabel_format(axis='x',style='plain')
        ax.grid(axis='x',alpha=.15)
    axes[0].set_yticks(np.arange(5),labels)
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False,loc='lower right',fontsize=9)
    fig.savefig(dest/'groups.png',bbox_inches='tight'); plt.close(fig)
    frame=pd.read_csv(root/'E4_rolling_v1/rolling_results.csv')
    fig,axes=plt.subplots(1,2,figsize=(9.1,2.85),layout='constrained')
    for ax,lane,title in zip(axes,['production_snapshot','public_etf_proxies'],
                            ['NAV Credit factor','HYG Credit proxy']):
        for method,color,label in [('M1_zero','#343b44','Zero'),('M2_auto','#ad5128','Automatic'),
                                   ('M3_fixed','#256b82','Fixed economic')]:
            part=frame[(frame.lane==lane)&(frame.span==0)&(frame.asset=='LQD')&(frame.method==method)].sort_values('date')
            assert len(part)==29
            ax.plot(pd.to_datetime(part.date),part.beta_Credit,label=label,color=color,lw=1.5)
        ax.set(title=title,ylabel='LQD Credit loading (native units)')
        ax.grid(alpha=.17)
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=9))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %Y'))
    axes[1].legend(frameon=False,fontsize=9,loc='best')
    fig.savefig(dest/'rolling.png',bbox_inches='tight'); plt.close(fig)
    return {p.name:digest(p) for p in dest.glob('*.png')}


def render_pdf(source, out, bundled):
    """Render portable Markdown, equations and exhibits using reportlab."""
    sys.path.append(str(bundled))
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
    from pypdf import PdfReader
    font_root=Path(matplotlib.get_data_path())/'fonts/ttf'
    for name,file in [('Body','DejaVuSerif.ttf'),('Body-Bold','DejaVuSerif-Bold.ttf'),
                      ('Body-Italic','DejaVuSerif-Italic.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(font_root/file)))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='Body-Bold',italic='Body-Italic')
    body=ParagraphStyle('body',fontName='Body',fontSize=10,leading=14,spaceAfter=8,textColor=colors.HexColor('#222a32'))
    styles={'body':body,
            'title':ParagraphStyle('title',parent=body,fontName='Body-Bold',fontSize=22,leading=27,spaceAfter=13),
            'h2':ParagraphStyle('h2',parent=body,fontName='Body-Bold',fontSize=13,leading=17,spaceBefore=9,spaceAfter=10,keepWithNext=True),
            'h3':ParagraphStyle('h3',parent=body,fontName='Body-Bold',fontSize=10.5,leading=14,spaceBefore=5,spaceAfter=6,keepWithNext=True),
            'caption':ParagraphStyle('caption',parent=body,fontSize=8.6,leading=11.5,spaceAfter=10),
            'table':ParagraphStyle('table',parent=body,fontSize=8,leading=11,spaceAfter=0),
            'reference':ParagraphStyle('reference',parent=body,fontSize=8.5,leading=11.6,spaceAfter=7)}
    math_dir=out/'equations'; math_dir.mkdir()
    parser=MathTextParser('path')
    def math_image(expr, size=10):
        """Render a formula with measured point dimensions."""
        expr=expr.replace('\\lVert','\\Vert').replace('\\rVert','\\Vert')
        prop=FontProperties(family='DejaVu Serif',size=size)
        result=parser.parse('$'+expr+'$',dpi=72,prop=prop)
        filename=hashlib.sha256((expr+str(size)).encode()).hexdigest()[:16]+'.png'
        path=math_dir/filename
        if not path.exists():
            math_to_image('$'+expr+'$',str(path),prop=prop,dpi=240,format='png',color='#222a32')
        return path,float(result.width),float(result.height),float(result.depth)
    def inline(raw):
        """Escape prose before resolving checked Markdown links and inline math."""
        pieces=re.split(r'(\$[^$]+\$)',raw)
        converted=[]
        for part in pieces:
            if part.startswith('$') and part.endswith('$'):
                path,width,height,depth=math_image(part[1:-1])
                converted.append(f'<img src="{path.as_posix()}" width="{width}" height="{height}" valign="{-depth}"/>')
            else:
                text=html.escape(part)
                def link(match):
                    """Keep bracketed numerical citations in the printable document."""
                    label, url = match.groups()
                    label = '['+label+']' if label.isdigit() else label
                    return f'<a href="{url}" color="#256b82">{label}</a>'
                text=re.sub(r'\[([^]]+)\]\(([^)]+)\)',link,text)
                text=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',text)
                converted.append(text)
        return ''.join(converted)
    clean=re.sub(r'<!-- evidence: [^>]+ -->\n|\n<!-- /evidence -->','',source)
    lines=clean.splitlines(); story=[]; i=0; table_no=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:
            i+=1; continue
        if line=='<!-- pagebreak -->':
            story.append(PageBreak()); i+=1; continue
        if line=='$$':
            end=lines.index('$$',i+1)
            expression=' '.join(lines[i+1:end])
            path,width,height,_=math_image(expression,11)
            factor=min(1.,475/width)
            img=Image(str(path),width=width*factor,height=height*factor)
            img.hAlign='CENTER'; story.extend([Spacer(1,5),img,Spacer(1,12)])
            i=end+1; continue
        if line.startswith('!['):
            url=re.search(r'\]\(([^)]+)\)',line).group(1)
            from PIL import Image as PILImage
            path=out/'figures'/Path(url).name
            with PILImage.open(path) as picture:
                width,height=picture.size
            story.extend([Image(str(path),width=475,height=475*height/width),Spacer(1,7)])
            i+=1; continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                row=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+',c) for c in row): rows.append(row)
                i+=1
            widths={2:[125,350],4:[139,72,72,192],5:[155,80,80,80,80]}.get(len(rows[0]))
            if len(rows[0])==4:
                widths=[139,63,213,60] if 'Secondary MSE' in rows[0] else [154,98,98,125]
            table=Table([[Paragraph(inline(c),styles['table']) for c in r] for r in rows],
                        colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef1')),
                ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#58717d')),
                ('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#87939a')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),
                ('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),4),
                ('BOTTOMPADDING',(0,0),(-1,-1),4)]))
            story.extend([table,Spacer(1,10)]); table_no+=1; continue
        style='body'
        if line.startswith('# '): style='title'; line=line[2:]
        elif line.startswith('## '): style='h2'; line=line[3:]
        elif line.startswith('### '): style='h3'; line=line[4:]
        elif line.startswith('**Table ') or line.startswith('**Figure '): style='caption'
        elif re.match(r'^\d+\. ',line): style='reference'
        paragraph=[line]; i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('<!--','$$','#','|','![')):
            paragraph.append(lines[i].strip()); i+=1
        story.append(Paragraph(inline(' '.join(paragraph)),styles[style]))
    pdf=out/'regularisation_targets_draft.pdf'
    doc=SimpleDocTemplate(str(pdf),pagesize=A4,leftMargin=60,rightMargin=60,topMargin=49,bottomMargin=47,
                          title='Regularisation Targets for Economic Factor Attribution',author='Artur Sepp')
    def page(canvas, document):
        """Apply consistent paper headers and page numbers."""
        canvas.saveState(); canvas.setFont('Body',7.4); canvas.setFillColor(colors.HexColor('#61717a'))
        canvas.drawString(60,821,'REGULARISATION TARGETS FOR ECONOMIC FACTOR ATTRIBUTION')
        canvas.drawRightString(535,25,f'Internal draft - 22 September 2026   |   {document.page}')
        canvas.restoreState()
    doc.build(story,onFirstPage=page,onLaterPages=page)
    reader=PdfReader(str(pdf))
    pages=[p.extract_text() for p in reader.pages]
    text='\n\n'.join(pages)
    assert len(pages)>=7 and len(pages)<=14, len(pages)
    assert 'TODO' not in text and '{{' not in text and '\ufffd' not in text
    for needle in ['57.3%', '4.89', '0.197544', '0.178937', '29 months', 'The Stationary Bootstrap']:
        assert needle in text, needle
    links=[a.get_object().get('/A',{}).get('/URI') for p in reader.pages for a in p.get('/Annots',[])]
    assert len([x for x in links if x])>=15
    (out/'extracted_text.txt').write_text(text,encoding='utf-8')
    return dict(pages=len(pages),pdf_sha256=digest(pdf),hyperlinks=len([x for x in links if x]),
                text_checks=True,table_count=table_no)


def main():
    """Verify evidence, produce a fresh PDF build and save complete receipts."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--bundled-site-packages',type=Path,required=True)
    parser.add_argument('--populate-source',action='store_true')
    args=parser.parse_args()
    out=validate_root(args.output_root)
    if out.exists(): raise FileExistsError(out)
    metadata=read_json(PAPER_ROOT/'paper_sources.json')
    root=Path(metadata['study_root'])
    from papers.prior_targets_2026.replication.run import check_snapshot
    from papers.prior_targets_2026.replication.decision import RUNS,validate_manifest,verify
    from papers.prior_targets_2026.replication.multiasset import check_manifest
    check_snapshot(root)
    sys.path.insert(0,str(root/'source_snapshot/src'))
    for run in RUNS: validate_manifest(root,run)
    check_manifest(root/'E5_decision_v1')
    previous=verify(root,root/'E5_decision_v1')
    evidence=Evidence(root)
    blocks=evidence_blocks(evidence,metadata['references'])
    source=(PAPER_ROOT/'manuscript.md').read_text(encoding='utf-8')
    if args.populate_source:
        assert all('{{'+key+'}}' in source for key in blocks)
        for key,content in blocks.items():
            source=source.replace('{{'+key+'}}','<!-- evidence: '+key+' -->\n'+content+'\n<!-- /evidence -->')
        # Initial template expansion only; never rewrite an already populated manuscript.
        (PAPER_ROOT/'manuscript.md').write_text(source,encoding='utf-8')
    validate_source(source,blocks,metadata['references'])
    rejected=[]
    for label,broken in [('altered_table',source.replace(blocks['controlled'],blocks['controlled'].replace('0.1612','9.1612'))),
                          ('broken_reference',source.replace('[1](https://academic.oup.com/jrsssb/article/58/1/267/7027929)',
                                                              '[1](https://example.invalid/missing)'))]:
        assert broken!=source
        try: validate_source(broken,blocks,metadata['references'])
        except AssertionError: rejected.append(label)
        else: raise AssertionError('Corruption accepted: '+label)
    algebra=algebra_checks()
    out.mkdir(parents=True)
    for name in ['build_paper.py','manuscript.md','paper_sources.json']:
        shutil.copy2((HERE if name.endswith('.py') else PAPER_ROOT)/name,out/name)
    plot_hashes=figures(root,out)
    rendered=render_pdf(source,out,args.bundled_site_packages)
    receipt=dict(status='rendered_pending_visual_review',evidence_ledger=previous,
                 numerical_receipts=len(evidence.receipts),source_blocks=len(blocks),
                 negative_controls=rejected,algebra=algebra,figures=plot_hashes,rendered=rendered)
    save_json(out/'numerical_receipts.json',evidence.receipts)
    save_json(out/'verification.json',receipt)
    save_json(out/'manifest.json',{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
