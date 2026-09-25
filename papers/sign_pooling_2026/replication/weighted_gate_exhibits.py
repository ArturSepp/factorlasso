"""Reproduce the revised-gate tables and ROC figure from saved MC evidence."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def build(root, out):
    """Read paired results and write two LaTeX tables plus a vector figure."""
    (out/'figures').mkdir(parents=True, exist_ok=True)
    data=pd.read_csv(root/'summary.csv')
    matched=pd.read_csv(root/'matched_false_sign.csv')
    case_names={'independent':'Independent residuals','overlapping':'Correlated residuals',
        'duplicate':'Duplicate responses','unequal_histories':'Unequal histories',
        'heteroskedastic':'Heteroskedastic residuals','regime_change':'Changing sign'}
    rows=[]
    for case in case_names:
        for span in (0,60):
            d=data.query('case==@case and span==@span and threshold==1').set_index('gate')
            rows.append(f"{case_names[case]} & {'Equal' if span==0 else '60'} & " + ' & '.join(
                f'{100*d.loc[g,v]:.1f}' for g in ('independent','date') for v in ('false_sign','active_recovery'))+r' \\')
    table=r'''\begin{table}[htbp]
    \centering\small
    \caption{Revised gate validation at threshold one. False sign means any admission
    on the null predictor; recovery means admission with the correct active sign.
    Percentages over 500 paired replications per row. Equal denotes equal weighting;
    60 is the EWMA span.}\label{tab:revised-gate}
    \begin{tabular}{llrrrr}\toprule
     & & \multicolumn{2}{c}{Independent gate} & \multicolumn{2}{c}{Date-score gate}\\
    Design & Span & False & Recovery & False & Recovery\\\midrule
    '''+ '\n'.join(rows)+r'''
    \bottomrule\end{tabular}
    \end{table}
    '''
    matchrows=[]
    for case in case_names:
        d=matched.query('case==@case and span==60').set_index('gate')
        matchrows.append(case_names[case]+' & '+' & '.join(f'{100*d.loc[g,v]:.1f}'
            for g in ('independent','date') for v in ('evaluation_false_sign','evaluation_active_recovery'))+r' \\')
    table2=r'''\begin{table}[htbp]\centering\small
    \caption{Separate threshold calibration and evaluation, EWMA span 60. The first
    250 draws calibrate each threshold to 20\% null admission; the other 250 assess
    false-sign and active-recovery percentages. This compares operating points,
    not a claim of exact error control.}\label{tab:revised-matched}
    \begin{tabular}{lrrrr}\toprule
     & \multicolumn{2}{c}{Independent gate} & \multicolumn{2}{c}{Date-score gate}\\
    Design & False & Recovery & False & Recovery\\\midrule
    '''+ '\n'.join(matchrows)+r'''
    \bottomrule\end{tabular}\end{table}
    '''
    fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained',sharex=True,sharey=True)
    for ax,case in zip(axes.flat,case_names):
        for gate,color in [('independent','#aa634d'),('date','#245875')]:
            d=data.query('case==@case and span==60 and gate==@gate').sort_values('threshold')
            ax.plot(d.false_sign*100,d.active_recovery*100,'o-',label=gate,color=color,ms=3)
        ax.set_title(case_names[case],fontsize=10);ax.grid(alpha=.18)
        ax.set_xlim(0,100);ax.set_ylim(0,102)
    for ax in axes[-1]:ax.set_xlabel('Null admission (%)')
    for ax in axes[:,0]:ax.set_ylabel('Correct active admission (%)')
    axes[0,0].legend(fontsize=8)
    fig.savefig(out/'figures/revised_gate_roc.pdf',bbox_inches='tight');plt.close(fig)
    (out/'table_revised_gate.tex').write_text(table, encoding='utf-8')
    (out/'table_revised_matched.tex').write_text(table2, encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args()
    build(args.root, args.out)
