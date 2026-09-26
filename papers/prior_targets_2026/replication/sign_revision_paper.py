"""Evidence blocks and plots for the sign-revision appendix and the matched penalty controls of the FI note."""
from pathlib import Path
import numpy as np
import pandas as pd

POLICIES={'Z':'Zero prior','A':'Max-R2 prior','E1':'Expert prior','E2':'Conditional prior'}
VARIANTS={'old':'Old sign inputs','mask':'Valid observations only',
          'ewma':'Valid observations + EWMA','date':'EWMA + date-score gate'}


def evidence(root,table):
    """Bind the added tables to the saved ablations and matched prediction controls."""
    root=Path(root)/'sign_revision'
    endpoint=pd.read_csv(root/'endpoint_summary.csv')
    scores=pd.read_csv(root/'oos_summary.csv')
    rows=[]
    for lane in ('indices','funds'):
        for variant in ('mask','ewma','date'):
            e=endpoint.query('lane==@lane and variant==@variant and arm=="E2"').iloc[0]
            o=scores.query('lane==@lane and variant==@variant and arm=="E2" and span==60').iloc[0]
            rows.append([lane.title(),VARIANTS[variant],int(e.detected_changes),int(e.effective_changes),
                int(e.block_changes),f'{e.max_abs_cma_change_bp:.1f}',f'{o.monthly_rmse_pct:.3f}'])
    blocks={'sign_ablation':table(['Panel','Revision','Raw signs','Final signs','Blocks','Max CMA delta, bp','RMSE, %'],rows)}
    rows=[]
    for lane in ('indices','funds'):
        d=scores.query('lane==@lane and span==60 and variant=="date"').set_index('arm')
        # The Zero prior has no target; its zero-penalty control uses the detected signs.
        for arm in ('Z','A','E1','E2'):
            rows.append([lane.title(),POLICIES[arm],f'{d.loc[arm,"monthly_rmse_pct"]:.3f}',
                f'{d.loc[arm+"_target_only","monthly_rmse_pct"]:.3f}' if arm!='Z' else '--',
                f'{d.loc[arm+"_lambda_zero","monthly_rmse_pct"]:.3f}' if arm in ('Z','E2') else '--'])
    blocks['penalty_controls']=table(['Panel','Prior policy','FCGL, %','Target only, %','Zero penalty, %'],rows)
    return blocks,dict(sign_ablation=endpoint.to_dict('records'),penalty_controls=scores.to_dict('records'))


def figures(root,out):
    """Plot paired complete-policy RMSE using a common absolute scale."""
    import matplotlib.pyplot as plt
    scores=pd.read_csv(Path(root)/'sign_revision/oos_summary.csv').query('span==60')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained',sharey=True)
    colors=['#8894a2','#4a88a3','#1e5a73','#ac5739']
    for ax,lane in zip(axes,('indices','funds')):
        for variant,color in zip(VARIANTS,colors):
            d=scores.query('lane==@lane and variant==@variant').set_index('arm')
            ax.plot(np.arange(4),[d.loc[a,'monthly_rmse_pct'] for a in POLICIES],
                marker='o',color=color,label=VARIANTS[variant],lw=1.5)
        ax.set_xticks(np.arange(4),list(POLICIES.values()),rotation=15,ha='right')
        ax.set_title(lane.title());ax.grid(alpha=.18)
    axes[0].set_ylabel('Held-out monthly RMSE (%)')
    axes[1].legend(fontsize=8,loc='upper right')
    for ext in ('pdf','png'):fig.savefig(Path(out)/('fi_sign_revision.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
