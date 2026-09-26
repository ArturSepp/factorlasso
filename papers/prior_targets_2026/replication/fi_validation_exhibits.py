"""Labelled FI exhibits from frozen endpoint estimates and owner risk APIs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

INDEX_NAMES = dict(zip([
    'LUATTRUU','LGTRTRUH','LEGATRUH','LGCPTRUH','H09627US','H09629US','H09631US',
    'H23059US','LF98TRUU','I00182US','I00185US','H04386US','JBCDCORE','I12881US',
    'I05040US','I05039US','LF94TRUH','LBUTTRUU','H21247US','H30902US','IBXXC1D3',
    'H30914US','H30919US','H13203US','BGCLTRUH','H04401US','H04402US','H24641US',
    'I38941US','I13913US'], [
    'US govt','Global govt','Global IG agg','Global IG corp','IG 1-3Y','IG 5-7Y','IG 10+Y',
    'Global HY','US HY','US HY BB','US HY B','EM hard currency','EM corporate','EM BBB',
    'EM BB','EM B','Global IL','US IL','Global IL 1-10Y','Global CoCo','AT1 CoCo',
    'IG AT1/RT1','Non-IG AT1/RT1','Corp subordinated','Capital debt','Tier 2 A',
    'Tier 2 BBB','Convertibles','US loans','Structured credit']))
FUND_NAMES = dict(zip([
    'IEI US Equity','LQDE LN Equity','FRTGUSD LX Equity','IHYU LN Equity',
    'HHGI2AU LX Equity','IEMB LN Equity','ACMEMI2 LX Equity','TIP US Equity',
    'LGTGILI LE Equity','CWB US Equity','OBJCGAU FP Equity','NBCHEPA ID Equity'], [
    'iShares Treasury','iShares USD IG','Nordea US Corp','iShares USD HY',
    'Janus Global HY','iShares EM Bonds','AB EM Debt','iShares TIPS',
    'LGT IL Bonds','SPDR Convertibles','Lazard Convertibles','NB Corp Hybrid']))
SHORT_NAMES = {**{k+' Index':v for k,v in INDEX_NAMES.items()}, **FUND_NAMES}


def read(root, name, **kwargs):
    """Preserve round-trip precision from the frozen research evidence."""
    return pd.read_csv(Path(root)/name, float_precision='round_trip', **kwargs)


def digest(path):
    """Return file identity for the exhibit manifest."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    """Write strict JSON evidence."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str)+'\n', encoding='utf-8')


def compute(root, out):
    """Calculate systematic risk, a constrained frontier, and the fitted cluster tree."""
    import qis
    from scipy.cluster.hierarchy import fcluster
    from factorlasso.factor_covar import CurrentFactorCovarData
    from optimalportfolios.optimization.constraints import Constraints
    from optimalportfolios.optimization.config import OptimiserConfig
    from optimalportfolios.optimization.saa.min_variance_target_return import wrapper_min_variance_target_return
    from . import fi_validation as study

    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    roster = read(root, 'inputs/roster.csv', index_col=0).query("lane=='indices'")
    betas = read(root, 'R1/indices/E2_betas.csv', index_col=0).loc[roster.index]
    diagnostics = read(root, 'R1/indices/E2_diagnostics.csv', index_col=0).loc[roster.index]
    factors = read(root, 'inputs/factor_covar.csv', index_col=0).loc[betas.columns, betas.columns]
    residuals = pd.DataFrame({'residual_var':12*diagnostics.residual_var_monthly})
    model_risk = CurrentFactorCovarData(x_covar=factors, y_betas=betas, y_variances=residuals)
    vols = model_risk.get_model_vols()
    cov = model_risk.get_y_covar(residual_var_weight=0.0)
    cov.to_csv(out/'factor_only_covar.csv')
    assert np.allclose(cov, cov.T, atol=1e-13)
    assert np.linalg.eigvalsh(cov).min()>-1e-12
    cmas = read(root, 'summary/cmas.csv').query("arm=='E2'").set_index('ticker').loc[roster.index]
    assets = pd.DataFrame({'short_name':[SHORT_NAMES[t] for t in roster.index], 'group':roster['group'],
        'cma':cmas.base_total_cma, 'factor_vol':vols.sys_vol, 'total_vol':vols.total_vol})
    assert np.allclose(assets.total_vol, cmas.vol, atol=1e-10)
    assets.to_csv(out/'assets.csv', index_label='ticker')
    cutoff = pd.Timestamp('2026-06-30')
    risk = qis.RiskModel(covar={cutoff:cov})
    zero = pd.Series(0., index=cov.index)

    def volatility(weights):
        """Use the QIS covariance owner with the zero benchmark for absolute risk."""
        return float(risk.compute_tre_at_date(benchmark_weights=zero, portfolio_weights=weights, date=cutoff))

    independent_vols = []
    for ticker in assets.index:
        unit=zero.copy(); unit.loc[ticker]=1.
        independent_vols.append(volatility(unit))
    assert np.allclose(independent_vols, assets.factor_vol, atol=1e-12)
    mu = assets.cma
    constraints = Constraints(is_long_only=True, min_weights=zero.copy(), max_weights=zero+1.,
        min_exposure=1., max_exposure=1.)
    from optimalportfolios.optimization.covar_factorization import factorize_covariance
    alternate_covar=pd.DataFrame(factorize_covariance(cov.to_numpy(),eigenvalue_floor=1e-12).covar,index=cov.index,columns=cov.columns)
    checks = []

    def solve(floor, solver='MOSEK', factorize=True):
        """Use the existing SAA solver and reject fallback or infeasible solutions."""
        weights, result = wrapper_min_variance_target_return(pd_covar=cov if factorize else alternate_covar, expected_returns=mu,
            target_return=float(floor), constraints=constraints,
            optimiser_config=OptimiserConfig(solver=solver, factorize_covar=factorize, verbose=False),
            context='FI systematic frontier')
        assert result.accepted and result.compliant, str(result)
        assert result.fallback_source is None and result.status=='optimal', str(result)
        ret=float(weights.dot(mu)); vol=volatility(weights)
        assert abs(weights.sum()-1.)<2e-7 and weights.min()>-2e-7 and weights.max()<1.+2e-7
        assert ret>=floor-2e-7
        checks.append(dict(solver=solver, floor=float(floor), cma=ret, factor_vol=vol,
            budget_error=float(abs(weights.sum()-1)), min_weight=float(weights.min()),
            status=result.status, accepted=bool(result.accepted), compliant=bool(result.compliant),
            factorized=factorize))
        return weights, ret, vol

    w0, m0, s0 = solve(mu.min()-0.01)
    rows=[dict(target=float(mu.min()-0.01),cma=m0,factor_vol=s0)]
    weights=[w0]
    for target in np.linspace(m0,mu.max(),81)[1:]:
        w,m,s=solve(target);weights.append(w);rows.append(dict(target=float(target),cma=m,factor_vol=s))
    frontier=pd.DataFrame(rows)
    assert np.diff(frontier.cma).min()>-2e-7 and np.diff(frontier.factor_vol).min()>-2e-7
    frontier.to_csv(out/'frontier.csv',index=False)
    pd.DataFrame(weights).to_csv(out/'frontier_weights.csv', index_label='point')
    alternate=[]
    for i in [0,20,40,60,80]:
        w,m,s=solve(frontier.loc[i,'target'],solver='MOSEK',factorize=False)
        error=abs(s-frontier.loc[i,'factor_vol'])
        assert error<2e-5, (i,error)
        _,raw_m,raw_s=solve(frontier.loc[i,'target'],factorize=False)
        assert abs(raw_s-frontier.loc[i,'factor_vol'])<2e-5
        alternate.append(dict(point=i,risk_difference=error,cma_difference=abs(m-frontier.loc[i,'cma']),
            reduced_floor_risk_difference=abs(raw_s-frontier.loc[i,'factor_vol'])))
    for t,row in assets.iterrows():
        w,m,s=solve(row.cma)
        assert s<=row.factor_vol+2e-5,(t,s,row.factor_vol)
    pd.DataFrame(checks).to_csv(out/'solver_checks.csv',index=False)
    # Recover the actual estimator linkage, rather than cluster the new betas.
    study.ROOT=root
    x,y,full_roster=study.load('indices')
    fitted,audit=study.fit(x,y,full_roster,'Z')
    old=read(root,'R1/indices/Z_betas.csv',index_col=0)
    beta_error=float((fitted.estimated_betas-old).abs().max().max())
    assert beta_error<1e-7,beta_error
    linkage=np.asarray(fitted.linkage_,dtype=float)
    assert linkage.shape==(29,4)
    cutoff_distance=float(fitted.cutoff_)
    cluster=fcluster(linkage,cutoff_distance,criterion='distance')
    old_clusters=read(root,'R1/indices/Z_clusters.csv',index_col=0).iloc[:,0].reindex(y.columns).to_numpy()
    assert np.array_equal(cluster[:,None]==cluster[None,:],old_clusters[:,None]==old_clusters[None,:])
    pd.DataFrame(linkage,columns=['left','right','distance','size']).to_csv(out/'linkage.csv',index=False)
    pd.DataFrame({'ticker':y.columns,'short_name':[SHORT_NAMES[t] for t in y],
        'cluster':cluster}).to_csv(out/'dendrogram_labels.csv',index=False)
    pd.DataFrame({'ticker':list(SHORT_NAMES),'short_name':list(SHORT_NAMES.values())}).to_csv(out/'short_names.csv',index=False)
    dump(out/'checks.json',dict(status='passed',risk_convention='annual systematic MATF volatility; residual excluded',
        policy='E2',covariance_rank=int(np.linalg.matrix_rank(cov)),frontier_points=len(frontier),
        optimizer_eigenvalue_floor=1e-10,alternate_eigenvalue_floor=1e-12,plotted_risk_uses_original_covariance=True,
        constraints='long-only, fully invested; no cash sleeve, leverage or additional caps',
        alternate_solver_checks=alternate, asset_vol_reference_max_error=float(np.max(np.abs(independent_vols-assets.factor_vol))),
        dendrogram_cutoff=cutoff_distance,cluster_sizes=pd.Series(cluster).value_counts().sort_index().to_dict(),
        refit_beta_max_error=beta_error, refit_audit=audit,
        minimum_risk=dict(cma=m0,factor_vol=s0), maximum_return=dict(cma=float(frontier.iloc[-1].cma),factor_vol=float(frontier.iloc[-1].factor_vol))))
    dump(out/'manifest.json',{p.name:digest(p) for p in out.iterdir() if p.is_file()})
    print('Exhibit computation passed:',out,flush=True)


def verify(out):
    """Fail closed on changed evidence or missing keys."""
    out=Path(out)
    manifest=json.loads((out/'manifest.json').read_text())
    for name,value in manifest.items():
        assert digest(out/name)==value,name
    checks=json.loads((out/'checks.json').read_text())
    assert checks['status']=='passed' and checks['frontier_points']==81
    return checks


def figures(out, target):
    """Render the owner frontier and owner linkage with readable asset names."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import qis
    out,target=Path(out),Path(target)
    checks=verify(out)
    assets=read(out,'assets.csv',index_col=0)
    frontier=read(out,'frontier.csv')
    palette=['#244c78','#167d8d','#bd7730','#854c97','#39805c','#bd5667','#6c7184']
    groups=list(dict.fromkeys(assets.group));colors=dict(zip(groups,palette))
    fig,ax=plt.subplots(figsize=(10,8.6),layout='constrained')
    # QIS handles scatter coordinates and percentage formatting; labels are an editorial layer.
    for group,rows in assets.groupby('group',sort=False):
        qis.plot_scatter(df=rows,x='factor_vol',y='cma',full_sample_order=0,
            full_sample_color=colors[group],add_universe_model_label=False,markersize=48,
            xvar_format='{:.0%}',yvar_format='{:.0%}',x_limits=(0.,.132),y_limits=(.032,.098),
            xlabel='Annual MATF factor volatility',ylabel='Annual total CMA',ax=ax)
    ax.plot(frontier.factor_vol,frontier.cma,color='#172b40',lw=2,label='Long-only efficient frontier')
    ax.scatter(frontier.iloc[0].factor_vol,frontier.iloc[0].cma,marker='*',s=140,color='#172b40',zorder=5)
    # Reviewed label positions for this fixed-vintage exhibit; data coordinates remain unchanged.
    positions={
        'US govt':(.046,.046),'Global govt':(.030,.0445),'Global IG agg':(.024,.0525),
        'IG 1-3Y':(.005,.0475),'IG 5-7Y':(.076,.059),'IG 10+Y':(.105,.078),
        'Global IG corp':(.064,.057),'Global HY':(.016,.069),'US HY':(.026,.0707),
        'US HY BB':(.015,.0648),'US HY B':(.046,.076),'EM hard currency':(.087,.069),
        'EM corporate':(.073,.0658),'EM BBB':(.083,.073),'EM BB':(.094,.076),
        'EM B':(.108,.085),'Global IL':(.063,.0545),'US IL':(.045,.052),
        'Global IL 1-10Y':(.004,.052),'Global CoCo':(.025,.079),'AT1 CoCo':(.05,.0805),
        'IG AT1/RT1':(.08,.064),'Non-IG AT1/RT1':(.065,.0825),
        'Corp subordinated':(.005,.0605),'Capital debt':(.025,.0625),
        'Tier 2 A':(.063,.0607),'Tier 2 BBB':(.046,.0585),'Convertibles':(.113,.07),
        'US loans':(.005,.0563),'Structured credit':(.026,.0577)}
    assert set(positions)==set(assets.short_name)
    for _,row in assets.iterrows():
        ax.annotate(row.short_name,xy=(row.factor_vol,row.cma),xytext=positions[row.short_name],
            ha='left',va='center',fontsize=8.8,color=colors[row.group],
            bbox=dict(facecolor='white',edgecolor='none',alpha=.94,pad=1.5),
            arrowprops=dict(arrowstyle='-',color=colors[row.group],lw=.65,alpha=.65),zorder=4)
    ax.set_xlim(.0,max(.142,float(assets.factor_vol.max())+.012));ax.set_ylim(min(.04,float(assets.cma.min())-.008),max(.090,float(assets.cma.max())+.005))
    handles=[Line2D([0],[0],color='#172b40',lw=2,label='Long-only frontier')]+[
        Line2D([0],[0],marker='o',ls='',color=colors[g],label=g) for g in groups]
    ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.09),ncol=4,frameon=False,fontsize=9)
    ax.grid(alpha=.15);ax.set_title('Conditional prior: Table 12 universe',fontsize=12)
    for ext in ['pdf','png']:fig.savefig(target/('fi_frontier.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    linkage=read(out,'linkage.csv').to_numpy()
    labels=read(out,'dendrogram_labels.csv')
    fig,ax=plt.subplots(figsize=(10,8.6),layout='constrained')
    qis.plot_dendrogram(linkage=linkage,labels=labels.short_name.tolist(),
        cutoff=checks['dendrogram_cutoff'],ax=ax,orientation='right',fontsize=10,
        title='FactorLasso response clustering at 30 June 2026')
    ax.set_xlabel('Clustering distance');ax.grid(axis='x',alpha=.15)
    for ext in ['pdf','png']:fig.savefig(target/('fi_dendrogram.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
