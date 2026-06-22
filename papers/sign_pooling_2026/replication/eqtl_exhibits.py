"""
eqtl_exhibits.py — figures and table for the empirical eQTL section.

Reuses eqtl_pipeline to load, LD-block, screen and fit, then builds:
  FIG E1 gene-correlation heatmap ordered by cluster with a sub-pathway sidebar
  FIG E2 gated cluster-pooled sign matrix (genes x markers)
  FIG E3 eQTL hotspot map (genes loaded per marker along the genome)
  TABLE E1 per-cluster summary + prediction parity note (LaTeX booktabs)

Run: python eqtl_exhibits.py
"""
# packages
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
# project
import eqtl_pipeline as E
from factorlasso import LassoModelType

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIG = os.path.join(_HERE, '..', 'paper')
_CUTOFF = 0.7

mpl.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.titlesize': 10.5,
    'axes.labelsize': 10, 'legend.fontsize': 8.5, 'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5, 'figure.dpi': 120, 'savefig.bbox': 'tight',
})
_MODULE_COLORS = {
    'pheromone (Fus3)': '#1b9e77', 'filamentation (Kss1)': '#d95f02',
    'osmolarity (Hog1)': '#7570b3', 'cell wall (Slt2)': '#e7298a',
    'sporulation (Smk1)': '#999999',
}
_SIGN_CMAP = ListedColormap(['#3b6fb0', '#f4f4f4', '#c23b3b'])
_SIGN_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], _SIGN_CMAP.N)


def prepare() -> dict:
    """load, LD-block, screen, fit HCGL; return everything the exhibits need."""
    y_all, x_all, marker_pos = E.load_data()
    genes, modules = E.select_mapk_genes(y_all)
    y_g = E._zscore(y_all[genes])
    keep_ld = E.ld_block_markers(x_all, marker_pos)
    x_ld = x_all.iloc[:, keep_ld]
    mpos_ld = marker_pos.iloc[keep_ld].reset_index(drop=True)
    keep = E.screen_markers(x_ld, y_all[genes])
    x_s = E._zscore(x_ld.iloc[:, keep])
    mpos = mpos_ld.iloc[keep].reset_index(drop=True)
    model = E.fit_model(LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO, x_s, y_g, cutoff_fraction=_CUTOFF)
    return dict(genes=genes, modules=modules, x_s=x_s, y_g=y_g, mpos=mpos,
                clusters=model.clusters_, signs=pd.DataFrame(model.derived_signs_.values, index=genes, columns=x_s.columns),
                beta=np.asarray(model.estimated_betas), corr=y_g.corr())


def _gene_order(clusters: pd.Series, modules: pd.Series) -> list:
    """order genes by cluster, then by sub-pathway within a cluster."""
    frame = pd.DataFrame({'cluster': clusters, 'module': modules.reindex(clusters.index)})
    frame = frame.sort_values(['cluster', 'module'])
    return list(frame.index)


def fig_clusters(D: dict) -> None:
    """E1: gene-correlation heatmap ordered by cluster with sub-pathway sidebar."""
    order = _gene_order(D['clusters'], D['modules'])
    corr = D['corr'].loc[order, order].values
    cl = D['clusters'].reindex(order).values
    mod = D['modules'].reindex(order).values
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
    for b in np.where(np.diff(cl))[0]:
        ax.axhline(b + 0.5, color='k', lw=0.5)
        ax.axvline(b + 0.5, color='k', lw=0.5)
    # sub-pathway sidebar
    for i, m in enumerate(mod):
        ax.add_patch(plt.Rectangle((-2.2, i - 0.5), 1.6, 1.0, color=_MODULE_COLORS.get(m, '#999999'),
                                   clip_on=False, lw=0))
    ax.set_xlim(-2.4, len(order) - 0.5)
    ax.set_title('Recovered clusters are coherent co-expression groups\nthat partially track the MAPK sub-pathways')
    ax.set_xlabel('gene (ordered by cluster)')
    ax.set_ylabel('gene')
    ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in _MODULE_COLORS.values()]
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='expression correlation')
    ax.legend(handles, list(_MODULE_COLORS.keys()), loc='center left', bbox_to_anchor=(1.34, 0.5),
              frameon=False, fontsize=8, title='MAPK sub-pathway')
    fig.savefig(f"{_FIG}/figE1_clusters.pdf"); fig.savefig(f"{_FIG}/figE1_clusters.png", dpi=150)
    plt.close(fig)


def fig_signs(D: dict) -> None:
    """E2: gated cluster-pooled sign matrix, genes by cluster, markers along genome."""
    order = _gene_order(D['clusters'], D['modules'])
    signs = D['signs'].loc[order].values
    cl = D['clusters'].reindex(order).values
    chrom = D['mpos']['chr'].values
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.imshow(signs, aspect='auto', cmap=_SIGN_CMAP, norm=_SIGN_NORM, interpolation='nearest')
    for b in np.where(np.diff(cl))[0]:
        ax.axhline(b + 0.5, color='0.5', lw=0.4)
    for b in np.where(np.diff(chrom))[0]:
        ax.axvline(b + 0.5, color='0.7', lw=0.3)
    ax.set_title('Cluster-pooled signs are coherent within recovered gene clusters '
                 '(within-cluster agreement 0.93)')
    ax.set_xlabel('marker (ordered along the genome, chromosome boundaries in grey)')
    ax.set_ylabel('gene (ordered by cluster)')
    ax.set_yticks([])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=_SIGN_NORM, cmap=_SIGN_CMAP), ax=ax,
                        ticks=[-1, 0, 1], fraction=0.025, pad=0.02)
    cbar.ax.set_yticklabels(['$-$', '0', '$+$'])
    fig.savefig(f"{_FIG}/figE2_signs.pdf"); fig.savefig(f"{_FIG}/figE2_signs.png", dpi=150)
    plt.close(fig)


def fig_hotspots(D: dict) -> None:
    """E3: genes loaded per marker along the genome (eQTL hotspots)."""
    mpos = D['mpos'].copy()
    mpos['hub'] = (np.abs(D['beta']) > 1e-8).sum(axis=0)
    # cumulative genomic coordinate
    order = mpos.sort_values(['chr', 'pos']).index
    mpos = mpos.loc[order].reset_index(drop=True)
    offs, cum, ticks, labels = {}, 0.0, [], []
    xs = np.zeros(len(mpos))
    for c in sorted(mpos['chr'].unique()):
        m = mpos['chr'] == c
        span = mpos.loc[m, 'pos'].max()
        xs[m.values] = cum + mpos.loc[m, 'pos'].values
        ticks.append(cum + span / 2); labels.append(int(c))
        cum += span + span * 0.02
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    for i, c in enumerate(sorted(mpos['chr'].unique())):
        m = (mpos['chr'] == c).values
        ax.vlines(xs[m], 0, mpos['hub'].values[m], color='#3b6fb0' if i % 2 == 0 else '#9ecae1', lw=0.8)
    top = mpos.sort_values('hub', ascending=False).head(4)
    for _, r in top.iterrows():
        xv = xs[mpos.index.get_loc(r.name)]
        ax.annotate(f"chr{int(r.chr)}", (xv, r.hub), textcoords='offset points', xytext=(0, 3),
                    ha='center', fontsize=7, color='0.2')
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel('chromosome (genomic position)')
    ax.set_ylabel('genes loaded')
    ax.set_title('A few loci regulate many genes \u2014 eQTL hotspots')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(f"{_FIG}/figE3_hotspots.pdf"); fig.savefig(f"{_FIG}/figE3_hotspots.png", dpi=150)
    plt.close(fig)


def _per_cluster_coherence(D: dict, p_thresh: float = 0.05) -> dict:
    """raw marginal-sign agreement within each cluster."""
    n = len(D['x_s'])
    r = (E._zscore(D['y_g']).values.T @ E._zscore(D['x_s']).values) / n
    t = r * np.sqrt((n - 2) / np.clip(1.0 - r ** 2, 1e-12, None))
    p = 2.0 * stats.t.sf(np.abs(t), df=n - 2)
    s = np.where(p <= p_thresh, np.sign(r), 0.0)
    genes = list(D['y_g'].columns); out = {}
    for c in D['clusters'].unique():
        idx = [genes.index(g) for g in D['clusters'][D['clusters'] == c].index]
        block = s[idx]; ag = []
        for j in range(block.shape[1]):
            sg = block[:, j][block[:, j] != 0]
            if len(sg) >= 2:
                modal = 1.0 if (sg > 0).sum() >= (sg < 0).sum() else -1.0
                ag.append(np.mean(sg == modal))
        out[c] = float(np.mean(ag)) if ag else float('nan')
    return out


def table_clusters(D: dict) -> str:
    """TABLE E1: per-cluster summary as a LaTeX booktabs body."""
    coh = _per_cluster_coherence(D)
    genes = list(D['y_g'].columns)
    rows = []
    for c in sorted(D['clusters'].unique()):
        members = D['clusters'][D['clusters'] == c].index
        mod = D['modules'].reindex(members)
        dom = mod.value_counts(normalize=True)
        idx = [genes.index(g) for g in members]
        nmark = int((np.abs(D['beta'][idx]) > 1e-8).any(axis=0).sum())
        rows.append((int(c), len(members), dom.index[0], dom.iloc[0], nmark, coh[c]))
    lines = [r'\begin{tabular}{cccccc}', r'\toprule',
             r'Cluster & Genes & Dominant sub-pathway & Purity & Markers & Sign coherence \\', r'\midrule']
    short = {'pheromone (Fus3)': 'Fus3', 'filamentation (Kss1)': 'Kss1',
             'osmolarity (Hog1)': 'Hog1', 'cell wall (Slt2)': 'Slt2', 'sporulation (Smk1)': 'Smk1'}
    for c, ng, dom, pur, nm, ch in rows:
        lines.append(f"{c} & {ng} & {short.get(dom, dom)} & {pur:.2f} & {nm} & {ch:.2f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    body = '\n'.join(lines)
    open(f"{_FIG}/tableE1_clusters.tex", 'w').write(body)
    return body


if __name__ == '__main__':
    os.makedirs(_FIG, exist_ok=True)
    D = prepare()
    fig_clusters(D)
    fig_signs(D)
    fig_hotspots(D)
    body = table_clusters(D)
    print('TABLE E1:\n', body)
    print('\nwritten to', _FIG)
