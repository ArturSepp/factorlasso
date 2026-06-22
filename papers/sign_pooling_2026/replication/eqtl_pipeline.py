"""
eqtl_pipeline.py — empirical (eQTL) pipeline for the cluster-pooled sign paper.

Applies HCGL/FCGL to the yeast MAPK eQTL data (Brem & Kruglyak 2005): gene
expression (responses) regressed on genetic markers (predictors). The discovered
gene clusters are mapped to the four MAPK sub-pathways, the cluster-marker
structure is read as eQTL hotspots, and prediction is reported as a parity check
against per-response LASSO, consistent with the low cluster-recoverability regime
characterized in the simulation section.

Data
----
yeast_full.npz : Y (112 x 6216 expression), X (112 x 3244 genotypes), gene labels.
yeast.rda      : marker.pos (chromosome, position) for the 3244 markers.

Gene set
--------
KEGG MAPK signaling pathway (sce04011) four-module membership measured in the
data. The conditional-Gaussian-graphical-model literature (Yin & Li 2011) uses a
canonical 54-gene MAPK subset; we keep the full measured membership here.

Marker screen
-------------
Markers associated with the expression of at least two MAPK genes at p <= 0.01
(the screen of Yin & Li 2011), yielding on the order of 200 markers.

Run: python eqtl_pipeline.py
"""
# packages
import os
import numpy as np
import pandas as pd
from typing import Tuple
from scipy import stats
from sklearn.metrics import adjusted_rand_score
# factorlasso
from factorlasso import LassoModel, LassoModelType

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(_HERE, 'results')
_DATA = os.path.join(_HERE, 'data')

# KEGG MAPK sce04011 four-module membership (standard gene names)
MAPK_MODULES = {
    'pheromone (Fus3)':     ['STE2', 'STE3', 'GPA1', 'STE4', 'STE18', 'STE5', 'STE11', 'STE7',
                             'FUS3', 'STE12', 'FAR1', 'FUS1', 'SST2', 'KAR4', 'AGA1', 'BAR1', 'MFA1', 'MFA2'],
    'filamentation (Kss1)': ['KSS1', 'TEC1', 'STE20', 'STE50', 'MSB2', 'SHO1', 'CDC42', 'CDC24',
                             'BEM1', 'DIG1', 'DIG2', 'FLO11'],
    'osmolarity (Hog1)':    ['SLN1', 'YPD1', 'SSK1', 'SSK2', 'SSK22', 'PBS2', 'HOG1', 'GPD1', 'CTT1',
                             'MSN2', 'MSN4', 'GLO1', 'NBP2', 'RCK2', 'SKO1', 'HOT1', 'SMP1'],
    'cell wall (Slt2)':     ['WSC1', 'WSC2', 'WSC3', 'MID2', 'MTL1', 'ROM1', 'ROM2', 'RHO1', 'PKC1',
                             'BCK1', 'MKK1', 'MKK2', 'SLT2', 'MPK1', 'RLM1', 'SWI4', 'SWI6', 'FKS1', 'FKS2', 'GSC2', 'KNR4'],
    'sporulation (Smk1)':   ['SMK1'],
}


def _zscore(df: pd.DataFrame) -> pd.DataFrame:
    """standardize columns to zero mean and unit variance."""
    return (df - df.mean()) / df.std(ddof=0)


def load_data(npz: str = None, rda: str = None) -> tuple:
    """load expression, genotypes and marker positions.

    Returns
    -------
    y_all : pd.DataFrame, shape (n, 6216)  expression, columns are gene labels
    x_all : pd.DataFrame, shape (n, 3244)  genotypes, columns m0..m3243
    marker_pos : pd.DataFrame, shape (3244, 2)  columns 'chr', 'pos'
    """
    import rdata
    npz = npz or os.path.join(_DATA, 'yeast_full.npz')
    rda = rda or os.path.join(_DATA, 'yeast.rda')
    d = np.load(npz, allow_pickle=True)
    y_all = pd.DataFrame(d['Y'].astype(float), columns=[str(g) for g in d['genes']])
    x_all = pd.DataFrame(d['X'].astype(float), columns=[f"m{j}" for j in range(d['X'].shape[1])])
    mp = np.asarray(rdata.read_rda(rda)['yeast']['marker.pos'], dtype=float)
    marker_pos = pd.DataFrame(mp, columns=['chr', 'pos'], index=x_all.columns)
    return y_all, x_all, marker_pos


def select_mapk_genes(y_all: pd.DataFrame) -> tuple:
    """select the MAPK genes present in the data and their sub-pathway labels."""
    name_to_module = {nm: mod for mod, names in MAPK_MODULES.items() for nm in names}
    present = [g for g in y_all.columns if str(g).upper() in name_to_module]
    modules = pd.Series({g: name_to_module[str(g).upper()] for g in present}, name='module')
    return present, modules


def screen_markers(x_all: pd.DataFrame, y_genes: pd.DataFrame,
                   p_thresh: float = 0.01, min_genes: int = 2) -> np.ndarray:
    """keep markers associated with >= min_genes genes at p <= p_thresh (Yin & Li 2011).

    Uses the marginal correlation t-test, t = r sqrt((n-2)/(1-r^2)).
    Returns the integer column positions of the kept markers.
    """
    if not 0.0 < p_thresh < 1.0:
        raise ValueError(f"p_thresh must lie in (0, 1), got {p_thresh!r}")
    n = len(x_all)
    xz = _zscore(x_all).values
    yz = _zscore(y_genes).values
    r = (yz.T @ xz) / n                                   # genes x markers
    t = r * np.sqrt((n - 2) / np.clip(1.0 - r ** 2, 1e-12, None))
    p = 2.0 * stats.t.sf(np.abs(t), df=n - 2)             # genes x markers
    counts = (p <= p_thresh).sum(axis=0)                  # per marker, number of genes
    return np.where(counts >= min_genes)[0]


def ld_block_markers(x_all: pd.DataFrame, marker_pos: pd.DataFrame,
                     corr_thresh: float = 0.92) -> np.ndarray:
    """collapse consecutive markers in tight linkage to one representative each.

    Walks the markers in genomic order and keeps a marker when it starts a new
    chromosome or its genotype correlation with the current representative falls
    below corr_thresh. Reduces the redundant 3244-marker panel to a representative
    set on the order of 600, matching the preprocessing of Yin & Li (2011).
    """
    x = x_all.values
    chrom = marker_pos['chr'].values
    keep, rep = [], -1
    for i in range(x.shape[1]):
        if rep < 0 or chrom[i] != chrom[rep]:
            keep.append(i)
            rep = i
            continue
        r = np.corrcoef(x[:, i], x[:, rep])[0, 1]
        if not np.isfinite(r) or abs(r) < corr_thresh:
            keep.append(i)
            rep = i
    return np.asarray(keep)


def fit_model(model_type: LassoModelType, x_s: pd.DataFrame, y_genes: pd.DataFrame,
              reg_lambda: float = 0.3, threshold_t: float = 2.0,
              cutoff_fraction: float = 0.7, group_data: pd.Series = None) -> LassoModel:
    """fit a factorlasso model with gated sign constraints."""
    model = LassoModel(model_type=model_type, reg_lambda=reg_lambda, demean=True,
                       cutoff_fraction=cutoff_fraction, group_data=group_data,
                       auto_sign_constraints=True, auto_sign_threshold_t=threshold_t)
    model.fit(x=x_s, y=y_genes)
    return model


def cluster_subpathway(clusters: pd.Series, modules: pd.Series) -> tuple:
    """agreement between discovered clusters and MAPK sub-pathways (ARI and purity)."""
    genes = clusters.index
    lab_c = clusters.values
    lab_m = modules.reindex(genes).values
    ari = adjusted_rand_score(lab_m, lab_c)
    frame = pd.DataFrame({'cluster': lab_c, 'module': lab_m})
    purity = frame.groupby('cluster')['module'].agg(
        lambda s: s.value_counts(normalize=True).iloc[0]).mean()
    return float(ari), float(purity)


def within_cluster_sign_coherence(x_s: pd.DataFrame, y_genes: pd.DataFrame,
                                  clusters: pd.Series, p_thresh: float = 0.05) -> float:
    """raw marginal-sign agreement within discovered clusters (justifies pooling).

    For each cluster and marker, the fraction of significantly-signed genes that
    agree with the modal sign, averaged over cells with at least two signed genes.
    """
    n = len(x_s)
    xz = _zscore(x_s).values
    yz = _zscore(y_genes).values
    r = (yz.T @ xz) / n
    t = r * np.sqrt((n - 2) / np.clip(1.0 - r ** 2, 1e-12, None))
    p = 2.0 * stats.t.sf(np.abs(t), df=n - 2)
    s_raw = np.where(p <= p_thresh, np.sign(r), 0.0)      # genes x markers
    genes = list(y_genes.columns)
    agrees = []
    for c in clusters.unique():
        idx = [genes.index(g) for g in clusters[clusters == c].index]
        if len(idx) < 2:
            continue
        block = s_raw[idx]                                # |cluster| x markers
        for j in range(block.shape[1]):
            signed = block[:, j][block[:, j] != 0]
            if len(signed) >= 2:
                modal = 1.0 if (signed > 0).sum() >= (signed < 0).sum() else -1.0
                agrees.append(np.mean(signed == modal))
    return float(np.mean(agrees)) if agrees else float('nan')


def marker_hub_counts(model: LassoModel, marker_pos: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    """per-marker number of genes loaded on it (hub count) with chromosomal position."""
    beta = model.estimated_betas if hasattr(model, 'estimated_betas') else model.coef_
    beta = np.asarray(beta)
    hub = (np.abs(beta) > eps).sum(axis=0)                # per marker
    out = marker_pos.copy()
    out['hub_count'] = hub
    return out


def prediction_parity(x_s: pd.DataFrame, y_genes: pd.DataFrame, n_folds: int = 3,
                      reg_lambda: float = 0.3, seed: int = 0) -> pd.DataFrame:
    """k-fold out-of-sample median R^2 per gene for HCGL, FCGL and per-response LASSO."""
    rng = np.random.default_rng(seed)
    n = len(x_s)
    folds = np.array_split(rng.permutation(n), n_folds)
    specs = {'HCGL': dict(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO, cutoff_fraction=0.7),
             'FCGL': dict(model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO, cutoff_fraction=0.7,
                          group_penalty='yuan_lin'),
             'LASSO': dict(model_type=LassoModelType.LASSO)}
    rows = []
    for name, spec in specs.items():
        r2_per_gene = []
        for te in folds:
            tr = np.setdiff1d(np.arange(n), te)
            xtr, xte = x_s.iloc[tr], x_s.iloc[te]
            ytr, yte = y_genes.iloc[tr], y_genes.iloc[te]
            mu_x, sd_x = xtr.mean(), xtr.std(ddof=0).replace(0, 1)
            mu_y, sd_y = ytr.mean(), ytr.std(ddof=0).replace(0, 1)
            xtr_z, xte_z = (xtr - mu_x) / sd_x, (xte - mu_x) / sd_x
            ytr_z = (ytr - mu_y) / sd_y
            m = LassoModel(reg_lambda=reg_lambda, demean=True, auto_sign_constraints=True,
                           auto_sign_threshold_t=2.0, **spec)
            m.fit(x=xtr_z, y=ytr_z)
            beta = np.asarray(m.estimated_betas if hasattr(m, 'estimated_betas') else m.coef_)
            pred_z = xte_z.values @ beta.T                # test x genes (standardized)
            yte_z = ((yte - mu_y) / sd_y).values
            ss_res = ((yte_z - pred_z) ** 2).sum(axis=0)
            ss_tot = (yte_z ** 2).sum(axis=0)
            r2_per_gene.append(1.0 - ss_res / np.clip(ss_tot, 1e-12, None))
        r2 = np.concatenate([np.atleast_1d(x) for x in r2_per_gene]) if False else np.mean(r2_per_gene, axis=0)
        rows.append(dict(method=name, oos_r2_median=float(np.median(r2)),
                         oos_r2_mean=float(np.mean(r2))))
    return pd.DataFrame(rows)


def within_cluster_residual_corr(x_s: pd.DataFrame, y_genes: pd.DataFrame,
                                 beta: np.ndarray, clusters: pd.Series) -> Tuple[float, float]:
    """size-weighted within-cluster correlation of the fit residuals and of the raw expression.

    The residual correlation drives the design effect that attenuates the pooling
    gain, and it is reported against the total expression correlation that the
    clustering reads. Residual = Y - X beta', using the fitted loadings.
    """
    resid = y_genes.values - x_s.values @ np.asarray(beta).T
    total = y_genes.values - y_genes.values.mean(axis=0)
    lab = clusters.reindex(y_genes.columns).values

    def _mean_within(M: np.ndarray) -> float:
        vals, sizes = [], []
        for c in np.unique(lab):
            idx = np.where(lab == c)[0]
            if len(idx) < 2:
                continue
            R = np.corrcoef(M[:, idx], rowvar=False)
            q = len(idx)
            vals.append((R.sum() - q) / (q * (q - 1)))
            sizes.append(q)
        return float(np.average(vals, weights=sizes))
    return _mean_within(resid), _mean_within(total)


def run_pipeline(reg_lambda: float = 0.3, threshold_t: float = 2.0,
                 cutoff_fraction: float = 0.7) -> dict:
    """end-to-end pipeline; saves results to _OUT and returns key frames."""
    os.makedirs(_OUT, exist_ok=True)
    y_all, x_all, marker_pos = load_data()
    genes, modules = select_mapk_genes(y_all)
    y_g = _zscore(y_all[genes])
    keep_ld = ld_block_markers(x_all, marker_pos)
    x_ld = x_all.iloc[:, keep_ld]
    mpos_ld = marker_pos.iloc[keep_ld].reset_index(drop=True)
    keep = screen_markers(x_ld, y_all[genes])
    x_s = _zscore(x_ld.iloc[:, keep])
    mpos = mpos_ld.iloc[keep].reset_index(drop=True)

    hcgl = fit_model(LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO, x_s, y_g,
                     reg_lambda=reg_lambda, threshold_t=threshold_t, cutoff_fraction=cutoff_fraction)
    clusters = hcgl.clusters_
    ari, purity = cluster_subpathway(clusters, modules)
    coherence = within_cluster_sign_coherence(x_s, y_g, clusters)
    resid_rho, total_rho = within_cluster_residual_corr(x_s, y_g, hcgl.estimated_betas, clusters)
    hubs = marker_hub_counts(hcgl, mpos)
    parity = prediction_parity(x_ld.iloc[:, keep].reset_index(drop=True),
                               y_all[genes].reset_index(drop=True), reg_lambda=reg_lambda)

    # persist
    signs = pd.DataFrame(hcgl.derived_signs_.values, index=genes, columns=x_s.columns)
    cl = pd.DataFrame({'gene': clusters.index, 'cluster': clusters.values,
                       'module': modules.reindex(clusters.index).values})
    cl.to_csv(f"{_OUT}/clusters.csv", index=False)
    signs.to_csv(f"{_OUT}/derived_signs.csv")
    hubs.to_csv(f"{_OUT}/hotspots.csv", index=False)
    parity.to_csv(f"{_OUT}/parity.csv", index=False)
    np.savez(f"{_OUT}/fit.npz", beta=np.asarray(hcgl.estimated_betas), genes=np.array(genes),
             markers=np.array(x_s.columns), corr=y_g.corr().values)
    pd.Series(dict(n_genes=len(genes), n_markers=len(keep), n_clusters=clusters.nunique(),
                   ari=ari, purity=purity, sign_coherence=coherence,
                   resid_within_corr=resid_rho, total_within_corr=total_rho)).to_csv(f"{_OUT}/summary.csv")

    print(f"genes (MAPK in data): {len(genes)}   markers (>=2 genes, p<=0.01): {len(keep)}")
    print(f"HCGL clusters (cutoff {cutoff_fraction}): {clusters.nunique()}")
    print(f"cluster vs sub-pathway: ARI={ari:.3f}, purity={purity:.3f}")
    print(f"within-cluster raw sign coherence (p<=0.05): {coherence:.3f}")
    print(f"within-cluster correlation: residual={resid_rho:.3f}, total={total_rho:.3f}")
    print("\ntop hotspot loci (by genes loaded):")
    print(hubs.sort_values('hub_count', ascending=False).head(8).to_string(index=False))
    print("\nprediction parity (3-fold OOS R^2):")
    print(parity.round(3).to_string(index=False))
    return dict(clusters=cl, signs=signs, hotspots=hubs, parity=parity, modules=modules)


if __name__ == '__main__':
    run_pipeline()
