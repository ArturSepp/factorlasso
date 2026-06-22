"""
exhibits.py — figures and tables for the simulation section, built from the CSVs
written by sign_pooling_simulation.run_local_test.

Figures (PDF + PNG): S1 recoverability, S2 gate ROC, S3 sign matrices, S4 consistency,
S5 correlated predictors (reads sign_pooling_robustness.py CSVs).
Tables (LaTeX booktabs): S1 base metrics, S2 regime map.

Run: python exhibits.py
"""
# packages
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import norm
from scipy.optimize import curve_fit
# project
import sign_pooling_simulation as S

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.join(_HERE, 'results')
_FIG = os.path.join(_HERE, '..', 'paper')

mpl.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.titlesize': 10.5,
    'axes.labelsize': 10, 'legend.fontsize': 8.5, 'xtick.labelsize': 9,
    'ytick.labelsize': 9, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 120, 'savefig.bbox': 'tight',
})

# method styling: colorblind-safe colours plus distinct line styles for grayscale
_STYLE = {
    'LASSO':       dict(color='#444444', marker='o', ls='-',  label='per-response (LASSO)'),
    'HCGL':        dict(color='#1f77b4', marker='s', ls='--', label='pooled, estimated clusters (HCGL)'),
    'GROUP-fixed': dict(color='#d62728', marker='^', ls='-.', label='pooled, known clusters'),
}


def _save(fig: plt.Figure, name: str) -> None:
    """write a figure to PDF (vector) and PNG (preview)."""
    os.makedirs(_FIG, exist_ok=True)
    fig.savefig(os.path.join(_FIG, name + '.pdf'))
    fig.savefig(os.path.join(_FIG, name + '.png'), dpi=150)
    plt.close(fig)


def fig_recoverability() -> None:
    """S1: sign recovery vs cluster recoverability (r2 sweep)."""
    df = pd.read_csv(os.path.join(_RES, 'recoverability.csv'))
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.axvspan(0.10, 0.20, color='0.9', zorder=0)
    ax.text(0.15, 0.515, 'eQTL\nregime', ha='center', va='bottom', fontsize=8, color='0.45')
    for m in ['LASSO', 'HCGL', 'GROUP-fixed']:
        d = df[df.method == m].sort_values('r2'); st = _STYLE[m]
        ax.errorbar(d.r2, d.sign_recovery, yerr=d.sign_recovery_se, color=st['color'],
                    marker=st['marker'], ls=st['ls'], label=st['label'], capsize=2, markersize=5)
    ax.set_xlabel('population $R^2$ per response')
    ax.set_ylabel('sign recovery')
    ax.set_ylim(0.5, 1.02)
    ax.set_title('Sign-pooling\u2019s gain tracks cluster recoverability')
    ax.legend(loc='lower right', frameon=False)
    _save(fig, 'figS1_recoverability')


def fig_gate_roc() -> None:
    """S2: false-sign vs sensitivity across the gate threshold."""
    df = pd.read_csv(os.path.join(_RES, 'gate_roc.csv')).sort_values('threshold')
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(df.false_sign, df.sensitivity, color='#d62728', marker='o', ls='-', markersize=5)
    for _, r in df.iterrows():
        ax.annotate(f"{r.threshold:.2f}", (r.false_sign, r.sensitivity),
                    textcoords='offset points', xytext=(6, -1), fontsize=7, color='0.35')
    knee = df[df.threshold == 2.5].iloc[0]
    ax.scatter([knee.false_sign], [knee.sensitivity], s=90, facecolors='none', edgecolors='black', zorder=5)
    ax.annotate('threshold 2.5:\n0.87 sensitivity, 0.07 false', (knee.false_sign, knee.sensitivity),
                textcoords='offset points', xytext=(28, -30), fontsize=8,
                arrowprops=dict(arrowstyle='->', color='0.35'))
    ax.set_xlabel('false-sign rate (null cells)')
    ax.set_ylabel('sensitivity (true cells)')
    ax.set_title('The gate trades sensitivity for false-sign control')
    ax.set_xlim(-0.02, 0.75); ax.set_ylim(0.6, 1.01)
    _save(fig, 'figS2_gate_roc')


def fig_consistency() -> None:
    """S4: sign recovery vs n (empirical consistency), with the Gaussian-gate rate."""
    df = pd.read_csv(os.path.join(_RES, 'sign_consistency.csv'))
    cfg = S.DgpConfig()
    m_size = cfg.n_responses // cfg.n_clusters     # cluster size (effective-sample factor)
    tau = 0.75                                     # gate threshold used in this study
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for m in ['LASSO', 'GROUP-fixed']:
        d = df[df.method == m].sort_values('n_obs'); st = _STYLE[m]
        ax.errorbar(d.n_obs, d.sign_recovery, yerr=d.sign_recovery_se, color=st['color'],
                    marker=st['marker'], ls=st['ls'], label=st['label'], capsize=2, markersize=5)
    # Theoretical rate: on the support the gated sign is correct with probability
    # Phi(c*sqrt(n) - tau); pooling a cluster of size m scales the effective sample
    # to m*n, i.e. Phi(c*sqrt(m*n) - tau). Fit the single constant c to the
    # per-response points; the pooled curve uses the SAME c with no refit, so it is a
    # parameter-free prediction of the cluster-size scaling.
    lasso = df[df.method == 'LASSO'].sort_values('n_obs')
    rate = lambda n, c: norm.cdf(c * np.sqrt(n) - tau)
    c_hat, _ = curve_fit(rate, lasso.n_obs.values, lasso.sign_recovery.values, p0=[0.15])
    c_hat = float(c_hat[0])
    ng = np.geomspace(lasso.n_obs.min() * 0.9, lasso.n_obs.max() * 1.05, 200)
    ax.plot(ng, rate(ng, c_hat), color=_STYLE['LASSO']['color'], ls=':', lw=1.4, alpha=0.8,
            label='per-response rate (fit)')
    ax.plot(ng, rate(ng * m_size, c_hat), color=_STYLE['GROUP-fixed']['color'], ls=':', lw=1.4,
            alpha=0.8, label=fr'pooled rate, $m={m_size}$ (no refit)')
    ax.set_xscale('log', base=2)
    ax.set_xticks([40, 80, 160, 320, 640]); ax.set_xticklabels([40, 80, 160, 320, 640])
    ax.set_xlabel('sample size $T$ (log scale)')
    ax.set_ylabel('sign recovery')
    ax.set_title('Pooling reaches sign consistency at far smaller samples')
    ax.legend(loc='lower right', frameon=False, fontsize=8)
    _save(fig, 'figS4_consistency')


def fig_sign_matrices(seed: int = 0, r2: float = 0.20, threshold_t: float = 2.0) -> None:
    """S3: true vs pooled vs per-response sign matrices for one replicate."""
    cfg = S.DgpConfig(r2=r2)
    x, y, true_sign, labels = S.simulate(cfg, seed)
    d_pool = S.derive_signs(S.SignMethod.GROUP_FIXED, x, y, labels, threshold_t=threshold_t)
    d_lasso = S.derive_signs(S.SignMethod.LASSO, x, y, labels, threshold_t=threshold_t)
    cmap = ListedColormap(['#3b6fb0', '#f4f4f4', '#c23b3b'])  # -1, 0, +1
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    panels = [(true_sign, 'True signs'), (d_pool, 'Pooled (known clusters)'), (d_lasso, 'Per-response (LASSO)')]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.8), constrained_layout=True)
    im = None
    for ax, (mat, title) in zip(axes, panels):
        im = ax.imshow(mat, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('factor')
        for b in np.where(np.diff(labels))[0]:
            ax.axhline(b + 0.5, color='0.55', lw=0.4)
    axes[0].set_ylabel('response (ordered by cluster)')
    cbar = fig.colorbar(im, ax=list(axes), ticks=[-1, 0, 1], shrink=0.75)
    cbar.ax.set_yticklabels(['$-$', '0', '$+$'])
    fig.suptitle(f'Pooling suppresses spurious within-cluster sign alternations '
                 f'($R^2$={r2}, threshold {threshold_t})', fontsize=10)
    os.makedirs(_FIG, exist_ok=True)
    fig.savefig(os.path.join(_FIG, 'figS3_sign_matrices.pdf'))
    fig.savefig(os.path.join(_FIG, 'figS3_sign_matrices.png'), dpi=150)
    plt.close(fig)


def table_base() -> str:
    """TABLE S1: base-config metrics as a LaTeX booktabs body."""
    df = pd.read_csv(os.path.join(_RES, 'base_table.csv')).set_index('method')
    disp = {'LASSO': 'Per-response (LASSO)', 'GROUP-fixed': 'Pooled, known clusters',
            'HCGL': 'Pooled, estimated (HCGL)'}

    def cell(m: str, base: str) -> str:
        return f"{df.loc[m, base]:.3f} ({df.loc[m, base + '_se']:.3f})"
    lines = [r'\begin{tabular}{lcccc}', r'\toprule',
             r'Method & Sign recovery & Sign flip & Abstain (true) & False sign \\', r'\midrule']
    for m in ['LASSO', 'GROUP-fixed', 'HCGL']:
        lines.append(f"{disp[m]} & {cell(m, 'sign_recovery')} & {cell(m, 'flip')} & "
                     f"{cell(m, 'abstain_true')} & {cell(m, 'false_sign')} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    body = '\n'.join(lines)
    open(os.path.join(_FIG, 'tableS1_base.tex'), 'w').write(body)
    return body


def table_regime() -> str:
    """TABLE S2: regime map as a LaTeX booktabs body."""
    df = pd.read_csv(os.path.join(_RES, 'regime_map.csv'))
    lines = [r'\begin{tabular}{ccccc}', r'\toprule',
             r'$R^2$ & $T$ & Pooled (HCGL) & Per-response & Gap \\', r'\midrule']
    last = None
    for _, r in df.iterrows():
        if last is not None and r.r2 != last:
            lines.append(r'\addlinespace')
        lines.append(f"{r.r2:.2f} & {int(r.n_obs)} & {r.pooled:.3f} & {r.per_response:.3f} & {r.gap:+.3f} \\\\")
        last = r.r2
    lines += [r'\bottomrule', r'\end{tabular}']
    body = '\n'.join(lines)
    open(os.path.join(_FIG, 'tableS2_regime.tex'), 'w').write(body)
    return body


def fig_correlated() -> None:
    """S5: sign recovery, flip, and false-sign vs predictor correlation.

    Reads correlated_predictors.csv (sign_pooling_robustness.study_correlated).
    Stresses the marginal-sign-agreement condition (A6): as factor_corr rises the
    derived marginal sign disagrees with the partial sign, and pooling amplifies
    the resulting flips.
    """
    df = pd.read_csv(os.path.join(_RES, 'correlated_predictors.csv'))
    panels = [('recovery', 'recovery_se', 'sign recovery'),
              ('flip', 'flip_se', 'sign-flip rate'),
              ('false_sign', 'false_sign_se', 'false-sign rate')]
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.2), constrained_layout=True)
    for ax, (col, se, ylab) in zip(axes, panels):
        for m in ['LASSO', 'HCGL', 'GROUP-fixed']:
            d = df[df.method == m].sort_values('factor_corr')
            st = _STYLE[m]
            ax.errorbar(d.factor_corr, d[col], yerr=d[se], color=st['color'],
                        marker=st['marker'], ls=st['ls'], capsize=2, markersize=4.5,
                        label=st['label'])
        ax.set_xlabel(r'predictor correlation $\rho_X$')
        ax.set_ylabel(ylab)
        ax.set_xlim(-0.03, 0.83)
    axes[0].set_ylim(0.6, 1.02)
    axes[1].legend(loc='upper left', frameon=False)
    _save(fig, 'figS5_correlated')




def table_rhobar() -> str:
    """TABLE S3: design-effect attenuation of the pooling gain vs residual correlation.

    Reads rho_resid_attenuation.csv (sign_pooling_robustness.study_rho_resid).
    """
    df = pd.read_csv(os.path.join(_RES, 'rho_resid_attenuation.csv'))
    lines = [r'\begin{tabular}{ccccc}', r'\toprule',
             r'$\bar{\rho}$ & Recovery & Abstention & False-sign & Rate factor \\', r'\midrule']
    for _, r in df.iterrows():
        lines.append(f"{r.rho_bar:.1f} & {r.recovery:.3f} & {r.abstain:.3f} & "
                     f"{r.false_sign:.3f} & {r.rate_factor:.2f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    body = '\n'.join(lines)
    open(os.path.join(_FIG, 'tableS3_rhobar.tex'), 'w').write(body)
    return body


def table_coop() -> str:
    """TABLE S4: hard sign constraint vs cooperative LASSO vs predictor correlation.

    Reads coop_benchmark.csv (sign_pooling_robustness.study_cooperative).
    """
    df = pd.read_csv(os.path.join(_RES, 'coop_benchmark.csv'))
    H = df[df.method == 'hard-constrained'].set_index('factor_corr')
    C = df[df.method == 'cooperative'].set_index('factor_corr')
    lines = [r'\begin{tabular}{ccccccccc}', r'\toprule',
             r' & \multicolumn{4}{c}{Hard sign constraint} & \multicolumn{4}{c}{Cooperative LASSO} \\',
             r'\cmidrule(lr){2-5} \cmidrule(lr){6-9}',
             r'$\rho_X$ & Rec. & Flip & Abst. & $\beta$-MSE & Rec. & Flip & Abst. & $\beta$-MSE \\',
             r'\midrule']
    for fc in sorted(H.index):
        h, c = H.loc[fc], C.loc[fc]
        lines.append(f"{fc:.1f} & {h.recovery:.3f} & {h.flip:.3f} & {h.abstain:.3f} & {h.beta_mse:.4f} & "
                     f"{c.recovery:.3f} & {c.flip:.3f} & {c.abstain:.3f} & {c.beta_mse:.4f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    body = '\n'.join(lines)
    open(os.path.join(_FIG, 'tableS4_coop.tex'), 'w').write(body)
    return body


if __name__ == '__main__':
    os.makedirs(_FIG, exist_ok=True)
    fig_recoverability()
    fig_gate_roc()
    fig_consistency()
    fig_sign_matrices()
    fig_correlated()
    table_base()
    table_regime()
    table_rhobar()
    table_coop()
    print('written to', _FIG)
    for f in sorted(os.listdir(_FIG)):
        print('  ', f)
