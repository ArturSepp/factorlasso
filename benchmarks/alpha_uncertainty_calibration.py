"""Offline calibration/bias experiments with fixed evaluation seeds and explicit scope.

Run with --output pointing outside the checkout. The oracle Gaussian experiment
has at least 10,000 draws per cell. Bootstrap experiments are deliberately labelled
pilots: their output cannot authorize a nominal-coverage claim for fund data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import beta, norm

import factorlasso as fl
from factorlasso.utils._hac import bartlett_kernel


def weights_for(n, span, initialization):
    """Recover the exact production recursion from impulses, including its seed."""
    impulses = np.eye(n)
    if initialization == 'zero':
        impulses = np.vstack([np.full(n, np.nan), impulses])
    return fl.compute_ewm(impulses, span=span)[-1]


def coverage_row(covered, widths, family_size=1):
    """Report binomial uncertainty and a simultaneous lower coverage diagnostic."""
    count, n = int(np.count_nonzero(covered)), len(covered)
    return dict(coverage=count/n, draws=n, mean_width=float(np.mean(widths)),
                mc_se=float(np.sqrt(count/n*(1-count/n)/n)),
                lower_coverage_95=float(beta.ppf(.05/family_size, count, n-count+1))
                if count else 0.)


def known_shape_experiment(draws):
    """Measure normal and oracle Gaussian coverage separately from variance bias."""
    rng = np.random.default_rng(2026100407)
    cells = [(n, 60, 1) for n in [15, 16, 38, 50, 120, 320]]
    cells += [(n, 40, 3) for n in [16, 38, 50, 100]]
    rows = []
    for n, span, cadence in cells:
        calendar = cadence*np.arange(n)
        kernel = csr_matrix(bartlett_kernel(calendar, 6.))
        for initialization in ['zero', 'observation']:
            q = weights_for(n, span, initialization)
            geometry = fl.compute_weighted_mean_hac_geometry(q, calendar=calendar)
            for phi in [-.3, 0., .3, .6]:
                sigma = phi**np.abs(np.arange(n)[:, None]-np.arange(n))
                errors = rng.normal(size=(draws, n)) @ np.linalg.cholesky(sigma).T
                response = .03+errors
                estimate = response @ q
                target = .03*q.sum()
                scores = (response-estimate[:, None]/q.sum())*q*np.sqrt(n/(n-1))
                se = np.sqrt(np.maximum(np.sum(scores*kernel.dot(scores.T).T, axis=1), 0.))
                # Independent response evaluation confirms the public geometry.
                reference = geometry.statistics(response[:4])
                np.testing.assert_allclose(estimate[:4], reference[0], atol=1e-13)
                np.testing.assert_allclose(se[:4], reference[1], atol=1e-13)
                critical = fl.gaussian_critical_value(geometry, sigma)
                variance = float(q @ sigma @ q)
                expected = float(np.trace(geometry.quadratic @ sigma))
                for method, multiplier in [('normal', norm.ppf(.975)),
                                            ('known_shape_gaussian', critical)]:
                    row = dict(n=n, span=span, cadence=cadence, initialization=initialization,
                               phi=phi, method=method, critical_value=multiplier,
                               weight_mass=float(q.sum()), true_variance=variance,
                               expected_hac_variance=expected, variance_ratio=expected/variance)
                    row.update(coverage_row(np.abs(estimate-target) <= multiplier*se,
                                             2*multiplier*se, family_size=80))
                    rows.append(row)
        print(f'known-shape n={n} span={span} cadence={cadence}', flush=True)
    return pd.DataFrame(rows)


def bootstrap_pilot(outer, inner):
    """Stress the residual-bootstrap calibration without promoting it to validated CI."""
    rng = np.random.default_rng(2026100408)
    rows = []
    for n, span, cadence in [(16, 60, 1), (38, 60, 1), (120, 60, 1), (38, 40, 3)]:
        calendar = cadence*np.arange(n)
        q = weights_for(n, span, 'zero')
        for process in ['iid', 'ar1', 'negative_ar1', 'student5', 'heteroskedastic', 'ma1']:
            phi = .6 if process == 'ar1' else (-.3 if process == 'negative_ar1' else 0.)
            sigma = phi**np.abs(np.arange(n)[:, None]-np.arange(n))
            z = (rng.standard_t(5, size=(outer, n))/np.sqrt(5/3) if process == 'student5'
                 else rng.normal(size=(outer, n)))
            samples = z @ np.linalg.cholesky(sigma).T
            if process == 'heteroskedastic':
                samples *= np.linspace(.5, 2., n)
            if process == 'ma1':
                extended = rng.normal(size=(outer, n+1))
                samples = (extended[:, 1:]+.7*extended[:, :-1])/np.sqrt(1.49)
            covered, widths, normal_covered = [], [], []
            for i, sample in enumerate(samples):
                result = fl.bootstrap_weighted_means(sample[:, None], q[:, None],
                    calendar=calendar, draws=inner, seed=20261005+i)
                interval = fl.linear_confidence_intervals(result['estimate'], result['covariance'],
                    method='bootstrap_t', errors=result['errors'],
                    replicate_standard_errors=result['standard_errors'])
                lo, hi = interval['lower'][0], interval['upper'][0]
                covered.append(bool(np.isfinite(lo) and lo <= 0 <= hi))
                widths.append(hi-lo)
                normal_covered.append(abs(result['estimate'][0]) <=
                                      norm.ppf(.975)*np.sqrt(result['covariance'][0, 0]))
            row = dict(n=n, span=span, cadence=cadence, process=process,
                       method='bootstrap_t_pilot', inner_draws=inner,
                       normal_coverage=float(np.mean(normal_covered)),
                       status='pilot_not_a_validation_gate')
            row.update(coverage_row(covered, widths, family_size=24))
            rows.append(row)
            print(f'bootstrap pilot n={n} process={process}: {row["coverage"]:.3f}', flush=True)
    return pd.DataFrame(rows)


def joint_bias_experiment(draws):
    """Compare analytic joint expectations with independent panel simulation.

    The third response averages each native quarter of an underlying monthly
    process. Its covariance is therefore coherent with the monthly responses.
    """
    rng = np.random.default_rng(2026100409)
    n, count = 120, 3
    calendar = np.arange(n)
    temporal = .6**np.abs(calendar[:, None]-calendar)
    cross = np.full((count, count), .4)+np.eye(count)*.6
    maps = [np.eye(n), np.eye(n), np.zeros((n, n))]
    for t in range(2, n, 3):
        maps[2][t, t-2:t+1] = 1/3
    mask = np.ones((n, count), dtype=bool)
    mask[:30, 1] = False
    mask[:, 2] = calendar % 3 == 2
    mask[[57, 58], 1] = False
    q = np.zeros((n, count))
    for j, span in enumerate([60, 60, 40]):
        impulses = np.full((n, mask[:, j].sum()), np.nan)
        impulses[mask[:, j]] = np.eye(mask[:, j].sum())
        q[mask[:, j], j] = fl.compute_ewm(impulses, span=span)[-1]
    kernel = bartlett_kernel(calendar, 6.)
    correction = np.sqrt(mask.sum(axis=0)/(mask.sum(axis=0)-1))
    operators = [np.diag(q[:, j]) @ (np.eye(n)-np.outer(np.ones(n), q[:, j]/q[:, j].sum()))
                 * correction[j] for j in range(count)]
    truth, expected = np.zeros((count, count)), np.zeros((count, count))
    for i in range(count):
        for j in range(count):
            sigma = cross[i, j]*maps[i] @ temporal @ maps[j].T
            truth[i, j] = q[:, i] @ sigma @ q[:, j]
            expected[i, j] = np.sum(kernel*(operators[i] @ sigma @ operators[j].T))
    x = rng.normal(size=(draws, n, count)) @ np.linalg.cholesky(cross).T
    x = np.einsum('ts,bsj->btj', np.linalg.cholesky(temporal), x, optimize=True)
    x[:, :, 2] = x[:, :, 2] @ maps[2].T
    x[:, ~mask] = np.nan
    alpha = np.nansum(x*q, axis=1)
    scores = np.nan_to_num(x-alpha[:, None, :]/q.sum(axis=0))*q*correction
    empirical = np.empty_like(truth)
    for i in range(count):
        for j in range(count):
            empirical[i, j] = np.mean(np.sum(scores[:, :, i]*
                                      csr_matrix(kernel).dot(scores[:, :, j].T).T, axis=1))
    for b in range(3):
        frame = pd.DataFrame(x[b], columns=['monthly', 'ragged', 'quarterly'])
        fit = fl.estimate_alpha_uncertainty(frame, pd.Series([60, 60, 40], index=frame.columns),
                                            calendar=calendar)
        np.testing.assert_allclose(fit.covariance, scores[b].T @ kernel @ scores[b], atol=1e-13)
    rows = []
    for i in range(count):
        for j in range(count):
            rows.append(dict(i=i, j=j, true_covariance=truth[i, j],
                             expected_hac_covariance=expected[i, j],
                             simulated_hac_covariance=empirical[i, j],
                             simulated_estimate_covariance=np.cov(alpha.T)[i, j]))
    return pd.DataFrame(rows)


def quadratic_experiment(draws):
    """Assess known-V multivariate bounds and estimated-HAC rank-one bounds.

    The rank-one experiment re-estimates covariance for every response. It is
    deliberately not evidence of adequate coverage for a high-rank fund metric.
    """
    rng = np.random.default_rng(2026100410)
    rows = []
    covariance = np.array([[1., .6, .1], [.6, 2., .2], [.1, .2, .5]])
    metric = np.diag([2., .05, 0.])
    for signal in [0., .5, 2.]:
        mean = np.array([signal, -signal, 2*signal])
        sample = mean+rng.normal(size=(draws, 3)) @ np.linalg.cholesky(covariance).T
        observed = np.einsum('ij,ij->i', sample @ metric, sample)
        truth = float(mean @ metric @ mean)
        for method in ['spectral', 'weighted_chi2']:
            summary = fl.quadratic_confidence_summary(mean, covariance, metric, method=method)
            radius = summary['error_norm_radius']
            lower = np.maximum(0., np.sqrt(observed)-radius)**2
            upper = (np.sqrt(observed)+radius)**2
            row = dict(scope='known_covariance_rank2', method=method, signal=signal,
                       truth=truth, noise=float(np.trace(metric @ covariance)))
            row.update(coverage_row((lower <= truth) & (truth <= upper), upper-lower,
                                    family_size=6))
            rows.append(row)
    for n, span, cadence in [(16, 60, 1), (38, 60, 1), (120, 60, 1), (320, 60, 1),
                              (38, 40, 3), (100, 40, 3)]:
        calendar = cadence*np.arange(n)
        kernel = csr_matrix(bartlett_kernel(calendar, 6.))
        for initialization in ['zero', 'observation']:
            q = weights_for(n, span, initialization)
            for phi in [0., .6]:
                sigma = phi**np.abs(np.arange(n)[:, None]-np.arange(n))
                errors = rng.normal(size=(draws, n)) @ np.linalg.cholesky(sigma).T
                for signal in [0., .3]:
                    response = signal+errors
                    estimate = response @ q
                    scores = (response-estimate[:, None]/q.sum())*q*np.sqrt(n/(n-1))
                    variance = np.maximum(np.sum(scores*kernel.dot(scores.T).T, axis=1), 0.)
                    # Rank-one weighted chi-square equals the squared normal critical value.
                    radius = norm.ppf(.975)*np.sqrt(variance)
                    lower = np.maximum(0., np.abs(estimate)-radius)**2
                    upper = (np.abs(estimate)+radius)**2
                    truth = float((signal*q.sum())**2)
                    for i in range(3):
                        summary = fl.quadratic_confidence_summary(
                            [estimate[i]], [[variance[i]]], [[1.]])
                        np.testing.assert_allclose([lower[i], upper[i]],
                                                   [summary['lower'], summary['upper']])
                    row = dict(scope='estimated_hac_rank1', method='weighted_chi2', n=n,
                               span=span, cadence=cadence, initialization=initialization,
                               phi=phi, signal=signal, truth=truth,
                               mean_noise=float(variance.mean()),
                               true_noise=float(q @ sigma @ q),
                               mean_noise_adjusted=float(np.mean(estimate**2-variance)))
                    row.update(coverage_row((lower <= truth) & (truth <= upper), upper-lower,
                                            family_size=48))
                    rows.append(row)
        print(f'quadratic covariance assessment n={n} span={span}', flush=True)
    return pd.DataFrame(rows)


def main():
    """Write all experiment outputs and the precise executed-source manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--bootstrap-outer', type=int, default=200)
    parser.add_argument('--bootstrap-inner', type=int, default=399)
    parser.add_argument('--quadratic-only', action='store_true')
    args = parser.parse_args()
    if args.draws < 10000:
        parser.error('known-shape evaluation requires at least 10000 draws')
    args.output.mkdir(parents=True, exist_ok=False)
    if not args.quadratic_only:
        known_shape_experiment(args.draws).to_csv(
            args.output/'known_shape_coverage.csv', index=False)
        joint_bias_experiment(args.draws).to_csv(
            args.output/'joint_covariance_bias.csv', index=False)
        bootstrap_pilot(args.bootstrap_outer, args.bootstrap_inner).to_csv(
            args.output/'bootstrap_pilot.csv', index=False)
    quadratic_experiment(args.draws).to_csv(args.output/'quadratic_coverage.csv', index=False)
    sources = list((Path(fl.__file__).parent/'inference').glob('*.py'))
    sources += [Path(__file__), Path(fl.__file__).parent/'covariance'/'_alpha_uncertainty.py']
    sources += [Path(fl.__file__).parent/'utils'/name for name in ('_hac.py', '_ewm.py')]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    (args.output/'sources.json').write_text(json.dumps(hashes, indent=2))
    archive = args.output/'executed_sources'
    archive.mkdir()
    for source in sources:
        shutil.copy2(source, archive/source.name)


if __name__ == '__main__':
    main()
