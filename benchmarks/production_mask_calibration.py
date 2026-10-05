"""Conditional scalar calibration on representative actual recursive EWMA masks.

Input cases contain residuals.csv, weights.csv and metadata.csv exported from
frozen production fits. Reuse the inference API; do not fit or normalize weights.
Shortest regular support in each case/cadence is selected without seeing alpha.
This checks declared Gaussian AR models, not empirical or post-LASSO coverage.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, norm
import factorlasso as fl


def run(inputs, output, draws=10000, cells=101):
    """Evaluate off-grid AR values and an oracle-shape bias-correction candidate."""
    output.mkdir(parents=True, exist_ok=False)
    rows, selection, intervals = [], [], []
    rng = np.random.default_rng(2026100417)
    for folder in sorted(inputs.iterdir()):
        if not folder.is_dir() or not (folder/'weights.csv').exists():
            continue
        q = pd.read_csv(folder/'weights.csv', index_col=0, float_precision='round_trip')
        y = pd.read_csv(folder/'residuals.csv', index_col=0, float_precision='round_trip')
        meta = pd.read_csv(folder/'metadata.csv', index_col=0, float_precision='round_trip')
        t = pd.to_datetime(q.index).to_period('M').asi8
        selected = {}
        for asset in q:
            support = y[asset].notna().to_numpy()
            dt = np.diff(t[support])
            regular = len(dt) > 0 and (dt == dt[0]).all()
            selection.append(dict(case=folder.name, asset=asset, regular=regular,
                                  n=int(support.sum()), cadence=int(dt[0]) if regular else np.nan))
            if regular and (dt[0] not in selected or support.sum() < selected[dt[0]][0]):
                selected[dt[0]] = (int(support.sum()), asset, support)
        for cadence, (n, asset, support) in selected.items():
            weights = q.loc[support, asset].to_numpy()
            scale = float(meta.loc[asset, 'inference_scale'])
            geometry = fl.compute_weighted_mean_hac_geometry(
                weights, calendar=t[support], scale=scale)
            actual = y.loc[support, asset].to_numpy()
            calibrated = fl.compute_ar1_interval(geometry, actual, phi_max=.7, cells=cells)
            critical = calibrated.critical_value[0]
            intervals.append(dict(case=folder.name, asset=asset, n=n, cadence=cadence,
                lower=calibrated.lower[0], upper=calibrated.upper[0], critical=critical,
                cells=cells, calibration_alpha=calibrated.calibration_alpha,
                scope='shortest regular native support; pointwise bounded Gaussian AR1'))
            print(folder.name, asset, 'n', n, 'critical', critical, flush=True)
            for phi in [-.55, 0., .55, .7]:
                covariance = phi**abs(np.arange(n)[:, None]-np.arange(n))
                errors = rng.normal(size=(draws, n)) @ np.linalg.cholesky(covariance).T
                estimates, se = geometry.statistics(errors)
                moments = fl.weighted_mean_hac_expectation(weights[:, None],
                    observed=np.ones((n, 1), dtype=bool), calendar=t[support],
                    covariance_shape=covariance, scale=scale)
                ratio = moments['expected_hac_covariance'][0, 0]/moments['true_covariance'][0, 0]
                for method, multiplier in [('normal_hac', norm.ppf(.975)),
                    ('oracle_shape_variance_corrected_normal', norm.ppf(.975)/np.sqrt(ratio)),
                    ('bounded_ar1', critical)]:
                    covered = abs(estimates) <= multiplier*se
                    count = int(covered.sum())
                    rows.append(dict(case=folder.name, asset=asset, n=n, cadence=cadence,
                        phi_native=phi, method=method, draws=draws, coverage=covered.mean(),
                        mc_lower=float(beta.ppf(.025, count, draws-count+1)) if count else 0.,
                        mc_upper=(float(beta.ppf(.975, count+1, draws-count))
                                  if count < draws else 1.), expected_to_true_variance=ratio,
                        empirical_variance_ratio=float(
                            np.mean(se**2)/moments['true_covariance'][0, 0])))
            pd.DataFrame(rows).to_csv(output/'coverage.csv', index=False)
            pd.DataFrame(intervals).to_csv(output/'representative_intervals.csv', index=False)
    pd.DataFrame(selection).to_csv(output/'support_inventory.csv', index=False)
    paths = [Path(__file__)]+list((Path(fl.__file__).parent/'inference').glob('*.py'))
    (output/'MANIFEST.json').write_text(json.dumps(dict(draws=draws, cells=cells, seed=2026100417,
        scope='regular masks; pointwise Gaussian constant-mean native AR1, fixed model',
        empirical_coverage_validated=False,
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--cells', type=int, default=101)
    options = parser.parse_args()
    run(**vars(options))
