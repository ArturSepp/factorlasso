"""Hierarchical Gaussian means with correlated measurement errors.

The model is y | a ~ N(a, V), a | mu, tau ~ N(Z mu, diag(tau[group]**2)),
mu ~ N(m0, diag(s_mu**2)), and independent tau_g ~ HalfNormal(s_tau,g).
Integrate mu analytically and tau by tensor Gauss-Legendre quadrature in prior
probability coordinates. Gaussian conditional simulation then includes uncertainty
in a, mu and tau. No estimated hyperparameter is treated as known.

References
----------
Gelman and Pardoe (2006), Bayesian Measures of Explained Variance and Pooling in
Multilevel (Hierarchical) Models, Technometrics 48, 241-251,
doi:10.1198/004017005000000517, describes partial pooling and warns that dispersion
of posterior means understates latent dispersion. Correlated known-error inputs,
the half-normal prior and quadrature/conditional simulation are implementation
choices here. Plug-in error covariance does not inherit frequentist coverage.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import halfnorm, lognorm, norm

from factorlasso.inference._validation import _confidence, _psd_root


@dataclass(frozen=True)
class HierarchicalMeanPosterior:
    """Joint hierarchical posterior approximated by quadrature and conditional draws.

    Attributes
    ----------
    estimates : pandas.DataFrame
        Input mean/SE, posterior mean/SD and equal-tail credible endpoints per asset.
    covariance : pandas.DataFrame
        Sample posterior covariance of latent means, including hyperparameter uncertainty.
    draws : pandas.DataFrame
        Joint latent-mean draws; rows must remain coupled for nonlinear functionals.
    group_mean_draws, group_scale_draws : pandas.DataFrame
        Coupled draws of group centres and between-asset standard deviations.
    groups : pandas.DataFrame
        Posterior group summaries, with deterministic quadrature mean of group scales.
    quadrature : pandas.DataFrame
        Group-scale nodes and their normalized posterior probabilities.
    diagnostics : dict
        Integration settings, evidence, numerical qualifications and prior specifications.
    """

    estimates: pd.DataFrame
    covariance: pd.DataFrame
    draws: pd.DataFrame
    group_mean_draws: pd.DataFrame
    group_scale_draws: pd.DataFrame
    groups: pd.DataFrame
    quadrature: pd.DataFrame
    diagnostics: dict


def _group_parameter(value, names, label, positive=False):
    """Require explicit scalar or exactly aligned group-level prior parameters."""
    if isinstance(value, pd.Series):
        if not value.index.equals(names):
            raise ValueError(f'{label} must have the exact ordered group labels')
        result = value.to_numpy(dtype=float)
    else:
        result = np.asarray(value, dtype=float)
        if result.ndim == 0:
            result = np.full(len(names), result)
    if (result.shape != (len(names),) or not np.isfinite(result).all()
            or (positive and np.any(result <= 0))):
        raise ValueError(f'{label} must be finite and match groups; scales must be positive')
    return result


def _pooling_inputs(mean, covariance, groups):
    """Validate finite labelled estimates, their full PSD error covariance and groups."""
    if not isinstance(mean, pd.Series) or mean.empty or mean.index.has_duplicates:
        raise ValueError('mean must be a nonempty Series with unique asset labels')
    if not np.isfinite(mean.to_numpy(dtype=float)).all():
        raise ValueError('mean must be finite')
    if (not isinstance(covariance, pd.DataFrame) or not covariance.index.equals(mean.index)
            or not covariance.columns.equals(mean.index)):
        raise ValueError('covariance must have the exact ordered mean axes')
    root, _, _ = _psd_root(covariance, len(mean))
    if (not isinstance(groups, pd.Series) or not groups.index.equals(mean.index)
            or groups.isna().any()):
        raise ValueError('groups must have complete labels on the exact ordered mean index')
    codes, names = pd.factorize(groups, sort=False)
    names = pd.Index(names, name='group')
    design = np.eye(len(names))[codes]
    return root, codes, names, design


def _marginal_factor(base, variances, delta):
    """Evaluate Gaussian marginal evidence with group centres integrated out."""
    matrix = base.copy()
    matrix.flat[::len(matrix)+1] += variances
    factor = cho_factor(matrix, lower=True, check_finite=False)
    solved = cho_solve(factor, delta, check_finite=False)
    loglik = -.5*(len(delta)*np.log(2*np.pi)
                  + 2*np.log(np.diag(factor[0])).sum() + delta @ solved)
    return factor, float(loglik)


def _conditional_simulation(y, noise_root, design, codes, location, mean_scales,
                            tau, factor, count, rng):
    """Sample the joint conditional law by conditioning independent prior simulations."""
    centres = location + rng.normal(size=(count, len(location)))*mean_scales
    latent = centres @ design.T + rng.normal(size=(count, len(y)))*tau[codes]
    pseudo_y = latent + rng.normal(size=(count, len(y))) @ noise_root.T
    correction = cho_solve(factor, (y-pseudo_y).T, check_finite=False).T
    centre_shift = (correction @ design)*mean_scales**2
    return (latent + correction*tau[codes]**2 + centre_shift @ design.T,
            centres+centre_shift)


def _evaluate_grid(axes, log_axis_weights, codes, base, delta):
    """Evaluate tensor quadrature, keeping prior/proposal density corrections explicit."""
    indices = np.indices(tuple(len(a) for a in axes)).reshape(len(axes), -1).T
    grid = np.column_stack([axis[indices[:, g]] for g, axis in enumerate(axes)])
    logs = sum(w[indices[:, g]] for g, w in enumerate(log_axis_weights))
    for i, tau in enumerate(grid):
        _, loglik = _marginal_factor(base, tau[codes]**2, delta)
        logs[i] += loglik
    evidence = logsumexp(logs)
    weights = np.exp(logs-evidence)
    weights /= weights.sum()
    return grid, weights, float(evidence)


def _adaptive_axes(probabilities, prior_scales, pilot, weights):
    """Concentrate numerical nodes using a full-support prior/lognormal mixture proposal."""
    centre = weights @ np.log(pilot)
    spread = np.maximum(.35, 1.5*np.sqrt(weights @ (np.log(pilot)-centre)**2))
    axes, corrections = [], []
    for scale, mu, sigma in zip(prior_scales, centre, spread):
        proposal_scale = np.exp(mu)

        def cdf(value):
            """Retain prior tails with 20 percent mass in the original half-normal."""
            return (.2*halfnorm.cdf(value, scale=scale)
                    + .8*lognorm.cdf(value, s=sigma, scale=proposal_scale))

        values = []
        for p in probabilities:
            upper = max(halfnorm.ppf(p, scale=scale),
                        lognorm.ppf(p, s=sigma, scale=proposal_scale))
            values.append(brentq(lambda x: cdf(x)-p, 0., upper,
                                  xtol=max(scale, proposal_scale)*1e-13))
        axis = np.asarray(values)
        log_prior = halfnorm.logpdf(axis, scale=scale)
        log_proposal = np.logaddexp(np.log(.2)+log_prior,
            np.log(.8)+lognorm.logpdf(axis, s=sigma, scale=proposal_scale))
        axes.append(axis)
        corrections.append(log_prior-log_proposal)
    return axes, corrections


def _grid_quantiles(grid, weights, probabilities):
    """Interpolate marginal quadrature CDF midpoints to reduce node-quantized endpoints."""
    result = []
    for column in grid.T:
        values, indices = np.unique(column, return_inverse=True)
        mass = np.bincount(indices, weights=weights)
        midpoint_cdf = np.cumsum(mass)-mass/2
        result.append(np.interp(probabilities, midpoint_cdf, values))
    return np.asarray(result).T


def pool_gaussian_means(mean, covariance, groups, *, mean_prior_scale,
                        dispersion_prior_scale, mean_prior_location=0.,
                        quadrature_order=24, draws=8192, confidence=.95, seed=0,
                        max_nodes=250000, adapt_quadrature=True):
    """Partially pool correlated mean estimates while integrating hyperparameters.

    Parameters
    ----------
    mean : pandas.Series
        Finite labelled estimates in caller-declared units, without implicit annualization.
    covariance : pandas.DataFrame
        Full PSD sampling-error covariance on exactly the same ordered axes.
    groups : pandas.Series
        Complete group assignments on the mean index. Group order is first appearance.
    mean_prior_scale, dispersion_prior_scale : float or pandas.Series
        Required positive normal-centre and half-normal-dispersion prior scales, in mean
        units. A Series must follow group order exactly. Scales are never fitted silently.
    mean_prior_location : float or pandas.Series, default 0
        Normal prior centres for group means, not imposed common asset means.
    quadrature_order : int, default 24
        Gauss-Legendre points per group on the half-normal prior CDF. Repeat at higher
        order to check integration accuracy; no automatic convergence claim is made.
    draws : int, default 8192
        At least two joint posterior draws. Quantiles and covariance have simulation error.
    confidence : float, default .95
        Equal-tail posterior credible mass. This is not frequentist confidence coverage.
    seed : int, default 0
        Nonnegative reproducible NumPy generator seed.
    max_nodes : int, default 250000
        Resource guard for the tensor grid, whose size is order**number_of_groups.
    adapt_quadrature : bool, default True
        Use an initial grid of at most 12 points per group to concentrate integration
        nodes. A full-support prior/lognormal mixture and explicit density correction
        change the quadrature measure only, never the statistical prior.

    Returns
    -------
    HierarchicalMeanPosterior
        Conditional-on-input-V posterior summaries, coupled draws, grid and diagnostics.

    Notes
    -----
    Cross-group error correlations are retained. Normal random effects are conditionally
    independent given group parameters, which is a separate modelling assumption.
    A continuous half-normal prior has no atom at zero dispersion: a positive lower
    credible endpoint is not a test rejecting identical group means. Estimated V,
    selected factor fits and empirical calibration are outside the posterior model.
    Tensor quadrature targets a small number of groups; it is not an MCMC algorithm.
    """
    confidence = _confidence(confidence)
    if not isinstance(adapt_quadrature, (bool, np.bool_)):
        raise ValueError('adapt_quadrature must be boolean')
    for label, value, minimum in [('quadrature_order', quadrature_order, 2),
                                  ('draws', draws, 2), ('max_nodes', max_nodes, 1),
                                  ('seed', seed, 0)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f'{label} must be an integer')
        if value < minimum:
            raise ValueError(f'{label} must be at least {minimum}')
    root, codes, names, design = _pooling_inputs(mean, covariance, groups)
    if quadrature_order**len(names) > max_nodes:
        raise ValueError('tensor quadrature exceeds max_nodes; reduce groups or order')
    mean_scales = _group_parameter(mean_prior_scale, names, 'mean_prior_scale', True)
    tau_scales = _group_parameter(dispersion_prior_scale, names, 'dispersion_prior_scale', True)
    location = _group_parameter(mean_prior_location, names, 'mean_prior_location')
    y = mean.to_numpy(dtype=float)
    # Reconstruct only to remove the already validated negative eigensolver roundoff.
    noise = root @ root.T
    base = noise + (design*mean_scales**2) @ design.T
    delta = y-design @ location
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    probabilities, prior_weights = (nodes+1)/2, weights/2
    axes = [norm.ppf((1+probabilities)/2)*s for s in tau_scales]
    axis_weights = [np.log(prior_weights)]*len(names)
    if adapt_quadrature:
        pilot_nodes, pilot_weights = np.polynomial.legendre.leggauss(min(quadrature_order, 12))
        pilot_axes = [norm.ppf((3+pilot_nodes)/4)*s for s in tau_scales]
        pilot, pilot_mass, _ = _evaluate_grid(pilot_axes,
            [np.log(pilot_weights/2)]*len(names), codes, base, delta)
        axes, corrections = _adaptive_axes(probabilities, tau_scales, pilot, pilot_mass)
        axis_weights = [np.log(prior_weights)+c for c in corrections]
    tau_grid, posterior_weights, log_evidence = _evaluate_grid(
        axes, axis_weights, codes, base, delta)
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(tau_grid), size=draws, p=posterior_weights)
    samples = np.empty((draws, len(y)))
    centres = np.empty((draws, len(names)))
    for node in np.unique(selected):
        rows = np.flatnonzero(selected == node)
        tau = tau_grid[node]
        factor, _ = _marginal_factor(base, tau[codes]**2, delta)
        samples[rows], centres[rows] = _conditional_simulation(
            y, root, design, codes, location, mean_scales, tau, factor, len(rows), rng)
    scales = tau_grid[selected]
    tail = (1-confidence)/2
    lower, upper = np.quantile(samples, [tail, 1-tail], axis=0)
    estimates = pd.DataFrame(dict(group=groups, raw_mean=mean,
        raw_standard_error=np.sqrt(np.diag(noise)), posterior_mean=samples.mean(axis=0),
        posterior_sd=samples.std(axis=0, ddof=1), lower=lower, upper=upper), index=mean.index)
    mu_lower, mu_upper = np.quantile(centres, [tail, 1-tail], axis=0)
    tau_lower, tau_median, tau_upper = _grid_quantiles(
        tau_grid, posterior_weights, [tail, .5, 1-tail])
    summary = pd.DataFrame(dict(N=np.bincount(codes), mean=centres.mean(axis=0),
        mean_sd=centres.std(axis=0, ddof=1), mean_lower=mu_lower, mean_upper=mu_upper,
        dispersion_mean=posterior_weights @ tau_grid, dispersion_median=tau_median,
        dispersion_lower=tau_lower, dispersion_upper=tau_upper), index=names)
    grid = pd.DataFrame(tau_grid, columns=names)
    grid.columns = ['scale_'+str(name) for name in names]
    grid['posterior_weight'] = posterior_weights
    diagnostics = dict(method='Gaussian hierarchy; tensor prior-CDF quadrature',
        confidence=confidence, quadrature_order=int(quadrature_order), nodes=len(tau_grid),
        draws=int(draws), seed=int(seed), log_evidence=float(log_evidence),
        effective_grid_nodes=float(1/(posterior_weights @ posterior_weights)),
        maximum_grid_weight=float(posterior_weights.max()),
        adapt_quadrature=bool(adapt_quadrature),
        mean_prior_location=location.tolist(), mean_prior_scale=mean_scales.tolist(),
        dispersion_prior_scale=tau_scales.tolist(),
        input_covariance='full correlated, fixed or plug-in; not inferred here',
        numerical_convergence='unchecked; compare higher quadrature order and independent draws',
        intervals='posterior credible intervals, not frequentist confidence intervals',
        zero_dispersion_atom=False)
    return HierarchicalMeanPosterior(estimates,
        pd.DataFrame(np.atleast_2d(np.cov(samples, rowvar=False)),
                     index=mean.index, columns=mean.index),
        pd.DataFrame(samples, columns=mean.index), pd.DataFrame(centres, columns=names),
        pd.DataFrame(scales, columns=names), summary, grid, diagnostics)
