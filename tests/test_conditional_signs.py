"""Independent normal-law and least-squares references for conditional gates."""
from dataclasses import fields

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr

from factorlasso.inference._conditional_signs import (
    compute_conditional_gaussian_geometry, gaussian_sign_gate_risk,
)


def _density(z):
    return np.exp(-z*z/2)/np.sqrt(2*np.pi)


def _reverse_reference(beta, gate_mean, fit_sd, gate_sd, rho, threshold, action):
    """Condition on the raw individual fit, the opposite producer conditioning."""
    h = threshold*gate_sd
    noise = gate_sd*np.sqrt(max(1-rho*rho, 0.))
    cuts = [-np.inf, -8., -4., -beta/fit_sd, 0., 4., 8., np.inf]
    if rho:
        cuts += [(h-gate_mean)/(rho*gate_sd), (-h-gate_mean)/(rho*gate_sd)]
    cuts = sorted(set(cuts))

    def integrand(v, component):
        z = beta+fit_sd*v
        gate_conditional_mean = gate_mean+rho*gate_sd*v
        if noise:
            positive = ndtr((gate_conditional_mean-h)/noise)
            negative = ndtr((-h-gate_conditional_mean)/noise)
        else:
            positive = float(gate_conditional_mean > 0 and gate_conditional_mean >= h)
            negative = float(gate_conditional_mean < 0 and gate_conditional_mean <= -h)
        weak = max(1-positive-negative, 0.)
        estimates = np.array([max(z, 0.), min(z, 0.), z if action == 'release' else 0.])
        probs = np.array([positive, negative, weak])
        values = [probs @ estimates, probs @ (estimates**2), probs @ ((estimates-beta)**2),
                  positive*float(z < 0)+negative*float(z > 0)]
        return values[component]*_density(v)

    return np.array([sum(quad(lambda v: integrand(v, j), a, b,
                             epsabs=2e-11, epsrel=2e-11)[0]
                         for a, b in zip(cuts[:-1], cuts[1:])) for j in range(4)])


@pytest.mark.parametrize('rho', [-1., -.6, 0., .7, 1.])
@pytest.mark.parametrize('beta,gate_mean', [(0., 0.), (.4, .8), (-1., .7)])
@pytest.mark.parametrize('action', ['exclude', 'release'])
def test_risk_matches_reverse_conditioning(beta, gate_mean, rho, action):
    result = gaussian_sign_gate_risk(beta, gate_mean, .7, .3, rho,
                                    threshold=1.3, action=action)
    reference = _reverse_reference(beta, gate_mean, .7, .3, rho, 1.3, action)
    np.testing.assert_allclose([result.estimate_mean, result.estimate_second_moment,
                               result.mse, result.probability_clipped],
                              reference, atol=2e-9, rtol=2e-9)
    assert result.mse == pytest.approx(result.positive_mse+result.negative_mse+result.weak_mse)
    assert result.probability_positive+result.probability_negative+result.probability_weak == (
        pytest.approx(1., abs=2e-15)
    )
    assert result.bias == pytest.approx(result.estimate_mean-beta, abs=2e-12)
    assert result.variance+result.bias**2 == pytest.approx(result.mse, abs=2e-12)
    assert result.integration_error < 1e-8


def test_exact_clipping_and_hard_threshold_endpoints():
    hard_mse = 2*(ndtr(-1.)+_density(1.))
    same = gaussian_sign_gate_risk(0., 0., 1., 1., 1.)
    assert same.mse == pytest.approx(hard_mse, abs=2e-12)
    assert same.probability_clipped == 0
    reverse = gaussian_sign_gate_risk(0., 0., 1., 1., -1.)
    assert reverse.mse == 0
    assert reverse.probability_clipped == pytest.approx(2*ndtr(-1.), abs=2e-12)
    reverse_release = gaussian_sign_gate_risk(0., 0., 1., 1., -1., action='release')
    assert reverse_release.mse == pytest.approx(1-hard_mse, abs=2e-12)
    independent = gaussian_sign_gate_risk(2., 0., 1., 1., 0., threshold=0.)
    assert independent.estimate_mean == pytest.approx(1., abs=2e-12)
    assert independent.estimate_second_moment == pytest.approx(2.5, abs=2e-12)
    assert independent.mse == pytest.approx(2.5, abs=2e-12)
    assert independent.probability_clipped == pytest.approx(.5, abs=2e-12)


@pytest.mark.parametrize('rho', [-1., -.3, 0., .8, 1.])
def test_exclusion_release_gap_retains_gate_fit_dependence(rho):
    beta, gamma, sz, sg, threshold = .6, -.2, .7, .3, 1.2
    excluded = gaussian_sign_gate_risk(beta, gamma, sz, sg, rho, threshold=threshold)
    released = gaussian_sign_gate_risk(beta, gamma, sz, sg, rho,
                                      threshold=threshold, action='release')
    a, b = -threshold-gamma/sg, threshold-gamma/sg
    p = ndtr(b)-ndtr(a)
    exact = beta*beta*p-sz*sz*(p+rho*rho*(a*_density(a)-b*_density(b)))
    assert excluded.mse-released.mse == pytest.approx(exact, abs=3e-12)
    assert excluded.probability_false_exclusion == pytest.approx(p)
    assert released.probability_false_exclusion == 0
    assert excluded.probability_wrong_sign == pytest.approx(ndtr(a))


@pytest.mark.parametrize('action', ['exclude', 'release'])
def test_deterministic_gates_and_fits(action):
    right = gaussian_sign_gate_risk(2., 1., 0., 0., .8, action=action)
    wrong = gaussian_sign_gate_risk(2., -1., 0., 0., -.8, action=action)
    null = gaussian_sign_gate_risk(2., 0., 0., 0., .8, threshold=0., action=action)
    assert right.estimate_mean == 2 and right.mse == 0
    assert wrong.estimate_mean == 0 and wrong.mse == 4 and wrong.probability_clipped == 1
    assert null.probability_weak == 1
    assert null.mse == (4 if action == 'exclude' else 0)
    infinite = gaussian_sign_gate_risk(2., 9., 1., 0., 1., threshold=np.inf, action=action)
    assert infinite.mse == (4 if action == 'exclude' else 1)
    stochastic_gate = gaussian_sign_gate_risk(2., .3, 0., 1., -.7, action=action)
    expected = 4*(stochastic_gate.probability_negative
                  + (stochastic_gate.probability_weak if action == 'exclude' else 0))
    assert stochastic_gate.mse == pytest.approx(expected, abs=2e-12)


def test_tail_symmetry_and_wide_gate_limits():
    base = gaussian_sign_gate_risk(.4, .8, .7, .3, .6)
    flipped = gaussian_sign_gate_risk(-.4, -.8, .7, .3, .6)
    assert flipped.mse == pytest.approx(base.mse, abs=2e-12)
    assert flipped.estimate_mean == pytest.approx(-base.estimate_mean, abs=2e-12)
    assert flipped.probability_positive == base.probability_negative
    unbounded_gate = gaussian_sign_gate_risk(0., 1000., 1., 1., .4)
    assert unbounded_gate.estimate_mean == pytest.approx(_density(0.), abs=2e-12)
    assert unbounded_gate.mse == pytest.approx(.5, abs=2e-12)
    strong = gaussian_sign_gate_risk(1e4, 1e4, 1., 1., .9)
    assert strong.mse == pytest.approx(1., abs=2e-10)


def _geometry_fixture():
    rng = np.random.default_rng(68043)
    x = rng.normal(size=(24, 3)) @ np.array([[1., .8, .3], [0, .5, .1], [0, 0, .7]])
    beta = np.array([[1., -.5, .2], [.7, -.2, 0.], [0., .3, -.8]])
    omega = np.array([[1., .3, .1], [.3, .8, -.1], [.1, -.1, .5]])
    pooling = np.array([[.5, .5, 0.], [0., .25, .75]])
    return x, beta, omega, pooling


@pytest.mark.parametrize('intercept', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
def test_geometry_matches_independent_normal_equations(intercept, weighted):
    x, beta, omega, a = _geometry_fixture()
    w = np.exp(np.linspace(-2., 0., len(x))) if weighted else np.ones(len(x))
    gamma = .4**abs(np.subtract.outer(np.arange(len(x)), np.arange(len(x))))
    result = compute_conditional_gaussian_geometry(
        x, beta, omega, a, weights=w, fit_intercept=intercept, date_covariance=gamma,
    )
    design = np.column_stack([np.ones(len(x)), x]) if intercept else x
    inverse = np.linalg.inv(design.T @ (w[:, None]*design))
    linear = inverse @ (design.T*w)
    linear = linear[1:] if intercept else linear
    v = linear @ gamma @ linear.T
    np.testing.assert_allclose(result.coefficient_map, linear, atol=2e-14)
    np.testing.assert_allclose(result.factor_covariance, v, atol=2e-14)
    np.testing.assert_allclose(result.pooled_means, a @ beta, atol=2e-15)
    for i in range(len(beta)):
        for g in range(len(a)):
            for j in range(beta.shape[1]):
                fit = omega[i, i]*v[j, j]
                gate = (a[g] @ omega @ a[g])*v[j, j]
                cross = (omega[i] @ a[g])*v[j, j]
                assert result.fit_sd[i, j]**2 == pytest.approx(fit)
                assert result.gate_sd[g, j]**2 == pytest.approx(gate)
                assert result.fit_gate_covariance[i, g, j] == pytest.approx(cross)
                assert result.fit_gate_correlation[i, g, j] == pytest.approx(
                    cross/np.sqrt(fit*gate))
    scaled = compute_conditional_gaussian_geometry(
        x, beta, omega, a, weights=137*w, fit_intercept=intercept, date_covariance=gamma,
    )
    np.testing.assert_allclose(scaled.coefficient_map, result.coefficient_map, atol=2e-14)
    for field in fields(result):
        assert not getattr(result, field.name).flags.writeable
    omega[0, 0] = 99
    assert result.residual_covariance[0, 0] == 1


def test_pooling_precision_and_duplicate_invariance():
    x, beta, _, _ = _geometry_fixture()
    m, rho = 4, .25
    omega = (1-rho)*np.eye(m)+rho*np.ones((m, m))
    a = np.ones((1, m))/m
    b = np.tile(beta[0], (m, 1))
    result = compute_conditional_gaussian_geometry(x, b, omega, a)
    np.testing.assert_allclose((result.gate_sd/result.fit_sd[0])**2,
                               rho+(1-rho)/m, atol=2e-14)
    duplicate = compute_conditional_gaussian_geometry(x, b, np.ones((m, m)), a)
    np.testing.assert_allclose(duplicate.gate_sd, duplicate.fit_sd[:1], atol=2e-14)
    deterministic = compute_conditional_gaussian_geometry(x, b, np.zeros((m, m)), a)
    assert not deterministic.fit_sd.any() and not deterministic.fit_gate_correlation.any()


@pytest.mark.parametrize('alteration', ['singular_x', 'negative_weight', 'zero_weight',
                                       'bad_omega', 'bad_gamma', 'bad_pooling'])
def test_geometry_rejects_invalid_model(alteration):
    x, beta, omega, a = _geometry_fixture()
    w, gamma = np.ones(len(x)), np.eye(len(x))
    if alteration == 'singular_x':
        x[:, 2] = x[:, 1]
    elif alteration == 'negative_weight':
        w[0] = -1
    elif alteration == 'zero_weight':
        w[:] = 0
    elif alteration == 'bad_omega':
        omega[0, 0] = -1
    elif alteration == 'bad_gamma':
        gamma[0, 0] = -1
    else:
        a[0, 0] = -.5
    with pytest.raises(ValueError):
        compute_conditional_gaussian_geometry(x, beta, omega, a, weights=w, date_covariance=gamma)


@pytest.mark.parametrize('kwargs', [dict(fit_sd=-1), dict(gate_sd=-1), dict(correlation=1.01),
                                    dict(threshold=-1), dict(threshold=np.nan), dict(beta=np.inf),
                                    dict(action='unknown'), dict(beta='bad'), dict(threshold=True)])
def test_risk_rejects_invalid_parameters(kwargs):
    arguments = dict(beta=0., gate_mean=0., fit_sd=1., gate_sd=1., correlation=0.)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        gaussian_sign_gate_risk(**arguments)


@pytest.mark.parametrize('beta,fit_sd', [
    (0., 1e200), (1e200, 0.), (1e154, 1e154),
    (np.finfo(float).max, np.finfo(float).max),
])
def test_unrepresentable_raw_second_moment_raises_without_overflow(beta, fit_sd):
    # Infinite-threshold exclusion can otherwise hide an infinite baseline MSE.
    with np.errstate(over='raise', invalid='raise'):
        with pytest.raises(ValueError, match='raw second moment'):
            gaussian_sign_gate_risk(beta, 0., fit_sd, 1., 0., threshold=np.inf)
