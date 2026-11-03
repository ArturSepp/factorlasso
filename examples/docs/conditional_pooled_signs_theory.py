"""Fixed-group Gaussian signs and clipped risk, checked independently.

No fitted estimator defaults are changed. All panels and distribution parameters are deterministic.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import norm

from factorlasso import compute_conditional_gaussian_geometry, gaussian_sign_gate_risk


def reverse_conditioning_risk(beta, gate_mean, fit_sd, gate_sd, correlation, threshold,
                              action):
    """Integrate over the raw fit and evaluate conditional gate probabilities."""
    conditional_gate_sd = gate_sd*np.sqrt(1-correlation**2)
    h = threshold*gate_sd

    def integrand(u):
        z = beta+fit_sd*u
        gate_location = gate_mean+correlation*gate_sd*u
        positive = norm.sf((h-gate_location)/conditional_gate_sd)
        negative = norm.cdf((-h-gate_location)/conditional_gate_sd)
        weak = 1-positive-negative
        weak_estimate = z if action == 'release' else 0.
        loss = (positive*(max(z, 0)-beta)**2
                + negative*(min(z, 0)-beta)**2
                + weak*(weak_estimate-beta)**2)
        return norm.pdf(u)*loss

    split = -beta/fit_sd
    return (quad(integrand, -np.inf, split, epsabs=2e-11, epsrel=2e-11)[0]
            + quad(integrand, split, np.inf, epsabs=2e-11, epsrel=2e-11)[0])


def main():
    """Verify the geometry, exact scalar identities and reverse-conditioning risk."""
    t = np.arange(60)
    orthogonal = np.column_stack([np.cos(2*np.pi*t/60), np.sin(2*np.pi*t/60)])
    orthogonal *= np.sqrt(2)
    x = orthogonal @ np.array([[1., .9], [0., np.sqrt(1-.9**2)]])
    beta = np.tile([1., -.5], (4, 1))
    omega = .75*np.eye(4)+.25*np.ones((4, 4))
    pooling = np.full((1, 4), .25)
    geometry = compute_conditional_gaussian_geometry(x, beta, omega, pooling)

    # Independent least-squares coefficient operator, including the intercept.
    design = np.column_stack([np.ones(len(x)), x])
    operator = np.linalg.lstsq(design, np.eye(len(x)), rcond=None)[0][1:]
    np.testing.assert_allclose(geometry.coefficient_map, operator, atol=2e-15)
    np.testing.assert_allclose(geometry.factor_covariance, operator@operator.T, atol=2e-15)
    np.testing.assert_allclose(geometry.pooled_means, [[1., -.5]], atol=2e-15)
    marginal = (x.T@(x@beta[0]))/np.sum(x*x, axis=0)
    np.testing.assert_allclose(marginal, [.55, .4], atol=2e-15)
    variance_ratio = (geometry.gate_sd[0]/geometry.fit_sd[0])**2
    np.testing.assert_allclose(variance_ratio, .25+.75/4, atol=2e-15)

    # Correlated least squares projects in the Gram metric, not coordinatewise.
    gram, raw = np.array([[1., .9], [.9, 1.]]), np.array([-1., 1.])
    projection = minimize(lambda b: (b-raw)@gram@(b-raw), np.maximum(raw, 0),
                          jac=lambda b: 2*gram@(b-raw), bounds=[(0., None)]*2)
    assert projection.success
    np.testing.assert_allclose(projection.x, [0., .1], atol=1e-9)
    assert not np.allclose(projection.x, np.maximum(raw, 0))
    # The same realized nonnegative cone satisfies the Gram-metric bound.
    def gram_norm(value):
        return np.sqrt(value@gram@value)

    for truth in (np.array([1., .5]), np.array([-.5, .3])):
        truth_projection = minimize(lambda b: (b-truth)@gram@(b-truth),
                                    np.maximum(truth, 0),
                                    jac=lambda b: 2*gram@(b-truth),
                                    bounds=[(0., None)]*2)
        assert truth_projection.success
        distance = gram_norm(truth_projection.x-truth)
        assert gram_norm(projection.x-truth) <= gram_norm(raw-truth)+distance+1e-10
        if np.all(truth >= 0):
            np.testing.assert_allclose(distance, 0., atol=1e-10)
            assert gram_norm(projection.x-truth) <= gram_norm(raw-truth)+1e-10

    # When the fit and gate coincide, exclusion is ordinary hard thresholding.
    exclusion = gaussian_sign_gate_risk(0., 0., 1., 1., 1., threshold=1., action='exclude')
    release = gaussian_sign_gate_risk(0., 0., 1., 1., 1., threshold=1., action='release')
    exact_mse = 2*(norm.sf(1)+norm.pdf(1))
    np.testing.assert_allclose(exclusion.mse, exact_mse, atol=2e-12)
    np.testing.assert_allclose(release.mse, 1., atol=2e-12)
    np.testing.assert_allclose(exclusion.probability_false_admission, 2*norm.sf(1), atol=2e-12)

    # An independent, uninformative sign can increase risk even without exclusion.
    clipping = gaussian_sign_gate_risk(2., 0., 1., 1., 0., threshold=0., action='exclude')
    np.testing.assert_allclose(clipping.estimate_mean, 1., atol=2e-12)
    np.testing.assert_allclose(clipping.estimate_second_moment, 2.5, atol=2e-12)
    np.testing.assert_allclose(clipping.mse, 2.5, atol=2e-12)
    np.testing.assert_allclose(clipping.probability_clipped, .5, atol=2e-12)

    params = dict(beta=.4, gate_mean=.8, fit_sd=.7, gate_sd=.3, correlation=.6, threshold=1.)
    risks = {}
    for action in ('exclude', 'release'):
        risk = gaussian_sign_gate_risk(**params, action=action)
        reference = reverse_conditioning_risk(**params, action=action)
        np.testing.assert_allclose(risk.mse, reference, rtol=2e-9, atol=2e-11)
        np.testing.assert_allclose(risk.mse, risk.positive_mse+risk.negative_mse+risk.weak_mse,
                                   rtol=2e-10, atol=2e-12)
        risks[action] = risk.mse
    a = -params['threshold']-params['gate_mean']/params['gate_sd']
    b = params['threshold']-params['gate_mean']/params['gate_sd']
    probability_weak = norm.cdf(b)-norm.cdf(a)
    difference = params['beta']**2*probability_weak-params['fit_sd']**2*(
        probability_weak+params['correlation']**2*(a*norm.pdf(a)-b*norm.pdf(b)))
    np.testing.assert_allclose(risks['exclude']-risks['release'], difference, atol=2e-11)
    print({'conditional_mean': geometry.pooled_means.tolist(),
           'marginal_mean': marginal.tolist(), 'variance_ratio': variance_ratio.tolist(),
           'null_exclusion_mse': exclusion.mse, 'uninformative_sign_mse': clipping.mse,
           'independent_risk_checks': risks})


if __name__ == '__main__':
    main()
