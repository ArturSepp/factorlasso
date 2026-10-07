"""Conservative joint WLS coefficient/scale regions under fixed Gaussian models.

Imhof (1961), Biometrika 48, 419-426, doi:10.1093/biomet/48.3-4.419,
supplies the quadratic-distribution reference used by the existing quantile API.
Bonferroni's inequality combines coefficient and residual-scale events; see
https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm.
The error-budget split and plug-in upper-scale enclosure below are this
implementation's conservative construction, not a post-LASSO coverage theorem.
"""
import numpy as np
from scipy.stats import norm

from factorlasso.inference._geometry import compute_wls_hac_geometry
from factorlasso.inference._quadratic import gaussian_quadratic_quantile
from factorlasso.inference._validation import _confidence, _symmetric


def joint_wls_gaussian_region(x, responses, weights, *, covariance_shape,
                              confidence=.95, coefficient_scale=None):
    """Bound all prespecified regression coefficients and noise scales jointly.

    Parameters
    ----------
    x : array-like, shape (T, M)
        Fixed regressors without an intercept. A leading intercept is added.
        Nonfinite rows are allowed only when every response is missing there.
    responses : array-like, shape (T, N) or (draws, T, N)
        Response panels. NaNs denote fixed missing observations. Batched panels
        must share exactly the same masks; batches share geometry, not error budget.
    weights : array-like, shape (T, N)
        Fixed nonnegative observation weights, zero at missing cells. Positive
        support must identify every coefficient and leave residual degrees of freedom.
        WLS normalizes the loss weights. For initialized EWMA alpha, explicitly
        supply its retained mass in coefficient_scale for the intercept.
    covariance_shape : array-like, shape (T, T)
        Known positive-definite temporal shape R with unit diagonal. Each asset's
        marginal error law is Gaussian with covariance sigma_i squared times R.
        Cross-asset dependence is unrestricted. Subsetting retains calendar gaps.
    confidence : float, default .95
        Simultaneous level for all N*(M+1) coefficients and all N variances.
    coefficient_scale : array-like, shape (N, M+1), optional
        Positive multipliers of the coefficient targets, e.g. annualization times
        initialized weight mass for the intercept. None means unscaled coefficients.

    Returns
    -------
    dict
        coefficient_estimate/lower/upper have shape (N, M+1), or a leading batch
        axis; variance_estimate/lower/upper have shape (N,), or a batch axis.
        Variances describe response noise per observation, without annualization.
        Shape and mean assumptions are declared, not estimated or verified here.
        No ridge repair, dropped coefficient or selected-model inference is applied.

    Notes
    -----
    Half the failure budget covers all two-sided variance intervals; half covers
    all Gaussian coefficient errors. Coefficient radii use the variance upper
    bound. The union bound does not need independence between estimates/scales or
    across assets. Weighted residual quadratic quantiles account for fitting and
    EWMA concentration. Coverage requires fixed correct means, weights, masks and
    Gaussian temporal shape; an estimated shape does not inherit this guarantee.
    """
    confidence = _confidence(confidence)
    xx, yy, ww = np.asarray(x, float), np.asarray(responses, float), np.asarray(weights, float)
    batched = yy.ndim == 3
    if yy.ndim == 2:
        yy = yy[None, :, :]
    if (xx.ndim != 2 or yy.ndim != 3 or not yy.size or len(xx) != yy.shape[1]
            or ww.shape != yy.shape[1:] or np.isinf(yy).any()):
        raise ValueError('x, responses and weights must have aligned nonempty panel shapes')
    observed = np.isfinite(yy[0])
    if not np.all(np.isfinite(yy) == observed):
        raise ValueError('batched responses must share fixed missing masks')
    if (not np.isfinite(ww).all() or np.any(ww < 0) or np.any(ww[~observed] != 0)
            or np.any(~np.isfinite(xx).all(axis=1) & observed.any(axis=1))):
        raise ValueError('finite nonnegative weights and regressors must respect observed support')
    nassets, p = yy.shape[2], xx.shape[1]+1
    scales = (np.ones((nassets, p)) if coefficient_scale is None
              else np.asarray(coefficient_scale, float))
    if scales.shape != (nassets, p) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('coefficient_scale must be positive finite with shape (N, M+1)')
    shape = _symmetric(covariance_shape, len(xx), 'covariance_shape', True)
    if not np.allclose(np.diag(shape), 1., rtol=0, atol=1e-12):
        raise ValueError('covariance_shape must have unit diagonal for marginal variance targets')
    failure = 1-confidence
    variance_tail = failure/(4*nassets)
    critical = norm.isf(failure/(4*nassets*p))
    estimates = np.empty((len(yy), nassets, p))
    radii = np.empty_like(estimates)
    variance = np.empty((len(yy), nassets))
    lower, upper = np.empty_like(variance), np.empty_like(variance)
    counts, effective_n, cache = [], [], {}
    for asset in range(nassets):
        valid = observed[:, asset] & (ww[:, asset] > 0)
        w = ww[valid, asset]
        counts.append(int(valid.sum()))
        if len(w) <= p:
            raise ValueError('positive residual degrees of freedom required for every asset')
        w = w/w.max()
        effective_n.append(float(w.sum()**2/(w @ w)))
        key = (valid.tobytes(), w.tobytes())
        if key not in cache:
            design = np.column_stack([np.ones(len(w)), xx[valid]])
            geometries = [compute_wls_hac_geometry(design, w, coefficient=j) for j in range(p)]
            h = np.array([g.linear for g in geometries])
            residual = geometries[0].residual_map
            r = shape[np.ix_(valid, valid)]
            q = residual.T @ ((w/w.sum())[:, None]*residual)
            root = np.linalg.cholesky(r)
            transformed = root.T @ q @ root
            eigen = np.linalg.eigvalsh((transformed+transformed.T)/2)
            tolerance = 100*np.finfo(float).eps*len(w)*np.max(abs(eigen))
            if eigen.min() < -tolerance:
                raise ArithmeticError('weighted residual form is numerically indefinite')
            eigen = np.maximum(eigen, 0.)
            expectation = eigen.sum()
            quantiles = [gaussian_quadratic_quantile(eigen/expectation, prob)
                         for prob in (variance_tail, 1-variance_tail)]
            noise_variance = np.einsum('ij,jk,ik->i', h, r, h)
            cache[key] = h, residual, expectation, quantiles, noise_variance
        h, residual, expectation, quantiles, noise_variance = cache[key]
        panel = yy[:, valid, asset]
        errors = panel @ residual.T
        v = np.sum(errors**2*(w/w.sum()), axis=1)/expectation
        if not np.isfinite(v).all() or np.any(v <= 0):
            raise ValueError('positive finite residual scale required; degenerate fits unsupported')
        variance[:, asset] = v
        lower[:, asset], upper[:, asset] = v/quantiles[1], v/quantiles[0]
        estimates[:, asset] = (panel @ h.T)*scales[asset]
        radii[:, asset] = critical*np.sqrt(upper[:, asset, None]*noise_variance)*scales[asset]
    values = dict(coefficient_estimate=estimates, coefficient_lower=estimates-radii,
                  coefficient_upper=estimates+radii, variance_estimate=variance,
                  variance_lower=lower, variance_upper=upper)
    if not batched:
        values = {key: value[0] for key, value in values.items()}
    return dict(**values, observations=np.array(counts), effective_n=np.array(effective_n),
                confidence=confidence, coefficient_critical=float(critical),
                variance_tail_probability=variance_tail,
                scope='all_prespecified_coefficients_and_variances',
                status='known_shape_gaussian_only',
                assumptions='fixed full mean design, weights, masks and correct Gaussian shape')
