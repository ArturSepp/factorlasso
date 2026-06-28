"""single-asset (N=1) handling for the clustering primitive and every
clustering / group-based estimator, through both fit() and the lambda path."""
import numpy as np
import pandas as pd
import pytest

from factorlasso.cluster_utils import compute_clusters_from_corr_matrix
from factorlasso.lasso_estimator import LassoModel, LassoModelType

# production overlay that exposed the FCGL block-weights crash at N=1
_PROD = dict(
    auto_sign_constraints=True,
    auto_sign_threshold_t=1.0,
    auto_sign_adaptive_weights=True,
    auto_sign_adaptive_gamma=1.0,
    auto_sign_adaptive_floor=0.5,
)


def _single_asset_panel():
    rng = np.random.default_rng(0)
    t, m = 120, 4
    factors = [f'f{j}' for j in range(m)]
    x = pd.DataFrame(rng.standard_normal((t, m)), columns=factors)
    y = pd.DataFrame({'asset0': x['f0'] * 0.5 + 0.1 * rng.standard_normal(t)})
    return x, y


def _kwargs(model_type, x, y, production):
    kw = dict(model_type=model_type, reg_lambda=0.01, span=None)
    if production:
        kw.update(_PROD)
        kw['factors_beta_prior'] = pd.DataFrame(
            [[0.1, 0.0, 0.0, 0.0]], index=y.columns, columns=x.columns,
        )
    if model_type in (LassoModelType.GROUP_LASSO,
                      LassoModelType.COOPERATIVE_GROUP_LASSO):
        kw['group_data'] = pd.Series({'asset0': 'grpA'})
    return kw


def test_single_asset_corr_returns_lone_cluster():
    # a 1x1 correlation matrix has an empty condensed distance vector, so the
    # function must short-circuit instead of letting scipy.linkage raise.
    corr = pd.DataFrame([[1.0]], index=['solo'], columns=['solo'])
    clusters, linkage, cutoff = compute_clusters_from_corr_matrix(corr)
    assert clusters.to_dict() == {'solo': 1}
    assert linkage.shape == (0, 4)
    assert cutoff == 0.0


@pytest.mark.parametrize('model_type', list(LassoModelType))
@pytest.mark.parametrize('production', [False, True])
def test_single_asset_fit_all_estimators(model_type, production):
    # every estimator must fit a single asset without raising, vanilla and
    # production (auto-sign + adaptive weights + prior).
    x, y = _single_asset_panel()
    model = LassoModel(**_kwargs(model_type, x, y, production)).fit(x=x, y=y)
    assert model.coef_.shape == (1, x.shape[1])


@pytest.mark.parametrize('model_type', list(LassoModelType))
def test_single_asset_lambda_path_all_estimators(model_type):
    # the lambda-path entry point (LassoModelCV use_lambda_path) must also
    # handle a single asset for every estimator.
    x, y = _single_asset_panel()
    lambdas = [0.001, 0.01, 0.1]
    models = LassoModel(**_kwargs(model_type, x, y, True)).fit_reg_lambda_path(
        x=x, y=y, reg_lambdas=lambdas,
    )
    assert len(models) == len(lambdas)
    assert all(m.coef_.shape == (1, x.shape[1]) for m in models)
