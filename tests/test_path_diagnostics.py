"""Fitted-state completeness of regularisation-path models and refits.

``fit_reg_lambda_path`` promises that every returned model is equivalent to a fresh ``fit`` at
its ``reg_lambda``, with the same diagnostics. These tests compare the preparation diagnostics,
which do not depend on ``reg_lambda``, exactly. Solved coefficients are compared with the
existing path tolerances elsewhere (``test_lambda_path_dpp.py``, ``test_cv_lambda_path.py``).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from factorlasso import LassoModel, LassoModelType

#: Fitted attributes derived before the solve; a path model must equal a fresh fit on each.
PREPARATION_DIAGNOSTICS = (
    "derived_signs_", "detected_signs_", "sign_slopes_", "sign_t_stats_", "sign_effective_n_",
    "sign_valid_counts_", "sign_penalty_weights_", "sign_block_weights_", "effective_sign_span_",
    "ols_betas_", "ols_r2_", "ols_beta_prior_", "effective_beta_prior_", "ols_prior_span_",
    "prior_lower_bounds_", "prior_upper_bounds_", "prior_bound_diagnostics_",
    "effective_prior_hac_lags_", "clusters_", "linkage_", "cutoff_", "valid_mask_",
    "effective_span_", "effective_cluster_correlation_span_",
)

#: Mutable diagnostics that each returned model must own independently.
OWNED_DIAGNOSTICS = (
    "derived_signs_", "detected_signs_", "sign_slopes_", "sign_t_stats_", "sign_effective_n_",
    "sign_valid_counts_", "sign_penalty_weights_", "sign_block_weights_", "ols_betas_",
    "ols_r2_", "ols_beta_prior_", "effective_beta_prior_", "prior_lower_bounds_",
    "prior_upper_bounds_", "prior_bound_diagnostics_", "clusters_", "linkage_", "valid_mask_",
)

LAMBDAS = (1e-4, 1e-3, 1e-2)


def _panel():
    """Six responses in two blocks driven by three factors (the V3 review probe)."""
    rng = np.random.default_rng(20261002)
    index = pd.date_range("2000-01-31", periods=100, freq="ME")
    x = pd.DataFrame(rng.standard_normal((100, 3)), index=index, columns=["f0", "f1", "f2"])
    beta = np.array([[1, .3, 0], [1, .2, 0], [1, .1, 0], [0, 1, .3], [0, 1, .2], [0, 1, .1]])
    y = pd.DataFrame(x.to_numpy() @ beta.T + 0.1 * rng.standard_normal((100, 6)),
                     index=index, columns=[f"y{j}" for j in range(6)])
    return x, y


def _restricted(model_type):
    """A configuration that populates the sign, adaptive-weight and prior diagnostics."""
    params = dict(
        model_type=model_type, reg_lambda=1e-3, span=36, auto_sign_constraints=True,
        auto_sign_adaptive_weights=True, auto_sign_use_fit_span=True, apply_ols_prior=True,
    )
    if model_type == LassoModelType.GROUP_LASSO:
        params["group_data"] = pd.Series(["a", "a", "a", "b", "b", "b"],
                                         index=[f"y{j}" for j in range(6)])
    else:
        params["n_clusters"] = 2
    return params


def _equal(left, right):
    """Exact equality for the diagnostic value types a fit stores."""
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, (pd.DataFrame, pd.Series)):
        return type(left) is type(right) and left.equals(right) and (
            left.index.equals(right.index)
            and (not isinstance(left, pd.DataFrame) or left.columns.equals(right.columns)))
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right, equal_nan=True)
    return left == right


@pytest.mark.parametrize("model_type", [
    LassoModelType.GROUP_LASSO,
    LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
    LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
], ids=["GL", "HCGL", "FCGL"])
def test_path_models_carry_fresh_fit_preparation_diagnostics(model_type):
    """Every path model has the preparation diagnostics of a fresh fit at its lambda."""
    x, y = _panel()
    params = _restricted(model_type)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = LassoModel(**params).fit_reg_lambda_path(x, y, reg_lambdas=LAMBDAS)
        fresh = [LassoModel(**{**params, "reg_lambda": lam}).fit(x, y) for lam in LAMBDAS]
    mismatched = sorted({name for model, reference in zip(path, fresh)
                         for name in PREPARATION_DIAGNOSTICS
                         if not _equal(getattr(model, name), getattr(reference, name))})
    assert mismatched == [], ", ".join(mismatched)


def test_refit_without_signs_clears_derived_signs():
    """A refit whose configuration enforces no signs leaves ``derived_signs_`` unset."""
    x, y = _panel()
    model = LassoModel(model_type=LassoModelType.LASSO, reg_lambda=1e-3,
                       auto_sign_constraints=True).fit(x, y)
    assert model.derived_signs_ is not None
    model.set_params(auto_sign_constraints=False)
    model.fit(x, y)
    assert model.derived_signs_ is None


def test_path_from_refitted_template_has_no_stale_signs():
    """Path models never inherit the template's signs from an earlier configuration."""
    x, y = _panel()
    params = dict(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO, reg_lambda=1e-3,
                  n_clusters=2, auto_sign_constraints=True)
    template = LassoModel(**params).fit(x, y)
    template.set_params(auto_sign_constraints=False)
    path = template.fit_reg_lambda_path(x, y, reg_lambdas=LAMBDAS[:2])
    assert all(model.derived_signs_ is None for model in path)


def test_path_models_own_their_diagnostics():
    """No mutable diagnostic object is shared between path models or with the template."""
    x, y = _panel()
    template = LassoModel(**_restricted(LassoModelType.FACTOR_CLUSTER_GROUP_LASSO))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = template.fit_reg_lambda_path(x, y, reg_lambdas=LAMBDAS)
    shared = []
    for name in OWNED_DIAGNOSTICS:
        objects = [getattr(model, name) for model in path] + [getattr(template, name, None)]
        ids = [id(obj) for obj in objects if obj is not None]
        if len(ids) != len(set(ids)):
            shared.append(name)
    assert shared == [], ", ".join(shared)
