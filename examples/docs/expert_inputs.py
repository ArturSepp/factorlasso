"""Offline demonstration of named-set conditioning and peer loading budgets."""
from __future__ import annotations

import numpy as np
import pandas as pd

from factorlasso import LassoModel, LassoModelType


def run() -> LassoModel:
    """Check leave-one-out budgets against explicit peer averaging."""
    rng = np.random.default_rng(716)
    raw = rng.normal(size=(180, 3))
    raw -= raw.mean(axis=0)
    orthogonal = np.linalg.qr(raw)[0] * np.sqrt(len(raw))
    x = pd.DataFrame(orthogonal, columns=['a', 'b', 'c'])
    x['b'] = .35 * x.a + np.sqrt(1 - .35**2) * x.b
    y = pd.DataFrame({f'fund{i}': (1 + i * .1) * x.a + .4 * x.b for i in range(5)})
    model = LassoModel(
        model_type=LassoModelType.GROUP_LASSO,
        group_data=pd.Series('cluster', index=y.columns),
        apply_ols_prior=True,
        factor_for_prior={name: 'a' for name in y.columns},
        expert_prior_budget_rule='cluster_mean',
        expert_prior_bound_centre='floor',
        reg_lambda=0., loss_normalization='weight_sum',
    ).fit(x=x, y=y)
    for fund in y:
        peers = model.budget_first_pass_coef_.drop(index=fund).abs().mean()
        np.testing.assert_allclose(model.resolved_budgets_.loc[fund], peers, atol=1e-12)
    np.testing.assert_allclose(model.resolved_budgets_.b, .4, atol=1e-6)
    np.testing.assert_allclose(model.prior_bound_diagnostics_.slack, .35 * .4, atol=1e-6)
    return model


if __name__ == '__main__':
    run()
