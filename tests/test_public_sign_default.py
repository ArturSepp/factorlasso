"""Public method compatibility is explicit while corrected gates remain opt-in."""

import numpy as np
from factorlasso import LassoModel
from factorlasso.sign_constraints import _compute_sign_vector


def test_public_gate_defaults_to_independent_without_reverting_mask_fix():
    """A bare public model uses the submitted-method variance convention."""
    model = LassoModel()
    assert model.auto_sign_variance == "independent"
    assert model.auto_sign_use_fit_span is False
    assert model.solver == "CLARABEL"


def test_standalone_default_is_legacy_and_date_remains_explicit():
    """Default and explicit independent calls agree; duplicated-date correction is available."""
    rng = np.random.default_rng(93)
    x = rng.normal(size=(90, 1))
    y = 0.1 * x + rng.normal(size=(90, 1))
    copies = np.tile(y, (1, 4))
    default = _compute_sign_vector(x, copies, return_diagnostics=True)[2]["t_stats"]
    legacy = _compute_sign_vector(
        x, copies, variance_estimator="independent", return_diagnostics=True
    )[2]["t_stats"]
    date = _compute_sign_vector(x, copies, variance_estimator="date", return_diagnostics=True)[2][
        "t_stats"
    ]
    single = _compute_sign_vector(x, y, variance_estimator="date", return_diagnostics=True)[2][
        "t_stats"
    ]
    np.testing.assert_allclose(default, legacy)
    np.testing.assert_allclose(date, single)
    assert not np.allclose(default, date)
