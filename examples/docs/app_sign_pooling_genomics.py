"""Synthetic configuration check for docs/app_sign_pooling_genomics.md.

The empirical results come from the revised paper's nested validation, not this script.
This self-contained example standardizes a training panel, supplies its Ward partition
with a final-merge cut, fits HCGL with the date-score gate, and checks the pooled signs
and slopes against a direct calculation. No yeast data or archived outputs are read.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

import factorlasso as fl

SEED = 20261025
CONFIGURATION = {
    "model_type": fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
    "reg_lambda": 0.1,  # Illustrative fixed penalty; the empirical study tunes it in inner folds.
    "l1_weight": 0.5,
    "auto_sign_constraints": True,
    "auto_sign_threshold_t": 2.0,
    "auto_sign_variance": "date",
    "demean": True,
    "solver": "CLARABEL",
}
CUT = 0.7


def training_partition(y: pd.DataFrame) -> pd.Series:
    """Ward groups from standardized training responses, cut at 0.7 of final merge."""
    distances = pdist(y.to_numpy().T) / np.sqrt(len(y))
    tree = linkage(distances, method="ward")
    labels = fcluster(tree, CUT * tree[-1, 2], criterion="distance")
    return pd.Series(labels, index=y.columns, name="cluster")


def main() -> None:
    rng = np.random.default_rng(SEED)
    x_raw = pd.DataFrame(rng.normal(size=(100, 6)), columns=[f"marker_{j}" for j in range(6)])
    beta = np.zeros((8, 6))
    beta[:4, :2] = [1.0, 0.5]
    beta[4:, 2:4] = [1.0, -0.5]
    shared = rng.normal(scale=0.5, size=(100, 2))
    errors = shared[:, np.repeat([0, 1], 4)] + rng.normal(scale=0.25, size=(100, 8))
    y_raw = pd.DataFrame(x_raw.to_numpy() @ beta.T + errors,
                         columns=[f"gene_{k}" for k in range(8)])

    # All centering, scaling and clustering use only the 80 training observations.
    x_train, y_train = x_raw.iloc[:80], y_raw.iloc[:80]
    x_mean, x_sd = x_train.mean(), x_train.std(ddof=0)
    y_mean, y_sd = y_train.mean(), y_train.std(ddof=0)
    x, y = (x_train - x_mean) / x_sd, (y_train - y_mean) / y_sd
    clusters = training_partition(y)
    model = fl.LassoModel(**CONFIGURATION).fit(x=x, y=y, external_clusters=clusters)

    # Independent reference: aggregate residual scores within rows before squaring.
    for label in clusters.unique():
        genes = clusters.index[clusters == label]
        x_arr, y_arr = x.to_numpy(), y.loc[:, genes].to_numpy()
        denominator = len(genes) * np.sum(x_arr**2, axis=0)
        slope = x_arr.T @ y_arr.sum(axis=1) / denominator
        residuals = y_arr[:, :, None] - x_arr[:, None, :] * slope[None, None, :]
        scores = x_arr * residuals.sum(axis=1)
        variance = len(x) / (len(x) - 1) * np.sum(scores**2, axis=0) / denominator**2
        statistic = slope / np.sqrt(variance)
        expected = np.where(np.abs(statistic) >= 2, np.sign(slope), 0.)
        np.testing.assert_allclose(model.sign_slopes_.loc[genes],
                                   np.broadcast_to(slope, (len(genes), len(x.columns))))
        np.testing.assert_allclose(model.sign_t_stats_.loc[genes],
                                   np.broadcast_to(statistic, (len(genes), len(x.columns))))
        np.testing.assert_array_equal(model.derived_signs_.loc[genes],
                                      np.broadcast_to(expected, (len(genes), len(x.columns))))

    signs, coefficients = model.derived_signs_.to_numpy(), model.coef_.to_numpy()
    assert np.all(coefficients[signs > 0] >= -1e-7)
    assert np.all(coefficients[signs < 0] <= 1e-7)
    assert np.all(np.abs(coefficients[signs == 0]) < 1e-7)
    predictions = model.predict((x_raw.iloc[80:] - x_mean) / x_sd)
    assert predictions.shape == (20, 8) and np.isfinite(predictions.to_numpy()).all()
    print("Synthetic date-score HCGL configuration and direct gate reference passed.")


if __name__ == "__main__":
    main()
