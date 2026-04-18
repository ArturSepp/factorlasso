"""
Example: Gene expression factor model with biological sign constraints
======================================================================

Demonstrates factorlasso on a synthetic genomics problem where gene
expression levels are driven by latent pathway activity factors.

This example is relevant to the bioinformatics community:
- Gene expression is modelled as a linear combination of pathway factors
- Biological priors enforce sign constraints (activators vs repressors)
- Group LASSO discovers co-regulated gene modules
- The estimated factor covariance captures pathway correlations

The same methodology applies to any multi-output regression problem
where domain knowledge constrains coefficient signs.
"""

import numpy as np
import pandas as pd

from factorlasso import (
    LassoModel,
    LassoModelType,
)


def main():
    np.random.seed(123)

    # --- 1. Simulate pathway-driven gene expression ---
    n_samples = 150     # biological samples
    n_pathways = 4      # latent pathway factors
    n_genes = 20        # measured genes

    pathway_names = ['PI3K/AKT', 'MAPK/ERK', 'WNT', 'Apoptosis']
    gene_names = [f'Gene_{i:02d}' for i in range(n_genes)]

    # Pathway activities (regressors)
    X = pd.DataFrame(
        np.random.randn(n_samples, n_pathways),
        columns=pathway_names,
    )

    # True sparse loadings: most genes respond to 1-2 pathways
    beta_true = np.zeros((n_genes, n_pathways))
    # PI3K/AKT cluster (genes 0-4): positive regulation
    beta_true[0:5, 0] = np.array([1.2, 0.9, 0.7, 1.0, 0.5])
    # MAPK/ERK cluster (genes 5-9): positive regulation
    beta_true[5:10, 1] = np.array([1.0, 0.8, 1.1, 0.6, 0.9])
    # WNT cluster (genes 10-14): mixed regulation
    beta_true[10:15, 2] = np.array([0.8, -0.5, 0.9, 0.7, -0.3])
    # Apoptosis cluster (genes 15-19): mixed regulation
    beta_true[15:20, 3] = np.array([-0.9, -1.1, -0.7, 0.4, -0.8])
    # Cross-talk: some PI3K genes also respond to MAPK
    beta_true[0, 1] = 0.3
    beta_true[2, 1] = 0.2

    # Gene expression = pathway activity × loadings + noise
    noise = 0.3 * np.random.randn(n_samples, n_genes)
    Y = pd.DataFrame(X.values @ beta_true.T + noise, columns=gene_names)

    # --- 2. Biological sign constraints ---
    # Domain knowledge: PI3K/AKT and MAPK/ERK are known activators
    # for their target genes (non-negative loadings)
    signs = pd.DataFrame(np.nan, index=gene_names, columns=pathway_names)
    signs.iloc[0:5, 0] = 1    # PI3K target genes: non-negative
    signs.iloc[5:10, 1] = 1   # MAPK target genes: non-negative
    # Apoptosis pathway: genes 15-17 are known to be repressed
    signs.iloc[15:18, 3] = -1  # non-positive

    # --- 3. Estimate with HCGL (auto-discovers gene modules) ---
    model = LassoModel(
        model_type=LassoModelType.GROUP_LASSO_CLUSTERS,
        reg_lambda=1e-4,
        span=None,  # no EWMA for cross-sectional data
        demean=True,
        factors_beta_loading_signs=signs,
    )
    model.fit(x=X, y=Y)

    print("=== Estimated loadings (genes × pathways) ===")
    print(model.estimated_betas.round(2).to_string())
    print()

    print("=== Discovered gene modules (clusters) ===")
    print(model.clusters.sort_values().to_string())
    print()

    # --- 4. Compare to ground truth ---
    error = np.abs(model.estimated_betas.values - beta_true)
    print(f"Mean absolute estimation error: {error.mean():.3f}")
    print(f"Max absolute estimation error:  {error.max():.3f}")
    print()

    # --- 5. Check sign constraints are satisfied ---
    betas = model.estimated_betas.values
    print("Sign constraint check:")
    print(f"  PI3K targets (genes 0-4) all >= 0: "
          f"{np.all(betas[0:5, 0] >= -1e-8)}")
    print(f"  MAPK targets (genes 5-9) all >= 0: "
          f"{np.all(betas[5:10, 1] >= -1e-8)}")
    print(f"  Apoptosis repressed (genes 15-17) all <= 0: "
          f"{np.all(betas[15:18, 3] <= 1e-8)}")
    print()

    # --- 6. R² per gene ---
    r2 = model.estimation_result_.r2
    print("=== R² per gene ===")
    r2_series = pd.Series(r2, index=gene_names)
    print(f"  Mean R²:   {r2_series.mean():.3f}")
    print(f"  Median R²: {r2_series.median():.3f}")
    print(f"  Min R²:    {r2_series.min():.3f}")


if __name__ == '__main__':
    main()
