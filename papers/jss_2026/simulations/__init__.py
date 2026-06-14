"""
papers.jss_2026.simulations
===========================

Simulation harness for the cluster-pooled sign-derivation methodology
paper (JSS 2026 submission). Lives under ``papers/jss_2026/`` so all
artefacts tied to this publication — paper source, simulation study,
and SPY application — sit in a single self-contained module.

This package is **not** part of the published ``factorlasso`` wheel.
It is in the repository for version control and reproducibility, not
installed via ``pip install factorlasso``.

Layout
------
- ``dgp``         — data-generating processes (factor returns + true β
                    with controllable cluster structure, sign mix, SNR)
- ``estimators``  — unified ``fit(X, y, λ, …) → EstimatorResult``
                    wrappers around each estimator under comparison
- ``metrics``     — pure metric functions on (β_true, β_hat, …)
- ``run``         — orchestration CLI; expands ``study.yaml`` into the
                    full (regime, seed, estimator, λ) grid, runs in
                    parallel, saves long-form results
- ``study.yaml``  — the JSS 2026 regime grid and λ schedule
- ``results/``    — output parquet + manifest (gitignored)

Usage
-----
From the repository root::

    python -m papers.jss_2026.simulations.run \\
        --config papers/jss_2026/simulations/study.yaml \\
        --output papers/jss_2026/simulations/results

Or for a single-seed smoke test::

    python -m papers.jss_2026.simulations.run \\
        --config papers/jss_2026/simulations/study.yaml \\
        --output /tmp/smoke \\
        --seeds-limit 1
"""

__all__ = [
    "dgp",
    "estimators",
    "metrics",
]
