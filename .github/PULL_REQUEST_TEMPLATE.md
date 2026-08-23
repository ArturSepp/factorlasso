## What changed

Describe the problem and the smallest coherent change that solves it.

## Verification

List the exact commands run and their results. For numerical changes, include the independent
reference calculation and any replication-output comparison.

## Checklist

- [ ] Tests cover the changed public behavior or defect.
- [ ] Estimation remains point-in-time, with shapes, scaling, and missing-data behavior explicit.
- [ ] Estimator defaults, penalty scaling, sign constraints, and covariance assembly are unchanged, or the numerical and replication evidence is included.
- [ ] The scikit-learn API contract and core-install independence from scikit-learn remain intact.
- [ ] No credentials, private/licensed data, local paths, generated outputs, papers, or agent reports are included unintentionally.
- [ ] `uv run --no-sync pytest` and the relevant static/docs checks pass.
- [ ] User-visible changes are documented in `CHANGELOG.md` and relevant docs.
- [ ] New runtime dependencies or public-signature changes are called out explicitly.
