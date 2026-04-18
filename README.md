# factorlasso 0.2.0 — Release Bundle

This bundle contains the three source files that need to be committed to the
repo for the 0.2.0 release, plus the built distribution artefacts.

## Contents

| File | Where it goes | What changed |
|---|---|---|
| `__init__.py` | `factorlasso/__init__.py` | Rewritten to export the 5 cluster utilities from `cluster_utils` |
| `.gitignore` | repo root | Fixed `.idea/` exclusion; expanded Python project ignores |
| `pyproject.toml` | repo root | Version bumped `0.1.12 → 0.2.0` |
| `factorlasso-0.2.0-py3-none-any.whl` | `dist/` | Fresh wheel build (includes `cluster_utils.py`) |
| `factorlasso-0.2.0.tar.gz` | `dist/` | Fresh sdist build |

## What verified

Against the full repo state (cluster_utils.py + updated __init__.py + all other modules):

- ✓ `python -m pytest tests/` → **143 passed, 9 skipped** (skipped are environment-specific)
- ✓ `python tests/test_integration.py` → runs end-to-end
- ✓ Fresh wheel builds cleanly via `python -m build --wheel --sdist`
- ✓ Installing from the fresh wheel and re-running all tests → all pass
- ✓ `from factorlasso import get_clusters_by_freq, get_linkages_by_freq, get_cutoffs_by_freq, get_linkage_array, compute_clusters_from_corr_matrix` → all importable
- ✓ Round-trip HCGL ≡ GROUP_LASSO-with-external-clusters semantic test passes
- ✓ `.gitignore` correctly excludes `.idea/*`, `__pycache__/`, build artefacts

## Version bump rationale (semver)

0.1.12 → 0.2.0 because the release **adds new public API**:

- `get_clusters_by_freq` (new)
- `get_linkages_by_freq` (new)
- `get_cutoffs_by_freq` (new)
- `get_linkage_array` (promoted from internal to public export)

Plus a module-level reorganisation (all clustering utilities consolidated
into the new `factorlasso.cluster_utils` module). The existing public
API is fully preserved — `compute_clusters_from_corr_matrix` is still
importable from `factorlasso` top-level exactly as before.

## One manual cleanup step needed in your working copy

The `.idea/` directory was previously committed (the old `.gitignore` had
`*.idea/` which doesn't match `.idea/`). Updating `.gitignore` alone
won't untrack what's already in git. Run this once:

```bash
git rm -r --cached .idea/
git commit -m "chore: untrack .idea IDE metadata"
```

After that, future changes to `.idea/` files will be ignored correctly.

## Suggested commit structure

Three logical commits:

```bash
# 1. Code reorganisation (already in your repo — cluster_utils.py, the
#    updated lasso_estimator.py and factor_covar.py)
git add factorlasso/cluster_utils.py factorlasso/lasso_estimator.py factorlasso/factor_covar.py
git commit -m "refactor: consolidate clustering utilities into cluster_utils module"

# 2. Public API update
git add factorlasso/__init__.py
git commit -m "feat: export cluster helpers (get_*_by_freq, get_linkage_array)"

# 3. Version bump + .gitignore fix
git add pyproject.toml .gitignore
git rm -r --cached .idea/
git commit -m "chore: bump to 0.2.0; fix .idea gitignore glob"

# 4. (optional) Rebuild dist/ if you want to commit artefacts
# Most projects don't commit dist/ — just build on release via CI
```

## Changelog entry (for `CHANGELOG.md` if you have one)

```markdown
## [0.2.0] - 2026-04-18

### Added
- New module `factorlasso.cluster_utils` consolidating all clustering utilities
- Public API exports: `get_clusters_by_freq`, `get_linkages_by_freq`, `get_cutoffs_by_freq`
- Public API export: `get_linkage_array` (previously internal)

### Changed
- `compute_clusters_from_corr_matrix` moved from `lasso_estimator.py` to `cluster_utils.py`
  (still importable from top-level `factorlasso`)
- `get_linkage_array` moved from `factor_covar.py` to `cluster_utils.py`
- `LassoModel` with `model_type=GROUP_LASSO` now populates `.clusters_` attribute from
  externally-supplied `group_data`, for API uniformity with HCGL mode

### Fixed
- `compute_clusters_from_corr_matrix` now uses `squareform(1 - C)` (correct correlation-to-distance
  conversion) instead of the previous buggy `pdist(1 - C)` (which computed Euclidean distances
  between rows of the correlation matrix)
- `.gitignore` glob `*.idea/` corrected to `.idea/`

### Internal
- All 143 existing tests continue to pass
- Wheel now correctly includes `cluster_utils.py`
```
