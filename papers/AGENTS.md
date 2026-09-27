# Paper workspaces

This is FactorLasso's paper-workspace contract. Root AGENTS.md controls the
external Python environment, numerical invariants and C-local runtime workspace.

## Layout and ownership

Keep papers at `papers/<paper_id>/`, outside `src/factorlasso/`. Use these six
sections as needed; do not commit placeholders for ignored local directories.

| Section | Purpose | Git policy |
|---|---|---|
| `paper/` | One current manuscript, reading PDF, bibliography, table fragments and `figures/` | Exact approved files only |
| `drafts/<date_or_version>/` | Previous versions with their own figures and build dependencies | Always ignored |
| `presentations/<date_event>/` | Slides with their own figures | Ignored by default; exact approved exceptions |
| `private/` | Editor correspondence, referee reports, replies and permissions | Always ignored |
| `replication/` | Code, static `data/`, and automated `tests/` | Approved code and redistributable inputs tracked |
| `agents/` | Paper-specific roadmaps, execution reports and working notes | Always ignored |

Per-paper `agents/` is FL's explicit override of the generated shared core's
root-only working-record location. Repository-wide records stay in root
`agents/`. Tracked `AGENTS.md` files are instructions, not working records.

Reusable estimators belong in `src/factorlasso/`. Replication imports the
package; production modules must never import `papers.*`. Preserve existing
direct script commands and support repository-root module entry points. Paper
tests live in `replication/tests/test_*.py`; core tests remain in root `tests/`.

## Publication and preservation

- Only `sign_pooling_2026` (CSDA) is approved for tracking. JSS, prior-targets and
  any future paper workspaces remain entirely local. Top-level `papers/AGENTS.md`
  and `papers/README.md` are the tracked policy and availability index.
- A new public paper needs an explicit maintainer decision, an index update,
  root ignore exception and corresponding checker update. Do not infer GitHub
  redistribution rights from submission, acceptance, SSRN posting or a DOI.
- Deny `paper/`, `presentations/` and `replication/data/` by default. The
  per-paper `.gitignore` lists each approved file exactly, including necessary
  bibliography/table/figure dependencies. Never approve future files through
  wildcard exceptions. Elsevier CAS templates remain local or installed through
  the LaTeX distribution; do not vendor them as part of this migration.
- Public docs may display only tracked exhibits and must not link files in
  local-only workspaces. Keep citation metadata without promising public source.
- Restricted inputs belong in `replication/data/local/`, always ignored. Record
  provenance and hashes for tracked static inputs and frozen reference caches.
  Missing licensed inputs must be reported; never substitute different data
  silently. Preserve shared input modules rather than duplicating them.
- Current CSDA reference caches live in `replication/data/reference/`. Producers
  write new results externally. `exhibits.py` reads frozen inputs by default;
  `--from-run` reads only the selected external run, without cached fallbacks.
- Configure the root-prescribed runtime before running code. New runs, caches,
  logs, figures and LaTeX builds go outside the checkout and OneDrive, using
  `FACTORLASSO_PAPER_OUTPUT_DIR` or the configured C-local runtime. Promote
  reviewed figures/tables into `paper/` explicitly; never overwrite the
  manuscript or frozen inputs merely to match a new run.
- Preserve local files during untracking and moves. Archive drafts with their
  dependencies and record when historical figure provenance is unknown. Ignored
  material requires private backup/version history outside Git. Untracking does
  not erase existing Git history; history rewrites are a separate decision.
- Keep local JSS and prior-targets layouts intact during this adoption. Do not
  reorganize their mixed historical content without identifying each file's role.

## Verification

After configuring the external environment, run from the repository root:

```powershell
python .github/scripts/check_paper_policy.py --worktree
python .github/scripts/paper_policy_test.py
python -m unittest discover -s papers/sign_pooling_2026/replication/tests -p 'test_*.py'
```

Before committing, run `python .github/scripts/check_paper_policy.py` against the
actual staged index. It reads indexed ignore rules; unstaged exceptions cannot
make a restricted staged file pass. CI repeats these checks. Never force-add
private material; ignore rules alone do not remove already tracked files.

Run `python .github/scripts/check_paper_policy.py --artifacts <directory>` on the
wheel and source archive. Both must exclude all paper and agent workspaces.
For moves, update loaders, commands, docs and CI together, preserve data bytes
and run relevant offline checks. Record source revision, data hashes, versions,
parameters and seeds where known. Distinguish teaching examples, current smoke
validation and exact archived reproduction. Do not invent historical evidence.
