# Project conventions for Claude Code

## Notebook workflow (jupytext)

Every `.ipynb` in `notebooks/` is paired with a `.py:percent` file of the same
name (see `jupytext.toml`). Division of responsibility:

- Claude edits `notebooks/<name>.py` only — never hand-edits a `.ipynb`
  cell directly. Claude *does* sync and execute (see below); it just never
  bypasses the `.py` as the source of truth.
- Cells are separated by `# %%`. Use `# %% [markdown]` for markdown cells.
  Keep cells short and single-purpose.
- If a notebook in `notebooks/` doesn't have a paired `.py` yet, ask before
  creating one (`jupytext --set-formats ipynb,py:percent <file>.ipynb`)
  rather than assuming — the user may not want it paired.

## Syncing and verifying (Claude does this)

After editing `notebooks/<name>.py`, Claude should:

1. Sync the change into the paired notebook:
   `jupytext --sync notebooks/<name>.py`
2. Execute it headless, full top-to-bottom, to regenerate real outputs and
   surface real errors:
   `jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb`
3. Read the executed notebook's actual output/traceback before reporting
   back — never claim a change "should work" without having run it this way.
4. If execution fails, fix the `.py` and repeat from step 1. Don't hand-edit
   the `.ipynb` to patch around an error.

Note: a notebook that only works with cells run out of order is a bug to fix,
not a detail to preserve — the headless full run is the correctness bar.

## What Claude should NOT do

- **Never read `.ipynb` files — not via the Read tool, not via `NotebookRead`,
  not via `cat`/`less`/`head`, not for any reason.** All verification comes
  from the stdout/stderr of the `nbconvert` command itself: a failed cell
  raises its traceback there, which is enough to diagnose and fix. If you
  genuinely need to know what a specific cell currently outputs, ask the user
  rather than opening the file — the raw JSON (base64 images, execution
  metadata) is expensive and unnecessary for the `.py`-based workflow.
- Do not `git add`, commit, or push `.ipynb` (or `.py`) files. The user
  reviews and commits after checking the synced/executed result.
- Do not modify `.gitignore` notebook-related entries.
- Before syncing, check for a notebook edited more recently than the last
  sync (`git status`, `git diff notebooks/<name>.ipynb`) — if the user may
  have edited it interactively in Jupyter since the last sync, flag it
  rather than syncing over it silently. Checking git status/diff timestamps
  this way doesn't require reading the notebook's content.

## Version control

- Both `notebooks/*.py` and `notebooks/*.ipynb` are tracked in git.
- The `.py` is what Claude edits and what the user should review for logic
  diffs; the `.ipynb` is synced/executed by Claude but committed and pushed
  by the user, after they've spot-checked the outputs.

## Research-analysis conventions

- Prefer simple, interpretable methods first; justify before reaching for
  ensembles/deep models.
- Flag data leakage, class imbalance, multicollinearity, or multiple-comparisons
  risk as soon as they're visible in the data or pipeline structure — don't wait
  to be asked.
- At decisions that depend on domain judgment (missing-data handling, feature
  inclusion, effect-size thresholds), surface the decision and the tradeoffs
  rather than picking silently.
