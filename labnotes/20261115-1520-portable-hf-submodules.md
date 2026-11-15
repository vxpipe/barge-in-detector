# Portable HF submodules

## Correction

The owner wanted portable repository links preserved on GitHub. The previous
ignored symlinks were local conveniences and did not satisfy that requirement.
Replaced them with real Git submodules at huggingface/model and huggingface/dataset.
Removed the untracked link helper and its two tests. Existing sibling checkouts
are untouched. No history rewriting, commits or pushes.

## Implementation and evidence

- `.gitmodules` records public HTTPS URLs for the model and dataset repositories.
  `git submodule add` stages that file and two mode-160000 gitlinks automatically.
- Model initial pin: 1994527ac2dbd91d7b858f3f94349914ded5ae6a.
- Dataset initial pin: a2cc9dd3349ddcec0426bfa873fba6c1bce1614f.
- Used GIT_LFS_SKIP_SMUDGE=1 during initialization; model weights remain pointers
  until an explicit LFS pull. Neither submodule path is gitignored now.
- Updated AGENTS.md, README, release workflow and milestone to describe portable
  initialization, detached HEADs, explicit weight downloads and release pin updates.
- Direct API publishing remains available and does not implicitly change gitlinks.
- `git submodule status` shows both initialized at their recorded commits;
  each submodule working tree is clean. `git ls-files --stage` confirms gitlinks.
- GitHub will render the repository/revision links after the changes are committed
  and pushed; this task does not publish them automatically.

Final format/lint/default-suite checks recorded below after completion.

- `uv run ruff format --check`: 86 files already formatted.
- `uv run ruff check`: all checks passed.
- `uv run pytest`: 131 passed, one integration test deselected.
- `git diff --check` and `git diff --cached --check` passed. The only staged paths
  are `.gitmodules` and the two submodule gitlinks created by `git submodule add`.
