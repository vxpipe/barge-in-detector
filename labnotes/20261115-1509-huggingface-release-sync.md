# Hugging Face release sync

## Decisions and implementation

- User requested dataset/model release publishing from the canonical source repo,
  ignored local model exports after GPU rsync, and convenient HF subdirectories.
- Added `barge_in_detector.releases` with separate dataset/model modes. Existing
  versioned manifests select all artifact/data paths; HF cards and model NOTICE
  were copied unchanged from the existing release checkouts into `releases/cards/`.
  Root LICENSE supplies the MIT licence. No folder-wide upload or training imports.
- Added optional `release` dependency group via `uv add --group release`; lockfile
  updated. No Torch dependency is needed for publishing.
- Default dry run makes no API calls/writes. Explicit `--refresh-hashes` updates
  selected hashes only; counts, evaluation claims and training provenance stay
  author-reviewed. `--publish` creates a parent-pinned atomic Hub commit, checks
  source publication and dataset bytes, and checks the model’s dataset pin exists.
- Remote files omitted from the manifest are preserved. This avoids deleting
  remote guidance/configuration, but obsolete weights/shards need explicit cleanup.
- Added ignored symlinks `huggingface/model` and `huggingface/dataset` via
  `bin/link-hf-repos`. The helper requires existing sibling Git checkouts and
  refuses replacement of local paths. Direct API uploads do not update the clones.
  Source development remains here and GPU exports remain under ignored `models/`.
- Updated AGENTS.md’s previously added manual-sync note, README and docs/releases.md.
  The source pin is the actual published training/data code, not necessarily the
  commit adding release tooling; new model authors must retain actual provenance.

## Verification and barriers

- Tests first: initial focused release tests failed because the module did not
  exist. Implemented manifest selection, checksums, path/symlink rejection,
  immutable pins and mocked Hub commit tests. Link helper tests likewise failed
  before implementation, then passed.
- Dataset dry run initially exposed an incorrect language-manifest mapping.
  Added a failing regression test, then mapped generation/factors/languages.yaml
  to metadata/languages.yaml. Existing local and released hashes match exactly.
- Plain `python` was unavailable; used project-managed `uv run python` for edits.
- `uv run ruff format --check`: 81 files already formatted.
- `uv run ruff check`: all checks passed.
- `uv run pytest`: 133 passed, one integration test deselected.
- Both existing release dry runs pass: dataset-pilot.json selects ten files,
  model-e5-small-run1.json selects eight files including 470,641,664-byte weights.
  Existing artifact/data hashes validate unchanged.
- `git check-ignore` confirms models/e5-small-run1/model.safetensors and both
  workspace links are ignored. Helper runs successfully on the existing clones.
- Consulted official Hub upload/create_commit documentation. Hub writes are
  tested with injected/mocked clients; no live publish, GPU usage, data generation,
  model retraining, commits or pushes were performed. No latency run needed:
  this checkpoint changes release tooling, not the model or inference export.

## Superseded local-link approach

The owner clarified that repository links must be preserved online and portable
across machines. Replaced ignored symlinks with real tracked Git submodules and
removed the link helper/tests. See [the correction checkpoint](20261115-1520-portable-hf-submodules.md).
The original 133-test evidence above describes the earlier implementation; the
correction records the current suite and submodule verification.
