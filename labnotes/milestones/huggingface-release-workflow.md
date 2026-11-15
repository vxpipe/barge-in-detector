# Direct Hugging Face release workflow

## Runnable outcome

Publish datasets and local GPU model exports from this canonical checkout,
without requiring staging in separate clones. Existing sibling clones remain
accessible through tracked Git submodules with portable HTTPS URLs and commit pins.

Prerequisites: [repository ownership](../../AGENTS.md#repository-ownership),
[completed source migration](github-source-and-hub-releases.md), and the existing
versioned manifests under `releases/`. This checkpoint implements packaging and
publishing tools; it does not create a new model or dataset release.

## Contracts and acceptance gates

- [x] Explicit dataset/model commands; offline dry run is the default.
- [x] Manifest-selected files, checksum validation, exact code/dataset pins;
  scripts, local secrets, optimizer state and checkpoints excluded by path rules.
- [x] Atomic Hub commit with observed parent revision; no implicit remote deletion.
- [x] GitHub source publication check and byte comparison for dataset releases.
- [x] Ignored `models/` accepts GPU exports; release manifests/cards stay versioned.
- [x] Track HF model/dataset submodules with portable URLs and immutable gitlinks;
  preserve independent sibling clones and skip LFS downloads during initialization.
- [x] Focused offline tests and existing model/dataset dry runs pass.
- [x] Ruff format/lint and full default pytest suite pass.
- [x] Document authentication, rsync, preparing manifests, publishing and limitations.

No live publication or new benchmark is required for this tooling checkpoint.
A future release must run its quality/latency gates and record returned Hub
commit SHAs; no inference/export behavior changes in this checkpoint.

Evidence: [release-sync labnotes](../20261115-1509-huggingface-release-sync.md) and
[portable submodule correction](../20261115-1520-portable-hf-submodules.md).
