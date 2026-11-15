# GitHub source and release boundaries

## Owner decision

Supersedes the earlier source split: GitHub owns all generation/training code,
tests, docs, benchmarks, labnotes and milestones. Keep dataset copies here.
HF model owns model release artifacts only; HF dataset owns dataset releases
only. Small cards/licences/AGENTS release guidance/provenance are release metadata.
HF development scripts, test suites, helpers and labnotes move back here.

## Implementation and evidence

- Restored barge_in_detector/training (data loading, train/evaluate/infer/metrics),
  experiments and four model tests. Optional train dependency group/uv.lock keeps
  generation-only installs lightweight. Shared CSV IO comes from splitting.py.
- Inference defaults to the released HF model; benchmarks default to the ignored
  local models/e5-small-run1 cache. Dataset downloads resolve immutable commits.
- Copied original model reports into reports/e5-small-run1 without weights.
- Preserved all model and dataset labnotes here. Where historical repository-split
  records differ, retain both with an explicit historical heading. Existing scopes
  are superseded rather than silently rewriting historical evidence.
- Combined unit suite: 118 passed, one integration test deselected (9.23 s).
  Ruff format and lint pass. No GPU rental, model retraining or dataset regeneration.
- Source will be published before deleting HF code. Deletions are ordinary Git
  commits; earlier history remains recoverable. Weights/data contents stay intact.

Publication checks and commit references follow after completion.

## Verified publication

- source_commit: `836de6924c3da68cd871e6e08da34291e9778432`.
- model_revision: `e5336e80d2829cab9a78842e0a807ac3174c5914`.
- dataset_revision: `9e914ce8b50aa12ebd3fe212bb183e09a1dee370`.
- Source published to GitHub before HF deletions. All original histories retained.
- Current HF model tree has 11 artifact/config/card/provenance files; no scripts, reports, labnotes or data. HF dataset tree has 13 data/card/metadata/config files; no development scripts.
- Anonymous Hub CPU load reproduces stop probability 0.0134437. Model/tokenizer and dataset file hashes unchanged.
- Release-only dataset loads through datasets.load_dataset: 2640/333/331 plus challenge 98. The source model loader resolves the new dataset revision and loads it correctly.
- Tiny CPU train/save/reload/evaluate smoke passed after source relocation. Default source inference CLI loads the HF model correctly. Combined 118 tests, Ruff format/lint passed.
- Source snapshots of release metadata are under releases/. Model weights remain ignored and absent from GitHub Git history.
- Review UI now runs from this canonical GitHub checkout in tmux review-ui on the existing Tailscale endpoint; 98 binary reviews remain intact (50 backchannel, 48 barge-in).
- No GPU usage, model retraining or data regeneration. Existing classifier-quality and reference-CPU gates remain open.
