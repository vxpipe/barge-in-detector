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
