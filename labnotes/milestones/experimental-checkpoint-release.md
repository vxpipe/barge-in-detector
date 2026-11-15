# Experimental checkpoint release

## Runnable outcome

A self-contained Hugging Face model repo that loads the first checkpoint on CPU,
trains/evaluates with a separate versioned dataset, and records honest limitations.
Prerequisites: dataset HashNuke/barge-in-detector-data, original generation/training
history in that dataset repo, root README and AGENTS.md.

## Contracts and checkpoints

- [x] Copy and verify selected epoch-5 weights and tokenizer without retraining.
- [x] Move training/evaluation/metrics/benchmarks into independent model modules.
- [x] Resolve Hub dataset revisions to immutable commits; support local splits.
- [x] Copy labnotes helper; derive model AGENTS.md and establish milestone usage.
- [x] Tiny CPU train/save/reload/evaluate smoke and actual checkpoint inference.
- [x] Publish artifacts, instructions and code to the model repo.
- [x] Record dataset commit, source snapshot and artifact checksums in release.json.

## Acceptance gates

- [x] Four model unit tests, lint and formatting pass.
- [x] Model weights and tokenizer resolve as large-file artifacts on the Hub.
- [x] Published files match local release hashes; CPU inference works from files.
- [x] Model card separates synthetic and reviewed diagnostics and records failures.

This milestone does not claim production readiness, calibration or reference-CPU
latency acceptance. See repository-split labnotes for evidence.

Checkpoint evidence: [repository split labnotes](../20261115-1332-repository-split.md).

Ownership note: the earlier repository split is superseded by
github-source-and-hub-releases.md. Development and milestones now live in GitHub;
HF repositories are release-only. Historical checkpoint evidence remains valid.
