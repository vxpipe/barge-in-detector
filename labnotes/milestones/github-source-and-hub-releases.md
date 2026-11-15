# GitHub source and Hugging Face releases

## Runnable outcome

GitHub is the canonical development workspace for generation and model work.
HF model contains model release artifacts; HF dataset contains dataset release
files. Source and data copies stay in GitHub; model weights stay gitignored.
Prerequisites: current ownership decision in AGENTS.md, the existing dataset and
model releases, and the earlier repository-split notes retained as history.

## Checkpoints and acceptance gates

- [x] Restore training, inference, evaluation, benchmarks and model tests here.
- [x] Consolidate model/dataset labnotes and active milestones in this repository.
- [x] Keep source dataset copies and optional model dependencies with uv.lock.
- [x] Update source instructions and check the combined 118-test suite.
- [ ] Publish canonical source code to GitHub before trimming HF code trees.
- [ ] HF dataset main contains only dataset files, card, licence and release metadata.
- [ ] HF model main contains only weights/tokenizer/config, card, licence and provenance.
- [ ] Verify unchanged artifact/data hashes, loading and exact code/dataset release pins.
- [ ] Point the review service back to this canonical source checkout.

Use ordinary additive/deletion commits; preserve prior histories. No history
rewrite, branch deletion or force-push. Release-only AGENTS files point to this
repo for development and labnotes instructions. Existing classifier-quality,
calibration and reference-CPU acceptance gates remain open.

Evidence: [GitHub release-boundary labnotes](../20261115-1418-github-release-boundaries.md).
