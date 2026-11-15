# Publish phrase models

Owner explicitly authorized publication of the fine-tuned FP32 and INT8 models,
the updated dataset, and updates to both HF Git submodules. Earlier authorization
also requested FP32 ONNX; all three formats and all seeds are retained. No new
training, generation or GPU rental is needed. The GPU was already destroyed.

The model was trained on combined original and supplement training rows:
2,640 + 2,278 = 4,918; validation 333 + 582 = 915; test 331 + 577 = 908.
Keep original data/weights available. Add dataset configurations for the standalone
supplement and combined splits; the original utterances configuration is unchanged.

The release command previously allowed only root models and the original dataset
paths. Six focused tests failed for additive candidate/supplement paths, then
passed after narrowly extending the allowlist and supporting model_subdirectory.
Negative tests continue to reject traversal, scripts, optimizer state and private
metadata filenames. Model publication will preserve root weights and archive their
release metadata while adding candidates/e5-phrase-run1. All uploads remain
explicit and manifest-selected, with exact source/data pins and an observed
remote parent commit.

Inspected both authenticated HF repositories and the GitHub source remote.
Source main matches origin/main before publication. Reviewed 188 changed/new
development files (about 14.1 MB); scans found no known private machine identifiers,
credential-shaped strings or user home paths. Staging model hashes match verified
downloads. No operational records, scripts or datasets enter the model repository.

Validation before source publication: Ruff format/check pass; **181 tests passed,
one network integration test deselected**. Existing ONNX exporter warnings remain
documented by the training checkpoint; no weights or export behavior changed in
this publication step. Dataset dry run validates all selected files and hashes.

Publish source first, then dataset, then model pinned to the returned dataset
revision. Verify remote checksums/loadability and dataset configurations before
advancing both submodule gitlinks. Preserve local weights while reconciling the
model staging checkout; do not rewrite history or force-push.

Source snapshot published as `064ea7f` on GitHub main. Added explicit support
for manifest-selected model Git LFS attributes so nested tokenizer and ONNX
files behave correctly in Git submodule checkouts. Focused red/green test covers
that addition while continuing to reject `.git` configuration paths.

Dataset published at `02114f9cbdeed78c86dcd58ad13011425d010f8e` after source byte verification. Its submodule
was fetched and checked out at that immutable revision. Original files remain
unchanged, with separate supplement and combined configurations. Model metadata
now pins this exact dataset and the published source; reviewed 27 selected model
artifacts/metadata files, including FP32/INT8 ONNX and preserved legacy provenance.
Full completion checks after LFS support: **182 passed, one deselected**, Ruff clean.

## Owner correction: Git versions and stable paths

The owner rejected retaining `run1`/candidate directories for historical versions.
Use stable root FP32 and `onnx/` artifact paths, with Git commits and immutable
tags for releases. Only intentionally parallel model development warrants
separate model directories. Added this policy to `docs/releases.md` and AGENTS.md.
The intermediate candidate-layout publication is superseded, while its Git history
is retained. No history rewriting or force-push is used.

Prepared the selected checkpoint at stable filenames, preserving all seed
checkpoints locally. Extended the manifest tool with explicit allowlisted
remove_paths, applied atomically with uploads at a pinned parent. Five focused
tests first failed, then passed for root provenance/checksums and explicit cleanup;
unsafe paths and simultaneous upload/delete overlaps are rejected.

The intermediate remote publication passed checks for all 16 dataset inputs and
27 model files, plus downloaded CPU loads of all three inference formats.
The new layout changes filenames/metadata only; model weights are unchanged.
Plan tags v0.1.0 for the prior baseline and v0.2.0 for this selected combined-data
release, on both model and dataset repositories. Do not overwrite existing tags.

Stable-layout checks: full suite **186 passed, one integration test deselected**;
Ruff format/check pass. The verification script now accepts explicit FP32/INT8
filenames so the same parity/benchmark workflow can use the public stable paths.
Parity was rechecked using the renamed files; weight contents did not change.
Initialized the model submodule's local Git LFS filters and refreshed its index
with the verified files; no model-content commit was created by that refresh.

## Published tagged release and submodules

- Model v0.2.0: `9a227cdbdabf0f30476cb00905989f5962ea5219`.
- Dataset v0.2.0: `02114f9cbdeed78c86dcd58ad13011425d010f8e`.
- Model v0.1.0 preserves baseline `1994527ac2dbd91d7b858f3f94349914ded5ae6a`.
- Dataset v0.1.0 preserves `a2cc9dd3349ddcec0426bfa873fba6c1bce1614f`.

The current model tree has stable root FP32 and FP32/INT8 ONNX files; all 26
obsolete candidate/archive paths were removed in the same upload commit. Previous
versions remain in Git history. All 11 manifest-selected files were downloaded
from the tagged release and matched SHA-256; all three inference formats loaded
on CPU and FP32 probabilities matched within 1e-4. Both dataset/model tags resolve
to the intended commits. The initial tag check compared annotated tag-object
IDs rather than resolved commits; corrected verification uses repo_info.sha.

Both submodules are checked out at their v0.2.0 commits. Restored large model
files from verified local artifacts with hardlinks; Git LFS filters keep the
working trees clean. Other seed checkpoints remain under gitignored models/.
No additional GPU time or generation calls. Source records the new versioning
policy, exact release manifests, remote verification and the submodule gitlinks.

Evidence: `releases/published-v0.2.0.json` and
`reports/e5-phrase-run1/tagged-release-verification.json`.

## README follow-up and authorized tag correction

The owner asked whether all repository READMEs were current. Remote contents
matched the checkouts, but GitHub still led its results section with original
metrics and called ONNX unmeasured. Updated it to the published FP32/INT8 results,
corrected model-card ONNX link labels, and made dataset configuration/tag examples
explicit. No weights, data or training parameters changed. README Python snippets
parse and artifact download paths exist.

The owner explicitly authorized updating release tags. Updated AGENTS.md and the
release guide accordingly: tags can move for authorized release corrections;
full commit SHAs keep provenance and submodule pins immutable. Apply these README
corrections to v0.2.0 in both HF repositories; retain v0.1.0 unchanged. Record
documentation source separately from the actual training source/dataset pins.

All three README corrections are published. Verified each HF change contains
only README.md and its release metadata; model weights and dataset files are
unchanged. Moved both v0.2.0 tags under the explicit owner authorization, verified
resolved target commits, and retained both v0.1.0 tags. Both submodules now follow
the corrected releases with clean working trees. Added the requested AGENTS.md
checklist for README updates, tag creation/correction, remote verification and
submodule/source publication. Evidence: readme-release-verification.json under
reports/e5-phrase-run1/ and the updated releases/published-v0.2.0.json receipt.
