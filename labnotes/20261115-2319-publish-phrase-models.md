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
