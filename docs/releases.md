# Publishing Hugging Face releases

Develop here and publish directly through the Hub API. The HF repos are tracked as submodules for
portable inspection/staging; these commands do not read or update their working
trees or pinned commits. Existing sibling clones remain optional. This replaces manual file copying for routine releases. Publishing remains
an explicit action; there is no background or CI sync.

## Model filenames and release versions

Owner decision, 2026-11-15: **Git history and release tags version models.** The
current HF tree contains the selected model at stable paths:

```text
model.safetensors          selected unquantized FP32 checkpoint
config.json
tokenizer.json
tokenizer_config.json
onnx/model.onnx           FP32 ONNX
onnx/model_int8.onnx       INT8 ONNX
onnx/tokenizer.json
onnx/metadata.json
provenance.json
artifacts.sha256.json
release.json
```

Publish updates to these files, then create a release tag such as `v0.2.0` at the
verified commit. The owner may authorize moving a tag to include release
corrections, including documentation updates. Verify the new target and refresh
both submodule pins after such an update. Consumers use tags to select releases;
full commit SHAs provide immutable reproduction even when a tag moves.
Previous model versions remain accessible through Git history and tags; do not
retain `run1`, `run2`, training seeds or historical candidate folders in the
current HF model tree merely to version them. Multiple model directories are
appropriate only for models intentionally being developed in parallel. Keep
experimental checkpoints locally under gitignored `models/`.

Verify each tag resolves to the intended commit before recording it. An annotated
tag object has its own SHA; use the resolved commit (`git rev-parse TAG^{commit}`
or Hub `repo_info(..., revision=TAG).sha`) for submodule and provenance pins.

Dataset configurations are separate: preserve the original dataset and the
standalone supplement alongside the derived combined splits. Tag dataset
releases too, and pin the exact training dataset commit in each model release.
Documentation-only commits do not change the model's training provenance or
dataset content pins; record the documentation source revision separately.

After verified publication, fetch and update **both** HF submodule checkouts,
retain local model weights, and commit/push the corresponding GitHub gitlink
updates. A remote HF update alone does not update the submodules.

## Portable HF subdirectories

Both HF release repositories are Git submodules. `.gitmodules` records their
public HTTPS URLs and Git tracks the exact commits at these paths:

```text
huggingface/model    HF model release repository
huggingface/dataset  HF dataset release repository
```

GitHub renders these entries as links to the pinned repositories/revisions. They
do not depend on sibling directories or local symlinks. A normal source clone
does not download submodule contents. To initialize them on any machine without
fetching large LFS model weights:

```shell
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
git -C huggingface/model status --short
git -C huggingface/dataset status --short
```

Install Git LFS before working with the model repository. When weights are needed:

```shell
git -C huggingface/model lfs pull
```

Submodules normally check out detached HEADs at the recorded commits. To work on
a release checkout, create a branch there first. Development code belongs here;
GPU exports still go into ignored `models/<run>/`.

Direct API publishing does not advance these pins. After a release, fetch and
check out the returned immutable commit in the corresponding submodule, then
stage and commit its updated gitlink in this repository when requested. For
example, with `MODEL_COMMIT` set to the returned model release SHA:

```shell
git -C huggingface/model fetch origin
GIT_LFS_SKIP_SMUDGE=1 git -C huggingface/model checkout "$MODEL_COMMIT"
git add -- huggingface/model
```

Do the equivalent for `huggingface/dataset` using the dataset release SHA. Do not
switch a submodule with uncommitted work; preserve that work first. Updating pins
never copies model weights into this repository’s Git history. Existing sibling
clones can remain on disk and are independent of these submodules.

## Setup and preview

Run from this repository root:

```shell
uv sync --group release
uv run --group release hf auth login
uv run --group release python -m barge_in_detector.releases dataset \
  --manifest releases/dataset-pilot.json
uv run --group release python -m barge_in_detector.releases model \
  --manifest releases/model-e5-small-run1.json --artifacts models/e5-small-run1
```

The default is an offline dry run: it validates inputs and prints destination
paths, sizes and SHA-256 hashes. It makes no API calls and writes nothing. Only
publishing needs authentication; use the locally saved HF login or `HF_TOKEN`.
Never put tokens in manifests, command arguments or tracked files. The release
dependency group needs neither Torch nor Transformers.

## Dataset releases

1. Generate/clean/review/export/split using the existing project tools. Never
   edit the exported CSV by hand.
2. Copy `releases/dataset-pilot.json` to a new descriptive manifest filename.
   Update release ID, schema version, counts, limitations and `code_revision`.
   `files_sha256` explicitly selects the release files, using their HF paths.
   Export/splits/evaluation CSVs and review JSONs keep their paths. The one mapping
   is local `generation/factors/languages.yaml` → HF `metadata/languages.yaml`.
3. Update the versioned HF dataset card at `releases/cards/dataset/README.md`,
   including counts, configurations and any changed evaluation claims.
4. Refresh the selected file hashes locally, then inspect the diff:

   ```shell
   uv run --group release python -m barge_in_detector.releases dataset \
     --manifest releases/dataset-next.json --refresh-hashes
   ```

   This updates only the hash mapping, preserving the selected paths. Add/remove
   paths in that mapping explicitly; a placeholder hash can be used before refresh.
   Counts and other metadata must be reviewed by the release author.
5. Publish the source/data changes to GitHub first. Set `code_revision` to that
   exact published commit. Commit the release manifest with its provenance.
6. Preview, then publish:

   ```shell
   uv run --group release python -m barge_in_detector.releases dataset \
     --manifest releases/dataset-next.json --publish
   ```

Publishing checks that the source commit is reachable from GitHub `origin/main`
and that every selected dataset file matches the bytes at that source commit.
If the remote main commit is absent locally, run `git fetch origin` and retry.
Record the returned HF commit SHA; use it as the model release dataset revision.

## Model releases from a GPU box

Rsync a final model/tokenizer export into `models/<run>/`. That entire directory
is gitignored, including weights, checkpoints and any accidentally copied logs.
For example, after setting `GPU_HOST` and `GPU_EXPORT_DIR` for your SSH setup:

```shell
mkdir -p models/run2
rsync -av -- "$GPU_HOST:$GPU_EXPORT_DIR/" models/run2/
```

1. Copy `releases/model-e5-small-run1.json` to a new descriptive filename. Update
   the base model, training/evaluation provenance, status, exact GitHub
   `code_revision`, exact HF `dataset_revision` and dataset hashes. Preserve the
   training run’s actual code and data pins, rather than substituting current main.
2. Select the exported files in `artifacts_sha256`. Supported paths cover the
   standard config/tokenizer files, safetensors (including shards/index), and
   ONNX exports. Checkpoint pickles, optimizer state, logs and scripts are excluded.
   Every selected file must exist; hashes must match before publishing.
3. Update `releases/cards/model/README.md` and `NOTICE` for this release’s model,
   licence attribution, metrics and limitations. Cards are current release inputs,
   so review them when republishing an older manifest too.
4. Refresh hashes and preview:

   ```shell
   uv run --group release python -m barge_in_detector.releases model \
     --manifest releases/model-run2.json --artifacts models/run2 --refresh-hashes
   uv run --group release python -m barge_in_detector.releases model \
     --manifest releases/model-run2.json --artifacts models/run2
   ```

5. Publish the training source to GitHub before releasing. Version the manifest
   and card here, then run the preview command with `--publish`. The command also
   verifies that the pinned HF dataset revision exists. Keep the returned model
   commit SHA in the release checkpoint labnotes for consumers to pin.

## Boundaries and limitations

Additive dataset releases may select `data/supplements/<name>/utterances.csv`,
the supplement's split CSVs and README, and `data/experiments/<name>/splits/*.csv`
plus README. Keep the original paths/configuration intact and add explicit Hub
dataset configurations for the new inputs in the card.

For intentionally parallel model development only, `model_subdirectory` may
select `candidates/<name>/checkpoint`. That support must not be used to retain
historical versions or every training seed. Normal releases use the stable root
layout above and tags. The versioned manifest still records the training run and
source hashes even when public artifact filenames stay stable.

Both commands upload exactly the validated manifest-selected files, plus the
versioned card, root MIT `LICENSE`, optional card-directory `NOTICE`, and manifest
(as `release.json` or `dataset-release.json`). Symlinks and paths outside the
release allowlist are rejected. No repository-wide folder upload is used.

Uploads use a single Hub commit with its parent pinned to the observed remote
main commit. A concurrent update causes failure rather than silently overwriting
it; review and retry. Repeated identical uploads rely on the Hub’s handling of
unchanged content. No release tags or new repositories are created.

By default, selected paths are replaced and omitted remote files are preserved.
For an explicitly requested layout cleanup, list exact obsolete paths in the
manifest's `remove_paths`. The dry run shows every deletion; publication applies
the uploads and removals in the same parent-pinned Hub commit. Cleanup paths must
pass the release allowlist and cannot overlap uploaded files. This removes files
only from the current tree; Git history and tags remain. There is no force-push or
history rewrite. Existing metadata and weights are not edited by dry runs.
Never change local release inputs while an upload is running.

These commands verify packaging, hashes and revision pins. They do not train,
calibrate, evaluate, benchmark or certify model quality. Run the relevant model
and reference-CPU acceptance gates before making new performance claims.

Implementation uses the official [Hub commit API](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#create_commit).
