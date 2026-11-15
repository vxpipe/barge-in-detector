# Publishing Hugging Face releases

Develop here and publish directly through the Hub API. The HF repos are tracked as submodules for
portable inspection/staging; these commands do not read or update their working
trees or pinned commits. Existing sibling clones remain optional. This replaces manual file copying for routine releases. Publishing remains
an explicit action; there is no background or CI sync.

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

To preserve an existing root model, set `model_subdirectory` in the new manifest
to `candidates/<run>/checkpoint`. Select fully qualified candidate paths from
an artifact root containing `candidates/`: checkpoint/seed safetensors and
tokenizers, FP32/INT8 ONNX, ONNX metadata, candidate card, licence and provenance.
The root card must point readers to the new subfolder. Previous release metadata
can be retained at `releases/<run>.json`. This does not move or replace root weights.

Both commands upload exactly the validated manifest-selected files, plus the
versioned card, root MIT `LICENSE`, optional card-directory `NOTICE`, and manifest
(as `release.json` or `dataset-release.json`). Symlinks and paths outside the
release allowlist are rejected. No repository-wide folder upload is used.

Uploads use a single Hub commit with its parent pinned to the observed remote
main commit. A concurrent update causes failure rather than silently overwriting
it; review and retry. Repeated identical uploads rely on the Hub’s handling of
unchanged content. No release tags or new repositories are created.

This is an additive file sync: selected paths are replaced, and other remote
files are preserved, including `.gitattributes` and release-only guidance. When
changing artifact layouts, explicitly inspect and remove obsolete remote weights
or shards separately. Existing release metadata and model weights are not edited
by dry runs. Never change local release inputs while an upload is running.

These commands verify packaging, hashes and revision pins. They do not train,
calibrate, evaluate, benchmark or certify model quality. Run the relevant model
and reference-CPU acceptance gates before making new performance claims.

Implementation uses the official [Hub commit API](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#create_commit).
