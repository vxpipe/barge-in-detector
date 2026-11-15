# Hugging Face repository split

## Decision and ownership

Owner decision: model artifacts plus training/evaluation/benchmark scripts belong
in HashNuke/barge-in-detector (model). Training data, generation/split/review tools
and all historical labnotes belong in HashNuke/barge-in-detector-data (dataset).
Existing GitHub source remote remains. No model artifacts enter its Git history.
Both repos have scoped AGENTS.md, milestone indexes and bin/create-labnotes.
Owner approved the two HashNuke repo names. Existing authenticated login used;
no credential values are printed or stored in project files.

## Implementation and evidence

- Copied model artifacts and code to a sibling model checkout, verified training
  on a tiny local CPU model (train, save, reload, evaluate) and actual checkpoint
  inference reproducing p=0.0134437 for stop before removing original scripts.
- Dataset splitter moved to barge_in_detector/splitting.py; CLI and focused tests
  updated. Dataset repo has no torch/transformers dependency group.
- Model dataset loader resolves Hub references to immutable commits and records
  the revision; supports local split paths. Four focused model tests cover CSV
  IO, missing splits, immutable download revision and evaluation metrics.
- Weights/config/tokenizer are unchanged. Full original run source manifest is
  retained with model reports; original run predates this repository separation.
- Source export comparison: all 12,414 rows unchanged except dropped split
  column. Accepted splits remain 2,640/333/331 and contain 3,304 unique texts.
- Root dataset card excludes rejected samples from training configs; challenge
  has a separate config. All 98 owner reviews are binary, after authorized
  reconciliation; they match the frozen expected labels.
- Both repos use MIT licensing, consistent with the required permissive release;
  base-model card verified MIT and attribution retained in model NOTICE.
- Small JSON metadata uses plain Git in the dataset; model weights/tokenizer use
  Hub large-file storage. Working logs, environments, local DBs and caches excluded.

## Verification and publication

Record final commits and Hub artifact checks below. The migration does not
change model accuracy or satisfy uncompleted model acceptance gates.

## Publication outcome

Published public repos HashNuke/barge-in-detector (model) and
HashNuke/barge-in-detector-data (dataset). Active local checkouts are the
sibling directories barge-in-detector-model and barge-in-detector-data. Original
GitHub source changes are committed in four coherent local checkpoints; its
history was not rewritten and model weights remain excluded. Dataset Hub
publication uses large-file storage for the 17 MB source CSV, preserving its
checksum. Standard dataset loading verifies all splits and the challenge.
Anonymous model download and CPU inference reproduce the checkpoint output.
Both repos contain scoped AGENTS.md, executable bin/create-labnotes and milestone
indexes. Model release.json pins dataset dda7115ace1fe7b07c49139860131f39910521ad;
standalone model code published at 9aa77e5ff33f479ec5559f3f65bcfc092e5f86ab.
Dataset 114 tests and model 4 tests pass, with Ruff formatting/lint clean.
Coverage and classifier-quality work remain open. No new GPU use.

## Earlier HF dataset release record (historical)

# Hugging Face repository split

## Decision and ownership

Owner decision: model artifacts plus training/evaluation/benchmark scripts belong
in HashNuke/barge-in-detector (model). Training data, generation/split/review tools
and all historical labnotes belong in HashNuke/barge-in-detector-data (dataset).
Existing GitHub source remote remains. No model artifacts enter its Git history.
Both repos have scoped AGENTS.md, milestone indexes and bin/create-labnotes.
Owner approved the two HashNuke repo names. Existing authenticated login used;
no credential values are printed or stored in project files.

## Implementation and evidence

- Copied model artifacts and code to a sibling model checkout, verified training
  on a tiny local CPU model (train, save, reload, evaluate) and actual checkpoint
  inference reproducing p=0.0134437 for stop before removing original scripts.
- Dataset splitter moved to barge_in_detector/splitting.py; CLI and focused tests
  updated. Dataset repo has no torch/transformers dependency group.
- Model dataset loader resolves Hub references to immutable commits and records
  the revision; supports local split paths. Four focused model tests cover CSV
  IO, missing splits, immutable download revision and evaluation metrics.
- Weights/config/tokenizer are unchanged. Full original run source manifest is
  retained with model reports; original run predates this repository separation.
- Source export comparison: all 12,414 rows unchanged except dropped split
  column. Accepted splits remain 2,640/333/331 and contain 3,304 unique texts.
- Root dataset card excludes rejected samples from training configs; challenge
  has a separate config. All 98 owner reviews are binary, after authorized
  reconciliation; they match the frozen expected labels.
- Both repos use MIT licensing, consistent with the required permissive release;
  base-model card verified MIT and attribution retained in model NOTICE.
- Small JSON metadata uses plain Git in the dataset; model weights/tokenizer use
  Hub large-file storage. Working logs, environments, local DBs and caches excluded.

## Verification and publication

Record final commits and Hub artifact checks below. The migration does not
change model accuracy or satisfy uncompleted model acceptance gates.

The Hugging Face dataset checkout starts a clean release history from the
Hub initializer. Full historical labnotes are retained. GitHub source commits
remain in the original checkout. The large source CSV is stored via Hub
large-file storage here with identical contents; small split CSVs remain plain Git.

## Verified release

- Data release dda7115ace1fe7b07c49139860131f39910521ad published in two
  coherent commits: generation/splits, then owner reviews and historical notes.
- Standard datasets.load_dataset verifies train/validation/test counts
  2,640/333/331 and separate challenge 98. Model loader verifies immutable
  revision and file hashes. Source-export large-file SHA256 matches originals.
- Dataset checks: 114 tests passed, one integration test deselected (9.68 s);
  Ruff formatting and lint passed. No torch dependency is needed here.
- Model artifacts and standalone code are published in HashNuke/barge-in-detector;
  model code release 9aa77e5ff33f479ec5559f3f65bcfc092e5f86ab pins the data revision.
- Anonymous model download/CPU inference reproduces the known stop failure.
  This validates packaging, not classifier quality. Coverage audit remains open.
- Existing GitHub checkout has four small local source commits. Active Hugging
  Face checkouts are sibling barge-in-detector-data and barge-in-detector-model.
  Historical GitHub commits were not rewritten. Both helpers retain executable mode.
