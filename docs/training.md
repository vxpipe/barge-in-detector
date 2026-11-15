# Dataset splits and model training boundary

This repository owns training data, split assignment and review. The model,
training/evaluation scripts and CPU benchmarks live in
[HashNuke/barge-in-detector](https://huggingface.co/HashNuke/barge-in-detector).
Follow that model repository README to train or evaluate, specifying the exact
revision of [this dataset](https://huggingface.co/datasets/HashNuke/barge-in-detector-data).

## Splits


`uv run python -m barge_in_detector split` writes `data/splits/train.csv`,
`validation.csv` and `test.csv` (committed) from the accepted utterances.

Every combination is spread over the splits (user decision, 2026-11-15), so
no combination is only in training and none is missing from it:

- a combination with n ≥ 2 accepted samples puts n − 1 in train and 1 in
  validation or test (with 5 samples: 4 + 1);
- one with a single sample keeps it in train;
- within each language and label, combinations take validation and test in
  turn, so both cover every language and both labels;
- which sample is held out is fixed by a hash of its id: the same data always
  gives the same split.

Deduplication of the cleaned text means no text appears in two splits. The
first dataset (3,304 utterances) splits into 2,640 / 333 / 331 (80 / 10 / 10).


## Handoff

Publish data/splits and record the dataset commit. The model scripts resolve a
Hub revision to an immutable commit before downloading data; local splits are
also supported with explicit file hashes recorded in release.json. The model
repo owns PyTorch/transformers dependencies. No GPU is needed for generation,
splitting, annotation or dataset inspection.

Historical training commands in labnotes describe the layout at the time of
the first run. Current commands and model-specific milestones are in the model
repo. This separation changes ownership, not the existing labels or weights.
