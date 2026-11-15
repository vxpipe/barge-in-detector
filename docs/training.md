# Training, evaluation and dataset splits

All generation and model code lives in this GitHub repo. Hugging Face hosts
model/data releases; it does not own executable development code. See README.md
and AGENTS.md for ownership. Model dependencies are optional, keeping generation
and review lightweight.

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



## CPU-only model setup

```shell
uv sync
uv pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers==5.23.0 huggingface-hub==2.4.0
uv run --no-sync python -m barge_in_detector.training.infer "please stop for a moment"
```

CPU commands use --no-sync to retain the explicitly installed CPU wheel. The
inference CLI downloads the released model by default and always runs on CPU.
Its input must already follow cleaner rules; the token budget is 64.

## GPU training

```shell
uv sync --group train
uv run --group train python -m barge_in_detector.training.train --out models/run2 --splits data/splits
```

Defaults: intfloat/multilingual-e5-small, two-label head, inverse-frequency
weighted cross-entropy, AdamW, 3e-5 learning rate, 10% warm-up, batch 32, five
epochs, seed 20261115. Select the epoch on validation macro F1 only.
Always use the single RTX 5090 / gpubox policy in AGENTS.md.

Instead of local --splits, use --dataset-repo HashNuke/barge-in-detector-data
and --dataset-revision COMMIT. The loader resolves references to immutable commits
before download. Local split runs must retain input file hashes and code revision.
Model output/checkpoint directories are ignored by Git.

## Evaluation and benchmarks

```shell
uv run --no-sync python -m barge_in_detector.training.evaluate --model models/run2 --splits data/splits --split validation
uv run --no-sync python -m barge_in_detector.training.evaluate --model models/run2 --splits data/splits --split test
uv run --no-sync hf download HashNuke/barge-in-detector --local-dir models/e5-small-run1
uv run --no-sync python -m experiments.evaluate_english_challenge --dataset-dir .
uv run --no-sync python -m experiments.benchmark_cpu --threads 4 --splits data/splits
uv run --no-sync python -m experiments.benchmark_cpu_cold --threads 4
```

Report false/missed interruptions, per-language evidence, prefixes and calibration.
Threshold default 0.5 is diagnostic, not calibrated. The challenge is now a
regression set; never train on it. Fresh reviewed English/Hindi evidence is needed
for independent evaluation. CPU benchmarks include tokenization through label,
batch one warmed, and separately cold process through first label; they exclude
HTTP/network/queue/Vxpipe boundary. Original first-run evidence is retained under
reports/e5-small-run1/ and historical labnotes.

## Releases

Commit and publish source on GitHub first. Publish only model artifacts/cards/
provenance to HF model, and dataset files/cards/metadata to HF dataset. Record
exact code and dataset commits in release metadata; verify file hashes and public
loading. Development AGENTS, labnotes, milestones and bin/create-labnotes stay here.
