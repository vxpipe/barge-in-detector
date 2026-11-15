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

The completed owner-review audit and first character-baseline comparison are in
[Review follow-up](review-followup.md), including the coverage gaps and next
training checkpoint. Reproduce without GPU or Gemini calls:

```shell
uv run python -m experiments.audit_reviewed_errors
uv run --group baseline python -m experiments.compare_character_baseline
```

Frozen E5 experiment artifacts are under `models/frozen-e5-run1/` locally;
versioned results are under `reports/frozen-e5-run1/`. The encoder uses the
original pretrained revision, query prefix, masked mean pooling, L2 normalization
and the 64-token total budget; a validation-selected logistic head provides the
uncalibrated probability. Run downloaded-artifact inference and CPU measurement:

```shell
uv run --no-sync python -m barge_in_detector.training.frozen_e5 "please wait"
uv run --no-sync python -m experiments.benchmark_frozen_e5
```

This candidate did not improve the 98-case score or meet the model p99 gate.
The user's local machine is the designated CPU for this measurement; whole-path
latency still requires Vxpipe integration.

The subsequent character-plus-word candidate and all comparisons are documented
in [Model improvement results](model-improvement-results.md). Its local portable
artifact uses only the optional baseline dependencies for CPU inference:

```shell
uv run --group baseline python -m barge_in_detector.training.lexical "stop"
```

It scores 90/98 on the owner-reviewed development set and meets the measured
model-path CPU budget, but retains false stops and multilingual synthetic
regressions. It has not replaced the published model.

The additive-data experiment fine-tunes E5 on all 4,918 combined training rows
and compares 27 sparse configurations on the same validation/test splits.
See [E5 and lightweight results](e5-phrase-results.md). The E5 candidate retains
PyTorch safetensors at the HF repository root, with FP32 ONNX at `onnx/model.onnx`
and dynamic INT8 ONNX at `onnx/model_int8.onnx`. Release tag `v0.2.0` identifies
the combined-data model; `v0.1.0` identifies the original baseline. Exact source,
dataset and artifact pins are in `releases/model-v0.2.0.json`. All training seeds
remain local; historical versions are accessed through Git tags/history.

CPU export verification is separate from training and uses the export group:

```shell
uv sync --group export
uv run --no-sync python -m experiments.verify_e5_export \
  --run huggingface/model --fp32-file model.onnx --int8-file model_int8.onnx \
  --reports reports/e5-phrase-run1 --benchmark
```

Benchmark verification scores INT8 at serving batch size one because dynamic
quantization can depend on batch composition. Scores, p50/p95/p99 and artifacts
are recorded in the reports directory; no research reports enter HF staging.

## Releases

Commit and publish source on GitHub first. Publish only model artifacts/cards/
provenance to HF model, and dataset files/cards/metadata to HF dataset. Record
exact code and dataset commits in release metadata; verify file hashes and public
loading. Development AGENTS, labnotes, milestones and bin/create-labnotes stay here.
