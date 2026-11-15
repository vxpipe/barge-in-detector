# Barge-in detector

Canonical source for generating multilingual utterances and training a CPU
classifier that distinguishes overlapping caller acknowledgments from genuine
interruptions. All code, tests, docs, research notes and milestones live here.

| Repository | Purpose |
| --- | --- |
| [GitHub source](https://github.com/vxpipe/barge-in-detector) | Generation, cleaning, splits, review UI, training, inference, evaluation, benchmarks and labnotes; also a dataset copy |
| [HF model](https://huggingface.co/HashNuke/barge-in-detector) | Released weights, tokenizer/configuration, model card, licence and provenance |
| [HF dataset](https://huggingface.co/datasets/HashNuke/barge-in-detector-data) | Released dataset files, dataset card, licence and metadata |

HF checkouts are release staging directories. Implement and test changes here,
then publish releases pointing to the exact GitHub code and dataset revisions.
Model weights and local checkpoints remain gitignored in this source repo.

## Setup and generation

```shell
uv sync
uv run python -m barge_in_detector check-catalogue
uv run python -m barge_in_detector split
uv run python -m barge_in_detector.review
```

Generation uses environment API keys; see [generation](docs/generation.md).
The dataset copy contains 3,304 accepted utterances split 2,640/333/331 and a
separate 98-example reviewed English challenge. The source export also retains
rejected samples; they are excluded from training splits. See
[data format](docs/data-format.md), [splits/training](docs/training.md) and
[review UI](docs/review-ui.md).

## Model work

Training dependencies are optional: `uv sync --group train`. Training on Vast.ai
requires one datacenter RTX 5090 below $0.65/hour and tmux session gpubox.
CPU inference/evaluation should run locally. See [training](docs/training.md)
for CPU-only setup and commands.

```shell
uv run --group train python -m barge_in_detector.training.train --out models/run2 --splits data/splits
uv run --group train python -m barge_in_detector.training.evaluate --model models/run2 --splits data/splits --split test
```

The first checkpoint is experimental: 100% synthetic test accuracy but only
73.5% on the reviewed English challenge, including 25/30 missed short
interruptions. Probabilities are not calibrated. Current fp32 CPU timing on our
four-vCPU EPYC Genoa VM is 13.1 ms median / 25.3 ms p99 for typical warmed inputs;
full-64-token p99 is 37.9 ms. Reference-CPU and Vxpipe boundary gates are unmet.
Recorded evidence is under reports/e5-small-run1/ and labnotes/.

## Work and verification

Start with [AGENTS.md](AGENTS.md) and [milestones](labnotes/milestones/index.md).
Use `bin/create-labnotes` for research/implementation checkpoints. HF repos link
back here for development and labnotes; their current releases contain no scripts.

```shell
uv run ruff format --check
uv run ruff check
uv run pytest
```

Tests exclude external integration calls by default. Released model/data
limitations and per-language evidence are documented in their HF cards.
