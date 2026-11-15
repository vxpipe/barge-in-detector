# Barge-in detector

**Give acknowledgments room. Give interruptions the floor.**

[![GitHub source](https://img.shields.io/badge/GitHub-source-181717?logo=github&logoColor=white)](https://github.com/vxpipe/barge-in-detector)
[![Hugging Face model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-model-FFD21E)](https://huggingface.co/HashNuke/barge-in-detector)
[![Hugging Face dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-dataset-FFD21E)](https://huggingface.co/datasets/HashNuke/barge-in-detector-data)
[![MIT licence](https://img.shields.io/badge/license-MIT-3DA639)](https://github.com/vxpipe/barge-in-detector/blob/main/LICENSE)

Voice agents hear callers say “uh-huh”, “yes”, and “okay” while they are still
speaking. Those acknowledgments should let the agent continue. Corrections,
questions, and requests to stop should give the caller the floor.

Barge-in detector is an open-source, multilingual, text-only project exploring
that distinction with synthetic data and a learned classifier that runs on CPU.

## Built for overlapping speech

The detector considers the caller's utterance **while the agent is talking**.
Speech after the agent finishes belongs to the voice runtime.

| Caller intent | Label | Intended runtime response |
| --- | --- | --- |
| Acknowledge: “mm hmm” | `backchannel` | Keep speaking |
| Request a pause: “please stop for a moment” | `barge_in` | Stop and listen |
| Change direction: “yes yes but change the time” | `barge_in` | Stop and listen |

- **Text in, probability out.** Use cleaned ASR transcripts, including interim
  revisions, with a first-64-token budget.
- **Multilingual data.** Fourteen languages, including English, Hindi, Kannada,
  and Tamil, with native scripts and English code-switching.
- **CPU inference.** A fine-tuned multilingual encoder with measured cold and
  warmed inference timings.
- **A reproducible research workflow.** Deterministic generation criteria,
  versioned cleaning/splits, owner review, training, and evaluation live together.

> **Experimental model.** Reviewed results are development evidence, and
> probabilities are not calibrated. Fresh human evaluation and Vxpipe
> whole-path latency remain open acceptance gates.

The [E5 phrase-coverage release](https://huggingface.co/HashNuke/barge-in-detector)
includes FP32 PyTorch, FP32 ONNX and INT8 ONNX, trained on the combined original
and supplement dataset. See [matched results and limits](docs/e5-phrase-results.md).
The root checkpoint is the selected combined-data model. Pin tag `v0.2.0` or a
full commit SHA to reproduce this release; `v0.1.0` preserves the original baseline.

## Try the model on your CPU

This CPU quick start uses Linux, Python 3.14, and [uv](https://docs.astral.sh/uv/).
The model downloads
from Hugging Face on the first call; you do not need a GPU or a Gemini key.

```shell
git clone https://github.com/vxpipe/barge-in-detector.git
cd barge-in-detector
uv sync
uv pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers==5.23.0 huggingface-hub==2.4.0
uv run --no-sync python -m barge_in_detector.training.infer "please stop for a moment"
```

The command returns `p_barge_in` and a label at the diagnostic 0.5 threshold.
Keep the model and tokenizer in memory when serving repeated requests. Input
must already follow the [transcript cleaner](docs/transcript-format.md):
lowercase, punctuation removed, native scripts optionally mixed with English.
`--no-sync` retains the CPU wheel installed above.

## Explore the data and review examples

The repo includes **3,304 accepted synthetic utterances**, split into 2,640
training, 333 validation, and 331 test examples. A separate **98-example English
challenge** has been owner-reviewed. Rejected samples stay in the source export
for audit, and are excluded from training.

A separate phrase-coverage supplement adds 3,437 accepted examples while keeping
the original data intact. The combined experiment uses 4,918 train / 915 validation
/ 908 test examples across all 14 languages. See the [E5 and lightweight comparison](docs/e5-phrase-results.md)
for the new candidate, its FP32/INT8 exports, CPU measurements and evaluation limits.

```shell
uv run python -m barge_in_detector check-catalogue
uv run python -m barge_in_detector.review
```

Open **http://127.0.0.1:8765** to review utterances. The UI hides proposed labels,
saves decisions to disk, and supports keys **1** for backchannel and **2** for
barge-in. [Review instructions](docs/review-ui.md) cover SSH and Tailscale access.

For hosted data without cloning the source repo, start with the
[HF dataset quick start](https://huggingface.co/datasets/HashNuke/barge-in-detector-data#load-the-data).

## Generate, train, and evaluate

| Task | Guide |
| --- | --- |
| Generate or extend the synthetic dataset | [Generation guide](docs/generation.md) |
| Understand columns, labels, and storage | [Data format](docs/data-format.md) |
| Clean caller transcripts consistently | [Transcript format](docs/transcript-format.md) |
| Train, evaluate, and benchmark on CPU | [Training guide](docs/training.md) |
| Inspect recorded results and failures | [First-run reports](reports/e5-small-run1/) |
| Compare the additive-data E5 and sparse candidates | [Current experimental results](docs/e5-phrase-results.md) |

Generation uses an environment Gemini API key. Training dependencies are
optional: `uv sync --group train`. CPU model evaluation runs locally; GPU
training follows the single RTX 5090 / `gpubox` policy in [AGENTS.md](AGENTS.md).

```shell
uv run --group train python -m barge_in_detector.training.train --out models/run2 --splits data/splits
uv run --group train python -m barge_in_detector.training.evaluate --model models/run2 --splits data/splits --split test
```

Model outputs remain gitignored. Publish selected weights to the HF model repo
and data releases to the HF dataset repo, retaining code/data revision pins.

## What the first release tells us

| Measurement | Result |
| --- | --- |
| Synthetic test accuracy, 331 examples | 100% |
| Reviewed English challenge accuracy, 98 examples | 73.5% |
| Short interruptions detected in the challenge | 5 / 30 |
| Warm CPU latency, typical utterance | 13.1 ms median / 25.3 ms p99 |
| Warm CPU latency, full 64-token budget | 26.8 ms median / 37.9 ms p99 |
| Fresh Python process to first label | 4.22 s median |

Timing uses four virtual AMD EPYC Genoa cores, four threads, eager fp32 PyTorch,
batch one. Warm figures include tokenization through label output. Cold start
includes imports and local model loading; OS file cache may be warm. HTTP,
network, queueing, and the Vxpipe boundary are excluded. These are local results,
not reference-CPU acceptance measurements; ONNX/int8 is unmeasured.

The synthetic test score does not establish real-world accuracy. Criteria overlap
between splits, and template/semantic leakage remains unaudited. The challenge
is now a development/regression set. Fresh human-reviewed English/Hindi evidence,
better short-interruption coverage, calibration, and latency work remain ahead.

## Help improve it

Good next contributions include reviewing fresh English/Hindi examples,
strengthening short-interruption coverage, auditing near duplicates, and comparing
a character n-gram baseline. Read [AGENTS.md](AGENTS.md) and the
[milestones index](labnotes/milestones/index.md) before starting. Record research
and implementation checkpoints with `bin/create-labnotes`.

```shell
uv run ruff format --check
uv run ruff check
uv run pytest
```

External integration tests are excluded by default.

## One project, three repositories

| What you want to do | Start here |
| --- | --- |
| Generate data, train, evaluate, benchmark, or contribute | [GitHub source](https://github.com/vxpipe/barge-in-detector) |
| Download a released model and tokenizer | [Hugging Face model](https://huggingface.co/HashNuke/barge-in-detector) |
| Load a versioned dataset | [Hugging Face dataset](https://huggingface.co/datasets/HashNuke/barge-in-detector-data) |

GitHub owns all code, tests, docs, labnotes, and milestones, plus a dataset copy.
Hugging Face hosts model and dataset releases. Release metadata links artifacts
to exact source and dataset revisions. The project is available under the
[MIT licence](https://github.com/vxpipe/barge-in-detector/blob/main/LICENSE).

To publish dataset or model releases directly from this checkout, follow the
[release workflow](docs/releases.md). Local GPU exports stay under ignored
`models/`; release manifests and HF cards are versioned under `releases/`. HF repos are tracked as submodules at
`huggingface/model` and `huggingface/dataset`; initialize them without large model
downloads using `GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive`.
