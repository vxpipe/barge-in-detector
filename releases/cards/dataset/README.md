---
license: mit
language: [en, hi, kn, ta, te, ml, zh, ja, id, fil, fr, de, es, it]
task_categories: [text-classification]
tags: [synthetic, barge-in, backchannel, multilingual]
configs:
  - config_name: utterances
    default: true
    data_files:
      - split: train
        path: data/splits/train.csv
      - split: validation
        path: data/splits/validation.csv
      - split: test
        path: data/splits/test.csv
  - config_name: challenge
    data_files:
      - split: test
        path: data/evaluation/english-challenge-v1.csv
  - config_name: phrase-coverage-v1
    data_files:
      - split: train
        path: data/supplements/phrase-coverage-v1/splits/train.csv
      - split: validation
        path: data/supplements/phrase-coverage-v1/splits/validation.csv
      - split: test
        path: data/supplements/phrase-coverage-v1/splits/test.csv
  - config_name: original-plus-phrase-v1
    data_files:
      - split: train
        path: data/experiments/original-plus-phrase-v1/splits/train.csv
      - split: validation
        path: data/experiments/original-plus-phrase-v1/splits/validation.csv
      - split: test
        path: data/experiments/original-plus-phrase-v1/splits/test.csv
---

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

This repository is the **dataset release**: versioned utterances, split files,
review annotations, and metadata. It preserves the original **3,304 accepted
synthetic examples**, adds a separate **3,437-example phrase-coverage supplement**,
and provides their **6,741-example combined dataset across 14 languages**.
The separate **98-example reviewed English challenge** is retained unchanged.
Generation, cleaning, review tools, training, tests, and labnotes live in the
[GitHub source](https://github.com/vxpipe/barge-in-detector).

## Load the data

Create a Python 3.14 environment and install the datasets library with uv:

```shell
uv venv --python 3.14
uv pip install datasets
source .venv/bin/activate
```

Run this in Python (`python`) or a notebook using that environment:

```python
from datasets import load_dataset

repo = "HashNuke/barge-in-detector-data"
splits = load_dataset(repo, "utterances")
combined = load_dataset(repo, "original-plus-phrase-v1")  # New E5 training dataset
phrases = load_dataset(repo, "phrase-coverage-v1")
challenge = load_dataset(repo, "challenge", split="test")

sample = splits["train"][0]
print(sample["text_clean"], sample["label"])
```

Use `revision=DATASET_COMMIT` to pin a snapshot for training or evaluation.
The [model release metadata](https://huggingface.co/HashNuke/barge-in-detector/blob/main/release.json)
identifies the exact dataset revision used for the published checkpoint.

## What's inside

| Configuration | Split | Examples | Intended use |
| --- | --- | ---: | --- |
| `utterances` | `train` | 2,640 | Train a classifier |
| `utterances` | `validation` | 333 | Select settings and thresholds |
| `utterances` | `test` | 331 | Report synthetic held-out results |
| `challenge` | `test` | 98 | Reviewed English diagnostics and regression checks |
| `phrase-coverage-v1` | `train` / `validation` / `test` | 2,278 / 582 / 577 | Separate additive supplement |
| `original-plus-phrase-v1` | `train` / `validation` / `test` | 4,918 / 915 / 908 | Combined splits used by the new E5 fine-tune |

The original accepted dataset contains **2,220 barge-ins** and **1,084 backchannels**.
The combined dataset contains **3,879 barge-ins** and **2,862 backchannels**.
The default `utterances` configuration retains the original splits; select
`original-plus-phrase-v1` explicitly to reproduce the new training dataset.
Only accepted examples enter training splits. The challenge must stay outside
training; it is now a development/regression set.

| Field | Meaning |
| --- | --- |
| `text` | Natural caller utterance |
| `text_clean` | Lowercase text with punctuation removed; model input |
| `label` | `backchannel` or `barge_in` |
| `language` | Language code |
| `utterance_id`, `combination_id` | Synthetic example and generation criterion identifiers |

Other columns describe regional variants, code-switching, and caller intent.
The challenge has its own case IDs and prefix groups. See the
[complete source data format](https://github.com/vxpipe/barge-in-detector/blob/main/docs/data-format.md).

## Labels and the listening task

Assume the caller is speaking **while the agent is still talking**.

- **Backchannel:** acknowledgment, agreement, empathy, surprise, laughter, or a
  continuer that lets the agent keep speaking.
- **Barge-in:** a request to stop/wait, correction, disagreement, question, new
  request, early answer that makes the remaining agent turn moot, or a
  backchannel that develops into one of these.

Speech after the agent finishes belongs to the voice runtime. There are exactly
two labels; no context/ambiguous class. The target model consumes the first 64
tokenizer tokens of cleaned text. Native scripts may mix with English.

## Languages

English (US, India, Singapore), Hindi, Kannada, Tamil, Telugu, Malayalam,
Mandarin, Japanese, Indonesian, Filipino, French, German, Spanish, and Italian.
[metadata/languages.yaml](metadata/languages.yaml) records scripts and tiers.
The owner can verify English, Hindi, Kannada, and Tamil; this release includes
reviewed English diagnostics, not human-verified evaluation in every language.
Other language results remain synthetic/unverified.

## Files and provenance

- [data/splits/](https://huggingface.co/datasets/HashNuke/barge-in-detector-data/tree/main/data/splits) contains the accepted training, validation, and test CSVs.
- [data/evaluation/](https://huggingface.co/datasets/HashNuke/barge-in-detector-data/tree/main/data/evaluation) contains the frozen challenge and review decisions.
- [data/export/utterances.csv](data/export/utterances.csv) retains **all 12,414 sample
  rows**, including rejections. It is an audit export, not the training configuration.
- [dataset-release.json](dataset-release.json) records the code revision, schema,
  counts, and file checksums.
- [data/supplements/phrase-coverage-v1/](data/supplements/phrase-coverage-v1/)
  contains the separate supplement: 4,016 audit rows, including 3,437 accepted,
  576 rejected and 3 retired, plus its accepted split CSVs.
- [data/experiments/original-plus-phrase-v1/](data/experiments/original-plus-phrase-v1/)
  contains the derived combined splits, with the original split assignments
  preserved and zero exact cleaned-text overlap across train/validation/test.

Gemini generated natural text under deterministic project-selected criteria;
a criteria judge accepted or rejected the original dataset. The phrase supplement
was generated **without a judge**, as requested by the owner; deterministic code
checked format, script, deduplication and split constraints. Its labels follow
the generation criterion and remain synthetically assigned. These are utterances, not
actual customer recordings. The pinned GitHub source contains generation
configuration, cleaning rules, storage code, and historical labnotes. The large
source export uses Hub large-file storage with contents matching the GitHub copy.
Local request logs and API credentials are not published.

## Evaluation limits

> **Experimental checkpoint.** The first model misses 25 of 30 short
> interruptions in the reviewed English challenge. Use it for research and
> evaluation; its probabilities are not calibrated.

The first model scored 100% on the 331-example synthetic test split but 73.5%
on the reviewed English challenge. Synthetic backchannels tend to be shorter
than interruptions. Criteria overlap between train and validation/test; exact
accepted cleaned-text duplicates do not cross splits, but template/semantic
near-duplicate leakage remains unaudited. Synthetic scores do not establish
real-world accuracy or calibration.

The 98 challenge examples were authored before inference, owner-reviewed, and
reconciled with authorization. Final binary decisions are in
[data/evaluation/reviews/english-challenge-v1.json](data/evaluation/reviews/english-challenge-v1.json)
and match the frozen CSV labels. Its original pending `review_status` is preserved
as part of that frozen artifact. Three exact cleaned texts occur in original
training and one in original validation; the supplement introduces none.
The challenge has been reused for development. Repeated prefixes are
intentional. Fresh human-reviewed evaluation is needed for independent claims.

The supplement covers short acknowledgments, repetitions, combinations and
contrasting requests, but does not exhaust every expression or regional variety.
Criterion and expression-coverage gaps remain documented in the GitHub source.
The new E5 export reaches 94/98 on the reused reviewed cases at INT8 batch one;
that is development evidence, not an independent multilingual accuracy estimate.

To extend the data, fix labels, or contribute a new evaluation set, use the
[GitHub generation guide](https://github.com/vxpipe/barge-in-detector/blob/main/docs/generation.md)
and [review workflow](https://github.com/vxpipe/barge-in-detector/blob/main/docs/review-ui.md).
This repository contains released data and metadata, without development scripts.

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
