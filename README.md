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
---
# Barge-in detector data

Synthetic multilingual utterances for deciding whether a caller speaking over
an ongoing voice-agent turn is acknowledging the agent or claiming the floor.
Exactly two labels: `backchannel` and `barge_in`. Answers after the agent stops
are outside the detector's contract. Text is cleaned lowercase without punctuation,
using the versioned cleaner; natural text is retained alongside it.

This repository owns **data, generation, cleaning, splits, review and historical
labnotes**. [The separate model repo](https://huggingface.co/HashNuke/barge-in-detector)
owns **weights, tokenizer, training, inference, evaluation and benchmarks**.
The original [GitHub repository](https://github.com/vxpipe/barge-in-detector)
is the historical source archive for this dataset project.

## Load the dataset

```python
from datasets import load_dataset

splits = load_dataset("HashNuke/barge-in-detector-data", "utterances", revision="DATASET_COMMIT")
challenge = load_dataset(
    "HashNuke/barge-in-detector-data", "challenge", split="test", revision="DATASET_COMMIT"
)
```

Pin a dataset commit when training or reporting results. The model training
scripts resolve a requested Hub revision to an immutable commit before download.

| Split | Examples |
| --- | ---: |
| Train | 2,640 |
| Validation | 333 |
| Test | 331 |
| Reviewed English challenge (separate configuration) | 98 |

The 3,304 accepted synthetic examples contain 2,220 barge-ins and 1,084
backchannels. `data/export/utterances.csv` is the source export of **all 12,414
accepted, rejected and retired sample rows**, with generation factors and judge
answers. It is not the training configuration; only accepted rows enter splits.
The splits cover 14 languages. English includes US, India and Singapore variants.
Use `text_clean` as model input, with a first-64-token budget.

## Provenance and limitations

Gemini writes natural text under deterministic factor criteria; a criteria judge
accepts or rejects samples. See generation/config.yaml, generation/plans/,
[the generation guide](docs/generation.md) and [data format](docs/data-format.md).
Generation request logs and credentials are not published. Samples are synthetic,
not transcripts of real customers.

Every criterion contributes training rows and, when it has at least two accepted
samples, one validation or test row. Criteria therefore overlap between splits.
Exact cleaned-text duplicates are absent across accepted splits; template and
semantic near-duplicate leakage have not been ruled out. Split assignment and
its limits are documented in [docs/training.md](docs/training.md).

**Synthetic test accuracy is insufficient evidence of generalization.**
Backchannels tend to be shorter than interruptions in the first dataset.
The first model scored 100% on synthetic test but only 73.5% on the reviewed
English challenge, missing 25/30 short interruptions. The challenge was authored
before model inference, then owner-reviewed and reconciled with authorization.
Final binary review decisions are in data/evaluation/reviews/english-challenge-v1.json;
its labels match the frozen challenge CSV. The CSV's original review_status
column remains unchanged to preserve the frozen artifact. Five challenge rows
occur in synthetic train/validation; unseen examples must be reported separately.
The challenge is now a development/regression set and includes repeated prefixes;
it is not a representative independent test population.

Owner verification is available for English, Hindi, Kannada and Tamil. This
release has an English diagnostic review, not human-verified evaluation across
all languages. Other language results remain synthetic/unverified.
Native scripts optionally mixed with English are the target policy. Consult the
language manifest and historical generation labnotes for actual data provenance.

## Develop and review

```shell
uv sync
uv run python -m barge_in_detector split
uv run python -m barge_in_detector.review
uv run ruff format --check
uv run ruff check
uv run pytest
```

Generation needs an environment API key; never commit keys or local request logs.
The review UI saves annotations separately from frozen inputs. See
[review instructions](docs/review-ui.md) for SSH/Tailscale access.
Start at [AGENTS.md](AGENTS.md) and [the milestones index](labnotes/milestones/index.md).
Use `bin/create-labnotes` for research/implementation checkpoints. Historical
model labnotes remain here; new model-only work is recorded in the model repo.
