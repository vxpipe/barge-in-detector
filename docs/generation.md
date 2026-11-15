# Synthetic data generator

Status: 2026-11-15. How the generator talks to Gemini for the utterance track
([utterance-first decision](utterance-first-decision.md)); storage and
fill-to-count are in [data format](data-format.md), the factors in
[the factor catalogue](factor-catalogue.md). The earlier conversation
pipeline (2-3 line windows) was removed on 2026-11-15; it is in git history.

## Decision

- **Generator model**: `gemini-3.8-flash` (user decision, 2026-11-15, after
  `gemini-3.5-flash-lite` passed only 14% of utterances through the judge).
  Set in `generation/config.yaml`; every request records its model.
- **One request per criterion**, asking for up to `batch_size` (10)
  utterances, each said over a speaker who is still talking.
- **Two parts per call**:
  1. **System prompt** `generator_utterances_v5` (cached in an explicit
     Gemini context cache per model and prompt hash; needs API `v1beta`): the
     task, the track's intents with their labels, script form and
     code-switching meanings, writing rules, output
     format.
  2. **Request**: a **Languages and scripts** block with one line per language
     of the combination and its script, no language ranked above another,
     e.g.

     ```
     - Languages and scripts (every language listed here must appear in every utterance; mix these languages in the utterance):
       - Tamil (regional variety ta-IN): in Tamil script, never in Latin letters.
       - English: in Latin letters.
     ```

     Then the intent with its intended label and example hints.
- **Structured output** (`UTTERANCE_SCHEMA`): `text`, `label` (`backchannel`
  or `barge_in`), `rationale`. An utterance that starts as a backchannel and
  turns into a barge-in is simply `barge_in`; where it turns is not recorded
  (user decision, 2026-11-15). Validated in code;
  rejected, never repaired.
- **Indian languages** appear alone or mixed with English; Indian English
  mixes one Indian language (user decision, 2026-11-15; manifest 0.7.0).

## Acceptance of a sample

In order; the first failure is the row's `reason`:

1. **schema**: fields present, label `backchannel` or `barge_in`, not empty
   after cleaning, no ellipsis, no bracket tags. Extra samples are rejected.
2. **label_mismatch**: the generator's label equals the intended label.
3. **dedupe**: the cleaned text is not already accepted (or waiting for the
   judge).
4. **judge**: every question answered yes.

Words are whitespace tokens that keep some text after cleaning
(`cleaner.words`): "Mm-hmm," is one word.

## Judge

The Gemini Decisions API (`gemini-decision-1`), one request per utterance;
the input is `{"text": ...}` only. The questions restate the request's
criteria and nothing else, never the label (user decision, 2026-11-15),
built in `generation/utterance_criteria.py`. For Tamil with English mixed in:

| Key | Question |
| --- | --- |
| `language_ta` | Does this text contain Tamil words written in Tamil script? |
| `language_en` | Does this text contain English words written in Latin letters? |
| `intent` | Said over a speaker who is still talking, is this text: <intent description> |

Every answer must have P(yes) ≥ 0.5; otherwise the reason lists each key
answered no (`judge: language_ta=0.08`). Every row keeps the judge's raw
answers.

## Running it

```shell
uv run python -m barge_in_detector check-catalogue
uv run python -m barge_in_detector plan                       # the criteria of the plan file
uv run python -m barge_in_detector show-prompt --system       # the cached system prompt
uv run python -m barge_in_detector show-prompt --index 0      # one request
uv run python -m barge_in_detector generate --dry-run         # shortfall and cost estimate
uv run python -m barge_in_detector generate --languages en --budget-usd 0.50
uv run python -m barge_in_detector status
```

`--plan` picks another plan file (default
`generation/plans/utterance-sample.yaml`). `generation/plans/utterance-all.yaml`
takes every valid combination (`all_combinations: true`) at the same count;
`only:` keeps some factor values, e.g. `generation/plans/english.yaml` is English
with `only: {code_switching: [none]}`. `generate` reads `GEMINI_API_KEY`
(or `GOOGLE_API_KEY`), makes up to `concurrency` (10) Gemini calls at once,
and writes `data/export/utterances.csv` at the end. Worker threads only call
Gemini; the main thread writes every row, one response at a time, which keeps
deduplication exact. Every request is logged before its call, so an
interrupted run loses nothing: the next run marks unanswered requests
`api_error` and finishes stored responses first. A generate request stays
`received` until each of its utterances has a verdict.

A changed prompt is a new file with a new version; the request log records the
template of every call.

## Rejected alternatives

- **The conversation pipeline** (2-3 line windows, contrastive sets, cut
  positions): solved a problem the runtime already solves; removed
  2026-11-15.
- **A judge that also labels the sample**: the judge checks the criteria only.
- **Longer judge questions with explanations** (judge v1, v2): read as two
  judges and misfired (Thai in Thai script failed because English words were in
  Latin letters); replaced by one-line questions.
- **Several samples in one Decisions request**: the API guide says unrelated
  items in one input lower accuracy.
- **Repairing near-miss responses**: silently changes what Gemini produced.
- **Batch API first**: interactive calls are simpler to resume and debug; the
  Batch API is a milestone task for the full dataset.
