# Data format and storage

Status: decided 2026-11-15 (user decisions); schema version 7 (6 without
`barge_in_from_word`). Replaces the
earlier ten-table design of the conversation track (in git history).

## Decision

- **The plan file is the only record of the criteria.** A plan
  (`generation/plans/*.yaml`) plus the factor files build the criteria
  deterministically (`generation/utterances.py`); git keeps its history. No
  plan revisions, targets or runs are stored.
- **One table holds every generated sample**, accepted or rejected:
  `utterances`. It is the dataset and the only committed data, exported as
  **one CSV**, `data/export/utterances.csv`, in plain git (diffable, loads
  straight into DuckDB or any CSV reader).
- **A local request log**, `requests`, holds every Gemini call (raw response,
  tokens, cost, status) for resume, the attempt budget and costs. It lives
  only in the working DuckDB file and is not committed.
- **DuckDB is the working store** (`data/work/barge_in.duckdb`, gitignored).
  It can be rebuilt from the CSV with `import` (the request log then starts
  empty, which also resets attempt budgets).

## Generation: fill each criterion to its required count

- For each criterion of the plan, `need = required_count − accepted − in
  flight`; generate only `need`. Running twice with nothing short makes no
  API calls and leaves the CSV unchanged.
- Attempt budget: at most `max_attempts_per_sample × required_count` samples
  requested per criterion, counting requests with the current generator
  template and model only (from the request log).
- Generated text is never changed. A sample row is inserted once; the only
  update is accepted → retired (with a reason) for bad data, which leaves its
  criterion short so a later run replaces it. Rows are never deleted.
- Interrupted work is resumed before new requests: a request sent but never
  answered becomes `api_error`; a stored response is validated and judged.
- More data: raise `required_count` or `criteria_per_language` in the plan
  (the builder is prefix-stable, so existing criteria keep their IDs and
  splits).
- Every run ends with the export.

## `utterances`

One row per generated sample, ordered by `utterance_id`.

| Column | Type | Notes |
| --- | --- | --- |
| `utterance_id` | UUID | v7, time-ordered |
| `combination_id` | UUID | the criterion (see Identifiers) |
| `language`, `regional_variant`, `script_form`, `code_switching`, `code_switch_set`, `mixed_in_script`, `caller_intent`, `development`, `repetition`, `length_bucket` | VARCHAR | the criterion's factor values |
| `locales` | VARCHAR[] | every locale in the text (`["hi-Latn-IN", "en-IN"]`); a JSON list in the CSV |
| `split` | VARCHAR | `train`, `validation` or `test`, from `combination_id` |
| `intended_label` | VARCHAR | from the intent: `backchannel` or `barge_in` |
| `status` | VARCHAR | `accepted`, `rejected` or `retired` |
| `reason` | VARCHAR | why rejected or retired: `<stage>: <detail>`, e.g. `schema: length_bucket_mismatch`, `label_mismatch: intended=barge_in gemini=backchannel`, `dedupe: duplicate of an accepted utterance`, `judge: script_ur=0.08,switching=0.3` |
| `text` | VARCHAR | exactly as generated |
| `text_clean`, `cleaner_version` | VARCHAR | the model input after the cleaner |
| `label` | VARCHAR | the generator's label (the training label when accepted) |
| `raw_sample` | JSON | the sample object exactly as Gemini returned it |
| `judge_answers` | JSON | the judge's raw answers (question key → `noul`, `confidence`); null when not judged |
| `text_hash` | VARCHAR | SHA-256 of `text_clean`; an accepted text is never accepted again |
| `request_id`, `judge_request_id` | UUID | links to the local request log |
| `sample_index` | INTEGER | position in the response |
| `model_id`, `prompt_template`, `judge_model_id`, `judge_template` | VARCHAR | |
| `created_at` | TIMESTAMP | |

A response that cannot be parsed at all has no sample rows; its request is
`schema_invalid` in the request log.

## `requests` (local only)

`request_id`, `kind` (`generate_utterances` or `judge_utterance`),
`parent_request_id` (a judge request's generate request), `combination_id`,
`language`, `requested_count`, `model_id`, `prompt_template`, `prompt_hash`,
`status` (`submitted`, `received`, `processed`, `api_error`,
`schema_invalid`), `error`, `response` (raw, JSON), token counts, `cost_usd`,
`created_at`, `completed_at`.

## Identifiers

- `combination_id`: UUIDv5 in the fixed project namespace
  `3de823b9-152c-4f71-899d-8d2bde9cc65c` (**never change it**) of
  `utterance_criterion:` + sorted `factor=value` pairs joined by `;`, leaving
  out factors at `default`. The same factor values always give the same ID.
- `utterance_id`, `request_id`: UUIDv7, so rows sort in creation order and the
  CSV only grows at the end.

## Splits

Each criterion's split comes from its `combination_id` and the plan's
`split_ratios` (`generation/utterances.split_for`), before anything is
generated, so all samples of a criterion share a split and the test set never
depends on results. Deduplication on `text_clean` across all accepted rows
keeps the same text out of two splits.

## Labels

The intent sets the intended label; the generator classifies each sample and
must agree. The judge checks only that the sample matches its criterion
([generation](generation.md#judge)), never the label.

## Rejected alternatives

- **Parquet partitioned by language and run** (schema 1-5): hundreds of small
  files for a few thousand rows, binary, in Git LFS, no diffs.
- **Committing the request log**: raw responses are mostly the samples
  themselves, which each row already carries as `raw_sample` and
  `judge_answers`; the rest (tokens, cost, retries) is only needed locally.
- **Storing criteria, plan revisions and runs**: the plan file in git and the
  deterministic builder already give every criterion; storing them duplicated
  it and needed migrations.
- **Random row-level splits**: samples of one criterion would land on both
  sides.
- **Storing token IDs**: ties the data to one tokenizer.
