# Data generation and storage format

Status: decided 2026-11-14; schema version 4 implemented (2 added `locales`; 3
added `mixed_in_script`, script subtags in `locales` and
`examples.transcript_raw`; 4 added the utterance track's `utterance_targets`
and `utterances` tables, see the
[utterance-first decision](utterance-first-decision.md); older databases are
migrated on open) in
`barge_in_detector/storage/` (`schema.py` owns the columns, `store.py` every
write). Column lists change only through a schema version bump recorded here.
Gemini calls, prompts and acceptance are described in
[the generator](generation.md).

## Decision

- **DuckDB is the working store.** Generation, cleaning, review and snapshot
  building read and write one local DuckDB database through SQL, so specific
  rows and cells can be queried and updated in transactions.
- **Parquet is the delivery format.** A deterministic export writes every table
  to Parquet under `data/export/`. Those files are committed to git (in Git
  LFS), published, and used for training. The database can be rebuilt from them
  at any commit.
- **Generation fills each criterion to its required count.** A criterion is one
  factor combination with a required number of samples. A run counts what
  already exists for each criterion, generates only the shortfall, and moves on
  to the next criterion when there is none. Existing generated data is never
  overwritten or regenerated.

| Table | One row per | Main use |
| --- | --- | --- |
| `plan_revisions` | plan revision: quotas, ratios, seed, catalogue versions | Audit of how targets were set |
| `targets` | criterion: factor values, label, split, required count | What to generate |
| `requests` | Gemini request attempt, including failures, with the raw response | Provenance, cost, resume |
| `conversations` | accepted conversation, exactly as generated | Source for examples |
| `rejections` | rejected attempt and the reason | Prompt quality, attempt limits |
| `examples` | model input (turn window + candidate prefix + label), original and cleaned | Training and evaluation |
| `utterance_targets` | utterance-track criterion: factor values, label, split, required count | What to generate (utterance track) |
| `utterances` | accepted utterance, exactly as generated, with its cleaned text | Utterance-track training data |
| `annotations` | human review of an example | Human-verified evaluation sets |

**Factors are stored as columns** on targets, conversations and examples, next
to the input and the label. This allows training on any subset, rebalancing,
and breaking down metrics by any factor without joins.

**Originals are kept; cleaned transcripts are derived.** Conversations keep the
text exactly as generated (casing, punctuation, speaker names) as a list of
`{speaker, role, text}`. Examples store both that original window and the
cleaned model input (`s1: …` lines, lowercase, no punctuation) produced by the
versioned cleaner in [transcript format](transcript-format.md). Re-running a
cleaner updates the cleaned columns; it never touches originals and makes no API
calls. Tokenized data is not stored; it is a cache produced at training time.

## Working store and exports

```text
data/
  work/barge_in.duckdb            # working store; local, gitignored
  export/                         # committed, Git LFS
    plan_revisions.parquet
    runs.parquet
    annotations.parquet
    targets/language=hi/part-0.parquet
    requests/language=hi/run_id=<run_id>/part-0.parquet
    conversations/language=hi/run_id=<run_id>/part-0.parquet
    rejections/language=hi/run_id=<run_id>/part-0.parquet
    examples/builder=<version>/split=train/language=hi/run_id=<run_id>/part-0.parquet
    snapshots/<snapshot_id>/
      manifest.json               # see "Snapshots"
      coverage.parquet            # counts per factor value and factor pair
      splits.parquet              # split audit: proportions per factor value × split
```

- **The database file is not committed.** A DuckDB file is a single binary that
  changes as a whole: every commit would store a full new copy in LFS, and it
  cannot be diffed or merged. The exports are the durable record. An `import`
  command rebuilds `barge_in.duckdb` from `data/export/` at any commit.
- **Every generation run ends with an export**, so committed data never lags
  behind the working store.
- **The export is deterministic**: rows ordered by primary key, fixed row-group
  size, zstd compression, single-threaded writing. Unchanged partitions produce
  byte-identical files, so git sees no change for them. Partitioning by
  language and `run_id` means a new run adds new files and leaves older
  partitions alone; only rare updates (such as retiring a conversation) rewrite
  an older partition. Checkpoint B must prove determinism with a test (export
  twice → identical bytes).
- **Git LFS** stores every `*.parquet` and `*.json` file, as configured in
  `.gitattributes`. Partitions are split into more `part-N` files to stay under
  100 MB each.
- Small fixtures for tests live in `tests/fixtures/`.

## What may change and what may not

| Never changes once written | May be updated |
| --- | --- |
| Generated text and everything in `requests.response` | `targets.required_count` (raised or lowered by a plan revision) and `targets.priority` (generation order only); the same for `utterance_targets` |
| Factor values, `combination_id`, label | `targets.status`, `conversations.status`, `utterance_targets.status` and `utterances.status` (`active`/`retired`) with reason and time; a rebuilt plan applied with `--retire-missing` retires its own earlier criteria it no longer contains |
| A criterion's `split` and `split_group_id`, once it has generation requests | Cleaned columns and `cleaner_version` in `examples`; plan-assigned fields (split, split group, quota group, contrastive group) of criteria with no generation requests yet |
| Request records (other than status while in flight) | `examples` rows: derived, so they can be rebuilt for a new builder version |

A single storage module owns all writes. It allows only the updates in the
right-hand column; anything else is refused, and tests prove it.

## Generation: fill each criterion to its required count

### Rules

1. **Never overwrite or regenerate existing generated data.** Changing a prompt
   template, the generator model or the judge applies to new samples only; each
   row records the template and models that produced it.
2. **Only fill the shortfall.** For each criterion,
   `need = required_count − accepted − in_flight`. If `need ≤ 0`, move on to
   the next criterion.
3. **Running twice changes nothing.** With unchanged targets and nothing in
   flight, a second run makes no API calls and the export is unchanged.

### What a run does

1. Open the database (DuckDB allows one writer process, so two runs cannot
   overlap; API calls within a run use worker threads with their own cursors).
2. Finish earlier work first, without new generator calls: mark requests that
   never got a response `api_error` ("interrupted"), and process stored
   responses that were never turned into conversations or rejections, calling
   the judge first if needed. (Collecting submitted Batch API jobs belongs here
   once batch mode exists.)
3. Go through active criteria in a fixed order (plan priority, then
   `quota_group`, then `combination_id`); a contrastive group is one unit. For
   each, count accepted (non-retired) conversations and the samples in
   requests still in flight (`submitted` or `received`), and compute `need`.
4. Skip the criterion if `need ≤ 0`, or if it has used its attempt budget
   (`max_attempts_per_sample × required_count` samples requested, counted from
   `requests`, excluding `api_error` requests so an outage does not exhaust
   it). Rejected attempts use budget but never count toward the required
   count.
5. Otherwise request `need` samples, in requests of at most `batch_size` (10).
   Each request row is inserted before the call is made, so a crash never
   loses track of a request. Rounds repeat within the run until nothing is
   short, the budget is used, `--limit` is reached, or a round gets no
   response at all.
6. Validate responses, run the blind judge, check that the labels agree (see
   [Labels](#labels)) and deduplicate; insert accepted conversations and
   their examples, or rejections, in the same transaction that marks the
   request processed.
7. Export, then report accepted and rejected counts (by stage and reason),
   criteria still short, budget exhausted, and cost.

A run that finds nothing to do records no run row and changes no file; an
unchanged plan records no plan revision.

`--dry-run` performs steps 1–4 and prints the shortfall and estimated cost
without calling the API or writing anything. `--limit` and a cost budget cap a
run; the next run continues where it stopped.

### Growing the pool

A new plan revision either adds criteria or raises `required_count` on existing
ones. The plan builder is deterministic and **prefix-stable**: the criteria
chosen for a quota of N are always the first N chosen for any larger quota.
Existing criteria keep their IDs and splits. The next run generates only the
new shortfall.

Lowering a required count deletes nothing; the surplus stays and snapshots may
leave it out.

### Catalogue changes

- `combination_id` is computed from factor values only, not the catalogue
  version, so **adding** languages, values, domains or intents leaves every
  existing criterion unchanged and only creates new ones.
- **Removing or renaming** a value retires the affected criteria
  (`targets.status = retired`). Their conversations stay in the data;
  snapshots exclude them.

### Retiring bad data

If an accepted conversation turns out to be wrong (e.g. after human review), it
is marked `retired` with a reason; its text stays. Its criterion is then one
short, so the next run generates a replacement.

### Duplicates

A new conversation that exactly or nearly duplicates an existing accepted one
is rejected (stage `dedupe`). Existing conversations are never removed for
being duplicated by new ones.

## Labels

Every sample's label comes from Gemini, the reference model, not from our code:

1. The criterion's intent sets the **intended label** and steers generation, so
   label counts and balance stay deterministic.
2. The generation call returns the sample **and its classification**: the
   label of the final utterance, the word where a developing utterance becomes a
   barge-in (if it does), and a short rationale.
3. A separate **blind judge**, the Gemini Decisions API (`gemini-decision-1`),
   classifies each sample in its own request without seeing the intent, the
   intended label or the generator's answer, and returns calibrated
   probabilities (see [the generator](generation.md)).
4. The sample is accepted only if the generator's classification, the judge's
   classification and the intended label all agree. Otherwise it is rejected
   with stage `label_mismatch`; the criterion stays short and a later attempt
   fills it. Mismatch rates are reported per intent and language, which shows
   where the intent definitions or prompts are unclear.

The stored training label is Gemini's classification; per-prefix labels in
`examples` follow from the classified barge-in word. For a barge-in, the two
`barge_in_from_word` values may differ by at most one word. An intended
`ambiguous` sample needs only the generator's `ambiguous`; the judge's label is
kept for evaluation. The exact order of checks is in
[the generator](generation.md#acceptance-of-a-sample).

## Identifiers

**Every row in every table has a UUID primary key**, so any row can be
referred to directly. Two kinds are used:

- **UUIDv7 (time-ordered, random)** for rows the project *creates*: plan
  revisions, runs, requests, conversations, rejections, annotations and
  snapshots. Being time-ordered, they sort in creation order, which keeps
  primary-key-ordered exports stable as rows are added.
- **UUIDv5 (derived from content)** for rows that must get the *same* ID every
  time they are computed: criteria and examples. Rebuilding the plan or the
  examples therefore never changes their IDs, references stay valid, and the
  export does not churn.

UUIDv5 uses the fixed project namespace
`3de823b9-152c-4f71-899d-8d2bde9cc65c`. **Never change it**; doing so would
change every derived ID.

| ID | Kind | Derived from / notes |
| --- | --- | --- |
| `revision_id` | v7 | plan revision; also has a human-readable `name` |
| `run_id` | v7 | generation run; also has a `name` |
| `snapshot_id` | v7 | snapshot; also has a `name` |
| `combination_id` | v5 | `criterion:` + sorted `factor=value` pairs joined by `;`, leaving out factors at the value `default`. Identifies one criterion. Independent of the catalogue version, and a new factor whose existing criteria take `default` keeps their IDs, so it is stable as the catalogue grows |
| `split_group_id` | v5 | `split_group:` + the smallest `combination_id` in the group |
| `request_id` | v7 | one Gemini request attempt |
| `conversation_id` | v7 | one accepted conversation |
| `rejection_id` | v7 | one rejected attempt |
| `example_id` | v5 | `example:{conversation_id}:{turn_shape}:{prefix_index}`. Independent of the builder and cleaner versions, so annotations stay attached when cleaning changes |
| `annotation_id` | v7 | one human review entry |

UUIDs are generated in Python (`uuid.uuid5`, and `uuid.uuid7` from Python
3.14, which the project requires) in `barge_in_detector/ids.py`, so tests
check the derivations.
DuckDB stores them with its `UUID` type. Exports write them as canonical
lowercase strings so every Parquet reader handles them the same way. Human
names (e.g. `en-hi-pilot`) are separate columns, never used as keys.

## Common factor columns

Present on targets, conversations and examples. Values and their meanings are
defined in [the factor catalogue](factor-catalogue.md) and are validated
against it on insert. `null` means the factor does not apply; use an explicit
value such as `none` when the factor applies and is absent.

| Column | Type | Notes |
| --- | --- | --- |
| `combination_id` | UUID | |
| `factor_catalogue_version` | VARCHAR | catalogue version when the criterion was created |
| `language` | VARCHAR | code from `languages.yaml`, e.g. `hi`, `en`, `kn`, `ta` |
| `regional_variant` | VARCHAR | e.g. `en-IN`; the language code when it has no variants |
| `script_form` | VARCHAR | `native`, `romanized`, `mixed_script` |
| `code_switching` | VARCHAR | `none`, `between_sentences`, `within_sentence` |
| `code_switch_set` | VARCHAR | language(s) mixed in, joined by `+`, e.g. `en`, `en+hi`; `none` when `code_switching=none` |
| `mixed_in_script` | VARCHAR | `default`, `native` or `romanized`: how the mixed-in language is written, for sets that allow a choice (Mandarin in Malaysian English as characters or pinyin) |
| `locales` | VARCHAR[] | every locale in the transcript, the conversation's first, then each mixed-in language's in the same region, with a script subtag when a language is not in its usual script or is Chinese: `[en-US]`, `[en-MY, zh-Hans-MY]`, `[en-MY, zh-Latn-MY]`, `[en-HK, yue-Hant-HK]`, `[en-IN, hi-Latn-IN, kn-Latn-IN]`, `[hi-Latn-IN, en-IN]`. Derived from the language factors and the manifest; not part of `combination_id` |
| `domain` | VARCHAR | conversation category |
| `scenario` | VARCHAR | scenario within the domain |
| `speaker_pattern` | VARCHAR | `returning_speaker` (s1, s2, s1) or `new_speaker` (s1, s2, s3) for the 3-line window |
| `agent_activity` | VARCHAR | explaining, listing_options, reading_numbers, … |
| `cut_position` | VARCHAR | start, early, middle, near_end, just_finished |
| `agent_state` | VARCHAR | `speaking`, `just_finished`; derived from `cut_position` |
| `caller_intent` | VARCHAR | fine-grained intent; determines the label |
| `development` | VARCHAR | `stable`, `backchannel_then_barge_in`, `filler_then_barge_in` |
| `repetition` | VARCHAR | `single`, `repeated` |
| `length_bucket` | VARCHAR | `w1`, `w2_3`, `w4_8`, `w9_plus` |
| `register` | VARCHAR | formal, casual |
| `caller_mood` | VARCHAR | neutral, friendly, hurried, confused, frustrated |
| `label` | VARCHAR | `backchannel`, `barge_in`, `ambiguous` (evaluation only). On `targets` it is the intended label from the intent; on `conversations` and `examples` it is Gemini's classification, see [Labels](#labels) |
| `contrastive_group_id` | VARCHAR | links minimal pairs: `<group>:<language>:<12 hex of the members' split group id>`, unique to its members so a rebuilt plan never reuses it; null otherwise |
| `split_group_id` | UUID | |
| `split` | VARCHAR | `train`, `validation`, `test`, `heldout`; fixed when the criterion is created, see [Splits](#splits) |

## Tables

### `plan_revisions`

`revision_id` (primary key), `name`, `created_at`, `seed`, `quotas` (JSON),
`split_ratios` (JSON), `heldout` (JSON: factor values held out) and
`catalogue_versions` (JSON: the version of every catalogue file).

### `runs`

`run_id` (primary key), `name`, `revision_id`, `started_at`, `finished_at`,
`generator_model_id`, `judge_model_id`, settings (JSON: limit, budget, attempt
budget), requests made, conversations accepted, rejections, `cost_usd`.

### `targets`

Common factor columns, plus:

| Column | Type | Notes |
| --- | --- | --- |
| `combination_id` | UUID | primary key |
| `contrastive_role` | INTEGER | member index within the contrastive group; null otherwise |
| `required_count` | INTEGER | samples needed for this criterion |
| `quota_group` | VARCHAR | the stream that selected this criterion: `stratified` or `contrastive` |
| `priority` | INTEGER | generation order; lower first |
| `added_in_revision`, `count_set_in_revision` | UUID | plan revisions that created it and last set its count |
| `status` | VARCHAR | `active` or `retired` |
| `retired_reason`, `retired_at` | VARCHAR, TIMESTAMP | |

### `requests`

| Column | Type | Notes |
| --- | --- | --- |
| `request_id` | UUID | primary key |
| `kind` | VARCHAR | `generate` or `judge` |
| `parent_request_id` | UUID | for a judge request, the generate request whose sample it judges |
| `run_id` | UUID | |
| `combination_id` | UUID | the criterion; null for a contrastive group request |
| `contrastive_group_id` | VARCHAR | the group, for a contrastive group request |
| `language` | VARCHAR | |
| `requested_count` | INTEGER | samples asked for (judge: 1, one sample per Decisions request) |
| `batch_job_id` | VARCHAR | null for direct calls |
| `model_id`, `prompt_template`, `prompt_hash` | VARCHAR | |
| `status` | VARCHAR | `submitted`, `received`, `processed`, `api_error`, `schema_invalid` |
| `error` | VARCHAR | for `api_error` and `schema_invalid`, with any key redacted |
| `response` | JSON | curated raw response, no headers. Generate: the returned text exactly, model version, response ID, finish reason, token usage. Judge: `item_index` (the sample's position among the validated samples), every answer with its probabilities, usage; or `blocked` |
| `input_tokens`, `cached_input_tokens`, `output_tokens` | INTEGER | output includes thinking tokens |
| `cost_usd` | DOUBLE | |
| `created_at`, `completed_at` | TIMESTAMP | |

Never store API keys or request headers.

### `conversations`

Common factor columns, plus:

| Column | Type | Notes |
| --- | --- | --- |
| `conversation_id` | UUID | primary key |
| `request_id`, `judge_request_id`, `run_id` | UUID | provenance |
| `sample_index` | INTEGER | position of the sample in the response |
| `turns` | STRUCT(speaker VARCHAR, role VARCHAR, text VARCHAR)[] | the turns before the overlapped agent turn, oldest first, exactly as generated (currently one: the context turn); `speaker` is the name or role Gemini used, `role` is `listener` (the judged person) or `other` (a third person) |
| `agent_speaker` | VARCHAR | the speaker's name as generated (the person being spoken over; `agent_*` columns mean the speaker, `candidate_*` the listener) |
| `agent_full_text` | VARCHAR | what the agent intended to say in the overlapped turn, as generated |
| `agent_spoken_text` | VARCHAR | the part spoken before the caller started, as generated |
| `candidate_speaker`, `candidate_role` | VARCHAR | the listener's name as generated; role `listener` |
| `candidate_text` | VARCHAR | the caller's full final utterance, as generated |
| `intended_label` | VARCHAR | from the criterion's intent |
| `gemini_label` | VARCHAR | the generator's classification of the final utterance |
| `gemini_rationale` | VARCHAR | the generator's short reason |
| `barge_in_from_word` | INTEGER | the generator's word index where a developing utterance becomes barge-in; null if it never does |
| `judge_label` | VARCHAR | the blind judge's classification |
| `judge_barge_in_from_word` | INTEGER | the judge's word index, for agreement checks |
| `model_id`, `prompt_template`, `judge_model_id` | VARCHAR | |
| `judge_notes` | VARCHAR | the judge's probability for its label, and for any failed check |
| `text_hash` | VARCHAR | SHA-256 of the cleaned three-line window, for exact deduplication |
| `status` | VARCHAR | `active` or `retired` |
| `retired_reason`, `retired_at` | VARCHAR, TIMESTAMP | |
| `created_at` | TIMESTAMP | |

### `rejections`

`rejection_id` (primary key), `request_id`, `judge_request_id`,
`combination_id`, `language`, `run_id`, `sample_index`, `stage` (`schema`,
`judge`, `label_mismatch`, `dedupe`, `contrastive`), `reason`, `gemini_label`,
`judge_label`, `created_at`.

### `examples`

Built deterministically from active conversations by
`barge_in_detector/examples.py` (builder version 1): one row per (turn
shape × candidate prefix). Prefixes are cut at the original candidate's
whitespace tokens, then each window is cleaned; a token that adds no cleaned
word (a lone dash) gives no new prefix, and `barge_in_from_word` counts
cleaned words. Inputs are cleaned-up transcripts, so no ASR-noise
variants are produced. Common factor columns, plus:

| Column | Type | Notes |
| --- | --- | --- |
| `example_id` | UUID | primary key |
| `conversation_id`, `run_id` | UUID | provenance |
| `builder_version` | VARCHAR | example builder version |
| `turn_shape` | TINYINT | 2 or 3 |
| `turns_original` | STRUCT(speaker VARCHAR, role VARCHAR, text VARCHAR)[] | the window as generated; the agent turn holds `agent_spoken_text`; the last turn is the candidate prefix |
| `transcript_original` | VARCHAR | the window as `<speaker>: <text>` lines, as generated |
| `transcript_clean` | VARCHAR | the model input: `s<n>: <text>` lines after cleaning, see [transcript format](transcript-format.md) |
| `transcript_raw` | VARCHAR | the alternative model input: `s<n>: <text>` lines with casing and punctuation kept (NFC and whitespace only), for comparing training with and without cleaning |
| `speaker_labels` | VARCHAR[] | the `s<n>` label of each line, e.g. `[s1, s2, s1]` |
| `candidate_original` | VARCHAR | the candidate prefix as generated |
| `candidate_clean` | VARCHAR | the candidate prefix after cleaning |
| `cleaner_version` | VARCHAR | |
| `prefix_index` | SMALLINT | 0-based index of the prefix's last original token |
| `is_final_prefix` | BOOLEAN | |
| `label` | VARCHAR | label for this prefix (a prefix of a developing interruption can be `backchannel`) |
| `label_id` | TINYINT | 0 = backchannel, 1 = barge-in, null = ambiguous |

Sample weights for rebalancing belong to the snapshot or training config, not
the examples table.

### `utterance_targets`

Utterance-track criteria from `plan-utterances`
(`barge_in_detector/generation/utterances.py`). The factors are those in
`generation/factors/utterances.yaml`; the catalogue's constraints apply, plus
the track's own rules. `combination_id` is
`uuid5(NAMESPACE, "utterance_criterion:" + factor=value pairs)`
(`ids.utterance_combination_id`), so it never equals a conversation
criterion's ID. `split` is fixed from the ID alone (`split_for`: the ID modulo
10,000 against the cumulative split ratios); the split audit on cleaned text is
a milestone task. Partitioned by `language`.

| Column | Type | Notes |
| --- | --- | --- |
| `combination_id` | UUID | primary key |
| `factor_catalogue_version`, `utterance_track_version` | VARCHAR | |
| `language` … `length_bucket` | VARCHAR | the ten utterance factors |
| `locales` | VARCHAR[] | as on `targets` |
| `split` | VARCHAR | `train`, `validation` or `test` |
| `label` | VARCHAR | intended label, from the intent |
| `required_count`, `priority` | INTEGER | priority is the position in the plan |
| `added_in_revision`, `count_set_in_revision` | UUID | `plan_revisions` rows (quotas JSON holds `"track": "utterance"`) |
| `status`, `retired_reason`, `retired_at` | | |

### `utterances`

Accepted utterances, one row per sample. Requests are in `requests` with
`kind = generate_utterances` and rejections in `rejections`, as for
conversations. Partitioned by `language` and `run_id`.

| Column | Type | Notes |
| --- | --- | --- |
| `utterance_id` | UUID | primary key (v7) |
| factor columns | | copied from the criterion, as above |
| `label` | VARCHAR | the training label: the generator's label, which must equal `intended_label` |
| `intended_label` | VARCHAR | |
| `request_id`, `run_id` | UUID | provenance |
| `sample_index` | INTEGER | position in the response |
| `text` | VARCHAR | exactly as generated |
| `text_clean`, `cleaner_version` | VARCHAR | the model input after the cleaner |
| `gemini_label`, `gemini_rationale`, `barge_in_from_word` | | the generator's classification |
| `judge_label`, `judge_model_id` | VARCHAR | null: no judge in this version |
| `model_id`, `prompt_template` | VARCHAR | |
| `text_hash` | VARCHAR | SHA-256 of `text_clean`; an accepted text is never accepted again |
| `status`, `retired_reason`, `retired_at`, `created_at` | | |

### `annotations`

| Column | Type | Notes |
| --- | --- | --- |
| `annotation_id` | UUID | primary key |
| `example_id` | UUID | the example reviewed |
| `source` | VARCHAR | `synthetic_reviewed`, `recorded` |
| `human_label` | VARCHAR | may differ from `label`; `human_label` is the ground truth |
| `ambiguous` | BOOLEAN | reported separately in metrics |
| `annotator` | VARCHAR | a pseudonymous ID, not a name |
| `annotated_at` | TIMESTAMP | |
| `annotation_notes` | VARCHAR | |

A correction is a new row with a later `annotated_at`; the latest row wins, so
review history is kept. Recorded (non-synthetic) evaluation examples are
inserted into `examples` with `source` set in their annotation.

## Snapshots

Because the data keeps growing, training never reads "whatever is there now".
It reads a **snapshot**: a manifest listing the exact export files, their
content hashes and the git commit they came from, after excluding retired rows
and applying the cross-split near-duplicate filter. A snapshot is written once
and never changed; a later snapshot is a new directory. Training runs and model
cards cite the `snapshot_id`.

### Manifest (`manifest.json`)

`schema_version`, `snapshot_id`, git commit, plan revisions and run IDs
included, builder and cleaner versions, the included export files with content
hashes, near-duplicate exclusions (example IDs removed from training),
generator and judge model IDs, prompt template hashes, factor catalogue /
constraints / language manifest versions, seed, row counts per split ×
language × label, rejection counts per stage and reason, total cost, and the
licence.

## Splits

Splitting is deterministic and happens **when a criterion is created, before
any data is generated for it**, so the test set is fixed before anyone sees
results and cannot be tuned against. A criterion's split never changes, so
every sample generated for it, now or later, lands in the same split.

### Grouping: what must stay together

All rows in a split group go to the same split:

- every sample of one criterion (`combination_id`), so the same kind of sample
  never appears in both training and test;
- every criterion in one `contrastive_group_id`, so a minimal pair is never
  split;
- automatically, every prefix and turn shape of a conversation, because
  examples inherit the criterion's split.

Criteria and contrastive groups are merged with union-find; the result is
`split_group_id`.

### Assignment: same proportions in every split

1. Target ratios, initially **train 80 / validation 10 / test 10**, recorded in
   the plan revision.
2. New groups are ordered rarest first (by the least frequent factor value
   they carry), ties broken by SHA-256 of (`seed`, `split_group_id`), which is
   deterministic but unrelated to factor values.
3. Each new group is assigned in that order to the split that is furthest below
   its target share for the group's factor values and label, weighted by
   `required_count` and **counting existing criteria as already assigned**
   (iterative stratification). A group carrying a value train does not have yet
   goes to train.
4. A refinement pass then moves or swaps new groups between splits while that
   lowers a chi-square imbalance over the audited values (those with at least
   `split_min_support` criteria, 50 by default), never taking a value's last
   training criterion out of train. This keeps every common factor value,
   every label and every language at the target ratio in each split as the
   data grows, so validation and test don't end up with more of one kind.
5. Every factor value present in validation or test is also present in
   train. Combinations are unseen; individual values are not.

Same inputs (existing targets, quotas, ratios, seed) always give the same
assignment.

### Held-out slice

A separate `heldout` split takes whole factor values out of training to measure
generalization: for example one domain, one agent activity, and some Tier C
languages. These are listed in the plan revision and removed before
stratification. Results on `heldout` are reported separately and are never
used for tuning or threshold selection.

### Checks

- **Plan revision**: each factor value, label and language with at least
  `split_min_support` criteria is within ±2 percentage points of the target
  ratio in every split (by required count), or the revision is not applied;
  rarer values are reported but cannot be balanced. No `split_group_id` spans
  two splits, and every value has training criteria.
- **Snapshot**: the same proportions are recomputed on accepted examples,
  because rejections and exhausted budgets can shift them, and written to
  `splits.parquet`. Out-of-tolerance drift fails the snapshot.
- **Cross-split near-duplicates**: when a snapshot is built, text similarity is
  checked between splits. A training example that nearly duplicates a
  validation or test example is excluded **from training** in that snapshot,
  never from the evaluation side, and the exclusions are recorded in the
  manifest. No data is changed.

### Human-verified sets

Synthetic items chosen for human review come only from `test` (or `heldout`).
Human-verified examples are never used for training.

## Tooling

- `duckdb` (Python) for the working store, SQL updates and Parquet export and
  import; `polars` for analysis if useful; Hugging Face `datasets` loads the
  files listed in a snapshot manifest for training.
- A schema module owns the table definitions and factor-value validation.
- A storage module owns every write: inserts, the allowed updates, export and
  import. Tests prove that disallowed updates are refused, that export is
  deterministic, and that import followed by export reproduces the same files.

## Publishing

The committed exports are the open-source dataset (through Git LFS). A
snapshot's files and manifest can also be mirrored as a Hugging Face dataset
repository for easy loading.

## Rejected alternatives

- **Append-only Parquet files as the working store** (this document's previous
  draft): Parquet cannot update rows, so statuses, counts and cleaned columns
  needed extra retirement files and rebuilt trees. DuckDB handles updates
  directly; Parquet stays the delivery format.
- **Committing the DuckDB file**: one binary that changes as a whole, so every
  commit stores a full new copy in LFS, with no diffs or merges. Exports are
  committed instead and the database is rebuilt from them.
- **Hash strings or name-based IDs** (an earlier draft): UUIDs are a standard
  type in DuckDB and Parquet readers; v5 keeps the deterministic property where
  it matters, v7 gives every created row a sortable, collision-free ID.
- **Random UUIDv4 for every row**: rebuilding the plan or the examples would
  give rows new IDs, breaking references and changing every export file.
- **One sample per spec with replicate numbers**: replaced by one criterion
  with a required count, which matches "generate only the remaining count".
- **Regenerating data on each run**: wastes API spend and changes data under
  existing experiments.
- **JSON Lines for raw responses**: only needed for append-only files; raw
  responses now live in `requests.response` and export to Parquet.
- **Random row-level splits**: prefixes of one conversation, samples of one
  criterion, and minimal pairs would land on both sides, inflating test scores.
- **Splitting after generation**: makes the test set depend on generation
  outcomes and invites re-splitting until results look good.
- **Hash-only assignment (`hash(group) mod 100`)**: deterministic, but with
  many small factor values it leaves some values over- or under-represented in
  validation and test.
- **Only storing the cleaned transcript**: cleaning loses casing, punctuation
  and speaker names for good. Originals are kept and the cleaned columns are
  derived.
- **Cleaning during generation**: makes every cleaner change a generation
  change. Gemini writes natural text; the cleaner runs afterwards.
- **Storing token IDs**: ties the data to one tokenizer.
- **CSV**: no nested types for turns, weak typing.
