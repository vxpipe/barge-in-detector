# Data generation and storage format

Status: draft decision, 2026-11-14. Settle it in milestone Checkpoint A, before
the Checkpoint B generator writes any data. Column lists change only through a
schema version bump recorded here.

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
    targets/language=hi/part-0.parquet
    requests/language=hi/run_id=<run_id>/part-0.parquet
    conversations/language=hi/run_id=<run_id>/part-0.parquet
    rejections/language=hi/run_id=<run_id>/part-0.parquet
    examples/builder=<version>/split=train/language=hi/run_id=<run_id>/part-0.parquet
    annotations/language=hi/part-0.parquet
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
| Generated text and everything in `requests.response` | `targets.required_count` (raised or lowered by a plan revision) |
| Factor values, `combination_id`, label | `targets.status` and `conversations.status` (`active`/`retired`) with reason and time |
| A criterion's `split` and `split_group_id` | Cleaned columns and `cleaner_version` in `examples` |
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

1. Open the database (DuckDB allows one writer, so two runs cannot overlap).
2. Finish earlier work first, without new API calls: collect results of
   submitted batch jobs recorded in `requests`, and process stored responses
   that were never turned into conversations or rejections.
3. Go through active criteria in a fixed order (plan priority, then
   `combination_id`). For each one, count accepted (non-retired)
   conversations and requests in flight, and compute `need`.
4. Skip the criterion if `need ≤ 0`, or if it has used its attempt budget
   (`max_attempts_per_sample × required_count` attempts, counted from
   `requests`). Rejected attempts use budget but never count toward the
   required count.
5. Otherwise request `need` samples. Each request row is inserted before the
   call is made, so a crash never loses track of a request.
6. Validate responses, run the blind judge, check that the labels agree (see
   [Labels](#labels)) and deduplicate; insert accepted conversations and
   their examples, or rejections, in the same transaction that marks the
   request processed.
7. Export, then report per language, label and factor value: filled, still
   short, budget exhausted, and cost.

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
3. A separate **blind judge call** classifies the same sample without seeing the
   intended label or the generator's answer.
4. The sample is accepted only if the generator's classification, the judge's
   classification and the intended label all agree. Otherwise it is rejected
   with stage `label_mismatch`; the criterion stays short and a later attempt
   fills it. Mismatch rates are reported per intent and language, which shows
   where the intent definitions or prompts are unclear.

The stored training label is Gemini's classification; per-prefix labels in
`examples` follow from the classified barge-in word.

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
| `combination_id` | v5 | `criterion:` + sorted `factor=value` pairs joined by `;`. Identifies one criterion. Independent of the catalogue version, so it is stable as the catalogue grows |
| `split_group_id` | v5 | `split_group:` + the smallest `combination_id` in the group |
| `request_id` | v7 | one Gemini request attempt |
| `conversation_id` | v7 | one accepted conversation |
| `rejection_id` | v7 | one rejected attempt |
| `example_id` | v5 | `example:` + `conversation_id`, turn shape and prefix index. Independent of the builder and cleaner versions, so annotations stay attached when cleaning changes |
| `annotation_id` | v7 | one human review entry |

UUIDs are generated in Python (`uuid.uuid5`; `uuid.uuid7` from Python 3.14, or
a small dependency on older versions), so tests can check the derivations.
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
| `contrastive_group_id` | VARCHAR | links minimal pairs; null otherwise |
| `split_group_id` | UUID | |
| `split` | VARCHAR | `train`, `validation`, `test`, `heldout`; fixed when the criterion is created, see [Splits](#splits) |

## Tables

### `plan_revisions`

`revision_id` (primary key), `name`, `created_at`, `quotas` (JSON), `split_ratios`
(JSON), `heldout` (JSON: factor values held out), `seed`, and the factor
catalogue, constraints and language manifest versions.

### `runs`

`run_id` (primary key), `name`, `revision_id`, `started_at`, `finished_at`,
`generator_model_id`, `judge_model_id`, settings (JSON: limit, budget, attempt
budget), requests made, conversations accepted, rejections, `cost_usd`.

### `targets`

Common factor columns, plus:

| Column | Type | Notes |
| --- | --- | --- |
| `combination_id` | UUID | primary key |
| `required_count` | INTEGER | samples needed for this criterion |
| `quota_group` | VARCHAR | the quota that selected this criterion, e.g. `pairwise`, `hard_case`, `contrastive_pair` |
| `priority` | INTEGER | generation order; lower first |
| `added_in_revision`, `count_set_in_revision` | UUID | plan revisions that created it and last set its count |
| `status` | VARCHAR | `active` or `retired` |
| `retired_reason`, `retired_at` | VARCHAR, TIMESTAMP | |

### `requests`

| Column | Type | Notes |
| --- | --- | --- |
| `request_id` | UUID | primary key |
| `combination_id`, `run_id` | UUID | |
| `language` | VARCHAR | |
| `batch_job_id` | VARCHAR | null for direct calls |
| `model_id`, `prompt_template`, `prompt_hash` | VARCHAR | |
| `status` | VARCHAR | `submitted`, `received`, `processed`, `api_error`, `schema_invalid` |
| `response` | JSON | the raw structured response exactly as returned |
| `input_tokens`, `output_tokens` | INTEGER | |
| `cost_usd` | DOUBLE | |
| `created_at`, `completed_at` | TIMESTAMP | |

Never store API keys or request headers.

### `conversations`

Common factor columns, plus:

| Column | Type | Notes |
| --- | --- | --- |
| `conversation_id` | UUID | primary key |
| `request_id`, `run_id` | UUID | provenance |
| `turns` | STRUCT(speaker VARCHAR, role VARCHAR, text VARCHAR)[] | all turns before the overlap, oldest first, exactly as generated; `speaker` is the name or role Gemini used, `role` is `agent`, `caller` or `other` |
| `agent_full_text` | VARCHAR | what the agent intended to say in the overlapped turn, as generated |
| `agent_spoken_text` | VARCHAR | the part spoken before the caller started, as generated |
| `candidate_speaker` | VARCHAR | as generated |
| `candidate_text` | VARCHAR | the caller's full final utterance, as generated |
| `intended_label` | VARCHAR | from the criterion's intent |
| `gemini_label` | VARCHAR | the generator's classification of the final utterance |
| `gemini_rationale` | VARCHAR | the generator's short reason |
| `barge_in_from_word` | INTEGER | the generator's word index where a developing utterance becomes barge-in; null if it never does |
| `judge_label` | VARCHAR | the blind judge's classification |
| `judge_barge_in_from_word` | INTEGER | the judge's word index, for agreement checks |
| `model_id`, `prompt_template`, `judge_model_id` | VARCHAR | |
| `judge_notes` | VARCHAR | |
| `status` | VARCHAR | `active` or `retired` |
| `retired_reason`, `retired_at` | VARCHAR, TIMESTAMP | |
| `created_at` | TIMESTAMP | |

### `rejections`

`rejection_id` (primary key), `request_id`, `combination_id`, `language`,
`run_id`, `stage` (`schema`, `judge`, `label_mismatch`, `dedupe`), `reason`,
`gemini_label`, `judge_label`, `created_at`.

### `examples`

Built deterministically from active conversations: one row per (turn shape ×
candidate prefix). Prefixes are cut from the original candidate words, then
each window is cleaned. Inputs are cleaned-up transcripts, so no ASR-noise
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
| `speaker_labels` | VARCHAR[] | the `s<n>` label of each line, e.g. `[s1, s2, s1]` |
| `candidate_original` | VARCHAR | the candidate prefix as generated |
| `candidate_clean` | VARCHAR | the candidate prefix after cleaning |
| `cleaner_version` | VARCHAR | |
| `prefix_index` | SMALLINT | 0-based word count of the prefix |
| `is_final_prefix` | BOOLEAN | |
| `label` | VARCHAR | label for this prefix (a prefix of a developing interruption can be `backchannel`) |
| `label_id` | TINYINT | 0 = backchannel, 1 = barge-in |

Sample weights for rebalancing belong to the snapshot or training config, not
the examples table.

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
2. New groups are ordered by SHA-256 of (`seed`, `split_group_id`), which is
   deterministic but unrelated to factor values.
3. Each new group is assigned in that order to the split that is furthest below
   its target share for the group's factor values and label, weighted by
   `required_count` and **counting all existing criteria as already assigned**
   (iterative stratification). This keeps every factor value, every label and
   every language at roughly the target ratio in each split as the data grows,
   so validation and test don't end up with more of one kind.
4. Every factor value present in validation or test must also be present in
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

- **Plan revision**: each factor value, label and language is within ±2
  percentage points of the target ratio in every split (by required count), or
  the revision fails; no `split_group_id` spans two splits.
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
