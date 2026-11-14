# Project brief and setup

2026-11-14. Initial repository setup from the user's brief.

## Done

- Wrote `AGENTS.md`: brief, hard requirements (CPU, p99 < 20 ms, ≤ 50 ms only with
  justified accuracy gain, ≥ 100 languages, text-only), draft input/output
  contract, label definitions, language tiers, data-generation rules (Gemini 3.8
  Flash via `google-genai`), labnotes/milestone/git conventions.
- Copied `bin/create-labnotes` unchanged from the Vxpipe repository.
- Added milestone index and
  `milestones/multilingual-text-barge-in-classifier.md` with checkpoints A–E.

## Decisions and rationale

- Agent turn is truncated to what was actually played: the overlap position is
  the strongest context signal and the full planned text would leak future
  content the caller never heard.
- Partial ASR prefixes are first-class training examples because the runtime
  must decide before the caller finishes.
- A human-verified evaluation set is required; synthetic-only eval is circular.

## User decisions (same day, after review)

- Input is 2 or 3 turns (always at least 2), replacing the 5-turn window.
  Keeps input short and lets callers pass what they have.
- Licence checks dropped: everything is released under a permissive licence;
  Google allows Gemini output for synthetic training data.
- Tier A (human-verified by the user): English and Hindi first, then Kannada
  and Tamil. All other languages are unverified unless a reviewer appears.
- Text appears in native script or romanized, possibly mixed within one
  utterance (e.g. romanized Hindi followed by English or Devanagari).
- Latency split: model path (tokenization to output) p99 ≤ 20 ms; whole path
  including the Vxpipe boundary p99 ≤ 50 ms.
- Generator model ID: `gemini-3.8-flash`.
- AGENTS.md and the milestone were updated with all of the above.

## Open

- Reference CPU, final language list, serving boundary, dataset storage and
  the specific permissive licence.
- Repository is not yet a git repository.

## Deterministic generation plan (user decision, same day)

- The user chose to enumerate conversation categories and other factors
  (language, script, code-switching, intent/label, …) and generate data from
  those conditions, so generation conditions are deterministic.
- Recorded in AGENTS.md: versioned factor catalogue, constraint rules, seeded
  plan builder (stratified quotas + all-pairs coverage, since the full
  cross-product is too large), stable spec IDs on every record, labels derived
  from the intent factor rather than from Gemini, a judge pass that checks
  records against their spec, coverage reports and spec-level splits.
- Milestone Checkpoints A–C updated to match.

## Data format (user request, same day)

- The user asked for the generation/storage structure to be documented with
  training in mind, suggesting Parquet with factors as columns next to the
  input and label.
- Drafted `docs/data-format.md`: four stages (plan specs, raw Gemini responses,
  conversations, training examples). Parquet for all but raw responses, which
  use zstd JSON Lines because they must be appendable and resumable. Examples
  are partitioned by split and language, keep factor columns, store turns as
  structured lists, and are expanded by turn shape × prefix × noise variant.
  Human eval sets add annotation columns. Rejected alternatives recorded there.
- Added `.gitignore` for model artefacts, Python caches and `.env`.
- Status is draft; to be settled with the user in Checkpoint A.

## Deterministic splits (user decision, same day)

- The user wants train/validation/test splitting to be deterministic too, so
  validation and test don't get more of the same kind of sample.
- Documented in `docs/data-format.md#splits`: assignment happens in the plan
  builder before generation; whole groups (all replicates of a factor
  combination, plus minimal pairs) go to one split; groups are ordered by a
  seeded hash and assigned by iterative stratification so every factor value,
  label and language keeps the 80/10/10 ratio within ±2 points; a separate
  `heldout` split removes whole factor values from training. Post-generation
  audit and cross-split near-duplicate removal (from train only).
- Added `combination_id`, `replicate` and `split_group_id` columns.

## Data is committed to git (user decision, same day)

- The user wants generated data committed, not gitignored. Removed `/data/`
  from `.gitignore`; AGENTS.md, the data format doc and the milestone now say
  all stages (raw generations included) are committed, sharded under 50 MB per
  file. Git LFS is noted as the fallback if the repository grows too large.

## Git LFS for all Parquet and JSON (user decision, same day)

- The user requires every Parquet and JSON file to be stored in Git LFS.
  Added `.gitattributes` with `*.parquet`, `*.json`, `*.jsonl`, `*.jsonl.zst`;
  AGENTS.md, the data format doc and the milestone updated. Shards stay under
  100 MB.

## Idempotent, append-only generation (user decision, same day)

- User: the generation script must be idempotent: never overwrite or
  regenerate older data; only generate enough to fill the counts required for
  the factors/languages.
- Rewrote `docs/data-format.md` around one append-only pool: plan revisions
  that only add specs (prefix-stable quotas), per-run raw/conversation/
  rejection files, per-part example files, retirement files instead of
  deletion, and immutable snapshot manifests for training (replacing the
  earlier per-dataset folders). Each spec needs exactly one accepted
  conversation; runs fill only the deficit, resume unfinished work first, and
  a second run with nothing to fill does nothing.
- `combination_id` no longer includes the catalogue version, so adding values
  keeps existing IDs; removing values retires specs.
- Splits are assigned when a spec is created and never change; new groups are
  stratified against existing assignments.
- Sample weights moved out of example files into snapshot/training config.
- `.gitignore`: generation lock and `*.tmp` under `data/`.
- AGENTS.md and the milestone (new Checkpoint B task with tests-first list)
  updated.

## Transcript format and cleaning (user decision, same day)

- User: model input lines are `s<n>: <text>` (s1, s2, s1 or s1, s2, s3 in 3
  lines), punctuation removed, a single casing. Generation need not clean; keep
  both the original and the cleaned transcript as Parquet columns so cleanup
  can be re-run.
- Recommended lowercase (fewer, more common tokens; most scripts are uncased;
  STT casing is unreliable). Wrote `docs/transcript-format.md`: NFC, Python
  `str.lower()`, delete in-word apostrophes, replace other `P*` with spaces,
  keep letters, digits, marks, ZWJ/ZWNJ and symbols, collapse whitespace;
  speakers numbered by first appearance within the window.
- Cleaning lives in Stage 4 (derived, versioned via `builder=`), because Stage 3
  files are immutable under the append-only rule; Stage 3 keeps the original
  with `{speaker, role, text}` turns. Examples gained `turns_original`,
  `transcript_original`, `transcript_clean`, `speaker_labels`,
  `candidate_original`, `candidate_clean`, `cleaner_version`.
- Flagged: removing `?` loses a question cue on agent lines (option for a later
  cleaner version). Open: which speaker is the agent, especially in three-speaker
  windows, and whether three-speaker conversations need a factor.

## DuckDB working store, fill-to-count generation (user decision, same day)

- User: since Parquet cannot modify rows, use DuckDB to read/write rows and
  cells, and treat Parquet as the delivery format. Generation checks each
  criterion for existing data and generates only the remaining count; if it is
  already met, it moves to the next criterion.
- Rewrote `docs/data-format.md`: tables plan_revisions, targets, requests,
  conversations, rejections, examples, annotations in `data/work/barge_in.duckdb`;
  deterministic partitioned Parquet export to `data/export/` (committed, LFS)
  at the end of every run; import rebuilds the database from exports.
- Decided not to commit the DuckDB file: it is one binary that changes as a
  whole, so every commit would add a full new LFS copy, with no diffs. The
  exports are the durable record. `.gitignore` now ignores `data/work/`.
- Criteria replace replicate-numbered specs: one targets row per combination
  with `required_count`; `need = required − accepted − in flight`.
- Allowed updates are limited (required counts, active/retired status, cleaned
  columns, derived examples); generated text, factor values and splits never
  change. Retirement is a status update; the criterion is refilled.
- Raw responses moved from JSON Lines into `requests.response`. The `*.jsonl`
  LFS rules stay as the user asked for all JSON files.
- AGENTS.md, milestone (two Checkpoint B tasks), transcript format and
  catalogue comments updated.

## Speaker roles (user decision, same day)

- User: the model is not told which speaker is the agent; just know the last
  line is the transcript to judge. Recorded in `docs/transcript-format.md`
  ("Speakers and roles"), AGENTS.md and the milestone. Roles stay in the
  original turns for generation and example building only.
- Still open: whether to add a catalogue factor for three-participant
  conversations so `s1, s2, s3` windows exist in training data.

## UUID row IDs (user decision, same day)

- User asked for an ID on every row, suggesting UUIDs.
- Every table now has a UUID primary key: UUIDv7 for created rows (revisions,
  runs, requests, conversations, rejections, annotations, snapshots), UUIDv5 for
  reproducible rows (criteria via `combination_id`, `split_group_id`, examples)
  so rebuilding the plan or examples keeps IDs and does not churn the export.
- Fixed project namespace for UUIDv5: `3de823b9-152c-4f71-899d-8d2bde9cc65c`
  (generated with `uuid.uuid4()`; must never change). Derivation strings are
  documented in `docs/data-format.md#identifiers`.
- Added a `runs` table and `rejection_id`/`annotation_id` keys; human names are
  separate `name` columns. DuckDB `UUID` type; exported as lowercase strings.
