# Barge-in detector

A small, CPU-only, multilingual text classifier that decides whether a caller's
words spoken **while a voice agent is talking** are a backchannel (let the agent
keep speaking) or a genuine barge-in (stop the agent and give the caller the
floor). It is trained on synthetic conversations generated with Google Gemini
and is consumed by Vxpipe's speech runtime.

## Project brief

Voice agents currently treat any caller transcript during agent playback as an
interruption. Callers constantly produce acknowledgments ("hmm", "uh-huh",
"ok", "yeah yeah", "oui oui", Kannada "ಹೌದು"/"houdu"/"oudhu") that should not
cut the agent off, while short utterances such as "stop", "wait" or "no, the
other one" must interrupt immediately. The same word can be either, depending
on context: "yes" during an explanation is a continuer; "yes" after the agent
asks "shall I book it?" is an answer; "yes yes, but change the time" begins as a
backchannel and becomes an interruption.

The detector reads a short text window of the conversation and returns a
calibrated probability that the most recent caller utterance is a barge-in.

The training data is not written from an agent's point of view: it is
conversations between any people in the catalogue's situations (a hotel
manager and a guest, David and Irina), where one person has the floor (the
speaker) and another says the last line (the listener). Names never reach the
model; the cleaner turns them into `s1`, `s2`, `s3`.

## Requirements

### Hard requirements

- **CPU inference only.** No GPU at serving time. Training may use GPUs
  (Vast.ai instances; see the user-level instructions for SSH access).
- **Latency, measured at p99 on the reference CPU, batch size 1, warmed:**
  - **Model path ≤ 20 ms**: from the start of tokenization to the output
    probability. Faster is better.
  - **Whole path ≤ 50 ms**: including the Vxpipe integration boundary
    (serialization, NIF or sidecar call, queueing) around the model path.
- **Multilingual: 36 languages** (user decision, 2026-11-14, replacing the
  earlier "at least 100"), listed explicitly in the versioned language manifest
  with tiers (see below): English, Hindi, Kannada, Tamil, Telugu, Malayalam, Bengali, Gujarati, Marathi, Punjabi, Urdu, Assamese, Mandarin, Cantonese, Japanese, Korean, Malay (Bahasa Malaysia), Indonesian (Bahasa Indonesia), Filipino, Thai, Vietnamese, Arabic (Modern Standard), French, German, Spanish, Portuguese, Italian, Dutch, Irish, Swedish, Danish, Norwegian (Bokmål), Finnish, Hungarian, Russian, Polish. "Supports a
  language" means it has evaluation results, not that the tokenizer covers it.
  Rollout starts with **English and Hindi**, then Kannada and Tamil, then the
  remaining languages.
- **Text input only** (ASR transcripts). Prosody/audio is out of scope for v1.
- **Context-aware, not a word list.** Never implement or ship a universal
  ignored-word list as the classifier. Lexical shortcuts are what the model must
  learn to overrule.

### Input contract (draft; settle in the milestone's contract checkpoint)

**Superseded for the model input (user decision, 2026-11-15):** Vxpipe already
knows whether the caller speaks over the agent or after it finished, so the
model only classifies the caller's **overlapping utterance** as backchannel or
barge-in. See [docs/utterance-first-decision.md](docs/utterance-first-decision.md)
and the utterance-first milestone. The conversation contract below is kept for
the existing data and the planned context slice.

- **2 or 3 turns**, oldest first, as a cleaned transcript: one `s<n>: <text>`
  line per turn, speakers numbered by first appearance (`s1, s2, s1` or
  `s1, s2, s3`), lowercase, punctuation removed. The last line is always the
  utterance to judge. The model is not told which speaker is the agent; earlier
  lines are context. Callers pass one or two preceding turns,
  depending on what they have. The model is trained and evaluated on both shapes
  and must not require the third turn. Exact rules:
  [docs/transcript-format.md](docs/transcript-format.md).
- **Generation does not clean.** Gemini writes natural text with casing,
  punctuation and speaker names or roles. A versioned cleaner produces the
  model input afterwards; examples keep both original and cleaned columns.
- The **agent's in-flight turn is truncated to what had actually been played**
  when the caller started speaking, not the full planned response. This is the
  single most informative feature (mid-sentence vs. just asked a question).
- An `agent_state` field: `speaking`, `just_finished` (within a short grace
  window), or `silent`. Barge-in is only meaningful for the first two.
- The **candidate caller utterance**, which may be a *partial/interim* ASR
  transcript. The runtime re-evaluates on each revision, so the model must give
  sensible, monotone-ish answers on prefixes ("yes" → "yes yes but").
- Optional language hint (BCP-47). The model must not depend on it.
- A hard token budget (initially 128 tokens; measure 96/128/192) with
  left-truncation of the oldest turn first, then the start of the agent turn;
  never truncate the candidate utterance.

### Output contract (draft)

- `p_barge_in` in [0, 1], calibrated. The runtime owns thresholds, which may be
  per language tier. Optionally a third "need more words" signal may be derived
  from probabilities in the middle band rather than trained as a class; decide
  with evidence.

### Label definitions (must be written down before data generation)

- **Backchannel**: continuers, acknowledgments, agreement tokens, surprise or
  empathy tokens, laughter tokens, repetitions of these, and filler noise that
  does not claim the floor.
- **Barge-in**: stop/wait requests, corrections, disagreement, clarification
  questions, new requests or topic changes, early answers that make the rest of
  the agent turn moot, and backchannels that develop into any of these.
- **Out of scope**: echo of the agent's own voice, background speakers, TV or
  radio, and transcription glitches from noise. Noise cancellation and audio
  cleaning upstream handle these; every input to this model is assumed to be the
  caller speaking to the agent.
- Ambiguous items are labelled ambiguous in evaluation data and reported
  separately rather than forced into a class.

### Language manifest

- Keep `data/languages.yaml` (or equivalent) as the source of truth: ISO/BCP-47
  code, name, script(s), tier, common romanization/transliteration, and whether
  native-speaker verification exists.
- Tier A: human-verified evaluation sets. Only **English, Hindi, Kannada and
  Tamil** can be verified by the project owner, so these are the Tier A
  languages; English and Hindi come first.
  Tier B: generation quality spot-checked by a native speaker if one becomes
  available.
  Tier C: synthetic-only; results reported but flagged as unverified.
- **Scripts:** each language appears in its usual native script **or** in
  romanized form (Latin/English characters), and utterances may mix them, e.g.
  romanized Hindi followed by English or by Devanagari Hindi. Every language in
  the manifest records its native script(s) and whether romanized usage is
  common; generation and evaluation cover native, romanized and mixed forms.

### Things that are easy to get wrong

- Synthetic evaluation alone is circular. Keep a held-out, human-verified
  evaluation set (optionally including permitted real conversations) and report
  on it separately from synthetic test splits.
- **Inputs are cleaned-up transcripts.** Transcription errors and noise are
  handled upstream, so generate clean text and do not simulate ASR noise
  (misrecognitions, garbled words, missing punctuation). Script mixing and
  code-switching are part of how callers speak and stay in scope.
- Split train/validation/test by **conversation and scenario template**, never by
  row, so near-duplicate generated dialogues do not leak across splits. Also keep
  some languages and scenario categories fully held out to measure generalization.
- Build **contrastive minimal pairs**: same context with "yes" as backchannel vs.
  as an answer; same utterance at different agent truncation points.
- Measure the latency of the *whole* path on the *reference* CPU, warmed, at
  p50/p95/p99. Laptop numbers do not count. Vocabulary/embedding size dominates
  model size for multilingual encoders; account for memory too.
- Do not let class balance in training data drive the decision threshold.
  Calibrate on validation data that reflects realistic priors.
- The model, code and generated data will be released as open source under a
  permissive licence. Prefer base models whose licences are compatible with
  that (e.g. MIT/Apache-2.0).

## Synthetic data generation

### Deterministic generation plan

The conditions for every generated sample are decided by project code, not by
the LLM. Gemini only writes text that satisfies a given specification.

1. **Factor catalogue** (versioned data under `generation/factors/`, explained
   in [docs/factor-catalogue.md](docs/factor-catalogue.md)): each factor and its
   allowed values: language and regional variant, script form, code-switching,
   domain and scenario, agent activity, cut position, caller intent,
   development over interim transcripts, repetition, length, register and
   caller mood. Each intent maps to an **intended label**
   (backchannel, barge-in, or ambiguous for evaluation only) that steers
   generation. **The training label is Gemini's classification**: the generator
   returns each sample with its classification, a blind judge classifies it
   again, and the sample is accepted only if both agree with the intended label
   (see [docs/data-format.md#labels](docs/data-format.md#labels)).
   Turn shape and interim prefixes are not spec factors; they are derived from
   each conversation later.
2. **Constraint rules** remove impossible or meaningless combinations (e.g.
   "answer to a question" requires the agent to have asked one; romanized form
   only for languages where it is used; code-switching only where plausible).
   They live in `generation/factors/constraints.yaml`.
3. **Plan builder**: a seeded, deterministic function from (factor catalogue
   version, constraints version, quotas, seed) to a list of **criteria**: factor
   combinations, each with a required sample count. The full cross-product is far too large, so the plan uses stratified
   quotas plus all-pairs coverage of factors (every pair of factor values
   appears at least N times), with extra quotas for hard cases and contrastive
   pairs. Same inputs always produce the same plan.
4. **IDs**: every row in every table has a UUID primary key. Rows the project
   creates (requests, conversations, runs, …) get time-ordered UUIDv7s; rows
   that must be reproducible (criteria and examples) get UUIDv5s derived from
   their content in a fixed project namespace. Each criterion's
   `combination_id` is derived from its factor values only, and every generated
   record carries it, so any sample can be traced back to its conditions. See
   [docs/data-format.md#identifiers](docs/data-format.md#identifiers).
5. **Generation**: prompts are rendered from the criterion only. Gemini's output is
   not deterministic, but the conditions, labels and their counts are.
6. **Conformance check**: a judge pass confirms that each record actually
   matches its criterion (language, script, intent, overlap point). Records
   that don't match are rejected, so the criterion stays short and a later
   attempt fills it; rejection rates are reported per factor value.
7. **Coverage reports** come from the plan and the accepted records: counts per
   factor value and factor pair, so gaps are visible before training.
8. **Splits come from the plan** and are deterministic: decided before
   generation, by whole groups (all samples of one criterion plus any minimal
   pairs), stratified so each split has the same proportion of every
   factor value, label and language. A separate held-out slice removes whole
   factor values from training. See
   [docs/data-format.md#splits](docs/data-format.md#splits).

### Data format and storage

[docs/data-format.md](docs/data-format.md) defines the tables (plan revisions,
targets, requests, conversations, rejections, examples, annotations), the
directory layout, IDs and every column. In short: **DuckDB is the working
store** (SQL reads, inserts and the allowed row/cell updates); **Parquet is the
delivery format**, written by a deterministic export to `data/export/` and
committed. **Factor columns are stored next to the input and the label.** The
DuckDB file is local and gitignored; it can be rebuilt from the exports.
Conversations keep the original text as
`{speaker, role, text}` turns, never modified; examples store both the original
and the cleaned transcript (see
[docs/transcript-format.md](docs/transcript-format.md)). Never store token IDs.
All writers and loaders
use the shared schema module. Changing a column requires a schema version bump
recorded in that document.

### Generator rules

- Use the `google-genai` Python package with **Gemini 3.5 Flash-Lite**
  (model ID `gemini-3.5-flash-lite`) as the default generator (user decision,
  2026-11-14, for cost); switching to **Gemini 3.8 Flash** (`gemini-3.8-flash`)
  is the user's call, made in `generation/config.yaml`. Record the model ID in
  each conversation and snapshot manifest. The blind judge is the Gemini
  Decisions API (model `gemini-decision-1`, `client.decisions.create`, needs
  `google-genai` 3.3+): one request per sample with closed questions and
  calibrated probabilities (see [docs/generation.md](docs/generation.md)).
- Read the API key from the environment (`GEMINI_API_KEY`/`GOOGLE_API_KEY`).
  Never commit, print or log keys, raw authorization headers or `.env` contents.
- Use structured output (`response_schema`/JSON) and validate every record
  against a project schema. Reject, don't repair silently.
- Every training snapshot has a manifest as defined in `docs/data-format.md`.
- Prompts and taxonomies are code: version them under `generation/` and test
  their rendering. Prefer the Batch API for large runs.
- **Generation fills each criterion to its required count** (details in
  [docs/data-format.md](docs/data-format.md#generation-fill-each-criterion-to-its-required-count)):
  - for each criterion, `need = required_count − accepted − in flight`; if
    `need ≤ 0`, move on to the next criterion; otherwise generate only `need`;
  - never overwrite or regenerate existing generated text. Only the updates
    listed in the data format doc are allowed (required counts, active/retired
    status, cleaned columns, derived examples);
  - running twice with nothing short makes no API calls and leaves the export
    unchanged;
  - more data means a new plan revision that adds criteria or raises required
    counts; existing criteria keep their IDs and splits;
  - resume interrupted work (submitted batch jobs, unprocessed responses)
    before making new requests;
  - bad data is marked retired with a reason, never deleted; its criterion is
    then short and gets a replacement;
  - training reads an immutable snapshot manifest, not the live tables.
- **Generated data is committed to git** as the Parquet exports under
  `data/export/`, every table including raw responses, together with the code
  that produced it. Every generation run ends with an export.
  **Every Parquet and JSON file (`*.parquet`, `*.json`, `*.jsonl`,
  `*.jsonl.zst`) is stored in Git LFS**, as configured in `.gitattributes`.
  Do not remove or bypass those rules; run `git lfs install` once per machine,
  and check `git lfs ls-files` before committing data. Shard files to stay
  under 100 MB each.
  Never edit exported files by hand; change data only through the storage
  module and re-export. Check raw responses contain no keys or headers before
  committing.

## Tooling

- Python project managed with `uv` (`uv sync`, `uv run ...`). Add dependencies
  with `uv add` and commit `uv.lock` with dependency changes.
- Always source `~/.zshrc` or run commands through `zsh -lc` to get the
  preconfigured tools.
- Keep training, generation, evaluation and serving/export as separate modules
  with separate responsibilities. No `utils.py` dumping grounds.
- Inference export target is decided by the milestone (e.g. ONNX Runtime with
  int8 quantization). The consumer is Vxpipe (Elixir); the integration boundary
  (in-process NIF such as Ortex plus the `tokenizers` package, or a sidecar) is
  part of the 50 ms whole-path budget and must be measured, not assumed.
- Tests use `pytest`. Mark tests that call Gemini or other networks as
  `integration` and exclude them from the default run.
- Code layout (`barge_in_detector/`): `catalogue.py` (factor catalogue and
  spec validation), `ids.py`, `cleaner.py` (transcript cleaner),
  `examples.py` (example builder), `generation/` (plan builder, prompts,
  response validation, Gemini client, fill-to-count runner), `storage/`
  (DuckDB schema, the storage module, Parquet export and import), `cli.py`.
  The utterance track adds `generation/utterances.py` (track and plan builder),
  `generation/utterance_criteria.py` (per-language prompt lines and the
  criteria judge) and `generation/utterance_runner.py`.
  Prompt templates live in `generation/prompts/`, plans in
  `generation/plans/`, run settings in `generation/config.yaml`.
- Data commands run through `uv run python -m barge_in_detector …`: `check-catalogue`, `plan`,
  `generate` (`--dry-run`, `--limit`, `--languages`, `--budget-usd`),
  `plan-utterances`, `generate-utterances` (the utterance track, same flags),
  `status`, `show-prompt` (`--utterances`), `build-examples`, `export`, `import`. See
  [docs/generation.md](docs/generation.md).

## Related repositories

Sibling checkouts live alongside this repository under `../`. The Vxpipe
platform (`../qa`, `../vxpipe`) owns the speech runtime that will consume this
model; see its milestone
`labnotes/milestones/provider-turn-detection-and-contextual-barge-in.md`
(Checkpoint C) for the runtime-side acceptance scenarios this model must serve.

## Project hygiene

Work in small, coherent checkpoints that leave the repository usable. Keep the
implementation, focused tests and relevant documentation for a checkpoint
together.

Use red-green-refactor for behavior changes: write the smallest focused failing
test, implement the smallest change that passes it, then refactor while green.
Data-generation prompts, model experiments and benchmarks are verified with
recorded evidence (metrics, manifests, latency tables) rather than unit tests,
but deterministic code (schemas, parsers, splitters, tokenization, export,
serving) gets tests first.

For substantial research or architecture decisions (base model choice, label
definitions, input contract, export runtime), add a focused document under
`docs/` recording the decision, rejected alternatives, implications and
verification evidence.

### Labnotes

- For research and implementation tasks, create a labnotes file with
  `bin/create-labnotes` and a two-to-four-word hyphenated task name, e.g.
  `bin/create-labnotes base-model-latency`. It creates an empty
  `labnotes/YYYYMMDD-HHMM-task-name.md`.
- Use that file as checkpoint labnotes. Update it as work progresses with what
  worked, what did not, barriers encountered, workarounds, decisions and their
  rationale, and evidence (commands, metrics, latency tables, dataset manifest
  references, cost).
- Keep labnotes factual and useful to the next person resuming the task. They do
  not replace durable decisions under `docs/`.
- Do not create labnotes for simple status checks, read-only inspection, or a
  request that only runs an existing command or script.
- Commit labnotes with the work they document.
- `labnotes/` also holds planning records: implementation milestones in
  `labnotes/milestones/` and deferred issues in `labnotes/issues/`.
  `bin/create-labnotes` files stay at the top level of `labnotes/`.

### Milestones

- Specifications and progress checklists live in `labnotes/milestones/`. Start
  with [the milestones index](labnotes/milestones/index.md); its checklist owns
  the implementation order.
- Milestone filenames and titles are descriptive and unnumbered. Reordering or
  inserting a milestone must not require renaming files.
- Each milestone describes a runnable outcome, prerequisites, contracts,
  checkpoint tasks and acceptance gates. Read linked sources before starting.
- Track work with Markdown task lists (`- [ ]` / `- [x]`). A written spec is not
  an implemented milestone: leave the index entry unchecked until acceptance
  gates pass. Record blockers or approved scope changes instead of checking off
  partial work.
- Link each checkpoint's labnotes from the milestone so evidence is findable.

### Git and commit hygiene

- Inspect `git status --short` and relevant diffs before and after changes.
  Preserve unrelated work.
- Do not create commits unless the user asks. Each commit is one coherent
  checkpoint with its code, tests, docs, labnotes and lockfile changes.
- Concise imperative subjects; use the body for motivation and verification.
- Stage exact paths and inspect `git diff --cached`. No `git add -A` in a dirty
  worktree.
- Never commit credentials, PII, local absolute paths, model checkpoints,
  virtualenvs, caches or build artefacts. Generated datasets under `data/` are
  committed (see Synthetic data generation).
- Do not amend, rebase, reset, force-push, delete branches or undo commits
  unless the user explicitly requests that exact operation.

## Completion checks

Once the Python project exists, run before finishing code changes:

```shell
uv run ruff format --check
uv run ruff check
uv run pytest
```

For model or export changes, also run the latency benchmark on the reference
CPU and record model-path and whole-path p50/p95/p99 in the labnotes.
