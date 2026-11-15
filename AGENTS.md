# Barge-in detector

A small, CPU-only, multilingual text classifier that decides whether a caller's
words spoken **while a voice agent is talking** are a backchannel (let the agent
keep speaking) or a genuine barge-in (stop the agent and give the caller the
floor). It is trained on synthetic utterances generated with Google Gemini and
is consumed by Vxpipe's speech runtime.

## Repository ownership

This GitHub repository is the canonical development workspace for dataset
creation, cleaning, splitting/review, training, inference, evaluation, CPU
benchmarks, tests, documentation, labnotes and milestones. Keep a versioned copy
of the dataset here. Model weights and local checkpoints stay gitignored.

Hugging Face repositories are release destinations:
- `HashNuke/barge-in-detector`: model weights, tokenizer/configuration, model
  card, licence and provenance only; no scripts or datasets.
- `HashNuke/barge-in-detector-data`: versioned data, dataset card, licence and
  metadata only; no generation/training code, tests or labnotes.

The sibling HF checkouts are release staging directories, not development
workspaces. Link every model release to the exact dataset revision and GitHub
code commit. Publish source code before removing code from release repositories.
All implementation/research uses this repo's AGENTS.md, bin/create-labnotes and
labnotes/milestones/index.md. No history rewriting or force-push is needed.

## Project brief

Voice agents currently treat any caller transcript during agent playback as an
interruption. Callers constantly produce acknowledgments ("hmm", "uh-huh",
"ok", "yeah yeah", "oui oui", Kannada "ಹೌದು"/"houdu"/"oudhu") that should not
cut the agent off, while short utterances such as "stop", "wait" or "no, the
other one" must interrupt immediately. The same word can be either, depending
on context: "yes" during an explanation is a continuer; "yes" after the agent
asks "shall I book it?" is an answer; "yes yes, but change the time" begins as a
backchannel and becomes an interruption.

Vxpipe already knows whether the caller speaks over the agent or after it
finished, and handles answers itself. The detector is called only for speech
**over** the agent: it reads the caller's utterance and returns a calibrated
probability that it is a barge-in (user decision, 2026-11-15; see
[docs/utterance-first-decision.md](docs/utterance-first-decision.md)).

## Requirements

### Hard requirements

- **CPU inference only.** No GPU at serving time. Training may use GPUs
  (Vast.ai instances; see the user-level instructions for SSH access).
- **Latency, measured at p99 on the reference CPU, batch size 1, warmed:**
  - **Model path ≤ 20 ms**: from the start of tokenization to the output
    probability. Faster is better.
  - **Whole path ≤ 50 ms**: including the Vxpipe integration boundary
    (serialization, NIF or sidecar call, queueing) around the model path.
- **Multilingual: 14 languages** (user decisions, 2026-11-15, cutting the 36
  of 2026-11-14), listed explicitly in the versioned language manifest with
  tiers (see below): English (US, India, Singapore), Hindi, Kannada, Tamil,
  Telugu, Malayalam, Mandarin, Japanese, Indonesian (Bahasa
  Indonesia), Filipino, French, German, Spanish, Italian. Dropped: Dutch,
  Hungarian, Swedish, Danish, Norwegian, Finnish, Irish, Portuguese, Thai,
  Assamese, Arabic, Vietnamese, Korean, Marathi, Gujarati, Punjabi, Bengali,
  Malay, Polish, Russian and Cantonese. "Supports a
  language" means it has evaluation results, not that the tokenizer covers it.
  Rollout starts with **English and Hindi**, then Kannada and Tamil, then the
  remaining languages.
- **Text input only** (ASR transcripts). Prosody/audio is out of scope for v1.
- **Context-aware, not a word list.** Never implement or ship a universal
  ignored-word list as the classifier. Lexical shortcuts are what the model must
  learn to overrule.

### Input contract

- **The caller's overlapping utterance**, cleaned (lowercase, punctuation
  removed, any script mix; [docs/transcript-format.md](docs/transcript-format.md)),
  possibly a *partial/interim* ASR transcript. The runtime re-evaluates on each
  revision, so the model must give sensible, monotone-ish answers on prefixes
  ("yes" → "yes yes but").
- A **64-token input budget** (user decision, 2026-11-15): enough for a whole
  sentence; longer input keeps its first 64 tokens. Measure 32 only if 64
  misses the latency gates; 128 is not needed if 64 meets them.
- **Generation does not clean.** Gemini writes natural text with casing and
  punctuation; a versioned cleaner produces the model input afterwards, and
  both are stored.
- Optional language hint (BCP-47). The model must not depend on it.
- A preceding agent line is not part of the contract unless the milestone's
  ablation shows it is worth it.

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
- Answers to a question the agent finished asking are not this model's job:
  the runtime handles speech after the agent stops.
- **Two labels only** (user decision, 2026-11-15): every item is backchannel or
  barge-in; there is no ambiguous class.

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
- **Scripts:** each language is generated in its native script only (user
  decisions, 2026-11-15: no romanized or mixed-script forms, no pinyin),
  alone or mixed with English in Latin letters. Words of an Indian language
  mixed into Indian English are in that language's native script. The manifest records each
  language's native script(s) and `romanized_usage`.

### Things that are easy to get wrong

- Synthetic evaluation alone is circular. Keep a held-out, human-verified
  evaluation set (optionally including permitted real conversations) and report
  on it separately from synthetic test splits.
- **Inputs are cleaned-up transcripts.** Transcription errors and noise are
  handled upstream, so generate clean text and do not simulate ASR noise
  (misrecognitions, garbled words, missing punctuation). Script mixing and
  code-switching are part of how callers speak and stay in scope.
- Spread every criterion's samples over train, validation and test (user
  decision, 2026-11-15; [docs/training.md](docs/training.md)), and deduplicate
  on the cleaned text, so the same utterance never lands in two splits.
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
   in [docs/factor-catalogue.md](docs/factor-catalogue.md)); the utterance
   track's factors are in `generation/factors/utterances.yaml`: language and
   regional variant, script form, code-switching (none or mixed) and caller
   intent. Each intent maps to an
   intended label (backchannel or barge-in).
2. **Constraint rules** remove impossible combinations
   (`generation/factors/constraints.yaml` and the track's own rules).
3. **The plan file** (`generation/plans/*.yaml`) and a seeded, deterministic,
   prefix-stable builder give the **criteria**: factor combinations, each with
   a required sample count. The plan file in git is the only record
   of the criteria (user decision, 2026-11-15).
4. **IDs**: a criterion's `combination_id` is a UUIDv5 of its factor values;
   samples and requests get UUIDv7s. See
   [docs/data-format.md#identifiers](docs/data-format.md#identifiers).
5. **Generation**: prompts are rendered from the criterion only. Gemini's output
   is not deterministic, but the conditions, labels and their counts are.
6. **Judge**: the Gemini Decisions API asks one plain yes/no question per
   criterion of the sample (one per language for language and script), never
   about the label. A sample is accepted only when every answer is yes.

### Data format and storage

[docs/data-format.md](docs/data-format.md) defines the columns. In short:
**one table, `utterances`, holds every generated sample**, accepted or
rejected, with its factor columns, original and cleaned text, label, raw sample,
the judge's raw answers and the reason. It is exported as **one CSV**,
`data/export/utterances.csv`, committed in plain git. A local request log
(`requests`: raw responses, tokens, costs, retries) stays in the DuckDB file
(`data/work/`, gitignored) and is not committed. Never store token IDs. All
writers use the storage module; changing a column requires a schema version
bump recorded in the data format doc.

### Generator rules

- Use the `google-genai` Python package. The generator is **Gemini 3.8 Flash**
  (`gemini-3.8-flash`, user decision, 2026-11-15); the model is the user's call,
  made in `generation/config.yaml`, and every request records it. The judge is
  the Gemini Decisions API (`gemini-decision-1`, `client.decisions.create`,
  `google-genai` 3.3+). See [docs/generation.md](docs/generation.md).
- Read the API key from the environment (`GEMINI_API_KEY`/`GOOGLE_API_KEY`).
  Never commit, print or log keys, raw authorization headers or `.env` contents.
- Use structured output and validate every record in code. Reject, don't
  repair silently.
- Prompts are code: version them under `generation/prompts/` and test their
  rendering. Prefer the Batch API for very large runs.
- **Generation fills each criterion to its required count** (details in
  [docs/data-format.md](docs/data-format.md)):
  - `need = required_count − accepted − in flight`; generate only `need`;
  - never overwrite or regenerate existing generated text; the only update is
    accepted → retired (with a reason) for bad data, which leaves its criterion
    short for a replacement; rows are never deleted;
  - running twice with nothing short makes no API calls and leaves the CSV
    unchanged;
  - more data means raising counts or criteria in the plan file; existing
    criteria keep their IDs and splits;
  - resume interrupted work before making new requests.
- **Generated data is committed to git** as `data/export/utterances.csv`, with
  the code that produced it. Every generation run ends with an export. Never
  edit the CSV by hand; change data only through the storage module and
  re-export. Check the CSV contains no keys or headers before committing. Split
  it into numbered files before it reaches 100 MB.

## Tooling

- **Vast.ai:** run training and remote setup/work commands inside the tmux
  session named `gpubox` so work survives SSH disconnects. Create it if absent;
  reconnect with `tmux attach -t gpubox`. Keep logs and model artifacts on disk.
  Always rent an **RTX 5090 (32 GB VRAM)**: one GPU on one datacenter
  instance, below $0.65/hour, with good internet connectivity and the Vast.ai
  PyTorch image. Do not substitute another GPU model or rent multiple instances.

- Use installed `agent-browser` for browser checks. Do not install or use
  Playwright or disable browser sandboxing as a workaround.
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
  `generation/` (`utterances.py`: track and plan builder; `prompts.py`;
  `responses.py`: response validation; `utterance_criteria.py`: per-language
  prompt lines and the judge questions; `llm.py`: Gemini client;
  `utterance_runner.py`: the fill-to-count runner), `storage/` (DuckDB schema,
  the storage module, CSV export and import), `splitting.py` (dataset split assignment and CSV IO), `review/` (binary
  owner annotation UI), `training/` (model data loading, metrics, train,
  evaluate and infer), `cli.py`. Benchmark scripts live in `experiments/`.
  Model dependencies use `uv sync --group train`. Prompt templates live
  in `generation/prompts/`, plans in `generation/plans/`, run settings in
  `generation/config.yaml`.
- Data commands run through `uv run python -m barge_in_detector …`:
  `check-catalogue`, `plan`, `generate` (`--plan`, `--dry-run`, `--limit`,
  `--languages`, `--budget-usd`, `--concurrency`), `status`, `show-prompt`,
  `split`, `export`, `import`. See [docs/generation.md](docs/generation.md).
  Training and evaluation run here with `python -m barge_in_detector.training.train`
  and `.evaluate`; see [docs/training.md](docs/training.md).

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

All dataset and model labnotes/milestones live here. HF release repos link to
these instructions and contain no labnotes helper scripts or development history
in their current release trees. Earlier ownership decisions in historical notes
are superseded by the current Repository ownership section.

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
