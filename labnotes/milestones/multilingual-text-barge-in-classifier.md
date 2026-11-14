# Multilingual text barge-in classifier

Status: specified, 2026-11-14; requirements updated the same day after the
user's review. No research, data, training or benchmarks have been done yet.

## Runnable outcome

A CPU-only model that takes 2 or 3 conversation turns (the last being a possibly
partial caller transcript, with the agent turn cut at the point the caller
started speaking) and returns a calibrated barge-in probability. Tokenization to
output takes **at most 20 ms at p99** on the reference CPU, and the whole call
from Vxpipe takes **at most 50 ms at p99**. It is evaluated per language across
at least 100 languages, human-verified for English, Hindi, Kannada and Tamil,
and released as open source under a permissive licence.

## Prerequisites and sources

- Project requirements in [AGENTS.md](../../AGENTS.md).
- Runtime-side scenarios: Vxpipe milestone
  `labnotes/milestones/provider-turn-detection-and-contextual-barge-in.md`,
  Checkpoint C and its acceptance table (in `../qa` / `../vxpipe`).
- Prior art to review (verify claims from primary sources, do not assume):
  LiveKit's multilingual text turn detector, Pipecat Smart Turn (audio),
  academic backchannel prediction / overlap classification literature, and
  small multilingual encoders (multilingual MiniLM, multilingual-e5-small,
  mmBERT-small, distilled XLM-R, mDeBERTa-v3-xsmall/small).

## Requirements

Decided with the user on 2026-11-14.

### Input

- **2 or 3 turns**, oldest first, as a cleaned transcript of `s<n>: <text>`
  lines (speakers numbered by first appearance; lowercase; no punctuation). The
  last line is always the speaker being judged, and the model is not told who
  any speaker is. Windows are `s1, s2`, `s1, s2, s1` or `s1, s2, s3`; the
  `speaker_pattern` factor generates both 3-line patterns. Callers pass
  whichever shape they have. Train and evaluate on both shapes and report each separately.
  See [transcript format](../../docs/transcript-format.md).
- Generation writes natural text (casing, punctuation, speaker names or roles);
  a versioned cleaner derives the model input. Examples keep both the original
  and cleaned transcript columns.
- **Agent turn cut at the overlap point**: only the text that had been spoken
  when the caller started, not the full planned response.
- **`agent_state`**: `speaking`, `just_finished` (within a short grace window)
  or `silent`. Barge-in only applies to the first two.
- **Partial transcripts**: the candidate may be an interim ASR transcript, and
  the runtime re-evaluates on each revision. Train on every prefix with its own
  label (a prefix of an interruption can still be a backchannel).
- Optional language hint (BCP-47); the model must work without it.
- Token budget initially 128 (measure 96/128/192); truncate the oldest turn
  first, then the start of the agent turn, never the candidate utterance.

### Output

- `p_barge_in` in [0, 1], calibrated. The runtime owns thresholds (default and
  per tier). A "need more words" band may be derived from mid-range
  probabilities; decide with evidence before adding a third class.

### Labels

- Backchannel: continuers, acknowledgments, agreement, surprise/empathy tokens,
  laughter, repetitions of these, and filler that does not take the floor.
- Barge-in: stop/wait, corrections, disagreement, clarification questions, new
  requests or topic changes, early answers, and backchannels that turn into any
  of these.
- Out of scope: agent echo, background speech, media audio and transcription
  glitches from noise. Upstream noise cancellation and audio cleaning handle
  them; the model only judges the caller's own speech.
- Ambiguous evaluation items are labelled ambiguous and reported separately.

### Languages and scripts

- At least 100 languages in a versioned manifest with scripts, romanization
  usage and tier.
- Tier A, human-verified by the user: **English and Hindi first**, then
  **Kannada and Tamil**. Tier B: native-speaker spot checks if available.
  Tier C: synthetic-only, reported as unverified.
- Text appears in the language's usual native script **or** romanized (Latin
  characters), and may mix within one utterance: e.g. romanized Hindi followed
  by English, or by Devanagari Hindi. Cover native, romanized and mixed forms,
  including the Kannada "oudhu"/"houdu"/"ಹೌದು" variants and French "oui oui".

### Data quality

- Inputs are cleaned-up transcripts, so generated text is clean. ASR noise
  simulation is out of scope; native, romanized and mixed script and
  code-switching remain in scope.
- Contrastive minimal pairs (the same token as backchannel vs. as an answer; the
  same utterance at different overlap points) and hard negatives (short real
  interruptions such as "stop", "wait", "no").
- Split by conversation and scenario template, never by row; hold out some
  scenario categories (and, for Tier C, some languages) entirely.
- Synthetic evaluation alone is circular: Tier A languages need human-verified
  evaluation sets.

### Latency

- Model path: tokenization start to output, p99 ≤ 20 ms, batch 1, warmed, on
  the reference CPU. Faster is better.
- Whole path: including the Vxpipe integration boundary, p99 ≤ 50 ms.
- Measure memory as well; embedding tables dominate multilingual encoders.

### Licence and data

- Code, model and generated data released under a permissive licence. Google
  allows using Gemini output to generate synthetic training data. Prefer base
  models with compatible licences.
- Generator: `gemini-3.8-flash` via `google-genai`. All generated data,
  including raw generations, is committed to git with its manifests and the
  code that produced it. All Parquet and JSON files are stored in Git LFS.

## Open decisions (resolve in Checkpoint A)

- Reference CPU (vendor, generation, AVX2/AVX-512/ARM, cores, threads allowed).
- Final 100+ language list and Tier B availability.
- Serving boundary inside Vxpipe (in-process ONNX via an Elixir NIF vs. sidecar).
- Storage for model artefacts (datasets are committed to git); whether to
  mirror datasets on Hugging Face; licence choice (e.g. Apache-2.0 or MIT).
- Third output class vs. probability band.

## Checkpoint A — Contracts, taxonomy and baselines

- [ ] Record the reference CPU and a reproducible latency harness that reports
  model-path and whole-path p50/p95/p99 (warm-up, batch 1).
- [ ] Settle [the data format](../../docs/data-format.md) (stages, layout,
  IDs, columns) with the user before any data is written.
- [ ] Write a `docs/` decision covering the input/output contract, labels,
  `agent_state`, partial-transcript behaviour, 2/3-turn shapes, token budget and
  truncation.
- [ ] Write the language manifest (≥100 languages, scripts, romanization,
  tiers). Draft the factor catalogue (see AGENTS.md) for review with the user
  before Checkpoint B builds on it.
  Draft 0.1.0 written 2026-11-14: [factor catalogue](../../docs/factor-catalogue.md)
  with 116 languages, 22 domains / 70 scenarios, 25 intents, 17 constraints and
  6 contrastive groups (0.2.0 after removing non-caller speech from scope); all references and constraints validated
  ([labnotes](../20261114-1023-factor-catalogue-draft.md)). Awaiting user review
  of the points listed in the catalogue doc.
- [ ] Benchmark candidate base models (fp32 and int8 ONNX, 96/128/192 tokens) on
  the reference CPU before training. Drop any that cannot meet 20 ms on the
  model path, or that have incompatible licences.
- [ ] Baselines to beat: "always barge-in", a per-language backchannel word
  list, and Gemini zero-shot as a reference accuracy ceiling.

## Checkpoint B — Synthetic data generator

Follows the deterministic generation plan in [AGENTS.md](../../AGENTS.md):
project code picks every sample's conditions; Gemini only writes the text.

- [ ] Python project with `uv`, `google-genai`, `pyarrow` and tests.
- [ ] Schema module owning the Arrow schemas and factor-value validation for
  every stage in `docs/data-format.md`. Tests first: valid writes round-trip;
  rows with unknown factor values or wrong types are rejected.
- [ ] Transcript cleaner as specified in
  [transcript format](../../docs/transcript-format.md), with golden test cases
  for every rule and script family (Latin, Devanagari, Kannada, Tamil, Arabic,
  Cyrillic, CJK, ZWJ/ZWNJ, apostrophes, hyphens). Tests first. Share the golden
  cases with Vxpipe for runtime parity.
- [ ] Deterministic example builder (conversations → examples: turn shapes,
  prefixes, original and cleaned transcripts, speaker labels, per-prefix
  labels, partitioning). Tests first: same
  input gives identical output.
- [ ] Factor catalogue and constraint rules as versioned data, with a `docs/`
  decision listing every factor, its values and the intent → label mapping.
- [ ] Seeded plan builder (stratified quotas, all-pairs coverage, hard-case and
  contrastive-pair quotas) producing criteria (factor combinations with
  required counts) and stable `combination_id`s. Tests first: same inputs give
  the same plan; constraints are respected; coverage targets are met; IDs are
  stable across runs and catalogue additions.
- [ ] Prompt templates rendered only from a criterion, emitting 2- and 3-turn
  windows with the overlap point marked. Tests for rendering.
- [ ] Judge pass checking each record against its criterion; reject
  non-matching records; report rejections per factor value.
- [ ] Coverage report per factor value and factor pair, from both the plan and
  the accepted records.
- [ ] Native-script, romanized and mixed-script/code-switched variants
  (romanized Hindi + English, romanized Hindi + Devanagari, etc.).
- [ ] Contrastive minimal pairs and hard negatives as listed in Requirements.
- [ ] Prefix expansion with per-prefix labels.
- [ ] Exact and near-duplicate removal;
  balance reports per language/script/category/label; resumable sharded
  generation via the Batch API; cost tracking; snapshot manifests.
- [ ] DuckDB working store and Parquet delivery as specified in
  [data format](../../docs/data-format.md): tables, storage module that allows
  only the listed updates, deterministic partitioned export, import that
  rebuilds the database from exports, snapshots, UUID primary keys (v7 for
  created rows, v5 for criteria and examples). Tests first: v5 IDs are stable
  for the same inputs and match their documented derivation; disallowed
  updates (generated text, factor values, splits) are refused; exporting twice
  gives byte-identical files; import then export reproduces the same files.
- [ ] Fill-to-count generation as specified in
  [data format](../../docs/data-format.md#generation-fill-each-criterion-to-its-required-count):
  per-criterion `need = required − accepted − in flight`, skip when met;
  attempt budget; resume of submitted batch jobs and unprocessed responses;
  `--dry-run`, `--limit` and cost budget; retirement creates a replacement;
  prefix-stable plan growth that keeps IDs and splits; export at the end of
  every run. Tests first: a second run with nothing short makes no API calls
  and leaves the export unchanged; a partial run followed by another fills
  exactly the remaining counts; raising a count generates only the difference;
  an interrupted run resumes without duplicate requests; generated text is
  never changed by any run; a retired conversation is replaced and its text
  kept.
- [ ] Pilot on **English and Hindi**, user review, fix prompts. Then Kannada
  and Tamil with user review, then scale to the full manifest.

## Checkpoint C — Evaluation sets

- [ ] Deterministic split assignment in the plan builder as specified in
  [data format: splits](../../docs/data-format.md#splits): split groups via
  union-find, seeded ordering, iterative stratification to 80/10/10, explicit
  `heldout` factor values. Tests first: same inputs give the same assignment;
  no group spans splits; every factor value, label and language is within
  tolerance; validation/test values all appear in train.
- [ ] Post-generation split audit and cross-split near-duplicate removal (from
  training only), with results in `splits.parquet` and the manifest.
- [ ] Human-verified evaluation sets: English and Hindi first, then Kannada and
  Tamil, each covering native, romanized and mixed scripts and both turn shapes,
  with ambiguous items labelled as such.
- [ ] Metrics defined up front: false interruptions on backchannels, missed
  interruptions, words needed to decide on prefixes, per-language/tier/script
  breakdowns, calibration (ECE). Set pass thresholds **before** tuning.

## Checkpoint D — Training and distillation

- [ ] Fine-tune candidate encoders; compare against baselines on held-out data.
- [ ] Compare 2-turn and 3-turn inputs and token budgets; confirm the 2-turn
  shape is not materially worse, or record the gap for callers.
- [ ] If the most accurate model misses 20 ms: distill to a smaller student
  (fewer layers, vocabulary/embeddings pruned to the manifest) and quantize;
  record accuracy and latency for each step.
- [ ] Calibrate (temperature scaling) on validation data with realistic class
  priors; propose default and per-tier thresholds.

## Checkpoint E — Export, latency and hand-off

- [ ] Export the selected model (e.g. ONNX int8) with its tokenizer, plus a test
  that the exported runtime matches the training framework's outputs.
- [ ] Latency gates on the reference CPU: model path p99 ≤ 20 ms; whole path,
  including the chosen Vxpipe boundary, p99 ≤ 50 ms.
- [ ] Model card: languages and tiers, metrics, known failure modes, data
  provenance, licence. Publish code, model and dataset.
- [ ] Hand-off notes for Vxpipe Checkpoint C: call contract, threshold policy,
  re-evaluation on interim transcripts, failure/timeout fallback.

## Acceptance gates

- [ ] Per-language results for every manifest language (≥100); English, Hindi,
  Kannada and Tamil human-verified results meet the Checkpoint C thresholds.
- [ ] Beats every Checkpoint A baseline on false-interruption and
  missed-interruption rates on the human-verified sets.
- [ ] Both latency gates met on the reference CPU with recorded evidence.
- [ ] Vxpipe acceptance-table scenarios reproduced as model test cases and
  passing.
- [ ] Reproducibility: data manifests, generation code and training config can
  regenerate the shipped model.

## Design review — separate from implementation

Initial specification written 2026-11-14 from the user's brief and revised the
same day with the user's decisions (2–3 turns, Tier A languages, scripts,
latency split, licence, model ID). Not yet reviewed against a decided reference
CPU or final language list.
