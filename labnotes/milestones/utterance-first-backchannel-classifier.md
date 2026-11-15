# Utterance-first backchannel classifier

Status: started 2026-11-15 after the user's decision recorded in
[the utterance-first decision](../../docs/utterance-first-decision.md).
Checkpoint A is done except the user review of the samples
([labnotes](../20261115-0100-utterance-first-generation.md)).

## Runnable outcome

A CPU-only model that takes the caller's utterance spoken **over** the voice
agent (possibly a partial ASR transcript, cleaned) and returns a calibrated
probability that it is a real interruption rather than a backchannel. Vxpipe
calls it only for overlapping speech; answers after the agent finished are
handled by the runtime. Model path p99 ≤ 20 ms and whole path p99 ≤ 50 ms on
the reference CPU, as in [AGENTS.md](../../AGENTS.md).

## Prerequisites and sources

- [Utterance-first decision](../../docs/utterance-first-decision.md), including
  the rejected alternatives.
- The [first milestone](multilingual-text-barge-in-classifier.md): its language,
  latency, licence and evaluation requirements still apply; its 2-3 turn input
  contract is superseded.
- Language manifest and intents under `generation/factors/`; the transcript
  cleaner ([transcript format](../../docs/transcript-format.md)).
- Vxpipe milestone
  `labnotes/milestones/provider-turn-detection-and-contextual-barge-in.md`
  (Checkpoint C) for the runtime scenarios that reach the model.

## Contracts

- **Input**: one cleaned utterance (lowercase, no punctuation, any script mix),
  re-evaluated on each interim revision. Optional language hint, never
  required. A preceding agent line is not part of the contract unless
  Checkpoint D keeps it.
- **Output**: `p_barge_in` in [0, 1], calibrated on realistic priors.
- **Labels**: exactly two, `backchannel` and `barge_in` (user decision,
  2026-11-15), as defined in AGENTS.md minus answers to a finished question
  (the runtime's job). No ambiguous class: the plan, the prompt schema and
  validation accept only these two.
- **Data**: one utterance per row with its factor columns, original and cleaned
  text, the generator's label and `barge_in_from_word`. Generated through the
  same fill-to-count rules as the conversation pipeline
  ([data format](../../docs/data-format.md)).

## Checkpoint A — Utterance generation

- [x] Record the decision in `docs/` and this milestone.
- [x] Utterance factor file `generation/factors/utterances.yaml`: the factors
  kept, the intents excluded and why, reusing the language manifest, intents
  and constraints.
- [x] Schema 4: `utterance_targets` and `utterances` tables, export and import,
  migration from schema 3. Tests first.
- [x] Seeded utterance plan builder (`plan-utterances`): label shares, intents
  round-robin, a stable `combination_id` per criterion, a deterministic split.
  Tests first.
- [x] Generator prompt `generator_utterances_v1` and per-request rendering,
  with rendering tests.
- [x] Response validation: schema, length bucket, `barge_in_from_word` against
  development, no ellipses or tags, label equal to the intended label,
  deduplication on the cleaned text. Tests first.
- [x] Fill-to-count runner without a judge (`generate-utterances`): need =
  required − accepted, attempt budget, resume of stored responses, dry run,
  limit, budget, export at the end. Tests: a second run makes no calls and
  leaves the export unchanged.
- [x] Five English samples generated for user review.
- [ ] User review of the samples; fix the prompt.

## Checkpoint B — Pilot data

- [ ] English and Hindi plan with realistic counts (start at a few thousand
  utterances per language), all script forms and code-switching sets.
- [ ] Prefix expansion for utterances (interim prefixes with per-prefix
  labels, from `barge_in_from_word`) into `examples`, or an utterance examples
  table; decide which.
- [ ] Deduplication and split audit on the cleaned text: the same text never
  appears in two splits; report how many cleaned texts appear with both labels
  (the ceiling for any utterance-only model).
- [ ] Decide with evidence whether a judge pass (language and script only) is
  needed, from the user's review and per-factor rejection rates.
- [ ] Coverage report per factor value and pair.
- [ ] Batch API submission for large runs.

## Checkpoint C — Context slice

- [ ] A small context track (about 10-15% of samples): the agent's truncated
  line plus the caller's utterance, for readouts (`tracking_repeat` vs
  correction), "no" after a negative statement, and echoing versus choosing a
  list item. Reuse the conversation pipeline's cut checks where possible.

## Checkpoint D — Baselines, training and the ablation

- [ ] Benchmark candidate encoders on the reference CPU at 8, 16, 32 and 64
  tokens (fp32 and int8 ONNX), with memory; consider vocabulary pruning to the
  36 languages.
- [ ] Baselines: character n-gram logistic regression on the utterance, and
  Gemini zero-shot as a ceiling.
- [ ] Ablation on the same data: encoder on the utterance only, encoder on
  utterance plus the agent's last line (context slice included), n-gram
  baseline. Report false and missed interruptions overall, on the context
  slice, per language and script form, and on prefixes. Keep context only if it
  gains about 2 points or more within the latency budget.
- [ ] Calibrate on validation data with realistic priors.

## Checkpoint E — Evaluation, export and hand-off

- [ ] Human-verified English and Hindi evaluation sets (native, romanized,
  mixed), then Kannada and Tamil, each item labelled backchannel or barge-in.
- [ ] ONNX int8 export with tokenizer, parity test, whole-path latency through
  the Vxpipe boundary.
- [ ] Model card and hand-off notes for Vxpipe: call contract for overlapping
  speech only, re-evaluation on interim transcripts, thresholds.

## Acceptance gates

- [ ] Beats the n-gram baseline on false and missed interruptions on the
  human-verified sets.
- [ ] Latency gates met on the reference CPU with recorded evidence.
- [ ] Vxpipe overlapping-speech scenarios reproduced as model test cases and
  passing.
- [ ] Per-language results for every manifest language.
