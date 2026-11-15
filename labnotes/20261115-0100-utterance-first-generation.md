# Utterance-first generation

Milestone: [utterance-first backchannel classifier](milestones/utterance-first-backchannel-classifier.md),
Checkpoint A. Decision: [docs/utterance-first-decision.md](../docs/utterance-first-decision.md).

## Why

- Two small English runs on the conversation pipeline (commit 4a2d85e):
  6 requests, 8 accepted, 30 rejected, $0.031. Rejections: judge intent check
  12, code-switching missing 6, cut position 8 (5 judge, 3 schema), contrastive
  sets dropped together 6, label disagreement 3. In one rejected sample the
  judge's intent check was wrong: the user confirmed the listener was
  completing the speaker's sentence.
- Fable reviewed the approach twice (2026-11-15). First pass: keep context,
  because "yes" after a finished question must be an answer. The user then
  clarified that Vxpipe already separates speech over the agent from answers
  after it finishes; the model only sees overlapping speech. Second pass:
  switch to an utterance classifier, keep at most one optional agent line if an
  ablation shows it helps. Estimates from that review (not measured): context
  flips 3-8% of overlapping utterances (readouts, "no" after a negative
  statement, list echoes); multilingual-e5-small int8 about 2-4 ms at 8-16
  tokens; SetFit gives no inference gain.

## What was built

- `generation/factors/utterances.yaml` (track 0.2.0): ten factors (the six
  language factors, intent, development, repetition, length); excluded intents
  with reasons; two track rules (within-sentence switching needs 2+ words,
  between-sentence switching needs 4+).
- `generation/utterances.py`: track loading and a position-by-position plan
  builder (label by deficit against shares, intents round-robin per label,
  variant by deficit, 8 seeded candidates from the valid specs, least-covered
  factor values win). Prefix-stable by construction. Split from the ID.
- Schema 4: `utterance_targets`, `utterances`; schema 3 databases migrate on
  open. `runs.conversations_accepted` now counts utterances too.
- Prompt `generator_utterances_v1`, `UTTERANCE_SCHEMA`, `parse_utterances`.
- `generation/utterance_runner.py`: fill-to-count without a judge, resume of
  stored responses, attempt budget, dry run, limit, budget, export.
- CLI: `plan-utterances` (`--retire-missing`), `generate-utterances`,
  `show-prompt --utterances`, utterances in `status`.
- Tests: `tests/test_utterances.py` (27), written before the code; red first
  (import error), then green.

## Barrier and fix

The first sample plan contained a one-word utterance with Malayalam mixed in
between sentences: impossible. In the conversation track the switch could sit
in another turn. Added the two track rules; random draws then sometimes found
no valid candidate and skipped a slot, so candidates are now sampled from the
list of valid specs. The five unused criteria of the first plan were retired
with `plan-utterances --retire-missing` (nothing had been generated for them).

## Sample run (2026-11-15)

```shell
uv run python -m barge_in_detector plan-utterances generation/plans/en-utterance-sample.yaml --retire-missing
uv run python -m barge_in_detector generate-utterances --languages en --budget-usd 0.10 --name en-utterance-sample
```

Run `01a1dcc0-04d7-7182-9dd4-1399c31eb00f`, `gemini-3.5-flash-lite`: 5
requests, 5 accepted, 0 rejected, $0.0018. Input 11,726 tokens of which 10,610
from the explicit cache; output 453. A second dry run plans 0 requests.

| Criterion | Text | Label |
| --- | --- | --- |
| en-US continuer, w1 | Mhm | backchannel |
| en-IN + Kannada between sentences, floor request, filler then barge-in, repeated | Umm, wait wait madri, hold on. | barge_in from word 1 |
| en-MY + Mandarin romanized within sentence, acknowledgment | Ok lah | backchannel |
| en-US correction, backchannel then barge-in, repeated, 9+ words | yeah yeah, that is correct except for the last digit which should be a seven | barge_in from word 6 |
| en-IN + Tamil between sentences, agreement, repeated | Yes absolutely, sari romba sari. | backchannel |

Problems seen (no judge to catch them):

- "lah" is a Malay/Singlish particle, not Mandarin: the requested mix is missing.
- Both "between sentences" samples mix within one sentence.
- The correction starts in lowercase, against the sentence-case rule.
- The correction's turning point is "except" (word 5), not 6.
- The correction refers to digits the speaker read: it carries its label alone,
  but it is the kind of sample the context slice is for.

## Two labels only

User decision, 2026-11-15: the label is either backchannel or barge-in.
Track 0.3.0 lists `labels: [backchannel, barge_in]`; loading fails if a kept
intent maps elsewhere, the plan builder refuses other `label_shares`, and
validation rejects any other label as `unknown_label` (before, `ambiguous`
was accepted by the schema check and only rejected as a label mismatch).

## Checks

`ruff format --check`, `ruff check`: clean. `pytest`: 185 passed, 1
deselected (integration).

## Next

User review of the samples; then decide whether a language/script judge (or a
code check of `barge_in_from_word` against the text) is needed before the
English and Hindi pilot.
