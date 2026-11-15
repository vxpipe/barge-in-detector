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
- Tests: `tests/test_utterances.py` (27 then), written before the code; red first
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

## Clean start and the all-language sample (unjudged)

User decision: delete all data in the working DB and start clean. The DB was
moved aside (backup kept outside the repo) and recreated empty; the
conversation runs' exports were removed from `data/export/` (still in git at
4a2d85e). `generation/plans/en-utterance-sample.yaml` became
`generation/plans/utterance-sample.yaml`: 36 languages × 5 criteria × 5
utterances (3 backchannel and 2 barge-in criteria per language).

Prompt v1, no judge: 362 requests, $0.27, 870 of 900 accepted. Review found
Urdu (ur-IN) in Devanagari although the Script line said Arabic script,
Arabic + French written as French only, Hindi + Tamil without Tamil (the
Script line said "Tamil words in Devanagari script or romanized"), Cantonese
in Jyutping with tone numbers (the manifest's romanization name was copied
into the prompt), and `barge_in_from_word` off by one ("Sure sure, but" at 3).
These 870 are retired ("generator_utterances_v1 without a judge").

## Parallel requests

`concurrency: 10` (user: up to 10 in parallel). Worker threads only call
Gemini; the main thread writes. Tests: overlap reaches the concurrency, exact
fill, dedupe equal to a sequential run, `--limit` honoured.

## Prompt v2/v3 and the criteria judge

User decision: state the script of every language in the combination, and
bring back the Decisions API as a judge that asks whether each criterion is
fulfilled, nothing else, one question per language where a criterion covers
several. `generation/utterance_criteria.py` renders both from the same
`language_parts`; schema 5 keeps `judge_request_id` and `judge_answers`.
The manifest (0.6.0) gives Urdu `script_name: Urdu (Perso-Arabic)` and
`never_scripts: [Deva]`. The attempt budget now counts only requests of the
current generator template.

Run `utterance-sample-v2-judged` (generator v2, judge v1): 700 generate +
2110 judge requests, $0.74, 286 accepted. Judge v1 had question flaws, seen in
rejections failing a single question:

- script: "written in Thai script, never in Latin letters" failed Thai text
  whose English words were in Latin letters;
- language: "Mh" was not Italian;
- variant: "Parfait." (fr-FR) 0.44, "好" (yue-HK) 0.50;
- stable development: its "yes yes" example made "Ja ja, true" a change.

Judge v2 (script question only about the language's own words, no forbidden
script; sounds count for any language; variant fails only when something is
out of place; new stable wording). Re-asked on 40 single-failure rejections
and accepted samples (read-only, about $0.004): the false script, variant,
development and repetition rejections above passed; the rest stayed rejected
and looked right ("Ya benar, setuju sekali. Very true." has no repetition;
"Hmm, não, não" turns at word 1, not 2).

Run `utterance-sample-v3-judged` (generator v3: majority and between-sentence
wording, repetition meaning; judge v2): 813 generate + 1547 judge requests,
$0.65, 219 accepted. Judge pass rate per factor value (v3, judged samples):

| Factor | Pass rate |
| --- | --- |
| code-switching none / within / between | 14% / 13% / 16% |
| script native / romanized / mixed | 18% / 12% / 1% |
| intent continuer / acknowledgment / correction / agreement / floor request | 8% / 12% / 14% / 15% / 20% |
| length w1 / w2_3 / w4_8 / w9_plus | 7% / 10% / 13% / 27% |

Single-language native text fails mostly on `intent` (122 of 164): for
`continuer` the generator writes agreement words ("Sì.", "Genau.", "Juist",
once "Mysz", Polish for mouse). Five distinct one-word continuers per
language, with deduplication, is close to impossible. Mixed-script text
almost never passes.

State: 505 active utterances (286 v2, 219 v3), 60 of 180 criteria full, 120
with exhausted attempt budgets. Spend since the clean start: $1.66.

## gemini-3.8-flash round and plain judge questions

User decision: switch the generator to `gemini-3.8-flash` for another round
(`generation/config.yaml`). The attempt budget now counts requests per prompt
template and generator model, so the new model could retry exhausted
criteria. Run `utterance-sample-v3-flash` (judge v2): 222 generate + 568
judge requests, $2.05 (stopped at the $2.00 cap), 133 accepted, 560 rejected
(switching 196, dedupe 112, repetition 57, development 42). 638 active
utterances, 94 criteria still short.

User feedback: one judge, with plain questions like "Is the Chinese in this
text in pinyin?", built from each sample's criteria; it does not check the
label; do not overengineer. The two judge files (v1, v2: the same judge,
reworded) are replaced by one, `judge_utterances_v3.yaml`, with one-line
questions; v1 and v2 were never committed and are described above.

## Equal languages, two-language Indian sets

User decisions (2026-11-15): no language is primary; the judge asks "Does
this text contain <language> words?" for every language of the combination,
and the prompt lists the languages without a main one. Indian languages
appear alone or with English only, and Indian English with one Indian
language: the manifest (0.7.0) drops every three-language set and Punjabi +
Urdu. Rebuilding the sample plan keeps 132 criteria and replaces 48 (every
Indian language's criteria move, because the builder picks from each
language's valid combinations); 135 accepted utterances sit on the replaced
criteria, 21 of them three-language samples. Not applied yet.

## Input budget

User decision: 64 tokens, enough for a whole sentence (30-50 subword tokens
for a long Indian-language utterance); measure 32 only if 64 misses the
latency gates, no 128 if 64 meets them. The "8-16 tokens" was Fable's typical
size for a latency estimate, never a limit.

## One CSV, one local request log (schema 6)

User decisions: no runs or plan revisions (git keeps the plan file's
history); no conversation tables; every sample, accepted or rejected, in one
table exported as one CSV in plain git; the request log stays local. The
conversation pipeline code (runner, plan builder, example builder, their
prompts and tests) was removed; it is in git history (3ef2f65). The
partitioned Parquet exports (539 files for about 13,000 rows) are replaced by
`data/export/utterances.csv`. CLI commands are now `plan`, `generate`,
`status`, `show-prompt`, `export`, `import`, each reading `--plan` (default
`generation/plans/utterance-sample.yaml`).

## Clean sample run with gemini-3.8-flash (rebuilt plan)

Committed the code first (3ef2f65), cleared the DB, rebuilt the plan without
three-language sets (180 criteria, 900 utterances). Run (judge v3, plain
questions, concurrency 10, $5 cap): 584 generate + 1,519 judge requests,
$4.42 ($4.36 generator, $0.07 judge), 721 accepted, 1,040 rejected (switching
282, dedupe 227, development 184, variant 105, intent 86, repetition 44,
plus language and script per language), 0 API errors; 59 criteria short with
exhausted attempt budgets. Converted into schema 6 (one-off script, counts and
cost matched), exported as `data/export/utterances.csv` (1,761 rows, 2.4 MB);
import of the CSV reproduces every row.

Cost per accepted utterance: $0.0061; per required utterance (80% filled):
$0.0049. Full space: 40,197 valid criteria across the 36 languages; at 5 each,
201k utterances: about $990 at these rates for about 80% filled (161k), and
roughly $1,200-1,500 to approach 100% (the short criteria are the hard ones).
The Batch API (half price, if it applies to this model) would about halve
that. Estimates, not measured at that scale; short buckets (w1, w2_3) will
deduplicate more as counts grow.

## 15 languages

User decision after reviewing the per-language table (pass rates, combination
counts): drop Dutch, Hungarian, Swedish, Danish, Norwegian, Finnish, Irish,
Portuguese, Thai, Assamese, Arabic, Vietnamese, Korean, Marathi, Gujarati (manifest 0.8.0; Indian English no longer
mixes Assamese, Marathi or Gujarati). Their 213 accepted utterances are retired ("language dropped
from the manifest"); 508 accepted remain. Valid combinations: 32,733 (was
40,197); every combination at 5 each is about 164k utterances, roughly $800
at the sample run's rate for about 80% filled, $1,000-1,250 to approach 100%.

User decision (same day): drop Punjabi and Bengali too (19 languages);
English only en-US, en-IN, en-SG (Singaporean English mixes Mandarin; Hong
Kong English and its Cantonese set are gone); Tamil only ta-IN. Samples of
dropped languages and varieties are retired the same way.

Then Malay, Polish, Russian and Cantonese too (15 languages; Cantonese passed 16%: its interjections read as Mandarin and its romanization was unreliable). The old conversation plan
`generation/plans/en-hi-pilot.yaml` is removed. `generation/plans/utterance-all.yaml`
takes every valid combination (`all_combinations: true`) at 5 each.

Mandarin in Chinese characters only (user decision, 2026-11-15:
`romanized_usage: none`, so no pinyin and no characters-plus-pinyin mix);
then no pinyin anywhere: Singaporean English mixes in Mandarin as
characters only. Urdu: Indian Urdu (ur-IN) only. Spanish: Spain and Mexico (es-ES, es-MX) only. French: France only; German: Germany only; Mandarin: mainland China only (Simplified characters). Indian languages in native script only, alone or mixed with English (no romanized or mixed-script forms; manifest 0.9.0). Indian English mixes Indian words in their native script. `turn_continuation` excluded (track 0.4.0): it needs an earlier turn.

## Checks

`ruff format --check`, `ruff check`: clean. `pytest`: 108 passed, 1
deselected (integration). No test calls Gemini.

## Next

Decide before more spend: a stronger generator (`gemini-3.8-flash`, the
user's call), fewer or merged criteria where few distinct texts exist
(continuer at w1/w2_3), and whether mixed-script stays in the sample plan.
