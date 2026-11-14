# Synthetic data generator

Milestone: [multilingual text barge-in classifier](milestones/multilingual-text-barge-in-classifier.md),
Checkpoint B (and the split-assignment task of Checkpoint C).

## Asked for

The user accepted all of Fable's catalogue recommendations and asked to:

- apply the catalogue changes now;
- write the Python generator scripts and prompts, following AGENTS.md;
- generate in batches of 10;
- split the prompt in two: common rules and instructions, cached; and a
  per-request prompt naming the category, intent and other factors.

## Catalogue 0.7.0

Applied every recommendation from Fable's review:

- Intents (24): merged `understanding` into `acknowledgment`; merged
  `surprise_token`, `empathy_token` and `laughter` into `reaction_token`;
  merged `stop_request` and `hold_request` into `floor_request`; merged
  `urgent_interjection` into `complaint`; cut `mixed_signal`. Added
  `go_ahead` (backchannel) and `turn_continuation`, `skip_request`,
  `hello_check` and `language_switch_request` (barge-in). One-word lengths for
  `disagreement`, `clarification_question` and `request_human`; widened
  `repeat_request` (slower, louder) and `end_call` (call me later).
- `early_answer` collision: it must carry content (never a bare yes/ok), and
  only near the end for yes/no questions and read-backs. A new rule stops
  affirmative backchannels near the end of those, where they read as early
  answers.
- Agent activity `closing`. Domains: the merges and cuts listed in the
  catalogue doc, plus front desk, verification/onboarding, agri/rural, UPI
  failure and pre-approved loan offer (22 domains, 77 scenarios; Fable
  estimated 69, but the additions were kept whole). `prompt_notes` for
  healthcare (administrative only) and collections (non-threatening).
- Contrastive groups (9): `ok_backchannel_vs_early_answer` became
  `ok_backchannel_vs_choice` (same agent list, "ok" vs. picking an option),
  because "ok" as an early answer now breaks the content rule. Added
  `go_on_vs_hold_on`, `ok_vs_ok_bye_at_closing` and
  `reaction_stays_vs_complaint`.
- Validator (scratchpad script, then the `check-catalogue` command): no errors.
  45,160 of 216,000 intent-side combinations valid (8,630 backchannel, 36,050
  barge-in, 480 ambiguous).

## Generator code

Python 3.14 project (`uv`; `google-genai` 3.3, `duckdb` 2.0, `pyyaml`). Layout
and commands are in AGENTS.md; decisions in `docs/generation.md`.

- `catalogue.py`: loads the YAML, validates specs (membership, intent length
  and repetition, every rule including language attributes), language options,
  derived values, reference checks and satisfiability.
- `ids.py`: documented UUIDv5 derivations and UUIDv7.
- `cleaner.py`: cleaner v1 with 24 text and 3 window golden cases in YAML
  (not JSON, which would go to LFS).
- `generation/plan.py`: prefix-stable streams (stratified by label share with
  greedy pairwise coverage; contrastive instances), split groups, split
  assignment, split audit.
- `generation/prompts.py` + `generation/prompts/*_v1.md`: cached system
  prompts rendered from the catalogue; per-request prompts; JSON schemas.
- `generation/responses.py`: validation of generator and judge responses.
- `generation/llm.py`: Gemini client with an explicit context cache per
  (model, system prompt hash), fallback to `system_instruction`, retries,
  curated raw response (no headers), key redaction.
- `generation/runner.py`: fill-to-count runner with resume, rounds, limit,
  budget, concurrency and acceptance stages.
- `storage/`: schema and the storage module (allowed updates only;
  deterministic export with byte-identical unchanged files; import).
- `examples.py`: builder v1. `cli.py`: the commands, run with
  `uv run python -m barge_in_detector <command>`.

## Problems met and fixes

- **Received-but-unprocessed responses were requested again** when the judge
  failed: `need` ignored them. `need` now subtracts samples in requests that
  are `submitted` or `received`, as the data format says.
- **API errors used up the attempt budget**, so an outage could exhaust
  criteria. They are now excluded from attempts; a round with no response at
  all ends the run instead, so a dead API cannot loop.
- **The pilot plan failed its own split audit** (e.g. `cut_position=middle`
  7.5% in test). Hash order spread the forced-to-train first occurrences of
  rare values (77 scenarios, variants, code-switch sets) throughout. Fixes, in
  order of effect: rarest-first ordering (still 3 failures); a move-based
  refinement over all features (worse: rare features dominated); refinement
  restricted to audited features (1 failure); a swap pass (passes). The pilot
  plan now passes with tolerance ±2 points at min support 50; a test pins it.
- **Re-applying an unchanged plan** recorded an empty plan revision; it is now
  a no-op. The local working store and export were rebuilt from scratch
  (nothing had been committed).
- `%` is Unicode punctuation (Po) though the transcript format listed it as a
  kept symbol: the cleaner keeps `%` and `‰` explicitly, and the doc now says
  so. Other `Cf` characters (soft hyphen, bidi marks) are deleted.
- A test fake bug (shared candidate text in shared-agent-turn groups) first
  looked like a lockstep failure; the lockstep logic was correct.

## Decisions

- One request per criterion with up to 10 samples; contrastive groups as sets
  in one request (10 samples total), accepted or rejected together.
- Judge: one blind call per generator response; it sees the transcript and the
  expected language/script/agent state, not the intent or labels.
- Accept: generator label = judge label = intended label, barge-in words
  within ±1; intended ambiguous needs only the generator's `ambiguous`.
- Requests store a curated response (text exactly as returned, model version,
  response ID, finish reason, usage), never the SDK's HTTP response object.
- Prices are left null in `generation/config.yaml`: not known for
  `gemini-3.8-flash` here. Tokens are always recorded.
- Licence left out of `pyproject.toml`: still an open decision.

## Evidence

```text
uv run ruff format --check   -> 39 files already formatted
uv run ruff check            -> All checks passed!
uv run pytest                -> 99 passed, 1 deselected (live integration test)
uv run python -m barge_in_detector check-catalogue
  catalogue 0.7.0; 116 languages, 22 domains, 77 scenarios, 24 intents
  {'backchannel': 7, 'barge_in': 15, 'ambiguous': 2}, 20 rules, 9 contrastive groups; no errors
uv run python -m barge_in_detector plan generation/plans/en-hi-pilot.yaml
  272 criteria, 2,720 required samples, split audit passes
uv run python -m barge_in_detector generate --dry-run --languages en,hi
  272 generator requests for 2,720 samples
  en/hi each: 630 backchannel, 680 barge-in, 50 ambiguous
  estimated tokens: input 429k, cached input 971k, output 598k (rough)
```

System prompt sizes: generator about 2,800 tokens (cacheable); judge about 760
tokens, probably under the explicit-cache minimum, so it falls back to
implicit caching.

## Not done yet

- No live Gemini call has been made: the smoke test
  (`uv run pytest -m integration`) and the first pilot run wait for the user.
- Batch API mode, snapshots/manifests, coverage reports, near-duplicate
  filtering across splits.
- The working store and `data/export/` (pilot plan) are local; the export is
  not committed yet.

## Update: no installable tool

The user decided there will be no `bargein` tool or pip package; this stays a
project. The code moved from `src/barge_in_detector/` to a top-level
`barge_in_detector/`, `pyproject.toml` has `[tool.uv] package = false` with no
build backend or console script, and commands run from the repository root
with `uv run python -m barge_in_detector <command>`.

## First live run and fixes (2026-11-14)

The user ran `generate --languages en --limit 5` in their own shell (my
non-interactive shell's `GEMINI_API_KEY` is rejected by Google with
`ACCESS_TOKEN_TYPE_UNSUPPORTED`, so live calls are run by the user for now).

Result: 5 requests, 26 accepted, 24 rejected (14 `judge:script_ok`, 5
`schema:length_bucket_mismatch`, 5 `schema:contrastive_member_rejected`).
Generator tokens: 17,293 input, 17,383 output for 5 requests; no cache hits.

| Finding | Cause | Fix |
| --- | --- | --- |
| Judge failed script on romanized Bhojpuri/Bengali in en-IN | "script form native" was ambiguous for mixed-in languages; the generator wrote Bengali script in some samples | explicit per-language Script line in requests and judge input (`script_expectation`); generator prompt v2; 26 v1 samples with Bengali script retired (kept) |
| "Mm-hmm." / "Uh-huh." rejected as 2 words | words counted after cleaning | `cleaner.words`: original tokens with content, used by validator, judge options and example builder |
| Context cache 404, warned twice | `gemini-3.8-flash` refuses explicit caching | caching off per model after the first refusal, one warning; first request of a run runs alone for implicit caching |
| All 5 requests were contrastive | contrastive priorities 0–1 sorted first | priorities spread through the stratified order; plan revisions may update `priority` |

Judge replaced by the Gemini Decisions API at the user's request
(`gemini-decision-1`, `google-genai` 3.3.0 `client.decisions.create`, checked
in the installed SDK and the AI Studio docs): one request per sample, six
questions (label choice, barge-in word choice, four yes/no checks), argmax
label, checks pass at P(true) >= 0.5, probabilities stored. Prices are now
per model.

A test of mine passed for the wrong reason twice and was tightened: the
warm-up test (first call always sees one active call) and the judge-resume
test (a missing `LlmError` import raised NameError, which the runner recorded
as an API error).

Open, for the user: English specs are mostly `en-IN` with code-switching,
because language-side options are drawn uniformly from all valid
(variant, script, code-switching, set) tuples and en-IN has 54 of 60.
Drawing variant, then script, then code-switching, then set would balance it,
but changes the pilot plan's specs.

Evidence: `uv run ruff format --check`, `uv run ruff check` clean;
`uv run pytest`: 115 passed, 1 deselected.

## Conversations between any people; yes/no judge (2026-11-14)

User decisions:

- The data is not from an agent's perspective: conversations between any two
  or three people in the catalogue's situations, named by role ("Hotel
  Manager", "Guest") or by name ("David", "Irina"; "Tom", "Hardy", "Nick").
- The judge asks yes/no questions only, phrased like "While <speaker> is
  speaking, can the last turn in this conversation be labelled <label>?", with
  the conversation as a JSON array of `{speaker, transcript}`.

Changes:

- Catalogue 0.8.0 (intents 0.6.0, domains 0.3.0, contrastive 0.6.0,
  constraints 0.7.0): wording only, speaker/listener instead of agent/caller;
  `request_human` now means "someone else" (manager, supervisor or a human
  instead of an assistant); contrastive `agent_turn` renamed `speaker_turn`.
  Factor ids are unchanged, so every criterion keeps its `combination_id`.
- Generator prompt v3 and schema with neutral fields (`speaker`,
  `speaker_full_text`, `speaker_spoken_text`, `listener`, `listener_text`;
  `context_turn` without a role). Stored columns keep their names (`agent_*`
  = speaker, `candidate_*` = listener) to avoid a schema migration.
- Judge v2 (`judge_decisions_v2.yaml`): `is_backchannel`, `is_barge_in`, one
  `prefix_<k>` per proper prefix of the last turn, and the four checks, all
  `noul`. Label from the two label answers (ambiguous when both or neither are
  yes); turning point from the first prefix answered yes. The word agreement
  check now applies only to developing barge-ins.
- v2 generator and v1 judge templates were never used live; they stay for
  history.

Evidence: ruff clean; `uv run pytest`: 118 passed, 1 deselected.

## US English, Mandarin–English, locales, factor questions (2026-11-14)

User decisions: plain US English as its own slice next to Indian
code-switched English; Mandarin + English code-switching as in Malaysia; a
`locales` array per row because a transcript can span several locales; every
generation factor confirmed by its own Decisions API yes/no question.

- Plan: `variant_shares` per language (pilot: en-US 0.35, en-IN 0.35, en-MY
  0.10, others 0.02–0.04); variants follow a deterministic share schedule like
  labels; the language side is drawn one factor at a time (variant, script,
  code-switching, set).
- Coverage scoring first made code-switched English win almost every pick
  (en-IN plain 6 of 55): every code-switch set is a rare value, and "none" is
  already common through other variants. Excluding the set, then the whole
  language side, from coverage scoring gave 18 of 55 (33%), as drawn.
- Manifest 0.3.0: `en-MY`, `en-SG` (mix `zh`, `mixed_in_script: Hans`),
  `zh-MY`; `region: IN` for Indian languages without variants, for locales.
- Schema 2: `locales VARCHAR[]` on targets, conversations, examples; a v1
  database is migrated on open (column added and filled from the factors).
- Judge v3: `factor_scenario`, `factor_speaker_activity`,
  `factor_listener_intent`, `factor_development`, `factor_repetition`,
  `factor_register`, `factor_listener_mood`, gating at P >= 0.5 with the four
  checks. Cut position is now checked in code from the spoken share.
- Pilot plan rebuilt and applied with `--retire-missing`: 272 new criteria,
  272 earlier ones retired (none had accepted data). English: en-US 47 plain,
  en-IN 18 plain + 37 code-switched, en-MY 13 Mandarin-mixed + 1 plain.

Evidence: ruff clean; `uv run pytest`: 136 passed, 1 deselected; split audit
passes for the rebuilt pilot.

## Chinese in both scripts, raw-text input, plan rebuild bugs (2026-11-14)

User decisions: Mandarin and Cantonese in both characters and romanization
(one language, two scripts; locales tell them apart); try training without
lowercasing or removing punctuation.

- New spec factor `mixed_in_script` (default, native, romanized), allowed per
  code-switch set by `mixed_in_scripts`; `en-HK` + Cantonese (`yue-HK`) added;
  zh-TW writes Traditional (`variant_scripts`); romanization names pinyin and
  Jyutping. `combination_id` leaves out factors at `default`, so the new
  factor kept every other criterion's ID. Locales carry script subtags.
  Catalogue 0.9.0, manifest 0.4.0, schema 3.
- `examples.transcript_raw`: `s<n>` labels with casing and punctuation kept,
  to compare against `transcript_clean` in Checkpoint D.
- Bugs found while rebuilding the pilot:
  - contrastive group ids were `<group>:<language>:<index>`, so a rebuilt
    plan reused ids that earlier requests pointed at (old attempts would
    count against new groups, and stale splits stuck). Ids now derive from
    the members; legacy matches are limited to criteria that existed when the
    request was made.
  - split assignment kept stale splits of criteria that had never been
    requested. Rule now: plan-assigned fields are fixed once a criterion has
    generation requests; before that a revision may reassign them, and
    plan-retired criteria that return are reactivated.
  - the sampler chose only four of the five language-side factors, so
    `mixed_in_script` was always `native` (17 of 17). Fixed; a test pins that
    both scripts are drawn.
  - I chained the plan command after a failing test run and applied a plan
    with 7 failing tests; the failures were a test-fake bug (cut after
    "here."), the plan had no data, and a re-run confirmed the applied plan
    equals a fresh build.
- Pilot now: 272 active criteria; English en-US 47 plain, en-IN 15 plain + 34
  code-switched, en-MY 6 Mandarin-mixed + 6 plain, en-SG 3 + 5, en-HK 5
  Cantonese-mixed + 1; Chinese mixing 2 native / 7 romanized for Mandarin, 2 /
  3 for Cantonese.

Evidence: ruff clean; `uv run pytest`: 147 passed, 1 deselected.

## Flash-Lite, 5 per criterion, caching fixed (2026-11-14)

- User decisions: generator `gemini-3.5-flash-lite` by default ($0.30 /
  $2.50 per M tokens; switching to `gemini-3.8-flash` is the user's call);
  5 samples per criterion; the manifest reduced to the user's 36 languages.
  Pilot: 272 criteria, 1,360 samples, dry-run estimate $1.91 for one pass.
- Caching: the explicit cache returned 404 because google-genai 3.x defaults
  to API `v1`; `cachedContents` exists only in `v1beta` (REST create worked;
  SDK with `v1beta` worked). GeminiClient now uses `v1beta` for both clients.
  Live check: 3,146 of 3,498 input tokens cached (v3), 3,332 of 3,684 (v4).
- My shell's credentials now work from the project directory; the live smoke
  test passed (the earlier `ACCESS_TOKEN_TYPE_UNSUPPORTED` no longer occurs).
- Live diagnostic (2 samples, en-IN + Hindi, agreement_token, repeated):
  v3 prompts gave lowercase unpunctuated text, an unrepeated "haan exactly"
  and a cut at a possible sentence end; the judge caught them (repetition
  0.10, cut 0.04). Generator v4 (sentence case and punctuation, mid-phrase
  cuts, self-check) and judge v4 (development question for any response):
  cut 0.97, repetition 0.97, development 0.84. v4 copied my concrete cut
  example verbatim, so the example was replaced by a description.
- Estimator constants set from measurements: 350 output tokens per sample,
  3,500 judge input tokens per sample.

Evidence: ruff clean; `uv run pytest`: 150 passed, 1 deselected; live smoke
test passed.

## Hard and soft judge checks (2026-11-14)

Live samples from Flash-Lite often missed variety factors ("repeated" kept in
1 of 5 samples; mood, register and naturalness borderline) while the label,
intent and cut were right. User decision: the training label is only whether
the last turn is a backchannel; intents and other factors are generation
dimensions. Hard checks (reject): the label answers and turning point,
language, script, cut-off, listener intent (`judge_hard_checks`). Soft
(recorded in `judge_notes` and the raw response, never a rejection):
repetition, mood, register, scenario, speaker activity, development,
naturalness. Both samples of the last diagnostic would now be accepted.

Evidence: ruff clean; `uv run pytest`: 152 passed, 1 deselected.
