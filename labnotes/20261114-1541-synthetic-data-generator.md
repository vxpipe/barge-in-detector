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
