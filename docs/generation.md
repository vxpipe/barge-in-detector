# Synthetic data generator

Status: implemented 2026-11-14 (Checkpoint B), not yet run against the live
API. This records how the generator talks to Gemini; the storage and
fill-to-count rules are in [data format](data-format.md) and the factors in
[the factor catalogue](factor-catalogue.md).

## Decision

- **One request per criterion, up to 10 samples.** `batch_size: 10` in
  `generation/config.yaml`. A criterion needing more than 10 gets several
  requests; one needing fewer gets one request for exactly what it needs. A
  contrastive group is one request of up to 10 samples in total (5 sets of 2
  members), so every member of a set is written together and shares its text.
- **Every call has two parts:**
  1. **System prompt (cached)**: the common rules: the task, the three lines of
     a sample, label definitions with every intent, factor value meanings,
     writing and safety rules, output format. It is rendered from the
     catalogue only, so it is identical for every request of a template
     version. It goes into an explicit Gemini context cache per (model, prompt
     hash), created once per run and deleted at the end. If a cache cannot be
     created (for example because the prompt is below the model's minimum
     cache size), the prompt is sent as `system_instruction` on every call and
     Gemini's implicit prefix caching applies.
  2. **Request**: the criterion: language, variety, script form,
     code-switching set, domain and scenario (with domain notes), speaker
     pattern, agent activity, cut position, intent with its intended label and
     example hints, development, repetition, length, register, mood, and the
     sample count. For a contrastive group: the shared factors once, then each
     member's own factors, and which element the set must share.
- **Structured output**: both calls use `response_json_schema`
  (`GENERATOR_SCHEMA`, `JUDGE_SCHEMA` in `generation/prompts.py`). The
  response is validated in code and rejected, never repaired.
- **The generator classifies what it wrote**: label, `barge_in_from_word` and
  a one-sentence rationale per sample. It is told to report the label the text
  really has, even when that differs from the intended label.
- **One blind judge call per generator response**: the judge sees the
  transcript (context turn, the agent's spoken part with a "cut off" or "had
  just finished" marker, the candidate) and what the item was meant to show
  (language, variety, script form, code-switching, agent state). It does not
  see the intent, the intended label or the generator's answer. It returns a
  label, a barge-in word and four checks: language, script, cut, natural.

| Template | File | Used for |
| --- | --- | --- |
| `generator_system_v1` | `generation/prompts/generator_system_v1.md` | generator system prompt |
| `judge_system_v1` | `generation/prompts/judge_system_v1.md` | judge system prompt |

A changed prompt is a new file with a new version; old files stay, because
every request records the template it used (`requests.prompt_template`,
`prompt_hash`).

## Acceptance of a sample

In order; the first failure decides the rejection stage:

1. **schema**: the response parses and every sample passes the code checks:
   all fields present; no text empty after cleaning; `agent_spoken_text` is a
   prefix of `agent_full_text` at a word boundary; equal to it exactly when the
   cut position is `just_finished`, otherwise ending mid-sentence; speakers
   match the speaker pattern (the agent never speaks the context or candidate
   line); the candidate's word count is in the length bucket (skipped for
   unspaced native scripts); `barge_in_from_word` is null for backchannel and
   ambiguous, 0 for a stable barge-in, and greater than 0 for a developing one.
   Extra samples are rejected.
2. **judge**: all four judge checks pass.
3. **label_mismatch**: the generator's label, the judge's label and the
   intended label agree; for a barge-in the two `barge_in_from_word` values
   differ by at most `barge_in_word_tolerance` (1). An intended `ambiguous`
   sample needs only the generator's `ambiguous`; the judge's label is kept
   for evaluation.
4. **dedupe**: the cleaned three-line window is not identical to an accepted
   conversation or to another sample in the batch.
5. **contrastive**: in a contrastive set, if any member is rejected, the whole
   set is, so members stay in lockstep.

Rejected samples are stored in `rejections` with the stage, the reason and
both labels; their text stays in `requests.response`. Generator/judge
disagreements are a ready pool of ambiguous candidates for evaluation review.

## Safety and realism rules in the system prompt

- Everything is fictional; no real brands, people or addresses.
- Every number (phone, account, card, order, PNR, amount) is obviously fake,
  such as sequential or repeated digits.
- Collections calls stay polite and non-threatening; healthcare calls stay
  administrative, with no diagnosis or medical advice (domain `prompt_notes`).
- Clean text only: no simulated recognition errors, tags or stage directions.

## Running it

```shell
uv run python -m barge_in_detector check-catalogue
uv run python -m barge_in_detector plan generation/plans/en-hi-pilot.yaml        # add --dry-run to preview
uv run python -m barge_in_detector show-prompt --system                          # the cached prompts
uv run python -m barge_in_detector show-prompt --index 0                         # one request
uv run python -m barge_in_detector generate --dry-run                            # shortfall and token estimate
uv run python -m barge_in_detector generate --languages en --limit 5             # a small first run
uv run python -m barge_in_detector status
uv run python -m barge_in_detector build-examples
```

`generate` reads `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) from the environment,
exports to `data/export/` at the end of every run, and prints accepted and
rejected counts with reasons. `--budget-usd` needs the prices in
`generation/config.yaml`, which are left empty until filled from Google's
current price list. Thinking level and temperature default to the model's
defaults and can be set there.

Concurrency: `concurrency` worker threads make API calls, each with its own
DuckDB cursor in the same process. Every request row is written before its
call, so an interrupted run loses nothing: the next run marks requests that
never got a response as `api_error` and processes stored responses (judging
them first if needed) before making new requests.

## Rejected alternatives

- **All rules in every request with no cache**: about 2,800 tokens of fixed
  instructions per request. Caching them makes the per-request cost the
  criterion and the output.
- **One sample per request**: ten times the requests and fixed-prompt tokens,
  and no within-batch variety instruction.
- **Mixing criteria in one request**: samples would drift between
  specifications and a rejection could not be traced to one criterion's
  prompt.
- **Letting the judge see the intended label**: it would confirm rather than
  classify; the agreement check would mean little.
- **Repairing near-miss responses** (trimming the agent text, recounting
  words): silently changes what Gemini produced and hides prompt problems.
- **Batch API first**: interactive calls are simpler to resume and debug for
  the pilot. The Batch API is still preferred for the large multilingual run
  and remains a Checkpoint B task (`requests.batch_job_id` is reserved for it).

## Verification

- `uv run pytest`: 99 tests, including runner tests against a fake client
  that answers from the rendered prompts: filling to count, no calls on a
  second run with an unchanged export, a limited run followed by one that
  fills exactly the rest, raising a count, resuming stored responses without
  new generator calls, replacing a retired conversation, label mismatches
  until the attempt budget runs out, API errors, contrastive lockstep and dry
  runs.
- Live smoke test (`uv run pytest -m integration`): not yet run.
