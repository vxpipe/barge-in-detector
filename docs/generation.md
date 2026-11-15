# Synthetic data generator

Status: implemented 2026-11-14 (Checkpoint B). First live run (5 English
requests) on 2026-11-14; its findings changed the judge, the script rule and
word counting (see "Pilot findings" below). This records how the generator talks to Gemini; the storage and
fill-to-count rules are in [data format](data-format.md) and the factors in
[the factor catalogue](factor-catalogue.md).

## Decision

- **Generator model**: `gemini-3.5-flash-lite` by default (user decision,
  2026-11-14, for cost: $0.30 / $2.50 per million input / output tokens
  against $0.75 / $3.75 for `gemini-3.8-flash`). The user decides whether to
  switch to `gemini-3.8-flash` (`generator_model` in `generation/config.yaml`);
  every request records the model that produced it.

- **Conversations between any people** (user decision, 2026-11-14). The
  detector is deployed where a voice agent is speaking, but the data is not
  written from an agent's point of view: every sample is a conversation
  between people in the domain's situation (a hotel manager and a guest, two
  friends, a pharmacist and a customer). The person with the floor is the
  **speaker**; the person who says the judged last line is the **listener**; a
  third person may say the first line (speaker pattern `new_speaker`). People
  are named by role ("Hotel Manager", "Guest") in about half the samples and
  by personal name ("David", "Irina"; "Tom", "Hardy", "Nick") in the rest.
  Names never reach the model: the cleaner turns them into `s1`, `s2`, `s3`.
  Stored columns keep their earlier names: `agent_*` is the speaker and
  `candidate_*` the listener; the factor `agent_activity` is what the speaker
  is doing.
- **One request per criterion, up to 10 samples.** `batch_size: 10` in
  `generation/config.yaml`. A criterion needing more than 10 gets several
  requests; one needing fewer gets one request for exactly what it needs. A
  contrastive group is one request of up to 10 samples in total (5 sets of 2
  members), so every member of a set is written together and shares its text.
- **Every generator call has two parts:**
  1. **System prompt (cached)**: the common rules: the task, the three lines of
     a sample, label definitions with every intent, factor value meanings,
     writing and safety rules, output format. It is rendered from the
     catalogue only, so it is identical for every request of a template
     version. It goes into an explicit Gemini context cache per (model, prompt
     hash), created once per run and deleted at the end. Caching needs API
     version `v1beta`: google-genai 3.x defaults to `v1`, where creating a
     cache returns 404 (the cause of the 404 in the first run), so the client
     is created with `v1beta`. Verified 2026-11-14 on `gemini-3.5-flash-lite`:
     3,332 of 3,684 input tokens served from the cache. If a model still
     refuses explicit caching, caching is switched off for that model for the
     run and the prompt is sent as `system_instruction`, where implicit prefix
     caching can apply; the first request of a run is sent alone so later ones
     can hit it. The Decisions API (the judge) has no caching.
  2. **Request**: the criterion: language, variety, script form,
     code-switching set, an explicit **Script** line (below), domain and
     scenario (with domain notes), speaker
     pattern, speaker activity, cut position, listener intent with its intended
     label and
     example hints, development, repetition, length, register, mood, and the
     sample count. For a contrastive group: the shared factors once, then each
     member's own factors, and which element the set must share.
- **Structured output**: the generator uses `response_json_schema`
  (`GENERATOR_SCHEMA` in `generation/prompts.py`). The response is validated
  in code and rejected, never repaired.
- **The generator classifies what it wrote**: label, `barge_in_from_word` and
  a one-sentence rationale per sample. It is told to report the label the text
  really has, even when that differs from the intended label.
- **The judge is the Gemini Decisions API** (`gemini-decision-1`, user
  decisions 2026-11-14), asked **yes/no questions only**. One request per
  sample; the input is just the conversation as JSON:

  ```json
  {"conversation": [
    {"speaker": "Guest", "transcript": "Do you have a room with a view?"},
    {"speaker": "Hotel Manager", "transcript": "Sir, we have a suite on the highest floor, and"},
    {"speaker": "Guest", "transcript": "ok"}
  ]}
  ```

  Questions, from `generation/prompts/judge_decisions_v3.yaml`, each naming
  the people and the moment ("While Hotel Manager is speaking" or "Right
  after Hotel Manager finished speaking"):

  | Key | Asks |
  | --- | --- |
  | `is_backchannel` | While Hotel Manager is speaking, can the last turn in this conversation, said by Guest, be labelled a backchannel? (with the definition and the label rules) |
  | `is_barge_in` | …can it be labelled a barge-in? |
  | `prefix_1` … `prefix_<n-1>` | Suppose Guest had so far said only the first k words: could that much already be labelled a barge-in? |
  | `language_ok` | Is the conversation in English (en-IN), with no other language mixed in? |
  | `script_ok` | Is every turn written in the scripts stated here: (the Script line)? |
  | `cut_ok` | Does Hotel Manager's turn visibly break off mid-sentence? / Is it complete? |
  | `natural_ok` | Does this read like a real conversation between these people…? |
  | `factor_scenario` | Is this conversation set in this situation: (domain and scenario descriptions)? |
  | `factor_speaker_activity` | In the second-to-last entry, is Hotel Manager doing this: (activity description)? |
  | `factor_listener_intent` | Does this describe what Guest does in the last turn: (intent description)? |
  | `factor_development` | Does Guest's last turn have a single intent from its first word? / start like a backchannel, or with filler, and then take the floor? |
  | `factor_repetition` | Are the key words repeated? / said once? |
  | `factor_register` | Do the people speak in a formal / casual register? |
  | `factor_listener_mood` | Does Guest come across as (mood)? |

  **Every generation factor is confirmed** (user decision, 2026-11-14): the
  ones code can check exactly are checked in code (length bucket, speaker
  pattern, and cut position from the share of the planned turn already said:
  start at most 4 words, early 8–45%, middle 30–75%, near end at least 60% with
  at most 6 words left); every other factor has its own `factor_*` question,
  and language, variety, code-switching and script are the `language_ok` and
  `script_ok` checks. Questions are evaluated independently, so the intent
  question does not inform the blind label questions.

  Answers are calibrated P(yes) with no generated text and no output tokens.
  The label is `barge_in` when P(is_barge_in) ≥ 0.5 and P(is_backchannel) <
  0.5, `backchannel` for the reverse, and `ambiguous` when both or neither are
  yes. `barge_in_from_word` is the first prefix answered yes (k words → word
  k − 1), or the last word when only the full turn is a barge-in. A check
  passes when P(yes) ≥ 0.5. The full answers (all probabilities) are stored in
  `requests.response` and the verdict's probabilities in
  `conversations.judge_notes`. A blocked input fails every check. The judge
  never sees the intent, the intended label or the generator's answer. One
  sample per request follows the API guide ("send only the fields your
  questions need"); items judged before an interruption are reused on resume.

| Template | File | Used for |
| --- | --- | --- |
| `generator_system_v4` | `generation/prompts/generator_system_v4.md` | generator system prompt (current): v3 plus sentence case and punctuation, mid-phrase cuts, a final self-check of every factor |
| `judge_decisions_v4` | `generation/prompts/judge_decisions_v4.yaml` | judge questions (current): v3 with the development question worded for any response |
| `generator_system_v3` | `generation/prompts/generator_system_v3.md` | conversations between any people; never used for stored data |
| `judge_decisions_v3` | `generation/prompts/judge_decisions_v3.yaml` | judge questions (current): v2 plus one yes/no question per generation factor |
| `judge_decisions_v2` | `generation/prompts/judge_decisions_v2.yaml` | yes/no, named people, JSON input; never used live |
| `generator_system_v2` | `generation/prompts/generator_system_v2.md` | agent/caller wording with the Script line; never used live |
| `judge_decisions_v1` | `generation/prompts/judge_decisions_v1.yaml` | choice questions; never used live |
| `generator_system_v1` | `generation/prompts/generator_system_v1.md` | first live run |
| `judge_system_v1` | `generation/prompts/judge_system_v1.md` | first live run (generative judge on `gemini-3.8-flash`) |

### The Script line

Every request and judge input states which script each language's words use,
computed from the spec (`prompts.script_expectation`):

- the conversation language follows `script_form` (native, romanized, or
  mixed within an utterance);
- English and other Latin-script languages mixed in: Latin letters;
- a code-switch set may allow a choice of script for the mixed-in language
  (`mixed_in_scripts` in the manifest), chosen per spec by the
  `mixed_in_script` factor: Mandarin in Malaysian or Singaporean English, and
  Cantonese in Hong Kong English, as characters ("…in Simplified Chinese
  characters, not pinyin") or romanized ("…in pinyin (Latin letters), never
  in Chinese characters");
- other mixed-in languages: romanized when the conversation is written in
  Latin letters (a Latin-script language such as English, or the romanized
  form), and in the conversation language's script when it is native, e.g.
  "Indian English mixing in Bengali: Bengali words romanized, never in
  Bengali script".

### Counting words

Lengths and `barge_in_from_word` count whitespace-separated words that keep
some text after cleaning (`cleaner.words`): "Mm-hmm," is one word, a lone dash
is none. The generator, the validator, the judge's word options and the
example builder all count this way.

A changed prompt is a new file with a new version; old files stay, because
every request records the template it used (`requests.prompt_template`,
`prompt_hash`).

## Acceptance of a sample

In order; the first failure decides the rejection stage:

1. **schema**: the response parses and every sample passes the code checks:
   all fields present; no text empty after cleaning; `speaker_spoken_text` is
   a prefix of `speaker_full_text` at a word boundary; equal to it exactly when
   the cut position is `just_finished`, otherwise ending mid-sentence; people
   match the speaker pattern (the speaker never says the context or last
   line); the listener's word count is in the length bucket (skipped for
   unspaced native scripts); `barge_in_from_word` is null for backchannel and
   ambiguous, 0 for a stable barge-in, and greater than 0 for a developing one.
   Extra samples are rejected.
2. **judge**: the hard checks are answered yes (P ≥ 0.5): language, script,
   cut-off and listener intent (`judge_hard_checks` in
   `generation/config.yaml`). The rejection reason lists the failed ones. The
   label is the training target, backchannel or not (user decision,
   2026-11-14): intents and the other factors are dimensions for generating
   varied samples. So the variety factors (repetition, mood, register,
   scenario, speaker activity, development) and naturalness are **soft**:
   asked for in the prompt and answered by the judge, with any "no" recorded
   in `judge_notes` (`soft: P(...)`) and the raw response, but never a reason
   to reject.
3. **label_mismatch**: the generator's label, the judge's label and the
   intended label agree; for a developing barge-in (generator's
   `barge_in_from_word` > 0) the judge's turning point is within
   `barge_in_word_tolerance` (1) words. For a stable barge-in the judge's
   prefix answers are kept but not compared. An intended `ambiguous`
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
- Collections stay polite and non-threatening; healthcare stays
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

## Utterance track

Added 2026-11-15 after the [utterance-first decision](utterance-first-decision.md).
Each request asks for N single caller utterances, said over a speaker who is
still talking, for one criterion of `utterance_targets`:

- **System prompt** `generator_utterances_v1` (`generation/prompts/`), cached
  like the conversation prompt: the task, the track's intents with labels,
  development, repetition, length, script form and code-switching meanings,
  writing rules, output format.
- **Request**: the language lines (including the Script line), the intent
  with its intended label and example hints, development, repetition and
  length. No domain, scenario, cut or speaker; the generator imagines a
  different situation for each utterance.
- **Schema** `UTTERANCE_SCHEMA`: `text`, `label` (backchannel or barge_in),
  `barge_in_from_word`, `rationale`.
- **Labels**: `backchannel` or `barge_in` only (`labels` in
  `generation/factors/utterances.yaml`); anything else is rejected as
  `unknown_label`.
- **Acceptance**: schema checks (fields, not empty after cleaning, no
  ellipsis, no bracket tags, length bucket, `barge_in_from_word` against
  development), then the generator's label must equal the intended label,
  then the cleaned text must be new. **No judge** (user decision,
  2026-11-15); language, script and code-switching are not checked yet.

```shell
uv run python -m barge_in_detector plan-utterances generation/plans/utterance-sample.yaml
uv run python -m barge_in_detector show-prompt --utterances --system
uv run python -m barge_in_detector show-prompt --utterances --index 0 --count 10
uv run python -m barge_in_detector generate-utterances --dry-run
uv run python -m barge_in_detector generate-utterances --languages en --budget-usd 0.10
```

Up to `concurrency` requests (10 in `generation/config.yaml`, or
`--concurrency`) are in flight at once. Worker threads only call Gemini; the
main thread inserts each request before its call starts (so it counts as in
flight), records responses and validates them one at a time, which keeps
deduplication exact. After rejections the runner plans again once the
requests in flight are processed. The Batch API is a milestone task.

## Pilot findings (first live run, 2026-11-14)

Five English requests (contrastive groups only, by the generation order of the
time): 26 accepted, 24 rejected.

- **14 `judge:script_ok`**: the request said "script form native" for Indian
  English mixing in Bengali or Bhojpuri. The generator sometimes wrote the
  mixed-in words in their own script, sometimes romanized; the judge expected
  native script. The manifest's rule is that mixed-in words in English-led
  speech are romanized, so the 26 accepted samples (Bengali script) were the
  wrong ones. Fix: the explicit Script line; the 26 were retired (kept, with
  the reason) so their criteria are refilled.
- **5 `length_bucket_mismatch`**: "Mm-hmm." and "Uh-huh." counted as two
  words because words were counted after cleaning. Fix: count original words.
- **Explicit context cache: 404** for `gemini-3.8-flash`, and no implicit
  cache hits (four cold parallel requests). Fix: one warning, caching off for
  the model, first request alone.
- **All five requests were contrastive groups** because contrastive
  instances had priorities 0–1. Fix: instances are spread evenly through the
  stratified order; plan revisions may update `priority`.
- The judge moved to the Decisions API at the user's request, then to
  yes/no questions only, and the data stopped assuming a voice agent
  (generator v3, judge v2): conversations between any people, named by role
  or by name.

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
- **A generative judge** (`judge_system_v1` on `gemini-3.8-flash`): replaced by
  the Decisions API, which returns calibrated probabilities for closed
  questions without output tokens, so thresholds can be tuned later from
  reviewed examples.
- **Several samples in one Decisions request**: the input would hold unrelated
  items for each question, which the API guide says lowers accuracy.
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
