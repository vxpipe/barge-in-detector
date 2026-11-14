# Factor catalogue

Status: draft 0.5.0, 2026-11-14, awaiting user review. The catalogue itself is
versioned data under [`generation/factors/`](../generation/factors/); this
document explains it.

## Purpose

Every generated sample is defined by a spec: one value for each factor below.
Each distinct spec is a criterion in the `targets` table with a required sample
count.
The plan builder chooses specs deterministically, applies the constraints and
derives the label from the caller intent. Gemini only writes text that matches
the spec. See [data format](data-format.md) for how specs, conversations and
examples are stored.

## Scope

Every input is the caller speaking to the agent. Echo of the agent's own voice,
background speech, TV or radio, and transcription glitches from noise are out of
scope: noise cancellation and audio cleaning upstream handle them. Transcripts
arrive cleaned up, so generated text is clean and no ASR noise is simulated.
The catalogue therefore has no intents, labels, rules or variants for any of
these.

## Files

| File | Contents |
| --- | --- |
| `catalogue.yaml` | Factor list, values and descriptions; derived and plan-assigned fields |
| `languages.yaml` | 116 languages: scripts, romanization usage, code-switch partners, regional variants, tier, rollout phase |
| `domains.yaml` | 22 conversation categories with 70 concrete scenarios |
| `intents.yaml` | 26 caller intents, each with its label, allowed lengths and example hints |
| `constraints.yaml` | 18 rules removing impossible or meaningless combinations |
| `contrastive.yaml` | 7 groups of specs generated together so the same words get different labels |

Every file has a `version`. Changing any value bumps the catalogue version,
which changes `combination_id`s and therefore requires a new plan.

## Factors

| Factor | Values | Why it matters |
| --- | --- | --- |
| `language` | 116 languages | Coverage requirement |
| `regional_variant` | e.g. en-IN, en-US, es-MX, pt-BR | Vocabulary and backchannel habits differ by region |
| `script_form` | native, romanized, mixed_script | Transcripts arrive in native script or Latin characters, sometimes mixed |
| `code_switching` | none, between_sentences, within_sentence | Callers mix languages, e.g. romanized Hindi with English |
| `code_switch_set` | e.g. `en`, `en+hi`, from the language's sets | Which language(s) are mixed in; may be two, e.g. Kannada + English + Hindi |
| `domain` / `scenario` | 22 / 70 | Vocabulary and what callers react to |
| `speaker_pattern` | returning_speaker, new_speaker | 3-line windows can be `s1, s2, s1` or `s1, s2, s3`; the last speaker is always the one judged |
| `agent_activity` | 14, from greeting to reading numbers to asking a question | What the caller is reacting to; the strongest context signal after cut position |
| `cut_position` | start, early, middle, near_end, just_finished | Mid-sentence vs. just asked a question |
| `caller_intent` | 26 intents | Determines the label |
| `development` | stable, backchannel_then_barge_in, filler_then_barge_in | Interim transcripts that change meaning as they grow |
| `repetition` | single, repeated | "yes yes", "oui oui", "no no no" |
| `length_bucket` | 1, 2–3, 4–8, 9+ words | Avoids the model learning "short means backchannel" |
| `register` | formal, casual | Wording varies with formality |
| `caller_mood` | neutral, friendly, hurried, confused, frustrated | Wording and urgency |

Derived, not chosen: `agent_state` (from `cut_position`) and `label` (from
`caller_intent`).

Not spec factors: turn shape (2 or 3 turns) and interim prefixes.
These are produced later from each conversation, so one generated conversation
yields all of them.

## Labels

Each intent sets the **intended label**. The training label is Gemini's own
classification of the generated sample, accepted only when it and a blind
judge both agree with the intended label; see
[data format: labels](data-format.md#labels).

| Intended label | Intents | Training label |
| --- | --- | --- |
| `backchannel` | continuer, acknowledgment, agreement_token, understanding, surprise_token, empathy_token, laughter, gratitude_token, tracking_repeat | 0 |
| `barge_in` | stop_request, hold_request, correction, disagreement, clarification_question, repeat_request, new_request, early_answer, answer_to_question, acknowledgment_response, complaint, request_human, urgent_interjection, end_call | 1 |
| `ambiguous` | collaborative_completion, thinking_filler_after_question, mixed_signal | evaluation only |

With `agent_state=just_finished`, `barge_in` means "the caller takes a real
turn", for example answering the agent's question with "yes", or
`acknowledgment_response`: an "ok" that hands the turn back after the agent
finishes a statement. The same "ok" after the agent has only paused
mid-explanation is `acknowledgment` (backchannel). The runtime uses
the same output either way: the caller has the floor.

## Code-switching in Indian languages

Sets in `languages.yaml` follow the user's rule (2026-11-14):

- Hindi mixes English, or English + a regional language.
- Regional Indian languages (kn, ta, te, ml, mr, gu, pa, bn, or, as, kok, mai,
  bho) mix English, or English + Hindi.
- Indian English (`en-IN`) mixes Hindi, a regional language, or both.

Sets involving Hindi are limited to Indian varieties (`ta-IN`, `bn-IN`,
`pa-IN`); Pakistani Punjabi mixes Urdu instead. English is written in Latin
script only, so Hindi or regional words in English-led speech are romanized.

## Constraints, in brief

- Romanized or mixed script only for languages where that is used; no
  code-switching for languages without a partner (English).
- `answer_to_question` needs a finished question; `early_answer` needs a
  question, list or pitch still being spoken; backchannel intents are not
  allowed right after a finished question (there, "yes" is an answer).
- `tracking_repeat` needs the agent to be reading something out; corrections
  need something already said; stop and hold requests need the agent
  still speaking.
- Only barge-in intents can develop from a backchannel or filler, and only in
  utterances of four or more words.
- Per intent: allowed lengths and whether repetition is allowed.

## Size of the space

Checked on 2026-11-14 with a validation script (see the labnotes):

- 37,439 of 218,400 combinations of the intent-related factors are valid:
  10,029 backchannel, 26,260 barge-in and 1,150 ambiguous.
- 1,025 language-side combinations (language × variant × script ×
  code-switching set).
- About 10.7 billion valid full specs when scenarios, register and speaker pattern are included.

The full space is never enumerated. The plan builder samples with stratified
quotas and all-pairs coverage. Its quotas, not these raw counts, set the class
balance, so barge-in's larger share of the raw space does not carry into the
training data.

## Points for review

1. **Ambiguous intents** are evaluation-only. Are three enough?
2. **Kannada and Tamil example hints** in `intents.yaml` (haudu/ಹೌದು,
   sari/ಸರಿ, aamaa/ஆமா, sari/சரி) are unverified and need your check, as do all
   Hindi hints.
3. **Domains and scenarios**: anything missing for the calls Vxpipe handles?
4. **Language manifest**: romanization usage, code-switch sets and variants
   for tier C languages are first estimates.

## Rejected alternatives

- **Free-form prompts ("generate diverse barge-in examples")**: coverage and
  class balance would depend on the model's habits, and nothing could be
  regenerated or audited.
- **Letting Gemini choose the label**: labels would drift with prompt wording.
  The intent fixes the label; the judge pass only checks the text matches.
- **Turn shape and prefixes as spec factors**: would multiply
  generation cost for variations that can be derived from one conversation.
- **Handling non-caller audio in this model** (0.1.0 had a
  `not_caller_speech` label for echo, background speech, media audio, ASR
  artefacts and side talk): removed by the user's decision. Those are the job of
  noise cancellation and audio cleaning; this model only detects barge-in.
