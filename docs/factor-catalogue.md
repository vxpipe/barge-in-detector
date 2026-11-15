# Factor catalogue

Status: draft 0.8.0, 2026-11-14: Fable's review applied and accepted by the
user; wording made neutral between speaker and listener (no agent assumed). The catalogue itself is
versioned data under [`generation/factors/`](../generation/factors/); this
document explains it.

## Purpose

Every generated sample is defined by a spec: one value for each factor below.
Each distinct spec is a criterion in the `targets` table with a required sample
count.
The plan builder chooses specs deterministically, applies the constraints and
derives the label from the listener's intent. Gemini only writes text that matches
the spec. See [data format](data-format.md) for how specs, conversations and
examples are stored.

## Scope

Every sample is a conversation between people in the domain's situation: the
**speaker** has the floor and the **listener** says the last line, the one
judged (user decision, 2026-11-14). The model is deployed where the speaker
is a voice agent, but the data does not assume one. Factor names that say
`agent` (`agent_activity`, `agent_state`) mean the speaker; `caller_intent`
and `caller_mood` mean the listener.

Every input is the listener speaking to the speaker. Echo of the speaker's own voice,
background speech, TV or radio, and transcription glitches from noise are out of
scope: noise cancellation and audio cleaning upstream handle them. Transcripts
arrive cleaned up, so generated text is clean and no ASR noise is simulated.
The catalogue therefore has no intents, labels, rules or variants for any of
these.

## Files

| File | Contents |
| --- | --- |
| `catalogue.yaml` | Factor list, values and descriptions; derived and plan-assigned fields |
| `languages.yaml` | 14 languages (2026-11-15; was 36): scripts, romanization usage, code-switch partners, regional variants, tier, rollout phase |
| `domains.yaml` | 22 conversation categories with 77 concrete scenarios; optional `prompt_notes` per domain |
| `intents.yaml` | 24 listener intents, each with its label and example hints |
| `constraints.yaml` | 20 rules removing impossible or meaningless combinations |

Every file has a `version`, and any change bumps the catalogue version.
`combination_id`s depend on factor values only, so added values leave existing
criteria alone; removed or renamed values retire the criteria that used them.
`uv run python -m barge_in_detector check-catalogue` validates every reference and that every
intent has at least one valid combination.

## Factors

| Factor | Values | Why it matters |
| --- | --- | --- |
| `language` | 14 languages | Coverage requirement (user's list, 2026-11-14, cut to 14 on 2026-11-15) |
| `regional_variant` | e.g. en-IN, en-US, es-MX, pt-BR | Vocabulary and backchannel habits differ by region |
| `script_form` | native, romanized, mixed_script | Transcripts arrive in native script or Latin characters, sometimes mixed |
| `code_switching` | none, mixed | Callers mix languages, e.g. "சரி சரி, but change the time" (2026-11-15: how they mix is not a factor) |
| `code_switch_set` | e.g. `en`, `en+hi`, from the language's sets | Which language(s) are mixed in; may be two, e.g. Kannada + English + Hindi |
| `domain` / `scenario` | 22 / 77 | Vocabulary and what listeners react to |
| `speaker_pattern` | returning_speaker, new_speaker | 3-line windows can be `s1, s2, s1` or `s1, s2, s3`; the last speaker is always the one judged |
| `agent_activity` | 15, from greeting to reading numbers, asking a question and closing | What the listener is reacting to; the strongest context signal after cut position |
| `cut_position` | start, early, middle, near_end, just_finished | Mid-sentence vs. just asked a question |
| `caller_intent` | 24 intents | Determines the label |
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
| `backchannel` | continuer, acknowledgment, agreement_token, reaction_token, gratitude_token, tracking_repeat, go_ahead | 0 |
| `barge_in` | floor_request, correction, disagreement, clarification_question, repeat_request, new_request, early_answer, answer_to_question, complaint, request_human, end_call, turn_continuation, skip_request, hello_check, language_switch_request | 1 |
| `ambiguous` | collaborative_completion, thinking_filler_after_question | evaluation only |

Generator/judge disagreements on the other intents are also kept (in
`rejections` and the raw responses) as a pool of ambiguous candidates for
evaluation review.

With `agent_state=just_finished`, `barge_in` means "the listener takes a real
turn", for example answering the speaker's question with "yes".

**A bare "ok" is always a backchannel** (user decision, 2026-11-14), whether
the speaker is mid-sentence, paused between sentences or finished with a
statement. This avoids identical inputs with opposite labels: the transcript
cannot show whether a complete-looking sentence had more queued. The one
exception is a finished question, where "ok" answers it (`answer_to_question`,
for example "shall I book it?" → "ok"). What the speaker does after a
backchannel at `just_finished` (continue, or wait briefly and move on) is the
runtime's choice.

## Code-switching in Indian languages

Sets in `languages.yaml` follow the user's rule (2026-11-15, replacing the
three-language sets of 2026-11-14):

- Every Indian language (hi, kn, ta, te, ml, pa, bn, ur) appears
  alone or mixed with English.
- Indian English (`en-IN`) mixes one Indian language.

English varieties (user decision, 2026-11-15): US, Indian and Singaporean
English only (`en-US`, `en-IN`, `en-SG`); Tamil is Indian Tamil only
(`ta-IN`). Singaporean English mixes in Mandarin in Simplified characters
(`mixed_in_script: native`, locale `zh-Hans-SG`). No pinyin anywhere (user
decision, 2026-11-15): Mandarin is written in characters only, and Mandarin-led
utterances mix in English. The sample plan gives English en-US 40%, en-IN 40%, en-SG 20%
(`variant_shares`).

Words of an Indian language mixed into Indian English are written in that
language's native script (`mixed_in_script: native`, user decision,
2026-11-15), for example "हाँ हाँ, but change the time".

## Constraints, in brief

- Romanized or mixed script only for languages where that is used; no
  code-switching for languages without a partner (English).
- `answer_to_question` needs a finished question. `early_answer` needs a
  question, list or read-back still being spoken, only near the end for yes/no
  questions and read-backs, and must carry content (an option, or "yes book
  it"), never a bare "yes". Backchannel intents are not allowed right after a
  finished question (there, "yes" is an answer), and affirmative backchannels
  are not generated near the end of a yes/no question or read-back, where they
  read as early answers.
- `tracking_repeat` needs the speaker to be reading something out; corrections
  need something already said; floor and skip requests need the speaker still
  speaking; `hello_check` needs the speaker to have gone quiet;
  `turn_continuation` needs the judged speaker's earlier turn in the window.

## Size of the space

The utterance track (2026-11-15) has 666 valid combinations across the 14
languages: 444 barge-in and 222 backchannel. The `utterance-all` plan takes
every one of them with the same sample count.

## Points for review

1. **Ambiguous evaluation items** come from two intents plus generator/judge
   disagreements. Check the pilot's disagreements before adding intents.
2. **Kannada and Tamil example hints** in `intents.yaml` (haudu/ಹೌದು,
   sari/ಸರಿ, aamaa/ஆமா, sari/சரி) are unverified and need your check, as do all
   Hindi hints.
3. **Domains and scenarios**: anything missing for the calls Vxpipe handles?
4. **Language manifest**: romanization usage, code-switch sets and variants
   for tier C languages are first estimates. The manifest holds exactly the
   36 languages the user chose on 2026-11-14 (0.5.0 removed 80 others, with
   every code-switch set that mixed one of them in; they remain in git
   history).

## Rejected alternatives

- **Free-form prompts ("generate diverse barge-in examples")**: coverage and
  class balance would depend on the model's habits, and nothing could be
  regenerated or audited.
- **Letting Gemini choose which label to write**: label balance would drift
  with prompt wording. The intent fixes the intended label; Gemini's own
  classification and the judge's must agree with it.
- **Finer-grained intents merged in 0.7.0** (Fable's review): `understanding`
  into `acknowledgment`; `surprise_token`, `empathy_token` and `laughter` into
  `reaction_token`; `stop_request` and `hold_request` into `floor_request`;
  `urgent_interjection` into `complaint`. The pairs were hard to tell apart in
  text and gave the same label. `mixed_signal` was cut: an intent asking for
  "either way" text is hard to generate on purpose, and disagreements give a
  more natural ambiguous pool.
- **Domains merged or cut in 0.7.0**: e-commerce + logistics, food delivery +
  ride-hailing, utilities + government (`public_services`); software issues
  into device setup; cleaning into repair; personal-assistant small talk
  removed. Added front desk, verification and onboarding, agriculture and rural
  services, a failed UPI payment and a pre-approved loan offer.
- **Turn shape and prefixes as spec factors**: would multiply
  generation cost for variations that can be derived from one conversation.
- **A separate "ok that takes the turn" intent** (0.5.0
  `acknowledgment_response`, barge-in, paired with `acknowledgment` after a
  pause): removed in 0.6.0. Both members produced the same transcript with
  opposite labels, since a pause after a complete sentence and a finished turn
  look identical. A bare "ok" is now always a backchannel.
- **Handling non-caller audio in this model** (0.1.0 had a
  `not_caller_speech` label for echo, background speech, media audio, ASR
  artefacts and side talk): removed by the user's decision. Those are the job of
  noise cancellation and audio cleaning; this model only detects barge-in.
