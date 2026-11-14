# Factor catalogue draft

2026-11-14. User asked for the categories and factors that drive deterministic
data generation (milestone Checkpoint A).

## Done

- `generation/factors/` (version 0.1.0, status draft):
  - `catalogue.yaml`: 14 spec factors, derived `agent_state` and `label`,
    plan-assigned fields, and the factors deliberately left out.
  - `languages.yaml`: 116 languages. Tier A: en, hi (phase 1), kn, ta
    (phase 2). 112 tier C. Romanized usage: 19 common, 28 occasional, 69 none.
  - `domains.yaml`: 22 domains, 70 scenarios.
  - `intents.yaml`: 30 intents: 9 backchannel, 13 barge_in, 5
    not_caller_speech, 3 ambiguous (evaluation only).
  - `constraints.yaml`: 22 rules.
  - `contrastive.yaml`: 8 groups.
- `docs/factor-catalogue.md`: rationale, label table, constraints summary,
  space size, review questions, rejected alternatives.
- `docs/data-format.md` factor columns synced (added regional_variant, scenario,
  repetition, caller_mood; removed noise_type; renamed development values;
  `mixed_script`; ambiguous label). AGENTS.md factor list now points to the
  catalogue.
- Also this session: `.gitattributes` puts every `*.parquet`, `*.json`,
  `*.jsonl`, `*.jsonl.zst` in Git LFS (user requirement). git-lfs 3.7.1 is
  installed locally.

## Decisions

- Noise (echo, background speech, media, ASR artefacts, side talk) became
  intents with label `not_caller_speech` instead of a separate factor: it is
  what the candidate is, not something layered on it.
- `agent_state` is derived from `cut_position` instead of chosen, removing
  contradictory combinations.
- Turn shape, prefixes and ASR noise stay out of the spec (derived in Stage 4).
- With `agent_state=just_finished`, label `barge_in` means "real caller turn".
- Backchannel intents are forbidden right after a finished question; "hmm"
  there is `thinking_filler_after_question` (ambiguous).
- One contrastive group originally paired two same-label members
  ("ok" mid-explanation vs. after a finished statement); replaced with "ok"
  during read-back vs. "ok" confirming after the question.

## Verification

Throwaway validator (scratchpad, run with
`uv run --no-project --with pyyaml python -I check_catalogue.py generation/factors`)
checked: all YAML parses; unique language codes, scenario and intent IDs;
code-switch partners exist; every constraint and contrastive member names
real factors/values; every intent has at least one valid combination; every
contrastive member is satisfiable. Result: no errors.

Space size: 40,013 of 252,000 intent-side combinations valid (backchannel
10,029; barge_in 25,810; not_caller_speech 3,024; ambiguous 1,150); 869
language-side combinations; ~4.9 billion full specs. Confirms sampling with
quotas + all-pairs coverage rather than enumeration.

The validator was not committed: Checkpoint B's schema module owns catalogue
validation, with tests written first.

## Open (for user review; listed in docs/factor-catalogue.md)

- Side talk label; "ok" after a finished statement; number of ambiguous
  intents; Hindi/Kannada/Tamil example hints; missing domains; tier C language
  estimates (Kannada lists Hindi as a code-switch partner, Tamil does not).

## Scope change: non-caller speech removed (user decision, same day)

- User: echo of the agent's voice, background speech and similar are the job of
  noise cancellation and audio cleaning; this model only detects barge-in.
- Catalogue bumped to 0.2.0: removed the `not_caller_speech` label and its 5
  intents (agent_echo, background_speech, media_audio, asr_artefact,
  side_talk), 5 constraint rules that only served them, and 2 contrastive
  groups (echo vs. repeat request, side talk vs. hold request). The side-talk
  review question is gone with it.
- ASR-style rendering of the caller's own words in Stage 4 stays: it models
  transcription of caller speech, not non-caller audio.
- AGENTS.md, milestone, data format (`label_id` 0 = backchannel) and catalogue
  doc updated.
- Revalidated: no errors. 25 intents (9 backchannel, 13 barge_in, 3
  ambiguous); 36,989 of 210,000 intent-side combinations valid (backchannel
  10,029, barge_in 25,810, ambiguous 1,150); ~4.5 billion full specs.

## Code-switching sets (user decision, same day)

- User: Indians mix Hindi + English, regional + English, or Hindi + English +
  regional. A single code-switch language could not express three languages.
- Catalogue 0.3.0: `code_switch_language` became `code_switch_set` (codes
  joined by `+`). `languages.yaml` now has `code_switch_sets`, each a list of
  languages optionally limited to regional variants. Hindi: en, en+R. Regional
  Indian languages: en, en+hi. en-IN: hi, R, hi+R. Hindi sets only for Indian
  varieties (ta-IN, bn-IN, pa-IN); pa-PK mixes ur / en+ur. Other languages
  keep their single-language partners. Converted with a scratchpad script.
- Revalidated: no errors; 1,025 language-side combinations; ~5.3 billion full
  specs. The "Kannada lists Hindi, Tamil does not" review question is resolved.

## ASR noise out of scope (user decision, same day)

- User: generating data with transcription noise is not our concern; the
  transcripts arrive cleaned up. This corrects the earlier note above that
  ASR-style rendering "stays".
- Removed the Stage 4 noise variants (`noise_variant` column, noise in
  `example_id`), the ASR-noise/TTS→STT task, the `tts_stt_roundtrip` eval
  source, and the "real transcripts are noisy" guidance. AGENTS.md now says
  inputs are clean and generated text is clean. Script mixing and
  code-switching stay in scope.

## Speaker pattern (user decision, same day)

- User: the last line is always the speaker barging in (or backchannelling);
  there is no need to know who s3 is. Generate 2- or 3-speaker samples with the
  judged speaker last and convert speaker names/roles to s1, s2, s3 in the
  cleaned transcript.
- Catalogue 0.4.0 adds `speaker_pattern`: `returning_speaker` (s1, s2, s1) or
  `new_speaker` (s1, s2, s3) for the 3-line window. The line before the last is
  the turn being spoken over, so consecutive lines never share a speaker; 2-line
  windows are always s1, s2. No new constraints were needed.
- `docs/transcript-format.md` "Speakers and roles" rewritten; data format column
  added; milestone open decision removed.
- Revalidated with no errors (the factor has no constraints, so it only
  doubles the full space to ~10.6 billion).

## Gemini classification and "ok" after a statement (user decision, same day)

- User: "ok" right after the agent finishes can be a response or a backchannel;
  Gemini must return the classification with every sample, because the labels
  come from the reference model.
- Catalogue 0.5.0: new intent `acknowledgment_response` (barge_in: "ok" that
  hands the turn back after a finished statement), rule requiring
  `cut_position=just_finished` and a non-question activity, and contrastive
  group `ok_pause_vs_ok_response` against `acknowledgment` after a pause.
  Derived `label` renamed `intended_label`.
- `docs/data-format.md` gained a Labels section: generator returns sample +
  classification + barge-in word + rationale; a blind judge classifies again;
  accept only if both agree with the intended label, otherwise reject with
  stage `label_mismatch`. New conversation columns `intended_label`,
  `gemini_label`, `gemini_rationale`, `judge_label`,
  `judge_barge_in_from_word`.
- Revalidated, no errors: 26 intents; 37,439 of 218,400 intent-side
  combinations valid (backchannel 10,029, barge_in 26,260, ambiguous 1,150).
