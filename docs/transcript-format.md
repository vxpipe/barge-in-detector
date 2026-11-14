# Transcript format and cleaning

Status: decided 2026-11-14; cleaner version 1 implemented. This is the
model's input contract: the training data and the runtime caller (Vxpipe) must
produce exactly the same cleaned transcript for the same conversation.

## Decision

- **Generation keeps the original.** Gemini writes natural text with normal
  casing and punctuation, and with speaker names or roles as it likes
  ("Hotel Manager", "Guest", "Priya"). Generation does no cleaning.
- **Cleaning is a separate, versioned step** owned by one cleaner module. It
  turns an original transcript window into the cleaned model input.
- **Both are stored.** The `conversations` table keeps the original, never
  modified. The `examples` table has both the original and the cleaned
  transcript as columns, plus the `cleaner_version`. Changing the cleaner and
  re-running it updates the cleaned columns and re-exports; it never touches
  originals and makes no API calls. See [data format](data-format.md).

## Cleaned transcript

One line per turn, oldest first, joined by `\n`:

```text
s1: haan theek hai lekin
s2: aapka order kal tak pahunch jayega aur
s1: haan haan but change the time
```

- 2 or 3 lines. The last line is always the candidate utterance.
- Speaker labels are `s1`, `s2`, `s3`, numbered by **first appearance within
  the window**, so the same person keeps the same label in that window. Two
  speakers give patterns such as `s1, s2, s1`; three speakers give
  `s1, s2, s3`.
- Labels are ASCII, followed by `: ` (colon and one space), then the cleaned
  text. Cleaning rules apply to the text only, never the label.

## Cleaning rules (cleaner version 1)

Applied to each turn's text, in order:

1. **Unicode NFC** normalization.
2. **Lowercase** with locale-independent Unicode lowercasing (Python
   `str.lower()`, not `casefold()`). Uncased scripts (Devanagari, Kannada,
   Tamil, Arabic, CJK, …) are unaffected.
3. **Apostrophes inside a word are deleted** without a space: `don't` →
   `dont`, `d'accord` → `daccord`.
4. **Every other punctuation character** (Unicode general category `P*`,
   including `।`, `،`, `؟`, `¿`, `…`, quotes, dashes and hyphens) is replaced by
   a space: `mm-hmm` → `mm hmm`.
5. **Kept as they are:** letters, digits, combining marks (`M*`, e.g. Indic
   vowel signs and virama), zero-width joiner and non-joiner (needed in Indic
   scripts and Persian), symbols (`S*`, e.g. `₹`, `+`), and `%` and `‰`, which
   Unicode classes as punctuation but which belong to the number.
6. **Other invisible format characters** (`Cf`, e.g. soft hyphen, bidi marks)
   are deleted; separators and control characters become a space.
7. **Whitespace collapsed** to single spaces and trimmed.
8. A turn whose text becomes empty is invalid; the example builder rejects it
   rather than emitting an empty line.

Numbers are not converted between digits and words; the text keeps whatever
form was generated.

### Why lowercase rather than uppercase

- Multilingual tokenizers are trained mostly on ordinary text, where lowercase
  dominates. Uppercase splits into more, rarer tokens, which costs input budget
  and latency.
- Most scripts the detector covers have no case, so the choice only affects
  Latin, Cyrillic, Greek, Armenian and Georgian text; lowercase keeps those
  closest to normal written text.
- Casing from speech-to-text is inconsistent anyway; removing it removes a cue
  the model should not rely on.

### Training without cleaning (to be compared)

User idea, 2026-11-14: train on text with casing and punctuation kept. Every
example also stores `transcript_raw`: the same `s<n>` labels with the text as
written (NFC and collapsed whitespace only). Checkpoint D trains on both and
compares on held-out data, ideally including real speech-to-text samples:
generated punctuation is perfect while speech-to-text punctuation and casing
are inconsistent, so the raw model may learn cues the runtime does not
reliably have. The runtime contract follows whichever wins.

### Known trade-off

Removing punctuation also removes `?` from the agent's turn, which is a useful
sign that the agent just asked a question. The model still has the question's
wording and `agent_state`. If evaluation shows a real loss, a later cleaner
version could keep sentence-final `?` on agent lines only, since agent text
comes from the agent's own response rather than speech-to-text. That would
need the runtime to send agent text the same way.

## Runtime parity

- The cleaner (`barge_in_detector/cleaner.py`) ships with golden test
  cases in `tests/fixtures/cleaner_golden.yaml` (original text or window →
  cleaned) covering every rule and script family. The fixture is YAML so it
  stays diffable outside Git LFS. The Vxpipe side must pass the same cases,
  whether it reuses this cleaner or reimplements it.
- `cleaner_version` is part of the model card. A model trained on one cleaner
  version must be served with input cleaned by that version.

## Speakers and roles

Decided by the user on 2026-11-14: **the last line is always the speaker being
judged** (with 2 lines, the second; with 3 lines, the third). The model is not
told who any speaker is; earlier lines are context.

- Generation produces 2- or 3-speaker conversations whose last line is the
  speaker who barges in or backchannels. The line before it is the turn being
  spoken over, so consecutive lines always have different speakers.
- The possible windows are therefore `s1, s2` (2 lines), and `s1, s2, s1` or
  `s1, s2, s3` (3 lines). The `speaker_pattern` factor chooses between the two
  3-line patterns.
- The cleaned transcript **always uses `s<n>` labels**, numbered by first
  appearance in the window. The names or roles Gemini wrote ("Hotel Manager",
  "Guest", "Priya") are kept only in the original columns and never reach the
  model.
- Conversations are between any people, not only an agent and a caller. Roles
  in the original turns are `agent` for the speaker being spoken over (the
  name is historical), `listener` for the judged person and `other` for a
  third person; the example builder uses them, for example to cut the turn
  being spoken over at the overlap point.
