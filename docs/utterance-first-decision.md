# Decision: classify the overlapping utterance, not the conversation

Status: decided by the user, 2026-11-15. The first implementation checkpoint
is in progress; see the
[utterance-first milestone](../labnotes/milestones/utterance-first-backchannel-classifier.md).

## Context

The project brief (AGENTS.md) and the
[first milestone](../labnotes/milestones/multilingual-text-barge-in-classifier.md)
describe a model that reads a 2-3 line window and decides two things at once:

- whether the caller's words interrupt the agent or answer a question the
  agent had just finished asking (`agent_state` `just_finished`);
- whether overlapping words are a backchannel or a real interruption.

The user clarified on 2026-11-15 that **Vxpipe already makes the first
decision.** The runtime knows whether the caller's speech overlaps the agent's
playback or comes after the agent finished, and handles answers itself. The
model is called only when the caller is speaking **over** the agent, and
answers one question: is this overlapping speech a backchannel (let the agent
keep talking) or a real interruption (stop and give the caller the floor)?

## Decision

1. **The model input is the caller's overlapping utterance**, possibly a
   partial (interim) transcript, cleaned as in
   [transcript format](transcript-format.md). One optional preceding line (the
   tail of what the agent had said) may be added later, but only if the
   ablation in the milestone shows it is worth it.
2. **The main training set is single labelled utterances** generated per
   language, script form, code-switching, intent, development, repetition and
   length. No domain, scenario, speaker pattern, cut position, register or mood.
3. **A small context slice** (about 10-15% of samples) covers the cases where
   the agent's words decide the label: number and name readouts (the caller
   echoing digits is tracking; different digits are a correction), "no" after a
   negative statement, and echoing a list item versus choosing it. It is a later
   checkpoint.
4. **Intents that need a finished agent turn leave the utterance track**:
   `answer_to_question`, `hello_check` and `thinking_filler_after_question`.
   `tracking_repeat` and `collaborative_completion` need the agent's words and
   move to the context slice.
5. **No judge for the first utterance runs** (user decision, 2026-11-15). The
   generator's own label must equal the intended label; whether a judge pass is
   needed is decided from the human review of the first samples.
6. **Two labels only** (user decision, 2026-11-15): `backchannel` or
   `barge_in`. The track has no ambiguous class; the catalogue's ambiguous
   intents are excluded, and any other label is rejected as `unknown_label`.
7. **The model**: fine-tune a small permissively licensed multilingual encoder
   (multilingual-e5-small or Multilingual-MiniLM-L12, both MIT) on the
   utterance. SetFit is a training method, not a faster model; at thousands of
   samples plain fine-tuning matches it at the same inference cost.

## Why

- With the runtime deciding answer versus interruption, the cases that
  needed the window (a bare "yes" after "shall I book it?") no longer reach the
  model. Fable's review (2026-11-15, recorded in the
  [labnotes](../labnotes/20261115-0100-utterance-first-generation.md)) estimates
  that the agent's words flip the label of only about 3-8% of overlapping
  utterances (estimate, not measured).
- The conversation pipeline's first runs accepted 21% of samples. Most
  rejections came from the parts the utterance track drops: cut position,
  contrastive sets and intents that depend on context.
- Inputs of 8-16 tokens instead of 60-128 cut encoder latency to an estimated
  2-4 ms at int8 (to be measured on the reference CPU).

## Rejected alternatives

- **Keep the 2-3 line conversation as the only input.** It solves a problem
  the runtime already solves, and costs more per accepted sample.
- **A tiered system** (word-level fast path, then a context model). The fast
  path needs to know when a word is ambiguous, which is a word list; AGENTS.md
  forbids shipping one.
- **A character n-gram linear model as the product.** It is a useful baseline,
  especially for romanized spellings, but it is close to a word list and cannot
  learn the development of "yes yes but".

## Implications

- AGENTS.md's input contract (2 or 3 turns, `agent_state`) and the first
  milestone's Checkpoint A contract are superseded for the model input. The
  conversation tables and their exports stay; nothing generated is deleted.
- Partial transcripts still matter: "yes yes" must score low and "yes yes but"
  high, so `barge_in_from_word` and prefix expansion carry over.
- Duplicates matter more: "ok" will be generated thousands of times. Splits and
  deduplication work on the cleaned text.
- Calibration uses realistic priors: most overlapping speech is backchannel.

## Verification

- The ablation in the milestone (utterance only, utterance plus the agent's
  last line, character n-gram baseline) decides whether the context line is
  kept: keep it only if it gains about 2 points or more on the context slice and
  the whole test set, within the latency budget.
