# Common-expression coverage supplement

Owner decisions, 2026-11-15: audit common words and phrases across every language
and regional variety; generate missing forms with the existing tools, without an
LLM judge; estimate cost before paid calls and keep total expense below $30.
Keep the previous dataset alongside the new dataset. This is an additive data
experiment, not a decision to replace the classifier with a word list.

## Preserved original and separate supplement

`data/export/utterances.csv`, `data/splits/`, existing model checkpoints and the
reviewed challenge stay intact. New samples are written through the existing
storage module into `data/work/phrase-coverage-v1.duckdb` and exported to
`data/supplements/phrase-coverage-v1/utterances.csv`. Each store has one utterances
table and one local request log. Neither store replaces the other.

The supplement uses the existing cleaner, language manifest, intent definitions,
UUID functions, Gemini client and storage/export. Schema 11 columns are unchanged.
Its `raw_sample` uses a documented envelope: `sample` is the exact generated
object; `coverage` contains the complete deterministic criterion; `verification`
is `local_checks_only_no_judge`. The extra `coverage_form` and `anchor_intent`
factors participate in criterion UUIDs and are preserved in that envelope. They
are defined by the separate versioned plan, not added to legacy criteria.
Judge fields are null. Here, accepted means passed local checks, not human or
judge verification. All generated languages remain synthetically labeled.

## Checklist and generation

The source of criteria is
[`phrase-coverage-v1.yaml`](../generation/plans/phrase-coverage-v1.yaml).
It covers all 14 languages, 17 regional varieties, and every permitted two-language
combination from the existing manifest. Native scripts are required.

For each of the six backchannel intents, request:

- A common standalone expression.
- Repetitions of the same expression.
- Combinations of acknowledgments.
- Longer acknowledgments that leave the speaker the floor.
- A backchannel followed by an explicit stop/wait request.
- A backchannel followed by a correction.

Standalone/repetition criteria use unmixed speech. Combinations, expansions and
contrasts also cover the 20 permitted language-mixture specifications. Separate
short interruption criteria cover floor requests, corrections and clarification
questions. Each generated variant records its opening backchannel as `anchor`,
allowing an auditable per-expression inventory rather than a bare row count.
Different criteria may discover different anchors; missing variants for an
anchor remain gaps, not evidence of exhaustive coverage.

The plan requests three distinct accepted phrases for each of 1,203 criteria
(3,609 total), with at most six requested samples per criterion, including
failed attempts. Each request asks only for the remaining accepted count.
This allowance grows when the required count grows, without resetting the cost
ledger or changing criterion IDs. It favors common wording
over forced novelty. Duplicate original/supplement strings, exact evaluation
strings, malformed output, wrong scripts, label mismatches and mechanically
invalid repetitions are rejected and retained with their reason. Linguistic
naturalness, semantics and regional authenticity still require human review.
Quota shortfalls after bounded attempts are reported rather than silently filled
with invented phrases. A candidate inventory is not a complete lexicon.

## Cost estimate and safeguards, recorded before generation

Google's [Standard Gemini 3.8 Flash pricing](https://aistudio.google.com/docs/pricing#gemini-3.8-flash)
on 2026-11-15 is $0.75/million input tokens and $3.75/million output tokens,
including thinking, through 2026-12-31. The unchanged generator selection is
`gemini-3.8-flash`; use its supported `low` thinking setting.

The initial offline estimate was **$3.53 for the first pass**, **$7.07 for two
full passes**. After tightening English-only script instructions in prompt v2,
the estimate is **$3.65 first pass**, **$7.29 for two full passes**. This assumes
UTF-8 prompt bytes/3 input tokens and
120 output tokens per phrase plus 256 thinking tokens per request. These are
estimates, not provider quotes. The hard new-run cap is **$8**. Historical logged
generation cost is **$13.1659674**; adding a **$2 uncertainty reserve** and the new
cap gives **$23.1659674**, below the owner's $30 ceiling. This is based on local
logs, not a reconciliation of every charge on the provider account.

Before submission, every request durably reserves **$0.013824**: 8,192 input
tokens plus a provider-enforced 2,048 output-token limit (thinking included).
The runner bounds prompt/schema bytes conservatively, disables SDK and wrapper
retries and explicit caches, and charges cached input at full price. A complete
usage response replaces the reservation with usage-based cost. Missing usage,
lost responses and crashes retain the full reservation across resume. In-flight
reservations share the same cap. The runner stops before the next request would
exceed $8, even if some coverage remains incomplete. At simultaneous token
maxima, all planned attempts would exceed that cap; the cap takes precedence.

Keep the local request log: deleting/rebuilding it would discard cost accounting.
DuckDB's exclusive writer lock prevents two processes writing this same ledger.
Pricing validity is checked before a paid run. Existing generic generation is
unchanged apart from removing hidden SDK retries and supporting an optional
output-token cap. Old received requests in the original store are untouched.

```sh
# Offline estimate; zero API calls.
uv run python -m barge_in_detector.generation.phrase_runner
# Paid, resumable supplement run; existing logged spend counts toward its cap.
uv run python -m barge_in_detector.generation.phrase_runner --run
```

## Evaluation protocol

Keep the original train/validation/test files fixed. Split the supplement
separately, deduplicating cleaned text against the original and every challenge
row before combining train portions. Preserve new assignments when appending
data. Report original test results and supplement test results separately, with
per-language errors and prefix transitions. The 98 owner-reviewed English cases
are development evidence after repeated use; improvement needs a fresh reviewed
set. Do not claim a new model is better merely because it fits these phrases.

Completed run, coverage gaps, cost and the fixed-architecture additive experiment:
[phrase coverage results](phrase-coverage-results.md).
