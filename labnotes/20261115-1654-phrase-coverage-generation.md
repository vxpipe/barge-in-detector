# Phrase coverage generation

## Scope and decisions

Owner authorized a common-expression coverage audit and generation across all
14 languages/regions, explicitly without a judge for phrases, and with a cost
estimate before paid calls and a $30 total expense ceiling. Follow-up requires
keeping the old dataset alongside the new data. Original data and splits remain
unchanged; use a separate supplement store/export, then compare additive training.

Existing accepted data: 3,304 rows, 1,084 backchannels. Many canonical standalone
phrases are absent (including English yes/okay and Hindi हाँ/जी/ठीक है), although
some appear as parts of longer utterances. The old plan controls intent/language,
not expression form. Exact reviewed strings are reserved for evaluation.

## Before paid generation

Read-only original request-log audit: Gemini 3.8 Flash $12.7393689 and Decisions
$0.4265985, total $13.1659674; 12 requests have unknown costs. Five old received
requests remain in that original store and are not part of this new track.

Prepared `generation/plans/phrase-coverage-v1.yaml`: 1,203 criteria, 3 samples
each, up to two requests per criterion. All 14 languages, 17 varieties, 20
code-switch specifications; native scripts. Standalone, repetition, combination,
expanded acknowledgment, stop/correction contrasts and short interruptions.

Official pricing checked before generation: $0.75/M input, $3.75/M output
including thinking through 2026-12-31. Model documentation rules out `minimal`;
plan uses `low`. Offline estimate $3.5326 first pass, $7.0652 full two passes.
New cap $8; historical spend + $2 uncertainty reserve + cap = $23.1659674.
Source: https://aistudio.google.com/docs/pricing#gemini-3.8-flash and
https://aistudio.google.com/docs/models/gemini-3.8-flash.

Request reservations include concurrent in-flight calls and unknown outcomes;
resuming preserves reservations. No hidden SDK/wrapper retries, no explicit
cache creation, 2,048 output-token limit. Tests first failed for missing phrase
module/output cap; focused tests now cover fill/no-op, resume, exact exclusions,
script/repetition validation and durable budget accounting.

Implementation and usage: [phrase coverage decision](../docs/phrase-coverage.md).

## Pilot

First 18 submissions returned 401 because the noninteractive login shell inherited
an invalid credential instead of loading the owner's configured environment.
No samples/token usage were returned. Loading `zsh -lic` fixed the read-only model
access check. Added a preflight and stop-on-400/401/403/404 behavior (red/green
test); retain $0.248832 in conservative reservations for those 18 requests.
No credentials were printed or stored. All subsequent live commands use `zsh -lic`.

Successful first pilot: 18 calls, 23 accepted, $0.01456425 usage cost. Inspection
found three romanized Hindi phrases in English-only criteria: Haan, Oh achha,
Arre wah. Retired them through Store with reasons (no text edits or deletion).
Preserved prompt v1; prompt v2 explicitly rules these out. Prompt hashes and
templates identify each response; the same bounds/accounting continue across
prompt versions. Revised estimate $3.6451 / $7.2901, still capped at $8.

Second pilot: 100 calls, 234 accepted cumulative, all 14 languages represented,
$0.110244 usage cost cumulative. Hindi includes हाँ, जी, अच्छा, ठीक है, समझ गया,
बिल्कुल. Existing/reviewed duplicates are recorded as rejected. Full run resumed
after this inspection; no judge calls are used.

## Completion, coverage and preservation

Initial full pass plus bounded retries: 3,413 accepted, $1.33114125 usage cost.
A growth test exposed a request-count attempt limit that would prevent raising
required counts after two previous requests. Changed it to requested-sample
accounting (max_attempts × required_count), matching the original generator's
growth contract, counting failed submissions too. Resuming the remaining
allowances made 176 more requests and reached **3,437 accepted**. No plan counts,
generator model or monetary limits were raised.

Final 1,567 submissions; 1,549 returned usage and 18 credential failures retain
maximum reservations. Usage-based cost $1.43615175; including new reservations
$1.68498375. Historical + new usage $14.60211915; with all uncertainty reserves
$16.85095115. No GPU rented, zero judge calls. Observed input/output maxima
624 / 779 tokens, below reservations/caps. All 14 languages and all 17 regional
varieties are represented. 1,087/1,203 criteria full, 1,191 represented; 116
quota gaps remain after bounded attempts. 1,149 expression checklist entries,
977 missing at least one variant form: not exhaustive lexical coverage.

Every new sample retained: 3,437 accepted, 576 rejected, three retired. Zero
accepted exact overlap with original/evaluation; zero exact duplicate accepted
texts; no exact duplicate label conflicts. Original export, splits and review
files match their six saved hashes. All writes went through Store/export.

No-op verification patched GeminiClient construction to raise, reran the live
entrypoint, and observed zero submissions with unchanged CSV SHA/mtime. Preflight
is now lazy, so an exhausted/completed run makes no API calls and needs no key.
Received responses resume offline before deciding whether a client is needed.

Supplement split 2,278/582/577; derived combined split 4,918/915/908. The original
2,640/333/331 assignments are unchanged. Stable append logic is tested, including
a singleton gaining a held-out example without moving its existing training row.

## Controlled data ablation and CPU evidence

Ran `uv run --group baseline python -m experiments.compare_phrase_supplement`.
Same selected char/word TF-IDF linear architecture, C=10, word scale=.25,
balanced weights, threshold=.5. Train-only vocabularies; all 14 languages; no
hyperparameter/test tuning. Feature construction+fit .568174 s, fit .140747 s
on local CPU; 4,918 training rows. No inputs required 64-token truncation.

Original test 320/331 → 323/331; supplement test 529/577 → 570/577; reviewed
English 90/98 → 91/98. False interruptions 4/50 → 1/50, missed interruptions
4/48 → 6/48, short recall unchanged 26/30, prefix cases 24/24 → 22/24.
Validation 330/333 → 327/333. This is a tradeoff, not a winner: preserve models.
Review misses include new failures on cancel, before-you-continue clarification
and acknowledgment-to-floor-claim transitions. Three original training strings
and one original validation string overlap the challenge; supplement overlap is
zero. Reviewed results remain development evidence, not fresh generalization.

Ran `uv run --group baseline python -m experiments.verify_lexical --run lexical-phrase-v1`.
Separate portable export: 24,243,695 bytes; exact probability parity, zero label
disagreements on original validation/test/challenge. Local AMD EPYC-Genoa, one
thread, warmed batch one, 100 warmups/1,000 samples:

| Scenario | p50 ms | p95 ms | p99 ms |
| --- | ---: | ---: | ---: |
| Multilingual model path | .752 | .883 | .952 |
| Full 64-token model path | 1.899 | 4.168 | 4.777 |
| Whole path | unmeasured | unmeasured | unmeasured |

Vxpipe boundary remains unimplemented. Peak verification-process RSS 494.7 MiB
includes multiple objects and is not isolated serving memory. No deployment,
HF upload, commit, or original data/model replacement was performed.

Reports: `reports/phrase-coverage-v1/` (budget, coverage/checklists, split/no-op
checks and original hashes), `reports/lexical-phrase-v1/` (ablation/predictions,
portable parity/CPU measurements). Durable conclusion:
[phrase coverage results](../docs/phrase-coverage-results.md).

## Final verification

`uv run ruff format --check`: 117 files formatted. `uv run ruff check`: passed.
`uv run pytest`: **165 passed, 1 network test deselected** (12.33 s). Budget tests
also refuse a generator that does not match the plan's verified model prices.
Secret/local-path scans of new data and reports passed, every CSV is below
100 MB, original file hashes match, and source/data SHA-256 manifests are saved.
Inspected final git status/diffs; unrelated release/submodule work is preserved.
