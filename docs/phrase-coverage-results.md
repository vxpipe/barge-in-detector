# Phrase coverage: data and controlled ablation

2026-11-15. The original dataset is preserved. The new supplement contains 3,437
accepted samples across all 14 languages, generated with Gemini 3.8 Flash and
**zero judge calls**. It adds common expressions, repetitions, combinations,
longer acknowledgments, contrasting interruptions and short direct requests.
The label quality remains synthetic and unverified.

## Coverage and cost

The [criterion checklist](../reports/phrase-coverage-v1/criterion-checklist.csv)
has 1,203 criteria: 1,191 represented and 1,087 filled to three accepted examples.
116 remain short after the attempt allowance. The
[expression checklist](../reports/phrase-coverage-v1/expression-checklist.csv)
contains 1,149 candidate entries by language/region/mix; 977 still lack one or
more forms. It makes gaps visible and does not establish exhaustive coverage.
Some missing training cores, such as English “okay,” are intentionally reserved
in the existing human review set. They are recorded in the inventory, not added
to training. Hindi core additions include हाँ, जी, ठीक है and बिल्कुल.

| Language | New backchannels | New barge-ins | New accepted total |
| --- | ---: | ---: | ---: |
| English (US, India, Singapore) | 373 | 403 | 776 |
| Hindi | 105 | 90 | 195 |
| Kannada | 96 | 90 | 186 |
| Tamil | 100 | 88 | 188 |
| Telugu | 100 | 90 | 190 |
| Malayalam | 98 | 90 | 188 |
| Mandarin | 107 | 90 | 197 |
| Japanese | 100 | 89 | 189 |
| Indonesian | 103 | 90 | 193 |
| Filipino | 101 | 90 | 191 |
| French | 99 | 90 | 189 |
| German | 100 | 90 | 190 |
| Spanish (Spain, Mexico) | 191 | 179 | 370 |
| Italian | 105 | 90 | 195 |

Generation made 1,567 submissions, including 18 initial credential failures.
Usage-based cost, conservatively charging cached tokens at full input price, was
**$1.43615175**. Including maximum-cost reservations for those failed requests:
**$1.68498375**. Historical recorded generation plus new usage is **$14.60211915**;
including the historical $2 uncertainty reserve and new reservations gives
**$16.85095115**, below the $30 ceiling. The pre-run estimate was $3.53 first pass,
revised to $3.65 for prompt v2; the enforced new-run cap remained $8.

The [audit](../reports/phrase-coverage-v1/coverage.json) verifies original file
hashes, zero accepted supplement overlap with original/evaluation texts, zero
judge calls and all costs. The final no-op rerun created no API client, made no
API calls, and preserved both the CSV hash and modification time. Rejected and
retired rows remain in the supplement: 183 original duplicates, 355 supplement
duplicates, 31 reserved evaluation strings, seven script failures and three
retired pilot script violations. No conflicting labels were found among exact
original duplicates.

## Same classifier, changed training data

This experiment compares the existing **character/word TF-IDF linear classifier**
with the same architecture trained on the additive dataset. It does not retrain
E5 or MiniLM and does not rent a GPU. Word scale 0.25, logistic-regression C=10,
balanced class weights and threshold 0.5 were fixed before this experiment.
There was no selection on test results or reviewed cases. Tokenization keeps the
same 64-token budget; no training strings required truncation.

All 6,741 accepted examples are allocated to train/validation/test. Original
assignments remain 2,640 / 333 / 331; supplement assignments are 2,278 / 582 / 577.
The additive classifier trains on **4,918 rows**, with separate validation and
test portions. Feature construction plus classifier fitting took **0.568 seconds
on the local CPU**; the linear-classifier fit itself took 0.141 seconds.

| Evaluation set, unchanged between models | Original-data model | Additive-data model |
| --- | ---: | ---: |
| Original synthetic validation | 330/333 (99.1%) | 327/333 (98.2%) |
| Original synthetic test | 320/331 (96.7%) | 323/331 (97.6%) |
| Supplement synthetic test | 529/577 (91.7%) | 570/577 (98.8%) |
| Owner-reviewed English development set | 90/98 (91.8%) | 91/98 (92.9%) |
| False interruptions on its 50 backchannels | 4 | 1 |
| Missed interruptions on its 48 barge-ins | 4 | 6 |
| Short interruptions recognized | 26/30 | 26/30 |
| Prefix cases correct | 24/24 | 22/24 |

The supplement substantially helps acknowledgment recognition, including all
previously missed long backchannels. It also catches “go back.” However, it loses
“cancel,” “before you continue i have a question,” and “got it okay now let me
explain”; “sure sure” becomes a false interruption just above the 0.5 threshold.
Original validation also regresses. **Do not promote this model as the winner.**
Preserve both checkpoints; the next targeted work is short requests and the
transition from acknowledgment to claiming the floor, using matched contrasting
phrases, preserving coverage of all existing barge-in intents, and
validation-selected weighting/calibration. Do not adjust thresholds
to rescue these reviewed examples and call that independent improvement.

Detailed results, per-language errors and source hashes are in
[the ablation report](../reports/lexical-phrase-v1/report.json).
The 98 English examples have been reused for development; three challenge texts
already occur in original training and one in original validation. No supplement
sample adds exact challenge overlap. Fresh human-reviewed data, realistic prior
calibration and real-call evaluation remain necessary.

## CPU serving verification

The additive model is saved separately under `models/lexical-phrase-v1/`; the
original stays under `models/lexical-run1/`. Portable JSON/tokenizer export has
exact probability parity on original validation, test and reviewed cases.
Artifact size is 24,243,695 bytes. The owner-designated local reference CPU is
AMD EPYC-Genoa, one thread, warmed batch size one, 100 warmups/1,000 measurements.

| Model path (tokenization through probability) | p50 ms | p95 ms | p99 ms |
| --- | ---: | ---: | ---: |
| Original multilingual test inputs | 0.752 | 0.883 | 0.952 |
| Full 64-token input | 1.899 | 4.168 | 4.777 |

This passes the 20 ms model-path gate. Whole-path p50/p95/p99 is **unmeasured**:
the Vxpipe integration boundary is not implemented, so the 50 ms whole-path gate
is still open. Verification-process peak RSS was 494.7 MiB, including both loaded
training/export objects; this is not isolated serving RSS. See the
[benchmark](../reports/lexical-phrase-v1/cpu-benchmark.json).
