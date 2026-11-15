# Lightweight model comparison

Owner authorized further local architecture experiments after E5 download and
GPU destruction, while retaining the E5 fine-tune in the HF model repository.
No additional paid generation or training is needed.

Protocol fixed before execution: all 4,918/915/908 combined multilingual splits,
shared E5 64-token truncation, character 2–5 grams and word 1–2 grams. Compare
balanced logistic regression, balanced linear SVM with three-fold training-only
sigmoid fitting, and Complement Naive Bayes. For each, word scale 0/.25/1 and
C or alpha .1/1/10: 27 fixed configurations. Fit vocabularies on training only;
choose each family and the overall candidate by validation macro F1, preserving
listed order on ties. Save selection before loading test/reviewed cases.

Report the same combined/original/supplement heldouts and 98 reviewed English
development cases, including false/missed stops and prefix/short-request errors.
Measure batch-one model p50/p95/p99 with 100 warmups/1,000 repetitions locally.
These small linear models are learned classifiers, not ignored-word lists.
SVM sigmoid fitting is not evidence of realistic-prior calibration. Fresh human
evaluation and Vxpipe whole-path timing remain acceptance gates.

## Results

Ran after the verified E5 download, GPU destruction and E5 CPU benchmark.
`uv run --no-sync python -m experiments.compare_lightweight` completed all 27
trials, saved validation-only selection, then evaluated and benchmarked in
20.59 seconds. No paid API calls or GPU rental.

| Family | Selected configuration | Validation macro F1 | Test / 908 | Review / 98 | False / 50 | Missed / 48 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Logistic | C10, word scale 0 | .97904 | 894 | 91 | 1 | 6 |
| Linear SVM | C10, word scale .25 | .98896 | 898 | 91 | 0 | 7 |
| Complement NB | alpha .1, word scale 1 | .98455 | 890 | 93 | 2 | 3 |

SVM is the lightweight validation winner. Naive Bayes' higher reviewed score
does not override the prespecified selection rule. All fits are retained under
gitignored `models/lightweight-v2/`; selection, every trial, errors, categories,
per-language results and file hashes are in `reports/lightweight-v2/`.

| Model/scenario | p50 | p95 | p99 |
| --- | ---: | ---: | ---: |
| Logistic, multilingual | .95 ms | 1.16 ms | 1.25 ms |
| Logistic, full 64 tokens | 1.93 ms | 5.02 ms | 5.74 ms |
| SVM, multilingual | 1.42 ms | 1.89 ms | 2.21 ms |
| SVM, full 64 tokens | 2.53 ms | 5.72 ms | 6.83 ms |
| Complement NB, multilingual | 1.66 ms | 2.03 ms | 2.51 ms |
| Complement NB, full 64 tokens | 2.55 ms | 6.19 ms | 7.36 ms |

Owner-designated local CPU, one thread, warmed batch one. All pass the model-path
gate; whole-path p50/p95/p99 remain unmeasured. E5 INT8 is the leading development
candidate overall (94/98, no false stops, 899/908 synthetic test), while SVM has
more CPU timing margin. No production winner is declared from reused English
cases and synthetic multilingual labels. See [full comparison and next evidence](../docs/e5-phrase-results.md).
