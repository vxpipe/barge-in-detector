# Model improvement results

The strongest lightweight candidate so far combines learned character fragments
with word unigrams/bigrams. On the same 98 owner-reviewed development cases it
improves 72/98 for the original fine-tuned E5 to **90/98**, while meeting the
model-path CPU latency budget on the owner's local machine. It is an experimental
candidate, not a released replacement: false interruptions and multilingual
generalization still need work, and the whole-path gate remains unmeasured.

## Fixed dataset and results

Every approach uses all 14 languages in exactly the same 2,640 train / 333
validation / 331 test split. Feature fitting and classifier weights use train
only; configuration selection uses validation macro F1. No new data was generated,
no review annotations were changed, and the challenge was not used for fitting
or numerical hyperparameter selection. Architecture exploration informed by its
errors makes it development evidence, not an independent holdout.

| Approach | Synthetic test | Reviewed English | Short requests caught | False stops / 50 | Prefix rows correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fine-tuned E5 | 331/331 | 72/98 | 5/30 | 0 | 23/24 |
| Character fragments | 320/331 | 88/98 | 26/30 | 5 | 23/24 |
| Frozen E5 + linear head | 327/331 | 72/98 | 7/30 | 2 | 23/24 |
| Characters + frozen E5 | 326/331 | 90/98 | 25/30 | 3 | 24/24 |
| **Characters + word phrases** | **320/331** | **90/98** | **26/30** | **4** | **24/24** |

All table entries use the diagnostic threshold 0.5; probabilities are not
calibrated. Validation-only macro-F1 threshold selection retains 0.5 for both
combined-feature candidates, so it offers no further gain here. It selects
0.5173 for frozen E5, with unchanged challenge decisions. These are not agreed
runtime thresholds or representative-prior calibration results.

The lexical candidate's two corrections versus the original character model are
“that is interesting yes i see” (backchannel) and “yeah yeah no that will not work
for me” (barge-in). It introduces no new mistakes on this reviewed set. It still
misses **enough**, **go back**, **you misunderstood** and **how much**, and makes
four false stops on longer acknowledgments. Its score on the 93 examples unseen
in any original split is 85/93 (91.4%); five historical challenge overlaps remain
explicitly accounted for.

Per-language synthetic results also matter. The lexical candidate is 16/18 on
Hindi, 16/18 on Italian and 33/36 on Spanish, versus the original encoder's
perfect synthetic test. Thus the reviewed English gain is not evidence of an
improvement in every language. Full counts and metrics are recorded in
[`reports/model-comparison.json`](../reports/model-comparison.json).

## What the experiments establish

- Freezing the original pretrained E5 does not resolve short requests. The issue
  cannot be attributed solely to damaging its representations during fine-tuning.
  This experiment also changes pooling/head setup, so it is not a causal isolation
  of one factor.
- Combining local character clues with longer-range representations improves
  the development result. A frozen-E5 hybrid reaches 90/98, but misses “stop”
  at the selected threshold and retains the transformer inference cost.
- Word features provide a cheaper complementary signal: the lexical candidate
  reaches the same total with more short requests caught. It uses learned weights,
  not a universal ignored-word list. It has limited contextual expressiveness;
  paired sentences and unseen phrasing remain essential evaluation cases.
- The largest remaining data concern is unchanged: no English/Hindi 1–2-word
  interruptions in training, and no short interruptions in their synthetic
  validation/test slices. Repeatedly selecting against those holdouts cannot
  reliably reward improvements on the missing cases.

## Downloaded and local artifacts

- `models/frozen-e5-run1/`: downloaded original frozen encoder/tokenizer, linear
  head, reports and source manifest. All ten artifact hashes and archive hash
  verified before destroying Vast.ai the training instance. No instances remain.
- `models/hybrid-run1/`: local diagnostic hybrid weights and embedding cache.
- `models/lexical-run1/portable/`: runnable JSON classifier plus tokenizer. This
  inference artifact loads no pickle, enforces the same first-64-token budget
  using the E5 tokenizer's offsets, and needs no transformer forward pass.

The lexical vocabulary, IDF values and coefficients are learned only on training
data. Validation selects C=10 and word-feature scale=0.25 from a fixed nine-setting
grid (C=1/10/100, scale=0.25/1/4). Export/reload reproduces every validation, test
and challenge probability exactly, including the new token-budget enforcement.

```shell
uv run --group baseline python -m barge_in_detector.training.lexical "stop"
```

This returns an uncalibrated `p_barge_in` of approximately 0.5326. The score's
proximity to 0.5 is another reason to retain “stop” as an explicit regression case.
Do not infer that the confidence has become reliable from the correct label alone.

## CPU measurements

Owner-designated local AMD EPYC-Genoa virtual CPU; batch one, warmed; 100 warmups,
1,000 measured iterations per scenario. Model path includes tokenization through
output probability. Frozen E5 uses four threads; lexical uses one.

| Model / scenario | p50 | p95 | p99 |
| --- | ---: | ---: | ---: |
| Frozen E5 / multilingual test | 14.53 ms | 19.32 ms | 25.45 ms |
| Frozen E5 / full 64 tokens | 27.66 ms | 35.71 ms | 53.19 ms |
| Lexical / multilingual test | 0.90 ms | 1.27 ms | **1.42 ms** |
| Lexical / full 64 tokens | 2.16 ms | 4.43 ms | **5.35 ms** |

The lexical model passes the <=20 ms model-path p99 gate in these measurements.
Whole-path p50/p95/p99 are **not measured**; there is no Vxpipe integration yet.
Artifact size is 22.9 MB including tokenizer; verification-process peak RSS is
470 MiB (includes export and comparison objects, not an isolated serving RSS).
Frozen E5 CPU/GPU parity differs by at most 6.63e-7 with zero label changes.

Evidence: [lexical CPU report](../reports/lexical-run1/cpu-benchmark.json) and
[frozen-E5 CPU report](../reports/frozen-e5-run1/cpu-benchmark.json).

## Next improvement checkpoint

Use the lexical model as the lightweight development candidate and retain E5 as
the multilingual comparator. Do not replace the published model based on this
English development set. Before another large model sweep:

1. Add a fresh evaluation batch that covers short floor requests, longer
   backchannels, contextual reversals and prefixes, with all 14 languages in the
   synthetic evaluation and owner verification where available. Freeze evaluation
   texts/families outside training before scoring. Keep the old 98 as regressions.
2. Implement the deferred deterministic length/coverage criteria and stable
   growing splits in [review follow-up](review-followup.md), across the full
   multilingual dataset. Preserve equal counts per criterion and all old rows.
   Review the exact plan/counts and spending cap before paid generation.
3. Retrain the same candidate/comparator on the expanded data, measuring per-language
   missed and false interruptions separately. Select calibration/thresholds using
   separate validation that reflects runtime priors; agree acceptable error rates.
4. Add the Vxpipe boundary measurement before considering a release.

The completed change improves development accuracy and CPU latency. Remaining
work concerns coverage and independent verification, rather than another identical
E5 training run on the same missing short-request examples.
