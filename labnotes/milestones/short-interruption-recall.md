# Improve short-interruption recall

## Runnable outcome and prerequisites

Improve the current checkpoint's 5/30 short-interruption recall on reviewed
English development examples, then establish performance on fresh owner-reviewed
English/Hindi evaluation data. Prerequisite: experimental checkpoint release;
coordinate data changes with the dataset coverage-audit milestone in this source repo.

## Contract and checkpoint tasks

- [ ] Agree acceptance floors for false and missed interruptions before tuning.
- [x] Compare a character n-gram baseline and encoder on exactly the same splits.
- [x] Complete the fixed-dataset comparison, including frozen E5 embeddings plus
  a linear classifier, using all 3,304 accepted samples across all 14 languages
  and the shared 2,640 / 333 / 331 train/validation/test split. Assess this
  comparison before changing the dataset (owner clarification, 2026-11-15).
- [x] Run an additive lexical data ablation on the revised multilingual dataset,
  excluding all exact challenge strings from the supplement. Original data is
  preserved, including three pre-existing challenge overlaps in training.
- [x] Fine-tune E5 on the additive all-language dataset with fixed seeds and
  validation-only checkpoint selection; retain all checkpoints and FP32/INT8 exports.
- [x] Compare 27 local character/word classifiers on the same combined splits,
  selecting each architecture using validation only.
- [x] Evaluate short interruptions, longer backchannels, prefix transitions and
  per-language errors. Preserve old checkpoint and use validation for selection.
- [x] Record immutable dataset/source revisions, parameters and costs.
- [ ] Calibrate on realistic validation priors.
- [ ] Measure ONNX/int8 parity and reference-CPU latency/memory; measure Vxpipe
  serialization/queue/boundary overhead for the whole-path budget.

## Acceptance gates

- [ ] Fresh human-reviewed English/Hindi results meet agreed floors and beat baseline.
- [ ] Model-path p99 <=20 ms; whole path p99 <=50 ms on reference CPU, warmed batch one.
- [ ] Prefix behaviour and Vxpipe overlapping-speech scenarios pass with evidence.

Use `bin/create-labnotes` and link each checkpoint's factual evidence here.

Checkpoint evidence: [review audit and baseline](../20261115-1529-reviewed-error-audit.md).
The character baseline reaches 88/98 reviewed cases versus the encoder's 72/98,
with 5/50 false interruptions versus 0/50. This is development evidence only;
the fresh evaluation, calibration and latency gates remain open. See
[the next training checkpoint](../../docs/review-followup.md).

Training and synthetic evaluation use the full multilingual dataset. The
English/Hindi human-review scope does not restrict the training languages.

Further evidence: [frozen E5 experiment and teardown](../20261115-1551-frozen-e5-comparison.md)
and [combined-feature improvement](../20261115-1600-combined-feature-improvement.md).
The portable lexical candidate scores 90/98, catches 26/30 short requests and
passes the model-path CPU latency gate locally. Four false stops, multilingual
synthetic regressions, calibration, fresh human evaluation and the Vxpipe whole
path remain open; the milestone is not complete. See
[model improvement results](../../docs/model-improvement-results.md).

Phrase-coverage evidence: [generation/ablation labnotes](../20261115-1654-phrase-coverage-generation.md)
and [results](../../docs/phrase-coverage-results.md). Adding 3,437 accepted samples
reduces reviewed false interruptions from 4 to 1, but misses rise from 4 to 6 and
prefix correctness drops from 24/24 to 22/24. Overall 91/98 is not a decisive win;
keep both candidates. The next target is matched short-request/prefix contrasts
and validation-only weighting/calibration, followed by fresh owner review.

Further evidence: [E5 fine-tuning and teardown](../20261115-1754-e5-supplement-finetune.md),
[lightweight comparison](../20261115-1818-lightweight-model-comparison.md), and
[matched results](../../docs/e5-phrase-results.md). Selected E5 scores 93/98 in
FP32; INT8 at serving batch one scores 94/98 with zero false stops and four missed
interruptions. INT8 model-path p99 is 5.69 ms typical / 18.85 ms full 64 tokens.
All variants and seed checkpoints are verified in the HF model submodule staging
directory. The GPU instance was destroyed. Exact published source/data pins,
fresh human evaluation, realistic-prior calibration and whole-path latency remain
open; neither the milestone nor production acceptance is complete.

The experimental HF release is published with exact source/dataset pins and
both HF submodules updated. See [publication verification](../20261115-2319-publish-phrase-models.md).
Fresh human evaluation, realistic-prior calibration and whole-path acceptance
remain open despite publication.
