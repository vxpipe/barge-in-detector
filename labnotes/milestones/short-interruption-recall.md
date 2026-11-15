# Improve short-interruption recall

## Runnable outcome and prerequisites

Improve the current checkpoint's 5/30 short-interruption recall on reviewed
English development examples, then establish performance on fresh owner-reviewed
English/Hindi evaluation data. Prerequisite: experimental checkpoint release;
coordinate data changes with the dataset coverage-audit milestone in this source repo.

## Contract and checkpoint tasks

- [ ] Agree acceptance floors for false and missed interruptions before tuning.
- [ ] Compare a character n-gram baseline and encoder on exactly the same splits.
- [ ] Train against the revised dataset distribution; never train on challenge rows.
- [ ] Evaluate short interruptions, longer backchannels, prefix transitions and
  per-language errors. Preserve old checkpoint and use validation for selection.
- [ ] Record immutable dataset/source revisions, parameters and costs.
- [ ] Calibrate on realistic validation priors.
- [ ] Measure ONNX/int8 parity and reference-CPU latency/memory; measure Vxpipe
  serialization/queue/boundary overhead for the whole-path budget.

## Acceptance gates

- [ ] Fresh human-reviewed English/Hindi results meet agreed floors and beat baseline.
- [ ] Model-path p99 <=20 ms; whole path p99 <=50 ms on reference CPU, warmed batch one.
- [ ] Prefix behaviour and Vxpipe overlapping-speech scenarios pass with evidence.

Use `bin/create-labnotes` and link each checkpoint's factual evidence here.
