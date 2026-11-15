# Barge-in detector milestones

This checklist owns implementation order. Check an entry only when the
milestone's acceptance gates pass.

- [ ] [Utterance-first backchannel classifier](utterance-first-backchannel-classifier.md)
  — classify only the caller's overlapping utterance (user decision,
  2026-11-15); supersedes the first milestone's 2-3 turn input contract.
- [ ] [Multilingual text barge-in classifier](multilingual-text-barge-in-classifier.md)
  — contracts, synthetic data, training, CPU latency and Vxpipe hand-off for a
  first model covering 15 languages (36 until 2026-11-15).
