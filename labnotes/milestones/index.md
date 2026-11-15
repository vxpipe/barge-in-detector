# Barge-in detector milestones

This checklist owns implementation order. Check an entry only when the
milestone's acceptance gates pass.

- [x] [GitHub source and Hugging Face releases](github-source-and-hub-releases.md)
  — current ownership and publication boundary.

- [x] [Direct Hugging Face release workflow](huggingface-release-workflow.md)
  — validated dataset/model publishing and portable HF release submodules.

- [ ] [Dataset release and coverage audit](dataset-release-and-coverage-audit.md)
  — active dataset work after separating model ownership.

- [ ] [Improve short-interruption recall](short-interruption-recall.md)
  — model work coordinated with the dataset coverage audit.

Historical cross-project milestones follow; all new model and dataset work
uses this GitHub repository and this index.

- [ ] [Utterance-first backchannel classifier](utterance-first-backchannel-classifier.md)
  — classify only the caller's overlapping utterance (user decision,
  2026-11-15); supersedes the first milestone's 2-3 turn input contract.
- [ ] [Multilingual text barge-in classifier](multilingual-text-barge-in-classifier.md)
  — contracts, synthetic data, training, CPU latency and Vxpipe hand-off for a
  first model covering 14 languages (36 until 2026-11-15).
