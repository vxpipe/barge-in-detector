# Dataset release and coverage audit

## Runnable outcome and prerequisites

Versioned Hugging Face data with generator, split/review tools and historical
labnotes, separated from the model/training repo. Then a measured plan to address
short-interruption undercoverage. Sources: README.md, AGENTS.md, docs/data-format.md,
reviewed challenge and historical CPU/evaluation labnotes.

## Checkpoints

- [x] Retain dataset generation/cleaning, move model scripts to the model repo.
- [x] Keep all existing labnotes and milestones in this repository.
- [x] Dataset card with split counts, source export, leakage and review limitations.
- [ ] Publish and verify HashNuke/barge-in-detector-data.
- [ ] Audit coverage by label, language, intent and length; inspect near duplicates.
- [ ] Propose short-interruption and long-backchannel generation changes with
  matched-length examples; preserve the reviewed challenge outside training.
- [ ] Obtain approval for plan/count changes before paid generation runs.
- [ ] Prepare fresh English/Hindi owner review separate from development data.

## Acceptance gates

- [ ] Published split counts and hashes match source; no exact cross-split duplicates.
- [ ] CLI split/review/generation work without model or torch dependencies.
- [ ] Focused tests, Ruff formatting and lint pass; credentials and local state excluded.
- [ ] Coverage findings and revised generation plan are recorded with evidence.

New model-specific milestones live in the model repository. This dataset milestone
stays unchecked until the coverage work and release gates are complete.

Checkpoint evidence: [repository split labnotes](../20261115-1323-huggingface-repo-split.md).
