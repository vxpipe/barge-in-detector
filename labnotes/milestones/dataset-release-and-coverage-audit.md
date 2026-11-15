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
- [x] Audit coverage by label, language, intent and length; inspect near duplicates.
- [x] Propose short-interruption and long-backchannel generation changes with
  matched-length examples; preserve the reviewed challenge outside training.
- [x] Owner authorized a separate judge-free common-expression supplement across
  all 14 languages, with a pre-run estimate and a $30 total expense ceiling
  (2026-11-15); preserve the original dataset alongside it.
- [x] Implement the separate plan/runner, durable budget reservations and
  supplement splits that preserve original assignments.
- [x] Complete the bounded supplement run and record checklist coverage/gaps:
  3,437 accepted samples, all 14 languages, zero judge calls, $1.4362 usage cost;
  see [results](../../docs/phrase-coverage-results.md).
- [ ] Prepare fresh English/Hindi owner review separate from development data.

## Acceptance gates

- [ ] Published split counts and hashes match source; no exact cross-split duplicates.
- [ ] CLI split/review/generation work without model or torch dependencies.
- [ ] Focused tests, Ruff formatting and lint pass; credentials and local state excluded.
- [ ] Coverage findings and revised generation plan are recorded with evidence.

New model-specific milestones live in the model repository. This dataset milestone
stays unchecked until the coverage work and release gates are complete.

Checkpoint evidence: [repository split labnotes](../20261115-1323-huggingface-repo-split.md).

Ownership note: the earlier repository split is superseded by
github-source-and-hub-releases.md. Development and milestones now live in GitHub;
HF repositories are release-only. Historical checkpoint evidence remains valid.

Coverage checkpoint: [review audit and baseline](../20261115-1529-reviewed-error-audit.md),
with [findings and generation proposal](../../docs/review-followup.md).
Follow-up: [phrase coverage generation](../20261115-1654-phrase-coverage-generation.md)
implements the owner-authorized additive workflow and preserves the original data.
