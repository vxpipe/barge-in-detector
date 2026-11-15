# Phrase coverage supplement v1

This dataset sits alongside the unchanged original at `data/export/utterances.csv`.
It contains **3,437 accepted synthetic samples** across all 14 supported languages,
plus 576 rejected and 3 retired rows retained for provenance. No judge was used.

The [plan](../../../generation/plans/phrase-coverage-v1.yaml) specifies common
standalone expressions, repetitions, combinations, longer acknowledgments,
stop/correction contrasts, and short interruptions, including permitted language
mixtures. All generation uses Gemini 3.8 Flash and native scripts. Accepted means
local structure/script/label/deduplication checks passed; it does not mean
native-speaker verification. Three bad pilot rows were retired without changing
or deleting their generated text.

- [Expression checklist](../../../reports/phrase-coverage-v1/expression-checklist.csv):
  1,149 candidate expression entries by language, region and language mix,
  with observed and missing forms. These are not 1,149 universally valid words.
- [Criterion checklist](../../../reports/phrase-coverage-v1/criterion-checklist.csv):
  1,087 of 1,203 criteria filled; 1,191 have at least one accepted sample.
  Duplicates, reserved review strings and failed checks explain the remaining gaps.
- [Coverage/cost audit](../../../reports/phrase-coverage-v1/coverage.json).
- [Generation and budget decision](../../../docs/phrase-coverage.md).
- [Data ablation results](../../../docs/phrase-coverage-results.md).

`utterances.csv` is exported through the storage module, never edited by hand.
The original schema-11 columns are unchanged. The supplement's `raw_sample`
envelope holds the exact generated object in `sample`, all criterion factors in
`coverage`, and `verification: local_checks_only_no_judge`. Judge columns are null.
The cost/resume ledger is local at `data/work/phrase-coverage-v1.duckdb`; retain it.

`splits/` contains 2,278 training, 582 validation and 577 test rows. No accepted
supplement text exactly matches an accepted original sample or any challenge row.
There are no exact cross-split duplicates. Semantic near duplicates can still
cross splits, so synthetic accuracy is not evidence of unseen real-call accuracy.
Original-plus-supplement experiment splits live separately under
`data/experiments/original-plus-phrase-v1/splits/`. Original data/splits/checkpoints
remain preserved. Nothing has been published to Hugging Face by this workflow.
