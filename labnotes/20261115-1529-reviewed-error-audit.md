# Reviewed errors and next training checkpoint

## Request and scope

The owner completed the 98-case UI review and asked how to improve the roughly
75% result, then asked to proceed. Completed the offline coverage/error audit and
a learned character baseline comparison before further paid generation/training.
Existing unrelated release/submodule edits were present and preserved. No commits,
publishing, annotation edits, generated-data changes or GPU rentals were made.

## Findings

- Reconciled all 98 saved binary decisions by source hash/case ID/text against
  frozen checkpoint probabilities. They match authored labels. Encoder remains
  72/98 correct, misses 26/48 interruptions, including 25/30 short requests;
  false interruptions 0/50. Five challenge rows overlap train/validation; unseen
  score 67/93. The challenge remains development/regression evidence.
- Train has zero English/Hindi interruptions of 1–2 whitespace words. English
  interruptions: 0 / 8 / 424 in bands 1–2 / 3–5 / 6+; Hindi: 0 / 2 / 94.
  All English/Hindi validation/test interruptions are 6+ words. Coverage by
  language, intent and label is in `reports/reviewed-error-audit-v1.json`.
- All 3,304 accepted rows match splits exactly; no within/cross-split exact
  duplicates. Character-trigram Jaccard >=0.8 finds four validation and one test
  near-matching rows against train, all acknowledgments/continuers among the
  displayed examples. This does not exclude semantic/template similarity.
- Descriptive encoder threshold sweep: 0.0133 catches 29/30 short requests but
  interrupts 25/50 backchannels. No threshold selected or changed.

## Baseline experiment

`uv run --group baseline python -m experiments.compare_character_baseline` fits
character 2–5-gram TF-IDF plus balanced logistic regression on the same 2,640
training rows. Original encoder source-manifest hashes guard all three splits.
Only validation macro F1 chooses C from 0.1/1/10; 10 wins. Threshold fixed at 0.5;
one thread, seed 20261115; convergence warnings fail the run. No challenge fitting.

- Baseline reviewed score: 88/98 (89.8%); unseen 83/93 (89.2%).
- Short interruptions 26/30; all interruptions 43/48; false interruptions 5/50.
- Long backchannels 19/24; ordinary categories 20/20; prefix rows 23/24.
- Synthetic test 320/331 (96.7%), versus encoder 331/331.
- Baseline false interruptions are longer acknowledgments; remaining short
  misses include enough, go back, you misunderstood and how much.
- Configuration, validation trials, runtime versions, hashes and source base
  commit/dirty flag are recorded under `reports/character-baseline-v1/`.
- Added optional `baseline` dependency group (scikit-learn 1.9.1 in uv.lock),
  retaining existing release dependency edits. No serving artifact was exported.

The evidence supports improving data coverage while retaining both methods in
the comparison. It does not establish general accuracy or calibrated performance.
Full cleaned strings are used for this diagnostic baseline; production token
budget behavior, reference CPU and Vxpipe whole-path latency remain unmeasured.
Existing benchmark evidence identifies its CPU as local/unspecified reference,
so no acceptance latency table can be supplied for a serving candidate here.
Cost: no Gemini calls or paid compute; local CPU only.

## Next steps and barriers

`docs/review-followup.md` records the length-factor proposal, matched-length
positive/negative coverage, independent English/Hindi review, calibration and
serving gates. Existing equal-count policy remains intact; executable criteria
counts and paid budget must be reviewed before generation. Default factor IDs
must remain stable, new schema columns versioned, old data preserved.

Code inspection also found that the current splitter chooses its held-out row
by minimum ID hash and validation/test side by sorted criterion position. Dataset
growth can therefore move existing samples. Fix and test append stability before
adding criteria; this is a required dependency of the next generation run.

## Verification

- Red: `uv run pytest tests/test_reviewed_audit.py -q` failed because the new
  experiment module did not yet exist.
- Green: 12 focused tests passed, covering owner-label precedence, stale data,
  invalid/missing binary decisions, invalid probabilities, duplicate prediction
  IDs, length grouping and near-duplicate exclusions.
- Audit and baseline ran successfully with saved reports. Baseline hyperparameters
  were fixed before observing its challenge output; no post-result tuning.
- Final checks: `uv run ruff format --check` (91 files), `uv run ruff check`
  and `uv run pytest` passed: **143 passed, 1 integration test deselected** in
  9.46 seconds. `git diff --check` passed. Reviewed the final status/diffs;
  original release/submodule work remains present and nothing was staged.

## Owner clarification: fixed multilingual comparison

The owner requested using the whole dataset with shared train/test/validation
splits so architectures have comparable evidence, rather than narrowing to
particular languages. Verified that both completed runs already use all 14
languages in each split: 2,640 train, 333 validation, 331 test. The 98-case human
challenge is English-only; that is an evaluation limitation, not a training
filter. Updated the durable follow-up plan and model milestone to assess the
fixed-dataset comparison before generating more data. Frozen E5 embeddings plus
a linear classifier is the proposed remaining comparison, not a completed run.
No dataset, split assignment or model artifact changed for this clarification.
