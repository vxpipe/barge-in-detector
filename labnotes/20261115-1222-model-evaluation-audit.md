# Model evaluation audit

## Motivation and frozen protocol

The synthetic test score was 331/331. This is weak evidence of generalization:
train and test share a synthetic generation process. The user requested an
additional challenge evaluation after clarification that no challenge set
previously existed.

- Authored 98 English cases with expected labels before model inference.
- Frozen dataset: `data/evaluation/english-challenge-v1.csv`, SHA256
  `3d46b9e024f4d3bedd69f07603cde934e85ec2272988e90d2acebc30d5e99519`.
- Cases and labels authored by the coding agent; pending owner review. This
  is diagnostic evidence, not human-verified or representative evaluation.
- Frozen checkpoint: `models/e5-small-run1`, no retraining or threshold tuning;
  existing threshold 0.5, 64-token budget. Evaluate inside tmux `gpubox` on
  training instance. A separate report leaves original synthetic metrics intact.
- Report exact overlap between splits and challenge; score unseen cases
  separately. Repeated prefixes intentionally remain grouped.
- English word-length baseline chooses its cutoff on validation macro F1,
  then reports test and challenge performance, with no challenge tuning.
- Synthetic test backchannels average 3.36 whitespace words, interruptions
  9.56; only 6/222 test interruptions have at most two whitespace words.
  Multilingual whitespace lengths are descriptive, not comparable token counts.
- Reproducible experiment: `uv run --group train python -m experiments.evaluate_english_challenge`.

## Completed evaluation and owner review

Local CPU evaluation reproduced 72/98 correct (73.5%); unseen-in-any-split
subset 67/93 (72.0%). Short interruptions: 5/30 correct; long backchannels 24/24;
ordinary backchannels 10/10; ordinary interruptions 10/10; prefixes 23/24.
Missed 26/48 interruptions and falsely interrupted 0/50 backchannels.
Seven of eight prefix groups had all labels correct. Stop, wait and stop talking
were missed with probabilities approximately 0.0134, 0.0135 and 0.0140.

Exact cleaned-text cross-split overlap is zero. Five challenge rows occur in
train/validation (3/2), none in test. A word-count baseline selected on English
validation reached 93.8% on English synthetic test, but 55.1% on challenge.
This supports a shortcut concern without proving that length alone drives the model.
Batch-vs-single maximum probability difference was 0.00000829 with no label changes.

Owner reviewed all 98 cases in the UI. Identical-text labels and all eight prefix
label sequences were consistent. Thirteen marks were reconsidered with explicit
authorization; subsequent binary-only clarification resolved en-031 to backchannel.
Final 50 backchannel / 48 barge-in labels match the frozen authored expectations.
This is now development/regression evidence, not a fresh independent holdout.

Model evaluation scripts and full reports now live in
HashNuke/barge-in-detector under experiments/ and reports/. Historical labnotes
remain in the dataset repo by owner decision. No retraining or threshold changes
were made during these diagnostics. All three rented instances were destroyed.
