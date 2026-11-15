# Evaluation diagnostics

`english-challenge-v1.csv` contains 98 agent-authored English diagnostic cases,
labelled before evaluating the frozen `models/e5-small-run1` checkpoint. This
is a deliberately adversarial set, not a representative accuracy estimate.
All 98 rows have been owner-reviewed and reconciled with authorization in
`reviews/english-challenge-v1.json`; final binary decisions match the frozen v1
labels. Earlier result metadata saying pending review reflects the first run. It is separate from the Gemini dataset
and must not be added to the training splits.

Assume each utterance is the caller speaking over an ongoing agent turn.
Acknowledgments let the agent continue; corrections, questions and requests
claim the floor. No agent context is supplied. If a reviewer finds a label
requires missing context, record the issue before treating that row as evidence.

Categories: 30 short interruptions, 24 longer backchannels, 10 ordinary
backchannels, 10 ordinary interruptions and 24 prefixes in eight groups.
Each prefix group has two continuer prefixes and a final interruption. Groups
share text intentionally; individual rows are not independent observations.
Common phrases can overlap the synthetic dataset. The report marks these and
scores examples unseen in **all** synthetic splits separately.

Frozen v1 SHA256:
`3d46b9e024f4d3bedd69f07603cde934e85ec2272988e90d2acebc30d5e99519`.

Run from the repo root:

```shell
uv run --no-sync python -m experiments.evaluate_english_challenge --dataset-dir .
```

The source-repo experiment writes `challenge-report.json` and
`challenge-predictions.csv` beside its checkpoint, without changing the checkpoint or original test
reports. It uses the existing 64-token budget and 0.5 threshold. Its length
baseline chooses a cutoff on English validation macro F1 only. This baseline
is a diagnostic for dataset shortcuts, not a proposed runtime classifier.

Review labels before viewing predictions where possible. Save revisions as a
new version; keep v1 intact. Once results inform changes to training, this set
becomes a regression/development set. A fresh human-labelled English/Hindi set
and consented real overlapping speech are needed for independent evaluation.
