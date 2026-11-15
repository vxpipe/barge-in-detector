# Model repository split

Owner moved model artifacts, training, inference, evaluation and benchmarks into
this standalone Hugging Face model repo. Dataset generation, splitting, review
and all earlier labnotes remain in HashNuke/barge-in-detector-data. This repo has
its own scoped AGENTS.md, bin/create-labnotes and milestone index.

- New package barge_in_detector_model isolates training/metrics/data-loading
  code from the dataset generator. Model code downloads a resolved dataset
  commit, or uses explicit local splits; it does not assign splits.
- Four unit tests cover metrics, local CSVs, missing split detection and pinned
  Hub downloads. Tiny CPU smoke trained one epoch, saved, reloaded and evaluated.
- Actual unchanged checkpoint predicts stop as backchannel, p=0.0134437,
  reproducing the known failure. No GPU rental or retraining of release weights.
- Token budget remains 64; two labels; predictions are not calibrated.
- Root weights and tokenizer match the original artifact hashes. Original source
  manifest and sanitized training metrics retain run provenance. Model card
  reports synthetic and reviewed diagnostics and local CPU timing limitations.
- AGENTS.md derives CPU serving gates, fourteen languages, single RTX 5090,
  under-$0.65/hour datacenter/PyTorch policy, gpubox, tests, and labnotes/milestone
  usage from the dataset instructions. Existing history is linked, not duplicated.

Historical run and latency evidence is in the dataset repo labnotes. Follow
labnotes/milestones/index.md for new model work. Publication verification follows.

## Publication verified

- Initial weights/tokenizer commit dcbadfd7572e08ef3ccd19e35576e85faffe00f8;
  standalone code commit 9aa77e5ff33f479ec5559f3f65bcfc092e5f86ab.
- release.json pins dataset dda7115ace1fe7b07c49139860131f39910521ad.
- Downloaded that dataset through the model loader: 2,640/333/331 rows and
  exact file hashes match. Standard datasets.load_dataset also loads those
  splits and the separate 98-example challenge configuration correctly.
- Anonymous Hub model download loaded on local CPU and reproduced stop
  probability 0.0134437. Remote large-file checksums match local model/tokenizer.
- Four model unit tests, Ruff format and lint passed. Tiny local train/save/
  reload/evaluate smoke passed before original scripts were removed.
- bin/create-labnotes is executable in the Git tree. Each repo has its own
  AGENTS.md and milestone index. New model work stays here; history stays in data.
- No model or tokenizer weights were committed to the original GitHub repo.
  No GPU was rented or restarted during this publication work.
