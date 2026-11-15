# Vast.ai model training

## Setup

- User requires all remote work and training inside tmux session `gpubox`.
- Single GPU, datacenter, below $0.65/hour, PyTorch image, good internet.
- Inspected `~/projects/bert-disfluency`, including README, instructions,
  scripts and commit messages: no recorded Vast.ai GPU or template.
- Initial A4000 instance stopped after GPU preference clarification.
- RTX 4090 instance could not allocate resources; kept stopped.
- RTX 5090 instance: one GPU, 32 GB VRAM, verified datacenter, $0.507222/hour including 30 GB disk. Image `vastai/pytorch:@vastai-automatic-tag`
  resolved to `vastai/pytorch:cuda-13.0.3-auto`.
- GitHub HEAD matches local HEAD `5ce0ed1283401f8c73bafcf5a7dbd2101c4bbee6`.
  Local uncommitted training code and splits require a source overlay after
  cloning; snapshot manifest records SHA256 for each tracked/unignored file.
- Dataset audit: 3,304 accepted, 2,220 barge-in / 1,084 backchannel;
  train 2,640 / validation 333 / test 331. Split IDs exactly match accepted
  export IDs; no duplicate cleaned text among accepted samples.
- Intended run: multilingual-e5-small, default 5 epochs, 64 tokens; select
  epoch by validation macro F1, then evaluate saved model on validation/test
  at threshold 0.5. These are synthetic results, not human-verified results
  or calibration evidence. Reference CPU and whole-path latency gates remain.

## Execution

- SSH: use instance via `vastai ssh-url` or direct SSH, then
  `tmux attach -t gpubox`.
- Source cloned in `/workspace/barge-in-detector`; local overlay and
  `barge-source-manifest.json` transferred without GitHub credentials.
- Setup runs in `gpubox` via `/workspace/barge-gpubox-setup.sh`;
  persistent setup log `/workspace/barge-setup.log`.
- Python 3.14.8 downloaded by uv; frozen train dependency installation started.
- Local completion checks: ruff format/check passed; pytest 111 passed,
  1 integration test deselected (8.68 seconds).

## Final outcome

Five epochs completed; epoch 5 selected by validation macro F1 0.99324.
Synthetic validation 331/333 correct; test 331/331 at threshold 0.5. Later
English challenge evaluation exposed 25/30 missed short interruptions.
Artifacts and tokenizer were copied locally and checked against SHA256 manifest.
All three Vast.ai reservations were destroyed at
the owner request. No instances remain. Training/evaluation/benchmark code and
selected artifacts now belong in the separate Hugging Face model repo; original
labnotes stay in the dataset repo. Model acceptance gates remain unfulfilled.
