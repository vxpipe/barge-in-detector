# Frozen E5 comparison

## Protocol and authorization

- Owner authorized the third experiment on one Vast.ai RTX 5090 below $0.65/hour,
  followed by model download and shutdown. No dataset changes: all 14 languages,
  existing 2,640 train / 333 validation / 331 test, plus the 98-case reviewed
  English development set scored separately.
- Owner designated this local machine for CPU latency measurement during the run.
  Vxpipe integration/queueing is still unavailable; whole-path latency remains open.
- Use the original pretrained `intfloat/multilingual-e5-small`, revision
  `614241f622f53c4eeff9890bdc4f31cfecc418b3`, frozen in evaluation mode. No encoder
  gradients. Use documented `query: ` prefix, masked mean pooling and L2
  normalization; FP32, batch 128, 64-token budget including prefix/special tokens.
- Fit balanced logistic regression on training embeddings only; select C from
  0.1/1/10 using validation macro F1, ties favoring smaller C. Threshold fixed at
  0.5; uncalibrated. Load the challenge after selection. This protocol was fixed
  before observing the third model's results.
- Export the full encoder/tokenizer plus a plain JSON linear head, source
  snapshot manifest, reports and checksums under gitignored models/. Version
  reports and code here; no HF publication or source commit requested.

## Instance and setup

- No pre-existing Vast.ai instances. Rented one training instance:
  one RTX 5090, 32 GB VRAM, verified datacenter, 30 GB storage,
  **$0.507222/hour** including disk. Network transfer is separately metered at $0.005208/GB each direction.
- Image `vastai/pytorch:@vastai-automatic-tag` resolved to
  `vastai/pytorch:cuda-13.0.3-auto`. All setup/work runs in tmux `gpubox`.
- The proxy SSH endpoint initially refused connections; direct mapped SSH worked,
  using the owner-specified key. No key material was copied to the instance.
- Prepared and tested code before renting. Uploaded an explicit 79-file source
  snapshot with per-file hashes and Git base commit; no credentials, models,
  virtualenvs or sibling HF checkouts. Remote `uv sync --frozen --group train
  --group baseline` installed the locked dependencies.
- Vast's raw creation/status responses unexpectedly included instance-scoped
  credentials in tool output. Removed them from saved local metadata and filtered
  subsequent API output to explicit status fields; no secrets enter versioned
  evidence. Instance teardown ends this instance's lifecycle.
- GPU reports NVIDIA GeForce RTX 5090, 32607 MiB, driver 580.178.04.

## Verification so far

- Red: focused tests failed importing the not-yet-created frozen-E5 module.
- Green: five focused tests pass locally and on the GPU host: masked pooling,
  normalization, query prefix/64-token budget, disabled gradients, portable head
  round-trip, label orientation and invalid feature rejection.
- Local formatting/lint pass; full suite **148 passed, 1 integration deselected**
  in 10.04 seconds, with the CPU torch environment (`uv run --no-sync`).

## Completed run and teardown

- Selected C=10. Synthetic test **327/331 (98.8%)**; reviewed challenge
  **72/98 (73.5%)**, unseen subset 67/93. Short interruptions **7/30**, all
  interruptions 24/48; false interruptions **2/50**. Long backchannels 22/24,
  ordinary categories 20/20, prefixes 23/24. Freezing E5 did not solve the issue.
- Timings: download/load 10.534 s; train/validation embedding extraction 0.563 s;
  three linear-head fits plus selection 0.029 s; run through export 11.787 s.
- Downloaded `models/frozen-e5-run1/` with encoder, tokenizer and head. Verified
  all 10 artifact checksums and the downloaded archive SHA256
  `92bb4b61eaa3a3da0c0b57bc637a7bd6f2602d8d327fe8c7717164e95700f227`.
- Destroyed the training instance after verification; confirmed the Vast.ai
  instance list is empty. First destroy invocation hit the CLI confirmation and
  aborted; repeated with `--yes` under the existing owner shutdown authorization.
  Sanitized lifecycle evidence is in `reports/frozen-e5-run1/instance-lifecycle.json`.
  Runtime was approximately 4–5 minutes; compute/storage about $0.04, with
  separately metered traffic. API traffic counters have unverified units, so
  they are recorded without pretending to know the final invoice.

## Local CPU verification

The owner designated this local machine (AMD EPYC-Genoa virtual CPU). PyTorch
2.15.0+cpu, four intra-op / one inter-op threads, batch one, 100 warmups and 1,000
measured iterations per scenario. CPU/GPU maximum probability difference
0.0000006633 with zero label differences across all 98 challenge cases.

| Scenario | Model p50 | Model p95 | Model p99 | Whole path |
| --- | ---: | ---: | ---: | --- |
| Multilingual test | 14.53 ms | 19.32 ms | 25.45 ms | Unmeasured |
| Full 64-token input | 27.66 ms | 35.71 ms | 53.19 ms | Unmeasured |

Model p99 fails the 20 ms gate in this FP32 implementation; no acceptance claim.
Peak process RSS 1,389.95 MiB. Whole-path p50/p95/p99 remain unmeasured because
Vxpipe integration is not present. Evidence: `reports/frozen-e5-run1/cpu-benchmark.json`.
The benchmark includes tokenization, prefix, pooling and linear probability.

## Continuing toward improvement

Owner clarified that experiments are a means to improving the model, not the
endpoint. Continued with a local combined-feature candidate using the same
dataset; see `20261115-1600-combined-feature-improvement.md`. No more GPU rental.
