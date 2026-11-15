# E5 supplement fine-tuning

Owner authorized Vast.ai fine-tuning, optional GPU analysis, download, then
instance destruction. Constraints persist: one verified datacenter RTX 5090,
32 GB, below $0.65/hour including disk, Vast.ai PyTorch image, remote work in
tmux `gpubox`; local CPU is the inference reference. No new data generation.

Read milestones/training/evaluation code and prior results. Fixed protocol in
[E5 supplement fine-tune](../docs/e5-supplement-finetune.md): same architecture,
pretrained revision and optimizer settings, all 14 languages in fixed
4,918/915/908 splits, three predetermined seeds, validation-only selection before
held-out analysis. Estimate under $1 rental/transfers; retain earlier $30 ceiling.

Preparation: no pre-existing Vast instances. Unqualified `uvx vastai` selected
Python 3.15 and failed building Pillow (missing jpeg headers). Using
`uvx --python 3.14 vastai` works without installing system libraries. Added an
optional export group via `uv add --group export onnx onnxruntime --no-sync`.
ONNX serving tests failed first for missing module, then passed for CPU-only
provider selection, finite/stable binary probabilities, masks and 64-token limit.

The owner subsequently freed local disk space and requested all checkpoints
and FP32/INT8 ONNX exports in the HF model submodule candidate directory.
Preserve the existing release and original datasets/models. Prepare source
snapshot/protocol before rental to minimize idle time.

Rented one training instance, one RTX 5090 32 GB via the verified-datacenter
filter, **$0.509537037/hour including 35 GB disk**. Transfer $0.005208333/GB each direction. Image
`vastai/pytorch:cuda-13.0.3-auto`. Direct SSH uses the owner's specified key;
raw lifecycle responses stay out of output and only allowlisted fields are saved.
Uploaded explicit source/data snapshot, verified every SHA-256 inside gpubox,
then installed frozen dependencies on Python 3.14. Local checks: Ruff clean,
167 tests passed / one network test deselected. Training/export logs stay on disk.

## Completion and export correction

All three fixed seeds completed; training durations 22.45 / 18.27 / 18.31 seconds,
with all training/selection/GPU analyses taking 70.42 seconds. Validation selected
seed 20261117, epoch 4 (macro F1 .99559). Combined test 899/908 and reviewed English
93/98 in FP32: zero false stops, five missed interruptions, short requests 26/30,
prefix cases 23/24. Same expanded test on the unchanged original E5 is 841/908;
its reviewed score remains 72/98. The baseline was not changed.

The first quantization attempt failed because fused ORT BERT operators omitted
intermediate type metadata. A tiny BERT end-to-end export test reproduced the
same error. Set DefaultTensorType to FLOAT for this wholly FP32 graph; the test
then passed for FP32/INT8, masks, mixed lengths and 64-token truncation. Re-export
and parity verification succeeded without retraining. Record the exact patched
source hashes in `reports/e5-phrase-run1/export-source-update.json` alongside the
original training snapshot. The legacy exporter emits deprecation/tracing warnings;
dynamic batch/sequence tests and full saved-prediction comparisons passed.

## Download, privacy and teardown

Downloaded all three seed checkpoints, the selected checkpoint and FP32/INT8 ONNX
to `huggingface/model/candidates/e5-phrase-run1/` at the owner's request. The old
HF root release is preserved. Research metrics/predictions remain under reports;
the HF candidate contains models, tokenizer/configuration, cards and provenance.
Added ONNX/nested-tokenizer Git LFS patterns. No commits or publication were made.
Source/data release revisions must be pinned after publishing that work; staging
metadata truthfully records the uncommitted snapshot and unset new dataset pin.

Verified SHA-256 for all 55 downloaded files (20 model files) and loaded PyTorch,
FP32 ONNX and INT8 ONNX locally. Selected weights match the validation-selected
seed. Then destroyed the training instance and confirmed no instances remained.
Compute/storage cost approximately $0.158, with metered transfers additional;
under the pre-run estimate. Operational teardown evidence and identifiers are
kept only in gitignored local records. Public lifecycle evidence contains costs,
verification counts and teardown status, not machine identifiers.

Owner corrected inclusion of private infrastructure details. Removed identifiers,
location and connection details from current notes/reports; added the rule to
AGENTS.md and scanned HF staging. Earlier committed notes also required working
tree edits; Git history was not rewritten. Those historical commits retain their
old text until any separately authorized history work. No force-push was done.

## Local reference CPU

AMD EPYC-Genoa, four threads, batch one, 100 warmups/1,000 measured iterations.
Model path includes tokenization through probability. Inference benchmarks ran
after download/teardown and before the lightweight model experiments.

| Export/scenario | p50 | p95 | p99 |
| --- | ---: | ---: | ---: |
| FP32 ONNX, multilingual | 3.86 ms | 6.88 ms | 10.62 ms |
| FP32 ONNX, full 64 tokens | 13.24 ms | 17.75 ms | 20.82 ms |
| INT8 ONNX, multilingual | 2.85 ms | 4.45 ms | 5.69 ms |
| INT8 ONNX, full 64 tokens | 9.76 ms | 13.88 ms | 18.85 ms |

INT8 meets the model-path gate in this run; FP32 misses it at maximum length.
Whole-path p50/p95/p99 remain unmeasured because Vxpipe integration is absent.
FP32 has zero decision disagreements and <1e-4 probability drift. INT8 at batch
one changes two test decisions and one reviewed decision, scoring 899/908 and
94/98 respectively. `start over` improves; other large probability differences
require caution in calibration. Batch-32 quantization gives different outputs,
so the batch-one result is the serving comparison. INT8 is 406.8 MB, FP32 ONNX
470.7 MB; process peak RSS across both is 1,173 MiB, not isolated model memory.

Evidence and remaining acceptance gates: [results](../docs/e5-phrase-results.md),
`reports/e5-phrase-run1/cpu-verification.json`. Full checks after the export fix:
Ruff format/check clean, **168 passed, one integration test deselected**.
