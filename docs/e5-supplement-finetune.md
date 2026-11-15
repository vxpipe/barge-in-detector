# E5 fine-tuning on the additive dataset

Protocol fixed before this run, 2026-11-15. The owner authorizes one Vast.ai
RTX 5090 below $0.65/hour, GPU training/analysis, verified artifact download,
then destruction of the instance. Original data/models stay untouched; no HF
publication or Git commit is requested.

## Training and selection

- Same `intfloat/multilingual-e5-small` encoder and two-label classification head
  as the original fine-tuning experiment; fine-tune all parameters. Initialize
  from pretrained revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`.
- All 14 languages, fixed original-plus-phrase split: 4,918 train / 915 validation
  / 908 test. Original and supplement remain separate datasets. Exact cleaned
  train/validation/test overlap is zero. Existing challenge overlap in original
  data remains disclosed; supplement adds none.
- Match original training settings: five epochs, batch 32, AdamW lr 3e-5,
  weight decay .01, 10% warmup, inverse-frequency class-weighted cross-entropy,
  gradient clipping 1, bf16 autocast on GPU, 64-token input budget, no text prefix
  added (same classification input as original fine-tuning).
- Fixed seeds 20261115, 20261116, 20261117. For each, select the epoch with highest
  combined-validation macro F1; ties retain earlier epoch. Select the seed by
  that validation score, ties retaining listed seed order. Record selection
  before loading held-out or reviewed examples. No hyperparameter sweep.
- After selection, report every seed on combined/original/supplement heldouts
  and the same owner-reviewed English cases. Diagnostic threshold .5, uncalibrated.
  Report false/missed interruptions, prefixes, per-language performance and
  seed variability. Fresh human evaluation is still required for a product choice.

## CPU export

Export the selected checkpoint to ONNX, optimize the BERT graph, then apply
dynamic per-channel int8 quantization to MatMul/Gemm/Attention; vocabulary
embeddings stay fp32. [ONNX Runtime documents](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
the transformer optimization and dynamic-quantization workflow and its possible
accuracy loss. Quantization is an experiment, not an assumed accuracy-preserving
transformation. Verify FP32 output differences <=1e-4 against GPU probabilities;
report int8 drift and changed decisions separately on validation/test/review.

Run tokenization-through-probability batch-one timings on the owner-designated
local CPU, 100 warmups/1,000 samples for multilingual inputs and full 64 tokens,
recording p50/p95/p99, memory and artifact size. The model-path gate is 20 ms p99.
The Vxpipe boundary is absent: whole-path timings and its 50 ms gate remain open.
Do not call an export a production replacement based only on synthetic accuracy.

## Budget and artifact lifecycle

Expected rental plus transfers under $1; the earlier recorded generation spend
is $14.60 ($16.85 including reserves), leaving room under the owner's $30 total
ceiling. Select a verified datacenter RTX 5090 with good network and the Vast.ai
PyTorch image; include 35 GB disk when checking the hourly price. All remote
setup, training and export run inside tmux `gpubox`, with logs on disk.

The owner freed local disk space and requested staging in the HF model submodule.
Download all three seed checkpoints, the selected unquantized PyTorch checkpoint,
FP32 ONNX and INT8 ONNX under `huggingface/model/candidates/e5-phrase-run1/`.
Preserve the existing release at the root. Only models, tokenizer/configuration,
cards and provenance belong there; research reports remain in this repository.
Use explicit file lists and per-file SHA-256 verification before destroying it.
Version sanitized reports/source manifests under `reports/e5-phrase-run1/`.
