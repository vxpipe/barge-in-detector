---
language: [en, hi, kn, ta, te, ml, zh, ja, id, fil, fr, de, es, it]
license: mit
library_name: transformers
pipeline_tag: text-classification
base_model: intfloat/multilingual-e5-small
tags: [barge-in, backchannel, multilingual, experimental, onnx]
---

# E5 phrase-coverage candidate

This candidate fine-tunes multilingual E5 on the original dataset plus a separate
phrase-coverage supplement. It classifies caller text spoken **over** an agent
turn as backchannel (`0`) or barge-in (`1`). It uses the first 64 tokenizer tokens,
without a `query:` prefix, preceding agent line or language hint. Clean inputs
using the versioned project transcript cleaner.

**Experimental release.** See [provenance.json](provenance.json) for the actual training
snapshot, dataset hashes and selected checkpoint. The original release at the
repository root is preserved.

| Variant | Files | Use |
| --- | --- | --- |
| Original unquantized fine-tune | [checkpoint/](checkpoint/) | PyTorch/Transformers, FP32 safetensors |
| FP32 ONNX | [onnx/model-fp32.onnx](onnx/model-fp32.onnx) | ONNX Runtime CPU |
| Quantized ONNX | [onnx/model-int8.onnx](onnx/model-int8.onnx) | Dynamic signed INT8 matrix operations; embeddings remain FP32 |
| Other training seeds | `seed-20261115/`, `seed-20261116/`, `seed-20261117/` | Retained full checkpoints for reproducibility |

Tokenizer/configuration accompany the checkpoints. ONNX uses `onnx/tokenizer.json`
and `onnx/metadata.json`; supply int64 `input_ids` and `attention_mask`, then
softmax the two output logits and read index 1. The quantized version is itself
an ONNX export. Its outputs can differ from FP32 and with batch composition.
The sparse classifier experiments are separate candidates.

Load the fine-tuned FP32 model on CPU:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = "HashNuke/barge-in-detector"
subfolder = "candidates/e5-phrase-run1/checkpoint"
tokenizer = AutoTokenizer.from_pretrained(repo, subfolder=subfolder)
model = AutoModelForSequenceClassification.from_pretrained(repo, subfolder=subfolder).cpu().eval()
inputs = tokenizer("please wait", return_tensors="pt", truncation=True, max_length=64)
with torch.inference_mode():
    p_barge_in = model(**inputs).logits.softmax(-1)[0, 1].item()
```

Training and evaluation code, research reports and datasets belong in the
[source repository](https://github.com/vxpipe/barge-in-detector), not this model
repository. [artifacts.sha256.json](artifacts.sha256.json) records verified model
file hashes. Licence and base-model attribution are in LICENSE and NOTICE.

## Training and evaluation

All 14 languages share fixed 4,918 training / 915 validation / 908 test examples.
The original dataset is preserved alongside the 3,437-row accepted supplement.
Base: `intfloat/multilingual-e5-small` at
`614241f622f53c4eeff9890bdc4f31cfecc418b3`. Full-encoder fine-tuning uses five
epochs, batch 32, AdamW 3e-5, weight decay .01, 10% warmup, class-weighted
cross-entropy and BF16 training. The saved checkpoint is FP32.

Epoch and seed selection used combined-validation macro F1 before held-out
analysis: seed 20261117, epoch 4. Each of three seeds trained for 18–22 seconds;
training, selection and GPU evaluations together took about 70 seconds.

| Evaluation, threshold .5 | Original E5 | Selected new E5, FP32 |
| --- | ---: | ---: |
| Same combined synthetic test | 841/908 (92.6%) | 899/908 (99.0%) |
| Reviewed English development cases | 72/98 (73.5%) | 93/98 (94.9%) |
| False interruptions in reviewed backchannels | 0/50 | 0/50 |
| Missed reviewed interruptions | 26/48 | 5/48 |
| Correct short interruptions | 5/30 | 26/30 |

Reviewed results across the three new seeds range from 90/98 to 93/98; none
produces a false interruption on these 50 backchannels. The selected seed was
chosen on validation, not on its reviewed score.

The multilingual test is synthetic. The 98 English examples have been reused
for development and contain pre-existing overlaps with the original dataset;
they are not a fresh independent accuracy estimate. Other languages do not yet
have human-verified acceptance results. Probabilities are uncalibrated under
realistic traffic priors, and .5 is a diagnostic threshold. Vxpipe whole-path
latency and fresh human evaluation remain open acceptance gates.

## Local CPU export results

Owner-designated AMD EPYC-Genoa CPU, four threads, warmed batch one; 100 warmups
and 1,000 measurements per scenario. Includes tokenization through probability.

| Export | Typical p50 / p95 / p99 | Full 64-token p50 / p95 / p99 |
| --- | --- | --- |
| FP32 | 3.86 / 6.88 / 10.62 ms | 13.24 / 17.75 / 20.82 ms |
| INT8 | 2.85 / 4.45 / 5.69 ms | 9.76 / 13.88 / 18.85 ms |

INT8 meets the 20 ms model-path p99 gate in this measurement; FP32 misses it at
the full input budget. This excludes the Vxpipe boundary, whose 50 ms whole-path
gate remains unmeasured. Peak process RSS across both loaded formats was about
1,173 MiB; it is not an isolated per-model memory measurement. Files are 470.7 MB
(FP32 ONNX) and 406.8 MB (INT8 ONNX); the large embedding table stays FP32.

At serving batch size one, INT8 scores **94/98** on the reviewed set: no false
interruptions and four missed interruptions, including 27/30 correct short
requests. It changes one reviewed decision and two synthetic-test decisions
relative to FP32; synthetic test accuracy remains 899/908. Probability drift can
be large on individual examples. Batch-32 verification gives different INT8
outputs, so these reported serving results explicitly use batch one.
