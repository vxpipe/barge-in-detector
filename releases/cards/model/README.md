---
language: [en, hi, kn, ta, te, ml, zh, ja, id, fil, fr, de, es, it]
license: mit
library_name: transformers
pipeline_tag: text-classification
base_model: intfloat/multilingual-e5-small
datasets: [HashNuke/barge-in-detector-data]
tags: [barge-in, backchannel, multilingual, experimental, onnx]
---

# Multilingual barge-in detector

A text classifier for caller speech **over an ongoing agent turn**: backchannel
(`0`, let the agent continue) or barge-in (`1`, give the caller the floor).
It consumes the first 64 tokenizer tokens of cleaned caller text, without a
`query:` prefix, preceding agent line or language hint. Probabilities remain
uncalibrated; .5 is a diagnostic threshold.

The new **E5 phrase-coverage model** is trained on the combined original dataset
and phrase supplement, across all 14 languages.

| Format | Download |
| --- | --- |
| Unquantized fine-tune, FP32 PyTorch | [model.safetensors](model.safetensors) |
| FP32 ONNX | [onnx/model-fp32.onnx](onnx/model.onnx) |
| Quantized INT8 ONNX | [onnx/model-int8.onnx](onnx/model_int8.onnx) |

Tokenizer/configuration accompany the checkpoint. The ONNX directory includes
its tokenizer and input/output metadata. Quantization applies to matrix
operations; embeddings remain FP32. See [verified hashes](artifacts.sha256.json) and [provenance](provenance.json).

**The root checkpoint is the new combined-data fine-tune.** Release `v0.2.0`
contains the FP32 checkpoint and both ONNX formats at stable paths. The original
baseline is available at tag `v0.1.0`; Git history preserves earlier versions.
Training seeds and historical model directories are not part of the current tree.

## Load the new FP32 model on CPU

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = "HashNuke/barge-in-detector"
revision = "v0.2.0"
tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
model = AutoModelForSequenceClassification.from_pretrained(repo, revision=revision).cpu().eval()
inputs = tokenizer("please wait", return_tensors="pt", truncation=True, max_length=64)
with torch.inference_mode():
    p_barge_in = model(**inputs).logits.softmax(-1)[0, 1].item()
```

Pin a release tag or full commit SHA for reproducibility. Clean inputs using the
[source transcript contract](https://github.com/vxpipe/barge-in-detector/blob/main/docs/transcript-format.md).
ONNX Runtime CPU inference uses int64 `input_ids` and `attention_mask`; apply
softmax to the two output logits and read index 1. Serving code lives in the
[GitHub source](https://github.com/vxpipe/barge-in-detector).

## Training data and evaluation

| Split | Original + supplement | Combined |
| --- | ---: | ---: |
| Train | 2,640 + 2,278 | 4,918 |
| Validation | 333 + 582 | 915 |
| Test | 331 + 577 | 908 |

Use dataset configuration `original-plus-phrase-v1` at the exact revision in
[release.json](release.json). The original dataset and standalone supplement
are also available. The supplement was generated without a judge; multilingual
labels remain synthetic.

The multilingual E5 encoder and binary head are fully fine-tuned. Three seeds
use five epochs, batch 32, AdamW 3e-5 and class-weighted cross-entropy. Validation
macro F1 selects seed 20261117, epoch 4 before heldout evaluation.

| Evaluation, threshold .5 | Original E5 | New E5 FP32 | New E5 INT8, batch one |
| --- | ---: | ---: | ---: |
| Combined synthetic test | 841/908 | 899/908 | 899/908 |
| Reviewed English development cases | 72/98 | 93/98 | 94/98 |
| False interruptions | 0/50 | 0/50 | 0/50 |
| Missed interruptions | 26/48 | 5/48 | 4/48 |

The 98 English cases are reused development examples with pre-existing overlaps
in the original dataset. They are not an independent estimate of real-world or
multilingual accuracy. Fresh English/Hindi review and realistic-prior calibration
remain open. Other languages are synthetic-only.

Owner-designated CPU, warmed batch one, four threads, 100 warmups/1,000 calls:

| Export | Typical p50 / p95 / p99 | Full 64-token p50 / p95 / p99 |
| --- | --- | --- |
| FP32 ONNX | 3.86 / 6.88 / 10.62 ms | 13.24 / 17.75 / 20.82 ms |
| INT8 ONNX | 2.85 / 4.45 / 5.69 ms | 9.76 / 13.88 / 18.85 ms |

INT8 meets the 20 ms model-path p99 target in this measurement. The Vxpipe boundary
is unmeasured, so the 50 ms whole-path target is not established. INT8 outputs
can differ from FP32 and with batch composition; these results use batch one.
This is an experimental release.

## Provenance and source

[release.json](release.json) pins the published source and dataset revisions and
artifact hashes. [Training provenance](provenance.json)
retains the actual source snapshot, base revision and selection.

Code, tests, research reports and labnotes live in
[GitHub](https://github.com/vxpipe/barge-in-detector). Versioned data lives in
[HashNuke/barge-in-detector-data](https://huggingface.co/datasets/HashNuke/barge-in-detector-data).
This repository contains model artifacts, cards, licence and provenance only.
MIT licence; see LICENSE and NOTICE for base-model attribution.
