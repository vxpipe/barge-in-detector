# E5 and lightweight models on the additive dataset

The selected E5 fine-tune is the strongest current development candidate. Its
INT8 ONNX export reaches **94/98** on the reviewed English cases with no false
interruptions, and meets the local 20 ms model-path p99 gate in this measurement.
It is retained in the HF model submodule regardless of the sparse-model results.
This is a candidate recommendation, not a production acceptance decision.

## Matched comparison

Training uses all 14 languages and the same original-plus-supplement split:
4,918 train / 915 validation / 908 test. Original data and prior models are
preserved. The unchanged original E5 is evaluated on the same expanded heldout.
New models choose their settings on validation before reading heldout/reviewed
labels. All use the same 64-token input budget and diagnostic threshold .5.

| Model | Combined synthetic test | Reviewed English | False interruptions / 50 | Missed interruptions / 48 | Short requests / 30 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original E5 | 841/908 | 72/98 | 0 | 26 | 5 |
| New E5, selected FP32 | 899/908 | 93/98 | 0 | 5 | 26 |
| New E5, INT8 ONNX, batch one | 899/908 | **94/98** | **0** | 4 | 27 |
| Character TF-IDF + logistic regression | 894/908 | 91/98 | 1 | 6 | 26 |
| Character/word TF-IDF + linear SVM | 898/908 | 91/98 | 0 | 7 | 25 |
| Character/word TF-IDF + Complement Naive Bayes | 890/908 | 93/98 | 2 | **3** | 27 |

E5 improves substantially over its original training: on the same synthetic
test, errors fall from 67 to 9; on reviewed English, FP32 errors fall from 26 to
5. The supplement improves the existing encoder, without needing a different
neural architecture. The original synthetic test does regress from 331/331 to
329/331, so retain the original result rather than silently changing its baseline.

The SVM wins the prespecified lightweight validation comparison (macro F1 .98896)
over logistic regression (.97904) and Naive Bayes (.98455). E5 validation macro F1
is .99559. Naive Bayes has fewer missed reviewed interruptions but introduces two
false stops and has lower multilingual synthetic performance. Its higher reviewed
score does not make it the validation winner. None of the sparse models clearly
displaces E5 on the measured accuracy/false-stop tradeoff.

E5's three seeds score 91/98, 90/98 and 93/98 in FP32. The chosen seed (20261117,
epoch 4) was selected by validation macro F1 before evaluation, not because it
scored best on reviewed cases. Each seed trained for 18–22 seconds; all training
and GPU evaluation took about 70 seconds. See the
[fixed E5 protocol](e5-supplement-finetune.md) and
[lightweight protocol](../labnotes/20261115-1818-lightweight-model-comparison.md).

## CPU latency and quantization

Owner-designated AMD EPYC-Genoa CPU, warmed batch one, 100 warmups and 1,000
measurements. E5 uses four threads; sparse models use one. Model path includes
tokenization, truncation, features/inference and probability. Loading is excluded.

| Model | Typical p50 / p95 / p99, ms | Full 64-token p50 / p95 / p99, ms |
| --- | --- | --- |
| E5 FP32 ONNX | 3.86 / 6.88 / 10.62 | 13.24 / 17.75 / 20.82 |
| E5 INT8 ONNX | 2.85 / 4.45 / 5.69 | 9.76 / 13.88 / 18.85 |
| Logistic regression | 0.95 / 1.16 / 1.25 | 1.93 / 5.02 / 5.74 |
| Linear SVM | 1.42 / 1.89 / 2.21 | 2.53 / 5.72 / 6.83 |
| Complement Naive Bayes | 1.66 / 2.03 / 2.51 | 2.55 / 6.19 / 7.36 |

INT8 meets the model-path gate in this run, with limited margin at the maximum
input length. **Whole-path p50/p95/p99 remain unmeasured for every candidate**:
the Vxpipe boundary is not implemented. No claim is made about its 50 ms gate.
Sparse models remain useful when resource use or additional timing margin matters.

FP32 ONNX has no decision differences from the GPU checkpoint across validation,
test and reviewed cases; maximum probability difference is below 1e-4. INT8 has
no validation decision changes, two test changes and one reviewed change at batch
one. Reviewed `start over` changes to a correct interruption. Other individual
probabilities can drift substantially (maximum test difference .9193), and dynamic
quantization can depend on batch composition. Batch-32 reviewed results remain
93/98; use the batch-one results for serving, not the batched diagnostic.

FP32 ONNX is 470.7 MB; INT8 ONNX is 406.8 MB because embeddings remain FP32.
Peak verification-process RSS is 1,173 MiB across the two formats; isolated runtime
memory and realistic concurrency still need measurement.

## Files, verification and limits

The local HF staging directory is
`huggingface/model/candidates/e5-phrase-run1/`: selected FP32 PyTorch checkpoint,
FP32 ONNX, INT8 ONNX, all three seed checkpoints, tokenizer/configuration,
checksums, card, licence and provenance. The old release at the root is preserved.
No code, datasets, training logs or research reports are included there.
The source/dataset changes are not yet committed or published: provenance records
their actual snapshot hashes and leaves the new HF dataset revision unset.
Complete those exact revision pins before publishing this candidate.

All 55 downloaded artifacts/reports passed SHA-256 verification, including 20
model files. All three selected inference formats loaded locally before the
training instance was destroyed; no Vast.ai instances remained. Rental compute
and storage were approximately $0.16, plus separately metered transfers. Local
architecture comparisons used no paid APIs or additional GPU rental.

Evidence: `reports/e5-phrase-run1/{selection,gpu-report,original-e5-comparison,
export-verification,cpu-verification,instance-lifecycle}.json` and
`reports/lightweight-v2/{selection,report}.json`. Predictions are retained beside
the reports. Infrastructure identifiers remain only in gitignored private records.

The 98 reviewed English cases are a development/regression set. Three exact texts
occur in original training and one in original validation; the supplement adds
none. They are too small and reused to establish a reliable generalization gain
from 93 to 94 cases, or multilingual production accuracy. Synthetic test scores
are not human verification. Realistic-prior calibration remains open for all
models, including the SVM's training-only sigmoid mapping.

## Recommended next evidence

Keep E5 INT8 as the leading serving candidate and the SVM as the lightweight
alternative. E5 INT8 still misses `repeat`, `enough`, `go back` and
`got it okay now let me explain` in the development set. The next data improvement
should cover these underlying request/transition families across the manifest,
with contrasting backchannels, rather than copy evaluation phrases into training.
Existing generation-coverage gaps remain recorded separately.

Before promotion, collect a new, frozen owner-reviewed English/Hindi evaluation
set, agree false-stop versus missed-stop limits, calibrate using validation with
realistic class priors, and measure the Vxpipe boundary. Preserve the current
98-case set for regression and prefix tests. No further paid generation is run
by this checkpoint.
