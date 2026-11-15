# CPU inference latency

## Requested measurements

- Cold process: new Python process through its first label, including imports,
  local tokenizer/model load and first inference. OS disk cache may be warm;
  no model downloads and no HTTP/Vxpipe boundary.
- Warm inference: model and tokenizer already in memory, batch one, from
  tokenization through label output (threshold 0.5).
- Local machine: four virtual AMD EPYC-Genoa CPUs, KVM, x86-64, AVX2/AVX512.
  This is not a designated reference CPU and does not establish serving gates.
- Saved fp32 multilingual-e5-small checkpoint, PyTorch eager 2.15.0+cpu,
  transformers 5.23.0. No ONNX or quantization applied.
- Warm sweep: 100 warm-up iterations and 1,000 measured iterations per
  scenario/thread count (one, two and four intra-op threads; one inter-op).
  Multilingual test corpus and a separate synthetic 64-output-token stress
  input. Long inputs still tokenize before truncation; stress text is longer
  than 64 tokens before truncation.
- Raw reports and logs saved under models/e5-small-run1 (gitignored).

Reproduce with the CPU environment:

```shell
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --no-project --python ~/.cache/barge-in-detector-cpu/bin/python \
  python -m experiments.benchmark_cpu
uv run --no-project --python ~/.cache/barge-in-detector-cpu/bin/python \
  python -m experiments.benchmark_cpu_cold
```

## Results

Initial thread sweep measures tokenization through probability; final four-thread run explicitly includes label selection. Both use 100 warm-ups and 1,000 measured calls per scenario. Final corpus is 331 synthetic test examples across 14 languages, cycled in seeded order: median 13, p95 19, maximum 26 tokens. Full-64 stress is a repeated English input, not a representative multilingual length distribution.

| Run | Threads | Input | p50 ms | p95 ms | p99 ms |
| --- | --- | --- | --- | --- | --- |
| Probability sweep | 1 | multilingual_test | 20.40 | 25.48 | 28.26 |
| Probability sweep | 1 | full_64_tokens | 44.70 | 50.55 | 53.76 |
| Probability sweep | 2 | multilingual_test | 19.41 | 23.16 | 25.65 |
| Probability sweep | 2 | full_64_tokens | 38.02 | 45.52 | 53.98 |
| Probability sweep | 4 | multilingual_test | 13.18 | 19.21 | 24.81 |
| Probability sweep | 4 | full_64_tokens | 26.39 | 32.49 | 41.04 |
| Final label output | 4 | multilingual_test | 13.09 | 19.42 | 25.32 |
| Final label output | 4 | full_64_tokens | 26.79 | 33.27 | 37.86 |

Cold process: ten runs with four threads, local files, model-independent process restart; OS file cache not flushed. Median 4.223 seconds; range 3.848–4.811 seconds. This includes Python/imports/model+tokenizer load and the first label, stopping when the parent receives the flushed output. First inference after loading (without request warm-up) median 23.50 ms on one short English input.

Model SHA256: `bddb65c08f6260cfc9d320e65b4229229ec9f3ca95204aed98ab5ae45ff758d3`. Parameter storage 448.8 MiB; peak process RSS 1163.6 MiB in final warm run. RSS includes Python/runtime and is not model-only memory.

A persistent server pays model startup once, then serves warmed inference. These timings exclude serialization, HTTP/network and queueing. Local model-path p99 exceeds the 20 ms project goal; no reference-CPU or whole-path acceptance claim. ONNX/int8 remains unmeasured. No model weights or training data changed.

Completion checks after timing finished (avoiding CPU contention): Ruff format
and lint passed; pytest 116 passed, one integration test deselected (10.22 s).
