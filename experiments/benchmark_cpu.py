"""Measure warmed batch-one tokenization through label output on local CPU.

Experimental timing harness: run from the repo root with a CPU PyTorch environment.
It measures no runtime boundary, queue, HTTP call or cold model load.
"""

import argparse
import hashlib
import json
import platform
import random
import resource
import statistics
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from barge_in_detector.training.data import read_csv, resolve_splits


def percentile(values, quantile):
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def summary(values):
    return {
        "p50_ms": percentile(values, 0.5),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "mean_ms": statistics.mean(values),
        "max_ms": max(values),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--model", type=Path, default=Path("models/e5-small-run1"))
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--splits", type=Path, help="Local dataset split directory")
    parser.add_argument("--dataset-revision")
    args = parser.parse_args()
    split_dir, dataset_revision = resolve_splits(args.splits, revision=args.dataset_revision)
    if args.iterations < 100 or args.warmup < 1 or min(args.threads) < 1:
        parser.error("Use >=100 measured iterations, positive warmup and threads")
    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = (
        AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True)
        .cpu()
        .eval()
    )
    load_seconds = time.perf_counter() - started
    rows = read_csv(split_dir / "test.csv")
    texts = [r["text_clean"] for r in rows]
    random.Random(20261115).shuffle(texts)
    stress = (
        "please stop for a moment because i need to correct the information you are reading " * 12
    )
    assert len(tokenizer(stress, truncation=True, max_length=64)["input_ids"]) == 64
    report = {
        "dataset_revision": dataset_revision,
        "model": str(args.model),
        "runtime": "PyTorch eager fp32, CPU only",
        "model_sha256": hashlib.sha256((args.model / "model.safetensors").read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "cpuinfo": next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "load_seconds_excluded": load_seconds,
        "iterations": args.iterations,
        "warmup_per_scenario": args.warmup,
        "batch_size": 1,
        "max_length": 64,
        "interop_threads": 1,
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "corpus_sha256": hashlib.sha256((split_dir / "test.csv").read_bytes()).hexdigest(),
        "corpus_rows": len(rows),
        "languages": sorted({r["language"] for r in rows}),
        "measurements": [],
        "scope": "Local virtual CPU; reference CPU unspecified. Excludes Vxpipe boundary and queueing; no whole-path acceptance claim.",
    }
    lengths = [len(tokenizer(t, truncation=True, max_length=64)["input_ids"]) for t in texts]
    report["corpus_token_lengths"] = {
        "p50": percentile(lengths, 0.5),
        "p95": percentile(lengths, 0.95),
        "max": max(lengths),
    }
    for threads in args.threads:
        torch.set_num_threads(threads)
        for name, inputs in [("multilingual_test", texts), ("full_64_tokens", [stress])]:
            samples, token_times, forward_times = [], [], []
            for index in range(args.warmup + args.iterations):
                text = inputs[index % len(inputs)]
                start = time.perf_counter_ns()
                enc = tokenizer(text, truncation=True, max_length=64, return_tensors="pt")
                tokenized = time.perf_counter_ns()
                logits = model(**enc).logits
                probability = torch.softmax(logits, dim=-1)[
                    0, model.config.label2id["barge_in"]
                ].item()
                label = "barge_in" if probability >= 0.5 else "backchannel"
                finish = time.perf_counter_ns()
                assert 0 <= probability <= 1
                assert label in model.config.label2id
                if index >= args.warmup:
                    samples.append((finish - start) / 1e6)
                    token_times.append((tokenized - start) / 1e6)
                    forward_times.append((finish - tokenized) / 1e6)
            result = {
                "threads": threads,
                "scenario": name,
                "model_path": summary(samples),
                "tokenization": summary(token_times),
                "forward_and_probability": summary(forward_times),
            }
            report["measurements"].append(result)
            print(json.dumps(result), flush=True)
    report["peak_process_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    (args.model / "cpu-benchmark.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
