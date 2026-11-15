"""CPU parity and warmed batch-one latency for the downloaded frozen E5 artifact.

uv run --no-sync python -m experiments.benchmark_frozen_e5
"""

import argparse
import json
import platform
import random
import resource
import time
from pathlib import Path

import torch

from barge_in_detector.splitting import read_csv
from barge_in_detector.training.frozen_e5 import MAX_LENGTH, PREFIX, FrozenE5
from experiments.audit_reviewed_errors import sha256
from experiments.benchmark_cpu import summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/frozen-e5-run1"))
    parser.add_argument(
        "--out", type=Path, default=Path("reports/frozen-e5-run1/cpu-benchmark.json")
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.iterations < 100 or args.warmup < 1 or args.threads < 1:
        parser.error("Use >=100 iterations and positive warmup/threads")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    predictor = FrozenE5(args.model)
    load_seconds = time.perf_counter() - started
    predictions = read_csv(args.model / "predictions-challenge.csv")
    cpu_probabilities = predictor.predict([r["text_clean"] for r in predictions])
    gpu_probabilities = [float(r["p_barge_in"]) for r in predictions]
    difference = max(abs(a - b) for a, b in zip(cpu_probabilities, gpu_probabilities, strict=True))
    disagreements = sum(
        (a >= 0.5) != (b >= 0.5) for a, b in zip(cpu_probabilities, gpu_probabilities, strict=True)
    )
    if difference > 1e-4 or disagreements:
        raise ValueError(f"CPU/GPU parity failed: {difference=}, {disagreements=}")
    rows = read_csv(Path("data/splits/test.csv"))
    texts = [r["text_clean"] for r in rows]
    random.Random(20261115).shuffle(texts)
    stress = (
        "please stop for a moment because i need to correct the information you are reading " * 12
    )
    assert (
        len(
            predictor.tokenizer(PREFIX + stress, truncation=True, max_length=MAX_LENGTH)[
                "input_ids"
            ]
        )
        == MAX_LENGTH
    )
    measurements = {}
    for name, inputs in (("multilingual_test", texts), ("full_64_tokens", [stress])):
        samples = []
        for index in range(args.warmup + args.iterations):
            started = time.perf_counter_ns()
            probability = predictor.predict([inputs[index % len(inputs)]], batch_size=1)[0]
            elapsed = (time.perf_counter_ns() - started) / 1e6
            assert 0 <= probability <= 1
            if index >= args.warmup:
                samples.append(elapsed)
        measurements[name] = summary(samples)
    report = {
        "scope": "Local machine designated by owner for CPU measurement, 2026-11-15. Tokenization through probability; excludes Vxpipe boundary/queueing.",
        "cpu": next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "threads": args.threads,
        "interop_threads": 1,
        "batch_size": 1,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "max_length": MAX_LENGTH,
        "head_sha256": sha256(args.model / "head.json"),
        "encoder_sha256": sha256(args.model / "encoder/model.safetensors"),
        "test_sha256": sha256(Path("data/splits/test.csv")),
        "cpu_gpu_max_probability_difference": difference,
        "cpu_gpu_label_disagreements": disagreements,
        "load_seconds_excluded": load_seconds,
        "model_path": measurements,
        "whole_path": None,
        "whole_path_reason": "Vxpipe integration boundary is not implemented in this experiment",
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
