"""Time a new Python process through its first label from local model files."""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

CHILD = """
import json
import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import sys
model_path, threads = sys.argv[1], int(sys.argv[2])
torch.set_num_threads(threads)
torch.set_num_interop_threads(1)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).cpu().eval()
with torch.inference_mode():
    loaded_start = time.perf_counter_ns()
    enc = tokenizer('please stop for a moment', truncation=True, max_length=64, return_tensors='pt')
    p = torch.softmax(model(**enc).logits, dim=-1)[0, model.config.label2id['barge_in']].item()
    label = 'barge_in' if p >= 0.5 else 'backchannel'
    loaded_ms = (time.perf_counter_ns() - loaded_start) / 1e6
print(json.dumps({'label': label, 'p_barge_in': p, 'loaded_to_first_label_ms': loaded_ms}), flush=True)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/e5-small-run1"))
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    samples = []
    loaded_first = []
    env = {
        **os.environ,
        "OMP_NUM_THREADS": str(args.threads),
        "MKL_NUM_THREADS": str(args.threads),
        "TOKENIZERS_PARALLELISM": "false",
    }
    for run in range(args.runs):
        start = time.perf_counter_ns()
        with subprocess.Popen(
            [sys.executable, "-c", CHILD, str(args.model), str(args.threads)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        ) as process:
            line = process.stdout.readline()
            elapsed = (time.perf_counter_ns() - start) / 1e9
            result = json.loads(line)
            if process.wait() != 0:
                raise RuntimeError("Cold-start child failed")
        samples.append(elapsed)
        loaded_first.append(result["loaded_to_first_label_ms"])
        print(f"cold start {run + 1}: {elapsed:.3f}s {result}", flush=True)
    report = {
        "cold_process_seconds": samples,
        "loaded_to_first_label_ms": loaded_first,
        "loaded_to_first_label_median_ms": statistics.median(loaded_first),
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "runs": args.runs,
        "threads": args.threads,
        "scope": "Python process creation through reading first flushed label. Includes imports, tokenizer/model load, first inference. Local files only; OS file cache may be warm. Excludes process shutdown, downloads and HTTP/network boundary.",
    }
    (args.model / "cpu-cold-benchmark.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
