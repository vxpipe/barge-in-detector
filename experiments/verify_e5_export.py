"""Verify ONNX probabilities against saved GPU predictions; optionally benchmark local CPU."""

import argparse
import csv
import gc
import json
import random
import resource
import time
from pathlib import Path

import numpy as np
import onnxruntime

from barge_in_detector.splitting import read_csv
from barge_in_detector.training.onnx_infer import OnnxClassifier
from experiments.audit_reviewed_errors import score, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("models/e5-phrase-run1"))
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reports", type=Path, help="Prediction inputs and verification output")
    parser.add_argument("--fp32-file", default="model-fp32.onnx")
    parser.add_argument("--int8-file", default="model-int8.onnx")
    args = parser.parse_args()
    reports = args.reports or args.run / "reports"
    selection = json.loads((reports / "selection.json").read_text())
    seed = selection["seed"]
    sets = {
        name: read_csv(reports / f"predictions-seed-{seed}-{name}.csv")
        for name in ("combined_validation", "combined_test", "challenge")
    }
    report = {
        "onnxruntime": onnxruntime.__version__,
        "threads": args.threads,
        "whole_path": None,
        "whole_path_reason": "Vxpipe boundary is not implemented",
        "reference": "Owner-designated local CPU"
        if args.benchmark
        else "Remote export verification only; not a latency reference",
        "runtimes": {},
    }
    if args.benchmark:
        report["cpu"] = next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        )
    for precision in ("fp32", "int8"):
        model_file = args.fp32_file if precision == "fp32" else args.int8_file
        started = time.perf_counter()
        model = OnnxClassifier(args.run / "onnx", threads=args.threads, model_file=model_file)
        result = {
            "load_seconds_excluded": time.perf_counter() - started,
            "prediction_batch_size": 1 if args.benchmark else 32,
            "parity": {},
            "model_path": {},
        }
        for name, rows in sets.items():
            probs = []
            batch_size = result["prediction_batch_size"]
            for i in range(0, len(rows), batch_size):
                probs.extend(model.predict([r["text_clean"] for r in rows[i : i + batch_size]]))
            predicted = [{**r, "p_barge_in": p} for r, p in zip(rows, probs, strict=True)]
            if args.benchmark:
                with (reports / f"predictions-onnx-{precision}-{name}.csv").open(
                    "w", newline=""
                ) as f:
                    writer = csv.DictWriter(f, fieldnames=list(predicted[0]))
                    writer.writeheader()
                    writer.writerows(predicted)
            delta = max(abs(float(r["p_barge_in"]) - p) for r, p in zip(rows, probs, strict=True))
            disagreements = sum(
                (float(r["p_barge_in"]) >= 0.5) != (p >= 0.5)
                for r, p in zip(rows, probs, strict=True)
            )
            result["parity"][name] = {
                "max_probability_difference": delta,
                "label_disagreements": disagreements,
                "gpu_metrics": score([{**r, "p_barge_in": float(r["p_barge_in"])} for r in rows]),
                "cpu_metrics": score(predicted),
            }
            if precision == "fp32" and delta > 1e-4:
                raise RuntimeError(f"FP32 export probability mismatch: {delta}")
        if args.benchmark:
            texts = [r["text_clean"] for r in sets["combined_test"]]
            random.Random(20261115).shuffle(texts)
            stress = (
                "please stop for a moment because i need to correct the information you are reading "
                * 12
            )
            assert len(model.tokenizer.encode(stress).ids) == 64
            for name, inputs in (("multilingual_test", texts), ("full_64_tokens", [stress])):
                elapsed = []
                for i in range(1100):
                    start = time.perf_counter_ns()
                    probability = model.predict([inputs[i % len(inputs)]])[0]
                    duration = (time.perf_counter_ns() - start) / 1e6
                    assert 0 <= probability <= 1
                    if i >= 100:
                        elapsed.append(duration)
                result["model_path"][name] = dict(
                    zip(
                        ("p50_ms", "p95_ms", "p99_ms"),
                        np.percentile(elapsed, [50, 95, 99]).tolist(),
                        strict=True,
                    )
                )
        result["model_file"] = model_file
        result["model_bytes"] = (args.run / "onnx" / model_file).stat().st_size
        report["runtimes"][precision] = result
        del model
        gc.collect()
    report["peak_process_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    report["artifact_sha256"] = {
        p.name: sha256(p)
        for p in (args.run / "onnx").iterdir()
        if p.is_file() and p.name != "model-raw.onnx"
    }
    report["code_sha256"] = sha256(Path(__file__))
    destination = reports / (
        "cpu-verification.json" if args.benchmark else "export-verification.json"
    )
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
