"""Export the selected local candidate, verify parity and benchmark CPU inference.

uv run --group baseline python -m experiments.verify_lexical
"""

import argparse
import json
import platform
import random
import resource
import time
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from barge_in_detector.splitting import read_csv
from barge_in_detector.training.lexical import LexicalModel, export_model
from experiments.audit_reviewed_errors import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="lexical-run1")
    args = parser.parse_args()
    if Path(args.run).name != args.run:
        raise ValueError("Run must be a directory name")
    root = Path("models") / args.run
    reports = Path("reports") / args.run
    destination = root / "portable"
    # This pickle was produced locally by our own experiment; exported inference
    # loads JSON and tokenizer files, never pickle.
    candidate = joblib.load(root / "local-experiment.joblib")
    export_model(
        candidate["chars"],
        candidate["words"],
        candidate["classifier"],
        candidate["word_scale"],
        Path("models/frozen-e5-run1/encoder/tokenizer.json"),
        destination,
    )
    started = time.perf_counter()
    model = LexicalModel(destination)
    load_seconds = time.perf_counter() - started
    comparisons = {}
    for split in ("validation", "test", "challenge"):
        rows = read_csv(reports / f"predictions-{split}.csv")
        probs = model.predict([r["text_clean"] for r in rows])
        difference = max(abs(p - float(r["p_barge_in"])) for p, r in zip(probs, rows, strict=True))
        disagreements = sum(
            (p >= 0.5) != (float(r["p_barge_in"]) >= 0.5) for p, r in zip(probs, rows, strict=True)
        )
        assert difference < 1e-10 and disagreements == 0
        comparisons[split] = {
            "max_probability_difference": difference,
            "label_disagreements": disagreements,
        }
    texts = [r["text_clean"] for r in read_csv(Path("data/splits/test.csv"))]
    random.Random(20261115).shuffle(texts)
    stress = (
        "please stop for a moment because i need to correct the information you are reading " * 12
    )
    measurements = {}
    with threadpool_limits(limits=1):
        for scenario, inputs in (("multilingual_test", texts), ("full_64_tokens", [stress])):
            samples = []
            for index in range(1100):
                started = time.perf_counter_ns()
                probability = model.predict([inputs[index % len(inputs)]])[0]
                elapsed = (time.perf_counter_ns() - started) / 1e6
                assert 0 <= probability <= 1
                if index >= 100:
                    samples.append(elapsed)
            measurements[scenario] = dict(
                zip(
                    ("p50_ms", "p95_ms", "p99_ms"),
                    np.percentile(samples, [50, 95, 99]).tolist(),
                    strict=True,
                )
            )
    files = {str(p.relative_to(destination)): sha256(p) for p in sorted(destination.iterdir())}
    (destination / "artifacts.sha256.json").write_text(json.dumps(files, indent=2) + "\n")
    report = {
        "scope": "Owner-designated local CPU; batch one, 100 warmups, 1000 iterations. Includes 64-token truncation, character/word TF-IDF and probability. No Vxpipe integration.",
        "cpu": next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "platform": platform.platform(),
        "load_seconds_excluded": load_seconds,
        "threads": 1,
        "parity": comparisons,
        "model_path": measurements,
        "whole_path": None,
        "whole_path_reason": "Vxpipe boundary is not implemented",
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "artifact_bytes": sum(p.stat().st_size for p in destination.iterdir()),
        "artifact_sha256": files,
        "verification_code_sha256": sha256(Path(__file__)),
        "inference_code_sha256": sha256(Path("barge_in_detector/training/lexical.py")),
    }
    (reports / "cpu-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
