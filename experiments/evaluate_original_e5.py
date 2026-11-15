"""Score the preserved original E5 checkpoint on the same expanded evaluation sets."""

import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from barge_in_detector.training.evaluate import predict
from experiments.audit_reviewed_errors import sha256
from experiments.finetune_e5_supplement import evaluation_sets, summarize


def main():
    torch.set_num_threads(4)
    checkpoint = Path("models/original-e5")
    expected = "bddb65c08f6260cfc9d320e65b4229229ec9f3ca95204aed98ab5ae45ff758d3"
    assert sha256(checkpoint / "model.safetensors") == expected
    model = (
        AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True)
        .cuda()
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    out = Path("models/e5-phrase-run1/reports")
    results = {}
    for name, rows in evaluation_sets().items():
        probs = predict(model, tokenizer, rows, max_length=64)
        rows = [{**r, "p_barge_in": p} for r, p in zip(rows, probs, strict=True)]
        results[name] = summarize(rows)
        with (out / f"predictions-original-e5-{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (out / "original-e5-comparison.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": expected,
                "results": results,
                "code_sha256": sha256(Path(__file__)),
                "scope": "Unmodified original fine-tuned E5; fixed 0.5 threshold; same evaluation sets as new E5.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({k: v["all"]["accuracy"] for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
