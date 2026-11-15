"""Evaluate a trained model on one split (README.md).

    uv run python -m barge_in_detector.training.evaluate --model models/run1 --split test

Reports accuracy, macro F1, false and missed interruption rates, and results
per language; writes them to <model>/eval-<split>.json and the mistakes to
<model>/errors-<split>.csv.
"""

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from barge_in_detector.training.data import read_csv, resolve_splits
from barge_in_detector.training.metrics import LABELS, compute

BARGE_IN = LABELS.index("barge_in")


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, str]],
    *,
    batch_size: int = 128,
    max_length: int = 64,
) -> list[float]:
    """P(barge_in) for each row's cleaned text."""
    model.eval()
    dev = next(model.parameters()).device
    out: list[float] = []
    for i in range(0, len(rows), batch_size):
        texts = [r["text_clean"] for r in rows[i : i + batch_size]]
        enc = tokenizer(
            texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt"
        ).to(dev)
        logits = model(**enc).logits.float()
        out += torch.softmax(logits, dim=-1)[:, BARGE_IN].tolist()
    return out


def evaluate(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, str]],
    *,
    threshold: float = 0.5,
    max_length: int = 64,
) -> tuple[dict[str, Any], list[float]]:
    probs = predict(model, tokenizer, rows, max_length=max_length)
    preds = ["barge_in" if p >= threshold else "backchannel" for p in probs]
    metrics = compute([r["label"] for r in rows], preds, [r["language"] for r in rows])
    return metrics, probs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="a directory written by train")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5, help="P(barge_in) cut-off")
    parser.add_argument("--max-length", type=int, default=64, help="token budget")
    parser.add_argument("--dataset-repo", default="HashNuke/barge-in-detector-data")
    parser.add_argument(
        "--dataset-revision", help="Exact dataset commit; resolves current main when omitted"
    )
    args = parser.parse_args(argv)
    args.splits, args.dataset_revision = resolve_splits(
        args.splits, repo_id=args.dataset_repo, revision=args.dataset_revision
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device())
    rows = read_csv(args.splits / f"{args.split}.csv")
    metrics, probs = evaluate(
        model, tokenizer, rows, threshold=args.threshold, max_length=args.max_length
    )
    metrics.update(split=args.split, threshold=args.threshold)
    (args.model / f"eval-{args.split}.json").write_text(json.dumps(metrics, indent=2))
    with (args.model / f"errors-{args.split}.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "p_barge_in", "language", "caller_intent", "text"])
        for r, p in zip(rows, probs, strict=True):
            if (p >= args.threshold) != (r["label"] == "barge_in"):
                writer.writerow(
                    [r["label"], f"{p:.3f}", r["language"], r["caller_intent"], r["text"]]
                )
    print(
        f"{args.split}: n={metrics['count']} accuracy={metrics['accuracy']:.3f} "
        f"macro_f1={metrics['macro_f1']:.3f} "
        f"false_interruptions={metrics['false_interruption_rate']:.3f} "
        f"missed_interruptions={metrics['missed_interruption_rate']:.3f}"
    )
    for lang, m in metrics["per_language"].items():
        print(f"  {lang:>4} n={m['count']:<4} accuracy={m['accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
