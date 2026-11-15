"""Fine-tune a multilingual encoder on the utterance splits (README.md).

    uv sync
    uv run python -m barge_in_detector.training.train --out models/run1

Trains on train.csv's cleaned text, evaluates on validation.csv after every
epoch, and keeps the epoch with the best validation macro F1 in --out (model,
tokenizer, metrics.json). Backchannel is the smaller class, so the loss
weights each class by its inverse frequency. Uses the GPU when there is one.
"""

import argparse
import json
import random
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from barge_in_detector.training.data import read_csv, resolve_splits
from barge_in_detector.training.evaluate import device, evaluate
from barge_in_detector.training.metrics import LABELS

# MIT licence, 118M parameters, about 100 languages (dataset repo docs/utterance-first-decision.md).
DEFAULT_MODEL = "intfloat/multilingual-e5-small"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id or path")
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=float, default=0.1, help="share of steps")
    parser.add_argument("--max-length", type=int, default=64, help="token budget")
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--dataset-repo", default="HashNuke/barge-in-detector-data")
    parser.add_argument(
        "--dataset-revision", help="Exact dataset commit; resolves current main when omitted"
    )
    args = parser.parse_args(argv)
    args.splits, args.dataset_revision = resolve_splits(
        args.splits, repo_id=args.dataset_repo, revision=args.dataset_revision
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = device()
    train_rows = read_csv(args.splits / "train.csv")
    validation_rows = read_csv(args.splits / "validation.csv")
    label_id = {label: i for i, label in enumerate(LABELS)}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=label_id,
    ).to(dev)

    counts = Counter(r["label"] for r in train_rows)
    weights = torch.tensor(
        [len(train_rows) / (len(LABELS) * counts[label]) for label in LABELS], device=dev
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    steps = args.epochs * -(-len(train_rows) // args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup * steps), steps)
    use_bf16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()

    print(
        f"device={dev} bf16={use_bf16} model={args.model} train={len(train_rows)} "
        f"validation={len(validation_rows)} class_weights={weights.tolist()}"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_rows)))
        random.shuffle(order)
        started, total = time.time(), 0.0
        for i in range(0, len(order), args.batch_size):
            batch = [train_rows[j] for j in order[i : i + args.batch_size]]
            enc = tokenizer(
                [r["text_clean"] for r in batch],
                truncation=True,
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            ).to(dev)
            labels = torch.tensor([label_id[r["label"]] for r in batch], device=dev)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(**enc).logits
            loss = loss_fn(logits.float(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total += loss.item() * len(batch)
        metrics, _ = evaluate(model, tokenizer, validation_rows, max_length=args.max_length)
        record = {
            "epoch": epoch,
            "train_loss": total / len(train_rows),
            "seconds": round(time.time() - started, 1),
            "validation": metrics,
        }
        history.append(record)
        print(
            f"epoch {epoch}: loss={record['train_loss']:.4f} "
            f"validation macro_f1={metrics['macro_f1']:.3f} accuracy={metrics['accuracy']:.3f} "
            f"false_interruptions={metrics['false_interruption_rate']:.3f} "
            f"missed_interruptions={metrics['missed_interruption_rate']:.3f}"
        )
        if best is None or metrics["macro_f1"] > best["validation"]["macro_f1"]:
            best = record
            model.save_pretrained(args.out)
            tokenizer.save_pretrained(args.out)
    (args.out / "metrics.json").write_text(
        json.dumps(
            {"args": {k: str(v) for k, v in vars(args).items()}, "best": best, "history": history},
            indent=2,
        )
    )
    print(f"best epoch {best['epoch']}: validation macro_f1={best['validation']['macro_f1']:.3f}")
    print(f"saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
