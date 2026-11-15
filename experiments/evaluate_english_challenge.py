"""Frozen-checkpoint diagnostic experiment; labels are authored before inference.

Run from the repo root with uv run --group train python -m experiments.evaluate_english_challenge.
Evidence is saved under the model directory, outside the training splits.
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from barge_in_detector.training.data import read_csv, resolve_splits
from barge_in_detector.training.evaluate import device, predict
from barge_in_detector.training.metrics import compute

MODEL = Path("models/e5-small-run1")


def score(rows, probs):
    return compute(
        [r["label"] for r in rows],
        ["barge_in" if p >= 0.5 else "backchannel" for p in probs],
        [r["language"] for r in rows],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("."), help="Local dataset repository root"
    )
    parser.add_argument("--dataset-revision")
    args = parser.parse_args()
    if args.dataset_dir is None:
        split_dir, revision = resolve_splits(None, revision=args.dataset_revision)
        dataset_dir = split_dir.parents[1]
    else:
        dataset_dir = args.dataset_dir
        revision = None
    challenge = dataset_dir / "data/evaluation/english-challenge-v1.csv"
    rows = read_csv(challenge)
    splits = {
        s: read_csv(dataset_dir / f"data/splits/{s}.csv") for s in ("train", "validation", "test")
    }
    texts = {s: {r["text_clean"] for r in rs} for s, rs in splits.items()}
    all_seen = set.union(*texts.values())
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device())
    probs = predict(model, tokenizer, rows, max_length=64)
    single_probs = predict(model, tokenizer, rows, batch_size=1, max_length=64)
    report = {
        "dataset_revision": revision,
        "challenge_sha256": hashlib.sha256(challenge.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256((MODEL / "model.safetensors").read_bytes()).hexdigest(),
        "threshold": 0.5,
        "runtime": {
            "device": str(next(model.parameters()).device),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "threads": torch.get_num_threads(),
        },
        "batch_vs_single_max_probability_difference": max(
            abs(a - b) for a, b in zip(probs, single_probs, strict=True)
        ),
        "batch_vs_single_label_disagreements": sum(
            (a >= 0.5) != (b >= 0.5) for a, b in zip(probs, single_probs, strict=True)
        ),
        "review_status": "Owner-reviewed v1 labels match authored expectations after authorized reconciliation. Diagnostic development set; not independent representative accuracy evidence.",
        "all": score(rows, probs),
        "by_category": {},
        "overlap_counts": {s: sum(r["text_clean"] in ts for r in rows) for s, ts in texts.items()},
        "cross_split_exact_overlap": {
            f"{a}/{b}": len(texts[a] & texts[b])
            for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
        },
    }
    for category in sorted({r["category"] for r in rows}):
        pairs = [(r, p) for r, p in zip(rows, probs, strict=True) if r["category"] == category]
        report["by_category"][category] = score(*zip(*pairs, strict=True))
    unseen = [(r, p) for r, p in zip(rows, probs, strict=True) if r["text_clean"] not in all_seen]
    report["unseen"] = score(*zip(*unseen, strict=True))
    groups = defaultdict(list)
    for r, p in zip(rows, probs, strict=True):
        if r["group"]:
            groups[r["group"]].append(
                {"text": r["text_clean"], "label": r["label"], "p_barge_in": p}
            )
    report["prefix_groups"] = dict(groups)
    report["prefix_groups_all_labels_correct"] = sum(
        all((r["p_barge_in"] >= 0.5) == (r["label"] == "barge_in") for r in group)
        for group in groups.values()
    )
    # A diagnostic baseline: select its word-count cutoff using validation only.
    english = {s: [r for r in rs if r["language"] == "en"] for s, rs in splits.items()}

    def length_probs(rs, cutoff):
        return [float(len(r["text_clean"].split()) >= cutoff) for r in rs]

    cutoff = max(
        range(1, 25),
        key=lambda n: score(english["validation"], length_probs(english["validation"], n))[
            "macro_f1"
        ],
    )
    report["english_length_baseline"] = {
        "cutoff_words": cutoff,
        "validation": score(english["validation"], length_probs(english["validation"], cutoff)),
        "test": score(english["test"], length_probs(english["test"], cutoff)),
        "challenge": score(rows, length_probs(rows, cutoff)),
    }
    with (MODEL / "challenge-predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*rows[0], "p_barge_in", "prediction", "seen_in"])
        writer.writeheader()
        for r, p in zip(rows, probs, strict=True):
            writer.writerow(
                {
                    **r,
                    "p_barge_in": p,
                    "prediction": "barge_in" if p >= 0.5 else "backchannel",
                    "seen_in": ";".join(s for s, ts in texts.items() if r["text_clean"] in ts),
                }
            )
    (MODEL / "challenge-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "prefix_groups"}, indent=2))


if __name__ == "__main__":
    main()
