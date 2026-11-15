"""Metrics for the two labels. Pure Python, so they are tested without torch."""

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

LABELS = ("backchannel", "barge_in")


def _prf(labels: Sequence[str], preds: Sequence[str], cls: str) -> dict[str, float]:
    tp = sum(1 for y, p in zip(labels, preds, strict=True) if y == cls and p == cls)
    predicted = sum(1 for p in preds if p == cls)
    actual = sum(1 for y in labels if y == cls)
    precision = tp / predicted if predicted else 0.0
    recall = tp / actual if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "count": actual}


def compute(
    labels: Sequence[str], preds: Sequence[str], languages: Sequence[str]
) -> dict[str, Any]:
    per_class = {cls: _prf(labels, preds, cls) for cls in LABELS}
    backchannel = [p for y, p in zip(labels, preds, strict=True) if y == "backchannel"]
    barge_in = [p for y, p in zip(labels, preds, strict=True) if y == "barge_in"]
    by_language: dict[str, list[bool]] = defaultdict(list)
    for y, p, lang in zip(labels, preds, languages, strict=True):
        by_language[lang].append(y == p)
    return {
        "count": len(labels),
        "accuracy": sum(y == p for y, p in zip(labels, preds, strict=True)) / len(labels),
        "macro_f1": sum(c["f1"] for c in per_class.values()) / len(LABELS),
        # A backchannel predicted as barge-in stops the agent for nothing.
        "false_interruption_rate": (
            sum(p == "barge_in" for p in backchannel) / len(backchannel) if backchannel else 0.0
        ),
        # A barge-in predicted as backchannel lets the agent talk over the caller.
        "missed_interruption_rate": (
            sum(p == "backchannel" for p in barge_in) / len(barge_in) if barge_in else 0.0
        ),
        "per_class": per_class,
        "per_language": {
            lang: {"accuracy": sum(ok) / len(ok), "count": len(ok)}
            for lang, ok in sorted(by_language.items())
        },
    }
