"""Train / validation / test split of the accepted utterances (docs/training.md).

Every combination is spread over the splits (user decision, 2026-11-15): one
with n >= 2 accepted samples puts n - 1 in train and 1 in validation or test,
so no combination is only in training or missing from it; one with a single
sample keeps it in train. Within each (language, label), combinations take
validation and test in turn, so both cover every language and label. Which
sample is held out is fixed by a hash of its id, so the split is
deterministic.
"""

import csv
import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SPLITS = ("train", "validation", "test")
COLUMNS = (
    "utterance_id",
    "combination_id",
    "language",
    "regional_variant",
    "code_switching",
    "code_switch_set",
    "caller_intent",
    "label",
    "text",
    "text_clean",
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def split_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    accepted = sorted(
        (dict(r) for r in rows if r["status"] == "accepted"), key=lambda r: r["utterance_id"]
    )
    by_combination: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in accepted:
        by_combination[r["combination_id"]].append(r)
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for cid, items in by_combination.items():
        groups[(items[0]["language"], items[0]["label"])].append(cid)
    held = {}
    for cids in groups.values():
        for i, cid in enumerate(sorted(cids)):
            held[cid] = "validation" if i % 2 == 0 else "test"
    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLITS}
    for cid, items in by_combination.items():
        out = min(items, key=lambda r: _hash(r["utterance_id"])) if len(items) > 1 else None
        for r in items:
            splits[held[cid] if r is out else "train"].append(r)
    for name in SPLITS:
        splits[name].sort(key=lambda r: r["utterance_id"])
    return splits


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_splits(splits: Mapping[str, Sequence[Mapping[str, Any]]], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        with (out_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(items)
