"""Add a separately versioned supplement without moving original split rows."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from barge_in_detector.cleaner import clean_text
from barge_in_detector.splitting import SPLITS, read_csv, split_rows, write_splits


def additive_splits(original, supplement, *, reserved, previous=None):
    accepted = [dict(r) for r in supplement if r["status"] == "accepted"]
    seen = {r["text_clean"] for rows in original.values() for r in rows}
    ids = {r["utterance_id"] for rows in original.values() for r in rows}
    for row in accepted:
        text, uid = row["text_clean"], row["utterance_id"]
        if text in reserved:
            raise ValueError("reserved evaluation text in supplement")
        if text in seen or uid in ids:
            raise ValueError("duplicate text or ID in supplement")
        seen.add(text)
        ids.add(uid)
    if not previous:
        extra = split_rows(accepted)
    else:
        old = {r["utterance_id"]: (s, r) for s, rows in previous.items() for r in rows}
        extra = {s: [] for s in SPLITS}
        pending, occupied = defaultdict(list), defaultdict(set)
        for row in accepted:
            uid, cid = row["utterance_id"], row["combination_id"]
            if uid in old:
                name, before = old[uid]
                if any(before[k] != row[k] for k in ("text_clean", "label", "combination_id")):
                    raise ValueError("previously split sample changed")
                extra[name].append(row)
                occupied[cid].add(name)
            else:
                pending[cid].append(row)
        fresh = [r for cid, rows in pending.items() if cid not in occupied for r in rows]
        for name, rows in split_rows(fresh).items():
            extra[name].extend(rows)
        for cid, names in occupied.items():
            rows = sorted(
                pending[cid], key=lambda r: hashlib.sha256(r["utterance_id"].encode()).hexdigest()
            )
            if rows and not names.intersection({"validation", "test"}):
                held = (
                    "validation"
                    if int(hashlib.sha256(cid.encode()).hexdigest(), 16) % 2
                    else "test"
                )
                extra[held].append(rows.pop(0))
            extra["train"].extend(rows)
        for rows in extra.values():
            rows.sort(key=lambda r: r["utterance_id"])
    combined = {s: [*original[s], *extra[s]] for s in SPLITS}
    return extra, combined


def main():
    source = Path("data/supplements/phrase-coverage-v1")
    out = source / "splits"
    original = {s: read_csv(Path("data/splits") / f"{s}.csv") for s in SPLITS}
    previous = {s: read_csv(out / f"{s}.csv") for s in SPLITS} if out.exists() else None
    reserved = {
        clean_text(r.get("text_clean") or r["text"])
        for p in Path("data/evaluation").glob("*.csv")
        for r in read_csv(p)
    }
    extra, combined = additive_splits(
        original, read_csv(source / "utterances.csv"), reserved=reserved, previous=previous
    )
    write_splits(extra, out)
    write_splits(combined, Path("data/experiments/original-plus-phrase-v1/splits"))
    report = {
        "original": {s: len(rows) for s, rows in original.items()},
        "supplement": {s: len(rows) for s, rows in extra.items()},
        "combined": {s: len(rows) for s, rows in combined.items()},
        "original_assignments_preserved": True,
        "exact_cross_split_duplicates": 0,
        "supplement_exact_evaluation_overlap": 0,
        "existing_evaluation_overlap_by_original_split": {
            s: sum(r["text_clean"] in reserved for r in rows) for s, rows in original.items()
        },
    }
    Path("reports/phrase-coverage-v1/splits.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
