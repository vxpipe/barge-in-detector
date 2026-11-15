"""Audit saved owner reviews, frozen predictions and training coverage, offline.

Run: uv run python -m experiments.audit_reviewed_errors
No inference, threshold selection, annotation changes or dataset writes.
"""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from barge_in_detector.review.store import ReviewStore
from barge_in_detector.splitting import SPLITS, read_csv
from barge_in_detector.training.metrics import LABELS, compute


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reviewed_predictions(source, reviews, predictions):
    store = ReviewStore(source, reviews)
    decisions = {r["case_id"]: r["review"] for r in store.snapshot()["cases"]}
    saved = read_csv(predictions)
    by_id = {r["case_id"]: r for r in saved}
    if len(by_id) != len(saved) or set(by_id) != store.ids:
        raise ValueError("Predictions must contain exactly the source's unique case IDs")
    rows = []
    for row in store.rows:
        cid = row["case_id"]
        review = decisions[cid]
        if not review or review.get("decision") not in LABELS:
            raise ValueError(f"Missing binary review for {cid}")
        prediction = by_id[cid]
        if prediction["text_clean"] != row["text_clean"]:
            raise ValueError(f"Prediction text mismatch for {cid}")
        p = float(prediction["p_barge_in"])
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(f"Invalid probability for {cid}")
        rows.append(
            {
                **row,
                "authored_label": row["label"],
                "label": review["decision"],
                "p_barge_in": p,
            }
        )
    return rows


def score(rows, threshold=0.5):
    if not rows:
        return {"count": 0}
    return compute(
        [r["label"] for r in rows],
        ["barge_in" if r["p_barge_in"] >= threshold else "backchannel" for r in rows],
        [r["language"] for r in rows],
    )


def coverage(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[f] for f in fields)].append(len(row["text_clean"].split()))
    return [
        {
            **dict(zip(fields, key, strict=True)),
            "count": len(lengths),
            "mean_whitespace_words": sum(lengths) / len(lengths),
            "length_counts": {
                "1-2": sum(n <= 2 for n in lengths),
                "3-5": sum(3 <= n <= 5 for n in lengths),
                "6+": sum(n >= 6 for n in lengths),
            },
        }
        for key, lengths in sorted(groups.items())
    ]


def near_duplicates(train, heldout, threshold=0.8):
    """Same-language character-trigram Jaccard, diagnostic only; exact pairs excluded."""

    def grams(text):
        return {text[i : i + 3] for i in range(max(1, len(text) - 2))}

    train_grams = [grams(r["text_clean"]) for r in train]
    postings = defaultdict(set)
    for index, (row, tokens) in enumerate(zip(train, train_grams, strict=True)):
        for token in tokens:
            postings[(row["language"], token)].add(index)
    matches = []
    for row in heldout:
        tokens = grams(row["text_clean"])
        intersections = Counter(
            index for token in tokens for index in postings[(row["language"], token)]
        )
        candidates = []
        for index, intersection in intersections.items():
            if row["text_clean"] == train[index]["text_clean"]:
                continue
            similarity = intersection / (len(tokens) + len(train_grams[index]) - intersection)
            if similarity >= threshold:
                candidates.append((similarity, -index))
        if candidates:
            similarity, negative_index = max(candidates)
            neighbor = train[-negative_index]
            matches.append(
                {
                    "language": row["language"],
                    "heldout_text": row["text_clean"],
                    "heldout_label": row["label"],
                    "train_text": neighbor["text_clean"],
                    "train_label": neighbor["label"],
                    "jaccard": similarity,
                }
            )
    return {
        "method": "same-language character-trigram Jaccard; excludes exact matches",
        "threshold": threshold,
        "heldout_rows": len(heldout),
        "matched_rows": len(matches),
        "examples": sorted(
            matches, key=lambda r: (-r["jaccard"], r["heldout_text"], r["train_text"])
        )[:20],
    }


def audit(root):
    paths = {
        "challenge": root / "data/evaluation/english-challenge-v1.csv",
        "reviews": root / "data/evaluation/reviews/english-challenge-v1.json",
        "predictions": root / "reports/e5-small-run1/challenge-predictions.csv",
        "original_report": root / "reports/e5-small-run1/challenge-report.json",
        "export": root / "data/export/utterances.csv",
        **{s: root / f"data/splits/{s}.csv" for s in SPLITS},
    }
    challenge = reviewed_predictions(paths["challenge"], paths["reviews"], paths["predictions"])
    original = json.loads(paths["original_report"].read_text())
    if original["challenge_sha256"] != sha256(paths["challenge"]):
        raise ValueError("Original report refers to a different challenge")
    splits = {s: read_csv(paths[s]) for s in SPLITS}
    exported = read_csv(paths["export"])
    accepted = {r["utterance_id"]: r for r in exported if r["status"] == "accepted"}
    split_rows = [r for rows in splits.values() for r in rows]
    if len(split_rows) != len(accepted) or {r["utterance_id"] for r in split_rows} != set(accepted):
        raise ValueError("Splits do not contain each accepted row exactly once")
    for row in split_rows:
        if any(value != accepted[row["utterance_id"]][key] for key, value in row.items()):
            raise ValueError("Split content differs from accepted export")
    texts = {s: {r["text_clean"] for r in rows} for s, rows in splits.items()}
    seen = set.union(*texts.values())
    overlap = {
        f"{a}/{b}": len(texts[a] & texts[b])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    by_category = {
        c: score([r for r in challenge if r["category"] == c])
        for c in sorted({r["category"] for r in challenge})
    }
    return {
        "input_sha256": {k: sha256(p) for k, p in paths.items()},
        "checkpoint_sha256_from_original_report": original["model_sha256"],
        "protocol": {
            "predictions": "frozen saved probabilities; no inference or model change",
            "labels": "saved owner decisions, joined by case_id and checked against source hash",
            "purpose": "development/regression diagnostic, not representative accuracy",
            "length": "whitespace words; compare within a language, not across scripts",
            "threshold": 0.5,
            "threshold_sensitivity": "descriptive only; no threshold selected or deployed",
        },
        "reviewed_count": len(challenge),
        "review_changes_from_authored_labels": sum(
            r["label"] != r["authored_label"] for r in challenge
        ),
        "challenge": score(challenge),
        "by_category": by_category,
        "challenge_unseen_in_any_split": score(
            [r for r in challenge if r["text_clean"] not in seen]
        ),
        "challenge_overlap": {
            s: [r["case_id"] for r in challenge if r["text_clean"] in ts] for s, ts in texts.items()
        },
        "errors": [
            {k: r[k] for k in ("case_id", "category", "text_clean", "label", "p_barge_in")}
            for r in challenge
            if (r["p_barge_in"] >= 0.5) != (r["label"] == "barge_in")
        ],
        "threshold_sensitivity": [
            {
                "threshold": t,
                "all": score(challenge, t),
                "short": score([r for r in challenge if r["category"] == "short_interruption"], t),
            }
            for t in (0.5, 0.1, 0.02, 0.014, 0.0135, 0.0133)
        ],
        "export_status_counts": dict(sorted(Counter(r["status"] for r in exported).items())),
        "split_integrity": {
            "accepted_rows_match_splits": True,
            "cross_split_exact_overlap": overlap,
            "within_split_duplicate_rows": {s: len(splits[s]) - len(texts[s]) for s in SPLITS},
        },
        "coverage": {
            s: {
                "count": len(rows),
                "language_label": coverage(rows, ("language", "label")),
                "language_intent": coverage(rows, ("language", "label", "caller_intent")),
            }
            for s, rows in splits.items()
        },
        "near_duplicates_against_train": {
            s: near_duplicates(splits["train"], splits[s]) for s in ("validation", "test")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("reports/reviewed-error-audit-v1.json"))
    args = parser.parse_args()
    report = audit(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(args.out), "challenge": report["challenge"]}, indent=2))


if __name__ == "__main__":
    main()
