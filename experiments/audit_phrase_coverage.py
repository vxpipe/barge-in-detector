"""Record checklist coverage and gaps, original-data preservation and costs.

uv run python -m experiments.audit_phrase_coverage
No model/API calls. The inventory is a synthetic candidate checklist, not a
claim of exhaustive lexical coverage or native-speaker verification.
"""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import duckdb

from barge_in_detector.catalogue import load_catalogue
from barge_in_detector.cleaner import clean_text
from barge_in_detector.generation.phrase_coverage import build_targets, load_plan
from barge_in_detector.generation.phrase_runner import exclusions, request_reserve
from barge_in_detector.splitting import SPLITS, read_csv

OUT = Path("reports/phrase-coverage-v1")
SOURCE = Path("data/supplements/phrase-coverage-v1/utterances.csv")


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    cat, plan = load_catalogue(), load_plan()
    targets = {str(t["combination_id"]): t for t in build_targets(cat, plan)}
    rows = read_csv(SOURCE)
    original = read_csv(Path("data/export/utterances.csv"))
    original = {r["text_clean"]: r for r in original if r["status"] == "accepted"}
    prior_splits = {
        r["text_clean"]: s for s in SPLITS for r in read_csv(Path("data/splits") / f"{s}.csv")
    }
    excluded = exclusions()
    manifest = json.loads((OUT / "original-source-manifest.json").read_text())
    assert all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p, sha in manifest.items()
    )
    accepted = [r for r in rows if r["status"] == "accepted"]
    assert len({r["text_clean"] for r in accepted}) == len(accepted)
    assert not {r["text_clean"] for r in accepted}.intersection(excluded)
    assert all(
        not r["judge_request_id"] and not r["judge_model_id"] and not r["judge_answers"]
        for r in rows
    )
    by_criterion = defaultdict(list)
    expressions = {}
    conflicts = []
    for row in rows:
        by_criterion[row["combination_id"]].append(row)
        raw = json.loads(row["raw_sample"])
        sample, spec = raw["sample"], raw["coverage"]
        if (
            row["reason"] == "duplicate_original"
            and original[row["text_clean"]]["label"] != row["label"]
        ):
            conflicts.append(
                {
                    "text_clean": row["text_clean"],
                    "original_label": original[row["text_clean"]]["label"],
                    "proposed_label": row["label"],
                    "language": row["language"],
                    "utterance_id": row["utterance_id"],
                }
            )
        if row["status"] == "retired" or row["reason"] not in (
            "",
            "duplicate_original",
            "duplicate_supplement",
            "reserved_evaluation",
        ):
            continue
        if not isinstance(sample, dict) or not sample.get("anchor"):
            continue
        anchor = clean_text(sample["anchor"])
        key = (row["language"], row["regional_variant"], row["code_switch_set"], anchor)
        if key not in expressions:
            expected = [
                f
                for f, definition in plan["forms"].items()
                if row["code_switching"] in definition["switching"]
            ]
            expressions[key] = {
                "language": key[0],
                "regional_variant": key[1],
                "code_switch_set": key[2],
                "anchor": anchor,
                "observed_forms": set(),
                "accepted_forms": set(),
                "expected_forms": expected,
                "source_ids": [],
                "original_exact_split": prior_splits.get(anchor, ""),
                "evaluation_reserved": excluded.get(anchor) == "reserved_evaluation",
            }
        item = expressions[key]
        item["observed_forms"].add(spec["coverage_form"])
        if row["status"] == "accepted":
            item["accepted_forms"].add(spec["coverage_form"])
        item["source_ids"].append(row["utterance_id"])
    expression_rows = []
    for _key, item in sorted(expressions.items()):
        missing = set(item["expected_forms"]) - item["observed_forms"]
        expression_rows.append(
            {
                **item,
                "observed_forms": "|".join(sorted(item["observed_forms"])),
                "accepted_forms": "|".join(sorted(item["accepted_forms"])),
                "missing_forms": "|".join(sorted(missing)),
                "source_ids": "|".join(item["source_ids"]),
                "native_verified": False,
            }
        )
    write_csv(
        OUT / "expression-checklist.csv",
        expression_rows,
        [
            "language",
            "regional_variant",
            "code_switch_set",
            "anchor",
            "original_exact_split",
            "evaluation_reserved",
            "observed_forms",
            "accepted_forms",
            "missing_forms",
            "native_verified",
            "source_ids",
        ],
    )
    checklist = []
    for cid, target in targets.items():
        proposed = by_criterion[cid]
        kept = [r for r in proposed if r["status"] == "accepted"]
        credited = {
            r["text_clean"]
            for r in proposed
            if r["status"] == "accepted"
            or r["reason"] in ("duplicate_original", "duplicate_supplement", "reserved_evaluation")
        }
        checklist.append(
            {
                **target,
                "accepted": len(kept),
                "valid_distinct_proposals": len(credited),
                "remaining": max(0, target["required_count"] - len(kept)),
                "reasons": json.dumps(
                    dict(Counter(r["reason"] or r["status"] for r in proposed)), sort_keys=True
                ),
                "examples": " | ".join(r["text"] for r in kept),
            }
        )
    write_csv(
        OUT / "criterion-checklist.csv",
        checklist,
        [
            "combination_id",
            "language",
            "regional_variant",
            "code_switching",
            "code_switch_set",
            "caller_intent",
            "anchor_intent",
            "coverage_form",
            "required_count",
            "accepted",
            "valid_distinct_proposals",
            "remaining",
            "reasons",
            "examples",
        ],
    )
    write_csv(
        OUT / "label-conflicts.csv",
        conflicts,
        ["utterance_id", "language", "text_clean", "original_label", "proposed_label"],
    )
    con = duckdb.connect("data/work/phrase-coverage-v1.duckdb", read_only=True)
    costs = con.execute(
        "SELECT count(*), coalesce(sum(cost_usd), 0), count(*) FILTER (WHERE cost_usd IS NULL), "
        "max(input_tokens), max(output_tokens) FROM requests"
    ).fetchone()
    kinds = dict(con.execute("SELECT kind, count(*) FROM requests GROUP BY 1").fetchall())
    errors = con.execute(
        "SELECT error, count(*) FROM requests WHERE status = 'api_error' GROUP BY 1"
    ).fetchall()
    con.close()
    committed = costs[1] + costs[2] * request_reserve(plan)
    assert committed <= plan["new_budget_usd"]
    language_rows = []
    for language in plan["languages"]:
        items = [r for r in accepted if r["language"] == language]
        checks = [r for r in checklist if r["language"] == language]
        language_rows.append(
            {
                "language": language,
                "accepted": len(items),
                "backchannel": sum(r["label"] == "backchannel" for r in items),
                "barge_in": sum(r["label"] == "barge_in" for r in items),
                "criteria": len(checks),
                "criteria_with_samples": sum(r["accepted"] > 0 for r in checks),
                "criteria_full": sum(r["remaining"] == 0 for r in checks),
                "anchors": sum(key[0] == language for key in expressions),
            }
        )
    report = {
        "original_files_preserved": True,
        "original_accepted": len(original),
        "supplement_accepted": len(accepted),
        "statuses": dict(Counter(r["status"] for r in rows)),
        "rejections": dict(Counter(r["reason"] for r in rows if r["status"] != "accepted")),
        "languages": language_rows,
        "criteria": len(targets),
        "criteria_full": sum(r["remaining"] == 0 for r in checklist),
        "criteria_with_samples": sum(r["accepted"] > 0 for r in checklist),
        "candidate_expression_entries": len(expressions),
        "candidate_expressions_missing_forms": sum(
            bool(r["missing_forms"]) for r in expression_rows
        ),
        "label_conflicts_for_review": len(conflicts),
        "supplement_exact_overlap_original_or_evaluation": 0,
        "judge_calls": 0,
        "request_kinds": kinds,
        "requests": costs[0],
        "usage_based_cost_usd": costs[1],
        "unknown_cost_requests_reserved": costs[2],
        "cost_including_reservations_usd": committed,
        "historical_plus_new_usage_usd": plan["historical_recorded_usd"] + costs[1],
        "total_with_all_reserves_usd": plan["historical_recorded_usd"]
        + plan["historical_uncertainty_reserve_usd"]
        + committed,
        "observed_max_input_tokens": costs[3],
        "observed_max_output_tokens": costs[4],
        "request_errors": errors,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "limitations": "Synthetic candidate checklist, not exhaustive coverage. Local checks only; no judge or native verification. Exact deduplication does not prevent semantic near-duplicate overlap.",
    }
    (OUT / "coverage.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
