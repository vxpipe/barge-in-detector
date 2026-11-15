"""Deterministic phrase checklist, prompts and local screening (no judge).

Coverage-only factors live in raw_sample.coverage, so the original schema-11
dataset and its CSV remain byte-for-byte unchanged. They participate in UUIDv5
criterion IDs, and are defined by the separate versioned phrase plan.
"""

import json
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from barge_in_detector import ids
from barge_in_detector.catalogue import LANGUAGE_FACTORS, Catalogue
from barge_in_detector.cleaner import clean_text
from barge_in_detector.generation.prompts import PROMPT_DIR, UTTERANCE_SCHEMA
from barge_in_detector.generation.utterance_criteria import describe_languages, language_parts
from barge_in_detector.storage.schema import UTTERANCE_FACTORS

PLAN_PATH = Path("generation/plans/phrase-coverage-v1.yaml")
TEMPLATE = "generator_phrases_v2"
KIND = "generate_phrases_v1"
SCHEMA = json.loads(json.dumps(UTTERANCE_SCHEMA))
SCHEMA["properties"]["samples"]["items"]["properties"]["anchor"] = {"type": "string"}
SCHEMA["properties"]["samples"]["items"]["required"].append("anchor")


def load_plan(path: Path = PLAN_PATH) -> dict[str, Any]:
    plan = yaml.safe_load(path.read_text())
    if plan["required_count"] < 1 or plan["max_attempts"] < 1:
        raise ValueError("Counts and attempts must be positive")
    if plan["new_budget_usd"] < 0 or (
        plan["historical_recorded_usd"]
        + plan["historical_uncertainty_reserve_usd"]
        + plan["new_budget_usd"]
        > plan["total_ceiling_usd"]
    ):
        raise ValueError("Plan exceeds the total expense ceiling")
    return plan


def build_targets(cat: Catalogue, plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    sides = [
        dict(zip(LANGUAGE_FACTORS, (language, *option), strict=True))
        for language in plan["languages"]
        for option in cat.language_options(language)
    ]
    targets = []

    def add(side, anchor_intent, form, intent, description):
        spec = {**side, "caller_intent": intent}
        problems = cat.violations(spec, partial_spec=True)
        if problems:
            raise ValueError(problems)
        spec.update(coverage_form=form, anchor_intent=anchor_intent)
        targets.append(
            {
                **spec,
                "combination_id": ids.utterance_combination_id(spec),
                "label": cat.intents[intent]["label"],
                "locales": cat.locales(side),
                "required_count": plan["required_count"],
                "description": description,
            }
        )

    # Cover every language for a form before spending on the next form.
    for form, definition in plan["forms"].items():
        for side in sides:
            if side["code_switching"] not in definition["switching"]:
                continue
            for anchor_intent in plan["anchor_intents"]:
                add(
                    side,
                    anchor_intent,
                    form,
                    definition.get("caller_intent", anchor_intent),
                    definition["description"],
                )
    for side in sides:
        for intent in plan["short_intents"]:
            add(
                side,
                "default",
                "short_interruption",
                intent,
                "A very short direct interruption, with no backchannel prefix. Use the shortest "
                "natural wording that expresses this intent clearly. Set anchor to empty string.",
            )
    return targets


def coverage_spec(target: Mapping[str, Any]) -> dict[str, str]:
    return {k: target[k] for k in (*UTTERANCE_FACTORS, "coverage_form", "anchor_intent")}


def render(cat: Catalogue, target: Mapping[str, Any], count: int) -> tuple[str, str]:
    anchor = target["anchor_intent"]
    lines = [
        f"Write {count} phrases for this criterion.",
        describe_languages(cat, target),
        f"Form: {target['coverage_form']}. {target['description']}",
        f"Required label: {target['label']}.",
        "Complete utterance intent: " + cat.intents[target["caller_intent"]]["description"],
    ]
    if anchor != "default":
        lines.append("Anchor intent: " + cat.intents[anchor]["description"])
    return (PROMPT_DIR / f"{TEMPLATE}.md").read_text(), "\n".join(lines)


_SCRIPT_NAMES = {
    "Latn": ("LATIN",),
    "Deva": ("DEVANAGARI",),
    "Knda": ("KANNADA",),
    "Taml": ("TAMIL",),
    "Telu": ("TELUGU",),
    "Mlym": ("MALAYALAM",),
    "Hans": ("CJK",),
    "Jpan": ("CJK", "HIRAGANA", "KATAKANA"),
}


def sample_problem(cat: Catalogue, target: Mapping[str, Any], sample: Any) -> str | None:
    if not isinstance(sample, dict) or not all(
        isinstance(sample.get(k), str) for k in ("text", "label", "anchor", "rationale")
    ):
        return "malformed_sample"
    if sample["label"] != target["label"]:
        return "label_mismatch"
    text, anchor = clean_text(sample["text"]), clean_text(sample["anchor"])
    if not text:
        return "empty_after_cleaning"
    if any(mark in sample["text"] for mark in ("…", "...", "[", "]", "(", ")", "<", ">")):
        return "tag_or_ellipsis"
    scripts = {_SCRIPT_NAMES[p.script] for p in language_parts(cat, target)}
    letters = [unicodedata.name(c, "") for c in text if unicodedata.category(c).startswith("L")]
    if any(not any(any(s in name for s in group) for group in scripts) for name in letters):
        return "wrong_script"
    if any(not any(any(s in name for s in group) for name in letters) for group in scripts):
        return "wrong_script"
    form = target["coverage_form"]
    if form == "short_interruption":
        return None if not anchor else "unexpected_anchor"
    if not anchor or not text.startswith(anchor):
        return "anchor_not_prefix"
    if form == "core" and text != anchor:
        return "core_not_anchor"
    if form == "repetition":
        compact, root = text.replace(" ", ""), anchor.replace(" ", "")
        if compact not in (root * 2, root * 3, root * 4):
            return "not_a_repetition"
    elif form != "core" and text == anchor:
        return "missing_extension"
    if len(text) > 500 or len(text.split()) > 30:
        return "too_long"
    return None
