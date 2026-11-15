"""What an utterance criterion asks for, language by language, and the judge's
yes/no questions that check it (docs/generation.md#utterance-track).

The generator prompt and the judge read the same `language_parts`. The judge's
questions restate the request's criteria and nothing else (user decision,
2026-11-15): one per language line ("Does this text contain Tamil words
written in Tamil script?") and one for the intent.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from barge_in_detector.catalogue import Catalogue
from barge_in_detector.generation.prompts import SCRIPT_NAMES

JUDGE_TEMPLATE = "criteria_questions_v1"  # questions built in code from the criteria
SAID_OVER = "Said over a speaker who is still talking"
JUDGE_KIND = "judge_utterance"
YES = 0.5

_HAN = ("Hans", "Hant")


@dataclass(frozen=True)
class LanguagePart:
    """One language of a criterion and how its words are written."""

    code: str
    name: str
    main: bool
    script: str  # ISO 15924 code, or "mixed" (partly native, partly romanized)
    native: str  # the language's own native script in this region
    romanized: bool  # a non-Latin language written in Latin letters
    host_script: str | None = None  # for a mixed-in "mixed": the main language's script


def _script_label(cat: Catalogue, code: str, script: str) -> str:
    label = SCRIPT_NAMES.get(script, script)
    return f"{label} characters" if script in _HAN else f"{label} script"


def language_parts(cat: Catalogue, spec: Mapping[str, str]) -> list[LanguagePart]:
    """The main language first, then each mixed-in language in set order."""
    code, form = spec["language"], spec["script_form"]
    region = cat.region(code, spec["regional_variant"])
    native = cat.native_script(code, region)
    if cat.latin(code):
        script, romanized = "Latn", False
    elif form == "romanized":
        script, romanized = "Latn", True
    elif form == "mixed_script":
        script, romanized = "mixed", False
    else:
        script, romanized = native, False
    name = str(cat.languages[code]["name"])
    parts = [LanguagePart(code, name, True, script, native, romanized)]
    for other, other_script in cat.mixed_in_script_codes(spec).items():
        other_native = cat.native_script(other, region)
        parts.append(
            LanguagePart(
                code=other,
                name=str(cat.languages[other]["name"]),
                main=False,
                script=other_script,
                native=other_native,
                romanized=other_script == "Latn" and not cat.latin(other),
                host_script=native if other_script == "mixed" else None,
            )
        )
    return parts


def script_name(cat: Catalogue, part: LanguagePart) -> str:
    """The script a part's words are written in: "Devanagari script",
    "romanized Latin letters", "both Devanagari script and Latin letters"."""
    if part.romanized:
        return "romanized Latin letters"
    if part.script == "Latn":
        return "Latin letters"
    if part.script == "mixed":
        if part.main:
            return f"both {_script_label(cat, part.code, part.native)} and Latin letters"
        return f"{_script_label(cat, part.code, part.host_script or part.native)} or Latin letters"
    return _script_label(cat, part.code, part.script)


def script_phrase(cat: Catalogue, part: LanguagePart) -> str:
    """The prompt's script instruction: the script, plus scripts to avoid."""
    phrase = f"in {script_name(cat, part)}"
    if part.script == "mixed":
        return phrase + (", within the same utterance" if part.main else "")
    never: list[str] = []
    if part.romanized:
        phrase += " as people type it in messages (no tone numbers or tone marks)"
        never.insert(0, _script_label(cat, part.code, part.native))
    elif part.script != "Latn":
        never.append("Latin letters")
    return phrase + (", never in " + " or ".join(never) if never else "")


def describe_languages(cat: Catalogue, spec: Mapping[str, str]) -> str:
    """The generator prompt's language block: one indented line per language."""
    switching = {
        "none": "this language only",
        "mixed": "mix these languages in the utterance",
    }[spec["code_switching"]]
    lines = [
        "Languages and scripts (every language listed here must appear in every "
        f"utterance; {switching}):"
    ]
    for part in language_parts(cat, spec):
        variety = (
            f" (regional variety {spec['regional_variant']})"
            if part.main and "-" in spec["regional_variant"]
            else ""
        )
        lines.append(f"  - {part.name}{variety}: {script_phrase(cat, part)}.")
    return "\n".join(lines)


# ---------------------------------------------------------------- judge


def judge_questions(
    cat: Catalogue, spec: Mapping[str, str], text: str
) -> tuple[str, dict[str, Any]]:
    """The Decisions API input ({"text": ...}) and the questions: the request's
    criteria restated, one per language line and one for the intent."""

    def ask(question: str) -> dict[str, Any]:
        return {"type": "noul", "instructions": question}

    questions = {
        f"language_{part.code}": ask(
            f"Does this text contain {part.name} words written in {script_name(cat, part)}?"
        )
        for part in language_parts(cat, spec)
    }
    intent = " ".join(cat.intents[spec["caller_intent"]]["description"].split())
    questions["intent"] = ask(f"{SAID_OVER}, is this text: {intent}")
    return json.dumps({"text": text}, ensure_ascii=False), questions


class JudgementError(ValueError):
    """The judge response is unusable."""


@dataclass(frozen=True)
class Verdict:
    answers: dict[str, float]  # question key -> P(yes)
    failed: list[str]  # keys answered no, in question order

    def reason(self) -> str:
        return ",".join(
            f"{k}={self.answers[k]:.2f}".rstrip("0").rstrip(".") if k in self.answers else k
            for k in self.failed
        )


def parse_verdict(response: Mapping[str, Any], keys: Sequence[str]) -> Verdict:
    """Every question must be answered yes (P(yes) >= 0.5); a blocked input fails."""
    if response.get("blocked"):
        return Verdict({}, [f"blocked:{response['blocked']}"])
    try:
        answers = response["answers"]
        p = {k: float(answers[k]["noul"]) for k in keys}
    except (KeyError, TypeError, ValueError) as exc:
        raise JudgementError(f"incomplete decision: {type(exc).__name__}: {exc}") from exc
    return Verdict(p, [k for k in keys if p[k] < YES])
