"""What an utterance criterion asks for, language by language, and the judge's
yes/no questions that check it (docs/generation.md#utterance-track).

The generator prompt and the judge read the same `language_parts`, so the
script the prompt asks for is the script the judge checks. The judge asks one
question per criterion of the specification and nothing else (user decision,
2026-11-15); the language and script criteria flatten into one question per
language in the combination. Length is checked in code.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from string import Template
from typing import Any

import yaml

from barge_in_detector.catalogue import Catalogue
from barge_in_detector.cleaner import words
from barge_in_detector.generation.prompts import PROMPT_DIR, SCRIPT_NAMES

JUDGE_TEMPLATE = "judge_utterances_v3"
_PUNCTUATION = ".,!?;:\"'()،。、，！？"
JUDGE_KIND = "judge_utterance"
YES = 0.5

_HAN = ("Hans", "Hant")

REGION_NAMES = {
    "AR": "Argentina",
    "AT": "Austria",
    "AU": "Australia",
    "BD": "Bangladesh",
    "BE": "Belgium",
    "BR": "Brazil",
    "CA": "Canada",
    "CH": "Switzerland",
    "CN": "mainland China",
    "CO": "Colombia",
    "DE": "Germany",
    "ES": "Spain",
    "FR": "France",
    "GB": "the United Kingdom",
    "HK": "Hong Kong",
    "IN": "India",
    "LK": "Sri Lanka",
    "MX": "Mexico",
    "MY": "Malaysia",
    "NG": "Nigeria",
    "NL": "the Netherlands",
    "PH": "the Philippines",
    "PK": "Pakistan",
    "PT": "Portugal",
    "SG": "Singapore",
    "SN": "Senegal",
    "TW": "Taiwan",
    "US": "the United States",
    "ZA": "South Africa",
}


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
    name = (
        cat.languages[code].get("script_name")
        if script == cat.languages[code]["scripts"][0]
        else None
    )
    label = name or SCRIPT_NAMES.get(script, script)
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
    never = [
        _script_label(cat, part.code, s)
        for s in cat.languages[part.code].get("never_scripts") or []
    ]
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
        "within_sentence": "switch languages within a sentence",
        "between_sentences": (
            "switch languages between sentences: each language in its own sentence, "
            "ended with a full stop"
        ),
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


@cache
def _template() -> dict[str, Any]:
    return yaml.safe_load((PROMPT_DIR / f"{JUDGE_TEMPLATE}.yaml").read_text(encoding="utf-8"))


def judge_questions(
    cat: Catalogue, spec: Mapping[str, str], text: str, barge_in_from_word: int | None
) -> tuple[str, dict[str, Any]]:
    """The Decisions API input ({"text": ...}) and one question per criterion."""
    q = _template()["questions"]

    def ask(raw: str, **fields: str) -> dict[str, Any]:
        return {"type": "noul", "instructions": Template(raw).substitute(fields)}

    parts = language_parts(cat, spec)
    switching = spec["code_switching"]
    questions: dict[str, Any] = {}
    for part in parts:
        questions[f"language_{part.code}"] = ask(q["language"], name=part.name)
    for part in parts:
        questions[f"script_{part.code}"] = ask(
            q["script"], name=part.name, script=script_name(cat, part)
        )
    region = cat.region(spec["language"], spec["regional_variant"])
    if "-" in spec["regional_variant"] and region in REGION_NAMES:
        questions["variant"] = ask(q["variant"], name=parts[0].name, region=REGION_NAMES[region])
    if switching != "none":
        questions["switching"] = ask(q["switching"][switching])
    intent = " ".join(cat.intents[spec["caller_intent"]]["description"].split())
    questions["intent"] = ask(q["intent"], description=intent)
    development = spec["development"]
    if development == "stable":
        questions["development"] = ask(q["development"]["stable"])
    else:
        tokens = words(text)
        word = tokens[min(barge_in_from_word or 0, len(tokens) - 1)] if tokens else ""
        questions["development"] = ask(q["development"][development], word=word.strip(_PUNCTUATION))
    questions["repetition"] = ask(q["repetition"][spec["repetition"]])
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
