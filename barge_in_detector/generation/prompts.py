"""Prompt rendering for the generator. Prompts are code: templates live in
generation/prompts/.

A generator call has two parts:

- a **system prompt** with the common rules, rendered from the catalogue and
  the track only. It is identical for every request of a template version, so
  it is cached (explicit Gemini context cache, or implicit prefix caching);
- a **request** naming the criterion and the count.

The judge's questions are in utterance_criteria.py.
"""

import hashlib
from collections.abc import Mapping
from pathlib import Path
from string import Template
from typing import Any

from barge_in_detector.catalogue import Catalogue

PROMPT_DIR = Path(__file__).resolve().parents[2] / "generation" / "prompts"

SCRIPT_NAMES = {
    "Latn": "Latin",
    "Cyrl": "Cyrillic",
    "Arab": "Arabic",
    "Deva": "Devanagari",
    "Ethi": "Ethiopic",
    "Beng": "Bengali-Assamese",
    "Thai": "Thai",
    "Telu": "Telugu",
    "Taml": "Tamil",
    "Sinh": "Sinhala",
    "Orya": "Odia",
    "Mymr": "Myanmar",
    "Mlym": "Malayalam",
    "Laoo": "Lao",
    "Kore": "Hangul",
    "Knda": "Kannada",
    "Khmr": "Khmer",
    "Jpan": "Japanese (kanji and kana)",
    "Hebr": "Hebrew",
    "Hant": "Traditional Chinese",
    "Hans": "Simplified Chinese",
    "Guru": "Gurmukhi",
    "Gujr": "Gujarati",
    "Grek": "Greek",
    "Geor": "Georgian",
    "Armn": "Armenian",
}


def prompt_hash(system: str, user: str) -> str:
    return hashlib.sha256(f"{system}\x00{user}".encode()).hexdigest()


def _template(name: str) -> Template:
    return Template((PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8"))


def _value_list(cat: Catalogue, factor: str) -> str:
    return "\n".join(f"- `{v}`: {cat.descriptions.get((factor, v), v)}" for v in cat.values[factor])


def language_name(cat: Catalogue, code: str) -> str:
    return str(cat.languages[code]["name"])


def _hints(cat: Catalogue, intent: str, language: str) -> str:
    examples = cat.intents[intent].get("examples") or {}
    if language in examples:
        hints = examples[language]
        return "Example hints (kinds of words only; vary them): " + ", ".join(
            f'"{h}"' for h in hints
        )
    if "en" in examples:
        hints = ", ".join(f'"{h}"' for h in examples["en"])
        return (
            f"English example hints, to be expressed naturally in "
            f"{language_name(cat, language)} (kinds of words only): {hints}"
        )
    return "No example hints: rely on the intent description."


UTTERANCE_TEMPLATE = "generator_utterances_v3"
_REPETITION = {
    "repeated": 'say a key word or short phrase two or more times ("wait wait", "haan haan").',
    "single": 'no word or phrase is said twice (not "yes yes").',
}
UTTERANCE_REQUEST_KIND = "generate_utterances"

_UTTERANCE_FIELDS: dict[str, Any] = {
    "text": {"type": "string"},
    "label": {"type": "string", "enum": ["backchannel", "barge_in"]},
    "barge_in_from_word": {"type": ["integer", "null"]},
    "rationale": {"type": "string"},
}

UTTERANCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "samples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": _UTTERANCE_FIELDS,
                "required": list(_UTTERANCE_FIELDS),
            },
        }
    },
    "required": ["samples"],
}


def render_utterance_system(cat: Catalogue, track: Any) -> str:
    """The cached system prompt of the utterance track (track: UtteranceTrack)."""
    intents = "\n".join(
        f"- `{i}` ({cat.intents[i]['label']}): {' '.join(cat.intents[i]['description'].split())}"
        for i in track.intents
    )
    return _template(UTTERANCE_TEMPLATE).substitute(
        intent_table=intents,
        developments=_value_list(cat, "development"),
        length_buckets=_value_list(cat, "length_bucket"),
        script_forms=_value_list(cat, "script_form"),
        code_switching=_value_list(cat, "code_switching"),
    )


def render_utterance_request(cat: Catalogue, spec: Mapping[str, str], *, count: int) -> str:
    from barge_in_detector.generation.utterance_criteria import describe_languages

    d = cat.descriptions
    intent = cat.intents[spec["caller_intent"]]
    lines = [describe_languages(cat, spec)] + [
        f"Intent: `{spec['caller_intent']}` (intended label: {intent['label']}): "
        f"{' '.join(intent['description'].split())}",
        _hints(cat, spec["caller_intent"], spec["language"]),
        f"Development: `{spec['development']}`: {d[('development', spec['development'])]}",
        f"Repetition: `{spec['repetition']}`: {_REPETITION[spec['repetition']]}",
        f"Length: `{spec['length_bucket']}`: {d[('length_bucket', spec['length_bucket'])]}",
    ]
    body = "\n".join(f"- {line}" for line in lines)
    head = (
        "Write 1 utterance for this specification, said over a speaker who is still talking."
        if count == 1
        else f"Write {count} utterances for this specification, each said over a speaker "
        "who is still talking, each in a different situation."
    )
    return f"{head}\n\n{body}\n"
