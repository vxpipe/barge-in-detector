"""Prompt rendering. Prompts are code: templates live in generation/prompts/.

Each call has two parts:

- a **system prompt** with the common rules, rendered from the catalogue only.
  It is identical for every request of a template version, so it is cached
  (explicit Gemini context cache, or implicit prefix caching as a fallback);
- a **request** naming the criterion (or contrastive set) and the count.
"""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from barge_in_detector.catalogue import Catalogue

PROMPT_DIR = Path(__file__).resolve().parents[2] / "generation" / "prompts"
GENERATOR_TEMPLATE = "generator_system_v1"
JUDGE_TEMPLATE = "judge_system_v1"

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

SHARED_ELEMENTS = {
    "candidate_text": (
        "candidate_text is exactly the same in every member of a set; the context "
        "and agent turns differ as each member's factors require."
    ),
    "agent_turn": (
        "context_turn, agent_speaker, agent_full_text and agent_spoken_text are "
        "exactly the same in every member of a set; only the candidate line differs."
    ),
    "candidate_opening": (
        "candidate_text of every member starts with the same first word or words; "
        "after that the members differ as each one requires."
    ),
}

_TURN = {
    "type": "object",
    "properties": {
        "speaker": {"type": "string"},
        "role": {"type": "string", "enum": ["caller", "other"]},
        "text": {"type": "string"},
    },
    "required": ["speaker", "role", "text"],
}

GENERATOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "samples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "set_index": {"type": "integer"},
                    "member_index": {"type": "integer"},
                    "context_turn": _TURN,
                    "agent_speaker": {"type": "string"},
                    "agent_full_text": {"type": "string"},
                    "agent_spoken_text": {"type": "string"},
                    "candidate_speaker": {"type": "string"},
                    "candidate_role": {"type": "string", "enum": ["caller", "other"]},
                    "candidate_text": {"type": "string"},
                    "label": {"type": "string", "enum": ["backchannel", "barge_in", "ambiguous"]},
                    "barge_in_from_word": {"type": ["integer", "null"]},
                    "rationale": {"type": "string"},
                },
                "required": [
                    "set_index",
                    "member_index",
                    "context_turn",
                    "agent_speaker",
                    "agent_full_text",
                    "agent_spoken_text",
                    "candidate_speaker",
                    "candidate_role",
                    "candidate_text",
                    "label",
                    "barge_in_from_word",
                    "rationale",
                ],
            },
        }
    },
    "required": ["samples"],
}

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "label": {"type": "string", "enum": ["backchannel", "barge_in", "ambiguous"]},
                    "barge_in_from_word": {"type": ["integer", "null"]},
                    "language_ok": {"type": "boolean"},
                    "script_ok": {"type": "boolean"},
                    "cut_ok": {"type": "boolean"},
                    "natural_ok": {"type": "boolean"},
                    "notes": {"type": "string"},
                },
                "required": [
                    "index",
                    "label",
                    "barge_in_from_word",
                    "language_ok",
                    "script_ok",
                    "cut_ok",
                    "natural_ok",
                    "notes",
                ],
            },
        }
    },
    "required": ["items"],
}


def prompt_hash(system: str, user: str) -> str:
    return hashlib.sha256(f"{system}\x00{user}".encode()).hexdigest()


def _template(name: str) -> Template:
    return Template((PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8"))


def _value_list(cat: Catalogue, factor: str) -> str:
    return "\n".join(f"- `{v}`: {cat.descriptions.get((factor, v), v)}" for v in cat.values[factor])


def render_generator_system(cat: Catalogue) -> str:
    intents = "\n".join(
        f"- `{i}` ({data['label']}): {' '.join(data['description'].split())}"
        for i, data in cat.intents.items()
    )
    return _template(GENERATOR_TEMPLATE).substitute(
        intent_table=intents,
        agent_activities=_value_list(cat, "agent_activity"),
        cut_positions=_value_list(cat, "cut_position"),
        developments=_value_list(cat, "development"),
        length_buckets=_value_list(cat, "length_bucket"),
        script_forms=_value_list(cat, "script_form"),
        code_switching=_value_list(cat, "code_switching"),
    )


def render_judge_system(cat: Catalogue) -> str:
    return _template(JUDGE_TEMPLATE).substitute()


# ---------------------------------------------------------------- requests


def language_name(cat: Catalogue, code: str) -> str:
    return str(cat.languages[code]["name"])


def describe_language(cat: Catalogue, spec: Mapping[str, str]) -> list[str]:
    lang = cat.languages[spec["language"]]
    scripts = ", ".join(SCRIPT_NAMES.get(s, s) for s in lang["scripts"])
    lines = [
        f"Language: {lang['name']} ({spec['language']}), regional variety {spec['regional_variant']}. "
        f"Native script: {scripts}.",
        f"Script form: `{spec['script_form']}`: "
        f"{cat.descriptions.get(('script_form', spec['script_form']), '')}",
    ]
    switching = spec["code_switching"]
    if switching == "none":
        lines.append("Code-switching: `none`: only this language, no other languages mixed in.")
    else:
        others = " and ".join(language_name(cat, c) for c in spec["code_switch_set"].split("+"))
        lines.append(
            f"Code-switching: `{switching}`: "
            f"{cat.descriptions.get(('code_switching', switching), '')} Mixed in: {others}."
        )
    return lines


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


def describe_spec(
    cat: Catalogue, spec: Mapping[str, str], *, include_language: bool = True
) -> list[str]:
    d = cat.descriptions
    domain = cat.domains[spec["domain"]]
    intent = cat.intents[spec["caller_intent"]]
    state = "just_finished" if spec["cut_position"] == "just_finished" else "speaking"
    lines = describe_language(cat, spec) if include_language else []
    lines += [
        f"Domain: `{spec['domain']}`: {domain['description']}",
        f"Scenario: `{spec['scenario']}`: {cat.scenarios[spec['scenario']]['description']}",
    ]
    if domain.get("prompt_notes"):
        lines.append(f"Domain notes: {' '.join(domain['prompt_notes'].split())}")
    lines += [
        f"Speaker pattern: `{spec['speaker_pattern']}`: {d[('speaker_pattern', spec['speaker_pattern'])]}",
        f"Agent activity: `{spec['agent_activity']}`: {d[('agent_activity', spec['agent_activity'])]}",
        f"Cut position: `{spec['cut_position']}`: {d[('cut_position', spec['cut_position'])]} "
        f"(agent state: {state})",
        f"Caller intent: `{spec['caller_intent']}` (intended label: {intent['label']}): "
        f"{' '.join(intent['description'].split())}",
        _hints(cat, spec["caller_intent"], spec["language"]),
        f"Development: `{spec['development']}`: {d[('development', spec['development'])]}",
        f"Repetition: `{spec['repetition']}`",
        f"Length of candidate_text: `{spec['length_bucket']}`: {d[('length_bucket', spec['length_bucket'])]}",
        f"Register: `{spec['register']}`. Caller mood: `{spec['caller_mood']}`.",
    ]
    return lines


def render_generation_request(
    cat: Catalogue,
    specs: Sequence[Mapping[str, str]],
    *,
    count: int,
    group: Mapping[str, Any] | None = None,
) -> str:
    """The per-request prompt. `count` is samples, or sets for a contrastive group."""
    if group is None:
        (spec,) = specs
        body = "\n".join(f"- {line}" for line in describe_spec(cat, spec))
        return (
            f"Write {count} samples for this specification. Set set_index and "
            f"member_index to 0.\n\n{body}\n"
        )

    members = group["members"]
    varied = {k for m in members for k in m}
    shared = [f for f in cat.spec_factors if f not in varied]
    shared_spec = {f: specs[0][f] for f in shared}
    shared_lines = describe_language(cat, specs[0]) + [
        line
        for line in describe_spec(cat, specs[0], include_language=False)
        if any(line.startswith(prefix) for prefix in _prefixes_for(shared_spec))
    ]
    parts = [
        f"Write {count} sets of contrastive samples ({count * len(members)} samples in total). "
        f"Each set has {len(members)} samples, one per member below, in member order. "
        f"Give every sample of set k set_index k (0-based) and its member's member_index.",
        f"Contrast: {' '.join(group['description'].split())}",
        f"Shared within a set: {SHARED_ELEMENTS[group['shared']]}",
        "",
        "Factors shared by all members:",
        *(f"- {line}" for line in shared_lines),
    ]
    for index, spec in enumerate(specs):
        lines = [
            line
            for line in describe_spec(cat, spec, include_language=False)
            if not any(line.startswith(p) for p in _prefixes_for(shared_spec))
        ]
        parts += ["", f"Member with member_index {index}:", *(f"- {line}" for line in lines)]
    return "\n".join(parts) + "\n"


_FACTOR_PREFIXES = {
    "domain": "Domain:",
    "scenario": "Scenario:",
    "speaker_pattern": "Speaker pattern:",
    "agent_activity": "Agent activity:",
    "cut_position": "Cut position:",
    "caller_intent": "Caller intent:",
    "development": "Development:",
    "repetition": "Repetition:",
    "length_bucket": "Length of candidate_text:",
    "register": "Register:",
}


def _prefixes_for(shared_spec: Mapping[str, str]) -> list[str]:
    prefixes = [p for f, p in _FACTOR_PREFIXES.items() if f in shared_spec]
    if "domain" in shared_spec:
        prefixes.append("Domain notes:")
    if "caller_intent" in shared_spec:
        prefixes += ["Example hints", "English example hints", "No example hints"]
    return prefixes


@dataclass(frozen=True)
class JudgeItem:
    index: int
    spec: Mapping[str, str]
    context_speaker: str
    context_text: str
    agent_speaker: str
    agent_spoken_text: str
    candidate_speaker: str
    candidate_text: str


def render_judge_request(cat: Catalogue, items: Sequence[JudgeItem]) -> str:
    parts = [f"Review these {len(items)} items. Return one entry per item with its index."]
    for item in items:
        spec = item.spec
        state = "just_finished" if spec["cut_position"] == "just_finished" else "speaking"
        if spec["code_switching"] == "none":
            switching = "no code-switching"
        else:
            others = " and ".join(language_name(cat, c) for c in spec["code_switch_set"].split("+"))
            switching = f"code-switching ({spec['code_switching']}) with {others}"
        marker = (
            "[the agent had just finished this turn]"
            if state == "just_finished"
            else "[cut off here: the last line was said while the agent was still speaking]"
        )
        parts += [
            "",
            f"Item {item.index}",
            f"Meant to show: {language_name(cat, spec['language'])} ({spec['regional_variant']}); "
            f"script form {spec['script_form']}; {switching}; agent state {state}.",
            "Transcript:",
            f"{item.context_speaker}: {item.context_text}",
            f"{item.agent_speaker}: {item.agent_spoken_text} {marker}",
            f"{item.candidate_speaker}: {item.candidate_text}",
        ]
    return "\n".join(parts) + "\n"
