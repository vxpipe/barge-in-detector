"""Prompt rendering. Prompts are code: templates live in generation/prompts/.

Generator calls have two parts:

- a **system prompt** with the common rules, rendered from the catalogue only.
  It is identical for every request of a template version, so it is cached
  (explicit Gemini context cache, or implicit prefix caching as a fallback);
- a **request** naming the criterion (or contrastive set) and the count.

The judge uses the Gemini Decisions API instead: one request per sample with
that sample as input and the questions from judge_decisions_v1.yaml.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

import yaml

from barge_in_detector.catalogue import Catalogue
from barge_in_detector.cleaner import words

PROMPT_DIR = Path(__file__).resolve().parents[2] / "generation" / "prompts"
GENERATOR_TEMPLATE = "generator_system_v4"
JUDGE_TEMPLATE = "judge_decisions_v4"

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
        "listener_text is exactly the same in every member of a set; the context "
        "and speaker turns differ as each member's factors require."
    ),
    "speaker_turn": (
        "context_turn, speaker, speaker_full_text and speaker_spoken_text are "
        "exactly the same in every member of a set; only the listener's line differs."
    ),
    "candidate_opening": (
        "listener_text of every member starts with the same first word or words; "
        "after that the members differ as each one requires."
    ),
}

_SAMPLE_FIELDS: dict[str, Any] = {
    "set_index": {"type": "integer"},
    "member_index": {"type": "integer"},
    "context_turn": {
        "type": "object",
        "properties": {"speaker": {"type": "string"}, "text": {"type": "string"}},
        "required": ["speaker", "text"],
    },
    "speaker": {"type": "string"},
    "speaker_full_text": {"type": "string"},
    "speaker_spoken_text": {"type": "string"},
    "listener": {"type": "string"},
    "listener_text": {"type": "string"},
    "label": {"type": "string", "enum": ["backchannel", "barge_in", "ambiguous"]},
    "barge_in_from_word": {"type": ["integer", "null"]},
    "rationale": {"type": "string"},
}

GENERATOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "samples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": _SAMPLE_FIELDS,
                "required": list(_SAMPLE_FIELDS),
            },
        }
    },
    "required": ["samples"],
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


# ---------------------------------------------------------------- requests


def language_name(cat: Catalogue, code: str) -> str:
    return str(cat.languages[code]["name"])


def _script_names(cat: Catalogue, code: str) -> str:
    return " or ".join(SCRIPT_NAMES.get(s, s) for s in cat.languages[code]["scripts"])


def _latin(cat: Catalogue, code: str) -> bool:
    return cat.languages[code]["scripts"][0] == "Latn"


_HAN = ("Hans", "Hant")


def _in_script(cat: Catalogue, name: str, script: str) -> str:
    label = SCRIPT_NAMES.get(script, script)
    return f"{name} in {label} characters" if script in _HAN else f"{name} in {label} script"


def script_expectation(cat: Catalogue, spec: Mapping[str, str]) -> str:
    """Which script each language's words are written in, stated explicitly.

    The conversation language follows the script form. Mixed-in languages
    follow catalogue.mixed_in_script_codes: Latin-script languages in Latin
    letters; a set's mixed_in_script choice (Chinese characters or pinyin)
    when it has one; otherwise romanized in Latin-led speech and in the
    conversation's script in native-script speech.
    """
    code, form = spec["language"], spec["script_form"]
    region = cat.region(code, spec["regional_variant"])
    name, scripts = language_name(cat, code), _script_names(cat, code)
    native = cat.native_script(code, region)
    romanization = cat.languages[code].get("romanization")
    if form == "native":
        main = f"{name} in Latin letters" if _latin(cat, code) else _in_script(cat, name, native)
    elif form == "romanized":
        main = (
            f"{name} in {romanization} (Latin letters), not {SCRIPT_NAMES.get(native, native)}"
            if romanization
            else f"{name} romanized (Latin letters), not {scripts}"
        )
    else:
        main = (
            f"{name} partly in {scripts} script and partly romanized (Latin letters) "
            "within the same utterance"
        )
    parts = [main]
    choice = spec.get("mixed_in_script", "default")
    for other, script in cat.mixed_in_script_codes(spec).items():
        other_name = language_name(cat, other)
        other_native = cat.native_script(other, region)
        other_rom = cat.languages[other].get("romanization")
        if _latin(cat, other):
            parts.append(f"{other_name} words in Latin letters")
        elif script == "Latn" and choice == "romanized" and other_rom:
            never = (
                "Chinese characters"
                if other_native in _HAN
                else f"{_script_names(cat, other)} script"
            )
            parts.append(f"{other_name} words in {other_rom} (Latin letters), never in {never}")
        elif script == "Latn":
            parts.append(
                f"{other_name} words romanized (Latin letters), never in "
                f"{_script_names(cat, other)} script"
            )
        elif choice == "native":
            parts.append(
                _in_script(cat, f"{other_name} words", script) + f", not {other_rom or 'romanized'}"
            )
        elif script == "mixed":
            parts.append(f"{other_name} words in {scripts} script or romanized")
        else:
            parts.append(f"{other_name} words written in {scripts} script")
    return "; ".join(parts) + "."


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
    lines.append(f"Script: {script_expectation(cat, spec)}")
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
        f"Speaker activity: `{spec['agent_activity']}`: {d[('agent_activity', spec['agent_activity'])]}",
        f"Cut position: `{spec['cut_position']}`: {d[('cut_position', spec['cut_position'])]} "
        f"({'the speaker has just finished' if state == 'just_finished' else 'the speaker is still speaking'})",
        f"Listener intent: `{spec['caller_intent']}` (intended label: {intent['label']}): "
        f"{' '.join(intent['description'].split())}",
        _hints(cat, spec["caller_intent"], spec["language"]),
        f"Development: `{spec['development']}`: {d[('development', spec['development'])]}",
        f"Repetition: `{spec['repetition']}`",
        f"Length of listener_text: `{spec['length_bucket']}`: {d[('length_bucket', spec['length_bucket'])]}",
        f"Register: `{spec['register']}`. Listener mood: `{spec['caller_mood']}`.",
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
    "agent_activity": "Speaker activity:",
    "cut_position": "Cut position:",
    "caller_intent": "Listener intent:",
    "development": "Development:",
    "repetition": "Repetition:",
    "length_bucket": "Length of listener_text:",
    "register": "Register:",
}


def _prefixes_for(shared_spec: Mapping[str, str]) -> list[str]:
    prefixes = [p for f, p in _FACTOR_PREFIXES.items() if f in shared_spec]
    if "domain" in shared_spec:
        prefixes.append("Domain notes:")
    if "caller_intent" in shared_spec:
        prefixes += ["Example hints", "English example hints", "No example hints"]
    return prefixes


# ---------------------------------------------------------------- utterance track

UTTERANCE_TEMPLATE = "generator_utterances_v1"
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
    d = cat.descriptions
    intent = cat.intents[spec["caller_intent"]]
    lines = describe_language(cat, spec) + [
        f"Intent: `{spec['caller_intent']}` (intended label: {intent['label']}): "
        f"{' '.join(intent['description'].split())}",
        _hints(cat, spec["caller_intent"], spec["language"]),
        f"Development: `{spec['development']}`: {d[('development', spec['development'])]}",
        f"Repetition: `{spec['repetition']}`",
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


@dataclass(frozen=True)
class JudgeItem:
    """One generated sample as the judge sees it."""

    spec: Mapping[str, str]
    context_speaker: str
    context_text: str
    speaker: str
    speaker_spoken_text: str
    listener: str
    listener_text: str


def _judge_template() -> dict[str, Any]:
    return yaml.safe_load((PROMPT_DIR / f"{JUDGE_TEMPLATE}.yaml").read_text(encoding="utf-8"))


def _switching_text(cat: Catalogue, spec: Mapping[str, str]) -> str:
    if spec["code_switching"] == "none":
        return "no other language mixed in"
    others = " and ".join(language_name(cat, c) for c in spec["code_switch_set"].split("+"))
    how = "within sentences" if spec["code_switching"] == "within_sentence" else "between sentences"
    return f"{others} mixed in {how}"


def render_judge_decision(cat: Catalogue, item: JudgeItem) -> tuple[str, dict[str, Any]]:
    """The Decisions API input (the conversation as JSON) and yes/no questions."""
    template = _judge_template()
    spec = item.spec
    finished = spec["cut_position"] == "just_finished"
    conversation = [
        {"speaker": item.context_speaker, "transcript": item.context_text},
        {"speaker": item.speaker, "transcript": item.speaker_spoken_text},
        {"speaker": item.listener, "transcript": item.listener_text},
    ]
    judge_input = json.dumps({"conversation": conversation}, ensure_ascii=False, indent=2)
    lang = cat.languages[spec["language"]]
    fields = {
        "speaker": item.speaker,
        "listener": item.listener,
        "moment": (
            f"Right after {item.speaker} finished speaking"
            if finished
            else f"While {item.speaker} is speaking"
        ),
        "language": f"{lang['name']} ({spec['regional_variant']})",
        "switching": _switching_text(cat, spec),
        "script": script_expectation(cat, spec),
    }

    def text(raw: str, **extra: str) -> str:
        return " ".join(Template(raw).substitute(fields, **extra).split())

    note = text(template["conversation_note"])
    rules = text(template["label_rules"])
    q = template["questions"]
    cut = q["cut_ok"]["finished" if finished else "speaking"]

    def noul(
        raw: str,
        *,
        with_rules: bool = False,
        true: str | None = None,
        false: str | None = None,
        **extra: str,
    ):
        question: dict[str, Any] = {
            "type": "noul",
            "instructions": " ".join(
                [note, text(raw, **extra), rules if with_rules else ""]
            ).strip(),
        }
        criteria = {}
        if true:
            criteria["true_description"] = text(true)
        if false:
            criteria["false_description"] = text(false)
        if criteria:
            question["criteria"] = criteria
        return question

    questions: dict[str, Any] = {
        "is_backchannel": noul(
            q["is_backchannel"]["instructions"],
            with_rules=True,
            true=q["is_backchannel"]["true_description"],
            false=q["is_backchannel"]["false_description"],
        ),
        "is_barge_in": noul(
            q["is_barge_in"]["instructions"],
            with_rules=True,
            true=q["is_barge_in"]["true_description"],
            false=q["is_barge_in"]["false_description"],
        ),
    }
    tokens = words(item.listener_text)
    for k in range(1, len(tokens)):
        questions[f"prefix_{k}"] = noul(
            q["prefix"]["instructions"], with_rules=True, prefix=" ".join(tokens[:k]), count=str(k)
        )
    for key in ("language_ok", "script_ok", "natural_ok"):
        questions[key] = noul(q[key]["instructions"])
    questions["cut_ok"] = noul(cut)

    f = template["factors"]
    domain = cat.domains[spec["domain"]]["description"]
    scenario = cat.scenarios[spec["scenario"]]["description"]
    activity = cat.descriptions[("agent_activity", spec["agent_activity"])]
    intent = " ".join(cat.intents[spec["caller_intent"]]["description"].split())
    questions["factor_scenario"] = noul(f["scenario"], description=f"{domain} {scenario}")
    questions["factor_speaker_activity"] = noul(f["speaker_activity"], description=activity)
    questions["factor_listener_intent"] = noul(f["listener_intent"], description=intent)
    questions["factor_development"] = noul(f["development"][spec["development"]])
    questions["factor_repetition"] = noul(f["repetition"][spec["repetition"]])
    questions["factor_register"] = noul(f["register"][spec["register"]])
    questions["factor_listener_mood"] = noul(
        f["listener_mood"], description=f["moods"][spec["caller_mood"]]
    )
    return judge_input, questions


def judge_hash(judge_input: str, questions: Mapping[str, Any]) -> str:
    return prompt_hash(judge_input, json.dumps(questions, sort_keys=True, ensure_ascii=False))
